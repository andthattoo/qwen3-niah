"""
LongCoT-Mini baseline runner for an OpenAI-compatible local LLM server.

Designed to point at a local `llama-cpp-python` server hosting the Qwen3.6
GGUF, but works unchanged against any OpenAI-compatible endpoint (OpenRouter,
Anthropic-compat, vLLM, SGLang, etc.).

What it does:
  1. Loads LongCoT-Mini (split="easy" of LongHorizonReasoning/longcot, 507 Qs).
  2. Filters to --domains and caps at --n-per-domain.
  3. Appends a format directive so the model emits `solution = <answer>`
     (Qwen3.6's thinking mode is fine — the marker just needs to appear
     anywhere in the response).
  4. Posts to the local server's /v1/chat/completions.
  5. Scores with a regex for `solution = ...` plus a substring fallback
     against the canonical answer JSON.
  6. Writes JSONL in LongCoT's expected format + a summary.

Typical two-step usage on a single machine (see README):
    # pane 1
    ./run_server.sh

    # pane 2
    python longcot_mini_eval.py --domains logic math --n-per-domain 20

Usage:
  python longcot_mini_eval.py [flags]

  --base-url        OpenAI-compatible base URL (default http://127.0.0.1:8000/v1)
  --model           Model name reported in output JSONL (default qwen3.6-35b-a3b)
  --domains         logic cs chemistry chess math (default: all 5)
  --n-per-domain    Cap problems per domain (0 = all)
  --max-new-tokens  Token cap for each completion (default 16384)
  --temperature     Sampling temperature (default 0.0 — greedy)
  --n-workers       Parallel requests (default 1 for local single-stream server)
  --out-dir         Where to write results (default longcot_mini_eval)
"""

from __future__ import annotations

import argparse
import json
import os
import re
import sys
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path


# ---------------------------------------------------------------------------
# Scoring
# ---------------------------------------------------------------------------

SOLUTION_RE = re.compile(r"solution\s*=\s*(.+?)(?:\n|$)", re.IGNORECASE)


def _normalize(s: str) -> str:
    return re.sub(r"\s+", " ", s.strip().lower())


def proxy_score(response: str, canonical_answer_json: str) -> dict:
    resp = response or ""
    ans_raw = canonical_answer_json or ""
    try:
        parsed = json.loads(ans_raw)
        if isinstance(parsed, (str, int, float, bool)):
            ans_strs = [str(parsed)]
        elif isinstance(parsed, list):
            ans_strs = [str(x) for x in parsed] + [ans_raw]
        else:
            ans_strs = [ans_raw]
    except Exception:
        ans_strs = [ans_raw]
    m = SOLUTION_RE.search(resp)
    extracted = m.group(1).strip() if m else ""
    solution_match = any(
        a and _normalize(a) and _normalize(a) in _normalize(extracted) for a in ans_strs
    )
    substring_match = any(
        a and _normalize(a) and _normalize(a) in _normalize(resp) for a in ans_strs
    )
    return {
        "solution_match": solution_match,
        "substring_match": substring_match,
        "has_solution_tag": bool(m),
        "extracted_solution": extracted[:200] if extracted else "",
    }


# ---------------------------------------------------------------------------
# API call
# ---------------------------------------------------------------------------

def call_once(client, model: str, prompt: str, max_tokens: int, temperature: float):
    resp = client.chat.completions.create(
        model=model,
        messages=[{"role": "user", "content": prompt}],
        max_tokens=max_tokens,
        temperature=temperature,
    )
    text = resp.choices[0].message.content or ""
    usage = {}
    if getattr(resp, "usage", None):
        usage = {
            "prompt_tokens": getattr(resp.usage, "prompt_tokens", 0),
            "completion_tokens": getattr(resp.usage, "completion_tokens", 0),
            "total_tokens": getattr(resp.usage, "total_tokens", 0),
        }
    return text, usage


def call_with_retries(
    client, model, prompt, max_tokens, temperature, max_retries=3, backoff=2.0
):
    last_err = None
    for attempt in range(max_retries):
        try:
            return True, *call_once(client, model, prompt, max_tokens, temperature)
        except Exception as e:
            last_err = e
            if attempt < max_retries - 1:
                time.sleep(backoff * (2 ** attempt))
    return False, f"[ERROR: {type(last_err).__name__}: {last_err}]", {}


# ---------------------------------------------------------------------------
# Runner
# ---------------------------------------------------------------------------

FORMAT_HINT = (
    "\n\nWhen you have finished reasoning, output your final answer on its own "
    "line in the exact format:\n  solution = <your final answer>\n"
    "Use a single concise value (number, string, list, etc.) as the answer."
)


def run(args):
    try:
        from openai import OpenAI
    except ImportError:
        print("Error: openai not installed. Run: uv pip install openai", file=sys.stderr)
        sys.exit(1)
    try:
        from datasets import load_dataset
    except ImportError:
        print("Error: datasets not installed. Run: uv pip install datasets", file=sys.stderr)
        sys.exit(1)

    api_key = os.environ.get(args.api_key_env, "dummy")
    client = OpenAI(base_url=args.base_url, api_key=api_key)

    print(f"[1/3] Loading LongCoT-Mini (split=easy) …")
    ds = load_dataset("LongHorizonReasoning/longcot", "all", split="easy")
    questions = list(ds)
    if args.domains:
        keep = set(args.domains)
        questions = [q for q in questions if q["domain"] in keep]
    if args.n_per_domain > 0:
        by_d: dict[str, list[dict]] = {}
        for q in questions:
            by_d.setdefault(q["domain"], []).append(q)
        questions = []
        for d in sorted(by_d):
            questions.extend(by_d[d][: args.n_per_domain])
    print(f"  {len(questions)} questions  ({sorted({q['domain'] for q in questions})})")

    print(f"[2/3] Calling {args.model} via {args.base_url}  (workers={args.n_workers})")

    jsonl_rows: list[dict] = []
    scores: list[dict] = []
    t0 = time.time()
    ctr = {"done": 0, "errs": 0}

    def process_one(q):
        prompt = q["prompt"] + (FORMAT_HINT if args.format_hint else "")
        ok, response, usage = call_with_retries(
            client, args.model, prompt, args.max_new_tokens, args.temperature,
            max_retries=args.max_retries,
        )
        score = proxy_score(response, q.get("answer", "") or "")
        score.update({
            "question_id": q["question_id"],
            "domain": q["domain"],
            "template": q.get("template", ""),
            "completion_tokens": usage.get("completion_tokens", 0),
        })
        row = {
            "question_id": q["question_id"],
            "successful": ok,
            "response_text": response,
            "model": args.model,
            "usage": usage,
        }
        return q, row, score, ok

    with ThreadPoolExecutor(max_workers=args.n_workers) as pool:
        futures = {pool.submit(process_one, q): q["question_id"] for q in questions}
        for fut in as_completed(futures):
            q, row, score, ok = fut.result()
            jsonl_rows.append(row)
            scores.append(score)
            ctr["done"] += 1
            if not ok:
                ctr["errs"] += 1
            n = ctr["done"]
            if n % max(1, args.log_every) == 0 or n == len(questions):
                elapsed = time.time() - t0
                rate = n / max(elapsed, 1e-6)
                correct = sum(1 for s in scores if s["solution_match"])
                substr = sum(1 for s in scores if s["substring_match"])
                tag = sum(1 for s in scores if s["has_solution_tag"])
                mean_ct = sum(s["completion_tokens"] for s in scores) / max(n, 1)
                print(f"  [{n}/{len(questions)}]  sol={correct} sub={substr} tag={tag} "
                      f"errs={ctr['errs']}  mean_ct={mean_ct:.0f}  "
                      f"{rate:.3f} q/s  elapsed={elapsed:.0f}s")

    # Save JSONL (LongCoT-compatible)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    slug = re.sub(r"[^a-zA-Z0-9._-]+", "_", args.model).strip("_")
    jsonl_path = out_dir / f"results_{slug}.jsonl"
    with jsonl_path.open("w") as f:
        for r in jsonl_rows:
            f.write(json.dumps(r) + "\n")

    # Summary
    n = len(scores)
    by_domain: dict[str, list[dict]] = {}
    for s in scores:
        by_domain.setdefault(s["domain"], []).append(s)

    print(f"\n[3/3] Results  (model={args.model}, n={n})")
    if n:
        print(f"  solution_match:   {sum(1 for s in scores if s['solution_match'])/n*100:.1f}%")
        print(f"  substring_match:  {sum(1 for s in scores if s['substring_match'])/n*100:.1f}%")
        print(f"  has_solution_tag: {sum(1 for s in scores if s['has_solution_tag'])/n*100:.1f}%")
        mean_ct = sum(s["completion_tokens"] for s in scores) / n
        print(f"  mean completion tokens: {mean_ct:.0f}")
        print("\n  Per-domain (solution_match):")
        for d in sorted(by_domain):
            subset = by_domain[d]
            c = sum(1 for s in subset if s["solution_match"])
            print(f"    {d:<12s}  {c}/{len(subset)}  ({c/len(subset)*100:.1f}%)")

    summary = {
        "args": vars(args),
        "n": n,
        "solution_match_acc": sum(1 for s in scores if s["solution_match"]) / n if n else 0,
        "substring_match_acc": sum(1 for s in scores if s["substring_match"]) / n if n else 0,
        "has_solution_tag_rate": sum(1 for s in scores if s["has_solution_tag"]) / n if n else 0,
        "mean_completion_tokens": sum(s["completion_tokens"] for s in scores) / n if n else 0,
        "per_domain": {
            d: {
                "n": len(sub),
                "solution_match_acc": sum(1 for x in sub if x["solution_match"]) / len(sub),
                "substring_match_acc": sum(1 for x in sub if x["substring_match"]) / len(sub),
            }
            for d, sub in by_domain.items()
        },
    }
    summary_path = out_dir / f"summary_{slug}.json"
    with summary_path.open("w") as f:
        json.dump(summary, f, indent=2)

    print(f"\nSaved → {jsonl_path}")
    print(f"Saved → {summary_path}")


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--base-url", default="http://127.0.0.1:8000/v1",
                   help="OpenAI-compatible endpoint. Default: local llama-cpp-python server.")
    p.add_argument("--api-key-env", default="DUMMY_KEY",
                   help="Env var holding API key. Local server ignores it — any non-empty string works.")
    p.add_argument("--model", default="qwen3.6-35b-a3b",
                   help="Model name reported in output JSONL.")

    p.add_argument("--domains", nargs="*", default=None,
                   help="Filter to these domains: logic cs chemistry chess math.")
    p.add_argument("--n-per-domain", type=int, default=0,
                   help="Cap problems per domain (0 = all).")
    p.add_argument("--max-new-tokens", type=int, default=16384)
    p.add_argument("--temperature", type=float, default=0.0)

    p.add_argument("--format-hint", action="store_true", default=True,
                   help="Append an instruction asking for `solution = ...`.")
    p.add_argument("--no-format-hint", dest="format_hint", action="store_false")

    p.add_argument("--n-workers", type=int, default=1,
                   help="Parallel requests. Keep at 1 for single-stream local servers.")
    p.add_argument("--max-retries", type=int, default=3)
    p.add_argument("--log-every", type=int, default=1)

    p.add_argument("--out-dir", default="longcot_mini_eval")
    args = p.parse_args()
    run(args)


if __name__ == "__main__":
    main()
