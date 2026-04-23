"""
FSM vs. free thinking — zero-training comparison on a code benchmark.

For each problem, run Qwen3.6-35B-A3B (via the local llama.cpp OpenAI-compatible
server) in two modes:

  FREE:  standard thinking-mode generation.  Model produces its native verbose
         <think>...</think> followed by an answer.

  FSM :  grammar-constrained generation.  The same model is forced via GBNF to
         emit a compact structured plan inside <think>...</think>, then the
         grammar becomes permissive so the model can write the code freely.

We measure:
  - pass@1 on the benchmark's hidden tests
  - thinking-token count (the tokens inside <think>...</think>)
  - total-completion-token count

and report a side-by-side table + compression ratio + accuracy delta.

No training.  No data pipeline.  Just constrained decoding.

Usage:
  # Make sure the server is up (see run_server.sh)
  uv run python fsm_vs_free_eval.py --n-problems 30 --dataset humaneval

  # Try MBPP+
  uv run python fsm_vs_free_eval.py --n-problems 50 --dataset mbpp

  # Free-only or FSM-only (for debugging)
  uv run python fsm_vs_free_eval.py --only free --n-problems 10
  uv run python fsm_vs_free_eval.py --only fsm  --n-problems 10
"""

from __future__ import annotations

import argparse
import json
import os
import re
import subprocess
import sys
import time
from pathlib import Path
from typing import Optional


# ---------------------------------------------------------------------------
# Dataset loading
# ---------------------------------------------------------------------------

def load_benchmark(name: str, n: int):
    from datasets import load_dataset
    if name == "humaneval":
        ds = load_dataset("evalplus/humanevalplus", split="test")
        # Fields: task_id, prompt, canonical_solution, test, entry_point
    elif name == "mbpp":
        ds = load_dataset("evalplus/mbppplus", split="test")
    else:
        raise ValueError(f"unknown dataset {name}")
    rows = list(ds)
    return rows[:n] if n > 0 else rows


# ---------------------------------------------------------------------------
# Prompt / response helpers
# ---------------------------------------------------------------------------

SYSTEM_PROMPT = (
    "You are an expert Python programmer.  Think carefully in your <think> "
    "block, then write correct, efficient, well-tested code.  "
    "Wrap your final code in a ```python ... ``` fenced block."
)


def build_user_prompt(problem: dict, dataset: str) -> str:
    if dataset == "humaneval":
        return (
            "Complete the following Python function.  Return the full function "
            "including the signature and docstring.\n\n"
            f"```python\n{problem['prompt']}```\n"
        )
    elif dataset == "mbpp":
        return (
            f"{problem['prompt']}\n\n"
            "Write the complete implementation in Python."
        )
    raise ValueError(dataset)


# Regex for extracting tokens.
THINK_RE = re.compile(r"<think>(.*?)</think>", re.DOTALL)
CODE_FENCED_RE = re.compile(r"```(?:python)?\s*\n(.*?)```", re.DOTALL)
CODE_DEF_RE = re.compile(r"(def\s+\w+.*?)(?=\n\S|\Z)", re.DOTALL)


def extract_think(text: str) -> str:
    m = THINK_RE.search(text)
    return m.group(1) if m else ""


def extract_code(text: str) -> str:
    # Prefer fenced code block after </think>.
    after_think = text.split("</think>", 1)[-1]
    m = CODE_FENCED_RE.search(after_think)
    if m:
        return m.group(1)
    # Fallback: any fenced block in the whole text
    m = CODE_FENCED_RE.search(text)
    if m:
        return m.group(1)
    # Last resort: the first def found after </think>
    m = CODE_DEF_RE.search(after_think)
    if m:
        return m.group(1)
    return after_think.strip()


# ---------------------------------------------------------------------------
# Tokenizer (for thinking-token counts)
# ---------------------------------------------------------------------------

_TOK = None


def get_tokenizer(model_name: str):
    global _TOK
    if _TOK is None:
        try:
            from transformers import AutoTokenizer
            _TOK = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
        except Exception as e:
            print(f"  (Falling back to char-based count: {e})", file=sys.stderr)
            _TOK = "fallback"
    return _TOK


def count_tokens(text: str, model_name: str) -> int:
    tok = get_tokenizer(model_name)
    if tok == "fallback":
        # char/4 is a rough BPE approximation
        return max(1, len(text) // 4)
    return len(tok.encode(text, add_special_tokens=False))


# ---------------------------------------------------------------------------
# Test execution (sandboxed-ish via subprocess)
# ---------------------------------------------------------------------------

def run_tests(code: str, test_code: str, entry_point: str, timeout: int = 30) -> tuple[bool, str]:
    # HumanEval/MBPP tests usually define a `check(func)` or run asserts.
    full = f"{code}\n\n{test_code}\n\ntry:\n    check({entry_point})\nexcept NameError:\n    pass\n"
    try:
        proc = subprocess.run(
            [sys.executable, "-c", full],
            timeout=timeout,
            capture_output=True,
        )
        ok = proc.returncode == 0
        err = proc.stderr.decode("utf-8", errors="ignore")[-500:] if not ok else ""
        return ok, err
    except subprocess.TimeoutExpired:
        return False, "TIMEOUT"
    except Exception as e:
        return False, f"{type(e).__name__}: {e}"[:200]


# ---------------------------------------------------------------------------
# Generation (via OpenAI-compatible server)
# ---------------------------------------------------------------------------

def make_client(args):
    from openai import OpenAI
    client = OpenAI(
        base_url=args.base_url,
        api_key=os.environ.get(args.api_key_env, "dummy"),
    )
    # Pre-flight: verify the server is reachable before running any problems.
    try:
        _ = client.models.list()
    except Exception as e:
        print(
            f"ERROR: cannot reach the server at {args.base_url}\n"
            f"  ({type(e).__name__}: {e})\n\n"
            "Start the local llama-cpp-python server first:\n"
            "  nohup ./run_server.sh > server.log 2>&1 &\n"
            "  tail -f server.log   # wait for 'Uvicorn running on ...'\n\n"
            "Or point --base-url at whatever OpenAI-compatible endpoint you want.",
            file=sys.stderr,
        )
        sys.exit(1)
    return client


def generate_free(client, model: str, user_prompt: str, max_tokens: int) -> tuple[str, int]:
    """Standard chat completion — no grammar, no constraint."""
    r = client.chat.completions.create(
        model=model,
        messages=[
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": user_prompt},
        ],
        max_tokens=max_tokens,
        temperature=0.0,
    )
    text = r.choices[0].message.content or ""
    completion_tokens = r.usage.completion_tokens if r.usage else count_tokens(text, model)
    return text, completion_tokens


def generate_fsm(client, model: str, user_prompt: str, grammar: str, max_tokens: int) -> tuple[str, int]:
    """Chat completion with GBNF grammar applied to the whole assistant response."""
    r = client.chat.completions.create(
        model=model,
        messages=[
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": user_prompt},
        ],
        max_tokens=max_tokens,
        temperature=0.0,
        extra_body={"grammar": grammar},
    )
    text = r.choices[0].message.content or ""
    completion_tokens = r.usage.completion_tokens if r.usage else count_tokens(text, model)
    return text, completion_tokens


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    p = argparse.ArgumentParser()
    p.add_argument("--base-url", default="http://127.0.0.1:8000/v1")
    p.add_argument("--api-key-env", default="DUMMY_KEY")
    p.add_argument("--model", default="qwen3.6-35b-a3b",
                   help="Model name sent in the request (local server ignores it).")
    p.add_argument("--tokenizer", default="Qwen/Qwen3.6-35B-A3B",
                   help="HF tokenizer id for counting think tokens.")

    p.add_argument("--dataset", choices=["humaneval", "mbpp"], default="humaneval")
    p.add_argument("--n-problems", type=int, default=30,
                   help="Problems to evaluate (0 = all).")
    p.add_argument("--grammar-file", default="fsm_grammar.gbnf")
    p.add_argument("--max-tokens", type=int, default=8192)
    p.add_argument("--timeout", type=int, default=30,
                   help="Per-test execution timeout (seconds).")

    p.add_argument("--only", choices=["both", "free", "fsm"], default="both",
                   help="Run one mode only (for debugging).")
    p.add_argument("--out-dir", default="fsm_vs_free")
    args = p.parse_args()

    try:
        grammar = Path(args.grammar_file).read_text()
    except Exception as e:
        print(f"ERROR: could not read grammar file {args.grammar_file}: {e}", file=sys.stderr)
        sys.exit(1)

    client = make_client(args)

    print(f"[1/3] Loading {args.dataset} problems")
    problems = load_benchmark(args.dataset, args.n_problems)
    print(f"  {len(problems)} problems")

    print(f"[2/3] Running {'both modes' if args.only == 'both' else args.only}")
    results = []
    t_start = time.time()

    for i, prob in enumerate(problems):
        user_prompt = build_user_prompt(prob, args.dataset)
        entry_point = prob.get("entry_point") or "candidate"
        test_code = prob.get("test", "")

        row = {"task_id": prob["task_id"]}
        t_prob = time.time()

        if args.only in ("both", "free"):
            try:
                free_text, free_total_tokens = generate_free(
                    client, args.model, user_prompt, args.max_tokens
                )
                free_think = extract_think(free_text)
                free_code = extract_code(free_text)
                free_pass, free_err = run_tests(
                    free_code, test_code, entry_point, args.timeout
                )
                row["free"] = {
                    "pass": free_pass,
                    "err": free_err[:100],
                    "think_tokens": count_tokens(free_think, args.tokenizer),
                    "total_tokens": int(free_total_tokens),
                    "code_first_200": free_code[:200],
                }
            except Exception as e:
                row["free"] = {"pass": False, "err": f"gen_error: {e}"[:200]}

        if args.only in ("both", "fsm"):
            try:
                fsm_text, fsm_total_tokens = generate_fsm(
                    client, args.model, user_prompt, grammar, args.max_tokens
                )
                fsm_think = extract_think(fsm_text)
                fsm_code = extract_code(fsm_text)
                fsm_pass, fsm_err = run_tests(
                    fsm_code, test_code, entry_point, args.timeout
                )
                row["fsm"] = {
                    "pass": fsm_pass,
                    "err": fsm_err[:100],
                    "think_tokens": count_tokens(fsm_think, args.tokenizer),
                    "total_tokens": int(fsm_total_tokens),
                    "code_first_200": fsm_code[:200],
                }
            except Exception as e:
                row["fsm"] = {"pass": False, "err": f"gen_error: {e}"[:200]}

        dt = time.time() - t_prob
        results.append(row)

        def tag(d): return "✓" if d and d.get("pass") else "✗"
        def tt(d):  return d.get("think_tokens", "-") if d else "-"
        err_bits = []
        for m in ("free", "fsm"):
            d = row.get(m)
            if d and not d.get("pass"):
                e = (d.get("err") or "").strip()
                if e:
                    err_bits.append(f"{m}: {e[:80]}")
        err_str = ("  |  " + " ; ".join(err_bits)) if err_bits else ""
        print(
            f"  [{i+1}/{len(problems)}] {prob['task_id']:<16s}  "
            f"free={tag(row.get('free'))} ({tt(row.get('free'))}tt)   "
            f"fsm={tag(row.get('fsm'))}  ({tt(row.get('fsm'))}tt)  "
            f"{dt:.0f}s{err_str}"
        )

    elapsed = time.time() - t_start

    # ---- Summary ----
    def mean(xs):
        return sum(xs) / max(len(xs), 1)

    free_rows = [r["free"] for r in results if "free" in r and r["free"].get("pass") is not None]
    fsm_rows = [r["fsm"]  for r in results if "fsm"  in r and r["fsm"].get("pass")  is not None]

    print(f"\n[3/3] Summary  (n={len(results)}, elapsed={elapsed:.0f}s)")
    print("  " + "-" * 70)
    if free_rows:
        free_pass_rate = mean([1.0 if r["pass"] else 0.0 for r in free_rows])
        free_think_mean = mean([r.get("think_tokens", 0) for r in free_rows])
        free_total_mean = mean([r.get("total_tokens", 0) for r in free_rows])
        print(f"  FREE  :  pass@1 = {free_pass_rate*100:5.1f}%   "
              f"mean think = {free_think_mean:6.0f} tok   "
              f"mean total = {free_total_mean:6.0f} tok")
    if fsm_rows:
        fsm_pass_rate = mean([1.0 if r["pass"] else 0.0 for r in fsm_rows])
        fsm_think_mean = mean([r.get("think_tokens", 0) for r in fsm_rows])
        fsm_total_mean = mean([r.get("total_tokens", 0) for r in fsm_rows])
        print(f"  FSM   :  pass@1 = {fsm_pass_rate*100:5.1f}%   "
              f"mean think = {fsm_think_mean:6.0f} tok   "
              f"mean total = {fsm_total_mean:6.0f} tok")
    if free_rows and fsm_rows:
        acc_delta = (fsm_pass_rate - free_pass_rate) * 100
        compression = free_think_mean / max(fsm_think_mean, 1)
        print("  " + "-" * 70)
        print(f"  Accuracy delta (FSM − FREE): {acc_delta:+5.1f} pp")
        print(f"  Think-token compression    : {compression:5.2f}×")

    # ---- Save ----
    out = Path(args.out_dir)
    out.mkdir(parents=True, exist_ok=True)
    (out / "results.jsonl").write_text(
        "\n".join(json.dumps(r) for r in results) + "\n"
    )
    summary = {
        "args": vars(args),
        "n": len(results),
        "elapsed_sec": elapsed,
    }
    if free_rows:
        summary["free"] = {
            "pass_rate": free_pass_rate,
            "think_tokens_mean": free_think_mean,
            "total_tokens_mean": free_total_mean,
        }
    if fsm_rows:
        summary["fsm"] = {
            "pass_rate": fsm_pass_rate,
            "think_tokens_mean": fsm_think_mean,
            "total_tokens_mean": fsm_total_mean,
        }
    (out / "summary.json").write_text(json.dumps(summary, indent=2))
    print(f"\nSaved → {out / 'results.jsonl'}")
    print(f"Saved → {out / 'summary.json'}")


if __name__ == "__main__":
    main()
