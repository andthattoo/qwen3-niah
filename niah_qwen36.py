"""
Needle-in-a-Haystack passkey-retrieval eval for unsloth/Qwen3.6-35B-A3B-GGUF.

Goal: characterize how recall degrades with context length on the hybrid
DeltaNet + Attention architecture.  If recall holds >80% across 128K-512K, the
model's native DeltaNet state handles long-range fine and Memory Caching is
unnecessary for this base.  If recall drops steeply past 128K, MC has a clear
job to do.

Pipeline:
  1. Pull GGUF quant (default UD-Q4_K_M) via hf_hub_download.
  2. Load with llama-cpp-python built against CUDA, set n_ctx to the largest
     context we want to test so every inner run reuses the same model.
  3. For each (context_len, needle_depth, seed):
       - build a passkey prompt of target length, needle inserted at depth
       - generate up to 40 tokens greedily
       - score: does the output contain the 7-digit passkey?
  4. Report a grid of accuracy by (context_len × depth) and save raw results.

Usage:
  python niah_qwen36.py --context-lengths 8192 32768 131072 \\
      --depths 0.1 0.5 0.9 --seeds 2

  # Stretch to longer contexts (much slower per query)
  python niah_qwen36.py --context-lengths 131072 262144 524288 \\
      --depths 0.1 0.5 0.9 --seeds 2

Notes:
  - The needle depth is the *fractional position* of the needle in the prompt,
    0.1 ≈ near the start, 0.9 ≈ near the end.
  - Prompt-processing time dominates for long contexts.  At 512K tokens expect
    several minutes per query even on H100.  Budget accordingly.
"""

from __future__ import annotations

import argparse
import ctypes
import glob
import json
import os
import random
import re
import site
import time
from pathlib import Path


def _preload_cuda_libs() -> None:
    """Preload CUDA shared libs from the venv's bundled nvidia/* packages so
    llama-cpp-python finds them without needing LD_LIBRARY_PATH set in the shell.

    llama-cpp-python's prebuilt cu124 wheel dynamically links libcudart.so.12 etc.
    Modern torch bundles cu13 by default, so those libs sit in site-packages under
    nvidia/cuda_runtime/lib/, nvidia/cublas/lib/, etc.  We dlopen them with
    RTLD_GLOBAL before importing llama_cpp so the symbols are visible.
    """
    search_dirs = set()
    for sp_dir in list(site.getsitepackages()) + [site.getusersitepackages()]:
        nv = os.path.join(sp_dir, "nvidia")
        if os.path.isdir(nv):
            for sub in os.listdir(nv):
                lib = os.path.join(nv, sub, "lib")
                if os.path.isdir(lib):
                    search_dirs.add(lib)

    loaded: set[str] = set()
    # Multiple passes: some libs depend on others; keep looping until no progress.
    for _ in range(4):
        progress = False
        for d in sorted(search_dirs):
            for pattern in ("*.so.12", "*.so.13", "*.so"):
                for so in glob.glob(os.path.join(d, pattern)):
                    if so in loaded:
                        continue
                    try:
                        ctypes.CDLL(so, mode=ctypes.RTLD_GLOBAL)
                        loaded.add(so)
                        progress = True
                    except OSError:
                        pass
        if not progress:
            break


# ---------------------------------------------------------------------------
# Defaults
# ---------------------------------------------------------------------------

DEFAULT_REPO = "unsloth/Qwen3.6-35B-A3B-GGUF"
DEFAULT_QUANT = "UD-Q4_K_M"  # 22.1 GB — fits on H100 80GB with room for 512K KV
DEFAULT_CTXS = [8192, 32768, 131072]
DEFAULT_DEPTHS = [0.1, 0.5, 0.9]
DEFAULT_SEEDS = 3

PASSKEY_RE = re.compile(r"\b(\d{7})\b")


# ---------------------------------------------------------------------------
# Prompt construction
# ---------------------------------------------------------------------------

SYSTEM_PREFIX = (
    "You will be given a long passage of filler text.  Somewhere inside there "
    "is a single important sentence containing a special 7-digit passkey.  "
    "Read carefully.  At the end you will be asked to recall the passkey.\n\n"
)

NEEDLE_TEMPLATE = (
    "\n\n**IMPORTANT**: The special passkey is {passkey}.  "
    "Remember this number — you will be asked about it later.\n\n"
)

QUESTION_SUFFIX = (
    "\n\nQuestion: What is the special 7-digit passkey mentioned in the text above?\n"
    "Answer: The passkey is"
)


def generate_passkey(seed: int) -> str:
    rng = random.Random(seed * 1_000_003 + 17)
    return "".join(rng.choices("0123456789", k=7))


def load_filler_text() -> str:
    """Return a big blob of semi-coherent English filler."""
    candidates = [Path("/tmp/filler.txt"), Path("./data/filler.txt")]
    for p in candidates:
        if p.exists() and p.stat().st_size > 1_000_000:
            return p.read_text()

    # Build one from PG-19 (public domain books) — streaming, tokenize-friendly.
    from datasets import load_dataset
    print("  Streaming PG-19 to build filler corpus…")
    ds = load_dataset("deepmind/pg19", split="validation", streaming=True)
    chunks: list[str] = []
    total_chars = 0
    for row in ds:
        text = row.get("text", "")
        if not text:
            continue
        chunks.append(text)
        total_chars += len(text)
        if total_chars > 8_000_000:   # ~8MB, plenty for 1M tokens worth
            break
    blob = "\n\n".join(chunks)
    Path("/tmp/filler.txt").write_text(blob)
    print(f"  Filler corpus: {len(blob):,} chars cached → /tmp/filler.txt")
    return blob


def build_prompt(
    llm,
    context_length: int,
    depth: float,
    passkey: str,
    filler: str,
) -> tuple[str, int]:
    """Construct a prompt of approximately `context_length` tokens with the
    passkey needle at fractional depth `depth` (0.0–1.0).  Returns (prompt, actual_tokens)."""
    sys_ids = llm.tokenize(SYSTEM_PREFIX.encode(), add_bos=False, special=False)
    q_ids = llm.tokenize(QUESTION_SUFFIX.encode(), add_bos=False, special=False)
    needle_text = NEEDLE_TEMPLATE.format(passkey=passkey)
    needle_ids = llm.tokenize(needle_text.encode(), add_bos=False, special=False)

    overhead = len(sys_ids) + len(q_ids) + len(needle_ids) + 64  # slack for bos / formatting
    filler_budget = max(1, context_length - overhead)

    # Tokenize the full filler once, then slice
    filler_ids = llm.tokenize(filler.encode(), add_bos=False, special=False)
    if len(filler_ids) < filler_budget:
        # Repeat if needed
        repeats = (filler_budget // len(filler_ids)) + 2
        filler_ids = filler_ids * repeats
    filler_ids = filler_ids[:filler_budget]

    n_before = int(filler_budget * depth)
    before_ids = filler_ids[:n_before]
    after_ids = filler_ids[n_before:]

    # Reassemble.  We need strings to re-tokenize cleanly; llama.cpp's detokenize
    # gives us that.
    before_text = llm.detokenize(before_ids).decode("utf-8", errors="ignore")
    after_text = llm.detokenize(after_ids).decode("utf-8", errors="ignore")

    prompt = SYSTEM_PREFIX + before_text + needle_text + after_text + QUESTION_SUFFIX
    actual_ids = llm.tokenize(prompt.encode(), add_bos=True, special=True)
    return prompt, len(actual_ids)


# ---------------------------------------------------------------------------
# Scoring
# ---------------------------------------------------------------------------

def score_output(output: str, passkey: str) -> dict:
    output = output or ""
    matches = PASSKEY_RE.findall(output)
    first_seven = matches[0] if matches else ""
    return {
        "correct": passkey in matches,
        "first_7digit_in_output": first_seven,
        "any_digit_match": passkey in output,
    }


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def run_eval(args):
    # Preload CUDA shared libs from the venv so llama-cpp-python can dlopen
    # them without requiring the user to set LD_LIBRARY_PATH.
    _preload_cuda_libs()
    try:
        from llama_cpp import Llama
    except ImportError:
        print("Error: llama-cpp-python not installed.")
        raise
    from huggingface_hub import hf_hub_download

    ctxs = sorted(set(int(x) for x in args.context_lengths))
    max_ctx = max(ctxs)
    depths = sorted(set(float(x) for x in args.depths))

    print(f"[1/4] Downloading GGUF: {args.repo}  quant={args.quant}")
    t0 = time.time()
    # Unsloth's GGUF files are named `{model}-{quant}.gguf` under the repo.
    # Some large quants are sharded; we accept a single-file or sharded pattern.
    target_filename = args.filename or f"{args.quant}/*"
    if "*" in target_filename:
        # Sharded case — list and download all parts
        from huggingface_hub import list_repo_files
        files = list_repo_files(args.repo)
        pattern = args.quant
        matches = [f for f in files if f.startswith(pattern) and f.endswith(".gguf")]
        if not matches:
            # Maybe it's flat in root
            matches = [f for f in files if pattern in f and f.endswith(".gguf")]
        if not matches:
            raise RuntimeError(f"No GGUF files matching '{pattern}' in {args.repo}")
        local_paths = [
            hf_hub_download(args.repo, filename=f) for f in sorted(matches)
        ]
        model_path = local_paths[0]  # llama.cpp auto-loads sharded files from first
    else:
        model_path = hf_hub_download(args.repo, filename=target_filename)
    print(f"  resolved model_path={model_path}  ({time.time()-t0:.0f}s)")

    print(f"[2/4] Loading model into llama.cpp  (n_ctx={max_ctx})")
    t0 = time.time()
    llm = Llama(
        model_path=model_path,
        n_ctx=max_ctx,
        n_gpu_layers=-1,
        n_batch=1024,
        flash_attn=True,
        type_k=8,  # 8-bit K cache
        type_v=8,  # 8-bit V cache
        verbose=False,
        seed=1,
    )
    print(f"  loaded in {time.time()-t0:.0f}s")

    print(f"[3/4] Building filler corpus")
    filler = load_filler_text()

    print(f"[4/4] Running NIAH grid: ctxs={ctxs}  depths={depths}  seeds={args.seeds}")
    results: list[dict] = []
    grid: dict[tuple[int, float], list[bool]] = {}

    total_runs = len(ctxs) * len(depths) * args.seeds
    run_idx = 0
    grand_start = time.time()

    for ctx in ctxs:
        for depth in depths:
            for seed in range(args.seeds):
                run_idx += 1
                passkey = generate_passkey(ctx * 31 + int(depth * 100) * 7 + seed)
                try:
                    prompt, n_tokens = build_prompt(llm, ctx, depth, passkey, filler)
                except Exception as e:
                    print(f"    [{run_idx}/{total_runs}] ctx={ctx} depth={depth} "
                          f"seed={seed}  PROMPT-BUILD-FAILED: {e}")
                    continue

                t_gen0 = time.time()
                try:
                    out = llm(
                        prompt,
                        max_tokens=40,
                        temperature=0.0,
                        top_p=1.0,
                        top_k=1,
                        echo=False,
                    )
                    completion = out["choices"][0]["text"]
                except Exception as e:
                    print(f"    [{run_idx}/{total_runs}] ctx={ctx} depth={depth} "
                          f"seed={seed}  INFER-FAILED: {e}")
                    continue
                dt = time.time() - t_gen0

                score = score_output(completion, passkey)
                results.append({
                    "context_length": ctx,
                    "actual_tokens": n_tokens,
                    "depth": depth,
                    "seed": seed,
                    "passkey": passkey,
                    "completion": completion.strip()[:200],
                    "correct": score["correct"],
                    "any_digit_match": score["any_digit_match"],
                    "infer_seconds": round(dt, 2),
                })
                grid.setdefault((ctx, depth), []).append(score["correct"])

                tag = "✓" if score["correct"] else "✗"
                elapsed = time.time() - grand_start
                print(f"    [{run_idx}/{total_runs}] ctx={ctx:>7}  depth={depth:.2f}  "
                      f"seed={seed}  tok={n_tokens:>7}  {tag}  "
                      f"gen={dt:.1f}s  total={elapsed:.0f}s  "
                      f"pk={passkey}  got={score['first_7digit_in_output'] or '-'}")

    # Print grid
    print("\n" + "=" * 78)
    print("  RECALL GRID  (fraction correct, mean over seeds)")
    print("=" * 78)
    header = f"  {'depth →':<10s}" + "".join(f"{d:>8.2f}" for d in depths) + "  | n"
    print(header)
    for ctx in ctxs:
        row = f"  ctx={ctx:<7}"
        for d in depths:
            vals = grid.get((ctx, d), [])
            acc = sum(vals) / len(vals) if vals else float("nan")
            row += f"{acc:>8.2f}"
        n = sum(len(grid.get((ctx, d), [])) for d in depths)
        row += f"  | {n}"
        print(row)

    # Save
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    with (out_dir / "niah_raw.jsonl").open("w") as f:
        for r in results:
            f.write(json.dumps(r) + "\n")
    summary = {
        "args": vars(args),
        "total_runs": len(results),
        "grid": {
            f"ctx={c}|depth={d}": {
                "n": len(grid.get((c, d), [])),
                "accuracy": sum(grid.get((c, d), [])) / len(grid.get((c, d), []))
                if grid.get((c, d)) else None,
            }
            for c in ctxs for d in depths
        },
    }
    with (out_dir / "niah_summary.json").open("w") as f:
        json.dump(summary, f, indent=2)
    print(f"\nSaved → {out_dir / 'niah_raw.jsonl'}")
    print(f"Saved → {out_dir / 'niah_summary.json'}")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main():
    p = argparse.ArgumentParser()
    p.add_argument("--repo", default=DEFAULT_REPO)
    p.add_argument("--quant", default=DEFAULT_QUANT,
                   help="Quant tag within the repo (e.g., UD-Q4_K_M).")
    p.add_argument("--filename", default="",
                   help="Override: exact .gguf filename inside the repo. "
                        "If empty we search for files matching --quant.")

    p.add_argument("--context-lengths", nargs="+", type=int, default=DEFAULT_CTXS)
    p.add_argument("--depths", nargs="+", type=float, default=DEFAULT_DEPTHS)
    p.add_argument("--seeds", type=int, default=DEFAULT_SEEDS)

    p.add_argument("--out-dir", default="niah_qwen36")
    args = p.parse_args()

    run_eval(args)


if __name__ == "__main__":
    main()
