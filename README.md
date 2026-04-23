# qwen3-niah

Needle-in-a-Haystack passkey-retrieval evaluation for [`unsloth/Qwen3.6-35B-A3B-GGUF`](https://huggingface.co/unsloth/Qwen3.6-35B-A3B-GGUF) across context lengths.

**Question**: does recall degrade as context grows on Qwen3.6's hybrid DeltaNet + Attention architecture? If recall holds at 128K-512K, the model's built-in DeltaNet state already handles long-range recall. If it drops steeply, external memory augmentation (Titans-style memory caching) has a clear job.

## Setup

Tested on an H100 with CUDA 12.4.

### Option A — uv (recommended)

```bash
uv sync
# llama-cpp-python is pulled from the prebuilt cu124 index via tool.uv.sources.
# cu12 runtime libs are pinned explicitly so they coexist with whatever CUDA
# version torch brings (torch ≥ 2.7 defaults to cu13).

# Before running, point the dynamic loader at the bundled CUDA libs:
export LD_LIBRARY_PATH=$(uv run python -c "
import site, os
sp = site.getsitepackages()[0]
nv = os.path.join(sp, 'nvidia')
print(':'.join(os.path.join(nv, d, 'lib') for d in os.listdir(nv)
               if os.path.isdir(os.path.join(nv, d, 'lib'))))
")
```

### Option B — pip

```bash
pip install --upgrade pip
pip install -r requirements.txt
pip install llama-cpp-python \
    --extra-index-url https://abetlen.github.io/llama-cpp-python/whl/cu124
```

### Fallback (build llama-cpp-python from source)

If the prebuilt wheel can't load Qwen3.6 GGUF (support for the hybrid arch may be newer than what the wheel ships):

```bash
CMAKE_ARGS='-DGGML_CUDA=on' uv pip install llama-cpp-python --no-binary llama-cpp-python
# or with pip
CMAKE_ARGS='-DGGML_CUDA=on' pip install llama-cpp-python --no-binary llama-cpp-python
```

### Model download

```bash
export HF_TOKEN=hf_...   # optional but helps download speed

# Pre-download the GGUF (~22 GB for UD-Q4_K_M)
huggingface-cli download unsloth/Qwen3.6-35B-A3B-GGUF \
    --include "*Q4_K_M*" --local-dir ~/models/qwen3.6-gguf
```

## Run

```bash
# Smoke test
python niah_qwen36.py \
    --context-lengths 8192 32768 131072 \
    --depths 0.1 0.5 0.9 \
    --seeds 2

# Longer contexts (much slower per query — budget hours)
python niah_qwen36.py \
    --context-lengths 131072 262144 524288 \
    --depths 0.1 0.5 0.9 \
    --seeds 2
```

Output lands in `niah_qwen36/`:
- `niah_raw.jsonl` — per-query rows
- `niah_summary.json` — aggregated grid
- a pretty-printed recall grid on stdout at the end

## Knobs

| Flag | Default | Notes |
|---|---|---|
| `--quant` | `UD-Q4_K_M` | 22 GB, recommended. Use `UD-Q3_K_XL` (16.8 GB) or `UD-IQ3_XXS` (13.2 GB) for tighter VRAM. |
| `--context-lengths` | `8192 32768 131072` | Max should be ≤ model's native 262K unless you configure YaRN. |
| `--depths` | `0.1 0.5 0.9` | Fractional position of the needle in the prompt. |
| `--seeds` | `3` | Samples per (ctx, depth) cell. |
| `--filename` | `""` | Exact GGUF filename; defaults to pattern-match on `--quant`. |

## LongCoT-Mini reasoning benchmark

Tests the model on [LongHorizonReasoning/longcot](https://huggingface.co/datasets/LongHorizonReasoning/longcot)'s easy split (507 problems across 5 domains: logic, cs, chemistry, chess, math). Complements NIAH — NIAH is *retrieval*; LongCoT-Mini is *multi-step reasoning*.

Two processes, two tmux panes:

### Pane 1 — start the llama.cpp server

```bash
./run_server.sh
# Overridable via env:
#   MODEL_PATH=/custom/path.gguf N_CTX=131072 PORT=8000 KV_TYPE=q8_0 ./run_server.sh
```

Or run it in the background (survives disconnect):

```bash
# nohup — logs to server.log, PID saved to server.pid
nohup ./run_server.sh > server.log 2>&1 &
echo $! > server.pid
tail -f server.log               # Ctrl-C stops following; server keeps running
kill $(cat server.pid)           # stop it later

# tmux — attachable session
tmux new -d -s qwen-server './run_server.sh'
tmux attach -t qwen-server       # Ctrl-b d to detach
tmux kill-session -t qwen-server # stop
```

Check the server is up:
```bash
curl -s http://127.0.0.1:8000/v1/models
```

Auto-discovers the Q4_K_M GGUF from the HF cache. Uses 8-bit KV (`type_k/type_v=q8_0`) to fit larger contexts. Default `n_ctx=65536` is enough for Qwen3.6's reasoning traces on LongCoT-Mini prompts (≤18K input chars, ≤12K output tokens typical). Bump if you need more.

### Pane 2 — run the benchmark

```bash
# Smoke test (40 questions, ~1-2 h on H100)
uv run python longcot_mini_eval.py --domains logic math --n-per-domain 20

# Full mini (507 questions, overnight)
uv run python longcot_mini_eval.py
```

Scoring: a proxy `solution = ...` regex + substring fallback against the canonical answer JSON. Output JSONL is compatible with LongCoT's official `run_eval.py` if you want the real verifiers later.

## FSM vs. free thinking — zero-training compression experiment

Does grammar-constrained decoding alone reduce the model's verbose thinking while preserving correctness? No training, no distillation — just GBNF applied to the `<think>` block.

The script runs each HumanEval+ (or MBPP+) problem twice:

- **FREE**: standard thinking-mode generation.
- **FSM** : same model, same prompt, but the `<think>` block is grammar-constrained to emit a compact structured plan (`GOAL`/`CASES`/`APPROACH`/`EDGE`). Code after `</think>` is unconstrained.

It measures pass@1, mean thinking-token count, and total completion tokens, then reports the accuracy delta and compression ratio.

```bash
# Server needs to be running (see above).
uv run python fsm_vs_free_eval.py --n-problems 30 --dataset humaneval

# Or MBPP+
uv run python fsm_vs_free_eval.py --n-problems 50 --dataset mbpp

# FSM-only or FREE-only for debugging
uv run python fsm_vs_free_eval.py --only fsm --n-problems 5
```

The grammar lives in `fsm_grammar.gbnf` — edit it to experiment with different symbolic formats. Results land in `fsm_vs_free/`.

Key dependency: the local llama-cpp-python server must accept the `grammar` parameter in request bodies. It does by default.

**What to look for in the output:**

- If FSM `pass@1` is within ~5pp of FREE and compression is 5–10×, the model can be grammar-constrained into symbolic thinking with negligible quality loss. No training needed.
- If FSM drops ≥15pp on pass@1, the model genuinely can't reason under this constraint — grammar redesign or actual distillation would be needed.
- If FSM outputs look templated (same `GOAL/CASES/...` for very different problems), grammar is too rigid or too-narrow slots.

## Architecture context

Qwen3.6-35B-A3B is a hybrid:
- **30 of 40 layers**: Gated DeltaNet (linear-attention RNN variant, bounded state)
- **10 of 40 layers**: Standard Gated Attention (KV cache, O(L²) attention)
- 256 experts MoE (9 active per token), ~3B active params out of 35B total
- Native context **262K**, extensible to 1M via YaRN

DeltaNet's O(L) cost on most layers makes 1M context tractable. The open question is whether DeltaNet's fixed-capacity state can actually *retrieve* specific facts from early in a long context. Transformers do this via KV attention (exact but quadratic). DeltaNet's state is lossy compression — this benchmark measures how lossy.

## Expected outcomes

- **At depth=0.9** (needle near end): should be near 100% across all context lengths. The needle is within recent attention; this is the "it works" floor.
- **At depth=0.1** (needle near start) on 128K+: the informative cell. A clean drop here means DeltaNet's state is compressing away early content and external memory augmentation would help. A flat high recall means the base model is already sufficient.
