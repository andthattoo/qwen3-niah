# qwen3-niah

Needle-in-a-Haystack passkey-retrieval evaluation for [`unsloth/Qwen3.6-35B-A3B-GGUF`](https://huggingface.co/unsloth/Qwen3.6-35B-A3B-GGUF) across context lengths.

**Question**: does recall degrade as context grows on Qwen3.6's hybrid DeltaNet + Attention architecture? If recall holds at 128K-512K, the model's built-in DeltaNet state already handles long-range recall. If it drops steeply, external memory augmentation (Titans-style memory caching) has a clear job.

## Setup

Tested on an H100 with CUDA 12.4.

### Option A — uv (recommended)

```bash
uv sync
# llama-cpp-python is pulled from the prebuilt cu124 index via tool.uv.sources.
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
