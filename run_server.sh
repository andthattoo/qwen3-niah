#!/usr/bin/env bash
# Start llama-cpp-python's OpenAI-compatible server with Qwen3.6-35B-A3B-GGUF
# on GPU.  Customize env vars as needed:
#
#   MODEL_PATH  — explicit .gguf path (default: auto-discover from HF cache)
#   N_CTX       — context length to allocate (default: 65536)
#   PORT        — server port (default: 8000)
#   HOST        — bind address (default: 127.0.0.1)
#   KV_TYPE     — KV cache quant, e.g. q8_0 or q4_0 (default: q8_0)
#   N_GPU_LAYERS — -1 offloads everything to GPU (default: -1)
#
# Run in a tmux pane and leave it up; point longcot_mini_eval.py at
# http://$HOST:$PORT/v1 in another pane.

set -euo pipefail

N_CTX="${N_CTX:-65536}"
PORT="${PORT:-8000}"
HOST="${HOST:-127.0.0.1}"
KV_TYPE="${KV_TYPE:-q8_0}"
N_GPU_LAYERS="${N_GPU_LAYERS:--1}"

if [ -z "${MODEL_PATH:-}" ]; then
    MODEL_PATH="$(find "${HOME}/.cache/huggingface/hub/models--unsloth--Qwen3.6-35B-A3B-GGUF" \
        -name '*Q4_K_M*.gguf' 2>/dev/null | head -1)"
fi

if [ -z "${MODEL_PATH}" ] || [ ! -f "${MODEL_PATH}" ]; then
    echo "ERROR: could not find Qwen3.6 GGUF in HF cache."
    echo "Download it first:"
    echo "  huggingface-cli download unsloth/Qwen3.6-35B-A3B-GGUF \\"
    echo "      --include '*Q4_K_M*' --local-dir ~/models/qwen3.6-gguf"
    echo "Then set MODEL_PATH=/path/to/the.gguf and re-run this script."
    exit 1
fi

echo "Starting llama-cpp-python server"
echo "  model      = ${MODEL_PATH}"
echo "  n_ctx      = ${N_CTX}"
echo "  host:port  = ${HOST}:${PORT}"
echo "  kv_type    = ${KV_TYPE}"
echo "  gpu_layers = ${N_GPU_LAYERS}"
echo

# Make sure the CUDA libs from the venv are on the loader path.  This is the
# same trick niah_qwen36.py uses — llama-cpp-python dlopen()s libcudart.so.12
# which isn't found by default.
if [ -n "${VIRTUAL_ENV:-}" ]; then
    VENV_NVIDIA="${VIRTUAL_ENV}/lib/python3.10/site-packages/nvidia"
    if [ -d "${VENV_NVIDIA}" ]; then
        for sub in "${VENV_NVIDIA}"/*/lib; do
            if [ -d "${sub}" ]; then
                export LD_LIBRARY_PATH="${sub}:${LD_LIBRARY_PATH:-}"
            fi
        done
    fi
fi

exec python -m llama_cpp.server \
    --model "${MODEL_PATH}" \
    --n_gpu_layers "${N_GPU_LAYERS}" \
    --n_ctx "${N_CTX}" \
    --type_k "${KV_TYPE}" \
    --type_v "${KV_TYPE}" \
    --port "${PORT}" \
    --host "${HOST}"
