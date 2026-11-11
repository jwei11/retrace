#!/usr/bin/env bash
# Serve a VLM with vLLM (OpenAI-compatible API on 127.0.0.1:8000) from its own venv (~/.venvs/vllm),
# kept separate because vLLM pins its own torch (D-005).
# Turing: fp16 only (--dtype half), no FlashAttention; vLLM picks a supported attention backend.
#   bash scripts/serve_vllm.sh [model] [max_model_len]
source "$(dirname "$0")/wsl_env.sh"
MODEL="${1:-Qwen/Qwen3-VL-2B-Instruct}"
MAXLEN="${2:-6144}"  # 8192 doesn't fit the KV cache on 8 GB next to Windows' own VRAM use
mkdir -p "$RETRACE_DATA/logs"
exec "$HOME/.venvs/vllm/bin/vllm" serve "$MODEL" \
  --host 127.0.0.1 --port 8000 \
  --dtype half \
  --max-model-len "$MAXLEN" \
  --gpu-memory-utilization 0.85 \
  --limit-mm-per-prompt '{"image": 1}' \
  --max-num-seqs 8 \
  --seed 0
