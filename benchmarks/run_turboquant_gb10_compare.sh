#!/usr/bin/env bash
set -euo pipefail

# Compare GB10 TurboQuant against the existing FP8 KV-cache path on the same
# long-context workload.

VENV_BIN="${VENV_BIN:-.venv.clean/bin}"
MODEL="${MODEL:-cyankiwi/Qwen3.5-27B-AWQ-4bit}"
CUDA_DEVICE="${CUDA_DEVICE:-0}"
MAX_MODEL_LEN="${MAX_MODEL_LEN:-262144}"
GPU_MEMORY_UTILIZATION="${GPU_MEMORY_UTILIZATION:-0.70}"
MAX_NUM_BATCHED_TOKENS="${MAX_NUM_BATCHED_TOKENS:-16384}"
MAX_NUM_SEQS="${MAX_NUM_SEQS:-64}"
DOCUMENT_LENGTH="${DOCUMENT_LENGTH:-32768}"
NUM_DOCUMENTS="${NUM_DOCUMENTS:-4}"
REPEAT_COUNT="${REPEAT_COUNT:-2}"
OUTPUT_LEN="${OUTPUT_LEN:-32}"

export LD_LIBRARY_PATH="/usr/local/cuda/targets/sbsa-linux/lib:${LD_LIBRARY_PATH:-}"
export PYTORCH_ALLOC_CONF="${PYTORCH_ALLOC_CONF:-expandable_segments:True}"
export CUDA_VISIBLE_DEVICES="${CUDA_DEVICE}"

COMMON_ARGS=(
  --model "${MODEL}"
  --tensor-parallel-size 1
  --max-model-len "${MAX_MODEL_LEN}"
  --gpu-memory-utilization "${GPU_MEMORY_UTILIZATION}"
  --max-num-batched-tokens "${MAX_NUM_BATCHED_TOKENS}"
  --max-num-seqs "${MAX_NUM_SEQS}"
  --attention-backend TRITON_ATTN
  --enable-chunked-prefill
  --enable-prefix-caching
  --document-length "${DOCUMENT_LENGTH}"
  --num-documents "${NUM_DOCUMENTS}"
  --repeat-count "${REPEAT_COUNT}"
  --output-len "${OUTPUT_LEN}"
)

echo "=== FP8 baseline ==="
"${VENV_BIN}/python" benchmarks/benchmark_long_document_qa_throughput.py \
  "${COMMON_ARGS[@]}" \
  --kv-cache-dtype fp8

echo "=== TurboQuant35 mode ==="
"${VENV_BIN}/python" benchmarks/benchmark_long_document_qa_throughput.py \
  "${COMMON_ARGS[@]}" \
  --kv-cache-dtype turboquant35 \
  --enable-turboquant
