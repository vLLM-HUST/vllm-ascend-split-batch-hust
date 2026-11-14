#!/usr/bin/env bash
# Copyright (c) 2026 vLLM-HUST Project. All Rights Reserved.
# 标准端到端服务评测负载执行入口（供外环 serving_loop 与手工验收统一调用）

set -euo pipefail

WORKLOAD="${1:-random}"
PORT="${2:-8350}"
MODEL="${3:-qwen35-35b}"
OUT_DIR="${4:-/tmp/workload_results}"
DRY_RUN="${DRY_RUN:-0}"
CONCURRENCY="${CONCURRENCY:-}"
NUM_PROMPTS="${NUM_PROMPTS:-}"
TAG="${RUN_TAG:-$(date +%Y%m%d_%H%M%S)}"

mkdir -p "${OUT_DIR}"

find_sharegpt_dataset() {
  if [[ -n "${VLLM_HUST_DATASET_SHAREGPT:-}" && ! -f "${VLLM_HUST_DATASET_SHAREGPT}" ]]; then
    echo "Invalid VLLM_HUST_DATASET_SHAREGPT path" >&2
    return 1
  fi
  local candidates=(
    "${VLLM_HUST_DATASET_SHAREGPT:-}"
    "/models/datasets/ShareGPT/ShareGPT_V3_unfiltered_cleaned_split.json"
    "/data/vllm-hust-benchmark/scripts/traces/evoscientist-workload-sharegpt.json"
    "/vllm-workspace/bench/datasets/sharegpt/ShareGPT_V3_unfiltered_cleaned_split.json"
    "/root/vllm-workspace/bench/datasets/sharegpt/ShareGPT_V3_unfiltered_cleaned_split.json"
  )
  for p in "${candidates[@]}"; do
    if [[ -n "${p}" && -f "${p}" ]]; then
      echo "${p}"
      return 0
    fi
  done
  return 1
}

echo "=========================================="
echo "执行端到端评测 Workload: ${WORKLOAD}"
echo "目标服务端口: ${PORT}"
echo "模型标识: ${MODEL}"
echo "结果输出目录: ${OUT_DIR}"
echo "=========================================="

case "${WORKLOAD}" in
  random)
    IN_LEN="${RANDOM_INPUT_LEN:-512}"
    OUT_LEN="${RANDOM_OUTPUT_LEN:-256}"
    NP="${NUM_PROMPTS:-128}"
    CONC="${CONCURRENCY:-32}"
    echo "启动 random 受控定长负载 (in=${IN_LEN}, out=${OUT_LEN}, prompts=${NP}, conc=${CONC})..."
    BENCH_CMD=(
      vllm bench serve
      --model "${MODEL}"
      --tokenizer "${VLLM_HUST_TOKENIZER:-${MODEL}}"
      --dataset-name random
      --random-input-len "${IN_LEN}"
      --random-output-len "${OUT_LEN}"
      --ignore-eos
      --num-prompts "${NP}"
      --max-concurrency "${CONC}"
      --port "${PORT}"
      --percentile-metrics ttft,tpot,itl
      --save-result
      --result-dir "${OUT_DIR}"
      --result-filename "result_random_${TAG}.json"
    )
    if [[ "${DRY_RUN}" == "1" ]]; then
      echo "[DRY-RUN] ${BENCH_CMD[*]}"
    else
      "${BENCH_CMD[@]}"
    fi
    ;;
  sharegpt)
    NP="${NUM_PROMPTS:-256}"
    CONC="${CONCURRENCY:-32}"
    if DATASET_PATH="$(find_sharegpt_dataset)"; then
      echo "检测到本地 ShareGPT 数据集: ${DATASET_PATH}"
      DATASET_ARGS=(--dataset-name sharegpt --dataset-path "${DATASET_PATH}")
    else
      if [[ -n "${VLLM_HUST_DATASET_SHAREGPT:-}" || "${DRY_RUN}" != "1" ]]; then
        echo "Set VLLM_HUST_DATASET_SHAREGPT to an existing dataset file" >&2
        exit 1
      fi
      DATASET_ARGS=(--dataset-name sharegpt)
    fi
    BENCH_CMD=(
      vllm bench serve
      --model "${MODEL}"
      --tokenizer "${VLLM_HUST_TOKENIZER:-${MODEL}}"
      "${DATASET_ARGS[@]}"
      --num-prompts "${NP}"
      --max-concurrency "${CONC}"
      --port "${PORT}"
      --percentile-metrics ttft,tpot,itl
      --save-result
      --result-dir "${OUT_DIR}"
      --result-filename "result_sharegpt_${TAG}.json"
    )
    if [[ "${DRY_RUN}" == "1" ]]; then
      echo "[DRY-RUN] ${BENCH_CMD[*]}"
    else
      "${BENCH_CMD[@]}"
    fi
    ;;
  long-prefix)
    IN_LEN="${LONG_INPUT_LEN:-2048}"
    OUT_LEN="${LONG_OUTPUT_LEN:-128}"
    NP="${NUM_PROMPTS:-128}"
    CONC="${CONCURRENCY:-32}"
    echo "启动 long-prefix 长上下文负载 (in=${IN_LEN}, out=${OUT_LEN}, prompts=${NP}, conc=${CONC})..."
    BENCH_CMD=(
      vllm bench serve
      --model "${MODEL}"
      --tokenizer "${VLLM_HUST_TOKENIZER:-${MODEL}}"
      --dataset-name random
      --random-input-len "${IN_LEN}"
      --random-output-len "${OUT_LEN}"
      --ignore-eos
      --num-prompts "${NP}"
      --max-concurrency "${CONC}"
      --port "${PORT}"
      --percentile-metrics ttft,tpot,itl
      --save-result
      --result-dir "${OUT_DIR}"
      --result-filename "result_long_prefix_${TAG}.json"
    )
    if [[ "${DRY_RUN}" == "1" ]]; then
      echo "[DRY-RUN] ${BENCH_CMD[*]}"
    else
      "${BENCH_CMD[@]}"
    fi
    ;;
  *)
    echo "错误: 未知 Workload '${WORKLOAD}'。可选: random, sharegpt, long-prefix" >&2
    exit 1
    ;;
esac

if [[ "${DRY_RUN}" == "1" ]]; then
  echo "Dry-run only; no benchmark results written."
else
  echo "Workload 压测完成，结果已保存至 ${OUT_DIR}"
fi
