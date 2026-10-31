#!/usr/bin/env bash
# Copyright (c) 2026 vLLM-HUST Project. All Rights Reserved.
# 标准端到端负载执行脚本

set -euo pipefail

WORKLOAD="${1:-random}"
PORT="${2:-8350}"
MODEL="${3:-qwen35-35b}"
OUT_DIR="${4:-/tmp/workload_results}"

mkdir -p "${OUT_DIR}"

echo "=========================================="
echo "执行端到端评测 Workload: ${WORKLOAD}"
echo "目标服务端口: ${PORT}"
echo "模型标识: ${MODEL}"
echo "结果输出目录: ${OUT_DIR}"
echo "=========================================="

case "${WORKLOAD}" in
  random)
    echo "启动 random-input 标准受控负载 (1024 in / 256 out, 并发 64)..."
    vllm bench serve \
        --backend openai-chat \
        --model "${MODEL}" \
        --dataset-name random \
        --random-input-len 1024 \
        --random-output-len 256 \
        --num-prompts 256 \
        --max-concurrency 64 \
        --port "${PORT}" \
        --percentile-metrics ttft,tpot,itl \
        --save-result \
        --result-dir "${OUT_DIR}" \
        --result-filename "result_random_$(date +%Y%m%d_%H%M%S).json"
    ;;
  sharegpt)
    echo "启动 sharegpt 真实对话负载 (长尾分布, 并发 64)..."
    DATASET_PATH="/data/shared_models/huggingface_cache/hub/datasets--anon8231489123--ShareGPT_Vicuna_unfiltered"
    if [ ! -d "${DATASET_PATH}" ]; then
      echo "提示: 未检测到本地 sharegpt 数据集路径，使用在线/通用 sharegpt 模式"
      DATASET_ARGS=(--dataset-name sharegpt)
    else
      DATASET_ARGS=(--dataset-name sharegpt --dataset-path "${DATASET_PATH}")
    fi
    vllm bench serve \
        --backend openai-chat \
        --model "${MODEL}" \
        "${DATASET_ARGS[@]}" \
        --num-prompts 256 \
        --max-concurrency 64 \
        --port "${PORT}" \
        --percentile-metrics ttft,tpot,itl \
        --save-result \
        --result-dir "${OUT_DIR}" \
        --result-filename "result_sharegpt_$(date +%Y%m%d_%H%M%S).json"
    ;;
  long-prefix)
    echo "启动 long-prefix 前缀复用负载 (长前缀 2048 / 增量 128 / 输出 128)..."
    # long-prefix: 使用较长输入与短输出模拟长上下文/KV复用场景
    vllm bench serve \
        --backend openai-chat \
        --model "${MODEL}" \
        --dataset-name random \
        --random-input-len 2048 \
        --random-output-len 128 \
        --num-prompts 128 \
        --max-concurrency 32 \
        --port "${PORT}" \
        --percentile-metrics ttft,tpot,itl \
        --save-result \
        --result-dir "${OUT_DIR}" \
        --result-filename "result_long_prefix_$(date +%Y%m%d_%H%M%S).json"
    ;;
  *)
    echo "错误: 未知 Workload '${WORKLOAD}'。可选: random, sharegpt, long-prefix"
    exit 1
    ;;
esac

echo "Workload 压测完成，结果已保存至 ${OUT_DIR}"
