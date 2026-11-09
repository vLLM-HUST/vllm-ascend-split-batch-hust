#!/usr/bin/env bash
# Copyright (c) 2026 vLLM-HUST Project. All Rights Reserved.
# 一键端到端 Profiling 采集与热点子图折叠脚本（支持 CPU Dry-Run 与 2x 910B2 带卡运行）

set -euo pipefail

WORKLOAD="${1:-random}"
PORT="${2:-8350}"
MODEL="${3:-qwen35-35b}"
OUT_DIR="${4:-/root/prof/${WORKLOAD}_$(date +%Y%m%d_%H%M%S)}"
DRY_RUN="${DRY_RUN:-0}"

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"

mkdir -p "${OUT_DIR}"

echo "=========================================="
echo "=== 端到端 Profiling 采集任务启动 ==="
echo "Workload: ${WORKLOAD}"
echo "目标服务端口: ${PORT}"
echo "模型标识: ${MODEL}"
echo "Trace 存放目录: ${OUT_DIR}"
echo "Dry-Run 模式: ${DRY_RUN}"
echo "=========================================="

if [[ "${DRY_RUN}" == "1" ]]; then
  echo "[DRY-RUN] 生成合成微缩 Profiler CSV 用于离线验证..."
  python3 -c "
import csv, os
out_dir = '${OUT_DIR}'
os.makedirs(out_dir, exist_ok=True)
kd_path = os.path.join(out_dir, 'kernel_details.csv')
stat_path = os.path.join(out_dir, 'op_statistic.csv')

with open(kd_path, 'w', encoding='utf-8', newline='') as f:
    writer = csv.writer(f)
    writer.writerow(['Name', 'Type', 'Accelerator Core', 'Duration(us)', 'Wait Time(us)', 'aiv_mte2_ratio', 'aiv_mte3_ratio', 'aiv_vec_ratio', 'Input Shapes', 'Input Data Types', 'Output Shapes'])
    writer.writerow(['MatMul_1', 'MatMul', 'AI_CORE', '120.0', '0.0', '0.0', '0.0', '0.0', '[32, 2048]', 'bf16', '[32, 4096]'])
    writer.writerow(['CustomNorm_1', 'Norm', 'AI_VECTOR_CORE', '8.5', '1.2', '0.65', '0.0', '0.85', '[32, 4096]', 'bf16', '[32, 4096]'])
    writer.writerow(['CustomGate_1', 'Gate', 'AI_VECTOR_CORE', '6.2', '0.5', '0.40', '0.0', '0.90', '[32, 4096]', 'bf16', '[32, 4096]'])
    writer.writerow(['MatMul_2', 'MatMul', 'AI_CORE', '115.0', '0.0', '0.0', '0.0', '0.0', '[32, 4096]', 'bf16', '[32, 2048]'])

with open(stat_path, 'w', encoding='utf-8', newline='') as f:
    writer = csv.writer(f)
    writer.writerow(['OP Type', 'Count', 'Total Time(us)', 'Avg Time(us)'])
    writer.writerow(['MatMul', '60', '7050.0', '117.5'])
    writer.writerow(['Norm', '30', '255.0', '8.5'])
    writer.writerow(['Gate', '30', '186.0', '6.2'])
"
else
  echo "正在执行真实基准压测并触发 Profiler 采集..."
  DRY_RUN=0 bash "${REPO_ROOT}/workloads/run_workload.sh" "${WORKLOAD}" "${PORT}" "${MODEL}" "${OUT_DIR}"
fi

echo "=========================================="
echo "=== 执行子图折叠与热点分析 ==="
echo "=========================================="
SVG_OUT="${OUT_DIR}/timeline_hotspots.svg"
python3 -m tools.op_agent.timeline_insight \
    --kernel-details-csv "${OUT_DIR}/kernel_details.csv" \
    --svg-out "${SVG_OUT}"

echo "=========================================="
echo "=== 执行多算子批次候选发现 (Campaign Discover) ==="
echo "=========================================="
BATCH_OUT="${OUT_DIR}/campaign_batch.json"
python3 -m tools.op_agent.campaign discover \
    --out "${BATCH_OUT}" \
    --kernel-details-csv "${OUT_DIR}/kernel_details.csv" \
    --op-statistic-csv "${OUT_DIR}/op_statistic.csv" \
    --svg-out "${SVG_OUT}"

echo "=========================================="
echo "Profiling 采集与热点分析已全部就绪！"
echo "审核清单: ${BATCH_OUT}"
echo "泳道图预览: ${SVG_OUT}"
echo "=========================================="
