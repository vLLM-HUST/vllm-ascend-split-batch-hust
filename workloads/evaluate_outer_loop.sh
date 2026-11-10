#!/usr/bin/env bash
# Copyright (c) 2026 vLLM-HUST Project. All Rights Reserved.
# Real collection uses an explicit runtime config; dry-run data is never delivered.

set -euo pipefail

OP_NAME="${1:-gdn_qkv_norm}"
GATE_ENV="${2:-VLLM_HUST_GDN_QKV_NORM=1}"
PORT="${3:-8350}"
MODEL="${4:-qwen35-35b}"
OUT_DIR="${5:-/tmp/outer_eval_${OP_NAME}}"
DRY_RUN="${DRY_RUN:-0}"

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"

if [[ "${DRY_RUN}" != "1" ]]; then
  : "${SERVING_CONFIG:?Set SERVING_CONFIG to an explicit serving_collect JSON config}"
  exec "${SERVING_PYTHON:-python3}" "${REPO_ROOT}/tools/op_agent/serving_collect.py" \
    --config "${SERVING_CONFIG}" --output "${OUT_DIR}" \
    --gate "${GATE_ENV%%=*}" --port "${PORT}" --model "${MODEL}"
fi

mkdir -p "${OUT_DIR}"
OUT_DIR="$(cd "${OUT_DIR}" && pwd)"
cd "${REPO_ROOT}"
echo "[DRY-RUN] Synthetic collection only; operator evidence will not be updated."
python3 -m tools.op_agent.serving_loop plan \
    --op "${OP_NAME}" --gate-env "${GATE_ENV}" \
    --port "${PORT}" --model "${MODEL}" > "${OUT_DIR}/plan.json"

BASE_GREEDY="${OUT_DIR}/baseline_greedy.txt"
CAND_GREEDY="${OUT_DIR}/candidate_greedy.txt"
cat << 'EOF' > "${BASE_GREEDY}"
Paging divides memory into fixed-size pages while segmentation uses variable-sized segments.
Binary heap construction takes O(N) time using bottom-up sift-down.
Prefix sums can be computed in O(N) by running an accumulator across the array.
Write-through updates memory immediately while write-back delays until eviction.
TCP congestion control transitions upon reaching ssthresh.
RMSNorm avoids mean centering and relies strictly on root-mean-square scaling.
Two-phase locking ensures serializability by separating growing and shrinking phases.
Kahn's algorithm performs topological sorting by repeatedly picking in-degree 0 nodes.
EOF
cp "${BASE_GREEDY}" "${CAND_GREEDY}"
python3 -m tools.op_agent.serving_loop compare-greedy \
    --baseline "${BASE_GREEDY}" --candidate "${CAND_GREEDY}"

STAT_CSV="${OUT_DIR}/op_statistic.csv"
cat << 'EOF' > "${STAT_CSV}"
OP Type,Count,Total Time(us),Avg Time(us)
MatMul,30,3000.0,100.0
_qkv_norm_kernel,30,180.0,6.0
EOF
python3 -m tools.op_agent.serving_loop parse-l4 \
    --csv "${STAT_CSV}" --kernel "_qkv_norm" --expected-calls 30

OFF_1="${OUT_DIR}/off_run1.json"
OFF_2="${OUT_DIR}/off_run2.json"
ON_1="${OUT_DIR}/on_run1.json"
ON_2="${OUT_DIR}/on_run2.json"
echo '{"output_throughput": 760.0, "mean_tpot_ms": 37.1, "mean_ttft_ms": 1250.0}' > "${OFF_1}"
echo '{"output_throughput": 762.0, "mean_tpot_ms": 37.0, "mean_ttft_ms": 1245.0}' > "${OFF_2}"
echo '{"output_throughput": 782.0, "mean_tpot_ms": 36.1, "mean_ttft_ms": 1210.0}' > "${ON_1}"
echo '{"output_throughput": 784.0, "mean_tpot_ms": 36.0, "mean_ttft_ms": 1205.0}' > "${ON_2}"
python3 -m tools.op_agent.serving_loop summarize-ab \
    --workload random --off "${OFF_1}" "${OFF_2}" --on "${ON_1}" "${ON_2}"
