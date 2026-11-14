#!/usr/bin/env bash
# Copyright (c) 2026 vLLM-HUST Project. All Rights Reserved.
# Real collection uses an explicit runtime config; dry-run data is never delivered.

set -euo pipefail

OP_NAME="${1:?Specify the operator directory name}"
GATE_ENV="${2:?Specify an existing public gate as NAME=1}"
PORT="${3:-8350}"
MODEL="${4:-qwen35-35b}"
OUT_DIR="${5:-/tmp/outer_eval_${OP_NAME}}"
DRY_RUN="${DRY_RUN:-0}"

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"

if [[ "${DRY_RUN}" != "1" ]]; then
  : "${SERVING_CONFIG:?Set SERVING_CONFIG to an explicit serving_collect JSON config}"
  exec "${SERVING_PYTHON:-python3}" "${REPO_ROOT}/tools/operator_tools/serving_collect.py" \
    --config "${SERVING_CONFIG}" --output "${OUT_DIR}" \
    --gate "${GATE_ENV%%=*}" --port "${PORT}" --model "${MODEL}"
fi

echo "[DRY-RUN] Plan only; no benchmark results written."
cd "${REPO_ROOT}"
exec python3 -m tools.operator_tools.serving_loop plan \
  --op "${OP_NAME}" --gate-env "${GATE_ENV}" \
  --port "${PORT}" --model "${MODEL}"
