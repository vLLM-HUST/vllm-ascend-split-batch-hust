#!/usr/bin/env bash
# Copyright (c) 2026 vLLM-HUST Project. All Rights Reserved.
# D0 hardware constant calibration wrapper: preflight, then run the window.
#
# Usage:
#   bash experiments/ascend910b-2card/run_d0_window.sh --dry-run
#   SPLIT_BATCH_RESULTS_DIR=<abs dir outside the repo> \
#   D0_PYTHON=<device venv python> \
#   VLLM_ASCEND_REPO=<vllm-ascend checkout> VLLM_HUST_REPO=<vllm checkout> \
#     bash experiments/ascend910b-2card/run_d0_window.sh
#
# Artifacts are written outside the repo (D0 plan section "回填位置").

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"

# No absolute paths here on purpose: this file ships to other hosts, where a
# hard-coded venv or checkout location silently records the wrong fingerprint.
PYTHON="${D0_PYTHON:-python3}"
RESULTS_DIR="${SPLIT_BATCH_RESULTS_DIR:-}"
STAMP="${D0_STAMP:-$(date +%m%d-%H%M)}"
DRY_RUN=0
EXTRA_ARGS=()

for arg in "$@"; do
  case "${arg}" in
    --dry-run) DRY_RUN=1 ;;
    *) EXTRA_ARGS+=("${arg}") ;;
  esac
done

cd "${REPO_ROOT}"
export SPLIT_BATCH_RESULTS_DIR="${RESULTS_DIR}"
unset PYTHONPATH

echo "=== D0 window preflight ==="
echo "repo:     ${REPO_ROOT}"
echo "python:   ${PYTHON}"
echo "results:  ${RESULTS_DIR}/${STAMP}"
echo "dry-run:  ${DRY_RUN}"

case "${RESULTS_DIR}" in
  "") echo "FAIL: SPLIT_BATCH_RESULTS_DIR must be set to an absolute path" >&2; exit 2 ;;
  /*) ;;
  *) echo "FAIL: SPLIT_BATCH_RESULTS_DIR must be absolute" >&2; exit 2 ;;
esac
case "${RESULTS_DIR}/" in
  "${REPO_ROOT}/"*) echo "FAIL: results dir must stay outside the repo" >&2; exit 2 ;;
esac

# run_baseline_window.py overwrites artifacts, and the default stamp is only
# minute-precise; refuse to clobber an earlier window.
if [[ "${DRY_RUN}" != "1" && -e "${RESULTS_DIR}/${STAMP}" ]]; then
  if [[ "${D0_ALLOW_OVERWRITE:-0}" != "1" ]]; then
    echo "FAIL: ${RESULTS_DIR}/${STAMP} already exists; artifacts would be overwritten." >&2
    echo "      Pick another D0_STAMP, or set D0_ALLOW_OVERWRITE=1 to replace it." >&2
    exit 5
  fi
fi

for repo in "${VLLM_ASCEND_REPO:-}" "${VLLM_HUST_REPO:-}"; do
  [ -n "${repo}" ] || continue
  printf '%-32s %s %s\n' "$(basename "${repo}")" \
    "$(git -C "${repo}" rev-parse --short HEAD 2>/dev/null || echo unknown)" \
    "$(git -C "${repo}" rev-parse --abbrev-ref HEAD 2>/dev/null || echo unknown)"
done

# D0 measures hardware constants only, so a feature branch does not invalidate
# the numbers; the deviation is printed so the artifact records its provenance.
if [ -n "${VLLM_ASCEND_REPO:-}" ]; then
  host_branch="$(git -C "${VLLM_ASCEND_REPO}" rev-parse --abbrev-ref HEAD 2>/dev/null || echo unknown)"
  if [[ "${host_branch}" != "based/split-batch-baseline" ]]; then
    echo "NOTE: vllm-ascend is on '${host_branch}', not based/split-batch-baseline."
    echo "      D0 constants are hardware-only; recorded for provenance."
  fi
fi

if [[ "${DRY_RUN}" != "1" ]]; then
  if ! "${PYTHON}" -c 'import torch, torch_npu' >/dev/null 2>&1; then
    echo "FAIL: ${PYTHON} cannot import torch / torch_npu." >&2
    echo "      Set D0_PYTHON to the device venv python." >&2
    exit 1
  fi

  "${PYTHON}" - <<'PY'
import sys
from tools.op_agent.guard import check_npu_environment

status = check_npu_environment(required_cards=2)
print(f"npu:      has_npu={status.has_npu} devices={status.device_count} "
      f"tp2_ready={status.is_tp2_ready} name={status.device_name}")
if not status.has_npu or not status.is_tp2_ready:
    print(f"FAIL: preflight rejected: {status.message}", file=sys.stderr)
    raise SystemExit(3)
PY

  # The window allocates multi-GiB blocks; a co-tenant would skew the numbers
  # and can trip the ceiling probe. grep -c exits 1 on zero matches, so the
  # substitution must tolerate it under set -e.
  free_cards="$(npu-smi info | grep -c 'No running processes found' || true)"
  if [[ "${free_cards}" -lt 2 ]]; then
    echo "FAIL: NPU cards are not idle (idle markers: ${free_cards}/2)." >&2
    echo "      Stop serving workloads first, or set D0_ALLOW_BUSY=1 to override." >&2
    if [[ "${D0_ALLOW_BUSY:-0}" != "1" ]]; then
      exit 4
    fi
  fi
fi

echo "=== launching window ==="
WINDOW_ARGS=(--stamp "${STAMP}")
if [[ "${DRY_RUN}" == "1" ]]; then
  WINDOW_ARGS+=(--dry-run)
fi
WINDOW_ARGS+=(${EXTRA_ARGS[@]+"${EXTRA_ARGS[@]}"})

exec "${PYTHON}" experiments/ascend910b-2card/run_baseline_window.py "${WINDOW_ARGS[@]}"
