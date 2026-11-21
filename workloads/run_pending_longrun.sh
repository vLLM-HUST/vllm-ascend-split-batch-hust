#!/usr/bin/env bash
set -euo pipefail

ROOT=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)
PY=${PY:-"$ROOT/.openbitfun/mod-acceptance-venv/bin/python"}
RUN_ID=$(date -u +%Y%m%dT%H%M%SZ)
RUN_DIR=${RUN_DIR:-"$ROOT/.openbitfun/pending-longrun-$RUN_ID"}
ARCHIVE_ROOT=${ARCHIVE_ROOT:-"/root/vllm-workspace/benchmark-archives/pending-longrun-$RUN_ID"}

if [[ ! -x "$PY" ]]; then
  echo "python not found: $PY" >&2
  exit 1
fi

TEMPLATE=$(realpath -e -- "${SERVING_BASE_PLAN:?请指定已核对宿主身份的模板计划}")
mkdir -p "$RUN_DIR"
cd "$RUN_DIR"

env -u PYTHONPATH "$PY" "$ROOT/tools/operator_tools/pending_longrun.py" \
  --template "$TEMPLATE" --plan "$RUN_DIR/plan.json" --report "$RUN_DIR/readiness.json" \
  | tee "$RUN_DIR/plan-generation.log"

nohup env -u PYTHONPATH PYTHONUNBUFFERED=1 "$PY" \
  "$ROOT/tools/operator_tools/serving_e2e_auto.py" \
  --plan "$RUN_DIR/plan.json" \
  --output "$ARCHIVE_ROOT" \
  --checkpoint "$RUN_DIR/checkpoint.json" \
  --logfile "$RUN_DIR/events.log" \
  --status-file "$RUN_DIR/status.txt" \
  --offline-python "$PY" \
  --resume --continue-on-error \
  >> "$RUN_DIR/nohup.log" 2>&1 < /dev/null &

pid=$!
printf '%s\n' "$pid" > "$RUN_DIR/runner.pid"
printf 'PID=%s\nRUN_DIR=%s\nARCHIVE_ROOT=%s\nPLAN=%s\nSTATUS=%s\nLOG=%s\n' \
  "$pid" "$RUN_DIR" "$ARCHIVE_ROOT" "$RUN_DIR/plan.json" \
  "$RUN_DIR/status.txt" "$RUN_DIR/nohup.log"
