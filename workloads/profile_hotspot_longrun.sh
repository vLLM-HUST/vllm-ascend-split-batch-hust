#!/usr/bin/env bash
# 用法: BASE_PLAN=<含 environment 的 plan.json> [RUN_ID=...] bash workloads/profile_hotspot_longrun.sh
#
# 必需环境变量: BASE_PLAN —— 提供 `config.environment`（CANN 依赖路径等）的 plan JSON，
#   无默认值；脚本从该 plan 首条带 environment 的条目取运行环境，取不到直接退出。
# 可选环境变量: PY / RUNTIME_PY / RUN_ID / RUN_DIR / ARCHIVE_ROOT / PORT_BASE /
#   PROFILE_GATE / PROFILE_ROUNDS / PROFILE_SLEEP_SECONDS（下方取值即默认）。
# 产物落点: 仓内 $RUN_DIR（默认 <仓根>/.openbitfun/profile-hotspot-longrun-<RUN_ID>）落
#   config.json / status.json / hotspots_NN.json / nohup.log / summary.txt；
#   设备侧原始采集落仓外 $ARCHIVE_ROOT 下的 round_NN_off/。
set -euo pipefail

ROOT=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)
PY=${PY:-"$ROOT/.openbitfun/mod-acceptance-venv/bin/python"}
RUNTIME_PY=${RUNTIME_PY:-"/root/venvs/split-batch-runtime/bin/python"}
RUN_ID=${RUN_ID:-$(date -u +%Y%m%dT%H%M%SZ)}
RUN_DIR=${RUN_DIR:-"$ROOT/.openbitfun/profile-hotspot-longrun-$RUN_ID"}
ARCHIVE_ROOT=${ARCHIVE_ROOT:-"/root/vllm-workspace/benchmark-archives/profile-hotspot-longrun-$RUN_ID"}
PORT_BASE=${PORT_BASE:-8350}
BASE_PLAN=${BASE_PLAN:?需显式指定 BASE_PLAN（提供 config.environment 的 plan JSON），无默认值}
GATE=${PROFILE_GATE:-"VLLM_HUST_GDN_SPLIT_BA"}
ROUNDS=${PROFILE_ROUNDS:-3}
SLEEP_SECONDS=${PROFILE_SLEEP_SECONDS:-30}

if [[ ! -x "$PY" || ! -x "$RUNTIME_PY" ]]; then
  echo "python not found: py=$PY runtime=$RUNTIME_PY" >&2
  exit 1
fi

if [[ ! -f "$BASE_PLAN" ]]; then
  echo "base plan not found: $BASE_PLAN" >&2
  exit 1
fi

ENVIRONMENT_JSON=$(env -u PYTHONPATH "$PY" - "$BASE_PLAN" <<'PY'
import json
import sys
plan = json.load(open(sys.argv[1]))
for entry in plan.get("items") or plan.get("operators") or []:
    config = entry.get("config") if isinstance(entry, dict) else None
    if isinstance(config, dict) and config.get("environment"):
        print(json.dumps(config["environment"], ensure_ascii=False))
        break
else:
    raise SystemExit("base plan has no runtime environment")
PY
)

mkdir -p "$RUN_DIR" "$ARCHIVE_ROOT"
cd "$RUN_DIR"

cat > config.json <<JSON
{
  "python": "$RUNTIME_PY",
  "server_command": [
    "$RUNTIME_PY",
    "-m",
    "vllm.entrypoints.cli.main",
    "serve",
    "/models/Qwen3.5-35B-A3B",
    "--host",
    "127.0.0.1",
    "--port",
    "__PORT__",
    "--served-model-name",
    "qwen35-35b",
    "--tensor-parallel-size",
    "2",
    "--seed",
    "42",
    "--max-model-len",
    "4096"
  ],
  "host_commits": {
    "vllm": "f18cf803c5f63625e2c71253ddaf8b0bad0bad1a",
    "vllm-ascend": "320eced8c1bba71c0329bf59caa721a2382e7499"
  },
  "gate": "$GATE",
  "port": 0,
  "model_alias": "qwen35-35b",
  "model_path": "/models/Qwen3.5-35B-A3B",
  "sharegpt_path": "/models/datasets/ShareGPT/ShareGPT_V3_unfiltered_cleaned_split.json",
  "world_size": 2,
  "attention_layers": 30,
  "phase": "decode",
  "startup_timeout": 1200,
  "environment": $ENVIRONMENT_JSON,
  "guard_server_args": [
    "--enforce-eager",
    "--no-enable-prefix-caching"
  ]
}
JSON

printf 'RUN_DIR=%s\nARCHIVE_ROOT=%s\nGATE=%s\nROUNDS=%s\n' \
  "$RUN_DIR" "$ARCHIVE_ROOT" "$GATE" "$ROUNDS" | tee summary.txt

env -u PYTHONPATH PYTHONUNBUFFERED=1 "$PY" - \
  "$ROOT" "$RUN_DIR" "$ARCHIVE_ROOT" "$PORT_BASE" "$ROUNDS" "$SLEEP_SECONDS" \
  >> "$RUN_DIR/nohup.log" 2>&1 <<'PY' &
import json
import os
import sys
import time
from pathlib import Path

root = Path(sys.argv[1])
run_dir = Path(sys.argv[2])
archive_root = Path(sys.argv[3])
port_base = int(sys.argv[4])
rounds = int(sys.argv[5])
sleep_seconds = int(sys.argv[6])

sys.path.insert(0, str(root))
from tools.operator_tools import serving_collect

template = json.loads((run_dir / "config.json").read_text())

for index in range(rounds):
    port = port_base + index
    config = dict(template)
    config["port"] = port
    config["server_command"] = list(template["server_command"])
    config["server_command"][config["server_command"].index("__PORT__")] = str(port)
    output = archive_root / f"round_{index:02d}_off"
    (run_dir / "status.json").write_text(
        json.dumps({"round": index, "state": "running", "output": str(output)}, indent=2) + "\n"
    )
    try:
        serving_collect.collect_guard(config, output, "off")
        analysis_root = output / "profile"
        totals = {}
        for path in analysis_root.rglob("op_statistic.csv"):
            import csv
            with path.open(newline="") as handle:
                for row in csv.DictReader(handle):
                    op = row["OP Type"]
                    slot = totals.setdefault(op, {"count": 0, "us": 0.0})
                    slot["count"] += int(row["Count"])
                    slot["us"] += float(row["Total Time(us)"])
        grand = sum(v["us"] for v in totals.values()) or 1.0
        hotspots = [
            {"op": op, "count": v["count"], "total_us": round(v["us"], 3),
             "share_pct": round(v["us"] * 100 / grand, 3)}
            for op, v in totals.items()
        ]
        hotspots.sort(key=lambda x: x["total_us"], reverse=True)
        (run_dir / f"hotspots_{index:02d}.json").write_text(
            json.dumps(hotspots, indent=2) + "\n"
        )
        (run_dir / "status.json").write_text(
            json.dumps({"round": index, "state": "done", "output": str(output)}, indent=2) + "\n"
        )
    except Exception as exc:
        (run_dir / "status.json").write_text(
            json.dumps({"round": index, "state": "failed", "error": repr(exc),
                        "output": str(output)}, indent=2) + "\n"
        )
    if index + 1 < rounds:
        time.sleep(sleep_seconds)
PY

echo $! > "$RUN_DIR/runner.pid"
printf 'PID=%s\nSTATUS=%s\nLOG=%s\n' \
  "$(cat "$RUN_DIR/runner.pid")" "$RUN_DIR/status.json" "$RUN_DIR/nohup.log" \
  | tee -a "$RUN_DIR/summary.txt"
