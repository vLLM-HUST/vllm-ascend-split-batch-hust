#!/usr/bin/env bash
# 用法: BASE_PLAN=<含 environment 的 plan.json> [RUN_ID=...] bash workloads/profile_hotspot_longrun.sh
#
# 必需环境变量: BASE_PLAN —— 提供 `config.environment`（CANN 依赖路径等）的 plan JSON，
#   无默认值；脚本从该 plan 首条带 environment 的条目取运行环境，取不到直接退出。
# 可选环境变量（下方取值即默认）:
#   PY / RUNTIME_PY / RUN_ID / RUN_DIR / ARCHIVE_ROOT / PORT_BASE /
#   PROFILE_GATE / PROFILE_ROUNDS / PROFILE_SLEEP_SECONDS
#   采集形态: MODEL_PATH / MODEL_ALIAS / TP_SIZE / MAX_MODEL_LEN /
#   MAX_NUM_SEQS / MAX_BATCHED_TOKENS / BLOCK_SIZE / GPU_MEMORY_UTILIZATION /
#   ENABLE_EXPERT_PARALLEL / CUDAGRAPH_CAPTURE_SIZES / ENABLE_CHUNKED_PREFILL /
#   SCHEDULING_POLICY / DISABLE_CUSTOM_ALL_REDUCE / SERVER_SEED /
#   COMPILATION_CONFIG /
#   SERVER_MODE（eager|graph）/ WORKLOAD / PREFIX_CACHING /
#   HOST_VLLM_COMMIT / HOST_ASCEND_COMMIT /
#   WARMUP_TOKENS / L3_TOKENS / PROFILE_TOKENS / WARMUP_REPEATS /
#   PROFILE_REQUESTS / CONCURRENCY / SHARED_PREFIX_TOKENS
# 产物落点: 仓内 $RUN_DIR（默认 <仓根>/.openbitfun/profile-hotspot-longrun-<RUN_ID>）落
#   config.json / status.json / hotspots_NN.json / nohup.log / summary.txt；
#   设备侧原始采集落仓外 $ARCHIVE_ROOT 下的 round_NN_off/。
#
# 换形态只改环境变量，不改脚本：SERVER_MODE=graph 时不传 --enforce-eager，
# 采集落进 ACLGraph 形态，并按 CUDAGRAPH_CAPTURE_SIZES 捕获图。
# 服务形态默认值逐项对齐冻结验收 profile（max-model-len 32768、
# max-num-seqs 16、max-num-batched-tokens 8192、block-size 128、
# expert parallel 开、capture sizes 1 2 4 8 16、显存水位 0.85、
# prefix caching 关、chunked prefill 开、fcfs 调度、关 custom all-reduce、
# seed 0、编译配置 FULL_DECODE_ONLY），Profile 窗口默认 16 条请求按 16 并发发出。
# 参数来源：vLLM-HUST/hust-eval-client-mmlu-pro examples/profile.b0_35b.json
# （frozen，documented command 口径，非运行回执）。
# 要回退旧的单请求 eager 形态，设：
#   SERVER_MODE=eager MAX_MODEL_LEN=4096 PROFILE_REQUESTS=1 CONCURRENCY=1
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

MODEL_PATH=${MODEL_PATH:-"/models/Qwen3.5-35B-A3B"}
MODEL_ALIAS=${MODEL_ALIAS:-"qwen35-35b"}
TP_SIZE=${TP_SIZE:-2}
# 服务形态默认值 = 冻结验收 profile；要回退旧形态就逐项改环境变量。
MAX_MODEL_LEN=${MAX_MODEL_LEN:-32768}
MAX_NUM_SEQS=${MAX_NUM_SEQS:-16}
MAX_BATCHED_TOKENS=${MAX_BATCHED_TOKENS:-8192}
BLOCK_SIZE=${BLOCK_SIZE:-128}
GPU_MEMORY_UTILIZATION=${GPU_MEMORY_UTILIZATION:-0.85}
ENABLE_EXPERT_PARALLEL=${ENABLE_EXPERT_PARALLEL:-1}
CUDAGRAPH_CAPTURE_SIZES=${CUDAGRAPH_CAPTURE_SIZES:-"1 2 4 8 16"}
ENABLE_CHUNKED_PREFILL=${ENABLE_CHUNKED_PREFILL:-1}
SCHEDULING_POLICY=${SCHEDULING_POLICY:-"fcfs"}
DISABLE_CUSTOM_ALL_REDUCE=${DISABLE_CUSTOM_ALL_REDUCE:-1}
SERVER_SEED=${SERVER_SEED:-0}
# 冻结 profile 用 --compilation-config 把图模式钉在 FULL_DECODE_ONLY；只在
# graph 模式下发，eager 模式下发只会让 manifest 撒谎。
COMPILATION_CONFIG=${COMPILATION_CONFIG:-'{"mode":3,"cudagraph_mode":"FULL_DECODE_ONLY"}'}
# 值里带引号，落进 config.json 前必须转义成 \"，否则整份 JSON 失效。
ESCAPED_CONFIG=${COMPILATION_CONFIG//\"/\\\"}
WORKLOAD=${WORKLOAD:-"random"}
SERVER_MODE=${SERVER_MODE:-"graph"}
PREFIX_CACHING=${PREFIX_CACHING:-0}
# profile 窗口的请求形状默认对齐验收客户端的并发：16 条请求、并发 16。
# 要让 prefix caching 真正起作用，profile 窗口里必须有至少两个共享前缀的请求；
# 单个请求没有可复用的前缀，只剩查找开销。
PROFILE_REQUESTS=${PROFILE_REQUESTS:-16}
CONCURRENCY=${CONCURRENCY:-16}
SHARED_PREFIX_TOKENS=${SHARED_PREFIX_TOKENS:-0}
HOST_VLLM_COMMIT=${HOST_VLLM_COMMIT:-"f18cf803c5f63625e2c71253ddaf8b0bad0bad1a"}
HOST_ASCEND_COMMIT=${HOST_ASCEND_COMMIT:-"320eced8c1bba71c0329bf59caa721a2382e7499"}

# guard 服务参数由「图模式」与「prefix caching」两个开关共同决定，两个轴都显式写出来。
# 两点教训：
#   1. 不要事后 sed 字符串——单元素列表里没有逗号，替换会静默失效；
#   2. 不能靠「去掉 --no-enable-prefix-caching」来打开缓存，vLLM 不会因此开启，
#      必须显式传 --enable-prefix-caching，否则 manifest 写着命中、实际命中率恒为 0。
if [[ "$PREFIX_CACHING" == "1" ]]; then
  CACHE_ARG="--enable-prefix-caching"
else
  CACHE_ARG="--no-enable-prefix-caching"
fi
case "$SERVER_MODE" in
  eager) GUARD_BODY="\"--enforce-eager\", \"$CACHE_ARG\"" ;;
  graph) GUARD_BODY="\"$CACHE_ARG\"" ;;
  *)
    echo "SERVER_MODE must be eager or graph, got: $SERVER_MODE" >&2
    exit 1
    ;;
esac
# 服务形态参数全部显式下发并与冻结验收 profile 对齐；capture sizes 只在
# graph 模式下传（eager 模式传了也不会有图可回放，只会让 manifest 撒谎）。
GUARD_BODY="$GUARD_BODY, \"--max-num-seqs\", \"$MAX_NUM_SEQS\""
GUARD_BODY="$GUARD_BODY, \"--max-num-batched-tokens\", \"$MAX_BATCHED_TOKENS\""
GUARD_BODY="$GUARD_BODY, \"--block-size\", \"$BLOCK_SIZE\""
GUARD_BODY="$GUARD_BODY, \"--gpu-memory-utilization\", \"$GPU_MEMORY_UTILIZATION\""
if [[ "$ENABLE_EXPERT_PARALLEL" == "1" ]]; then
  GUARD_BODY="$GUARD_BODY, \"--enable-expert-parallel\""
fi
if [[ "$ENABLE_CHUNKED_PREFILL" == "1" ]]; then
  GUARD_BODY="$GUARD_BODY, \"--enable-chunked-prefill\""
fi
GUARD_BODY="$GUARD_BODY, \"--scheduling-policy\", \"$SCHEDULING_POLICY\""
if [[ "$DISABLE_CUSTOM_ALL_REDUCE" == "1" ]]; then
  GUARD_BODY="$GUARD_BODY, \"--disable-custom-all-reduce\""
fi
if [[ "$SERVER_MODE" == "graph" ]]; then
  GUARD_BODY="$GUARD_BODY, \"--cudagraph-capture-sizes\""
  for size in $CUDAGRAPH_CAPTURE_SIZES; do
    GUARD_BODY="$GUARD_BODY, \"$size\""
  done
  GUARD_BODY="$GUARD_BODY, \"--compilation-config\", \"$ESCAPED_CONFIG\""
fi
GUARD_ARGS="[$GUARD_BODY]"
# 自检：参数串与开关必须同向，防止再次出现「manifest 说开了、服务没开」。
if [[ "$PREFIX_CACHING" == "1" && "$GUARD_ARGS" != *"--enable-prefix-caching"* ]]; then
  echo "internal error: PREFIX_CACHING=1 but guard args lack --enable-prefix-caching" >&2
  exit 1
fi
if [[ "$PREFIX_CACHING" != "1" && "$GUARD_ARGS" != *"--no-enable-prefix-caching"* ]]; then
  echo "internal error: PREFIX_CACHING=0 but guard args lack --no-enable-prefix-caching" >&2
  exit 1
fi
if [[ "$SERVER_MODE" == "graph" && "$GUARD_ARGS" != *"--cudagraph-capture-sizes"* ]]; then
  echo "internal error: SERVER_MODE=graph but guard args lack --cudagraph-capture-sizes" >&2
  exit 1
fi
if [[ "$SERVER_MODE" == "eager" && "$GUARD_ARGS" == *"--cudagraph-capture-sizes"* ]]; then
  echo "internal error: SERVER_MODE=eager must not pass --cudagraph-capture-sizes" >&2
  exit 1
fi
if [[ "$ENABLE_CHUNKED_PREFILL" == "1" && "$GUARD_ARGS" != *"--enable-chunked-prefill"* ]]; then
  echo "internal error: ENABLE_CHUNKED_PREFILL=1 but guard args lack --enable-chunked-prefill" >&2
  exit 1
fi
if [[ "$SERVER_MODE" == "graph" && "$GUARD_ARGS" != *"--compilation-config"* ]]; then
  echo "internal error: SERVER_MODE=graph but guard args lack --compilation-config" >&2
  exit 1
fi
if [[ "$SERVER_MODE" == "eager" && "$GUARD_ARGS" == *"--compilation-config"* ]]; then
  echo "internal error: SERVER_MODE=eager must not pass --compilation-config" >&2
  exit 1
fi

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
    "$MODEL_PATH",
    "--host",
    "127.0.0.1",
    "--port",
    "__PORT__",
    "--served-model-name",
    "$MODEL_ALIAS",
    "--tensor-parallel-size",
    "$TP_SIZE",
    "--seed",
    "$SERVER_SEED",
    "--max-model-len",
    "$MAX_MODEL_LEN"
  ],
  "host_commits": {
    "vllm": "$HOST_VLLM_COMMIT",
    "vllm-ascend": "$HOST_ASCEND_COMMIT"
  },
  "gate": "$GATE",
  "port": 0,
  "model_alias": "$MODEL_ALIAS",
  "model_path": "$MODEL_PATH",
  "sharegpt_path": "/models/datasets/ShareGPT/ShareGPT_V3_unfiltered_cleaned_split.json",
  "world_size": $TP_SIZE,
  "attention_layers": 30,
  "phase": "decode",
  "startup_timeout": 1200,
  "environment": $ENVIRONMENT_JSON,
  "guard_server_args": $GUARD_ARGS,
  "collection_shape": {
    "server_mode": "$SERVER_MODE",
    "prefix_caching": $PREFIX_CACHING,
    "prefix_caching_flag": "$CACHE_ARG",
    "workload": "$WORKLOAD",
    "max_model_len": $MAX_MODEL_LEN,
    "max_num_seqs": $MAX_NUM_SEQS,
    "max_num_batched_tokens": $MAX_BATCHED_TOKENS,
    "block_size": $BLOCK_SIZE,
    "gpu_memory_utilization": $GPU_MEMORY_UTILIZATION,
    "expert_parallel": $ENABLE_EXPERT_PARALLEL,
    "cudagraph_capture_sizes": "$CUDAGRAPH_CAPTURE_SIZES",
    "chunked_prefill": $ENABLE_CHUNKED_PREFILL,
    "scheduling_policy": "$SCHEDULING_POLICY",
    "disable_custom_all_reduce": $DISABLE_CUSTOM_ALL_REDUCE,
    "server_seed": $SERVER_SEED,
    "compilation_config": "$ESCAPED_CONFIG",
    "warmup_tokens": ${WARMUP_TOKENS:-16},
    "l3_tokens": ${L3_TOKENS:-128},
    "profile_tokens": ${PROFILE_TOKENS:-16},
    "warmup_repeats": ${WARMUP_REPEATS:-10},
    "profile_requests": $PROFILE_REQUESTS,
    "concurrency": $CONCURRENCY,
    "shared_prefix_tokens": $SHARED_PREFIX_TOKENS
  }
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

# 归档自述：strict 校验要求归档能说清自己是什么形态、由哪个宿主产生。
manifest = {
    "schema": "vllm-hust/collection/collection-manifest/v1",
    "produced_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
    "host_commits": template.get("host_commits", {}),
    "collection_shape": template.get("collection_shape", {}),
    "rounds": rounds,
    "artifacts": [
        "op_statistic.csv",
        "kernel_details.csv",
        "step_trace_time.csv",
        "api_statistic.csv",
        "communication_matrix.json",
    ],
}
(archive_root / "collection_manifest.json").write_text(
    json.dumps(manifest, indent=2, ensure_ascii=False) + "\n"
)

from tools.operator_tools import artifact_contract

for strict in (False, True):
    report = artifact_contract.inspect(archive_root, strict=strict)
    mode = "strict" if strict else "legacy"
    (run_dir / f"contract_{mode}.json").write_text(
        json.dumps(report.as_dict(), indent=2, ensure_ascii=False) + "\n"
    )
    print(f"contract[{mode}]: ok={report.ok} findings={len(report.findings)}", flush=True)
PY

echo $! > "$RUN_DIR/runner.pid"
printf 'PID=%s\nSTATUS=%s\nLOG=%s\n' \
  "$(cat "$RUN_DIR/runner.pid")" "$RUN_DIR/status.json" "$RUN_DIR/nohup.log" \
  | tee -a "$RUN_DIR/summary.txt"
