#!/usr/bin/env bash
# bench-ladder —— 算子优化验证阶梯（0.28 主栈, 分钟级出信号）
#
# 用法: bench-ladder.sh <command> [args]
#   serve                 启动 plain 固定协议 serve，供 screen/L1/L2/L3；不得拿 profile serve 的结果判收益
#   serve --profile [light|deep]
#                         启动 L4 专用 serve。light 默认固定 32 worker iterations、无 shape/stack；
#                         deep 仅用于机制缺口，默认完整窗口 + shape。两档都必须先 stop 再切换。
#   stop                  停 ladder serve
#   wait [timeout_s]      等待 /health 就绪（默认 1800s）
#   l1 [iters]            L1 单请求延迟（~1 min, 对 per-op 改进最敏感）
#   l2 [prompts] [conc]   L2 并发吞吐突发（默认 128 题 2048/256 @ 并发32, ~3-5 min; 0.28 里 range-ratio 0=定长）
#                         LADDER_L2_LOAD=synthetic(默认)|mmlu-pro；mmlu-pro 需 LADDER_L2_DATASET=<prompt jsonl>
#                         （export-mmlupro-prompts.py 产出；形状 = 5-shot CoT 展平 + thinking 长输出 @并发5）
#                         产物带 load_caliber/cohort（ladder-stamp.py）；两形态只比方向不比数值
#   l3                    L3 数值护栏捕获（8 条固定 prompt 贪心输出）
#   compare <A> <B>       L3 对比两次捕获（不一致 = 数值回归）
#   smoke [N]             MMLU-Pro smoke N 题（默认 4, harness venv lm_eval）
#   screen <tag>          plain serve 晋级筛选：L1×3 + L2×3 + L3×1，生成 screen.json；
#                         LADDER_LOAD_CALIBER 可选（v4：仅导航，不再作门票）
#                         screen.json 还带 carriers（载体记录轴: find_spec + vllm.general_plugins
#                         入口点 + VLLM_HUST_* + 本 serve 日志 marker 计数; 只记录不当门）
#   carriers [serve_log]  打印载体状态记录（单行 JSON）; serve 就绪后也会自动打人读摘要
#   profile --tier light|deep [load_n]
#                         L4 窗口：light 只交付目标算子/all-kernel/通信并删 DB；deep 额外交付
#                         per-shape×dtype。两档都在交付后立即离线解析和瘦身。
#                         负载口径 LADDER_PROFILE_LOAD=synthetic(默认)|mmlu-pro|external（后者用
#                         LADDER_PROFILE_LOAD_CMD 跑任意自定负载）；
#                         产物带形状事实(window_coverage)与权重标签(cohort)；不同形状/权重只比方向
#   slim <asc|窗目录> [ts] [--opstat <p>]  解析后瘦身：抢救 csv + pt db + communication*.json 到
#                         records/raw/<窗>/ + 删中间物（默认; LADDER_SLIM=0 关; LADDER_SLIM_RESCUE_DB=0 关 pt db 抢救）
#                         交付门（默认保护, **逐 rank**, 无 force 逃生口）: 未交付 rank（输入不全 or
#                         无**关联有效 opstat**=load_caliber 已登记且非 unmarked）一律保留不删, 只瘦身
#                         已交付 rank; 无可瘦身 rank 即不删并返回非零 → 见 recover-profile
#   recover-profile <窗目录|asc_pt> [ts] [--opstat <p>] [--caliber <id>]
#                         只为**已有完整产物**的窗口生成/补齐 L4 opstat（离线、无 NPU、**无覆盖**）;
#                         不原地重解析（不覆盖既有 csv/db）; 口径须已登记（缺失请显式给）; 输入不完整
#                         则明确失败并说明"需另行受控解析", 不写任何文件
#   l0 [op] [bench-l0 flags]  L0 算子微基准（bench-l0.py; op=all 列全部;
#                             余下参数原样透传, 如 --sig 4 --decode-tokens 11912 --iters 100）
#                             op=GdnQkvUnpack 走 tools/bench_gdn_qkv_unpack.py
#                             （rearrange_mixed_qkv 配对 A/B; 落 <ts>-l0-gdn-qkv-unpack.json）
#   run <tag> [steps]     一条龙: l1,l2,l3,smoke -> records/ladder/<tag>/（默认全四步）
#                         产出: run.json 索引 + steps.log + smoke 目录 + 单页实验门禁报告
#                         （不再拷贝 l1/l2/l3 json; 权威产物是 records/ladder/<ts>-*.json）
#   report <A> <B> [--json] 单页实验门禁报告（L1/L2/L3/L4/smoke + 口径/cohort 前提与告警;
#                         L3 不一致退出码 3, 可直接当门禁）
#   logs [--dry-run]     日志收尾: gzip 已完结的 ladder-*.log（保留最近 LADDER_LOG_KEEP_DAYS=3 天
#                        未压缩; 被 L2/KB 按绝对路径引用的见 logs/.log-cited, 一律跳过）
#
# 纪律（S2/S3 运营项, 内置不省略）:
#   - console script `vllm serve`（python -m vllm 入口已移除）
#   - TORCHINDUCTOR_FX_GRAPH_CACHE=0 TORCHINDUCTOR_AUTOGRAD_CACHE=0（torch 2.13 AOT 缓存崩溃）
#   - --no-enable-prefix-caching（与 S3 基线同态）
#   - VLLM_HUST_GATE_FP32=1（2026-10-14 固化: D4 gate-fp32 权重外提, 吞吐侧; 见 L1 卡
#     gate_fp32_full_graph_ab_1014: L2 +2.58% 3/3 对, 设备 Cast[256,2048]->0 -148ms/窗, L3 8/8。
#     关: 显式 VLLM_HUST_GATE_FP32=0。载体(husk 插件)缺失时 fail-open, 不影响起服）
#   - L0/评测单变量: greedy + 固定 seed; 慢命令 setsid 后台落日志
#   - 离线 trace 解析: LADDER_PARSE_PYTHON（默认 serving python, torch_npu 自带 analyse）
#     + ascend_pt 父目录口径 + ulimit -v 45G（cgroup 64G, 与 serve 错峰;
#     要解析前停 serve 设 LADDER_STOP_SERVE_FOR_PARSE=1, 默认 0）
set -uo pipefail

SCRIPTS=/root/包体恢复/scripts
RECORDS=/root/包体恢复/records/ladder
LOGS=/root/包体恢复/logs
PY=/usr/local/python3.12.13/bin/python
VLLM=/usr/local/python3.12.13/bin/vllm
HARNESS_LM_EVAL=/root/venvs/harness/bin/lm_eval
PARSE_PYTHON=${LADDER_PARSE_PYTHON:-$PY}
MODEL=${LADDER_MODEL:-/models/Qwen3.5-35B-A3B}
# 路径兜底: 2026-10-13 实测 /models/Qwen3.5-35B-A3B 已不存在（模型在 modelscope 缓存里）,
# 少了这条会让 serve/L1/L2/L3/L4 全部起不来, 却只报"模型找不到"。
if [ ! -d "$MODEL" ]; then
  for cand in /models/modelscope_cache/Qwen/Qwen3.5-35B-A3B; do
    if [ -d "$cand" ]; then
      echo "bench-ladder: $MODEL 不存在, 回退 -> $cand （可用 LADDER_MODEL=... 覆盖）" >&2
      MODEL=$cand
      break
    fi
  done
fi
PORT=8000
# 让本脚本拉起的客户端子进程（ladder-l1.py / ladder-l3.py）用同一个模型 id
export LADDER_MODEL="$MODEL"
LADDER_PIDFILE=/tmp/ladder-serve.pid
SERVE_MODE_FILE=/tmp/ladder-serve-mode.json
# prof 目录指针: 主写 PVC（容器重建后仍在，可续解析既有 trace）；/tmp 仅作回退。
PROF_PTR_PVC="$RECORDS/.current-prof-dir"   # /root/包体恢复/records/ladder/.current-prof-dir
PROF_PTR_TMP=/tmp/ladder-prof-dir           # 旧位置（overlay, 重建即失效）, 只读回退
mkdir -p "$RECORDS" "$LOGS"

# 双 inductor 缓存禁用 —— serve 与 bench 客户端一律生效
export TORCHINDUCTOR_FX_GRAPH_CACHE=0
export TORCHINDUCTOR_AUTOGRAD_CACHE=0

serve_pid() { [ -f "$LADDER_PIDFILE" ] && cat "$LADDER_PIDFILE" 2>/dev/null || true; }

health_ok() { curl -sf "http://127.0.0.1:${PORT}/health" >/dev/null 2>&1; }

# ---------------------------------------------------------------------------
# 载体探针（carriers）: plain screen / serve 的**载体状态记录**（2026-10-17 加）
#
# 为什么加: plain screen 的 screen.json 原来完全没有载体记录, 于是唯一能看的东西是 vllm 的
#   `Unknown vLLM environment variable detected: VLLM_HUST_*` 警告 —— 而该警告在**载体存在与
#   不存在两种状态下都会出现**（/root/包体恢复/logs/ladder-serve-1017-081645.log:18-19 那份日志
#   的载体已生效, 同样有这两行）→ 它不是缺席判据, 拿它判缺席会得出反向结论。
# 可靠判据（本探针记录的三件）:
#   ① 服务实际 Python 的 find_spec('vllm_ascend_split_batch')（解释器优先
#      /usr/local/python3.12.13/bin/python = serve 用的那个; 不要用 /usr/bin/python3, 它看不到这个包）;
#   ② vllm.general_plugins 里 husk 入口点（name/value 全量列出; 装不上/未注册等价于未生效）;
#   ③ 本 serve 日志里的功能 marker 计数（cascade plugin loaded / Gate fp32 cache populated /
#      W3F2/W3F3 补丁行 / gdn merge fail-closed 警告）。
# 环境值从哪来（2026-10-17 修正 F1）: 之前探针读**调用 shell** 的 os.environ, 而 `opt bench screen`
#   的调用 shell 未必继承 cmd_serve 里的 `export VLLM_HUST_*=1` → 会写出"两个键为空 + marker
#   effective"的自证冲突。现在 env 只认 **serve 侧快照**：cmd_serve 起服时把 export 后的
#   `VLLM_HUST_*` 原子写到与 serve log 同 stem 的 `<log>.env.json`（`opt-ops-serve-env.v1`），
#   探针按 serve_log 关联读取; 读不到 → env 值一律 null + env_scope/env_source=unknown +
#   probe_errors（**不用空字符串冒充 off**）。
# 归因拆分（2026-10-17 修正 F2）: `attribution` 把「仓内 husk 插件」与「运行树补丁」**分开**记
#   （husk.cascade / husk.gate_fp32 / husk.gdn_merge_inproj / runtime_tree.w3f2 / runtime_tree.w3f3）,
#   **不再**用跨载体三元组混算一个 functional_markers。gdn_merge_inproj 仓内是
#   `present-but-fail-closed`（宿主没有可安全消费合并权重的公开 seam, 见 HOST_CONTRACT.md GDN 段）,
#   运行树 W3F2 若 applied 是**另一件事**, 两者共存但不得归因为插件做了合并。
# 纪律（/root/opt-ops/AGENTS.md 实验纪律）: 载体状态是**记录轴不是门禁** → 本探针只写记录,
#   **不返回失败码、不阻断 screen/serve**; 探针自身出错（包缺席 / 入口点枚举报错 / 日志或 env 文件缺失）
#   一律 fail-closed 记 unknown/null/false, 绝不崩调用方。
# 覆盖: LADDER_CARRIER_PYTHON=<解释器>（默认 /usr/local/python3.12.13/bin/python, 缺失回退 $PY）、
#   LADDER_SERVE_LOG=<日志>（默认读 /tmp/ladder-serve-log, 由 cmd_serve 写入）、
#   LADDER_SERVE_ENV=<env.json>（默认由 serve_log 推导 <log>.env.json）。
carrier_python() {
  if [ -n "${LADDER_CARRIER_PYTHON:-}" ]; then printf '%s\n' "$LADDER_CARRIER_PYTHON"; return 0; fi
  if [ -x /usr/local/python3.12.13/bin/python ]; then
    printf '%s\n' /usr/local/python3.12.13/bin/python; return 0
  fi
  printf '%s\n' "${PY:-python3}"
}

# serve 日志路径: 显式参数 > LADDER_SERVE_LOG > /tmp/ladder-serve-log（cmd_serve 写的指针）。
# 都取不到就返回空 → 探针把 marker 记 unknown（不猜、不扫全盘找日志）。
carrier_serve_log() {
  local log=${1:-}
  if [ -z "$log" ]; then log=${LADDER_SERVE_LOG:-}; fi
  if [ -z "$log" ] && [ -s /tmp/ladder-serve-log ]; then
    log=$(cat /tmp/ladder-serve-log 2>/dev/null || true)
  fi
  printf '%s\n' "$log"
}

# serve 侧 env 快照路径: 与 serve log **同 stem**（ladder-serve-<ts>.log → ladder-serve-<ts>.env.json）。
# 同 stem 保证"一个 serve ⇔ 一份 env 快照"，不会跨 serve 串用; 显式覆盖 LADDER_SERVE_ENV。
carrier_env_path() {
  local log=${1:-} override=${LADDER_SERVE_ENV:-}
  if [ -n "$override" ]; then printf '%s\n' "$override"; return 0; fi
  [ -n "$log" ] || return 0
  case "$log" in
    *.log) printf '%s\n' "${log%.log}.env.json" ;;
    *) printf '%s\n' "$log.env.json" ;;
  esac
}

# 起服时落 serve 侧 env 快照（F1）: 只记 `VLLM_HUST_*` 相关键的**实际导出值**,
# 原子写（temp + rename）且独占语义（同 stem 已存在则不覆盖, 记错误）; 失败只告警, 不阻断起服。
carrier_write_serve_env() {   # $1=env.json 路径 $2=serve log $3=serve pid（可空）
  local out=${1:?need env file} log=${2:-} pid=${3:-}
  "$PY" - "$out" "$log" "$pid" <<'PYEOF' || echo "   （serve env 快照写入失败: $out；carriers.env 将记 unknown）" >&2
import json
import os
import sys
import time

out, log, pid = sys.argv[1], sys.argv[2], sys.argv[3]
KEYS = ("VLLM_HUST_GATE_FP32", "VLLM_HUST_GDN_MERGE_INPROJ", "VLLM_HUST_SHARED_GATE_PACK")
# 空串 = 该键没导出/被显式清空 → 记 null（**不是** off；off 只能由 "0" 表达）
keys = {k: (os.environ.get(k) or None) for k in KEYS}
doc = {
    "schema": "opt-ops-serve-env.v1",
    "serve_log": log,
    "scope": "serve-process",
    "source": "bench-ladder.sh cmd_serve（export 之后、nohup 之前的进程环境快照）",
    "written_at": time.strftime("%FT%TZ", time.gmtime()),
    "serve_pid": int(pid) if str(pid).strip().isdigit() else None,
    "keys": keys,
    "keys_set": [k for k, v in keys.items() if v is not None],
    "keys_unset": [k for k, v in keys.items() if v is None],
    "env_all_husk": {k: v for k, v in sorted(os.environ.items())
                     if k.startswith("VLLM_HUST_")},
    "note": ("VLLM_HUST_* 的取值只在此文件里可信：screen/L1/L2 的调用 shell 未必继承 "
             "cmd_serve 的 export。null = 该键未导出（不是 off）。"),
}
if os.path.exists(out):
    print(f"   serve env 快照已存在, 不覆盖: {out}", file=sys.stderr)
    raise SystemExit(1)
tmp = f"{out}.tmp.{os.getpid()}"
with open(tmp, "w", encoding="utf-8") as fh:
    json.dump(doc, fh, ensure_ascii=False, indent=1)
    fh.flush()
    os.fsync(fh.fileno())
os.replace(tmp, out)          # 原子: 读侧不会看到半个 JSON
print(f"   serve env 快照 -> {out}"
      f"（keys_set={','.join(doc['keys_set']) or '-'}；未导出={','.join(doc['keys_unset']) or '-'}）")
PYEOF
  return 0
}

# 探针解释器不可用/没输出 JSON 时的 fail-closed 记录（纯 bash, 不依赖任何 python）。
carrier_probe_fallback() {
  local py=${1:-unknown} log=${2:-} envf=${3:-} pyj logj envj
  pyj=$(printf '%s' "$py" | sed 's/\\/\\\\/g; s/"/\\"/g')
  logj=$(printf '%s' "$log" | sed 's/\\/\\\\/g; s/"/\\"/g')
  envj=$(printf '%s' "$envf" | sed 's/\\/\\\\/g; s/"/\\"/g')
  printf '%s\n' "{\"schema\": \"opt-ops-carriers.v2\", \"recorded_by\": \"bench-ladder.sh carrier_probe_fallback\", \
\"probe_python\": \"$pyj\", \"husk_package\": \"\", \"husk_package_found\": false, \
\"husk_module_spec\": \"unknown\", \"husk_entry_points\": [], \"husk_entry_point_names\": [], \
\"general_plugins_entry_points_total\": null, \
\"env\": {\"VLLM_HUST_GATE_FP32\": null, \"VLLM_HUST_GDN_MERGE_INPROJ\": null, \
\"VLLM_HUST_SHARED_GATE_PACK\": null}, \"env_scope\": \"unknown\", \"env_source\": \"unknown\", \
\"env_file\": {\"path\": \"$envj\", \"present\": false}, \"env_all_husk\": {}, \
\"serve_log\": \"$logj\", \"serve_log_present\": false, \"serve_log_markers\": {}, \
\"unreliable_absence_signals\": {}, \
\"attribution\": {\"husk\": {\"package\": \"unknown\", \"entry_points\": \"unknown\", \
\"cascade\": {\"state\": \"unknown\", \"basis\": \"探针不可用\"}, \
\"gate_fp32\": {\"state\": \"unknown\", \"basis\": \"探针不可用\"}, \
\"gdn_merge_inproj\": {\"state\": \"unknown\", \"basis\": \"探针不可用\"}}, \
\"runtime_tree\": {\"w3f2\": {\"state\": \"unknown\", \"basis\": \"探针不可用\"}, \
\"w3f3\": {\"state\": \"unknown\", \"basis\": \"探针不可用\"}}}, \
\"verdict\": {\"axis\": \"record-only\", \"cross_carrier_merge\": \"not-computed\"}, \
\"probe_errors\": [\"载体探针不可用: 解释器 $pyj 未输出 JSON（fail-closed: 记 unknown/null, 不阻断）\"]}"
}

carrier_probe_json() {   # 用法: carrier_probe_json [serve_log]; 输出单行 JSON, 永不返回非零
  local log envf py raw last
  log=$(carrier_serve_log "${1:-}")
  envf=$(carrier_env_path "$log")
  py=$(carrier_python)
  raw=$("$py" - "$py" "$log" "$envf" <<'PYEOF'
import importlib.metadata as md
import importlib.util as iu
import json
import os
import re
import sys
import time

probe_py = sys.argv[1] if len(sys.argv) > 1 else "unknown"
log = sys.argv[2] if len(sys.argv) > 2 else ""
env_file = sys.argv[3] if len(sys.argv) > 3 else ""
PACKAGE = "vllm_ascend_split_batch"
ENV_KEYS = ("VLLM_HUST_GATE_FP32", "VLLM_HUST_GDN_MERGE_INPROJ", "VLLM_HUST_SHARED_GATE_PACK")
MARKERS = (
    ("cascade_plugin_loaded", r"cascade plugin loaded"),
    ("gate_fp32_cache_populated", r"Gate fp32 cache populated"),
    ("gdn_merge_inproj_fail_closed", r"GDN merge carrier is fail-closed"),
    ("w3f2_gdn_merge_applied", r"W3F2: GDN in_proj merge applied"),
    ("w3f2_gdn_merge_skipped", r"W3F2: GDN in_proj merge skipped"),
    ("w3f3_shared_gate_pack_applied", r"W3F3: shared scalar-gate pack \([^)]*\) applied"),
    ("w3f3_shared_gate_pack_failed", r"W3F3: shared scalar-gate pack failed"),
)
NOT_A_SIGNAL = (
    'vllm 的 "Unknown vLLM environment variable detected: VLLM_HUST_*" 警告在载体存在与不'
    "存在两种状态下都会出现（见 logs/ladder-serve-1017-081645.log:18-19），不是缺席判据"
)
GDN_NOTE = (
    "仓内 gdn_merge_inproj 载体 fail-closed: 宿主 AscendGatedDeltaNetAttention.forward 没有可安全"
    "消费合并权重的公开 seam, 插件拒绝合并、模型保持 stock（见 HOST_CONTRACT.md GDN 段）。"
    "运行树 W3F2 补丁若 applied 是**另一件事**; 两者共存但不得把 W3F2 的合并归因给本插件。"
)
doc = {
    "schema": "opt-ops-carriers.v2",
    "recorded_at": time.strftime("%FT%TZ", time.gmtime()),
    "recorded_by": "bench-ladder.sh carrier_probe_json",
    "probe_python": probe_py,
    "husk_package": "",
    "husk_package_found": False,
    "husk_module_spec": "unknown",
    "husk_entry_points": [],
    "husk_entry_point_names": [],
    "general_plugins_entry_points_total": None,
    # env: 只认 serve 侧快照; 未导出/未读到一律 null（不用 "" 冒充 off）
    "env": {k: None for k in ENV_KEYS},
    "env_scope": "unknown",
    "env_source": "unknown",
    "env_file": {"path": env_file, "present": False},
    "env_all_husk": {},
    "serve_log": log,
    "serve_log_present": False,
    "serve_log_markers": {},
    "unreliable_absence_signals": {},
    # attribution: 仓内 husk 插件 与 运行树补丁 **分开**记, 不合并成单一布尔
    "attribution": {
        "husk": {
            "package": "unknown",
            "entry_points": "unknown",
            "cascade": {"state": "unknown", "basis": ""},
            "gate_fp32": {"state": "unknown", "basis": ""},
            "gdn_merge_inproj": {"state": "unknown", "basis": "", "note": GDN_NOTE},
        },
        "runtime_tree": {
            "w3f2": {"state": "unknown", "basis": ""},
            "w3f3": {"state": "unknown", "basis": ""},
        },
    },
    "verdict": {"axis": "record-only", "cross_carrier_merge": "not-computed",
                "see": "attribution（husk 插件 / 运行树补丁 各自状态）"},
    "probe_errors": [],
}
errors = doc["probe_errors"]


def probe_package():
    """服务实际 Python 里的 find_spec 结果（不可 import 不等于不可 find）。"""
    try:
        spec = iu.find_spec(PACKAGE)
    except Exception as e:  # 包存在但父级 import 抛错等: 记 unknown, 不猜缺席
        doc["attribution"]["husk"]["package"] = "unknown"
        errors.append(f"find_spec({PACKAGE}) 失败: {type(e).__name__}: {e}")
        return
    if spec is None:
        doc["husk_module_spec"] = "absent"
        doc["attribution"]["husk"]["package"] = "absent"
        return
    doc["husk_module_spec"] = getattr(spec, "origin", None) or "present"
    doc["husk_package"] = PACKAGE
    doc["husk_package_found"] = True
    doc["attribution"]["husk"]["package"] = "present"


def probe_entry_points():
    """vllm.general_plugins 的 husk 入口点（装不上 / 未注册都等价于未生效）。"""
    try:
        eps = md.entry_points()
        group = (eps.select(group="vllm.general_plugins")
                 if hasattr(eps, "select") else (eps.get("vllm.general_plugins") or []))
        items = sorted(
            ({"name": str(getattr(e, "name", "")), "value": str(getattr(e, "value", ""))}
             for e in group),
            key=lambda d: (d["name"], d["value"]))
    except Exception as e:  # 枚举本身失败: 记 unknown（不是"没有入口点"）
        doc["attribution"]["husk"]["entry_points"] = "unknown"
        errors.append(f"vllm.general_plugins 入口点枚举失败: {type(e).__name__}: {e}")
        return
    doc["general_plugins_entry_points_total"] = len(items)
    husk = [d for d in items if PACKAGE in d["value"]]
    doc["husk_entry_points"] = husk
    doc["husk_entry_point_names"] = [d["name"] for d in husk]
    doc["attribution"]["husk"]["entry_points"] = "present" if husk else "empty"


def probe_env():
    """serve 侧 env 快照（F1）: 只认 cmd_serve 落的 <log>.env.json, 不读调用 shell。"""
    doc["env_file"]["path"] = env_file
    if not env_file:
        errors.append("serve 日志路径未知 → 无法定位 serve env 快照 → env 记 unknown/null")
        return
    if not os.path.isfile(env_file):
        errors.append(f"serve env 快照不存在: {env_file}"
                      "（该 serve 若由 2026-10-17 之前的 cmd_serve 启动就不会写）→ env 记 unknown/null,"
                      "不用空串冒充 off")
        return
    try:
        with open(env_file, encoding="utf-8") as fh:
            data = json.load(fh)
        if not isinstance(data, dict):
            raise ValueError("env 快照不是 JSON 对象")
    except Exception as e:
        errors.append(f"serve env 快照不可读: {env_file}: {type(e).__name__}: {e} → env 记 unknown/null")
        return
    # 关联校验: 快照里记的 serve_log 必须指向同一份日志, 否则不采信（防止跨 serve 串用）
    snap_log = data.get("serve_log")
    if snap_log and log and os.path.abspath(snap_log) != os.path.abspath(log):
        errors.append(f"serve env 快照(${snap_log}) 与 serve 日志({log}) 不匹配 → env 记 unknown/null")
        return
    keys = data.get("keys") or {}
    doc["env"] = {k: (keys.get(k) if keys.get(k) not in ("", None) else None) for k in ENV_KEYS}
    doc["env_scope"] = str(data.get("scope") or "unknown")
    doc["env_source"] = f"serve-env-file:{env_file}"
    doc["env_file"].update({
        "present": True,
        "schema": data.get("schema"),
        "written_at": data.get("written_at"),
        "serve_pid": data.get("serve_pid"),
        "source": data.get("source"),
        "keys_set": data.get("keys_set"),
        "keys_unset": data.get("keys_unset"),
    })
    doc["env_all_husk"] = dict(data.get("env_all_husk") or {})


def probe_log():
    """本 serve 日志的 marker 计数 + 逐载体归因（F2: 不把运行树 marker 算成插件生效）。"""
    husk = doc["attribution"]["husk"]
    tree = doc["attribution"]["runtime_tree"]
    if not log:
        for slot in ("cascade", "gate_fp32", "gdn_merge_inproj"):
            husk[slot]["basis"] = "serve 日志路径未知"
        for slot in ("w3f2", "w3f3"):
            tree[slot]["basis"] = "serve 日志路径未知"
        errors.append("serve 日志路径未知（cmd_serve 写 /tmp/ladder-serve-log 或给 LADDER_SERVE_LOG）"
                      "→ marker 记 unknown")
        return
    if not os.path.isfile(log):
        for slot in ("cascade", "gate_fp32", "gdn_merge_inproj"):
            husk[slot]["basis"] = f"serve 日志不存在: {log}"
        for slot in ("w3f2", "w3f3"):
            tree[slot]["basis"] = f"serve 日志不存在: {log}"
        errors.append(f"serve 日志不存在: {log} → marker 记 unknown")
        return
    pats = {name: re.compile(pat) for name, pat in MARKERS}
    counts = {name: {"count": 0, "first_line": None} for name, _ in MARKERS}
    unknown_env = 0
    # 按 b"\n" 切行（二进制读 + 逐行 decode）: 文本模式的 universal newlines 会把日志里的孤立
    # CR（加载进度条）也当换行, 行号就会比 `grep -n` 偏大 —— 行号必须能直接对上 grep。
    with open(log, "rb") as fh:
        for lineno, raw_line in enumerate(fh, 1):
            line = raw_line.decode("utf-8", "replace")
            for name, pat in pats.items():
                if pat.search(line):
                    counts[name]["count"] += 1
                    if counts[name]["first_line"] is None:
                        counts[name]["first_line"] = lineno
            if "Unknown vLLM environment variable detected" in line and "VLLM_HUST_" in line:
                unknown_env += 1
    doc["serve_log_present"] = True
    doc["serve_log_markers"] = counts
    doc["unreliable_absence_signals"] = {
        "unknown_vllm_env_warning": {"count": unknown_env, "is_absence_signal": False,
                                     "why": NOT_A_SIGNAL}}
    have = {name: counts[name]["count"] > 0 for name in counts}
    eps = set(doc["husk_entry_point_names"] or [])
    pkg_absent = doc["husk_module_spec"] == "absent"

    def husk_slot(name, ep_name, marker, absent_state="absent"):
        slot = husk[name]
        if pkg_absent:
            slot.update({"state": absent_state, "basis": "husk 包 find_spec=absent"})
        elif have[marker]:
            slot.update({"state": "effective",
                         "basis": f"{marker}×{counts[marker]['count']}"
                                  f"（首行 {counts[marker]['first_line']}）"})
        elif ep_name in eps:
            slot.update({"state": "present-not-observed",
                         "basis": f"入口点 {ep_name} 在, 但日志无 {marker}（开关关/graph 未加载/尚未打点）"})
        else:
            slot.update({"state": absent_state,
                         "basis": f"入口点 {ep_name} 不在 vllm.general_plugins 里"})

    husk_slot("cascade", "cascade-attention", "cascade_plugin_loaded")
    husk_slot("gate_fp32", "gate-fp32", "gate_fp32_cache_populated")
    # gdn_merge_inproj 专职: fail-closed 才是"插件在但拒绝合并"; 不是 effective
    gslot = husk["gdn_merge_inproj"]
    if pkg_absent:
        gslot.update({"state": "absent", "basis": "husk 包 find_spec=absent"})
    elif have["gdn_merge_inproj_fail_closed"]:
        gslot.update({"state": "present-but-fail-closed",
                      "basis": f"gdn_merge_inproj_fail_closed×{counts['gdn_merge_inproj_fail_closed']['count']}"
                               f"（首行 {counts['gdn_merge_inproj_fail_closed']['first_line']}）: "
                               "插件拒接合并, 模型保持 stock"})
    elif "gdn-merge-inproj" in eps:
        gslot.update({"state": "present-not-observed",
                      "basis": "入口点 gdn-merge-inproj 在, 日志无 fail-closed 警告"
                               "（VLLM_HUST_GDN_MERGE_INPROJ 未开或尚未加载到该层）"})
    else:
        gslot.update({"state": "absent",
                      "basis": "入口点 gdn-merge-inproj 不在 vllm.general_plugins 里"})

    def tree_slot(name, on_markers, off_marker, on_state, off_state):
        slot = tree[name]
        hit_on = [m for m in on_markers if have.get(m)]
        if hit_on:
            m = hit_on[0]
            slot.update({"state": on_state,
                         "basis": f"{m}×{counts[m]['count']}（首行 {counts[m]['first_line']}）"})
        elif off_marker and have.get(off_marker):
            slot.update({"state": off_state,
                         "basis": f"{off_marker}×{counts[off_marker]['count']}"})
        else:
            slot.update({"state": off_state,
                         "basis": "日志里无对应补丁行（补丁未应用/未开/尚未加载）"})

    tree_slot("w3f2", ("w3f2_gdn_merge_applied",), "w3f2_gdn_merge_skipped",
              "applied", "not-applied")
    tree_slot("w3f3", ("w3f3_shared_gate_pack_applied", "w3f3_shared_gate_pack_failed"), None,
              "enabled", "not-enabled")
    if have["w3f3_shared_gate_pack_failed"]:
        tree["w3f3"].update({"state": "failed",
                             "basis": f"w3f3_shared_gate_pack_failed×{counts['w3f3_shared_gate_pack_failed']['count']}"
                                      "（fail-closed raise）"})


for fn in (probe_package, probe_entry_points, probe_env, probe_log):
    try:
        fn()
    except Exception as e:   # 兜底: 任一节崩溃只记错误, 不让调用方崩
        errors.append(f"{fn.__name__} 崩溃: {type(e).__name__}: {e}")
print(json.dumps(doc, ensure_ascii=False))
PYEOF
) || raw=""
  # 只取最后一行且必须是 JSON 对象（防止解释器在 stdout 掺入无关输出）
  last=$(printf '%s\n' "$raw" | tail -1)
  case "$last" in
    \{*) printf '%s\n' "$last" ;;
    *) carrier_probe_fallback "$py" "$log" "$envf" ;;
  esac
  return 0
}

# 人读摘要（cmd_serve 就绪后 / cmd_screen 里打）; 记录轴: 只打印, 不判定成败, 失败不报错退出。
carrier_summary() {
  local raw=${1:-}
  "$PY" - "$raw" <<'PYEOF' || echo "   载体摘要不可用（探针输出不可解析；不影响 screen/serve）" >&2
import json
import sys

try:
    c = json.loads(sys.argv[1])
    if not isinstance(c, dict):
        raise ValueError("not an object")
except Exception:
    raise SystemExit(1)
print("== 载体（记录轴: 只记录不判定成败）==")
print(f"   探针 Python: {c.get('probe_python')}")
names = c.get("husk_entry_point_names") or []
print(f"   husk 包: {c.get('husk_package') or '(未见)'}  find_spec={c.get('husk_module_spec')}"
      f"  入口点 {len(names)} 个" + ((": " + ", ".join(names)) if names else ""))
print(f"   serve env（scope={c.get('env_scope')}, source={c.get('env_source')}）")
for k, val in (c.get("env") or {}).items():
    print(f"     {k}={val if val is not None else '<unknown/未导出>'}")
print(f"   serve 日志: {c.get('serve_log') or '(未知)'}  present={c.get('serve_log_present')}")
for name, hit in (c.get("serve_log_markers") or {}).items():
    print(f"     {name}: count={hit.get('count')} first_line={hit.get('first_line')}")
un = (c.get("unreliable_absence_signals") or {}).get("unknown_vllm_env_warning") or {}
if un:
    print(f"   注意: 日志里 Unknown vLLM env 警告 {un.get('count')} 次 —— **不是**缺席判据"
          "（载体存在/不存在都会出现）")
attr = c.get("attribution") or {}
husk = attr.get("husk") or {}
tree = attr.get("runtime_tree") or {}
print("   归因（仓内 husk 插件 / 运行树补丁 分开记, 不合并）:")
for slot in ("cascade", "gate_fp32", "gdn_merge_inproj"):
    item = husk.get(slot) or {}
    print(f"     husk.{slot}={item.get('state')}  ({item.get('basis')})")
for slot in ("w3f2", "w3f3"):
    item = tree.get(slot) or {}
    print(f"     runtime_tree.{slot}={item.get('state')}  ({item.get('basis')})")
v = c.get("verdict") or {}
print(f"   总体: axis={v.get('axis')} cross_carrier_merge={v.get('cross_carrier_merge')}")
if husk.get("gate_fp32", {}).get("state") != "effective":
    print("   提示: marker 计数是「此刻」的日志快照；模型/图还没加载完时 Gate fp32 与 W3F2"
          " 尚未打点 → 稍后重跑 `bench-ladder.sh carriers` 复核"
          "（screen.json 里的 carriers 取 screen 结束时刻）")
for e in (c.get("probe_errors") or []):
    print(f"   [probe] {e}")
PYEOF
}

# /metrics 快照（窗口覆盖元数据用）: 取不到就删文件（不伪造空指标 → 覆盖率走 declared=false）。
metrics_snap() {
  local out=${1:?need path}
  if curl -sf "http://127.0.0.1:${PORT}/metrics" -o "$out" 2>/dev/null && [ -s "$out" ]; then
    return 0
  fi
  rm -f "$out" 2>/dev/null || true
  echo "   （/metrics 未取到: $out → 该窗口缺指标口径, captured_request_count 只能靠设备证据）" >&2
  return 0
}

# window_coverage_ok <opstat.json>: 新 producer 的交付门——产物必须带窗口覆盖元数据 schema。
# 老窗口（无该块）由消费者按 legacy 处理, 不走本门（本门只拦**新采集**的交付）。
window_coverage_ok() {
  local p=${1:?need opstat path}
  "$PY" - "$p" <<'PYEOF'
import json, sys
try:
    doc = json.load(open(sys.argv[1]))
except Exception as e:
    print(f"   opstat 不可读: {e}", file=sys.stderr); sys.exit(1)
cov = ((doc.get("load_caliber") or {}).get("window_coverage") or {})
if cov.get("schema") != "opt-ops-window-coverage.v1":
    print("   缺 load_caliber.window_coverage.schema（新 producer 必须写; 不得用手工块代替）",
          file=sys.stderr)
    sys.exit(1)
print(f"   coverage: {cov.get('summary')}  declared={cov.get('declared')}")
for g in cov.get("gaps") or []:
    print(f"   [gap] {g}")
sys.exit(0)
PYEOF
}

# prof 目录指针读写: PVC 优先, /tmp 回退（见 PROF_PTR_* 定义处说明）。
set_prof_ptr() {
  printf '%s\n' "$1" > "$PROF_PTR_PVC" 2>/dev/null || printf '%s\n' "$1" > "$PROF_PTR_TMP" 2>/dev/null || true
  # /tmp 副本仅作回退, 尽力而为
  printf '%s\n' "$1" > "$PROF_PTR_TMP" 2>/dev/null || true
}
get_prof_ptr() {
  if [ -s "$PROF_PTR_PVC" ]; then
    cat "$PROF_PTR_PVC"
  elif [ -s "$PROF_PTR_TMP" ]; then
    echo "bench-ladder: 从 /tmp 回退读取 prof 指针（PVC 指针缺失；容器重建前的旧指针不可续用）" >&2
    cat "$PROF_PTR_TMP"
  fi
}

serve_env_preflight() {
  # nnal/atb 在位时要求 LD_LIBRARY_PATH 已含其条目, 否则 worker dlopen libatb.so 必败。
  [ "${LADDER_SKIP_ENV_CHECK:-0}" != "1" ] || return 0
  [ -d "${LADDER_ATB_HOME:-/usr/local/Ascend/nnal/atb}" ] || return 0
  case ":${LD_LIBRARY_PATH:-}:" in
    *:"${LADDER_ATB_HOME:-/usr/local/Ascend/nnal/atb}"/*) ;;
    *)
      echo "bench-ladder: LD_LIBRARY_PATH 缺 nnal/atb 路径 → worker dlopen libatb.so 将失败 → 拒绝起服" >&2
      echo "  修复: 起服 shell 依次 source ascend-toolkit / cann ascendnpu-ir / nnal-atb 三个 set_env.sh" >&2
      echo "  （持久修复: tools/repair-ascend-login-env.sh; LADDER_SKIP_ENV_CHECK=1 自担）" >&2
      return 1 ;;
  esac
  if "$PY" -c 'import ctypes; ctypes.CDLL("libatb.so")' 2>/dev/null; then return 0; fi
  echo "bench-ladder: LD_LIBRARY_PATH 含 nnal/atb 条目但 dlopen libatb.so 失败 → 拒绝起服" >&2
  echo "  修复: 起服 shell 依次 source ascend-toolkit / cann ascendnpu-ir / nnal-atb 三个 set_env.sh" >&2
  echo "  （持久修复: tools/repair-ascend-login-env.sh; LADDER_SKIP_ENV_CHECK=1 自担）" >&2
  return 1
}

cmd_serve() {
  local profile=0 tier=plain max_iters=0 require_shapes=0
  if [ "${1:-}" = "--profile" ]; then
    profile=1; shift
    tier=${1:-${LADDER_PROFILE_TIER:-light}}
    case "$tier" in
      light)
        max_iters=${LADDER_PROFILE_MAX_ITERS:-32}
        [[ "$max_iters" =~ ^[1-9][0-9]*$ ]] || {
          echo "light profile 要求 LADDER_PROFILE_MAX_ITERS 为正整数（当前 '$max_iters'）" >&2; return 2; }
        require_shapes=0 ;;
      deep)
        max_iters=${LADDER_PROFILE_MAX_ITERS:-0}
        [[ "$max_iters" =~ ^[0-9]+$ ]] || {
          echo "deep profile 要求 LADDER_PROFILE_MAX_ITERS 为非负整数（0=完整窗口）" >&2; return 2; }
        require_shapes=1 ;;
      *) echo "未知 profile tier=$tier（light | deep）" >&2; return 2 ;;
    esac
  fi
  if health_ok; then
    local running=unknown
    [ -s "$SERVE_MODE_FILE" ] && running=$("$PY" -c 'import json,sys; print(json.load(open(sys.argv[1])).get("mode","unknown"))' "$SERVE_MODE_FILE" 2>/dev/null || echo unknown)
    if [ "$running" = "$tier" ]; then
      echo "serve 已在 ${PORT} 运行（mode=$running）, 跳过启动"
      # 复用在跑的 serve 也要打一次载体摘要（同一 serve 的载体状态应与 screen 记录一致）
      carrier_summary "$(carrier_probe_json "$(carrier_serve_log)")"
      return 0
    fi
    echo "serve 已运行但 mode=$running，请先 opt bench stop 再切换到 mode=$tier（plain/profile 不混测）" >&2
    return 3
  fi
  # 起服前置: ATB 动态库路径自检（fail-fast; 回执 records/probes/20261021T024500Z_serve_env_clobber_diagnosis）
  if serve_env_preflight; then :; else return 1; fi
  # 起服前置: 运行树接口面校验（2026-10-16 加；纯静态、不 import、不占卡、秒级）
  #   拦两类错误: ①「调用点已改、定义点未改」——2026-10-16 11:53 就是这样漏过去的
  #   （fused_moe.py 调 AscendSharedExperts.prepare_input_before_routed_experts 而定义点无此方法），
  #   代价是连续 5 次起服未就绪 ~40min; ②「载体依赖的宿主契约被改掉」——不报错、静默 fail-open。
  #   断言表: scripts/tree-seam-assertions.json; 树可用 SEAM_TREE 覆盖（默认运行树）
  #   跳过: LADDER_SKIP_SEAM_CHECK=1（需自担）
  if [ "${LADDER_SKIP_SEAM_CHECK:-0}" != "1" ]; then
    if ! "$PY" "$SCRIPTS/check-tree-seams.py" --quiet; then
      echo "bench-ladder: 接口面校验未过 → 拒绝起服（LADDER_SKIP_SEAM_CHECK=1 可跳过）" >&2
      return 1
    fi
  fi
  # 0.28 栈唯一"能跑图"的 serve 配置（2026-10-13 实测固化，见 L1 卡
  # 028/global/host028_serve_cudagraph_restore）：
  #   - Ascend 平台 cudagraph_mode==NONE ⇒ mode=NONE（vllm_ascend/platform.py:1195），等于把
  #     inductor 一起关掉 → L1 TPOT 250ms（正常应 ~14ms，17x 退化）；
  #   - 默认捕获网格 / piecewise（含 breakable）在 capture_model 报
  #     Resource_Error(EE1023) "Too many streams are created"；
  #   - 收小网格 + FULL 档可捕获：L1 TPOT 14.08ms、L2 3983 tok/s（vs 退化 250ms / 1007）。
  # 覆盖口径：LADDER_SERVE_EXTRA 设为其它值即覆盖；设为空串（`LADDER_SERVE_EXTRA=`）退回无附加参数。
  local default_extra='--compilation-config {"mode":"VLLM_COMPILE","cudagraph_mode":"FULL","cudagraph_capture_sizes":[1,2,4,8,16,32]}'
  local serve_extra=${LADDER_SERVE_EXTRA-$default_extra}
  local extra=()
  if [ -n "$serve_extra" ]; then
    if [ -n "${LADDER_SERVE_EXTRA:-}" ]; then
      echo "bench-ladder: 附加 serve 参数(显式): $serve_extra" >&2
    else
      echo "bench-ladder: 附加 serve 参数(默认固化): $serve_extra" >&2
    fi
    read -r -a extra <<< "$serve_extra"
  fi
  local prof_args=()
  # 2026-10-14 cohort 记录（模型/宿主/配置档；与负载口径正交）:
  #   本栈结论只在同一 cohort 内成立。把**实际生效**的配置写成窗口目录的 cohort.json:
  #   模型 / 宿主版本（真读 vllm+vllm_ascend）/ TP / max_model_len / APC / 图档（从 serve 参数解析）;
  #   解析时由 ladder-opstat.py 嵌进 L4 产物（cohort.id 是 producer marker 强制字段）。
  #   覆盖: LADDER_COHORT_ID=<id> 指定 id（配置细节仍按实际写）。
  cohort_write() {
    local cid=${LADDER_COHORT_ID:-028-qwen35-35b-a3b-tp2-fullgraph}
    local vers model_name cmode csizes apc
    # 注意: import vllm 会往 stdout 打 INFO 日志（platform plugins 等）→ 必须静音 + 只取最后一行，
    # 否则版本串会掺进日志文本（本函数自检时踩到过）。
    vers=$(VLLM_LOGGING_LEVEL=ERROR "$PY" -c "
import logging; logging.disable(logging.CRITICAL)
import importlib.metadata as md
def v(pkg):
    try: return md.version(pkg)
    except Exception: return 'unknown'
import vllm
# vllm_ascend 不在包级暴露 __version__（实测），版本走发行元数据
print((getattr(vllm,'__version__','unknown') or 'unknown') + '|' + v('vllm-ascend'))
" 2>/dev/null | tail -1)
    model_name=$(basename "$MODEL")
    cmode=$(printf '%s' "$serve_extra" | grep -o '"cudagraph_mode"[[:space:]]*:[[:space:]]*"[A-Z_]*"' | sed 's/.*"\([A-Z_]*\)"$/\1/')
    csizes=$(printf '%s' "$serve_extra" | grep -o '"cudagraph_capture_sizes"[[:space:]]*:[[:space:]]*\[[0-9,]*\]' | sed 's/.*\[\(.*\)\]/\1/')
    apc=off
    case "$serve_extra" in *enable-prefix-caching*) apc=on ;; esac
    [ -n "$cmode" ] || cmode="unset(serve 默认；本栈默认档可能退化为 mode=NONE, 见 L1 卡 host028_serve_cudagraph_restore)"
    "$PY" - "$prof_dir/cohort.json" "$cid" "$model_name" "$vers" "$cmode" "$csizes" "$apc" "$serve_extra" <<'PYEOF'
import importlib.util, json, sys, time
out, cid, model, vers, cmode, csizes, apc, extra = sys.argv[1:9]
core, ascend = (vers.split("|") + ["unknown", "unknown"])[:2]
# 2026-10-17: 载体状态必须随窗口记录——容器重建会抹掉 husk 插件与运行树补丁,
# 而 bench-ladder 仍会导出 VLLM_HUST_* 环境变量（等于静默 fail-open）。
# 两臂 A/B 若载体状态不同就是第二个变量; 记录轴只提示不阻断, 但必须可见。
import os
carriers = {k: os.environ.get(k, "") for k in
            ("VLLM_HUST_GATE_FP32", "VLLM_HUST_GDN_MERGE_INPROJ")}
carriers["husk_package"] = ("vllm_ascend_split_batch"
                            if importlib.util.find_spec("vllm_ascend_split_batch") else "")
doc = {
  "schema": "opt-ops-window-cohort.v1",
  "id": cid,
  "model": model,
  "host": {"vllm_core": core or "unknown", "vllm_ascend": ascend or "unknown"},
  "serve": {"entry": "vllm serve (console script)", "tp": 2, "max_model_len": 8192,
            "prefix_caching": apc, "extra_args": extra},
  "graph": {"mode": "VLLM_COMPILE", "cudagraph_mode": cmode,
            "cudagraph_capture_sizes": csizes or "unknown"},
  "precision": "bf16",
  "carriers": carriers,
  "carriers_note": "VLLM_HUST_* 是 husk 插件开关; husk_package 为空 = 载体未安装（env 静默 fail-open）。记录轴: 不作门禁、不要求重跑（单变量替换结论具普遍价值）; 只有跨窗口绝对数值比较需要同载体状态; 两臂载体状态不一致才是第二个变量。容器重启后 /root/start-sshd.sh 恢复安装与补丁链。",
  "recorded_at": time.strftime("%FT%TZ", time.gmtime()),
  "recorded_by": "bench-ladder.sh cohort_write",
  "note": "本窗口数字只在本 cohort(模型/宿主/设备/服务配置/图档) 内成立；换坐标不得外推（见 /root/opt-ops/state/cohort.json）",
}
json.dump(doc, open(out, "w"), ensure_ascii=False, indent=1)
print(f"cohort 记录 -> {out} (id={cid}, model={model}, core={core}, ascend={ascend}, graph={cmode[:40]})")
PYEOF
  }
  # 2026-10-13 (修订): 默认落 PVC /root/prof（98G）。容器约每 70 分钟重建一次，overlay 层
  # （/tmp、/srv/prof-archive）随之清空——实测 09:35 的重建直接销毁了 16G trace + 23G 解析
  # 中间产物，before 窗口零产物。放 PVC 后即使容器重建，trace 仍在，可直接重解析而不必重跑负载；
  # 磁盘压力由 cmd_profile 的 ≥20G 预检兜住。LADDER_PROF_ROOT 可覆盖（回 overlay 需自担风险）。
  # 2026-10-14 固化(协议级): D4 gate-fp32 权重外提 (husk 插件 gate_fp32_plugin)。
  # 依据: 设备 Cast[256,2048] 20480->0 / -148ms/窗; L2 out_tok_s +2.58% (3/3 对同向); L3 8/8。
  # 记为吞吐/设备预算减负, 非降 TPOT (L1 在噪声底)。显式 VLLM_HUST_GATE_FP32=0 可关
  # (插件仅在 ==1 时启用; 载体缺失/容器重建未 bootstrap 时 fail-open, 不影响起服)。
  export VLLM_HUST_GATE_FP32="${VLLM_HUST_GATE_FP32:-1}"
  # 2026-10-16 固化(协议级): F2 GDN in_proj 合并 (candidate F2-gdn-ba-qkvz-merge accepted 2026-10-16)。
  # 依据: decode MatMulV2 -30.1/步 / -10.5% 时长; kernel 对照 -15.4%/pattern; L3 8/8 identical ×2;
  #       TPOT L1 成对 -1.10% (无回归)。证据: L1 卡 f2_accepted_verdict_1016 + 候选台账。
  # 显式 VLLM_HUST_GDN_MERGE_INPROJ=0 可关（做 OFF 对照窗时必须显式置 0）。
  # 前提: 运行树补丁三件套已由 apply-runtime-patches.sh 应用（容器重建后先重放再起 serve）。
  export VLLM_HUST_GDN_MERGE_INPROJ="${VLLM_HUST_GDN_MERGE_INPROJ:-1}"
  # 2026-10-17(修订 F1): 这两个 export **必须**排在 cohort_write / env 快照之前 ——
  #   cohort.json 的 carriers 与 <serve log>.env.json 记的都是"本 serve 实际继承的 env"，
  #   放到后面会记成未导出，与 serve 实际生效值矛盾（screen 的 carriers 曾因此自证冲突）。
  local prof_root=${LADDER_PROF_ROOT:-/root/prof}
  mkdir -p "$prof_root" 2>/dev/null || prof_root=/root/prof
  local prof_dir="$prof_root/ladder-$(date +%m%d-%H%M%S)"
  if [ "$profile" = 1 ]; then
    mkdir -p "$prof_dir"
    set_prof_ptr "$prof_dir"   # 供 cmd_profile 定位 trace 目录（PVC 优先, 重建后仍可续解析）
    # light 固定迭代、禁 shape；deep 才默认收 shape。stack/memory 始终关闭。
    local rs=false
    [ "$require_shapes" = "1" ] && rs=true
    prof_args=(--profiler-config.profiler=torch
               --profiler-config.torch_profiler_dir="$prof_dir"
               --profiler-config.torch_profiler_record_shapes="$rs"
               --profiler-config.torch_profiler_with_stack="${LADDER_PROFILE_WITH_STACK:-false}"
               --profiler-config.torch_profiler_with_memory=false
               --profiler-config.ignore_frontend=true)
    [ "$max_iters" -gt 0 ] && prof_args+=(--profiler-config.max_iterations="$max_iters")
    echo "profile 模式: tier=$tier max_iterations=$max_iters record_shapes=$rs, trace: $prof_dir"
    cohort_write
    "$PY" - "$prof_dir/profile-plan.json" "$tier" "$max_iters" "$require_shapes" <<'PYEOF'
import json, sys, time
out, tier, max_iters, shapes = sys.argv[1:5]
doc = {"schema": "opt-ops-profile-plan.v1", "tier": tier,
       "max_iterations": int(max_iters), "require_shapes": shapes == "1",
       "capture_scope": (f"vllm-max-iterations-{max_iters}" if int(max_iters) else "full-window"),
       "recorded_at": time.strftime("%FT%TZ", time.gmtime())}
json.dump(doc, open(out, "w"), ensure_ascii=False, indent=1)
PYEOF
  fi
  # 2026-10-13: 日志按启动时间命名，不再无限追加同一个 ladder-serve.log
  # （旧文件已被 L2/KB 按绝对路径引用，保持原样不动；新窗口各自独立落文件）。
  local log="$LOGS/ladder-serve-$(date +%m%d-%H%M%S).log"
  echo "$log" > /tmp/ladder-serve-log   # 供 cmd_wait 的失败提示定位
  echo "启动 serve -> $log"
  nohup setsid "$VLLM" serve "$MODEL" \
    --tensor-parallel-size 2 --max-model-len 8192 --port "$PORT" \
    --no-enable-prefix-caching "${prof_args[@]}" "${extra[@]}" > "$log" 2>&1 < /dev/null &
  echo $! > "$LADDER_PIDFILE"
  # 2026-10-17(F1): 落 serve 侧 env 快照（与 serve log 同 stem 的 <log>.env.json）。
  # 这里**按 stem 硬推导**（不认 LADDER_SERVE_ENV 覆盖），保证"一个 serve ⇔ 一份快照"的关联；
  # 该覆盖项只给探针/排障用。这是 carriers.env 的**唯一**来源：screen/L1/L2 的调用 shell
  # 未必继承这里的 export，读调用 shell 会写出"键为空 + marker effective"的自证冲突。
  # 失败只告警, 不阻断起服。
  local serve_env="${log%.log}.env.json"
  carrier_write_serve_env "$serve_env" "$log" "$(serve_pid)"
  cmd_wait 1800 || return $?
  # 2026-10-17: 就绪后打载体 marker 摘要（人工核验: 包/入口点/env 来源/日志逐载体归因）。
  # 只记录不阻断 —— 载体状态是记录轴（/root/opt-ops/AGENTS.md），起服失败另有其它门。
  carrier_summary "$(carrier_probe_json "$log")"
  "$PY" - "$SERVE_MODE_FILE" "$tier" <<'PYEOF'
import json, sys
json.dump({"mode": sys.argv[2]}, open(sys.argv[1], "w"))
PYEOF
}

cmd_stop() {
  local pid; pid=$(serve_pid)
  if [ -n "$pid" ]; then
    kill -- -"$pid" 2>/dev/null || kill "$pid" 2>/dev/null || true
    rm -f "$LADDER_PIDFILE"
  fi
  rm -f "$SERVE_MODE_FILE"
  # 兜底: 清理残存 vllm serve 进程
  pkill -f "vllm serve $MODEL" 2>/dev/null || true
  echo "serve 已停止"
}

cmd_wait() {
  local timeout=${1:-1800} i=0
  until health_ok; do
    sleep 5; i=$((i + 5))
    if [ "$i" -ge "$timeout" ]; then echo "等待 /health 超时(${timeout}s), 查 $(cat /tmp/ladder-serve-log 2>/dev/null || echo "$LOGS/ladder-serve.log")"; return 1; fi
  done
  echo "serve 就绪 (端口 $PORT)"
}

cmd_l1() {
  local ts; ts=$(date +%m%d-%H%M%S)
  # run 内联调用时不另写日志（统一进 <run>/steps.log）；单独调用仍落 logs/。
  if [ "${LADDER_RUN_ACTIVE:-0}" = "1" ]; then
    "$PY" "$SCRIPTS/ladder-l1.py" --port "$PORT" --iters "${1:-10}" --out "$RECORDS/${ts}-l1.json"
  else
    "$PY" "$SCRIPTS/ladder-l1.py" --port "$PORT" --iters "${1:-10}" \
      --out "$RECORDS/${ts}-l1.json" | tee "$LOGS/ladder-l1-${ts}.log"
  fi
}

cmd_l2() {
  local ts; ts=$(date +%m%d-%H%M%S)
  # 2026-10-14 负载形态（两种形态属不同统计范围, 只比方向不比数值）:
  #   synthetic（默认）合成固定协议: random 2048/256 定长 @并发 32 —— 算子 A/B 复现口径
  #   mmlu-pro         MMLU-Pro 真实评测流量形状: 5-shot CoT prompt（展平）+ thinking 长输出
  #                    @num_concurrent=5；需要 LADDER_L2_DATASET=<prompt jsonl>
  #                    （由 scripts/export-mmlupro-prompts.py 从真实评测样本导出）
  # 产物一律带 load_caliber/cohort 标记（产物自证；否则口径门禁看不见它, 见 ladder-stamp.py）
  local load=${LADDER_L2_LOAD:-synthetic}
  local n conc outfile
  local -a ds=() extra=()
  case "$load" in
    synthetic)
      n=${1:-128}; conc=${2:-32}
      outfile="${ts}-l2.json"
      ds=(--dataset-name random --num-prompts "$n"
          --random-input-len 2048 --random-output-len 256 --random-range-ratio 0) ;;
    mmlu-pro)
      : "${LADDER_L2_DATASET:?mmlu-pro 载荷需要 LADDER_L2_DATASET=<prompt jsonl>（export-mmlupro-prompts.py 产出）}"
      [ -f "$LADDER_L2_DATASET" ] || { echo "LADDER_L2_DATASET 不存在: $LADDER_L2_DATASET"; return 1; }
      local total; total=$(grep -c . "$LADDER_L2_DATASET")
      n=${1:-$total}; conc=${2:-${LADDER_EVAL_CONC:-5}}
      outfile="${ts}-l2-mmlupro.json"
      ds=(--dataset-name custom --dataset-path "$LADDER_L2_DATASET"
          --num-prompts "$n" --custom-output-len "${LADDER_L2_MAXTOK:-2048}")
      extra=(--metric-percentiles 50,90,99) ;;
    *) echo "未知 LADDER_L2_LOAD=$load（synthetic | mmlu-pro）"; return 1 ;;
  esac
  local args=(bench serve
    --backend openai-chat --endpoint /v1/chat/completions
    --host 127.0.0.1 --port "$PORT"
    "${ds[@]}"
    --max-concurrency "$conc" --temperature 0 --seed 42
    --percentile-metrics ttft,tpot,itl,e2el "${extra[@]}"
    --save-result --result-dir "$RECORDS" --result-file "$outfile")
  local rc=0
  if [ "${LADDER_RUN_ACTIVE:-0}" = "1" ]; then
    "$VLLM" "${args[@]}" || rc=$?
  else
    "$VLLM" "${args[@]}" 2>&1 | tee "$LOGS/ladder-l2-${ts}.log" || rc=$?
  fi
  [ "$rc" = "0" ] || { echo "L2 采集失败(rc=$rc), 不写口径标记"; return "$rc"; }
  local calib_id=${LADDER_LOAD_CALIBER:-} calib_detail=${LADDER_LOAD_CALIBER_DETAIL:-}
  local maxout=256
  if [ "$load" = "mmlu-pro" ]; then maxout=${LADDER_L2_MAXTOK:-2048}; fi
  if [ -z "$calib_id" ]; then
    if [ "$load" = "mmlu-pro" ]; then calib_id=mmlu-pro-eval-traffic
    else calib_id=synthetic-fixed-protocol; fi
  fi
  if [ -z "$calib_detail" ]; then
    if [ "$load" = "mmlu-pro" ]; then
      calib_detail="MMLU-Pro 真实评测流量形状 L2（数据集实例=lm-eval mmlu_pro_smoke 样本展平 prompt: $LADDER_L2_DATASET）: 5-shot CoT 形状 / 单轮 / 无状态 / APC off / num_concurrent=$conc / max_tokens=$maxout / greedy seed=42"
    else
      calib_detail="合成固定协议 L2: random 2048/256 定长 @并发 $conc（算子 A/B 复现口径）"
    fi
  fi
  "$PY" "$SCRIPTS/ladder-stamp.py" --file "$RECORDS/$outfile" \
    --load-caliber "$calib_id" --load-caliber-detail "$calib_detail" \
    --cohort-id "${LADDER_COHORT_ID:-028-qwen35-35b-a3b-tp2-fullgraph}" \
    --run-note "L2 load=$load n=$n conc=$conc max_out=$maxout dataset=${LADDER_L2_DATASET:-random}"
}

cmd_l3() {
  local ts; ts=$(date +%m%d-%H%M%S)
  if [ "${1:-}" = "gdn16" ]; then
    shift
    local gdn_dir; gdn_dir=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
    "$PY" "$gdn_dir/bench_gdn_l3.py" "$@" --port "$PORT" \
      --out "$RECORDS/${ts}-gdn16-l3.json"
    return $?
  fi
  "$PY" "$SCRIPTS/ladder-l3.py" capture --port "$PORT" --out "$RECORDS/${ts}-l3.json"
}

cmd_compare() { "$PY" "$SCRIPTS/ladder-l3.py" compare "$1" "$2"; }

cmd_smoke() {
  local ts; ts=$(date +%m%d-%H%M%S)
  if [ "${LADDER_RUN_ACTIVE:-0}" = "1" ]; then
    "$SCRIPTS/run-smoke-eval.sh" "${1:-4}" "ladder-smoke-${ts}" "/dev/null"
  else
    "$SCRIPTS/run-smoke-eval.sh" "${1:-4}" "ladder-smoke-${ts}" "$LOGS/ladder-smoke-${ts}.log"
  fi
}

cmd_l0() {
  local ts; ts=$(date +%m%d-%H%M%S)
  local op=${1:-all}
  [ $# -gt 0 ] && shift
  if [ "$op" = "GdnQkvNorm" ]; then
    local norm_dir; norm_dir=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
    "$PY" "$norm_dir/bench_gdn_qkv_norm.py" "$@" \
      --out "$RECORDS/${ts}-l0-gdn-qkv-norm.json"
    return $?
  fi
  if [ "$op" = "GdnQkvUnpack" ]; then
    # rearrange_mixed_qkv（split+reshape×3+cat 4 launch）配对 A/B：单算子单入口,
    # 其余算子仍走 bench-l0.py; 产物同落 records/ladder 的带时间戳新文件
    local self_dir; self_dir=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
    "$PY" "$self_dir/bench_gdn_qkv_unpack.py" "$@" \
      --out "$RECORDS/${ts}-l0-gdn-qkv-unpack.json"
    return $?
  fi
  # 余下参数原样透传给 bench-l0.py（--sig/--decode-tokens/--iters...）;
  # 自带 --out 放最后, 保证落 L2 的带时间戳新文件（只追加, 不覆盖既有归档）
  "$PY" "$SCRIPTS/bench-l0.py" --op "$op" "$@" \
    --out "$RECORDS/${ts}-l0.json"
}

cmd_profile() {
  local ts; ts=$(date +%m%d-%H%M%S)
  local requested_tier="" load_n=${LADDER_PROFILE_REQ:-4}
  while [ $# -gt 0 ]; do
    case "$1" in
      --tier) shift; requested_tier=${1:?--tier 缺 light|deep} ;;
      --tier=*) requested_tier=${1#--tier=} ;;
      *) load_n=$1 ;;
    esac
    shift
  done
  [ -n "$requested_tier" ] || { echo "profile 必须显式给 --tier light|deep" >&2; return 2; }
  case "$requested_tier" in light|deep) ;; *) echo "未知 tier=$requested_tier" >&2; return 2 ;; esac
  # trace 目录: 优先 PVC 指针（容器重建后仍在）, /tmp 回退, 再兜底找最新 ladder-* 目录
  local prof_dir
  prof_dir=$(get_prof_ptr || true)
  [ -z "$prof_dir" ] && prof_dir=$(ls -td /root/prof/ladder-* 2>/dev/null | head -1)
  if [ -z "$prof_dir" ] || [ ! -d "$prof_dir" ]; then
    echo "未找到 trace 目录（先 bench-ladder.sh stop 再 serve --profile <tier>）"; return 1
  fi
  local plan="$prof_dir/profile-plan.json" tier max_iters require_shapes capture_scope
  [ -f "$plan" ] || { echo "缺 profile-plan.json：拒绝无档位窗口 $prof_dir" >&2; return 3; }
  IFS=$'\t' read -r tier max_iters require_shapes capture_scope < <("$PY" - "$plan" <<'PYEOF'
import json, sys
p = json.load(open(sys.argv[1]))
print(p.get("tier", ""), p.get("max_iterations", ""),
      1 if p.get("require_shapes") else 0, p.get("capture_scope", ""), sep="\t")
PYEOF
)
  [ "$tier" = "$requested_tier" ] || {
    echo "请求 tier=$requested_tier，但当前 serve/trace 目录是 tier=$tier；先 stop 后按同档重启" >&2; return 3; }
  # 2026-10-13: 磁盘预检（必须）。/root 是 98G PVC，窗口 trace 约 12~17G；
  # 曾经在盘满时开窗口 → trace 写一半，torch 侧 TLV 损坏（"collected data has been lost"），
  # DbParser/op_statistic.csv 全挂且**不报错**，白跑 15 分钟还污染 before/after 对照。
  local min_free_gb=${LADDER_MIN_FREE_GB:-20}
  local free_gb; free_gb=$(df -BG --output=avail "$prof_dir" | tail -1 | tr -dc '0-9')
  if [ "${free_gb:-0}" -lt "$min_free_gb" ]; then
    echo "!! trace 所在盘仅剩 ${free_gb}G < ${min_free_gb}G（$prof_dir）：盘满会把 trace 写坏（静默，torch 侧全挂）。"
    echo "   先跑一次瘦身清中间物（单窗可回收 ~11G/rank）："
    echo "     bench-ladder.sh slim /root/prof/<窗目录>          # 抢救 csv + 删中间物"
    echo "     bench-ladder.sh slim /root/prof/<窗目录> 000000 --dryrun  # 只报账（LADDER_SLIM_DRYRUN=1）"
    echo "   注意: trace 现默认落 PVC /root/prof（持久）; 瘦身会把 csv/db 抢救到 records/ 后删中间物。"
    echo "   （或 LADDER_MIN_FREE_GB=<n> 覆盖门槛）"
    return 1
  fi
  # 2026-10-14 负载口径（必须显式标注，跨口径不得外推算子占比）:
  #   synthetic（默认）   合成固定协议 = 重复定长 unit prompt × load_n，chat，max_tokens=128，定长 decode；
  #                       适合算子 A/B 复现（单变量、可重复），**不代表 MMLU-Pro 真实流量结构**。
  #   mmlu-pro            数据集派生（评测流量）= run-prof-load.sh（默认 lm-eval mmlu_pro_smoke 前 N 条、
  #                       5-shot CoT、num_concurrent=5、max_gen_toks=2048）→ 长 prefill + 思考型长 decode。
  #   external            任意自定负载 = 由 LADDER_PROFILE_LOAD_CMD 提供的命令（vllm bench serve /
  #                       run-prof-load.sh <N> <task> <conc> / 自写客户端…）；用于新增"负载特征"（scenario）:
  #                       长上下文 prefill / 多轮 chat / 别的数据集（mmlu_pro_local / HLE / SWE…）。
  #                       必须同时给 LADDER_LOAD_CALIBER=<已登记的口径 id>（先登记到 state/load-caliber.json）。
  # 两口径的算子占比/热点排名不可互相外推；窗口产物带 load_caliber 标记（ladder-opstat.py），
  # 入库卡必须声明口径（opt note/ledger）。覆盖: LADDER_PROFILE_LOAD=mmlu-pro LADDER_PROFILE_MMLU_N=<n>。
  local load_mode=${LADDER_PROFILE_LOAD:-synthetic}
  local calib_id calib_detail
  case "$load_mode" in
    synthetic)
      calib_id="synthetic-fixed-protocol"
      calib_detail="反复定长 unit prompt x${load_n} (chat, max_tokens=128, temperature=0, seed=42) — 算子 A/B 复现口径, 非 MMLU-Pro 流量"
      ;;
    mmlu-pro|mmlu_pro|mmlupro)
      load_n=${LADDER_PROFILE_MMLU_N:-$load_n}
      calib_id="mmlu-pro-eval-traffic"
      calib_detail="lm-eval ${LADDER_EVAL_TASK:-mmlu_pro_smoke}(第一 ${load_n} 条, 5-shot CoT, num_concurrent=${LADDER_EVAL_CONC:-5}, max_gen_toks=2048, chat) — 数据集派生评测流量口径"
      ;;
    external)
      # 外部负载命令（v4：口径 id 可空，仅作导航）
      [ -n "${LADDER_PROFILE_LOAD_CMD:-}" ] || { echo "external 模式必须给 LADDER_PROFILE_LOAD_CMD='<负载命令>'"; return 1; }
      calib_id=${LADDER_LOAD_CALIBER:-}
      calib_detail=${LADDER_LOAD_CALIBER_DETAIL:-"外部负载命令: $LADDER_PROFILE_LOAD_CMD"}
      [ -n "$calib_id" ] && echo "   口径 id（导航，不设门）: $calib_id"
      ;;
    *)
      echo "未知 LADDER_PROFILE_LOAD=$load_mode（synthetic | mmlu-pro | external）"; return 1 ;;
  esac
  calib_detail="$calib_detail; capture_scope=$capture_scope; trace_tier=$tier; max_iterations=$max_iters"
  export LADDER_LOAD_CALIBER="$calib_id" LADDER_LOAD_CALIBER_DETAIL="$calib_detail"
  echo "== L4 $tier 窗口: start_profile -> 负载[${calib_id}] -> stop_profile (capture=$capture_scope, trace: $prof_dir, /root 余 ${free_gb}G) =="
  local cohort_file="$prof_dir/cohort.json"
  [ -f "$cohort_file" ] || cohort_file=""
  echo "   负载: $calib_detail"
  [ -n "$cohort_file" ] && echo "   cohort: $(basename "$prof_dir")/cohort.json（模型/宿主/配置档随窗口记录）" \
    || echo "   cohort: 本窗口无 cohort.json（老窗口或 serve 未以 --profile 起）→ 产物 cohort.id=unmarked"
  # 2026-10-17 新增（窗口覆盖元数据契约 opt-ops-window-coverage.v1）:
  #   同一 caliber/cohort 下，两个窗口的算子**总量**不可直接相减——采集窗（capture_scope）覆盖的
  #   prefill 体积可以差 2 倍（已发生的实例: GDN conc=8/16 的 ChunkGatedDeltaRule 2.04x 全由体积解释）。
  #   所以新窗口必须在窗口目录留下三份结构化证据，由 ladder-opstat.py 交叉核对成 window_coverage:
  #     ① /metrics 前后快照（真实 prompt/请求数增量, 窗口 ⊆ 墙钟窗）
  #     ② load-exec.json（负载执行体的逐请求完成记录, 不是"计划 N"）
  #     ③ load-manifest.json（负载特征参数 + capture 档位 + 执行体 + 指标增量 + 模型层结构）
  #   缺任何一份 → 产物 declared=false + gaps（不阻断分析, 阻断跨窗口总量比较）。
  metrics_snap "$prof_dir/metrics-before.txt"
  local exec_json="$prof_dir/load-exec.json"
  curl -sf -X POST "http://127.0.0.1:${PORT}/start_profile" || { echo "start_profile 失败（serve 需以 bench-ladder.sh serve --profile 启动）"; return 1; }
  if [ "$load_mode" = "synthetic" ]; then
    "$PY" - "$PORT" "$load_n" "$exec_json" <<'EOF'
import json, os, sys, time, urllib.request
port, n, out = int(sys.argv[1]), int(sys.argv[2]), sys.argv[3]
unit = ("Benchmark load for operator profiling across batched decode steps. " * 30)
done, failed = [], []
for i in range(n):
    body = json.dumps({"model": os.environ.get("LADDER_MODEL") or "/models/Qwen3.5-35B-A3B",
                       "messages": [{"role": "user", "content": unit + f" Request index {i}. Summarize in one sentence."}],
                       "max_tokens": 128, "temperature": 0, "seed": 42}).encode()
    req = urllib.request.Request(f"http://127.0.0.1:{port}/v1/chat/completions", data=body,
                                 headers={"Content-Type": "application/json"})
    try:
        urllib.request.urlopen(req, timeout=600).read()
        done.append(i)
    except Exception as e:                                   # 逐请求记录, 不因单条失败静默丢窗口
        failed.append({"i": i, "error": str(e)[:200]})
doc = {"schema": "opt-ops-load-exec.v1", "kind": "synthetic-client",
       "requested": n, "completed": len(done), "failed": len(failed),
       "failures": failed[:5], "recorded_at": time.strftime("%FT%TZ", time.gmtime()),
       "note": "客户端逐请求完成记录（不是计划 N）"}
json.dump(doc, open(out, "w"), ensure_ascii=False, indent=1)
print(f"profile load: {len(done)}/{n} requests done")
EOF
  elif [ "$load_mode" = "external" ]; then
    # 外部负载命令（scenario 自定：长上下文 prefill / 多轮 chat / 别的数据集 …）
    echo "   负载命令: $LADDER_PROFILE_LOAD_CMD"
    if ! bash -lc "$LADDER_PROFILE_LOAD_CMD"; then
      echo "LADDER_PROFILE_LOAD_CMD 失败，停 profile 避免窗口作废"
      curl -sf -X POST "http://127.0.0.1:${PORT}/stop_profile" || true; return 1
    fi
    # external 没有统一的结构化完成记录 → 明确记 null, 由 window_coverage 标 declared=false
    "$PY" - "$exec_json" "$LADDER_PROFILE_LOAD_CMD" <<'EOF'
import json, sys, time
out, cmd = sys.argv[1], sys.argv[2]
json.dump({"schema": "opt-ops-load-exec.v1", "kind": "external", "cmd": cmd,
           "requested": None, "completed": None,
           "recorded_at": time.strftime("%FT%TZ", time.gmtime()),
           "note": "external 负载无结构化完成记录 → captured_request_count 只靠设备证据与 /metrics 交叉核对"},
          open(out, "w"), ensure_ascii=False, indent=1)
EOF
  else
    # 数据集派生负载: 复用固化脚本（可传 task/并发/输出目录；默认 mmlu_pro_smoke / 5 / eval-prof-N）
    # 本次窗口的评测输出落到窗口目录（$prof_dir/eval），供执行体记录与事后复核（不污染共享 eval 目录）
    local eval_out="$prof_dir/eval"
    mkdir -p "$eval_out"
    if ! LADDER_EVAL_OUT="$eval_out" \
         "$SCRIPTS/run-prof-load.sh" "$load_n" "${LADDER_EVAL_TASK:-mmlu_pro_smoke}" "${LADDER_EVAL_CONC:-5}"; then
      echo "run-prof-load.sh 失败，停 profile 避免窗口作废"
      curl -sf -X POST "http://127.0.0.1:${PORT}/stop_profile" || true; return 1
    fi
    # 执行体记录: 完成数 = 本次窗口 samples_*.jsonl 的样本行数（lm-eval 逐样本落盘, 非计划 N）
    "$PY" - "$eval_out" "$exec_json" "$load_n" "${LADDER_EVAL_TASK:-mmlu_pro_smoke}" <<'PYEOF'
import glob, json, os, sys, time
eval_out, out, n, task = sys.argv[1], sys.argv[2], int(sys.argv[3]), sys.argv[4]
samples = sorted(glob.glob(os.path.join(eval_out, "**", "samples_*.jsonl"), recursive=True),
                 key=os.path.getmtime, reverse=True)
results = sorted(glob.glob(os.path.join(eval_out, "**", "results_*.json"), recursive=True),
                 key=os.path.getmtime, reverse=True)
completed = None
if samples:
    with open(samples[0]) as f:
        completed = sum(1 for line in f if line.strip())
doc = {"schema": "opt-ops-load-exec.v1", "kind": "lm-eval", "task": task,
       "requested": n, "completed": completed,
       "samples_path": samples[0] if samples else "",
       "results_path": results[0] if results else "",
       "eval_out": eval_out,
       "recorded_at": time.strftime("%FT%TZ", time.gmtime()),
       "note": "lm-eval 逐样本完成记录（completed = samples 行数, 不是计划 N）"}
json.dump(doc, open(out, "w"), ensure_ascii=False, indent=1)
print(f"  执行体记录: completed={completed}/{n} -> {out}")
PYEOF
  fi
  curl -sf -X POST "http://127.0.0.1:${PORT}/stop_profile"
  metrics_snap "$prof_dir/metrics-after.txt"
  # 窗口 manifest: 负载特征参数 + capture 档位 + 执行体 + 指标增量 + 模型层结构
  # （ladder-opstat.py 只读这份 + kernel_details.csv, 不读原始 trace）
  local man_task="" man_ds="" man_conc="" man_maxout=""
  case "$load_mode" in
    synthetic) man_task="synthetic-unit-prompt"; man_conc=$load_n; man_maxout=128
               man_ds="内置单元提示 ×${load_n}（无数据集文件; 负载特征=定长 prompt/128 输出）" ;;
    external)  man_task="external"; man_ds="${LADDER_PROFILE_LOAD_CMD:-}" ;;
    *)         man_task="${LADDER_EVAL_TASK:-mmlu_pro_smoke}"; man_conc="${LADDER_EVAL_CONC:-5}"
               man_maxout="${LADDER_PROFILE_MAXTOK:-2048}"
               man_ds="lm-eval ${man_task} 前 ${load_n} 条（5-shot CoT 展平 prompt）" ;;
  esac
  MAN_LOAD_MODE="$load_mode" MAN_CALIBER="$calib_id" MAN_CALIBER_DETAIL="$calib_detail" \
  MAN_MODEL="$MODEL" MAN_N="$load_n" MAN_TASK="$man_task" MAN_DATASET="$man_ds" \
  MAN_CONC="$man_conc" MAN_MAXOUT="$man_maxout" \
    "$PY" - "$prof_dir" <<'PYEOF'
import glob, json, os, sys, time
prof = sys.argv[1]
env = os.environ
def jload(p, d=None):
    try:
        return json.load(open(p))
    except Exception:
        return d

def metric_total(path, name, must_contain=""):
    """prometheus 文本 → 指定指标所有标签集的求和（缺文件/缺指标返回 None）。"""
    if not path or not os.path.exists(path):
        return None
    tot, seen = 0.0, False
    with open(path) as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            if not (line.startswith(name + "{") or line.startswith(name + " ")):
                continue
            if must_contain and must_contain not in line:
                continue
            try:
                tot += float(line.rsplit(None, 1)[1])
                seen = True
            except (IndexError, ValueError):
                continue
    return tot if seen else None

mode = env.get("MAN_LOAD_MODE", "")
model = env.get("MAN_MODEL", "")
plan = jload(os.path.join(prof, "profile-plan.json"), {}) or {}
exe = jload(os.path.join(prof, "load-exec.json"), {}) or {}
before, after = os.path.join(prof, "metrics-before.txt"), os.path.join(prof, "metrics-after.txt")
deltas = {}
for key, name, contains in (
        ("prompt_tokens", "vllm:prompt_tokens_total", ""),
        ("prompt_tokens_local_compute", "vllm:prompt_tokens_by_source_total", 'source="local_compute"'),
        ("generation_tokens", "vllm:generation_tokens_total", ""),
        ("request_success", "vllm:request_success_total", "")):
    a, b = metric_total(before, name, contains), metric_total(after, name, contains)
    if a is not None and b is not None:
        deltas[key] = int(round(b - a)) if (b - a) >= 0 else None
cfg = os.path.join(model, "config.json") if model else ""
layers = {}
try:
    tc = (json.load(open(cfg)).get("text_config") or json.load(open(cfg)))
    counts = {}
    for t in tc.get("layer_types") or []:
        counts[str(t)] = counts.get(str(t), 0) + 1
    layers = {"num_hidden_layers": tc.get("num_hidden_layers"), "layer_type_counts": counts,
              "linear_attention_layers": counts.get("linear_attention"),
              "full_attention_layers": counts.get("full_attention")}
except Exception:
    layers = {}
doc = {
    "schema": "opt-ops-load-manifest.v1",
    "window": os.path.basename(prof.rstrip("/")),
    "recorded_at": time.strftime("%FT%TZ", time.gmtime()),
    "recorded_by": "bench-ladder.sh cmd_profile",
    "load": {"mode": mode,
             "task": env.get("MAN_TASK", ""),
             "num_prompts": int(env.get("MAN_N") or 0) or None,
             "num_concurrent": int(env.get("MAN_CONC") or 0) or None,
             "dataset": env.get("MAN_DATASET", ""),
             "max_output_tokens": int(env.get("MAN_MAXOUT") or 0) or None,
             "temperature": 0, "seed": 42,
             "executor": {k: exe.get(k) for k in
                          ("kind", "cmd", "task", "samples_path", "results_path", "eval_out")
                          if exe.get(k) is not None}},
    "capture": {"tier": plan.get("tier", ""),
                "max_iterations": plan.get("max_iterations"),
                "capture_scope": plan.get("capture_scope", ""),
                "require_shapes": plan.get("require_shapes")},
    "execution": {"requested": exe.get("requested"), "completed": exe.get("completed"),
                  "completed_source": ("lm-eval samples 行数" if exe.get("kind") == "lm-eval"
                                       else "客户端逐请求完成记录" if exe.get("kind") == "synthetic-client"
                                       else "")},
    "metrics": {"before": before if os.path.exists(before) else "",
                "after": after if os.path.exists(after) else "",
                "delta": deltas,
                "note": "prometheus 快照增量＝墙钟窗口内的真实负载量；trace 窗 ⊆ 墙钟窗"},
    "model": {"path": model, "config": cfg if cfg and os.path.exists(cfg) else "",
              "layer_counts": layers},
    "load_caliber": {"id": env.get("MAN_CALIBER", ""), "detail": env.get("MAN_CALIBER_DETAIL", "")},
    "cohort_file": os.path.join(prof, "cohort.json") if os.path.exists(os.path.join(prof, "cohort.json")) else "",
    "note": ("窗口覆盖元数据（opt-ops-window-coverage.v1）的结构化来源: 负载特征参数/capture 档位/"
             "执行体完成记录/指标增量/模型层结构。缺任一项 → ladder-opstat 产物 declared=false。"),
}
out = os.path.join(prof, "load-manifest.json")
json.dump(doc, open(out, "w"), ensure_ascii=False, indent=1)
print(f"  窗口 manifest -> {out}（metrics Δ={deltas}）")
PYEOF
  echo; echo "等待 trace 落盘（ascend_pt 目录出现在 $prof_dir）..."
  local i=0
  while ! ls -d "$prof_dir"/*_ascend_pt >/dev/null 2>&1; do
    sleep 5; i=$((i + 5)); [ "$i" -ge 300 ] && break
  done
  # 2026-10-13: 目录刚出现 ≠ 落盘完成。CANN 侧原始数据是在 stop_profile 之后继续搬的，
  # 立即解析会「秒返回 PARSE_DONE 但什么都不产出」→ 必须等体积稳定（连续 3 次采样不变）。
  local prev=-1 cur stable=0
  while [ "$stable" -lt 3 ]; do
    cur=$(du -sk "$prof_dir" 2>/dev/null | cut -f1)
    [ "$cur" = "$prev" ] && stable=$((stable + 1)) || stable=0
    prev=$cur; sleep 20
  done
  echo "trace 落盘稳定（$(du -sh "$prof_dir" | cut -f1)），开始解析"
  local asc_dir; asc_dir=$(ls -td "$prof_dir"/*_ascend_pt 2>/dev/null | head -1)
  [ -z "$asc_dir" ] && { echo "未找到 ascend_pt 目录"; return 1; }
  # 2026-10-13: 空壳必须早退。曾在 serve 运行期间更换 --profiler-config.torch_profiler_dir 的目标
  # （我为了避开 PVC 盘满，把该路径的符号链接重指到空目录）→ CANN 侧写不进去，ascend_pt 只有空壳，
  # 而离线解析会「秒过 PARSE_DONE 但什么都不产出」，白等 12 分钟才报「缺 op_statistic.csv」。
  # 换 profiler 目录必须重启 serve，不能在飞行中换。
  local asc_n; asc_n=$(find "$asc_dir" -mindepth 1 2>/dev/null | wc -l)
  if [ "$asc_n" -lt 2 ]; then
    echo "!! ascend_pt 是空壳（仅 $asc_n 个条目）：profiler 没写进去，解析必然是空跑。"
    echo "   最常见原因: serve 运行中更换/重建了 --profiler-config.torch_profiler_dir 的目标；"
    echo "   换目录必须重启 serve（opt bench stop && opt bench serve --profile）。"
    echo "   目录: $asc_dir"
    return 1
  fi
  echo "ascend_pt: $asc_dir （离线解析走 LADDER_PARSE_PYTHON（默认 serving python）, ascend_pt 父目录口径, ulimit -v ${LADDER_PARSE_ULIMIT_MB:-45000}M）"
  # 改动 6｜解析与 serve 错峰（开关默认关, 不改变既有默认行为）。
  # 开启后 stop_profile 与 analyse 之间停一次 serve, 释放 ~20G 常驻内存, 把 cgroup 64G 留给解析。
  # 归因判据与结论见交付报告; 默认关是因为「停 serve」会打断主会话的在线观测, 需人工裁决。
  if [ "${LADDER_STOP_SERVE_FOR_PARSE:-1}" = "1" ]; then
    echo "LADDER_STOP_SERVE_FOR_PARSE=1 → 解析前停 serve（opt bench stop）释放常驻内存"
    ( cd /root 2>/dev/null; opt bench stop ) \
      || echo "  警告: opt bench stop 返回非零, 继续解析" >&2
  fi
  local parse_log="$LOGS/ladder-parse-${ts}.log"
  (
    # 2026-10-13: 旧值 20G 在「无图重放」(cudagraph_mode=NONE) 的窗口上会让 DbParser 等全部 parser 任务失败
    # （trace 体量翻 2~4 倍）→ 提到 45G; cgroup 64G, 必须与 serve 错峰跑。
    ulimit -v $(( ${LADDER_PARSE_ULIMIT_MB:-45000} * 1000 ))
    # 路径走 argv（不插值进 Python 源码, 避免引号/空格注入）
    "$PARSE_PYTHON" - "$asc_dir" > "$parse_log" 2>&1 <<'PYEOF'
import sys
from torch_npu.profiler.profiler import analyse
analyse(sys.argv[1], max_process_number=1)
print('PARSE_DONE')
PYEOF
  )
  grep -q PARSE_DONE "$parse_log" || { echo "离线解析未完成, 查 $parse_log"; return 1; }
  # 2026-10-15 根因修复（重复采集）: "秒过 PARSE_DONE"或只产出部分 csv ≠ 交付完成。
  # 旧逻辑只认 op_statistic.csv 就自动瘦身 → 删掉中间物后 kernel_details.csv 仍缺 →
  # ladder-opstat.py 退出 1 → 无法离线恢复 → 被迫重采同一窗（W4C /root/prof/ladder-1015-065857 即此）。
  # 现改为: 真正下游必需产物(op_statistic.csv / kernel_details.csv / pt db=shape 来源)齐备 **且**
  # 下游产物(all-kernel 口径的 L4 opstat, 含已登记 load_caliber)成功落盘 **且** shape 抽取成功后,
  # 才允许自动瘦身; 否则保留全部原始输入、给出离线恢复路径, 不再以"重采"为默认。
  local outd="$asc_dir/ASCEND_PROFILER_OUTPUT"
  local win; win=$(basename "$prof_dir")
  if ! rank_inputs_ok "$asc_dir" "$require_shapes"; then
    echo "!! profile 交付门: 窗口 '$win' 未成功交付（检查 rank: $(basename "$asc_dir")）→ 不瘦身, 原始输入全部保留。"
    echo "   离线恢复（无需重采）: opt bench recover-profile $asc_dir"
    echo "   若重解析仍缺 kernel_details.csv: 多为解析未跑完/内存不足 → 调大 LADDER_PARSE_ULIMIT_MB,"
    echo "   或 LADDER_STOP_SERVE_FOR_PARSE=1 与 serve 错峰后再试。parse_log=$parse_log"
    return 1
  fi
  local out_json; out_json=$(reserve_path "$RECORDS/${ts}-opstat.json") \
    || { echo "profile: 无法预留 L4 产物路径"; return 1; }
  # 2026-10-14 口径修复: 产物改由 ladder-opstat.py 生成 —— 目标算子表 + comm 段 + step_budget 段，
  # device 分母 = all-kernel（compute+comm）。旧内联块只读 op_statistic.csv（compute-only），
  # 通信（hcom_*，Accelerator Core=COMMUNICATION，本栈占 ~22% device）系统性不在视线里。
  # 2026-10-14 扩展: 同窗一并产出 op_profile（aic/aiv/mac/mte + cube_util）与 comm_detail
  # （analysis.db CommAnalyzer*: wait/sync/transit、消息大小、带宽），供"算子在算还是搬"归因。
  if ! "$PY" "$SCRIPTS/ladder-opstat.py" --asc-dir "$asc_dir" --out "$out_json" \
       --load-caliber "${LADDER_LOAD_CALIBER:-unmarked}" \
       --load-caliber-detail "${LADDER_LOAD_CALIBER_DETAIL:-未声明负载口径的窗口（消费端不得据此外推）}" \
       --window-manifest "$prof_dir/load-manifest.json" \
       --model-config "$MODEL/config.json" \
       $( [ -n "$cohort_file" ] && printf -- '--cohort-file %s' "$cohort_file" ); then
    echo "!! profile 交付门: L4 产物生成失败（$out_json）→ 不瘦身, 原始输入保留。"
    echo "   排查后重跑产物生成, 或: opt bench recover-profile $asc_dir"
    return 1
  fi
  # 把「采了负载的哪一段」写进产物；固定迭代轻窗不得伪装成完整流量窗口。
  "$PY" - "$out_json" "$plan" <<'PYEOF'
import json, os, sys
out, plan = sys.argv[1:3]
doc, scope = json.load(open(out)), json.load(open(plan))
doc["trace_tier"] = scope["tier"]
doc["capture_scope"] = {k: scope[k] for k in ("max_iterations", "capture_scope", "require_shapes")}
tmp = out + ".tmp"
json.dump(doc, open(tmp, "w"), ensure_ascii=False, indent=1)
os.replace(tmp, out)
PYEOF
  # 交付判据: L4 产物必须真正落盘、范围有效（load_caliber 已登记且非 unmarked）且与本 rank 关联。
  if ! window_opstat_ok "$asc_dir" "$out_json" >/dev/null; then
    echo "!! profile 交付门: L4 产物未落盘/范围无效/未关联（$out_json）→ 不瘦身, 原始输入保留。"
    echo "   load_caliber 须为**已登记**口径（≠ unmarked）; 排查后重跑产物生成, 或: opt bench recover-profile $asc_dir"
    return 1
  fi
  # 2026-10-17 窗口覆盖元数据交付门: 新采集必须带 window_coverage.schema（旧窗口兼容读, 不走本门）。
  if ! window_coverage_ok "$out_json"; then
    echo "!! profile 交付门: L4 产物缺窗口覆盖元数据（$out_json）→ 不瘦身, 原始输入保留。"
    echo "   该窗不得用于跨窗口总量比较; 补窗口 manifest 后: opt bench recover-profile $asc_dir"
    return 1
  fi
  if [ "$require_shapes" = "1" ]; then
    # deep 才把 per-shape 作为交付硬门。
    local ptdb out_shapes
    ptdb=$(ls "$outd"/ascend_pytorch_profiler_*.db 2>/dev/null | head -1)
    out_shapes=$(reserve_path "$RECORDS/${ts}-l0-shapes.json") || out_shapes=""
    if [ -z "$ptdb" ] || [ -z "$out_shapes" ] || \
       ! "$PY" "$SCRIPTS/extract-l0-shapes-db.py" "$ptdb" "$out_shapes"; then
      echo "!! deep profile 交付门: shape 抽取失败（ptdb=${ptdb:-缺}）→ 不自动瘦身, 原始输入保留。"
      return 1
    fi
    echo "  （刷新 L0 shape 库: opt truth ingest --shapes $out_shapes --version 028）"
  else
    echo "  light profile: 只交付算子/all-kernel/通信；shape 沿用同 cohort 已冻结基线"
  fi
  # 交付后立即瘦身；light 不保留大 DB，deep 保留 shape 来源。
  LADDER_REQUIRE_SHAPES="$require_shapes" \
    LADDER_SLIM_RESCUE_DB="$require_shapes" cmd_slim "$asc_dir" "$ts"
}

# ---- 交付门判据（自动瘦身与显式 `opt bench slim` 共用；**逐 rank** 判定, 不提供 force 逃生口）----
# L4 窗口常含多个 ascend_pt「rank」; 本协议 L4 产物可能只来自 rank0（cmd_profile 只解析最新一个
# ascend_pt）, 因此**不强制所有 rank 都交付**。改为**逐 rank** 判"该 rank 能否安全瘦身":
# 不可瘦身的 rank 一律**保留（不删任何文件）**, 其余 rank 照常瘦身（不会因某个 rank 未交付
# 就整体拒收, 也不会因某个 rank 已交付就去删别的未交付 rank）。
#
# 一个 rank 可瘦身 ⟺ 同时满足:
#   (输入) ASCEND_PROFILER_OUTPUT/{op_statistic.csv, kernel_details.csv, ascend_pytorch_profiler_*.db}
#            —— kernel_details=all-kernel 分母/通信输入; pt db=per-shape 签名来源
#   (产物) 存在**与该 rank 关联**的有效 opstat: load_caliber.id **已登记**(≠ unmarked),
#           且其 asc_dir == 该 rank 的 asc_dir（或由 --opstat 显式指定）。
# rank_inputs_ok <asc_dir>: 0=输入齐备; 1=缺输入（打印缺失项到 stderr）。
rank_inputs_ok() {
  local a=${1:?need asc_dir} require_shapes=${2:-${LADDER_REQUIRE_SHAPES:-1}} outd
  outd="$a/ASCEND_PROFILER_OUTPUT"
  local -a miss=()
  [ -f "$outd/op_statistic.csv" ] || miss+=("op_statistic.csv(目标算子表输入)")
  [ -f "$outd/kernel_details.csv" ] || miss+=("kernel_details.csv(all-kernel 分母/通信输入)")
  if [ "$require_shapes" = "1" ]; then
    ls "$outd"/ascend_pytorch_profiler_*.db >/dev/null 2>&1 || miss+=("ascend_pytorch_profiler_*.db(shape 来源)")
  fi
  [ "${#miss[@]}" -eq 0 ] && return 0
  echo "   输入不完整 $(basename "$a"): 缺 ${miss[*]}" >&2
  return 1
}

# window_opstat_ok <asc_dir> [opstat_hint]: 找与 asc_dir 关联、范围有效的 opstat; 打印命中路径, 0/1。
# 范围有效 = load_caliber.id 已登记且 ≠ unmarked（"仅非空字符串/占位 unmarked"不算有效）。
# hint = 显式 --opstat 路径（仍校验范围有效; 若其自带 asc_dir 须与目标一致）。
window_opstat_ok() {
  local asc_dir=${1:?need asc_dir} hint=${2:-}
  "$PY" - "$asc_dir" "$RECORDS" "$hint" <<'PYEOF'
import glob, json, os, sys
asc_dir, records, hint = sys.argv[1], sys.argv[2], sys.argv[3]

def caliber_ok(doc):
    """v4：范围有效 = 自带形状事实(window_coverage/prefill_token_volume) + 权重标签(cohort.model)。"""
    doc = doc or {}
    wc = (doc.get("load_caliber") or {}).get("window_coverage") or doc.get("shape_facts") or {}
    wl = doc.get("weight_label") or (doc.get("cohort") or {})
    return bool(wc.get("schema") or wc.get("prefill_token_volume")) and bool(wl.get("model"))

def ok(path):
    try:
        doc = json.load(open(path))
    except Exception:
        return False
    if not isinstance(doc, dict) or not caliber_ok(doc):
        return False
    ad = doc.get("asc_dir") or ""
    if ad:
        return os.path.realpath(ad) == os.path.realpath(asc_dir)
    return path == hint        # 无 asc_dir: 仅显式 hint 才接受匹配

def valid(p):
    return bool(p) and os.path.exists(p) and ok(p)

if hint:                       # 显式指定: 只认这一份, 无效即失败（不静默回落到扫描）
    if valid(hint):
        print(hint)
        sys.exit(0)
    sys.exit(1)
cands = sorted(glob.glob(os.path.join(records, "*-opstat*.json")),
               key=os.path.getmtime, reverse=True)
for p in cands:
    if valid(p):
        print(p)
        sys.exit(0)
sys.exit(1)
PYEOF
}

# 预留一个**不存在的**输出路径（append-only: 绝不覆盖既有证据）。打印可用路径; 1=无法预留。
reserve_path() {
  local p=${1:?need path} base ext i=1
  [ ! -e "$p" ] && { printf '%s\n' "$p"; return 0; }
  case "$p" in
    */*.*) base=${p%.*}; ext=.${p##*.} ;;
    *)     base=$p; ext= ;;
  esac
  while [ -e "${base}.${i}${ext}" ]; do i=$((i + 1)); [ "$i" -gt 10000 ] && return 1; done
  printf '%s\n' "${base}.${i}${ext}"
}

# 2026-10-13: 解析后瘦身。单窗口 13G（2 rank），真有用的只有 ASCEND_PROFILER_OUTPUT 里的
# csv（~50M/rank）与 torch 侧 pt db（per-shape 证据源; D4 的 Cast[256,2048] 计数唯一来源）。其余全是中间物：
#   FRAMEWORK/torch.op_range|op_mark   2.5G/rank  torch 逐 op 记录（大窗口解析必挂, 删）
#   PROF_*/host/                       2.4G/rank  host 侧记录
#   PROF_*/msprof_*.db                 1.5G/rank  CANN 解析中间库（csv 已产出）
#   PROF_*/device_*/data/              0.9G/rank  device 原始切片
# csv 与 pt db 一律抢救到 PVC（records/raw/<窗>/）。容器约每 70 分钟重建, overlay 层清空;
# trace 目录虽已默认落 PVC, 但中间物无保留价值, 删掉可把单窗从 16G 压到 ~1.5G（见交付报告）。
# 开关: LADDER_SLIM=0 关; LADDER_SLIM_KEEP_RAW=1 只抢救不删; LADDER_SLIM_DRYRUN=1 只报账;
#       LADDER_SLIM_RESCUE_DB=0 关闭 pt db 抢救（默认 1, 即抢救; 旧默认 0 会让 db 随重建失效）;
#       LADDER_SLIM_OPSTAT=<opstat 路径> 显式指定与该窗关联的 L4 产物（交付门用）
# 交付门（默认保护, 无 force 逃生口）: **逐 rank**; 未交付 rank（输入不全 or 无关联有效 opstat）
# 一律保留不删, 只瘦身已交付的 rank; 若无任何 rank 可安全瘦身 → 不删并返回非零。
cmd_slim() {
  local target=${1:?need ascend_pt 或包含 ascend_pt 的窗目录}; shift || true
  local ts="" opstat_hint="${LADDER_SLIM_OPSTAT:-}"
  while [ $# -gt 0 ]; do
    case "$1" in
      --dryrun) LADDER_SLIM_DRYRUN=1 ;;
      --opstat) shift || true; opstat_hint=${1:?--opstat 缺路径} ;;
      --opstat=*) opstat_hint=${1#--opstat=} ;;
      *) [ -z "$ts" ] && ts=$1 ;;
    esac
    shift || true
  done
  [ -n "$ts" ] || ts=$(date +%m%d-%H%M%S)
  if [ "${LADDER_SLIM:-1}" = "0" ]; then echo "LADDER_SLIM=0, 跳过瘦身"; return 0; fi
  local win asc_list base
  base=$(basename "$target")
  if [ -d "$target" ] && [ "${base%_ascend_pt}" != "$base" ]; then
    win=$(basename "$(dirname "$target")"); asc_list=$target
  elif [ -d "$target" ] && ls -d "$target"/*_ascend_pt >/dev/null 2>&1; then
    win=$(basename "$target"); asc_list=$(ls -d "$target"/*_ascend_pt)
  else
    echo "slim: 找不到 ascend_pt 目录: $target"; return 1
  fi
  local window_root plan require_shapes=${LADDER_REQUIRE_SHAPES:-}
  window_root=$(dirname "$(echo "$asc_list" | head -1)")
  plan="$window_root/profile-plan.json"
  if [ -z "$require_shapes" ] && [ -f "$plan" ]; then
    require_shapes=$("$PY" -c 'import json,sys; print(1 if json.load(open(sys.argv[1])).get("require_shapes") else 0)' "$plan")
  fi
  [ -n "$require_shapes" ] || require_shapes=1  # 老窗口保持原交付门
  local rescue_db=${LADDER_SLIM_RESCUE_DB:-$require_shapes}
  echo "== 瘦身窗口 '$win'（$(( $(echo "$asc_list" | wc -l) )) 个 rank; shapes=$require_shapes; dryrun=${LADDER_SLIM_DRYRUN:-0}）=="
  local raw_root="$RECORDS/raw/$win"
  local -a m_rows=() kept_rows=()
  local slimmed=0 a
  for a in $asc_list; do
    local rank; rank=$(basename "$a")
    # 交付门（逐 rank）: 输入齐备 + 与该 rank 关联的有效 opstat(范围已登记、非 unmarked)。
    # 不满足者**保留（不删任何文件）**, 不影响其它 rank。
    if ! rank_inputs_ok "$a" "$require_shapes"; then
      kept_rows+=("$rank|输入不完整"); continue
    fi
    if ! window_opstat_ok "$a" "$opstat_hint" >/dev/null; then
      kept_rows+=("$rank|无关联有效 opstat(load_caliber 未登记/unmarked/缺产物)"); continue
    fi
    slimmed=$((slimmed + 1))
    local bef aft
    bef=$(du -sk "$a" 2>/dev/null | cut -f1)
    # --- 抢救: csv 必保; pt db 默认也保（LADDER_SLIM_RESCUE_DB=0 才关） ---
    local out_dst="$raw_root/$rank/ASCEND_PROFILER_OUTPUT"
    if [ "${LADDER_SLIM_DRYRUN:-0}" != "1" ]; then
      mkdir -p "$out_dst"
      cp -n "$a"/ASCEND_PROFILER_OUTPUT/*.csv "$out_dst"/ 2>/dev/null || true
      # 2026-10-14 扩展: 通信视图（communication.json 42MB / communication_matrix.json）一起 rescue；
      # 只挑 communication*.json，绝不动 trace_view.json（1.2G+，本来就该删）。
      cp -n "$a"/ASCEND_PROFILER_OUTPUT/communication*.json "$out_dst"/ 2>/dev/null || true
      if [ "$rescue_db" = "1" ]; then
        cp -n "$a"/ASCEND_PROFILER_OUTPUT/*.db "$out_dst"/ 2>/dev/null || true
      fi
    fi
    # --- 删除中间物；light 的 opstat 已交付后不再保留大 DB ---
    local p kept_db
    local -a dead=(
      "$a/FRAMEWORK/torch.op_range" "$a/FRAMEWORK/torch.op_mark"
      "$a/ASCEND_PROFILER_OUTPUT/trace_view.json"
    )
    if [ "$rescue_db" != "1" ]; then
      for p in "$a"/ASCEND_PROFILER_OUTPUT/*.db; do [ -e "$p" ] && dead+=("$p"); done
    fi
    for p in "$a"/PROF_*/host "$a"/PROF_*/device_*/data "$a"/PROF_*/msprof_*.db; do
      [ -e "$p" ] && dead+=("$p")
    done
    local dead_kb=0 d
    for d in "${dead[@]}"; do
      [ -e "$d" ] || continue
      dead_kb=$(( dead_kb + $(du -sk "$d" 2>/dev/null | cut -f1) ))
      [ "${LADDER_SLIM_DRYRUN:-0}" = "1" ] || [ "${LADDER_SLIM_KEEP_RAW:-0}" = "1" ] || rm -rf "$d"
    done
    kept_db=$(ls "$a"/ASCEND_PROFILER_OUTPUT/*.db 2>/dev/null | wc -l)
    aft=$(du -sk "$a" 2>/dev/null | cut -f1)
    awk -v r="$rank" -v b="$bef" -v a="$aft" -v d="$dead_kb" -v n="$kept_db" \
      'BEGIN{printf "  %-46s %7.1fG -> %6.1fG (可回收 %.1fG, pt db 保留 %s 个)\n", r, b/1048576, a/1048576, d/1048576, n}'
    m_rows+=("$rank|$bef|$aft|$dead_kb|$kept_db")
  done
  if [ "${#kept_rows[@]}" -gt 0 ]; then
    echo "  -- 以下 rank 未交付 → 保留（未删任何文件）:"
    local kr
    for kr in "${kept_rows[@]}"; do echo "     ${kr%%|*} （${kr#*|}）"; done
  fi
  if [ "$slimmed" = 0 ]; then
    echo "!! slim 交付门: 窗口 '$win' 无任何可安全瘦身的 rank → 未删任何文件, 原始输入全部保留。"
    echo "   离线恢复（无需重采）: opt bench recover-profile $target"
    echo "   （本门无 force 逃生口; 确需人工清理请自行处置, 不要在窗口不完整时删中间物）"
    return 1
  fi
  if [ "${LADDER_SLIM_DRYRUN:-0}" = "1" ]; then echo "  (dryrun: 未删任何东西)"; return 0; fi
  # 瘦身台账（**独占文件名**, append-only: 绝不覆盖既有证据）
  local manifest; manifest=$(reserve_path "$RECORDS/${ts}-slim.json") \
    || { echo "slim: 无法预留台账路径"; return 1; }
  local -a margs=("$manifest" "$win" "$raw_root" "$ts")
  margs+=("${m_rows[@]}")
  [ "${#kept_rows[@]}" -gt 0 ] && margs+=(--kept "${kept_rows[@]}")
  "$PY" - "${margs[@]}" <<'EOF'
import json, os, sys, time
man, win, raw, ts = sys.argv[1], sys.argv[2], sys.argv[3], sys.argv[4]
rest = sys.argv[5:]
kept = []
if "--kept" in rest:
    i = rest.index("--kept"); slim, kept = rest[:i], rest[i + 1:]
else:
    slim = rest
rows = []
for r in slim:
    rank, bef, aft, dead, db = r.split("|")
    rows.append({"rank": rank, "before_kb": int(bef), "after_kb": int(aft),
                 "reclaimed_kb": int(dead), "kept_pt_db": int(db)})
kept_ranks = [{"rank": r.split("|", 1)[0], "reason": r.split("|", 1)[1]} for r in kept]
doc = {"window": win, "ts": ts,
       "saved_at": time.strftime("%FT%TZ", time.gmtime()),
       "rescued_csv_to": raw, "ranks": rows, "kept_ranks": kept_ranks,
       "rescue_db_default": os.environ.get("LADDER_SLIM_RESCUE_DB", "1"),
       "note": "逐 rank 交付门: 仅对已交付(输入完整 + 关联有效 opstat)的 rank 瘦身; "
               "csv 与 pt db(默认)已 rescue 到 PVC; 未交付 rank 保留未动"}
json.dump(doc, open(man, "w"), indent=1)
print(f"  台账 -> {man}")
EOF
  echo "  抢救产物 -> $raw_root/ （PVC 持久层）"
  [ "${#kept_rows[@]}" -gt 0 ] && echo "  注意: 有 rank 被保留（未瘦身）, 见上; 台账只记已瘦身的 rank"
  return 0
}

# 2026-10-15 新增: 只为**已存在**窗口生成/补齐 L4 产物（无 NPU / 无负载 / **无覆盖** / 不瘦身）。
# 严格 no-overwrite 安全: **不原地重解析**（避免覆盖既有 csv/db）—— 只从**已有完整产物**
# (op_statistic.csv + kernel_details.csv + pt db)重新生成一份新 opstat（独占文件名）。
# 若输入不完整, 明确失败并说明"不完整原始需另行受控解析", 不在本命令内隐式 analyse、不写任何文件。
# 口径必须能从**已登记窗口源**核实: 关联 opstat 的 load_caliber、或显式 --caliber/LADDER_LOAD_CALIBER;
# 缺失/未登记/unmarked 一律失败, 让用户显式提供（不得以 unmarked 成功）。
# 用法: bench-ladder.sh recover-profile <窗目录|*_ascend_pt> [--opstat <path>] [--caliber <id>]
cmd_recover_profile() {
  local target=${1:?need 窗目录或 ascend_pt 路径}; shift || true
  local ts="" opstat_hint="${LADDER_RECOVER_OPSTAT:-}" calib="${LADDER_LOAD_CALIBER:-}"
  while [ $# -gt 0 ]; do
    case "$1" in
      --opstat) shift || true; opstat_hint=${1:?--opstat 缺路径} ;;
      --opstat=*) opstat_hint=${1#--opstat=} ;;
      --caliber) shift || true; calib=${1:?--caliber 缺 id} ;;
      --caliber=*) calib=${1#--caliber=} ;;
      # 可选 ts（第 2 位置参数）: 只作文件名提示; 与既有文件冲突即靠 reserve_path 换名, 绝不覆盖
      *) [ -z "$ts" ] && ts=$1 ;;
    esac
    shift || true
  done
  local asc_dir=""
  if [ -d "$target" ] && [ "${target%_ascend_pt}" != "$target" ]; then
    asc_dir=$target
  elif [ -d "$target" ]; then
    asc_dir=$(ls -td "$target"/*_ascend_pt 2>/dev/null | head -1)
  fi
  if [ -z "$asc_dir" ] || [ ! -d "$asc_dir" ]; then
    echo "recover-profile: 找不到 ascend_pt 目录: $target"; return 1
  fi
  echo "== recover-profile（离线 / 无 NPU / 无覆盖）: $asc_dir =="
  local require_shapes=${LADDER_REQUIRE_SHAPES:-1} plan="$(dirname "$asc_dir")/profile-plan.json"
  if [ -f "$plan" ]; then
    require_shapes=$("$PY" -c 'import json,sys; print(1 if json.load(open(sys.argv[1])).get("require_shapes") else 0)' "$plan")
  fi
  # light 只需算子/all-kernel 输入；deep/老窗口仍要求 shape DB。
  if ! rank_inputs_ok "$asc_dir" "$require_shapes"; then
    echo "!! recover-profile: 既有产物不完整（见上）→ 本命令**不写任何文件**。"
    echo "   不完整原始需**另行受控解析**（本命令不原地 analyse, 以免覆盖既有 csv/db）:"
    echo "     1) 原始输入拷到新唯一目录（不用硬链接: parser 可能原地改写）:"
    echo "        cp -a \"$asc_dir\" \"$(dirname "$asc_dir")/recover-copy-$(date +%m%d-%H%M%S)\""
    echo "        （先确认目标盘空余 ≥ 单窗体积; 不足则先清理**已交付**窗再复制, 半盘复制会把新目录也写坏）"
    echo "     2) 对新目录离线 analyse 后, 再对**新目录**跑 recover-profile 生成 L4 产物。"
    echo "   现场保留、未改动: $asc_dir"
    return 1
  fi
  # 口径核实: 优先沿用**关联 opstat** 的 load_caliber; 否则用显式 --caliber/LADDER_LOAD_CALIBER。
  local assoc="" caliber="$calib" detail=""
  if assoc=$(window_opstat_ok "$asc_dir" "$opstat_hint"); then
    caliber=$("$PY" - "$assoc" <<'PYEOF'
import json, sys
print(((json.load(open(sys.argv[1])).get("load_caliber") or {}).get("id")) or "")
PYEOF
)
    detail="recover-profile: 沿用关联 opstat 的负载口径（$(basename "$assoc")）"
  fi
  [ -n "$detail" ] || detail="recover-profile: 口径 $caliber（导航，不设门）"
  [ -n "$ts" ] || ts=$(date +%m%d-%H%M%S)
  # 2026-10-17 窗口覆盖元数据: 与 live profile 走**同一 producer contract**（同一字段名/同一来源）。
  # 老窗口通常没有 load-manifest.json → 不补造、不猜: 传空 manifest, 由 ladder-opstat 产
  # load_caliber.window_coverage.declared=false + gaps（模型层结构仍用本机 config 交叉核对迭代数）。
  local manifest_path; manifest_path="$(dirname "$asc_dir")/load-manifest.json"
  [ -f "$manifest_path" ] || manifest_path=""
  local model_cfg="$MODEL/config.json"
  [ -f "$model_cfg" ] || model_cfg=""
  # 独占/预留文件名: 绝不覆盖既有证据（用户传入相同 ts 也不会覆盖）
  local out_json; out_json=$(reserve_path "$RECORDS/${ts}-recover-opstat.json") \
    || { echo "recover-profile: 无法预留产物路径"; return 1; }
  if ! "$PY" "$SCRIPTS/ladder-opstat.py" --asc-dir "$asc_dir" --out "$out_json" \
       --load-caliber "$caliber" --load-caliber-detail "$detail" \
       --window-manifest "$manifest_path" --model-config "$model_cfg"; then
    echo "!! recover-profile: L4 产物生成失败（$out_json）→ 不报恢复完成, 现场保留。"
    return 1
  fi
  if ! window_opstat_ok "$asc_dir" "$out_json" >/dev/null; then
    echo "!! recover-profile: 产物未落盘/范围无效/未关联（$out_json）→ 不报恢复完成。"
    echo "   范围有效 = 自带形状事实(load_caliber.window_coverage/prefill_token_volume) + 权重标签(cohort.model)；口径 id 仅导航。"
    return 1
  fi
  # 覆盖元数据: 同一 producer contract; 老窗口补不出执行体证据 → 明确 declared=false（不猜）。
  window_coverage_ok "$out_json" || {
    echo "!! recover-profile: 产物缺窗口覆盖元数据（$out_json）→ 不报恢复完成。"
    return 1
  }
  echo "   L4 产物 -> $out_json"
  echo "   恢复完成; 如需回收磁盘: opt bench slim $target"
}

cmd_screen() {
  local tag=${1:?need unique tag} reps=${LADDER_SCREEN_REPEATS:-3}
  [[ "$reps" =~ ^[0-9]+$ ]] && [ "$reps" -ge 3 ] || {
    echo "LADDER_SCREEN_REPEATS 必须 >=3（单变量取中位）" >&2; return 2; }
  local caliber=${LADDER_LOAD_CALIBER:-}   # v4: 仅导航，不再作 screen 门票
  health_ok || { echo "plain serve 未就绪；先 opt bench serve" >&2; return 1; }
  local mode=unknown
  [ -s "$SERVE_MODE_FILE" ] && mode=$("$PY" -c 'import json,sys; print(json.load(open(sys.argv[1])).get("mode","unknown"))' "$SERVE_MODE_FILE" 2>/dev/null || echo unknown)
  [ "$mode" = plain ] || {
    echo "screen 只接受 plain serve（当前 mode=$mode）；先 stop 再 opt bench serve" >&2; return 3; }
  local dir="$RECORDS/$tag"
  [ ! -e "$dir" ] || { echo "screen tag 已存在，证据只追加，换一个 tag: $dir" >&2; return 3; }
  mkdir -p "$dir"
  local marker; marker=$(mktemp /tmp/ladder-screen.XXXXXX); touch "$marker"
  local i
  export LADDER_RUN_ACTIVE=1
  for ((i=1; i<=reps; i++)); do
    echo "--- screen repeat $i/$reps: L1 ---" | tee -a "$dir/steps.log"
    cmd_l1 | tee -a "$dir/steps.log" || { unset LADDER_RUN_ACTIVE; rm -f "$marker"; return 1; }
    echo "--- screen repeat $i/$reps: L2 ---" | tee -a "$dir/steps.log"
    cmd_l2 | tee -a "$dir/steps.log" || { unset LADDER_RUN_ACTIVE; rm -f "$marker"; return 1; }
  done
  echo "--- screen: L3 ---" | tee -a "$dir/steps.log"
  cmd_l3 | tee -a "$dir/steps.log" || { unset LADDER_RUN_ACTIVE; rm -f "$marker"; return 1; }
  unset LADDER_RUN_ACTIVE
  # 2026-10-17: 载体状态随 screen 记录（screen.json 的 carriers）—— plain screen 原来**完全没有**
  # 载体记录, 于是只能拿日志里的 Unknown env 警告反推, 而那两者并不等价（见文件头 carriers 段）。
  # 探针失败也不阻断 screen: 失败只写 fail-closed 记录（unknown/false）。
  local carriers_json
  carriers_json=$(carrier_probe_json)
  echo "--- screen: carriers ---" | tee -a "$dir/steps.log"
  carrier_summary "$carriers_json" 2>&1 | tee -a "$dir/steps.log"
  "$PY" - "$RECORDS" "$dir/screen.json" "$marker" "$tag" "$reps" "$caliber" \
    "${LADDER_COHORT_ID:-028-qwen35-35b-a3b-tp2-fullgraph}" "$carriers_json" <<'PYEOF'
import glob, json, os, sys, time
records, out, marker, tag, reps, caliber, cohort, carriers_raw = sys.argv[1:9]
cut = os.stat(marker).st_mtime_ns
def fresh(pattern):
    return [p for p in sorted(glob.glob(os.path.join(records, pattern)))
            if os.stat(p).st_mtime_ns >= cut]
l1, l2, l3 = fresh("*-l1.json"), fresh("*-l2*.json"), fresh("*-l3.json")
reps = int(reps)
if len(l1) != reps or len(l2) != reps or len(l3) != 1:
    raise SystemExit(f"screen artifacts incomplete: l1={len(l1)} l2={len(l2)} l3={len(l3)}")
# carriers: 探针输出的 dict 原样嵌入; 不可解析就写 fail-closed 记录（不加门、不让 screen 崩）
try:
    carriers = json.loads(carriers_raw)
    if not isinstance(carriers, dict):
        raise ValueError("carriers 不是 JSON 对象")
except Exception as e:
    carriers = {
        "schema": "opt-ops-carriers.v2",
        "recorded_by": "bench-ladder.sh cmd_screen fallback",
        "probe_python": "unknown", "husk_package": "", "husk_package_found": False,
        "husk_module_spec": "unknown", "husk_entry_points": [], "husk_entry_point_names": [],
        "env": {"VLLM_HUST_GATE_FP32": None, "VLLM_HUST_GDN_MERGE_INPROJ": None,
                "VLLM_HUST_SHARED_GATE_PACK": None},
        "env_scope": "unknown", "env_source": "unknown",
        "env_file": {"path": "", "present": False}, "env_all_husk": {},
        "serve_log": "", "serve_log_present": False,
        "serve_log_markers": {}, "unreliable_absence_signals": {},
        "attribution": {"husk": {"package": "unknown", "entry_points": "unknown",
                                 "cascade": {"state": "unknown", "basis": "探针不可解析"},
                                 "gate_fp32": {"state": "unknown", "basis": "探针不可解析"},
                                 "gdn_merge_inproj": {"state": "unknown",
                                                      "basis": "探针不可解析"}},
                        "runtime_tree": {"w3f2": {"state": "unknown", "basis": "探针不可解析"},
                                         "w3f3": {"state": "unknown", "basis": "探针不可解析"}}},
        "verdict": {"axis": "record-only", "cross_carrier_merge": "not-computed"},
        "probe_errors": [f"carriers 不可解析: {type(e).__name__}: {e}"],
    }
doc = {"schema": "opt-ops-screen.v1", "tag": tag, "mode": "plain",
       "repetitions": reps, "load_caliber": {"id": caliber}, "cohort": {"id": cohort},
       "carriers": carriers,
       "carriers_note": ("载体状态是**记录轴不是门禁**（/root/opt-ops/AGENTS.md 实验纪律）: "
                         "只记录、不阻断、不要求重跑; 只有跨窗口绝对数值比较需要同载体状态, "
                         "两臂载体状态不一致才是第二个变量。可靠判据 = find_spec + "
                         "vllm.general_plugins 入口点 + serve 日志 marker; env 只认 serve 侧快照"
                         "（<serve log>.env.json, env_scope/env_source 见 carriers）; 归因分开记"
                         "（carriers.attribution.husk 插件 vs runtime_tree 补丁）, 不合并; "
                         "日志里的 'Unknown vLLM environment variable' 警告不是缺席判据。"),
       "artifacts": {"l1": l1, "l2": l2, "l3": l3},
       "created_at": time.strftime("%FT%TZ", time.gmtime()),
       "next": ("候选晋级 `ready|validating` 后才可开 light L4"
                "（前置: L3 compare PASS + L1/L2 无明显回归；台阶见 opt-ops/README 采集门）")}
json.dump(doc, open(out, "x"), ensure_ascii=False, indent=1)
print(f"screen report -> {out}")
PYEOF
  local rc=$?; rm -f "$marker"; return "$rc"
}

cmd_run() {
  local tag=${1:?need tag}; shift
  local steps=${1:-l1,l2,l3,smoke}
  local dir="$RECORDS/$tag"
  mkdir -p "$dir"
  echo "== ladder run '$tag' steps=$steps -> $dir =="
  health_ok || { echo "serve 未就绪, 先 bench-ladder.sh serve"; return 1; }
  # 2026-10-14 降噪: run 期间不再为每步单独写 logs/ladder-<step>-<ts>.log（统一进 steps.log）,
  # 也不再往 run 目录**拷贝** l1/l2/l3 json（改为 run.json 索引指向 records/ladder/<ts>-*.json 权威产物）。
  export LADDER_RUN_ACTIVE=1
  local IFS=','
  for step in $steps; do
    echo "--- step: $step ---"
    case "$step" in
      l1) cmd_l1 | tee -a "$dir/steps.log" ;;
      l2) cmd_l2 | tee -a "$dir/steps.log" ;;
      l3) cmd_l3 | tee -a "$dir/steps.log" ;;
      smoke) cmd_smoke 4 | tee -a "$dir/steps.log" ;;
      *) echo "未知 step: $step"; unset LADDER_RUN_ACTIVE; return 1 ;;
    esac
  done
  unset LADDER_RUN_ACTIVE
  # 收拢: 只搬 smoke 产物目录（harness 自己写的目录, 单份）, 其余写索引
  local smoke_out; smoke_out=$(ls -td "$RECORDS"/eval-ladder-smoke-* 2>/dev/null | head -1)
  [ -n "$smoke_out" ] && mv "$smoke_out" "$dir/" 2>/dev/null || true
  "$PY" - "$dir" "$tag" "$steps" "${LADDER_LOAD_CALIBER:-unmarked}" "${LADDER_COHORT_ID:-028-qwen35-35b-a3b-tp2-fullgraph}" "${LADDER_RUN_NOTE:-}" <<'EOF'
import glob, json, os, sys, time
d, tag, steps, cal, coh, note = sys.argv[1:7]
def newest(pat):
    hits = sorted(glob.glob(f"{os.path.dirname(d)}/{pat}"), key=os.path.getmtime)
    return hits[-1] if hits else ""
art = {"l1": newest("*-l1.json"), "l2": newest("*-l2.json"), "l3": newest("*-l3.json")}
smoke = sorted(glob.glob(f"{d}/eval-ladder-smoke-*/**/results_*.json", recursive=True))
doc = {"schema": "opt-ops-run-index.v1", "tag": tag, "steps": steps.split(","),
       "ts": time.strftime("%FT%TZ", time.gmtime()),
       "artifacts": art | {"smoke_results": smoke[0] if smoke else ""},
       "load_caliber": cal, "cohort": coh, "note": note,
       "why": "run 目录只放索引 + steps.log + smoke 目录; 权威产物是 artifacts 指向的 records/ladder/<ts>-*.json（不再拷贝副本）"}
json.dump(doc, open(f"{d}/run.json", "w"), ensure_ascii=False, indent=1)
print(f"run 索引 -> {d}/run.json（权威产物: {os.path.basename(art['l1'] or '-')} / "
      f"{os.path.basename(art['l2'] or '-')} / {os.path.basename(art['l3'] or '-')}）")
EOF
  echo "== run '$tag' 完成 =="
  # 单页实验门禁报告: 有基线就对比打印（LADDER_RUN_BASELINE=<tag|目录> 显式指定; 否则用上次 run 指针）
  local baseline=${LADDER_RUN_BASELINE:-}
  if [ -z "$baseline" ] && [ -s "$RECORDS/.last-run" ]; then
    local prev; prev=$(cat "$RECORDS/.last-run" 2>/dev/null || true)
    [ -n "$prev" ] && [ "$prev" != "$dir" ] && baseline=$prev
  fi
  if [ -n "$baseline" ]; then
    echo; echo "== 实验门禁报告（基线 $(basename "$baseline") → $tag）=="
    "$PY" "$SCRIPTS/ladder-report.py" "$baseline" "$dir" || true
  else
    echo "（无基线指针: 这是第一个 run；下次 run 会自动与它对比，或显式 LADDER_RUN_BASELINE=<tag>）"
  fi
  printf '%s\n' "$dir" > "$RECORDS/.last-run"
  # 收尾挂日志压缩（LADDER_LOG_KEEP_DAYS=0 可放开全压; 失败不影响主流程）
  cmd_logs || true
}

cmd_report() { "$PY" "$SCRIPTS/ladder-report.py" "$@"; }

# 日志收尾：gzip 已完结的 ladder-*.log，默认保留最近 LADDER_LOG_KEEP_DAYS=3 天未压缩。
# 只动 logs/ 下的日志，绝不触碰 records/** 的 csv/json（那些是被卡与台账引用的证据路径）。
# 已被 L2/KB 按绝对路径引用的历史日志（ladder-serve.log、ladder-parse-fixed-1013.log 等）跳过。
cmd_logs() {
  local keep_days=${LADDER_LOG_KEEP_DAYS:-3}
  local dry=0; [ "${1:-}" = "--dry-run" ] && dry=1
  local cited="$LOGS/.log-cited"          # 每行一个文件名，人工/脚本维护
  local n=0 freed=0
  echo "== 日志收尾: $LOGS（保留最近 ${keep_days} 天未压缩）=="
  local f s
  while IFS= read -r f; do
    [ -f "$f" ] || continue
    case "$f" in *.gz) continue ;; esac
    # 1) 被引用保护
    if [ -f "$cited" ] && grep -qxF "$(basename "$f")" "$cited"; then
      echo "   跳过(被引用): $(basename "$f")"; continue
    fi
    # 2) 近 keep_days 天内仍在用
    if [ "$keep_days" -gt 0 ] && [ -n "$(find "$f" -newermt "-${keep_days} days" 2>/dev/null)" ]; then
      echo "   跳过(近 ${keep_days} 天): $(basename "$f")"; continue
    fi
    s=$(stat -c%s "$f" 2>/dev/null || echo 0)
    if [ "$dry" = 1 ]; then
      echo "   [dry-run] 待压: $(basename "$f") ($s 字节)"
    else
      gzip -9 "$f" && { n=$((n + 1)); freed=$((freed + s - $(stat -c%s "$f.gz" 2>/dev/null || echo 0))); }
    fi
  done < <(find "$LOGS" -maxdepth 1 -name 'ladder-*.log' -type f | sort)
  [ "$dry" = 1 ] || echo "== 压缩 $n 个, 省 $((freed / 1024)) KB; 当前 logs: $(du -sh "$LOGS" | cut -f1) =="
}

case "${1:-help}" in
  serve) shift; cmd_serve "$@" ;;
  stop) cmd_stop ;;
  logs) shift; cmd_logs "$@" ;;
  wait) shift; cmd_wait "$@" ;;
  l1) shift; cmd_l1 "$@" ;;
  l2) shift; cmd_l2 "$@" ;;
  l3) shift; cmd_l3 "$@" ;;
  compare) cmd_compare "$2" "$3" ;;
  smoke) shift; cmd_smoke "$@" ;;
  l0) shift; cmd_l0 "$@" ;;
  screen) shift; cmd_screen "$@" ;;
  carriers) shift; carrier_probe_json "${1:-}" ;;
  profile) shift; cmd_profile "$@" ;;
  slim) shift; cmd_slim "$@" ;;
  recover-profile) shift; cmd_recover_profile "$@" ;;
  run) shift; cmd_run "$@" ;;
  report) shift; cmd_report "$@" ;;
  *) sed -n '2,26p' "$0" | sed 's/^# \{0,1\}//' ;;
esac
