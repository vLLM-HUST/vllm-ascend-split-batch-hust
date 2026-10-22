from __future__ import annotations

import json
import os
import subprocess
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "tools" / "bench-ladder.sh"

# 真实「载体生效」serve 日志版式（摘录自 ladder-serve-1017-081645.log）:
#   4 条 `cascade plugin loaded` = APIServer + EngineCore + 2 worker（每进程一条）;
#   `Gate fp32 cache populated` / `W3F2: GDN in_proj merge applied` 各 TP0/TP1 一条;
#   `gdn_merge_inproj_plugin.py:195` 的 fail-closed 警告（插件不重复合并, 不是失败）。
# 关键: 这份日志故意带着 `Unknown vLLM environment variable` 两行 —— 载体生效时也会出现,
# 所以它不是缺席判据; carriers 记录把两件事分开（marker 计数 vs unreliable）。
SERVE_LOG_MARKERS = """\
INFO [cascade_plugin.py:175] cascade plugin loaded (gate=0, graph_gate=0)
WARN Unknown vLLM environment variable detected: VLLM_HUST_GDN_MERGE_INPROJ
WARN Unknown vLLM environment variable detected: VLLM_HUST_GATE_FP32
INFO [cascade_plugin.py:175] cascade plugin loaded (gate=0, graph_gate=0)
INFO [cascade_plugin.py:175] cascade plugin loaded (gate=0, graph_gate=0)
INFO [cascade_plugin.py:175] cascade plugin loaded (gate=0, graph_gate=0)
INFO [model_runner_v1.py:3979] W3F2: GDN in_proj merge applied on 30 layer(s)
INFO [gate_fp32_plugin.py:87] Gate fp32 cache populated: 40 MoE gates
WARNING [gdn_merge_inproj_plugin.py:195] the GDN merge carrier is fail-closed
INFO [model_runner_v1.py:3979] W3F2: GDN in_proj merge applied on 30 layer(s)
INFO [gate_fp32_plugin.py:87] Gate fp32 cache populated: 40 MoE gates
WARNING [gdn_merge_inproj_plugin.py:195] the GDN merge carrier is fail-closed
"""

W3F3_APPLIED = (
    "INFO [model_runner_v1.py:4024] "
    "W3F3: shared scalar-gate pack (513direct) applied on 30 layer(s)\n"
)
W3F3_FAILED = (
    "WARNING [model_runner_v1.py:4026] "
    "W3F3: shared scalar-gate pack failed: target width 513 < rows+1=640\n"
)

# carriers 记录的稳定字段（消费者/台账按这些键读; 只能加键, 不得改名）
CARRIER_KEYS = {
    "schema",
    "probe_python",
    "husk_package",
    "husk_package_found",
    "husk_module_spec",
    "husk_entry_points",
    "husk_entry_point_names",
    "env",
    "env_scope",
    "env_source",
    "env_file",
    "serve_log",
    "serve_log_present",
    "serve_log_markers",
    "unreliable_absence_signals",
    "attribution",
    "verdict",
    "probe_errors",
}

ENV_KEYS = {
    "VLLM_HUST_GATE_FP32",
    "VLLM_HUST_GDN_MERGE_INPROJ",
    "VLLM_HUST_SHARED_GATE_PACK",
}

MARKER_COUNTS = {
    "cascade_plugin_loaded": 4,
    "gate_fp32_cache_populated": 2,
    "gdn_merge_inproj_fail_closed": 2,
    "w3f2_gdn_merge_applied": 2,
    "w3f2_gdn_merge_skipped": 0,
    "w3f3_shared_gate_pack_applied": 0,
    "w3f3_shared_gate_pack_failed": 0,
}

NOISY_ENV = {
    "LADDER_CARRIER_PYTHON",
    "LADDER_SERVE_LOG",
    "LADDER_SERVE_ENV",
    "LADDER_MODEL",
    "LADDER_RUN_ACTIVE",
    "VLLM_HUST_GATE_FP32",
    "VLLM_HUST_GDN_MERGE_INPROJ",
    "VLLM_HUST_SHARED_GATE_PACK",
}


def clean_env(extra: dict[str, str] | None = None) -> dict[str, str]:
    base = {k: v for k, v in os.environ.items() if k not in NOISY_ENV}
    base.update(extra or {})
    return base


def run_ladder(
    *args: str, env: dict[str, str] | None = None
) -> subprocess.CompletedProcess[str]:
    """跑 bench-ladder.sh 子命令（清掉会改变探针解释器/日志/env 的 LADDER_* 变量）。"""
    return subprocess.run(
        ["bash", str(SCRIPT), *args],
        capture_output=True,
        text=True,
        check=False,
        env=clean_env(env),
        timeout=120,
    )


def probe(*, python: str, log: Path | None, env_file: Path | None = None) -> dict:
    """跑 `bench-ladder.sh carriers` 取单行 JSON（取末行, 防 stdout 掺日志）。"""
    env = {"LADDER_CARRIER_PYTHON": python}
    # 必须给具体路径: 空串会回退读 /tmp/ladder-serve-log（不确定）
    env["LADDER_SERVE_LOG"] = str(log) if log else "/nonexistent/ladder-serve-test.log"
    if env_file is not None:
        env["LADDER_SERVE_ENV"] = str(env_file)
    proc = run_ladder("carriers", env=env)
    assert proc.returncode == 0, proc.stderr
    return json.loads(proc.stdout.strip().splitlines()[-1])


def write_serve_env(
    path: Path, log: Path, *, keys: dict[str, str | None] | None = None
) -> subprocess.CompletedProcess[str]:
    """用仓内**真实** writer 落 serve 侧 env 快照（与 cmd_serve 同一实现）。"""
    script = (
        'source "$1" __src >/dev/null 2>&1\n'
        'PY="$2"\n'
        'carrier_write_serve_env "$3" "$4" 4242\n'
    )
    env_vars = {k: v for k, v in (keys or {}).items() if v is not None}
    return subprocess.run(
        [
            "bash",
            "-c",
            script,
            "bash",
            str(SCRIPT),
            sys.executable,
            str(path),
            str(log),
        ],
        capture_output=True,
        text=True,
        check=False,
        env=clean_env(env_vars),
        timeout=120,
    )


def test_bench_ladder_shell_syntax() -> None:
    result = subprocess.run(
        ["bash", "-n", str(SCRIPT)],
        capture_output=True,
        text=True,
        check=False,
    )
    assert result.returncode == 0, result.stderr


def test_light_profile_is_bounded_and_minimal() -> None:
    body = SCRIPT.read_text()
    assert "LADDER_PROFILE_MAX_ITERS:-32" in body
    assert "--profiler-config.max_iterations" in body
    assert "--profiler-config.ignore_frontend=true" in body
    assert "LADDER_PROFILE_WITH_STACK:-false" in body
    assert "--profiler-config.torch_profiler_with_memory=false" in body


def test_plain_screen_precedes_profile() -> None:
    body = SCRIPT.read_text()
    assert "cmd_screen()" in body
    assert "LADDER_SCREEN_REPEATS:-3" in body
    assert '"schema": "opt-ops-screen.v1"' in body
    assert '"mode": "plain"' in body


def test_plain_screen_records_carriers() -> None:
    """plain screen 的 screen.json 必须带载体记录（记录轴, 不是门禁）。"""
    body = SCRIPT.read_text()
    assert "carrier_probe_json" in body
    assert '"schema": "opt-ops-carriers.v2"' in body
    assert '"carriers": carriers' in body
    assert "carriers_note" in body
    assert "carriers_raw" in body
    assert "carriers 不可解析" in body
    assert "carriers) shift;" in body
    # 只记录: 探针失败不得变成 screen 的返回码
    assert "carriers_json=$(carrier_probe_json)" in body


def test_screen_next_gate_matches_ready_or_validating() -> None:
    """F6: screen.json 的 next 文案与 opt-ops/README 的采集门口径一致。"""
    body = SCRIPT.read_text()
    assert "候选晋级 `ready|validating` 后才可开 light L4" in body
    assert "候选晋级 ready 才可开 light L4" not in body


def test_serve_env_exports_precede_cohort_record() -> None:
    """F1: export 必须早于 cohort_write，否则 cohort.json 的 carriers 记成未导出。"""
    body = SCRIPT.read_text()
    gate = body.index("export VLLM_HUST_GATE_FP32=")
    gdn = body.index("export VLLM_HUST_GDN_MERGE_INPROJ=")
    call = body.index("\n    cohort_write\n")
    assert gate < call and gdn < call


def test_serve_writes_env_snapshot_atomically() -> None:
    """F1: 起服时落同 stem 的 <log>.env.json（原子写 + 不覆盖 + 含 W3F3 键）。"""
    body = SCRIPT.read_text()
    assert "carrier_write_serve_env" in body
    assert "carrier_env_path" in body
    assert "os.replace" in body
    assert '"schema": "opt-ops-serve-env.v1"' in body
    assert "VLLM_HUST_SHARED_GATE_PACK" in body
    # cmd_serve 起服后必须真的落一次快照
    assert 'carrier_write_serve_env "$serve_env" "$log"' in body


def test_serve_reports_carrier_attribution_after_ready() -> None:
    body = SCRIPT.read_text()
    # 就绪后 + 复用已在跑的 serve 都打摘要; 摘要来自探针, 不是 Unknown env 警告
    assert body.count('carrier_summary "$(carrier_probe_json') == 2
    assert "Unknown vLLM env 警告" in body
    assert "is_absence_signal" in body
    assert "runtime_tree" in body


def test_carrier_probe_counts_serve_log_markers(tmp_path: Path) -> None:
    log = tmp_path / "ladder-serve-test.log"
    log.write_text(SERVE_LOG_MARKERS)
    doc = probe(python=sys.executable, log=log)

    assert doc["schema"] == "opt-ops-carriers.v2"
    assert doc.keys() >= CARRIER_KEYS
    assert doc["probe_python"] == sys.executable
    assert doc["serve_log"] == str(log)
    assert doc["serve_log_present"] is True

    markers = doc["serve_log_markers"]
    assert {k: v["count"] for k, v in markers.items()} == MARKER_COUNTS
    # 行号必须对得上 grep -n（不因日志里的孤立 CR 偏移）
    assert markers["cascade_plugin_loaded"]["first_line"] == 1
    assert markers["w3f2_gdn_merge_applied"]["first_line"] == 7
    assert markers["gate_fp32_cache_populated"]["first_line"] == 8

    # Unknown env 警告单独记、并显式标记「不是缺席判据」
    un = doc["unreliable_absence_signals"]["unknown_vllm_env_warning"]
    assert un["count"] == 2
    assert un["is_absence_signal"] is False


def test_attribution_splits_husk_plugin_and_runtime_tree(tmp_path: Path) -> None:
    """F2: 运行树 W3F2 不得被聚合成「仓内插件做了合并」。"""
    log = tmp_path / "serve.log"
    log.write_text(SERVE_LOG_MARKERS)
    doc = probe(python=sys.executable, log=log)
    husk = doc["attribution"]["husk"]
    tree = doc["attribution"]["runtime_tree"]

    assert husk["cascade"]["state"] == "effective"
    assert husk["gate_fp32"]["state"] == "effective"
    assert husk["gdn_merge_inproj"]["state"] == "present-but-fail-closed"
    assert "HOST_CONTRACT.md" in husk["gdn_merge_inproj"]["note"]
    assert "不得把 W3F2 的合并归因给本插件" in husk["gdn_merge_inproj"]["note"]
    assert tree["w3f2"]["state"] == "applied"
    assert tree["w3f3"]["state"] == "not-enabled"

    # 总体 verdict 不再有跨载体混算字段
    assert "functional_markers" not in doc["verdict"]
    assert "functional_markers" not in husk["gdn_merge_inproj"]
    assert doc["verdict"]["cross_carrier_merge"] == "not-computed"
    assert doc["verdict"]["axis"] == "record-only"


@pytest.mark.parametrize(
    ("extra", "expected"),
    [(W3F3_APPLIED, "enabled"), (W3F3_FAILED, "failed")],
)
def test_attribution_w3f3_states(tmp_path: Path, extra: str, expected: str) -> None:
    log = tmp_path / "serve.log"
    log.write_text(SERVE_LOG_MARKERS + extra)
    doc = probe(python=sys.executable, log=log)
    assert doc["attribution"]["runtime_tree"]["w3f3"]["state"] == expected


def test_carrier_probe_reads_serve_env_snapshot(tmp_path: Path) -> None:
    """F1 核心: 调用 shell 无 env, 但 serve 快照写 1 → carriers.env 必须是 1。"""
    log = tmp_path / "ladder-serve-x.log"
    log.write_text(SERVE_LOG_MARKERS)
    env_file = tmp_path / "ladder-serve-x.env.json"
    wrote = write_serve_env(
        env_file,
        log,
        keys={"VLLM_HUST_GATE_FP32": "1", "VLLM_HUST_GDN_MERGE_INPROJ": "1"},
    )
    assert wrote.returncode == 0, wrote.stderr
    snapshot = json.loads(env_file.read_text())
    assert snapshot["scope"] == "serve-process"
    assert snapshot["serve_pid"] == 4242
    assert snapshot["keys_unset"] == ["VLLM_HUST_SHARED_GATE_PACK"]

    doc = probe(python=sys.executable, log=log)
    assert set(doc["env"]) == ENV_KEYS
    assert doc["env"]["VLLM_HUST_GATE_FP32"] == "1"
    assert doc["env"]["VLLM_HUST_GDN_MERGE_INPROJ"] == "1"
    # 未导出的键是 null（不是 "" / 不是 "0"）
    assert doc["env"]["VLLM_HUST_SHARED_GATE_PACK"] is None
    assert doc["env_scope"] == "serve-process"
    assert doc["env_source"] == f"serve-env-file:{env_file}"
    assert doc["env_file"]["present"] is True
    assert doc["env_file"]["serve_pid"] == 4242
    assert doc["probe_errors"] == []
    # marker 与 env 自洽（不再出现 "键为空 + marker effective"）
    assert doc["attribution"]["husk"]["gate_fp32"]["state"] == "effective"


def test_carrier_probe_env_unknown_when_snapshot_missing(tmp_path: Path) -> None:
    """F1: 快照缺失 → env 全 null + env_scope/env_source=unknown, 不冒充 off。"""
    log = tmp_path / "serve.log"
    log.write_text(SERVE_LOG_MARKERS)
    doc = probe(python=sys.executable, log=log)

    assert set(doc["env"]) == ENV_KEYS
    assert all(v is None for v in doc["env"].values())
    assert doc["env_scope"] == "unknown"
    assert doc["env_source"] == "unknown"
    assert doc["env_file"]["present"] is False
    assert any("serve env 快照不存在" in e for e in doc["probe_errors"])
    # marker 仍然照实记（unknown env 与 effective marker 不是矛盾, 是两件事）
    assert doc["attribution"]["husk"]["cascade"]["state"] == "effective"
    assert doc["attribution"]["husk"]["gate_fp32"]["state"] == "effective"


def test_carrier_probe_rejects_mismatched_env_snapshot(tmp_path: Path) -> None:
    """F1: 快照里的 serve_log 指向别的日志 → 不采信（防跨 serve 串用）。"""
    log = tmp_path / "serve.log"
    log.write_text(SERVE_LOG_MARKERS)
    other = tmp_path / "other.log"
    other.write_text("")
    wrote = write_serve_env(
        tmp_path / "serve.env.json", other, keys={"VLLM_HUST_GATE_FP32": "1"}
    )
    assert wrote.returncode == 0, wrote.stderr

    doc = probe(python=sys.executable, log=log, env_file=tmp_path / "serve.env.json")
    assert all(v is None for v in doc["env"].values())
    assert doc["env_scope"] == "unknown"
    assert any("不匹配" in e for e in doc["probe_errors"])


def test_serve_env_snapshot_is_not_overwritten(tmp_path: Path) -> None:
    path = tmp_path / "s.env.json"
    log = tmp_path / "s.log"
    log.write_text("")
    assert write_serve_env(path, log, keys={"VLLM_HUST_GATE_FP32": "1"}).returncode == 0
    second = write_serve_env(path, log, keys={"VLLM_HUST_GATE_FP32": "0"})
    # 记录轴: 不覆盖已有快照, 但仍不阻断调用方（rc=0 + 明确告警）
    assert second.returncode == 0
    assert "不覆盖" in second.stderr
    assert json.loads(path.read_text())["keys"]["VLLM_HUST_GATE_FP32"] == "1"


def test_carrier_probe_records_absent_package_without_crashing(
    tmp_path: Path,
) -> None:
    """载体缺席（包找不到 / 入口点为 0）→ 记录 false/absent, 脚本仍退出 0。"""
    log = tmp_path / "serve.log"
    log.write_text(SERVE_LOG_MARKERS)
    # -S 起解释器 = 不处理 site-packages（等价于 overlay 里的 husk 安装被抹掉）
    site_blocker = tmp_path / "python-nosite"
    site_blocker.write_text(f'#!/bin/sh\nexec "{sys.executable}" -S "$@"\n')
    site_blocker.chmod(0o755)

    doc = probe(python=str(site_blocker), log=log)

    assert doc.keys() >= CARRIER_KEYS
    assert doc["husk_package"] == ""
    assert doc["husk_package_found"] is False
    assert doc["husk_module_spec"] == "absent"
    assert doc["husk_entry_points"] == []
    assert doc["husk_entry_point_names"] == []
    assert doc["attribution"]["husk"]["package"] == "absent"
    assert doc["attribution"]["husk"]["entry_points"] in {"empty", "unknown"}
    assert doc["attribution"]["husk"]["cascade"]["state"] == "absent"
    # 运行树补丁与 husk 包无关: 日志有 W3F2 就照记 applied
    assert doc["attribution"]["runtime_tree"]["w3f2"]["state"] == "applied"


def test_carrier_probe_is_fail_closed_when_probe_python_breaks(
    tmp_path: Path,
) -> None:
    """探针解释器坏掉 + 日志缺失 → unknown/null 记录, 绝不崩（rc=0, JSON 可解析）。"""
    broken = tmp_path / "broken-python"
    broken.write_text("#!/bin/sh\nexit 1\n")
    broken.chmod(0o755)

    doc = probe(python=str(broken), log=tmp_path / "missing.log")

    assert doc.keys() >= CARRIER_KEYS
    assert doc["husk_package_found"] is False
    assert doc["husk_package"] == ""
    assert doc["husk_module_spec"] == "unknown"
    assert doc["serve_log_present"] is False
    assert doc["serve_log_markers"] == {}
    assert all(v is None for v in doc["env"].values())
    assert doc["env_scope"] == "unknown"
    assert doc["attribution"]["husk"]["cascade"]["state"] == "unknown"
    assert doc["attribution"]["runtime_tree"]["w3f2"]["state"] == "unknown"
    assert doc["verdict"]["axis"] == "record-only"
    assert doc["probe_errors"]


def test_carrier_probe_marks_missing_log_as_unknown(tmp_path: Path) -> None:
    """日志路径给了但文件不存在 → marker 记 unknown + 原因, 不当「没有 marker」。"""
    doc = probe(python=sys.executable, log=tmp_path / "missing.log")
    assert doc["serve_log_present"] is False
    assert doc["attribution"]["husk"]["gate_fp32"]["state"] == "unknown"
    assert doc["attribution"]["runtime_tree"]["w3f2"]["state"] == "unknown"
    assert any("serve 日志不存在" in e for e in doc["probe_errors"])


@pytest.mark.parametrize("has_markers", [True, False])
def test_carrier_summary_is_human_readable(tmp_path: Path, has_markers: bool) -> None:
    """cmd_serve 就绪后的载体摘要（人读; 记录轴 + 归因拆分要写在脸上）。"""
    log = tmp_path / "serve.log"
    log.write_text(SERVE_LOG_MARKERS if has_markers else "plain stock serve log\n")
    env_file = tmp_path / "serve.env.json"
    wrote = write_serve_env(env_file, log, keys={"VLLM_HUST_GATE_FP32": "1"})
    assert wrote.returncode == 0, wrote.stderr
    doc = probe(python=sys.executable, log=log)

    script = 'source "$1" __src >/dev/null 2>&1\nPY="$3"\ncarrier_summary "$2"\n'
    proc = subprocess.run(
        ["bash", "-c", script, "bash", str(SCRIPT), json.dumps(doc), sys.executable],
        capture_output=True,
        text=True,
        check=False,
        timeout=120,
    )
    assert proc.returncode == 0, proc.stderr
    out = proc.stdout
    assert "载体（记录轴" in out
    assert "VLLM_HUST_GATE_FP32=1" in out
    assert "serve env（scope=serve-process" in out
    assert "不是**缺席判据" in out
    assert "归因（仓内 husk 插件 / 运行树补丁 分开记" in out
    if has_markers:
        assert "cascade_plugin_loaded: count=4" in out
        assert "husk.gate_fp32=effective" in out
        assert "husk.gdn_merge_inproj=present-but-fail-closed" in out
        assert "runtime_tree.w3f2=applied" in out
    else:
        assert "cascade_plugin_loaded: count=0" in out
        assert "husk.gate_fp32=present-not-observed" in out
        assert "runtime_tree.w3f2=not-applied" in out


def test_profile_output_records_capture_scope() -> None:
    body = SCRIPT.read_text()
    assert 'doc["trace_tier"] = scope["tier"]' in body
    assert 'doc["capture_scope"]' in body
    assert '"schema": "opt-ops-profile-plan.v1"' in body


def test_light_slim_drops_profiler_databases() -> None:
    body = SCRIPT.read_text()
    assert 'LADDER_SLIM_RESCUE_DB="$require_shapes"' in body
    assert 'if [ "$rescue_db" != "1" ]' in body
    assert "ASCEND_PROFILER_OUTPUT/*.db" in body
