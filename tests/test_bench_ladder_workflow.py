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
    "serve_log",
    "serve_log_present",
    "serve_log_markers",
    "unreliable_absence_signals",
    "verdict",
    "probe_errors",
}

MARKER_COUNTS = {
    "cascade_plugin_loaded": 4,
    "gate_fp32_cache_populated": 2,
    "w3f2_gdn_merge_applied": 2,
    "gdn_merge_inproj_fail_closed": 2,
}

NOISY_ENV = {
    "LADDER_CARRIER_PYTHON",
    "LADDER_SERVE_LOG",
    "LADDER_MODEL",
    "LADDER_RUN_ACTIVE",
}


def run_ladder(
    *args: str, env: dict[str, str] | None = None
) -> subprocess.CompletedProcess[str]:
    """跑 bench-ladder.sh 子命令（清掉会改变探针解释器/日志解析的 LADDER_* 变量）。"""
    base = {k: v for k, v in os.environ.items() if k not in NOISY_ENV}
    base.update(env or {})
    return subprocess.run(
        ["bash", str(SCRIPT), *args],
        capture_output=True,
        text=True,
        check=False,
        env=base,
        timeout=120,
    )


def probe(*, python: str, log: Path | None) -> dict:
    """跑 `bench-ladder.sh carriers` 取单行 JSON（取末行, 防 stdout 掺日志）。"""
    proc = run_ladder(
        "carriers",
        env={
            "LADDER_CARRIER_PYTHON": python,
            # 必须给具体路径: 空串会回退读 /tmp/ladder-serve-log（不确定）
            "LADDER_SERVE_LOG": (
                str(log) if log else "/nonexistent/ladder-serve-test.log"
            ),
        },
    )
    assert proc.returncode == 0, proc.stderr
    return json.loads(proc.stdout.strip().splitlines()[-1])


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
    assert "--profiler-config.torch_profiler_with_stack=false" in body
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
    # 探针接入点 + screen.json 的 carriers 键 + 探针不可解析时的 fail-closed 记录
    assert "carrier_probe_json" in body
    assert '"schema": "opt-ops-carriers.v1"' in body
    assert '"carriers": carriers' in body
    assert "carriers_note" in body
    assert "carriers_raw" in body
    assert "carriers 不可解析" in body
    assert "carriers) shift;" in body
    # 只记录: 探针失败不得变成 screen 的返回码
    assert "carriers_json=$(carrier_probe_json)" in body


def test_serve_reports_carrier_markers_after_ready() -> None:
    body = SCRIPT.read_text()
    # 就绪后 + 复用已在跑的 serve 都打摘要; 摘要来自探针, 不是 Unknown env 警告
    assert body.count('carrier_summary "$(carrier_probe_json') == 2
    assert "Unknown vLLM env 警告" in body
    assert "is_absence_signal" in body


def test_carrier_probe_counts_serve_log_markers(tmp_path: Path) -> None:
    log = tmp_path / "ladder-serve-test.log"
    log.write_text(SERVE_LOG_MARKERS)
    doc = probe(python=sys.executable, log=log)

    assert doc["schema"] == "opt-ops-carriers.v1"
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

    # 两个开关键都要在记录里（未设置记空串, 不是缺键）
    assert set(doc["env"]) == {"VLLM_HUST_GATE_FP32", "VLLM_HUST_GDN_MERGE_INPROJ"}
    assert doc["verdict"]["functional_markers"] == "effective"
    assert doc["verdict"]["axis"] == "record-only"

    # Unknown env 警告单独记、并显式标记「不是缺席判据」
    un = doc["unreliable_absence_signals"]["unknown_vllm_env_warning"]
    assert un["count"] == 2
    assert un["is_absence_signal"] is False


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
    assert doc["verdict"]["package"] == "absent"
    assert doc["verdict"]["entry_points"] in {"empty", "unknown"}


def test_carrier_probe_is_fail_closed_when_probe_python_breaks(
    tmp_path: Path,
) -> None:
    """探针解释器坏掉 + 日志缺失 → unknown/false 记录, 绝不崩（rc=0, JSON 可解析）。"""
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
    assert doc["verdict"]["functional_markers"] == "unknown"
    assert doc["verdict"]["axis"] == "record-only"
    assert doc["probe_errors"]


def test_carrier_probe_marks_missing_log_as_unknown(tmp_path: Path) -> None:
    """日志路径给了但文件不存在 → marker 记 unknown + 原因, 不当「没有 marker」。"""
    doc = probe(python=sys.executable, log=tmp_path / "missing.log")
    assert doc["serve_log_present"] is False
    assert doc["verdict"]["functional_markers"] == "unknown"
    assert any("serve 日志不存在" in e for e in doc["probe_errors"])


@pytest.mark.parametrize("has_markers", [True, False])
def test_carrier_summary_is_human_readable(tmp_path: Path, has_markers: bool) -> None:
    """cmd_serve 就绪后的载体摘要（人读; 记录轴语义要写在脸上）。"""
    log = tmp_path / "serve.log"
    log.write_text(SERVE_LOG_MARKERS if has_markers else "plain stock serve log\n")
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
    assert "VLLM_HUST_GATE_FP32=" in out
    assert "不是**缺席判据" in out
    if has_markers:
        assert "cascade_plugin_loaded: count=4" in out
        assert "gate_fp32_cache_populated: count=2" in out
        assert "functional_markers=effective" in out
    else:
        assert "cascade_plugin_loaded: count=0" in out
        assert "functional_markers=missing" in out


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
