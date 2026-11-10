"""`experiments/ascend910b-2card/run_baseline_window.py` 的护栏。

窗口脚本要长期跑在共享机器上，所以这里钉的是**不碰设备也必须成立**的部分：

* 产物目录只允许仓库外绝对路径（fail-closed，退出码 2）；
* 计划清单与实际产物文件名一致，未测项必须显式登记；
* 粒度台阶判定能在合成数据上稳定找出翻转点，平坦数据不报台阶；
* 摘要只写口径与结论，不遗漏未测项；
* 模块在 import 与 `--dry-run` 时不加载 torch（CPU 环境可自检）。
"""

from __future__ import annotations

import importlib.util
import json
import os
import subprocess
import sys
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parents[1]
SCRIPT = REPO / "experiments" / "ascend910b-2card" / "run_baseline_window.py"

spec = importlib.util.spec_from_file_location("run_baseline_window", SCRIPT)
window = importlib.util.module_from_spec(spec)
sys.modules["run_baseline_window"] = window
spec.loader.exec_module(window)


@pytest.fixture(autouse=True)
def _restore_results_env():
    saved = os.environ.get("SPLIT_BATCH_RESULTS_DIR")
    yield
    if saved is None:
        os.environ.pop("SPLIT_BATCH_RESULTS_DIR", None)
    else:
        os.environ["SPLIT_BATCH_RESULTS_DIR"] = saved


def test_out_dir_inside_repo_is_rejected():
    with pytest.raises(window.WindowError):
        window.resolve_out_dir(str(REPO / "tmp-window"))


def test_out_dir_must_be_absolute():
    with pytest.raises(window.WindowError):
        window.resolve_out_dir("relative/results")


def test_out_dir_outside_repo_is_accepted(tmp_path):
    assert window.resolve_out_dir(str(tmp_path)) == tmp_path.resolve()


def test_out_dir_missing_env_is_rejected(monkeypatch):
    monkeypatch.delenv("SPLIT_BATCH_RESULTS_DIR", raising=False)
    with pytest.raises(window.WindowError):
        window.resolve_out_dir(None)


def test_manifest_lists_every_artifact_and_not_measured_item(tmp_path):
    manifest = window.plan_manifest("1109-1400", tmp_path)
    names = [Path(p).name for p in manifest["artifacts"]]
    assert names == list(window.ARTIFACTS)
    assert {p["id"] for p in manifest["phases"]} == {"A", "B", "C", "D", "E", "F"}
    items = [row["item"] for row in manifest["not_measured"]]
    assert any("UB" in item for item in items)
    assert all(row["reason"] for row in manifest["not_measured"])


def test_granularity_step_found_on_synthetic_cliff():
    cases = [
        {"row_bytes": 2, "gb_per_s": 10.0},
        {"row_bytes": 16, "gb_per_s": 15.0},
        {"row_bytes": 32, "gb_per_s": 304.0},
        {"row_bytes": 128, "gb_per_s": 1177.0},
    ]
    verdict = window.detect_granularity_step(cases)
    assert verdict["step"]["row_bytes"] == window.ALIGN_BYTES
    # 台阶前的基线取滚动最大值（16 字节档的 15.0），不是最小档。
    assert verdict["step"]["previous_gb_per_s"] == 15.0
    assert verdict["step"]["ratio"] == pytest.approx(304.0 / 15.0, abs=0.1)


def test_granularity_flat_curve_reports_no_step():
    cases = [{"row_bytes": b, "gb_per_s": 100.0} for b in (2, 32, 128)]
    verdict = window.detect_granularity_step(cases)
    assert verdict["step"] is None


def test_granularity_rejects_empty_cases():
    with pytest.raises(window.WindowError):
        window.detect_granularity_step([])


def _fake_window(tmp_path: Path) -> dict:
    return {
        "stamp": "1109-1400",
        "generated_at": "2026-11-09T14:00:00Z",
        "output_dir": str(tmp_path / "1109-1400"),
        "env": {
            "device_count": 2,
            "devices": [{"name": "Ascend910B2"}],
            "python": "3.12.13",
            "torch": "2.10.0",
            "torch_npu": "2.10.0.post4",
            "host_repos": {"vllm_ascend": "abc", "vllm_hust": "def"},
        },
        "bandwidth": {"steady_gb_per_s": 1137.4, "peak_case": {"size_mib": 256}},
        "cube": {"peak": {"tflops": 288.6, "dtype": "fp16", "m": 4096}},
        "hbm_ceiling": {"allocated_gib": 40, "hit_cap": True},
        "granularity": {
            "verdict": {"step": {"row_bytes": 32}},
            "agrees_with_expectation": True,
        },
        "not_measured": window.plan_manifest("1109-1400", tmp_path)["not_measured"],
    }


def test_summary_carries_calibration_and_gaps(tmp_path):
    text = window.build_summary(_fake_window(tmp_path))
    assert "1137.4" in text and "288.6" in text
    assert "32" in text
    assert "未测" in text
    assert "片上 UB / L1 容量" in text
    for name in window.ARTIFACTS:
        assert name in text


def test_dry_run_prints_plan_and_exits_zero(tmp_path, capsys):
    assert window.main(["--dry-run", "--out-dir", str(tmp_path), "--stamp", "s1"]) == 0
    payload = json.loads(capsys.readouterr().out)
    assert payload["stamp"] == "s1"
    assert payload["output_dir"] == str(tmp_path / "s1")


def test_inside_repo_out_dir_exits_two(capsys):
    code = window.main(["--dry-run", "--out-dir", str(REPO / "tmp-window")])
    assert code == 2
    assert "fail-closed" in capsys.readouterr().err


def test_module_import_and_dry_run_do_not_import_torch(tmp_path):
    """CPU CI 没有 torch：import 与 --dry-run 都必须不碰它。"""
    code = (
        "import importlib.util,sys;"
        f"spec=importlib.util.spec_from_file_location('w',r'{SCRIPT}');"
        "m=importlib.util.module_from_spec(spec);sys.modules['w']=m;"
        "spec.loader.exec_module(m);"
        f"assert m.main(['--dry-run','--out-dir',r'{tmp_path}'])==0;"
        "assert 'torch' not in sys.modules, 'dry-run 不该加载 torch'"
    )
    out = subprocess.run(
        [sys.executable, "-c", code], capture_output=True, text=True, timeout=120
    )
    assert out.returncode == 0, out.stderr
