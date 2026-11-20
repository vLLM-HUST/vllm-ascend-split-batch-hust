"""l0_trace_record 的离线门禁：机械入账不改数字、不动结论标签。"""

from __future__ import annotations

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from tools.operator_tools import l0_trace_record as recorder  # noqa: E402


def _make_operator_dir(base: Path, operator: str = "demo_op") -> Path:
    op_dir = base / "ops" / operator
    op_dir.mkdir(parents=True)
    for name in recorder.CONTRACT_FILES:
        (op_dir / name).write_text("{}\n", encoding="utf-8")
    return op_dir


def _make_receipt(base: Path, operator: str, passed: bool = True) -> Path:
    receipt = {
        "operator": operator,
        "scope": "isolated operator microbenchmark",
        "torch": "2.10.0+cpu",
        "device": "Ascend910B2",
        "seed": 42,
        "warmup": 10,
        "samples": 100,
        "timing": "NPU event per replay",
        "argv": ["python", "graph_retest.py", operator],
        "script_sha256": "0" * 64,
        "cases": [
            {
                "shape": [[8, 1024]],
                "stride": [[1024, 1]],
                "atol": 0.001,
                "passed": passed,
                "speedup": 1.25,
                "baseline": {
                    "median_us": 4.0,
                    "p90_us": 5.0,
                    "p99_us": 6.0,
                    "samples_us": [4.0] * 100,
                },
                "candidate": {
                    "median_us": 3.2,
                    "p90_us": 4.0,
                    "p99_us": 5.0,
                    "samples_us": [3.2] * 100,
                },
            }
        ],
    }
    path = base / f"{operator}-graph.json"
    path.write_text(json.dumps(receipt), encoding="utf-8")
    return path


def test_receipt_trace_is_recorded_and_custody_refreshed(tmp_path, monkeypatch):
    op_dir = _make_operator_dir(tmp_path)
    monkeypatch.setattr(recorder, "OPERATORS_DIR", tmp_path / "ops")
    evidence = op_dir / "evidence.json"
    evidence.write_text(
        json.dumps(
            {"operator_name": "demo_op", "serving_end_to_end": {"tested": False}}
        ),
        encoding="utf-8",
    )
    receipt = _make_receipt(tmp_path, "demo_op")

    message = recorder.append_trace(
        "demo_op", recorder.build_receipt_trace("demo_op", receipt, None, None)
    )

    data = json.loads(evidence.read_text(encoding="utf-8"))
    trace = data["l0_traces"][-1]
    assert trace["kind"] == "l0_receipt_trace"
    case = trace["cases"][0]
    assert case["baseline"]["median_us"] == 4.0
    assert "samples_us" not in case["baseline"]
    assert trace["case_summary"] == {"total": 1, "passed": 1, "failed": 0}
    assert data["serving_end_to_end"] == {"tested": False}
    assert "matched checksums" in message


def test_failed_cases_are_recorded_not_dropped(tmp_path, monkeypatch):
    _make_operator_dir(tmp_path)
    monkeypatch.setattr(recorder, "OPERATORS_DIR", tmp_path / "ops")
    receipt = _make_receipt(tmp_path, "demo_op", passed=False)

    recorder.append_trace(
        "demo_op", recorder.build_receipt_trace("demo_op", receipt, None, None)
    )
    data = json.loads((tmp_path / "ops" / "demo_op" / "evidence.json").read_text())
    trace = data["l0_traces"][-1]
    assert trace["cases"][0]["passed"] is False
    assert trace["case_summary"]["failed"] == 1


def test_operator_mismatch_is_rejected(tmp_path):
    _make_operator_dir(tmp_path)
    receipt = _make_receipt(tmp_path, "other_op")
    try:
        recorder.build_receipt_trace("demo_op", receipt, None, None)
    except ValueError as error:
        assert "other_op" in str(error)
    else:
        raise AssertionError("operator 不匹配应当拒绝")


def test_bench_execution_records_placeholder_exit(tmp_path, monkeypatch):
    op_dir = _make_operator_dir(tmp_path)
    (op_dir / "bench_l0.py").write_text(
        "print('No standalone measurement provided.')\nraise SystemExit(3)\n",
        encoding="utf-8",
    )
    monkeypatch.setattr(recorder, "OPERATORS_DIR", tmp_path / "ops")
    monkeypatch.setattr(recorder, "ROOT", tmp_path)

    stdout, exit_code = recorder._run_bench("demo_op")
    record = recorder.build_bench_execution_trace("demo_op", stdout, exit_code, None)

    assert record["exit_code"] == 3
    assert record["no_measurement"] is True
    assert "No standalone measurement" in record["stdout"]
    assert len(record["bench_sha256"]) == 64
