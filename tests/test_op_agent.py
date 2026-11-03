from __future__ import annotations

import pytest

# The package CI runs without numpy / torch / smolagents installed.
# Mark the whole module to skip if optional dependencies are not present.
pytest.importorskip("numpy", reason="op agent tests need numpy")
pytest.importorskip("smolagents", reason="op agent tests need smolagents")

from tools.op_agent.custody import verify_dir_checksum
from tools.op_agent.evaluator import run_candidate_evaluation
from tools.op_agent.guard import check_npu_environment
from tools.op_agent.optimizer_loop import MockKernelModel, get_configured_model
from tools.op_agent.qwen35_shapes import CANONICAL_CASES


def test_guard_detection():
    hw = check_npu_environment(required_cards=2)
    if not hw.has_npu:
        pytest.skip("No NPU detected on runner")
    assert hw.device_count >= 2
    assert hw.is_tp2_ready is True
    assert "Ascend910B2" in hw.device_name


def test_qwen35_shape_registry():
    assert "moe_shared_epilogue_decode" in CANONICAL_CASES
    case = CANONICAL_CASES["moe_shared_epilogue_decode"]
    assert case["shared_out_shape"] == (32, 2048)
    assert case["dtype"] == "bfloat16"


def test_evaluator_l3_numerical_veto():
    case = {
        "baseline_fn": lambda a, b: a * b,
        "input_gen_fn": lambda: (2.0, 3.0),
        "iters": 5,
        "required_cards": 1,
    }
    # Candidate 1: wrong logic
    bad_code = "def optimized_op(a, b): return a + b"
    res_bad = run_candidate_evaluation(bad_code, case)
    assert res_bad["l3_passed"] is False
    assert "L3 VETO" in res_bad["error"]

    # Candidate 2: correct logic
    good_code = "def optimized_op(a, b): return a * b"
    res_good = run_candidate_evaluation(good_code, case)
    assert res_good["l3_passed"] is True
    assert res_good["l0_median_us"] > 0


def test_custody_fingerprint_and_verify(tmp_path):
    sub = tmp_path / "op_test"
    sub.mkdir()
    (sub / "REPORT.md").write_text("report content")
    (sub / "evidence.json").write_text('{"val": 1}')

    from tools.op_agent.custody import write_dir_checksum

    write_dir_checksum(str(sub))

    ok, msg = verify_dir_checksum(str(sub))
    assert ok is True

    # Tamper with a file
    (sub / "REPORT.md").write_text("tampered content")
    ok_tampered, msg_tampered = verify_dir_checksum(str(sub))
    assert ok_tampered is False
    assert "modified files" in msg_tampered


def test_llm_model_configuration(monkeypatch):
    monkeypatch.delenv("OP_AGENT_MODEL_ID", raising=False)
    m_default = get_configured_model()
    assert isinstance(m_default, MockKernelModel)

    monkeypatch.setenv("OP_AGENT_MODEL_ID", "test-model")
    monkeypatch.setenv("OP_AGENT_API_BASE", "http://localhost:8000/v1")
    monkeypatch.setenv("OP_AGENT_API_KEY", "dummy-key")
    m_custom = get_configured_model()
    assert m_custom.model_id == "test-model"
