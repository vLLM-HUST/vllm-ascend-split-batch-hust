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


def test_evaluate_kernel_short_circuit():
    import json

    import tools.op_agent.optimizer_loop as loop
    from tools.op_agent.optimizer_loop import EvaluateKernelTool

    loop.CURRENT_TEST_CASE = {
        "baseline_fn": lambda a, b: a * b,
        "input_gen_fn": lambda: (2.0, 3.0),
        "iters": 5,
        "required_cards": 1,
        "baseline_latency": 100.0,
    }
    tool = EvaluateKernelTool()

    # Case 1: L3 fails -> short-circuit without L0
    bad_code = "def optimized_op(a, b): return a + b"
    res_bad = json.loads(tool.forward(bad_code))
    assert res_bad["l3_passed"] is False
    assert "error" in res_bad
    assert "l0_median_us" not in res_bad

    # Case 2: L3 passes -> measures L0 directly in single call
    good_code = "def optimized_op(a, b): return a * b"
    res_good = json.loads(tool.forward(kernel_code=good_code))
    assert res_good["l3_passed"] is True
    assert res_good["l0_median_us"] > 0
    assert "speedup" in res_good

    # Case 3: Pass-by-reference via WriteCandidateFileTool
    from tools.op_agent.optimizer_loop import WriteCandidateFileTool

    writer = WriteCandidateFileTool()
    cand_path = "ops/tmp_test_cand.py"
    try:
        write_res = json.loads(writer.forward(cand_path, good_code))
        assert write_res["status"] == "saved"
        res_file = json.loads(tool.forward(file_path=cand_path))
        assert res_file["l3_passed"] is True
        assert res_file["l0_median_us"] > 0
    finally:
        import os

        if os.path.exists(cand_path):
            os.remove(cand_path)


def test_kb_in_memory_caching():
    import tempfile
    import time

    from tools.op_agent.kb import query_ascend_kb, read_kb_card

    # Query caching
    res1 = query_ascend_kb("triton", limit=2)
    t0 = time.perf_counter()
    res2 = query_ascend_kb("triton", limit=2)
    t_elapsed_ms = (time.perf_counter() - t0) * 1000
    assert res1 == res2
    assert t_elapsed_ms < 50.0  # sub-50ms cache response

    # Card condensed reading
    with tempfile.NamedTemporaryFile("w", suffix=".md", delete=False) as f:
        f.write("---\nauthor: me\ndate: 2026\n---\n\n## Summary\nUse tiling.")
        tmp_name = f.name
    try:
        content = read_kb_card(tmp_name)
        assert "author:" not in content
        assert "Use tiling." in content
    finally:
        import os

        if os.path.exists(tmp_name):
            os.remove(tmp_name)


def test_case_registry_contract():
    from tools.op_agent.cases import CASE_REGISTRY, get_case

    assert "gdn_qkv_norm" in CASE_REGISTRY
    assert "gdn_input_norm_decode" in CASE_REGISTRY
    assert "gdn_norm_gate" in CASE_REGISTRY
    assert "gdn_split_ba" in CASE_REGISTRY
    assert "moe_shared_epilogue_decode" in CASE_REGISTRY

    for name in CASE_REGISTRY:
        case = get_case(name)
        tc = case.build_test_case(is_tp2_ready=False)
        assert tc["name"] == name
        assert isinstance(tc["callsite"], str)
        assert isinstance(tc["shape_facts"], dict)
        assert callable(tc["input_gen_fn"])
        assert callable(tc["baseline_fn"])
        assert isinstance(tc["iters"], int)
        assert isinstance(tc["required_cards"], int)

        # Baseline execution on CPU mock data
        inputs = tc["input_gen_fn"]()
        out = tc["baseline_fn"](*inputs)
        assert out is not None


def test_gdn_qkv_norm_case_cpu():
    from tools.op_agent.cases import get_case

    case = get_case("gdn_qkv_norm")
    tc = case.build_test_case(is_tp2_ready=False)
    inputs = tc["input_gen_fn"]()
    assert len(inputs) == 1
    assert inputs[0].shape == (16, 4096)

    out = tc["baseline_fn"](*inputs)
    assert isinstance(out, tuple)
    assert len(out) == 3
    q_norm, k_norm, v = out
    assert q_norm.shape == (1, 16, 8, 128)
    assert k_norm.shape == (1, 16, 8, 128)
    assert v.shape == (1, 16, 16, 128)
