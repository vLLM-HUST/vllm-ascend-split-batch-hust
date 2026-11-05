from __future__ import annotations

import os
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

# The package CI runs without numpy / torch / smolagents installed.
# Mark the whole module to skip if optional dependencies are not present.
pytest.importorskip("numpy", reason="op agent tests need numpy")
pytest.importorskip("smolagents", reason="op agent tests need smolagents")

from tools.op_agent.custody import verify_dir_checksum  # noqa: E402
from tools.op_agent.evaluator import run_candidate_evaluation  # noqa: E402
from tools.op_agent.guard import check_npu_environment  # noqa: E402
from tools.op_agent.optimizer_loop import (  # noqa: E402
    MockKernelModel,
    get_configured_model,
)
from tools.op_agent.qwen35_shapes import CANONICAL_CASES  # noqa: E402


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
    assert "gdn_output_norm_fusion" in CASE_REGISTRY
    assert "gdn_split_ba" in CASE_REGISTRY
    assert "moe_shared_epilogue_decode" in CASE_REGISTRY

    for name in CASE_REGISTRY:
        case = get_case(name)
        tc = case.build_test_case(is_tp2_ready=False)
        assert tc["name"] == name
        assert isinstance(tc["callsite"], str)
        assert isinstance(tc["shape_facts"], dict)
        assert isinstance(tc["workload_spec"], dict)
        assert "primary_workload" in tc["workload_spec"]
        assert "stress_workload" in tc["workload_spec"]
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


def test_gdn_output_norm_fusion_strided_view_and_recompile_veto():
    from tools.op_agent.cases import get_case

    case = get_case("gdn_output_norm_fusion")
    tc = case.build_test_case(is_tp2_ready=False)
    tc["iters"] = 4

    # Verify non-contiguous column-sliced z layout on CPU numpy array
    core_out, z, weight = tc["bucket_input_gen_fn"](8)
    assert core_out.shape == (8, 16, 128)
    assert z.shape == (8, 16, 128)
    assert z.flags["C_CONTIGUOUS"] is False
    out = tc["baseline_fn"](core_out, z, weight)
    assert out.shape == (8, 2048)

    # Candidate that passes multi-bucket L3 and dynamic shape probes
    good_res = run_candidate_evaluation(case.get_baseline_code(), tc)
    assert good_res["l3_passed"] is True
    assert set(good_res["bucket_latencies_us"].keys()) == {1, 4, 8, 16, 32}

    # Candidate simulating per-shape Triton JIT recompilation stall on T=97
    stalling_code = (
        "import time\n"
        + case.get_baseline_code()
        + "\n_orig = optimized_op\n"
        + "def optimized_op(core_out, z, weight):\n"
        + "    if z.shape[0] == 97:\n"
        + "        time.sleep(0.06)\n"
        + "    return _orig(core_out, z, weight)\n"
    )
    tc_strict = dict(tc, recompile_budget_ms=20.0)
    veto_res = run_candidate_evaluation(stalling_code, tc_strict)
    assert veto_res["l3_passed"] is False
    assert veto_res.get("recompile_veto") is True
    assert "RECOMPILE VETO" in veto_res["error"]
    assert os.path.exists("ops") is True


def test_pluggable_campaign_discover_and_guards(tmp_path):
    from tools.op_agent.campaign import discover_hotspot_batch

    stat_csv = tmp_path / "op_statistic.csv"
    stat_csv.write_text(
        "OP Type,Count,Total Time(us),Avg Time(us)\ngdn_split_ba_op,100,10.0,0.0001\n",
        encoding="utf-8",
    )
    manifest_file = tmp_path / "batch.json"
    manifest = discover_hotspot_batch(
        output_manifest_path=str(manifest_file),
        case_names=["moe_shared_epilogue_decode", "gdn_split_ba"],
        op_statistic_csv=str(stat_csv),
        min_graph_baseline_us=0.0,
        iters=4,
    )
    assert manifest["schema"] == "op-agent-campaign-batch/v1"
    assert len(manifest["candidates"]) == 2
    by_name = {c["case_name"]: c for c in manifest["candidates"]}
    # gdn_split_ba has a tiny fake CSV avg (0.0001 us), so the CSV cross-check guard
    # flags it as `pending_review`
    assert by_name["gdn_split_ba"]["status"] == "pending_review"


def test_moe_gating_routing_case_contract():
    from tools.op_agent.cases import get_case

    case = get_case("moe_gating_routing")
    tc = case.build_test_case(is_tp2_ready=False)
    tc["iters"] = 4

    # Verify input gen shapes on CPU
    router_logits, hidden_states = tc["bucket_input_gen_fn"](16)
    assert router_logits.shape == (16, 256)
    assert hidden_states.shape == (16, 2048)

    topk_weights, expanded_x, expanded_row_idx, expert_tokens = tc["baseline_fn"](
        router_logits, hidden_states
    )
    assert topk_weights.shape == (16, 8)
    assert expanded_x.shape == (128, 2048)
    assert expanded_row_idx.shape == (128,)
    assert expert_tokens.shape == (256,)
    assert expert_tokens.sum() == 128

    res = run_candidate_evaluation(case.get_baseline_code(), tc)
    assert res["l3_passed"] is True
    assert set(res["bucket_latencies_us"].keys()) == {1, 4, 8, 16, 32}
