"""`tools/analysis/ascend910b_static_audit.py` 的算术护栏。

这些断言钉的是**由维度直接推出的字节账与可行性判据**，不含任何设备实测：
账目必须逐阶段可复算、单调性必须成立、32 字节判据必须在边界处翻转、
UB 预算必须与级数和分块线性一致。任何一条被改坏，说明静态审计的口径漂了。
"""

from __future__ import annotations

import importlib.util
import json
from pathlib import Path

import pytest

PATH = (
    Path(__file__).resolve().parents[1]
    / "tools"
    / "analysis"
    / "ascend910b_static_audit.py"
)
spec = importlib.util.spec_from_file_location("ascend910b_static_audit", PATH)
audit = importlib.util.module_from_spec(spec)
spec.loader.exec_module(audit)

HIDDEN = 2048
INTER = 256


def test_hbm_stage_ledger_is_hand_checkable():
    """m=10, k=256, n=2048, bf16：每个阶段的字节必须能口算出来。"""
    r = audit.fused_epilogue_hbm(10, INTER, HIDDEN, "bf16")
    tile = 10 * HIDDEN * 2
    assert r["tile_bytes"] == tile == 40960
    assert r["unfused_activation_bytes"] == 6 * tile
    assert r["fused_activation_bytes"] == 2 * tile
    assert r["saved_activation_bytes"] == 4 * tile
    assert r["weights_bytes_per_step"] == INTER * HIDDEN * 2
    total = 6 * tile + INTER * HIDDEN * 2
    assert r["unfused_total_bytes"] == total
    assert r["saved_fraction_including_weights"] == pytest.approx(
        4 * tile / total, abs=1e-6
    )
    assert r["saved_fraction_of_activation_only"] == pytest.approx(4 / 6, abs=1e-6)


def test_saved_bytes_scale_linearly_with_m():
    a = audit.fused_epilogue_hbm(8, INTER, HIDDEN, "bf16")
    b = audit.fused_epilogue_hbm(16, INTER, HIDDEN, "bf16")
    assert b["saved_activation_bytes"] == 2 * a["saved_activation_bytes"]


def test_weight_traffic_dilutes_the_saving_at_small_m():
    """小 M 时中间激活的往返只占总流量的很小一部分，这是必须如实报告的结论。"""
    small = audit.fused_epilogue_hbm(1, INTER, HIDDEN, "bf16")
    large = audit.fused_epilogue_hbm(512, INTER, HIDDEN, "bf16")
    assert small["saved_fraction_including_weights"] < 0.02
    assert (
        large["saved_fraction_including_weights"]
        > small["saved_fraction_including_weights"] * 20
    )
    assert small["saved_fraction_of_activation_only"] == pytest.approx(
        large["saved_fraction_of_activation_only"]
    )


def test_arithmetic_intensity_matches_definition():
    r = audit.fused_epilogue_hbm(32, INTER, HIDDEN, "bf16")
    assert r["gemm_flops"] == 2 * 32 * INTER * HIDDEN
    assert r["fused_arithmetic_intensity_flop_per_byte"] == pytest.approx(
        r["gemm_flops"] / r["fused_total_bytes"], rel=1e-6
    )


@pytest.mark.parametrize(
    "peak_tflops, expected",
    [(0.5, "compute"), (200.0, "memory")],
)
def test_roofline_verdict_follows_machine_balance(peak_tflops, expected):
    r = audit.fused_epilogue_hbm(1, INTER, HIDDEN, "bf16")
    verdict = audit.roofline_verdict(r, peak_tflops=peak_tflops, peak_gb_s=800.0)
    assert verdict["bound"] == expected
    assert verdict["assumption"]["peak_tflops"] == peak_tflops


def test_roofline_rejects_non_positive_peaks():
    r = audit.fused_epilogue_hbm(1, INTER, HIDDEN, "bf16")
    with pytest.raises(ValueError):
        audit.roofline_verdict(r, 0.0, 800.0)
    with pytest.raises(ValueError):
        audit.roofline_verdict(r, 200.0, 0.0)


def test_alignment_boundary_flips_at_32_bytes():
    """2 字节元素 16 个正好 32B；17 个越界；4 字节元素 8 个正好 32B。"""
    profile = {"hidden": 16, "head_dim": 17, "num_kv_heads": 1, "intermediate": 8}
    rows = {r["slice"]: r for r in audit.alignment_checks(profile, "bf16")}
    assert rows["hidden_row"]["row_bytes"] == 32
    assert rows["hidden_row"]["aligned"] is True
    assert rows["kv_head_row"]["row_bytes"] == 34
    assert rows["kv_head_row"]["aligned"] is False
    assert rows["kv_head_row"]["remedy"].startswith("use DataCopyPad")

    rows = {r["slice"]: r for r in audit.alignment_checks(profile, "fp32")}
    assert rows["shared_expert_inter_row"]["row_bytes"] == 32
    assert rows["shared_expert_inter_row"]["aligned"] is True


def test_alignment_checks_start_offset_in_bytes():
    profile = {"hidden": 16, "head_dim": 16, "num_kv_heads": 1, "intermediate": 16}
    ok = {
        r["slice"]: r
        for r in audit.alignment_checks(
            profile, "bf16", offsets={"hidden_row": 16, "kv_head_row": 1}
        )
    }
    assert ok["hidden_row"]["start_bytes"] == 32
    assert ok["hidden_row"]["aligned"] is True
    assert ok["kv_head_row"]["start_bytes"] == 2
    assert ok["kv_head_row"]["aligned"] is False


def test_default_profile_slices_are_aligned_at_both_dtypes():
    """Qwen3.5-35B-A3B(TP=2) 的切片在 bf16/fp32 下全部落在 32B 网格上。"""
    for dtype in ("bf16", "fp32"):
        rows = audit.alignment_checks(audit.PROFILES["qwen35-a3b-tp2"], dtype)
        assert rows, "审计结果不能为空"
        assert all(r["aligned"] for r in rows), [
            (r["slice"], r["row_bytes"]) for r in rows if not r["aligned"]
        ]


def test_extra_slice_is_audited_too():
    rows = audit.alignment_checks(
        audit.PROFILES["qwen35-a3b-tp2"], "bf16", None, {"odd_slice": 15}
    )
    odd = [r for r in rows if r["slice"] == "odd_slice"]
    assert len(odd) == 1
    assert odd[0]["row_bytes"] == 30
    assert odd[0]["aligned"] is False


def test_ub_budget_arithmetic_and_linearity():
    one = audit.ub_budget(
        tile_m=16, tile_n=INTER, dtype="bf16", operands=2, stages=1, ub_bytes=131072
    )
    assert one["bytes_per_stage"] == 16 * INTER * 2 * 2 == 16384
    assert one["required_bytes"] == 16384
    assert one["fits"] is True

    three = audit.ub_budget(
        tile_m=16, tile_n=INTER, dtype="bf16", operands=2, stages=3, ub_bytes=131072
    )
    assert three["required_bytes"] == 3 * 16384
    assert three["headroom_bytes"] == 131072 - 3 * 16384
    assert three["fill_ratio"] == pytest.approx(3 * 16384 / 131072, abs=1e-4)


def test_ub_budget_reports_overflow():
    r = audit.ub_budget(
        tile_m=64, tile_n=HIDDEN, dtype="fp32", operands=2, stages=4, ub_bytes=131072
    )
    assert r["required_bytes"] > 131072
    assert r["fits"] is False
    assert r["headroom_bytes"] < 0


def test_max_stages_and_max_tile_m():
    assert (
        audit.max_stages_that_fit(16, INTER, "bf16", 2, ub_bytes=131072, extra_bytes=0)
        == 8
    )
    assert audit.max_stages_that_fit(16, INTER, "bf16", 2, 16384) == 1
    assert audit.max_stages_that_fit(16, INTER, "bf16", 2, 1000) == 0
    # 131072 / (256 × 2 × 2 × 3) = 42.67 行，向下取 16 的整倍数。
    assert audit.max_tile_m_that_fit(INTER, "bf16", 2, 3, 131072) == 32
    assert audit.max_tile_m_that_fit(INTER, "bf16", 2, 3, 1024) == 0


@pytest.mark.parametrize(
    "m, advice, waste",
    [
        (1, "vector_gemv_fallback", 15 / 16),
        (4, "vector_gemv_fallback", 3 / 4),
        (8, "cube_gemm_with_token_packing", 1 / 2),
        (10, "cube_gemm_with_token_packing", 6 / 16),
        (16, "cube_gemm", 0.0),
        (17, "cube_gemm", 15 / 32),
    ],
)
def test_small_m_padding_waste(m, advice, waste):
    r = audit.small_m_report(m)
    assert r["dispatch_advice"] == advice
    assert r["wasted_mac_fraction"] == pytest.approx(waste, abs=1e-4)


def test_invalid_inputs_raise():
    with pytest.raises(ValueError):
        audit.fused_epilogue_hbm(0, INTER, HIDDEN, "bf16")
    with pytest.raises(ValueError):
        audit.ub_budget(0, INTER, "bf16", 2, 3, 131072)
    with pytest.raises(ValueError):
        audit.small_m_report(0)


def test_build_report_is_deterministic_and_complete():
    kwargs = dict(
        profile_name="qwen35-a3b-tp2",
        m_list=[1, 10],
        dtype="bf16",
        ub_kib_list=[128],
        stages_list=[2, 3, 4],
        operands=2,
        peak_tflops=200.0,
        peak_gb_s=800.0,
    )
    first = audit.build_report(**kwargs)
    second = audit.build_report(**kwargs)
    assert first == second
    assert [row["hbm"]["m"] for row in first["hbm_fusion"]] == [1, 10]
    assert [s["stages"] for s in first["ub_budget"][0]["sweep"]] == [2, 3, 4]
    assert first["assumption"] is True


def test_main_emits_parseable_json(capsys):
    assert audit.main(["--m", "1", "32", "--ub-kib", "192"]) == 0
    payload = json.loads(capsys.readouterr().out)
    assert payload["profile"] == "qwen35-a3b-tp2"
    assert payload["align_bytes"] == audit.ALIGN_BYTES
    assert len(payload["hbm_fusion"]) == 2
    assert payload["ub_budget"][0]["ub_kib"] == 192


def test_cli_slice_and_offset_flags_reach_the_audit(capsys):
    assert audit.main(["--m", "8", "--slice", "odd=15", "--offset", "odd=1"]) == 0
    payload = json.loads(capsys.readouterr().out)
    odd = [r for r in payload["alignment"] if r["slice"] == "odd"]
    assert len(odd) == 1
    assert odd[0]["aligned"] is False
