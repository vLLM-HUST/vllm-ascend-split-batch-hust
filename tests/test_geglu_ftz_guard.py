"""`tools/analysis/geglu_ftz_guard.py` 的精度护栏与 FTZ 结论护栏。

钉三件事：

1. 低精度舍入（RNE）与次正规数截断（FTZ）的语义实现正确 —— 用可手算的边界值；
2. 误差拆分口径稳定 —— 量化地板、FTZ 贡献、低精度运算次序三块互不串味；
3. 结论本身可复现 —— FTZ 在 GeGLU 上的贡献远小于目标精度的量化台阶，而
   `max_abs <= 1e-5` 这类绝对门禁对 fp16/bf16 输出任何实现都过不了。
"""

from __future__ import annotations

import importlib.util
import json
import math
import sys
from pathlib import Path

import pytest

PATH = Path(__file__).resolve().parents[1] / "tools" / "analysis" / "geglu_ftz_guard.py"
spec = importlib.util.spec_from_file_location("geglu_ftz_guard", PATH)
guard = importlib.util.module_from_spec(spec)
# dataclass 解析注解时需要模块已注册。
sys.modules["geglu_ftz_guard"] = guard
spec.loader.exec_module(guard)

FP16_MIN_NORMAL = 2.0**-14
FP16_SUBNORMAL_STEP = 2.0**-24
UP_VALUES = [-4.0, -2.0, 0.0, 2.0, 4.0]


def test_fp16_rounding_is_round_to_nearest_even():
    assert guard.round_fp16(1.0 + 2.0**-11) == 1.0  # 平局落到偶数尾数
    assert guard.round_fp16(1.0 + 3 * 2.0**-11) == 1.0 + 2.0**-9
    assert guard.round_fp16(1.0 + 2.0**-10) == 1.0 + 2.0**-10
    assert guard.round_fp16(FP16_SUBNORMAL_STEP) == FP16_SUBNORMAL_STEP
    assert guard.round_fp16(FP16_SUBNORMAL_STEP / 2) == 0.0
    assert guard.round_fp16(1e6) == math.inf
    assert guard.round_fp16(-1e6) == -math.inf


def test_bf16_rounding_is_round_to_nearest_even():
    assert guard.round_bf16(1.0 + 2.0**-9) == 1.0
    assert guard.round_bf16(1.0 + 3 * 2.0**-9) == 1.0 + 2.0**-7
    assert guard.round_bf16(2.0**-40) == 2.0**-40  # bf16 次正规数区间极深
    assert math.isnan(guard.round_bf16(math.nan))
    assert guard.round_bf16(math.inf) == math.inf


@pytest.mark.parametrize("dtype", ["fp16", "bf16", "fp32"])
def test_rounding_is_idempotent(dtype):
    for value in (0.0, 1.0, -1.5, 1e-7, 123.456, -0.841):
        once = guard.round_dtype(value, dtype)
        assert guard.round_dtype(once, dtype) == once


def test_flush_subnormal_keeps_normal_and_zeroes_subnormal():
    assert guard.flush_subnormal(FP16_MIN_NORMAL, "fp16") == FP16_MIN_NORMAL
    assert guard.flush_subnormal(FP16_MIN_NORMAL * 0.99, "fp16") == 0.0
    assert guard.flush_subnormal(-FP16_SUBNORMAL_STEP, "fp16") == 0.0
    assert math.copysign(1.0, guard.flush_subnormal(-FP16_SUBNORMAL_STEP, "fp16")) < 0
    assert guard.flush_subnormal(0.0, "fp16") == 0.0
    assert math.isinf(guard.flush_subnormal(math.inf, "fp16"))
    # fp32 的次正规区间在 1e-38 以下，普通量级一律不受影响。
    assert guard.flush_subnormal(1e-30, "fp32") == 1e-30
    assert guard.flush_subnormal(1e-40, "fp32") == 0.0


def test_ulp_at_matches_exponent_buckets():
    assert guard.ulp_at(1.0, "fp16") == 2.0**-10
    assert guard.ulp_at(2.0, "fp16") == 2.0**-9
    assert guard.ulp_at(0.5, "fp16") == 2.0**-11
    assert guard.ulp_at(1.0, "bf16") == 2.0**-7
    assert guard.ulp_at(2.0**-20, "fp16") == FP16_SUBNORMAL_STEP


def test_ulp_distance_semantics():
    assert guard.ulp_distance(1.0, 1.0, "fp16") == 0.0
    assert guard.ulp_distance(1.0, 1.0 + 2.0**-10, "fp16") == pytest.approx(1.0)
    assert guard.ulp_distance(0.0, FP16_SUBNORMAL_STEP, "fp16") == pytest.approx(1.0)
    assert guard.ulp_distance(-1.0, 1.0, "fp16") == math.inf


def test_gelu_tanh_reference_values():
    assert guard.gelu_tanh(0.0) == 0.0
    assert guard.gelu_tanh(1.0) == pytest.approx(0.841192, abs=1e-6)
    assert abs(guard.gelu_tanh(-30.0)) < 1e-10


def test_quantized_only_equals_high_precision_without_ftz():
    """只量化 与 高精度不 FTZ 是同一条计算，必须逐位一致。"""
    for gate in (-6.0, -1.0, 0.0, 2.5):
        for up in UP_VALUES:
            assert guard.geglu_quantized_only(gate, up, "fp16") == (
                guard.geglu_high_precision(gate, up, "fp16", False)
            )


def test_gate_grid_covers_the_activation_cliff():
    """网格必须覆盖"gelu 输出落进 fp16 次正规区间"的窄带，否则结论无意义。"""
    band = [
        g for g in guard.gate_grid() if 0.0 < abs(guard.gelu_tanh(g)) < FP16_MIN_NORMAL
    ]
    assert band, "网格没覆盖次正规输出区间"
    assert all(-8.0 < g < 0.0 for g in band)
    # 落在次正规区间、且在该精度里确实能表示成非零次正规数的那一段：
    # 不开 FTZ 保留原值，开 FTZ 一律归零。
    representable = [
        g
        for g in band
        if guard.low_precision_round(guard.gelu_tanh(g), "fp16", False) != 0.0
    ]
    assert len(representable) >= 10
    for g in representable:
        assert guard.low_precision_round(guard.gelu_tanh(g), "fp16", True) == 0.0


def test_ftz_contribution_stays_under_the_theoretical_bound():
    """FTZ 带来的单元素误差不得超过 最小正规数 × |up| 上界。"""
    report = guard.decompose(guard.gate_grid(), UP_VALUES, "fp16")
    bound = guard.ftz_error_bound("fp16", 4.0)["activation_error_bound"]
    assert bound == FP16_MIN_NORMAL * 4.0
    assert report["ftz_contribution_high"]["max_abs"] <= bound
    assert report["ftz_contribution_low"]["max_abs"] <= bound


def test_ftz_is_far_smaller_than_the_dtype_quantization_floor():
    """结论 1：fp16 下 FTZ 贡献占不到量化台阶的 5%，不是主因。"""
    report = guard.build_report("fp16", 4.0, guard.gate_grid(), UP_VALUES)
    assert report["intrinsic_floor_half_ulp"] == 0.5 * guard.ulp_at(
        report["output_abs_max"], "fp16"
    )
    assert report["ftz_vs_floor_ratio"] < 0.05


def test_bf16_ftz_is_inert_over_the_whole_grid():
    """结论 2：bf16 的次正规区间在 1e-38 以下，GeGLU 全网格都碰不到。"""
    report = guard.build_report("bf16", 4.0, guard.gate_grid(), UP_VALUES)
    assert report["ftz_contribution_high"]["max_abs"] == 0.0
    assert report["ftz_contribution_low"]["max_abs"] == 0.0
    assert report["ftz_vs_floor_ratio"] == 0.0


def test_low_precision_evaluation_order_is_negligible():
    report = guard.decompose(guard.gate_grid(), UP_VALUES, "fp16")
    assert report["low_precision_order_contribution"]["max_abs"] < 1e-6


@pytest.mark.parametrize("dtype", ["fp16", "bf16"])
def test_absolute_gate_below_the_dtype_floor_is_unreachable(dtype):
    """结论 3：`max_abs <= 1e-5` 对低精度输出任何实现都过不了，必须换相对判据。"""
    report = guard.build_report(dtype, 4.0, guard.gate_grid(), UP_VALUES)
    gate = report["gate"]["high_precision_ftz"]
    assert gate["unreachable_by_any_impl"] is True
    assert gate["max_abs_ok"] is False
    assert gate["passed"] is False


def test_gate_passes_once_the_threshold_is_relative_to_the_floor():
    report = guard.decompose(guard.gate_grid(), UP_VALUES, "fp16")
    floor = report["intrinsic_floor_half_ulp"]
    stats = guard.compare(
        [guard.geglu_reference(g, u) for g in guard.gate_grid() for u in UP_VALUES],
        [
            guard.geglu_high_precision(g, u, "fp16", True)
            for g in guard.gate_grid()
            for u in UP_VALUES
        ],
        "fp16",
    )
    relaxed = guard.l3_gate(
        stats, cosine_min=0.99999, max_abs_max=2 * floor, intrinsic_floor=floor
    )
    assert relaxed["unreachable_by_any_impl"] is False
    assert relaxed["passed"] is True
    assert relaxed["excess_over_floor"] <= 2.0


@pytest.mark.parametrize("dtype", ["fp16", "bf16"])
def test_cosine_criterion_holds_for_both_paths(dtype):
    report = guard.decompose(guard.gate_grid(), UP_VALUES, dtype)
    assert report["high_precision_ftz"]["cosine"] >= 0.99999
    assert report["full_low_precision_ftz"]["cosine"] >= 0.99999


def test_decompose_is_deterministic_and_complete():
    first = guard.decompose(guard.gate_grid(steps=101), UP_VALUES, "fp16")
    second = guard.decompose(guard.gate_grid(steps=101), UP_VALUES, "fp16")
    assert first == second
    assert first["samples"] == 101 * len(UP_VALUES)
    for key in (
        "quantization_only",
        "high_precision_noftz",
        "high_precision_ftz",
        "full_low_precision_noftz",
        "full_low_precision_ftz",
        "ftz_contribution_high",
        "low_precision_order_contribution",
    ):
        assert first[key]["count"] == first["samples"]


def test_compare_rejects_mismatched_or_empty_inputs():
    with pytest.raises(ValueError):
        guard.compare([1.0], [1.0, 2.0], "fp16")
    with pytest.raises(ValueError):
        guard.compare([], [], "fp16")
    with pytest.raises(ValueError):
        guard.gate_grid(steps=1)
    with pytest.raises(ValueError):
        guard.round_dtype(1.0, "int8")


def test_main_emits_parseable_json(capsys):
    assert guard.main(["--dtype", "fp16", "--up-max", "4"]) == 0
    payload = json.loads(capsys.readouterr().out)
    assert payload["dtype"] == "fp16"
    assert payload["assumption"] is True
    assert payload["ftz_bound"]["activation_error_bound"] == pytest.approx(
        FP16_MIN_NORMAL * 4.0
    )
    assert payload["gate"]["high_precision_ftz"]["unreachable_by_any_impl"] is True


def test_rounding_agrees_with_torch_casts():
    """与 torch 的 fp16/bf16 落位逐位对照（torch 不在时跳过）。"""
    torch = pytest.importorskip("torch", reason="cross-check needs torch")
    values = [
        -3.0,
        -1.0,
        -1e-7,
        -5.9604645e-08,
        0.0,
        5.9604645e-08,
        1e-7,
        0.1,
        0.841192,
        1.0009765625,
        65504.0,
    ]
    for dtype, cast in (
        ("fp16", torch.float16),
        ("bf16", torch.bfloat16),
    ):
        tensor = torch.tensor(values, dtype=torch.float32).to(cast).to(torch.float32)
        got = [float(v) for v in tensor.tolist()]
        want = [guard.round_dtype(v, dtype) for v in values]
        assert got == want, dtype
