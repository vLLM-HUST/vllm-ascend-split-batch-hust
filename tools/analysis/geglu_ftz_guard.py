#!/usr/bin/env python3
"""GeGLU 在低精度与次正规数截断（FTZ）下的误差分解仿真与 L3 门禁判据。

纯标准库（``struct`` / ``math``），无 torch、无 numpy、无设备。目的：在没有任何
昇腾硬件的情况下，把"低精度激活融合"这条路的数值后果**拆成互相独立的来源**，
量化各自量级，并写成可执行门禁，将来写 Ascend C kernel 时直接对接。

三条误差来源必须分开算，混在一起就会得出错误结论
------------------------------------------------
1. **输入/输出量化台阶**：把 fp32 结果落成 fp16 / bf16 时，邻居可表示值的间距。
   这是 dtype 自带的物理台阶，**任何实现都无法做得比它更好**。
2. **FTZ 贡献**：硬件把绝对值小于最小正规数的结果强制归零。只在 GELU 输出恰好
   落进次正规区间时出现，量级上界是 ``最小正规数 × |up|``。
3. **逐元素运算次序**：低精度路径把 GELU 中间量与乘积都留在目标精度，会额外
   引入台阶级误差。

只做 1 与 2 的对比就能判定：FTZ 是不是这条链路的主要风险。

``round_fp16`` / ``round_bf16`` 按 IEEE 754 就近取偶（RNE）实现，用 ``struct``
的 binary16 / binary32 位模式，逐位可比；``flush_subnormal`` 模拟 FTZ，符号保留。

用法::

    python tools/analysis/geglu_ftz_guard.py --dtype fp16
    python tools/analysis/geglu_ftz_guard.py --dtype bf16 --up-max 8
"""

from __future__ import annotations

import argparse
import json
import math
import struct
from dataclasses import dataclass

# 各精度最小正规数（2 的幂）；次正规数区间是 (0, min_normal)。
MIN_NORMAL = {
    "fp16": 2.0**-14,
    "bf16": 2.0**-126,
    "fp32": 2.0**-126,
}
# 显式有效位数（不含隐含位），用于算可表示值间距。
SIGNIFICAND_BITS = {"fp16": 10, "bf16": 7, "fp32": 23}


def round_fp16(x: float) -> float:
    """按 RNE 舍入到 binary16；溢出得到 ±inf，与硬件一致。"""
    try:
        return struct.unpack(">e", struct.pack(">e", x))[0]
    except OverflowError:
        return math.copysign(math.inf, x)


def round_bf16(x: float) -> float:
    """按 RNE 把 fp32 的 23 位尾数舍入到 bf16 的 7 位。"""
    if math.isnan(x) or math.isinf(x):
        return x
    if x == 0.0:
        return x
    bits = struct.unpack(">I", struct.pack(">f", x))[0]
    lsb = (bits >> 16) & 1
    bits = (bits + 0x7FFF + lsb) & 0xFFFF0000
    return struct.unpack(">f", struct.pack(">I", bits))[0]


def round_fp32(x: float) -> float:
    return struct.unpack(">f", struct.pack(">f", x))[0]


def round_dtype(x: float, dtype: str) -> float:
    if dtype == "fp16":
        return round_fp16(x)
    if dtype == "bf16":
        return round_bf16(x)
    if dtype == "fp32":
        return round_fp32(x)
    raise ValueError(f"不支持的 dtype: {dtype}")


def flush_subnormal(x: float, dtype: str) -> float:
    """FTZ：绝对值小于该精度最小正规数的有限值归零，符号保留。"""
    if math.isnan(x) or math.isinf(x) or x == 0.0:
        return x
    if abs(x) < MIN_NORMAL[dtype]:
        return math.copysign(0.0, x)
    return x


def low_precision_round(x: float, dtype: str, ftz: bool) -> float:
    """硬件一步到位的"舍入 + 可选 FTZ"。"""
    y = round_dtype(x, dtype)
    return flush_subnormal(y, dtype) if ftz else y


def ulp_at(x: float, dtype: str) -> float:
    """``x`` 所在二进制指数区间内的相邻可表示值间距（次正规区为固定步长）。"""
    subnormal_step = MIN_NORMAL[dtype] * 2.0 ** (-SIGNIFICAND_BITS[dtype])
    if x == 0.0 or not math.isfinite(x):
        return subnormal_step
    if abs(x) < MIN_NORMAL[dtype]:
        return subnormal_step
    exponent = math.frexp(abs(x))[1] - 1
    return math.ldexp(1.0, exponent - SIGNIFICAND_BITS[dtype])


def gelu_tanh(x: float) -> float:
    """tanh 近似的 GELU，双精度计算（参考实现用）。"""
    c = math.sqrt(2.0 / math.pi)
    return 0.5 * x * (1.0 + math.tanh(c * (x + 0.044715 * x**3)))


def geglu_reference(gate: float, up: float) -> float:
    """参考路径：全程双精度，不落低精度、不 FTZ。"""
    return gelu_tanh(gate) * up


def geglu_quantized_only(gate: float, up: float, dtype: str) -> float:
    """只做量化：输入落 dtype、内部 fp32 计算、只最后一跳落 dtype，不 FTZ。"""
    g = round_dtype(gate, dtype)
    u = round_dtype(up, dtype)
    return round_dtype(gelu_tanh(float(g)) * float(u), dtype)


def geglu_high_precision(gate: float, up: float, dtype: str, ftz: bool) -> float:
    """高精度路径：输入落 dtype，内部 fp32 计算，最后一跳落 dtype 并可 FTZ。"""
    g = round_dtype(gate, dtype)
    u = round_dtype(up, dtype)
    acc = gelu_tanh(float(g)) * float(u)
    return low_precision_round(acc, dtype, ftz)


def geglu_full_low_precision(gate: float, up: float, dtype: str, ftz: bool) -> float:
    """低精度路径：GELU 中间量与乘积都留在目标精度，每步都可能被 FTZ 截断。"""
    g = round_dtype(gate, dtype)
    u = round_dtype(up, dtype)
    act = low_precision_round(gelu_tanh(float(g)), dtype, ftz)
    return low_precision_round(float(act) * float(u), dtype, ftz)


def ulp_distance(a: float, b: float, dtype: str) -> float:
    """两个值在该精度下的 ULP 距离；跨零或含非有限值时返回 inf。"""
    if not (math.isfinite(a) and math.isfinite(b)):
        return math.inf if a != b else 0.0
    if a == b:
        return 0.0
    if (a < 0.0) != (b < 0.0):
        return math.inf
    step = ulp_at(max(abs(a), abs(b)), dtype)
    return abs(a - b) / step


@dataclass(frozen=True)
class ErrorStats:
    """逐元素误差统计。``ulp_inf_count`` 记跨零或非有限的不可比元素数。"""

    max_abs: float
    rel_l2: float
    cosine: float
    max_ulp: float
    ulp_inf_count: int
    count: int

    def as_dict(self) -> dict:
        return {
            "max_abs": self.max_abs,
            "rel_l2": self.rel_l2,
            "cosine": self.cosine,
            "max_ulp": None if math.isinf(self.max_ulp) else self.max_ulp,
            "ulp_inf_count": self.ulp_inf_count,
            "count": self.count,
        }


def compare(reference: list, candidate: list, dtype: str) -> ErrorStats:
    """逐元素对比两路输出；余弦相似度按向量内积算。"""
    if len(reference) != len(candidate):
        raise ValueError("两路输出长度不一致")
    if not reference:
        raise ValueError("空输出无法比较")
    max_abs = 0.0
    max_ulp = 0.0
    ulp_inf = 0
    diff_sq = 0.0
    ref_sq = 0.0
    cand_sq = 0.0
    dot = 0.0
    for r, c in zip(reference, candidate, strict=True):
        d = c - r
        if abs(d) > max_abs:
            max_abs = abs(d)
        u = ulp_distance(r, c, dtype)
        if math.isinf(u):
            ulp_inf += 1
        elif u > max_ulp:
            max_ulp = u
        diff_sq += d * d
        ref_sq += r * r
        cand_sq += c * c
        dot += r * c
    rel_l2 = math.sqrt(diff_sq / ref_sq) if ref_sq else math.inf
    denom = math.sqrt(ref_sq) * math.sqrt(cand_sq)
    cosine = dot / denom if denom else (1.0 if diff_sq == 0.0 else 0.0)
    return ErrorStats(
        max_abs=max_abs,
        rel_l2=rel_l2,
        cosine=cosine,
        max_ulp=max_ulp,
        ulp_inf_count=ulp_inf,
        count=len(reference),
    )


def l3_gate(
    stats: ErrorStats,
    cosine_min: float,
    max_abs_max: float,
    intrinsic_floor: float | None = None,
) -> dict:
    """L3 门禁：余弦下限 + 绝对误差上限，两条都要过。

    ``intrinsic_floor`` 给的是该 dtype 在该输出量级下的**量化地板**（半个台阶）。
    若地板本身就高于 ``max_abs_max``，这条绝对门禁任何实现都过不了，必须换成
    dtype 相对判据 —— 输出里用 ``unreachable_by_any_impl`` 明确标出来。
    """
    cosine_ok = stats.cosine >= cosine_min
    abs_ok = stats.max_abs <= max_abs_max
    result = {
        "cosine_min": cosine_min,
        "max_abs_max": max_abs_max,
        "cosine_ok": cosine_ok,
        "max_abs_ok": abs_ok,
        "passed": cosine_ok and abs_ok,
    }
    if intrinsic_floor is not None:
        result["intrinsic_floor"] = intrinsic_floor
        result["unreachable_by_any_impl"] = intrinsic_floor > max_abs_max
        result["excess_over_floor"] = (
            round(stats.max_abs / intrinsic_floor, 6) if intrinsic_floor else None
        )
    return result


def ftz_error_bound(dtype: str, up_abs_max: float) -> dict:
    """FTZ 在激活悬崖处引入的单元素误差上界，与门控取值无关。"""
    return {
        "dtype": dtype,
        "min_normal": MIN_NORMAL[dtype],
        "up_abs_max": up_abs_max,
        "activation_error_bound": MIN_NORMAL[dtype] * up_abs_max,
        "note": "门控使 gelu 输出落进次正规区间时，乘积误差不超过该上界",
    }


def gate_grid(gate_min: float = -30.0, gate_max: float = 3.0, steps: int = 661) -> list:
    """覆盖强负值区（含 GELU 输出为次正规数的窄带）到正值区的门控网格。"""
    if steps < 2:
        raise ValueError("steps 至少为 2")
    span = gate_max - gate_min
    return [gate_min + span * i / (steps - 1) for i in range(steps)]


def _pairs(gates: list, up_values: list) -> list:
    if not up_values:
        raise ValueError("up_values 不能为空")
    return [(g, u) for g in gates for u in up_values]


def decompose(gates: list, up_values: list, dtype: str) -> dict:
    """把误差拆成 量化地板 / FTZ 贡献 / 低精度次序 三块。"""
    pairs = _pairs(gates, up_values)
    ref = [geglu_reference(g, u) for g, u in pairs]
    quant = [geglu_quantized_only(g, u, dtype) for g, u in pairs]
    high_ftz = [geglu_high_precision(g, u, dtype, True) for g, u in pairs]
    high_noftz = [geglu_high_precision(g, u, dtype, False) for g, u in pairs]
    low_ftz = [geglu_full_low_precision(g, u, dtype, True) for g, u in pairs]
    low_noftz = [geglu_full_low_precision(g, u, dtype, False) for g, u in pairs]

    out_abs_max = max(abs(v) for v in ref)
    floor = 0.5 * ulp_at(out_abs_max, dtype)
    return {
        "dtype": dtype,
        "samples": len(pairs),
        "gate_range": [min(gates), max(gates)],
        "up_abs_max": max(abs(u) for u in up_values),
        "output_abs_max": out_abs_max,
        "intrinsic_floor_half_ulp": floor,
        "quantization_only": compare(ref, quant, dtype).as_dict(),
        "high_precision_noftz": compare(ref, high_noftz, dtype).as_dict(),
        "high_precision_ftz": compare(ref, high_ftz, dtype).as_dict(),
        "full_low_precision_noftz": compare(ref, low_noftz, dtype).as_dict(),
        "full_low_precision_ftz": compare(ref, low_ftz, dtype).as_dict(),
        "ftz_contribution_high": compare(high_noftz, high_ftz, dtype).as_dict(),
        "ftz_contribution_low": compare(low_noftz, low_ftz, dtype).as_dict(),
        "low_precision_order_contribution": compare(quant, low_noftz, dtype).as_dict(),
        "gate": {
            "high_precision_ftz": l3_gate(
                compare(ref, high_ftz, dtype), 0.99999, 1e-5, floor
            ),
            "full_low_precision_ftz": l3_gate(
                compare(ref, low_ftz, dtype), 0.99999, 1e-5, floor
            ),
        },
    }


def build_report(dtype: str, up_abs_max: float, gates: list, up_values: list) -> dict:
    result = decompose(gates, up_values, dtype)
    result["ftz_bound"] = ftz_error_bound(dtype, up_abs_max)
    result["ftz_vs_floor_ratio"] = (
        round(
            result["ftz_contribution_high"]["max_abs"]
            / result["intrinsic_floor_half_ulp"],
            6,
        )
        if result["intrinsic_floor_half_ulp"]
        else None
    )
    result["assumption"] = True
    result["note"] = (
        "参考实现用双精度，未在设备上跑过；结论是误差结构与量级，不是实测值。"
        "绝对门禁是否可达由 intrinsic_floor 判定，不要对低精度输出直接引用绝对阈值。"
    )
    return result


def main(argv: list | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="GeGLU 低精度 / FTZ 误差分解仿真与 L3 门禁判据（无设备）"
    )
    parser.add_argument("--dtype", default="fp16", choices=sorted(MIN_NORMAL))
    parser.add_argument("--up-max", type=float, default=4.0)
    args = parser.parse_args(argv)

    up_values = [-args.up_max, -0.5 * args.up_max, 0.0, 0.5 * args.up_max, args.up_max]
    print(
        json.dumps(
            build_report(
                dtype=args.dtype,
                up_abs_max=args.up_max,
                gates=gate_grid(),
                up_values=up_values,
            ),
            ensure_ascii=False,
            indent=2,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
