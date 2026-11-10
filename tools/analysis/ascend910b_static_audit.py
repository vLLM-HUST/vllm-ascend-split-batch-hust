#!/usr/bin/env python3
"""Ascend 910B 算子方向离线静态审计（纯 CPU、纯标准库、不读设备、不写文件）。

三件事都是**由维度事实直接推出的算术**，不依赖任何设备实测：

1. ``fused_epilogue_hbm``：Down GEMM -> Residual Add -> Norm 链路在"分算子各自
   落 HBM"与"片上闭环"两种写法下的搬运字节数。逐阶段列账（权重 / 残差 / 中间
   激活 / 输出），比值随 M 变化，不与实测数字混算。
2. ``alignment_checks``：按 32 字节物理粒度（16 个 2 字节元素 / 8 个 4 字节元素）
   审计每个切片行的连续段字节数与起始元素偏移。
3. ``ub_budget``：在给定片上 SRAM 上限下，多级乒乓（2/3/4 级）能否装下一个
   Vector 处理分块，以及可装下的最大级数 / 最大分块。

所有硬件常数（带宽、算力、UB 容量）都是**输入参数**，默认值只作为比较基准，
输出里一律带 ``"assumption": true``，禁止把默认值当作实测结论引用。

用法::

    python tools/analysis/ascend910b_static_audit.py --m 1 8 10 32 512
    python tools/analysis/ascend910b_static_audit.py --dtype fp32 --ub-kib 256
"""

from __future__ import annotations

import argparse
import json
import math

ALIGN_BYTES = 32
CUBE_ATOM_M = 16
DTYPE_BYTES = {"fp16": 2, "bf16": 2, "fp32": 4}

# 事实来源：Qwen3.5-35B-A3B (TP=2) 的层结构常量，与 ops/*/REPORT.md、
# host-proposals/GDN-qkv-norm/proposal_meta.json、tools/op_agent/qwen35_shapes.py
# 中登记的值保持一致；改模型时改这里，不要在函数内部硬编码。
PROFILES = {
    "qwen35-a3b-tp2": {
        "hidden": 2048,  # 4096 / TP2
        "intermediate": 256,  # shared expert 512 / TP2
        "head_dim": 128,
        "num_kv_heads": 1,  # k/v 合并后每秩 256 = 2 × 128
        "layers": 40,
    },
}


def _row_bytes(elems: int, dtype: str) -> int:
    return elems * DTYPE_BYTES[dtype]


def fused_epilogue_hbm(
    m: int,
    k: int,
    n: int,
    dtype: str,
    weights_per_step: bool = True,
) -> dict:
    """Down GEMM -> Residual -> Norm 两种写法的搬运字节账。

    ``m`` 行、``k`` 是归约维（中间维）、``n`` 是输出宽（hidden）。
    账目只列这一条链路自己的流量，权重是否计入由 ``weights_per_step`` 控制：
    权重每步都要读一次，若不与激活流量分开列，小 M 下的比值会被权重量稀释。
    """
    if m <= 0 or k <= 0 or n <= 0:
        raise ValueError("m/k/n 必须为正整数")
    b = DTYPE_BYTES[dtype]
    tile = m * n * b
    weights = k * n * b if weights_per_step else 0

    # 分算子写法：每跳都经 HBM。
    unfused_stages = [
        {"stage": "down_gemm_write_y", "bytes": tile, "dir": "write"},
        {"stage": "residual_read_y", "bytes": tile, "dir": "read"},
        {"stage": "residual_read_residual", "bytes": tile, "dir": "read"},
        {"stage": "residual_write_y_plus_r", "bytes": tile, "dir": "write"},
        {"stage": "norm_read_y_plus_r", "bytes": tile, "dir": "read"},
        {"stage": "norm_write_out", "bytes": tile, "dir": "write"},
    ]
    # 片上闭环：只有叶子输入与最终输出跨 HBM。
    fused_stages = [
        {"stage": "read_residual", "bytes": tile, "dir": "read"},
        {"stage": "write_norm_out", "bytes": tile, "dir": "write"},
    ]
    unfused = sum(s["bytes"] for s in unfused_stages)
    fused = sum(s["bytes"] for s in fused_stages)
    saved = unfused - fused
    unfused_total = unfused + weights
    fused_total = fused + weights

    flops = 2 * m * k * n
    return {
        "m": m,
        "k": k,
        "n": n,
        "dtype": dtype,
        "tile_bytes": tile,
        "weights_bytes_per_step": weights,
        "unfused_stages": unfused_stages,
        "fused_stages": fused_stages,
        "unfused_activation_bytes": unfused,
        "fused_activation_bytes": fused,
        "saved_activation_bytes": saved,
        "saved_fraction_of_activation_only": round(saved / unfused, 6),
        "saved_fraction_including_weights": round(
            saved / unfused_total if unfused_total else 0.0, 6
        ),
        "unfused_total_bytes": unfused_total,
        "fused_total_bytes": fused_total,
        "gemm_flops": flops,
        "fused_arithmetic_intensity_flop_per_byte": round(flops / fused_total, 3),
    }


def roofline_verdict(
    report: dict,
    peak_tflops: float,
    peak_gb_s: float,
) -> dict:
    """按机器平衡点判断该链路在给定 M 下是算力受限还是搬运受限。

    ``peak_tflops`` / ``peak_gb_s`` 由调用方给，默认值仅作比较基准。
    """
    if peak_tflops <= 0 or peak_gb_s <= 0:
        raise ValueError("peak_tflops / peak_gb_s 必须为正")
    ai = report["fused_arithmetic_intensity_flop_per_byte"]
    balance = peak_tflops * 1e12 / (peak_gb_s * 1e9)
    bound = "compute" if ai >= balance else "memory"
    return {
        "arithmetic_intensity_flop_per_byte": ai,
        "machine_balance_flop_per_byte": round(balance, 3),
        "bound": bound,
        "assumption": {"peak_tflops": peak_tflops, "peak_gb_s": peak_gb_s},
    }


def alignment_checks(
    profile: dict,
    dtype: str,
    offsets: dict | None = None,
    extra_slices: dict | None = None,
) -> list:
    """逐切片审计 32 字节物理粒度。

    判据：一行的连续段字节数是 32 的整数倍，且起始元素偏移换算成字节后也是
    32 的整数倍。任一条不满足，MTE 会退化到低效搬运，须用 ``DataCopyPad``
    或改内存排布。
    """
    elems = {
        "hidden_row": profile["hidden"],
        "kv_head_row": profile["head_dim"],
        "k_all_heads_row": profile["head_dim"] * profile["num_kv_heads"],
        "shared_expert_inter_row": profile["intermediate"],
        "gate_up_row": profile["intermediate"] * 2,
    }
    elems.update(extra_slices or {})
    extra = offsets or {}
    rows = []
    for name, count in elems.items():
        start = int(extra.get(name, 0))
        nbytes = _row_bytes(count, dtype)
        obytes = _row_bytes(start, dtype)
        rows.append(
            {
                "slice": name,
                "elements": count,
                "row_bytes": nbytes,
                "start_element": start,
                "start_bytes": obytes,
                "row_aligned": nbytes % ALIGN_BYTES == 0,
                "start_aligned": obytes % ALIGN_BYTES == 0,
                "aligned": nbytes % ALIGN_BYTES == 0 and obytes % ALIGN_BYTES == 0,
                "remedy": None
                if nbytes % ALIGN_BYTES == 0 and obytes % ALIGN_BYTES == 0
                else "use DataCopyPad or repack innermost dim to a 32B multiple",
            }
        )
    return rows


def ub_budget(
    tile_m: int,
    tile_n: int,
    dtype: str,
    operands: int,
    stages: int,
    ub_bytes: int,
    extra_bytes: int = 0,
) -> dict:
    """片上 UB 装一个 Vector 处理分块、按 ``stages`` 级乒乓所需的字节数。

    每级字节 = ``tile_m × tile_n × dtype 字节 × operands``；输出原地写回其中一个
    操作数缓冲，不额外占块。``extra_bytes`` 留给掩码 / 标量 / 归约缓存。
    """
    if tile_m <= 0 or tile_n <= 0 or operands <= 0 or stages <= 0:
        raise ValueError("tile_m/tile_n/operands/stages 必须为正")
    b = DTYPE_BYTES[dtype]
    per_stage = tile_m * tile_n * b * operands
    total = per_stage * stages + extra_bytes
    return {
        "tile_m": tile_m,
        "tile_n": tile_n,
        "dtype": dtype,
        "operands": operands,
        "stages": stages,
        "bytes_per_stage": per_stage,
        "ub_bytes": ub_bytes,
        "required_bytes": total,
        "fits": total <= ub_bytes,
        "headroom_bytes": ub_bytes - total,
        "fill_ratio": round(total / ub_bytes, 4) if ub_bytes else None,
    }


def max_stages_that_fit(
    tile_m: int,
    tile_n: int,
    dtype: str,
    operands: int,
    ub_bytes: int,
    extra_bytes: int = 0,
) -> int:
    """在给定分块下最多能开几级乒乓（0 表示一级都装不下）。"""
    per_stage = tile_m * tile_n * DTYPE_BYTES[dtype] * operands
    if per_stage <= 0:
        return 0
    return max(0, (ub_bytes - extra_bytes) // per_stage)


def max_tile_m_that_fit(
    tile_n: int,
    dtype: str,
    operands: int,
    stages: int,
    ub_bytes: int,
    extra_bytes: int = 0,
) -> int:
    """固定 ``tile_n`` 与级数时，可装下的最大 ``tile_m``（向下取 16 的整倍数）。"""
    per_row = tile_n * DTYPE_BYTES[dtype] * operands * stages
    if per_row <= 0:
        return 0
    rows = max(0, (ub_bytes - extra_bytes) // per_row)
    return rows - rows % CUBE_ATOM_M


def small_m_report(m: int) -> dict:
    """M < 16 时 Cube 的零填充浪费比例与派发建议（纯算术，不含实测）。"""
    if m <= 0:
        raise ValueError("m 必须为正整数")
    padded = math.ceil(m / CUBE_ATOM_M) * CUBE_ATOM_M
    waste = 1.0 - m / padded
    if m >= CUBE_ATOM_M:
        advice = "cube_gemm"
    elif m >= CUBE_ATOM_M // 2:
        advice = "cube_gemm_with_token_packing"
    else:
        advice = "vector_gemv_fallback"
    return {
        "m": m,
        "padded_m": padded,
        "pad_ratio": round(padded / m, 4),
        "wasted_mac_fraction": round(waste, 4),
        "dispatch_advice": advice,
    }


def build_report(
    profile_name: str,
    m_list: list,
    dtype: str,
    ub_kib_list: list,
    stages_list: list,
    operands: int,
    peak_tflops: float,
    peak_gb_s: float,
    offsets: dict | None = None,
    tile_m: int = CUBE_ATOM_M,
    tile_n: int | None = None,
    extra_slices: dict | None = None,
) -> dict:
    profile = PROFILES[profile_name]
    width = profile["intermediate"] if tile_n is None else tile_n

    hbm = []
    for m in m_list:
        chain = fused_epilogue_hbm(m, profile["intermediate"], profile["hidden"], dtype)
        hbm.append(
            {
                "hbm": chain,
                "roofline": roofline_verdict(chain, peak_tflops, peak_gb_s),
                "small_m": small_m_report(m),
            }
        )

    ub = []
    for kib in ub_kib_list:
        ub_bytes = kib * 1024
        best_stages = max_stages_that_fit(
            tile_m=tile_m,
            tile_n=width,
            dtype=dtype,
            operands=operands,
            ub_bytes=ub_bytes,
        )
        ub.append(
            {
                "ub_kib": kib,
                "sweep": [
                    ub_budget(
                        tile_m=tile_m,
                        tile_n=width,
                        dtype=dtype,
                        operands=operands,
                        stages=stages,
                        ub_bytes=ub_bytes,
                    )
                    for stages in stages_list
                ],
                "max_stages_for_tile": best_stages,
                "max_tile_m_at_3_stages": max_tile_m_that_fit(
                    tile_n=width,
                    dtype=dtype,
                    operands=operands,
                    stages=3,
                    ub_bytes=ub_bytes,
                ),
            }
        )
    return {
        "profile": profile_name,
        "profile_facts": profile,
        "dtype": dtype,
        "align_bytes": ALIGN_BYTES,
        "hbm_fusion": hbm,
        "alignment": alignment_checks(profile, dtype, offsets, extra_slices),
        "ub_budget": ub,
        "assumption": True,
        "note": (
            "所有字节数由维度与 dtype 宽度推出；峰值带宽/算力/UB 容量为输入参数，"
            "默认值仅作比较基准，不作为实测结论。"
        ),
    }


def main(argv: list | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Ascend 910B 算子方向离线静态审计（无设备，纯算术）"
    )
    parser.add_argument("--profile", default="qwen35-a3b-tp2", choices=sorted(PROFILES))
    parser.add_argument("--m", type=int, nargs="+", default=[1, 8, 10, 16, 32, 512])
    parser.add_argument("--dtype", default="bf16", choices=sorted(DTYPE_BYTES))
    parser.add_argument("--ub-kib", type=int, nargs="+", default=[128, 192, 256])
    parser.add_argument("--stages", type=int, nargs="+", default=[2, 3, 4])
    parser.add_argument("--operands", type=int, default=2)
    parser.add_argument("--tile-m", type=int, default=CUBE_ATOM_M)
    parser.add_argument(
        "--tile-n",
        type=int,
        default=None,
        help="分块宽度，默认取 profile 的 intermediate 局部宽度",
    )
    parser.add_argument("--peak-tflops", type=float, default=200.0)
    parser.add_argument("--peak-gb-s", type=float, default=800.0)
    parser.add_argument(
        "--offset",
        action="append",
        default=[],
        metavar="SLICE=ELEMS",
        help="切片起始元素偏移，可重复，例如 --offset kv_head_row=2048",
    )
    parser.add_argument(
        "--slice",
        action="append",
        default=[],
        metavar="NAME=ELEMS",
        help="追加一个自定切片的行元素数，可重复，例如 --slice moe_inter=512",
    )
    args = parser.parse_args(argv)

    offsets = {}
    for item in args.offset:
        name, _, value = item.partition("=")
        offsets[name.strip()] = int(value)

    extra_slices = {}
    for item in args.slice:
        name, _, value = item.partition("=")
        extra_slices[name.strip()] = int(value)

    report = build_report(
        profile_name=args.profile,
        m_list=args.m,
        dtype=args.dtype,
        ub_kib_list=args.ub_kib,
        stages_list=args.stages,
        operands=args.operands,
        peak_tflops=args.peak_tflops,
        peak_gb_s=args.peak_gb_s,
        offsets=offsets,
        tile_m=args.tile_m,
        tile_n=args.tile_n,
        extra_slices=extra_slices,
    )
    print(json.dumps(report, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
