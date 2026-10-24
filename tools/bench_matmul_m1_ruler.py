#!/usr/bin/env python3
# Copyright (c) 2025-2026 Huawei Technologies Co., Ltd. All Rights Reserved.
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""decode M=1 skinny GEMM kernel-headroom 标尺（matmulv2-decode-m1-l0-ruler）。

对窗 1021-031741 实测的 8 个 MatMulV2 权重桶，每桶测三量（全部 eager、单卡、
``torch.npu.Event`` 包夹）:

* ``t_m1``   —— ``F.linear(x[1,H], W)`` 中位时长（eager 现状标尺）；
* ``bw_floor`` —— 同权重整读整写 device copy 的实测带宽**按单向折算**的纯装载地板
  （经验值: 不宣称 HBM 峰值带宽，只取本卡本态可达值）；
* ``launch_floor`` —— 最小 GEMM ``F.linear(x[1,H], W[8,H])`` 的中位时长
  （kernel 启动/最小工作量地板探针）。

判读量（v2，FULL 图内）::

    window_us_per_call = window_ms * 1000 / (calls_per_it * WINDOW_ITERS)
    pct_of_read_ceiling = (weight_bytes / window_us_per_call) / READ_CEILING_GBPS

理由: eager 提交地板实测 34.6–36.6µs，**大于**这些桶在 FULL 图内的 per-call 设备时间
（v1 证据: 11.97–26.11µs），因此 eager 侧的 ``eager_headroom_pct`` 只记录不判读。
本标尺只立不裁决: kernel 替换/tiling 类方法属 W1R 相邻家族，进入前需先问。

退出码：0 成功｜1 失败｜2 用法错。
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import platform
import statistics
import subprocess
import sys
import time

SCHEMA = "opt-ops-l0-matmul-m1-ruler.v1"
RULER_VERSION = 2
HIDDEN = 2048
TIMER_SCOPE = "npu_event_elapsed_eager_chain_including_host_submission_gaps"
RC_OK, RC_FAIL, RC_USAGE = 0, 1, 2
# 窗 1021-031741 的捕获范围（optstat: capture_scope=vllm-max-iterations-32）
WINDOW_ITERS = 32
# 同卡权重纯读上限（D7 卡记录 lm_head 纯读 1.51TB/s; 只作"距离上限"的分母, 不作收益宣称）
READ_CEILING_GBPS = 1510.0
# 每迭代 callsite 计数（与 CELLS 的 callsite 文本同源; 用来把 window_ms 折成 per-call）
CALLS_PER_IT = {
    "gdn_inproj": 30,
    "out_o": 40,
    "fa_qkv": 10,
    "router": 40,
    "shared_gate_up": 40,
    "shared_down": 40,
    "shared_gate": 40,
    "lm_head": 1,
}

# 窗 1021-031741 decode M=1 实测桶（weight -> (N, K, window_ms, callsite)）
CELLS = {
    "gdn_inproj": (6176, HIDDEN, 25.069, "GDN in_proj（F2 已并）30/it"),
    "out_o": (2048, HIDDEN, 15.319, "GDN out(30)+FA o(10) 40/it"),
    "fa_qkv": (4608, HIDDEN, 7.077, "FA qkv 含 attn_output_gate 10/it"),
    "router": (256, HIDDEN, 9.674, "MoE router GateLinear 40/it"),
    "shared_gate_up": (512, HIDDEN, 9.672, "shared expert gate_up 40/it"),
    "shared_down": (2048, 256, 8.916, "shared expert down 40/it"),
    "shared_gate": (1, HIDDEN, 6.104, "shared_expert_gate 标量门 40/it"),
    "lm_head": (124160, HIDDEN, 11.724, "lm_head 1/it"),
}


def oneway_bw_gbps(bytes_moved: int, t_us: float) -> float:
    """``copy_`` 的读+写合计字节折成**单向**带宽（GB/s）。

    v1 直接拿合计字节除以耗时当单向用，`bw_floor` 因此恰好小一半
    （gdn_inproj: 11.532µs 应为 23.06µs）。
    """
    return bytes_moved / 2.0 / (t_us * 1e-6) / 1e9


def load_floor_us(nbytes: int, oneway_gbps: float) -> float:
    """按单向可达带宽换算的纯装载地板（µs）。"""
    return nbytes / (oneway_gbps * 1e9) * 1e6


def window_us_per_call(window_ms: float, calls_per_it: int, iters: int) -> float:
    """FULL 图内单次调用设备时长（µs）。"""
    return window_ms * 1000.0 / (calls_per_it * iters)


def read_efficiency_pct(
    nbytes: int, us: float, ceiling_gbps: float = READ_CEILING_GBPS
) -> float:
    """有效读带宽占同卡纯读上限的百分比（判读量）。"""
    eff = nbytes / (us * 1e-6) / 1e9
    return eff / ceiling_gbps * 100.0


CONVERGED_PCT = 70.0


def build_reading(cells: list[dict]) -> dict:
    """按 FULL 图内有效读占纯读上限分档（纯函数, CPU 可测）。"""
    pct = {c["id"]: c["pct_of_read_ceiling"] for c in cells}
    big = [c["id"] for c in cells if c["weight_mb"] > 4]
    return {
        "rule": (
            f"按 FULL 图内 per-call 有效读占同卡纯读上限判读: "
            f">={CONVERGED_PCT:.0f}% → 该桶装载面已收敛; "
            f"<{CONVERGED_PCT:.0f}% → 存在**未测**装载空间（可达性需 kernel 通路证据, 本标尺不裁决）"
        ),
        "judged_metric": "pct_of_read_ceiling",
        "read_ceiling_gbps": READ_CEILING_GBPS,
        "converged_threshold_pct": CONVERGED_PCT,
        "window": {"source": "窗 1021-031741", "iters": WINDOW_ITERS},
        "per_cell_pct_of_read_ceiling": pct,
        "converged": sorted(i for i, v in pct.items() if v >= CONVERGED_PCT),
        "unmeasured_headroom": sorted(i for i, v in pct.items() if v < CONVERGED_PCT),
        "big_cells": big,
        "big_pct_median": round(
            statistics.median([pct[i] for i in big]), 1
        )
        if big
        else None,
        "withdrawn": (
            "v1 规则「>4MB 桶 headroom<20% → D7 无 kernel 杠杆」作废: 其分母是 eager 提交地板"
            "（34.6–36.6µs）, 大于被测桶的 FULL 图内 per-call（11.97–26.11µs）, 等于测不到"
        ),
    }


def timed(fn, iters: int) -> float:
    import torch

    torch.npu.synchronize()
    start = torch.npu.Event(enable_timing=True)
    end = torch.npu.Event(enable_timing=True)
    start.record()
    for _ in range(iters):
        fn()
    end.record()
    torch.npu.synchronize()
    return start.elapsed_time(end) * 1000.0 / iters


def median_us(fn, warmup: int, iters: int, repeats: int) -> float:
    import torch

    for _ in range(warmup):
        fn()
    torch.npu.synchronize()
    return round(statistics.median(timed(fn, iters) for _ in range(repeats)), 3)


def run_cell(name: str, n: int, k: int, ctx: dict) -> dict:
    import torch
    import torch.nn.functional as F

    dev = ctx["device"]
    gen = torch.Generator(device="cpu").manual_seed(ctx["seed"] + n + k)
    x = torch.randn(1, k, generator=gen, dtype=torch.float32).to(torch.bfloat16).to(dev)
    w = torch.randn(n, k, generator=gen, dtype=torch.float32).to(torch.bfloat16).to(dev)
    iters, warmup, repeats = ctx["iters"], ctx["warmup"], ctx["repeats"]

    t_m1 = median_us(lambda: F.linear(x, w), warmup, iters, repeats)

    w2 = torch.empty_like(w)

    def copy_w():
        w2.copy_(w)

    t_copy = median_us(copy_w, warmup, max(iters // 4, 20), repeats)
    nbytes = w.numel() * w.element_size()
    copy_bw = oneway_bw_gbps(nbytes * 2, t_copy)
    bw_floor_us = round(load_floor_us(nbytes, copy_bw), 3)

    w_min = (
        torch.randn(8, k, generator=gen, dtype=torch.float32).to(torch.bfloat16).to(dev)
    )
    t_launch = median_us(lambda: F.linear(x, w_min), warmup, iters, repeats)

    effective_floor = max(bw_floor_us, t_launch)
    eager_headroom = round((t_m1 - effective_floor) / t_m1 * 100, 2)
    per_call = round(
        window_us_per_call(CELLS[name][2], CALLS_PER_IT[name], WINDOW_ITERS), 2
    )
    doc = {
        "id": name,
        "weight": [n, k],
        "weight_mb": round(nbytes / 1e6, 2),
        "callsite": CELLS[name][3],
        "calls_per_it": CALLS_PER_IT[name],
        "window_ms": CELLS[name][2],
        "window_us_per_call": per_call,
        "effective_read_gbps": round(nbytes / (per_call * 1e-6) / 1e9, 1),
        "pct_of_read_ceiling": round(read_efficiency_pct(nbytes, per_call), 1),
        "t_m1_us": t_m1,
        "t_copy_us": t_copy,
        "copy_bw_gbps": round(copy_bw * 2, 1),
        "copy_bw_oneway_gbps": round(copy_bw, 1),
        "bw_floor_us": bw_floor_us,
        "t_launch_probe_us": t_launch,
        "effective_floor_us": round(effective_floor, 3),
        "eager_headroom_pct": eager_headroom,
        "headroom_vs_effective_floor_pct": eager_headroom,
        "eager_headroom_judged": False,
        "binding": "bw" if bw_floor_us >= t_launch else "launch",
    }
    del x, w, w2, w_min
    torch.npu.empty_cache()
    return doc


def sha256_file(path: str) -> str | None:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--iters", type=int, default=200)
    ap.add_argument("--repeats", type=int, default=3)
    ap.add_argument("--warmup", type=int, default=20)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--cells", default=",".join(CELLS))
    ap.add_argument("--out", default="")
    args = ap.parse_args(argv)
    if args.iters <= 0 or args.repeats <= 0:
        print("--iters/--repeats 必须 >0", file=sys.stderr)
        return RC_USAGE
    try:
        import torch
        import torch_npu  # noqa: F401
    except Exception as exc:  # noqa: BLE001
        print(f"要求 NPU 但无法 import torch_npu: {exc!r}", file=sys.stderr)
        return RC_FAIL
    if torch.npu.device_count() == 0:
        print("torch.npu.device_count()==0", file=sys.stderr)
        return RC_FAIL

    cells = [c.strip() for c in args.cells.split(",") if c.strip()]
    unknown = [c for c in cells if c not in CELLS]
    if unknown:
        print(f"未知 cell {unknown}", file=sys.stderr)
        return RC_USAGE

    t_start = time.monotonic()
    repo = "/root/vllm-workspace/vllm-ascend-split-batch-hust"
    try:
        head = subprocess.run(
            ["git", "-C", repo, "rev-parse", "HEAD"],
            capture_output=True,
            text=True,
            timeout=20,
        ).stdout.strip()
    except Exception:  # noqa: BLE001
        head = None
    product = {
        "schema": SCHEMA,
        "ruler_version": RULER_VERSION,
        "ts": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "status": "ok",
        "command": {
            "argv": list(sys.argv if argv is None else argv),
            "cwd": os.getcwd(),
            "python": sys.executable,
        },
        "candidate": "matmulv2-decode-m1-l0-ruler",
        "route": "028/D7-matmulv2-decode-fusion",
        "goal": {
            "what": "decode M=1 skinny GEMM kernel headroom 标尺（只立标尺, 不裁决 kernel 替换）",
            "judged_metric": "pct_of_read_ceiling = FULL 图内 per-call 有效读 / 同卡纯读上限",
            "floors": {
                "bw": "同权重 copy_ 实测带宽**单向折算**后的纯装载地板（经验地板, 非峰值宣称）",
                "launch": "最小 GEMM [8,H] 探针（eager 启动/最小工作量地板, 只记录不判读）",
                "effective": "max(bw, launch)（eager 侧, v2 起不参与判读）",
            },
        },
        "hardware": {
            "device": "npu",
            "device_name": torch.npu.get_device_name(0),
            "host": platform.node(),
            "torch": torch.__version__,
        },
        "code": {
            "repo": repo,
            "repo_head": head,
            "tool_sha256": sha256_file(os.path.abspath(__file__)),
        },
        "cells": [],
        "not_claimed": [
            "eager Event 链延迟, 不是 FULL/图模式时长; 图模式 per-call 由 window_ms/calls_per_it 折算",
            "eager 提交地板实测 34.6–36.6µs, 大于这些桶的 FULL 图内 per-call(11.97–26.11µs), "
            "故 eager_headroom_pct 只记录不判读",
            "copy_ 是读+写混合实测; v2 按单向折算 bw_floor（v1 误用双向, 地板小一半）",
            "pct_of_read_ceiling 只说明与纯读上限的距离, 不构成 kernel 替换收益结论; "
            "W1R 相邻家族进入需先问",
        ],
    }
    ctx = {
        "device": "npu",
        "seed": args.seed,
        "iters": args.iters,
        "repeats": args.repeats,
        "warmup": args.warmup,
    }
    for name in cells:
        n, k, _, _ = CELLS[name]
        doc = run_cell(name, n, k, ctx)
        product["cells"].append(doc)
        print(
            f"  {name:16s} W={doc['weight']} {doc['weight_mb']:>7}MB  "
            f"per_call={doc['window_us_per_call']:>7.2f}us  "
            f"read={doc['effective_read_gbps']:>7.1f}GB/s "
            f"({doc['pct_of_read_ceiling']:>5.1f}% of {READ_CEILING_GBPS:.0f})  "
            f"eager t_m1={doc['t_m1_us']:>8.2f}us (bw_floor={doc['bw_floor_us']:>7.2f}us | "
            f"launch={doc['t_launch_probe_us']:>5.2f}us, 不判读)"
        )
    product["reading"] = build_reading(product["cells"])
    product["budget"] = {"elapsed_s": round(time.monotonic() - t_start, 2)}
    if args.out:
        with open(args.out, "w") as f:
            json.dump(product, f, indent=1, ensure_ascii=False)
        print(f"saved -> {args.out}  status={product['status']}")
    return RC_OK


if __name__ == "__main__":
    sys.exit(main())
