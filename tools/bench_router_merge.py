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

"""MoE router 并入合并 GEMM（router-merge-into-inproj-qkv）L0 配对 A/B 微基准。

单变量：基线 = 独立两次 ``F.linear``（in_proj/qkv 与 router GateLinear，与
``vllm-hust/vllm/model_executor/models/qwen3_next.py`` 的 ReplicatedLinear 路径同
语义，bias=False 假设）；候选 = 加载期 ``cat([W_main, W_router], dim=0)`` 一次
``F.linear`` + logits 切片 ``.contiguous()``（KB 载荷
ge_graph_moe_gate_router_weight_layout：路由权重/输出须连续）。

臂：``baseline_pair``（2 GEMM）｜``candidate_merged``（1 GEMM + 切片转连续）｜
``diag_router_only``（单 router GEMM，只作诊断不参与裁决）。

计时：``torch.npu.Event`` 包夹 iters 次 eager 调用 → /iters；偶数轮正序、奇数轮
反序，两臂同轮相邻。裁决量 = baseline_pair vs candidate_merged 配对中位差。

数值：int 视图 bitwise（in_proj 切片、router 切片分别对照基线）；GEMM 重结合允许
非 bitwise，bitwise 与 max_abs_diff 只记录不裁决，L3 才是数值门。

用法::

    timeout -k 30 600 python3 tools/bench_router_merge.py \
        --out /root/包体恢复/records/ladder/<ts>-l0-router-merge.json

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

import torch
import torch.nn.functional as F

SCHEMA = "opt-ops-l0-router-merge.v1"
MODEL = "Qwen3.5-35B-A3B"
HIDDEN = 2048
ROUTER_N = 256
VARIANTS = {
    "gdn_inproj": {"main_n": 6176, "callsite": "30 GDN 层 in_proj（F2 已并 qkvz+ba）"},
    "fa_qkv": {"main_n": 4608, "callsite": "10 FA 层 qkv（含 attn_output_gate 列）"},
}
TMS = {1: "decode", 353: "prefill"}
TIMER_SCOPE = "npu_event_elapsed_eager_chain_including_host_submission_gaps"
RC_OK, RC_FAIL, RC_USAGE = 0, 1, 2


def int_view(t: torch.Tensor) -> torch.Tensor:
    return t.detach().to("cpu").contiguous().view(torch.int16).reshape(-1)


def bitwise(a: torch.Tensor, b: torch.Tensor) -> dict:
    va, vb = int_view(a), int_view(b)
    eq = bool(torch.equal(va, vb))
    diff = (va != vb).sum().item()
    mad = (
        (a.detach().to("cpu").float() - b.detach().to("cpu").float()).abs().max().item()
    )
    return {
        "bitwise_ok": eq,
        "mismatch": int(diff),
        "numel": int(va.numel()),
        "max_abs_diff": mad,
    }


def timed(fn, iters: int) -> dict:
    torch.npu.synchronize()
    start = torch.npu.Event(enable_timing=True)
    end = torch.npu.Event(enable_timing=True)
    t0 = time.perf_counter()
    start.record()
    for _ in range(iters):
        fn()
    end.record()
    t_enq = time.perf_counter()
    torch.npu.synchronize()
    t_tot = time.perf_counter()
    total_us = start.elapsed_time(end) * 1000.0
    return {
        "event_elapsed_us_total": round(total_us, 3),
        "us_per_iter_event_elapsed": round(total_us / iters, 3),
        "us_host_enqueue": round((t_enq - t0) * 1e6 / iters, 3),
        "us_host_total": round((t_tot - t0) * 1e6 / iters, 3),
    }


def run_cell(variant: str, m: int, ctx: dict) -> dict:
    main_n = VARIANTS[variant]["main_n"]
    dev, seed, iters = ctx["device"], ctx["seed"], ctx["iters"]
    gen = torch.Generator(device="cpu").manual_seed(seed + m * 10 + main_n)
    x = (
        torch.randn(m, HIDDEN, generator=gen, dtype=torch.float32)
        .to(torch.bfloat16)
        .to(dev)
    )
    w_main = (
        torch.randn(main_n, HIDDEN, generator=gen, dtype=torch.float32)
        .to(torch.bfloat16)
        .to(dev)
    )
    w_r = (
        torch.randn(ROUTER_N, HIDDEN, generator=gen, dtype=torch.float32)
        .to(torch.bfloat16)
        .to(dev)
    )
    w_m = torch.cat([w_main, w_r], dim=0).contiguous()

    def base_pair():
        y = F.linear(x, w_main)
        logits = F.linear(x, w_r)
        return y, logits

    def cand_merged():
        y = F.linear(x, w_m)
        return y[:, :main_n], y[:, main_n:].contiguous()

    def diag_router():
        return F.linear(x, w_r)

    # ---- 数值（bitwise 记录，不裁决）----
    yb, lb = base_pair()
    yc, lc = cand_merged()
    numerics = {
        "main_slice_vs_baseline": bitwise(yc, yb),
        "router_slice_vs_baseline": bitwise(lc, lb),
        "weight_contiguous": {
            "w_merged": bool(w_m.is_contiguous()),
            "w_baseline": bool(w_main.is_contiguous()),
        },
        "router_slice_contiguous_after_copy": bool(lc.is_contiguous()),
        "merged_row_stride_pollutes_main_view": {
            "main_view_stride": list(yc.stride()),
            "baseline_stride": list(yb.stride()),
        },
    }

    # ---- 计时 ----
    arms = {
        "baseline_pair": base_pair,
        "candidate_merged": cand_merged,
        "diag_router_only": diag_router,
    }
    for f in arms.values():
        for _ in range(ctx["warmup"]):
            f()
    torch.npu.synchronize()
    schedule = [
        list(arms) if i % 2 == 0 else list(reversed(arms))
        for i in range(ctx["repeats"])
    ]
    rows: dict[str, list[dict]] = {a: [] for a in arms}
    for order in schedule:
        for arm in order:
            rows[arm].append(timed(arms[arm], iters))
    perf = {}
    for arm, rs in rows.items():
        vals = [r["us_per_iter_event_elapsed"] for r in rs]
        perf[arm] = {"us_median": round(statistics.median(vals), 3), "runs": rs}
    b_key = "us_per_iter_event_elapsed"
    paired = [
        round(
            (c["us_per_iter_event_elapsed"] / b["us_per_iter_event_elapsed"] - 1) * 100,
            3,
        )
        for b, c in zip(rows["baseline_pair"], rows["candidate_merged"])
    ]
    doc = {
        "id": f"{variant}-M{m}",
        "variant": variant,
        "callsite": VARIANTS[variant]["callsite"],
        "m": m,
        "phase": TMS.get(m, "undeclared"),
        "shapes": {
            "x": [m, HIDDEN],
            "w_main": [main_n, HIDDEN],
            "w_router": [ROUTER_N, HIDDEN],
            "w_merged": [main_n + ROUTER_N, HIDDEN],
        },
        "numerics": numerics,
        "perf": {
            "dtype": "bfloat16",
            "iters": iters,
            "repeats": ctx["repeats"],
            "warmup": ctx["warmup"],
            "timer_scope": TIMER_SCOPE,
            "schedule": schedule,
            "arms": perf,
            "paired_delta_pct": paired,
            "paired_delta_pct_median": round(statistics.median(paired), 3),
            "delta_pct_median_of_medians": round(
                (
                    perf["candidate_merged"]["us_median"]
                    / perf["baseline_pair"]["us_median"]
                    - 1
                )
                * 100,
                3,
            ),
        },
    }
    del x, w_main, w_r, w_m, yb, lb, yc, lc
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
    ap.add_argument("--iters", type=int, default=100)
    ap.add_argument("--repeats", type=int, default=3)
    ap.add_argument("--warmup", type=int, default=20)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--variants", default="gdn_inproj,fa_qkv")
    ap.add_argument("--ms", default="1,353")
    ap.add_argument("--out", default="")
    args = ap.parse_args(argv)
    if args.iters <= 0 or args.repeats <= 0 or args.warmup < 0:
        print("--iters/--repeats 必须 >0，--warmup >=0", file=sys.stderr)
        return RC_USAGE
    try:
        import torch_npu  # noqa: F401
    except Exception as exc:  # noqa: BLE001
        print(f"要求 NPU 但无法 import torch_npu: {exc!r}", file=sys.stderr)
        return RC_FAIL
    if torch.npu.device_count() == 0:
        print("torch.npu.device_count()==0", file=sys.stderr)
        return RC_FAIL

    variants = [v.strip() for v in args.variants.split(",") if v.strip()]
    ms = [int(v) for v in args.ms.split(",") if v.strip()]
    unknown = [v for v in variants if v not in VARIANTS]
    if unknown or not ms:
        print(f"未知 variant {unknown} 或空 --ms", file=sys.stderr)
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
        "ts": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "status": "ok",
        "command": {
            "argv": list(sys.argv if argv is None else argv),
            "cwd": os.getcwd(),
            "python": sys.executable,
        },
        "candidate": "router-merge-into-inproj-qkv",
        "route": "028/D7-matmulv2-decode-fusion",
        "caliber_note": "L0 eager 单卡；窗口 1021-031741 的 decode M=1 / prefill M=353 实测形状",
        "goal": {
            "baseline": "F.linear(x,W_main) + F.linear(x,W_router) 两次独立 GEMM（qwen3_next ReplicatedLinear 语义, bias=False 假设）",
            "candidate": "cat([W_main,W_router],0) 一次 F.linear + logits 切片 .contiguous()",
            "arbiter": "baseline_pair vs candidate_merged 配对中位差; diag_router_only 只诊断",
        },
        "kb_constraints": {
            "card": "ge_graph_moe_gate_router_weight_layout",
            "applied": "路由输出切片强制 contiguous; 权重 cat(dim=0) 保行连续 ND",
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
            "eager 链流程局部延迟，不是 FULL/图模式时长、不是 all-kernel 占比、不换算端到端",
            "bitwise 不裁决——GEMM 重结合允许; L3 8/8 输出一致才是数值门",
            "merged 行跨度会污染 in_proj 输出视图（stride 6432/4864）: 下游 unpack/reshape 的适配成本不在本基准内，属 L3/实现风险",
            "w_merged 为加载期一次性构造；本基准不含构造耗时（一次性 pay）",
        ],
    }
    ctx = {
        "device": "npu",
        "seed": args.seed,
        "iters": args.iters,
        "repeats": args.repeats,
        "warmup": args.warmup,
    }
    for variant in variants:
        for m in ms:
            doc = run_cell(variant, m, ctx)
            product["cells"].append(doc)
            p = doc["perf"]
            print(
                f"  {doc['id']:20s} pair={p['arms']['baseline_pair']['us_median']:.2f}us "
                f"merged={p['arms']['candidate_merged']['us_median']:.2f}us "
                f"router_only={p['arms']['diag_router_only']['us_median']:.2f}us "
                f"paired_median={p['paired_delta_pct_median']}% "
                f"bitwise(router)={doc['numerics']['router_slice_vs_baseline']['bitwise_ok']} "
                f"bitwise(main)={doc['numerics']['main_slice_vs_baseline']['bitwise_ok']}"
            )
    product["budget"] = {"elapsed_s": round(time.monotonic() - t_start, 2)}
    if args.out:
        with open(args.out, "w") as f:
            json.dump(product, f, indent=1, ensure_ascii=False)
        print(f"saved -> {args.out}  status={product['status']}")
    return RC_OK


if __name__ == "__main__":
    sys.exit(main())
