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

"""decode M=1 skinny GEMM 权重布局 A/B（kernel 领地 L0, matmulv2 m1 布局标尺续篇）。

eager 链被 ~35µs host 提交地板钳住（1021-132409）, 无法仲裁 sub-launch-floor;
本基准改用 **profiler 逐 kernel 设备时长**（与窗 1021-031741 kernel_details 同口径）
做仲裁。单变量 = 权重内存布局, 前向调用语义不变:

* ``nd_linear``  基线: ``F.linear(x, W)`` ND 行主序（现状）;
* ``tnd_matmul`` 物理转置: ``x @ Wt``, ``Wt = W.t().contiguous()``（加载期一次性）;
* ``nz_linear``  NZ 分形: ``npu_format_cast(W, FRACTAL_NZ)`` 后同 ``F.linear``
  （torch_npu 不支持则如实记 error, fail-open 单臂）。

数值: 每臂 vs 基线 int16 视图 bitwise + max_abs_diff（记录不裁决; L3 才是数值门）。
预声明门: in_proj 桶任一臂设备时长改进 ≥15% → 立 L3 候选; 否则 kernel 布局领地关闭。

退出码：0 成功｜1 失败｜2 用法错。
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import platform
import subprocess
import sys
import time

import torch
import torch.nn.functional as F

SCHEMA = "opt-ops-l0-matmul-m1-layout.v1"
HIDDEN = 2048
RC_OK, RC_FAIL, RC_USAGE = 0, 1, 2
NZ_FORMAT = 29  # ACL_FORMAT_FRACTAL_NZ
CELLS = {
    "gdn_inproj": (6176, 26.46, "窗内 0.96TB/s vs 同卡 1.51TB/s"),
    "fa_qkv": (4608, 22.91, "窗内 MTE2 65%"),
    "out_o": (2048, 11.46, "窗内 mixed"),
}


def int_view(t: torch.Tensor) -> torch.Tensor:
    return t.detach().to("cpu").contiguous().view(torch.int16).reshape(-1)


def timed_arm(fn, warmup: int, prof_iters: int, evict_mb: int = 0) -> dict:
    """NPU graph capture + replay: 每 graph 打包 k 个连续 GEMM 摊薄 host 提交。

    evict_mb>0 时为 L2 冷态仿真: graph 内每 GEMM 前插入 evict 缓冲整写（逐出 L2），
    另测 evict-only graph, 冷态 GEMM 时长 = (G1-G2)/k —— 对齐窗口的 HBM 流场景。"""
    k = 16
    iters = max(prof_iters, 40)
    for _ in range(warmup):
        fn()
    torch.npu.synchronize()

    def replay_time(g, n):
        for _ in range(5):
            g.replay()
        torch.npu.synchronize()
        start = torch.npu.Event(enable_timing=True)
        end = torch.npu.Event(enable_timing=True)
        start.record()
        for _ in range(n):
            g.replay()
        end.record()
        torch.npu.synchronize()
        return start.elapsed_time(end) * 1000.0 / n

    if evict_mb > 0:
        n_elem = int(evict_mb * 1e6 / 2)
        ev = torch.zeros(n_elem, dtype=torch.bfloat16, device="npu")

        def body():
            ev.fill_(1.0)
            fn()

        g1 = torch.npu.NPUGraph()
        with torch.npu.graph(g1):
            for _ in range(k):
                body()
        g2 = torch.npu.NPUGraph()
        with torch.npu.graph(g2):
            for _ in range(k):
                ev.fill_(1.0)
        t1 = replay_time(g1, iters)
        t2 = replay_time(g2, iters)
        per = (t1 - t2) / k
        return {
            "method": "npu_graph_capture_k16_replay_l2cold",
            "kernels_per_graph": k,
            "replays": iters,
            "evict_mb": evict_mb,
            "with_evict_us_per_block": round(t1, 3),
            "evict_only_us_per_block": round(t2, 3),
            "median_us": round(per, 3),
            "min_us": None,
            "n_kernels_found": iters * k,
            "caveat": "差值法; 含 evict 与 GEMM 的重叠扰动, 为冷态上界近似",
        }

    g = torch.npu.NPUGraph()
    with torch.npu.graph(g):
        for _ in range(k):
            fn()
    per = replay_time(g, iters)
    return {
        "method": "npu_graph_capture_k16_replay_l2warm",
        "kernels_per_graph": k,
        "replays": iters,
        "evict_mb": 0,
        "median_us": round(per, 3),
        "min_us": None,
        "n_kernels_found": iters * k,
        "caveat": "背靠背同权重 L2 热态, 不代表窗口 HBM 冷流",
    }


def run_cell(name: str, n: int, ctx: dict) -> dict:
    k = HIDDEN
    dev = ctx["device"]
    gen = torch.Generator(device="cpu").manual_seed(ctx["seed"] + n)
    x = torch.randn(1, k, generator=gen, dtype=torch.float32).to(torch.bfloat16).to(dev)
    w = torch.randn(n, k, generator=gen, dtype=torch.float32).to(torch.bfloat16).to(dev)
    w_t = w.t().contiguous()

    def f_nd():
        return F.linear(x, w)

    def f_tnd():
        return torch.matmul(x, w_t)

    arms = {"nd_linear": f_nd, "tnd_matmul": f_tnd}
    try:
        import torch_npu  # noqa: PLC0415

        w_nz = torch_npu.npu_format_cast(w.clone(), NZ_FORMAT)
        fmt = torch_npu.get_npu_format(w_nz)

        def f_nz():
            return F.linear(x, w_nz)

        f_nz()
        torch.npu.synchronize()
        arms["nz_linear"] = f_nz
        nz_meta = {
            "requested_format": NZ_FORMAT,
            "observed_format": int(fmt),
            "applied": True,
        }
    except Exception as exc:  # noqa: BLE001 — 单臂失败如实记录, 其余臂继续
        nz_meta = {"applied": False, "error": repr(exc)[:200]}

    ref = f_nd()
    torch.npu.synchronize()
    numerics = {}
    for arm, fn in arms.items():
        if arm == "nd_linear":
            continue
        out = fn()
        torch.npu.synchronize()
        va, vb = int_view(ref), int_view(out)
        numerics[arm] = {
            "bitwise_ok": bool(torch.equal(va, vb)),
            "mismatch": int((va != vb).sum()),
            "max_abs_diff": round((ref.float() - out.float()).abs().max().item(), 6),
            "shape_ok": tuple(ref.shape) == tuple(out.shape),
        }
        del out

    perf = {}
    for arm, fn in arms.items():
        perf[arm] = timed_arm(
            fn, ctx["warmup"], ctx["prof_iters"], evict_mb=ctx.get("evict_mb", 0)
        )
    base = perf["nd_linear"]["median_us"]
    for arm, p in perf.items():
        if arm != "nd_linear" and p["median_us"] and base:
            p["delta_pct_vs_nd"] = round((p["median_us"] / base - 1) * 100, 2)
    doc = {
        "id": name,
        "weight": [n, k],
        "weight_mb": round(w.numel() * w.element_size() / 1e6, 2),
        "window_dur_us": CELLS[name][1],
        "window_note": CELLS[name][2],
        "nz": nz_meta,
        "numerics": numerics,
        "perf": perf,
    }
    del x, w, w_t, ref
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
    ap.add_argument("--prof-iters", type=int, default=30)
    ap.add_argument("--warmup", type=int, default=20)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument(
        "--evict-mb",
        type=int,
        default=0,
        help=">0 时启用 L2 冷态仿真(graph 内逐出缓冲整写, 差值法)",
    )
    ap.add_argument("--cells", default=",".join(CELLS))
    ap.add_argument(
        "--gate-pct",
        type=float,
        default=15.0,
        help="预声明门: inproj 桶改进 >=此百分比 -> 立候选",
    )
    ap.add_argument("--out", default="")
    args = ap.parse_args(argv)
    try:
        import torch_npu  # noqa: F401
    except Exception as exc:  # noqa: BLE001
        print(f"要求 NPU: {exc!r}", file=sys.stderr)
        return RC_FAIL
    if torch.npu.device_count() == 0:
        print("torch.npu.device_count()==0", file=sys.stderr)
        return RC_FAIL

    cells = [c.strip() for c in args.cells.split(",") if c.strip()]
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
        "candidate": "matmulv2-decode-m1-layout",
        "route": "028/D7-matmulv2-decode-fusion",
        "goal": {
            "what": "权重布局(ND/物理转置/NZ)对 M=1 skinny GEMM 设备时长的影响",
            "arbiter": "profiler chrome-trace kernel 设备时长中位（与窗口 kernel_details 同口径）",
            "gate": f"in_proj 任一臂改进 >= {args.gate_pct}% -> 立 L3 候选; 否则布局领地关闭",
        },
        "hardware": {
            "device": "npu",
            "device_name": torch.npu.get_device_name(0),
            "host": platform.node(),
            "torch": torch.__version__,
            "torch_npu": getattr(torch_npu, "__version__", ""),
        },
        "code": {
            "repo": repo,
            "repo_head": head,
            "tool_sha256": sha256_file(os.path.abspath(__file__)),
        },
        "cells": [],
        "not_claimed": [
            "profiler 设备时长与 FULL 图模式非严格同构（含 profiling 开销差异），只比方向",
            "bitwise 记录不裁决; 若中选臂非 bitwise, L3 风险升级须如实上报（F3 教训）",
            "单变量=布局; 不含 kernel 模板替换（那属更进一步的新领地）",
        ],
    }
    ctx = {
        "device": "npu",
        "seed": args.seed,
        "warmup": args.warmup,
        "prof_iters": args.prof_iters,
        "evict_mb": args.evict_mb,
    }
    for name in cells:
        n, _, _ = CELLS[name]
        doc = run_cell(name, n, ctx)
        product["cells"].append(doc)
        row = "  ".join(
            f"{a}={p['median_us']}us{('(' + str(p.get('delta_pct_vs_nd')) + '%)') if p.get('delta_pct_vs_nd') is not None else ''}"
            for a, p in doc["perf"].items()
        )
        print(f"  {name:12s} {doc['weight']} {row}")
    gate = None
    for cell in product["cells"]:
        if cell["id"] != "gdn_inproj":
            continue
        for arm in ("tnd_matmul", "nz_linear"):
            p = cell["perf"].get(arm) or {}
            if (
                p.get("delta_pct_vs_nd") is not None
                and -p["delta_pct_vs_nd"] >= args.gate_pct
            ):
                gate = {
                    "arm": arm,
                    "delta_pct": p["delta_pct_vs_nd"],
                    "bitwise_ok": (cell["numerics"].get(arm) or {}).get("bitwise_ok"),
                }
    product["gate"] = gate or {
        "passed": False,
        "reason": f"无臂达 in_proj 改进 >= {args.gate_pct}%",
    }
    product["budget"] = {"elapsed_s": round(time.monotonic() - t_start, 2)}
    if args.out:
        with open(args.out, "w") as f:
            json.dump(product, f, indent=1, ensure_ascii=False)
        print(
            f"saved -> {args.out}  status={product['status']}  gate={product['gate']}"
        )
    return RC_OK


if __name__ == "__main__":
    sys.exit(main())
