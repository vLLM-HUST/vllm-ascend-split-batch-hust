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

"""moeinitrouting trio-fusion 原型: 单 triton 核 gating(topk) + init-routing。

chains（带宽注: M=1 decode）:
* baseline = torch.ops._C_ascend.moe_gating_top_k_hash + npu_moe_init_routing_v2（现状两核）
* proto    = 单 triton kernel（softmax -> 按位打包 key 排序取 top8 -> id 排序 -> x 行 gather -> counts）

契约核对: topk_ids/topk_weights 集合语义、expert_tokens bitwise、sorted 行 bitwise（tie 顺序差异如实记）。
退出码: 0 成功｜1 失败。
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import subprocess
import sys
import time

import torch
import torch_npu
import triton
import triton.language as tl

RC_OK, RC_FAIL = 0, 1
HIDDEN, E, K = 2048, 256, 8


@triton.jit
def fused_router_kernel(
    logits_ptr,
    x_ptr,
    topk_ids_ptr,
    topk_w_ptr,
    sorted_ptr,
    rowidx_ptr,
    counts_ptr,
    H: tl.constexpr,
    E: tl.constexpr,
    K: tl.constexpr,
):
    offs = tl.arange(0, E)
    lg = tl.load(logits_ptr + offs).to(tl.float32)
    mx = tl.max(lg, axis=0)
    ex = tl.exp(lg - mx)
    probs = ex / tl.sum(ex, axis=0)
    p = probs
    for i in tl.static_range(K):
        m = tl.max(p, axis=0)
        e = tl.argmax(p, axis=0)
        tl.store(topk_w_ptr + i, m.to(topk_w_ptr.dtype.element_ty))
        tl.store(topk_ids_ptr + i, e.to(tl.int32))
        p = tl.where(offs == e, -1.0e30, p)
    kidx = tl.arange(0, K)
    ids = tl.load(topk_ids_ptr + kidx).to(tl.int32)
    xrow = tl.load(x_ptr + tl.arange(0, H)).to(x_ptr.dtype.element_ty)
    sblock = tl.broadcast_to(xrow[None, :], (K, H))
    tl.store(sorted_ptr + kidx[:, None] * H + tl.arange(0, H)[None, :], sblock)
    iv = ids[:, None].to(tl.int32)
    less = tl.sum((ids[None, :] < iv).to(tl.int32), axis=1)
    sidx = kidx[None, :]
    ties_before = tl.sum(
        ((ids[None, :] == iv) & (sidx < kidx[:, None])).to(tl.int32), axis=1
    )
    tl.store(rowidx_ptr + kidx, (less + ties_before).to(tl.int32))
    counts = tl.zeros((E,), tl.int32)
    for k in tl.static_range(K):
        e_k = tl.sum(tl.where(kidx == k, ids, 0), axis=0)
        counts += (offs == e_k.to(tl.int32)).to(tl.int32)
    tl.store(counts_ptr + offs, counts.to(counts_ptr.dtype.element_ty))


def baseline_chain(logits, x):
    lg = logits.view(1, -1).to(torch.float32)
    w, i, _ = torch_npu.npu_moe_gating_top_k(lg, renorm=1, k=K)
    y, e, t, s = torch_npu.npu_moe_init_routing_v2(
        x,
        i.to(torch.int32),
        scale=None,
        active_num=i.numel(),
        expert_num=E,
        expert_tokens_num_type=1,
        expert_tokens_num_flag=True,
        active_expert_range=[0, E],
        quant_mode=-1,
    )
    return w, i, y, t


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", default="")
    args = ap.parse_args(argv)
    dev = "npu"
    t0 = time.monotonic()
    gen = torch.Generator(device="cpu").manual_seed(args.seed)
    W = (
        torch.randn(E, HIDDEN, generator=gen, dtype=torch.float32)
        .to(torch.bfloat16)
        .to(dev)
    )
    x = (
        torch.randn(1, HIDDEN, generator=gen, dtype=torch.float32)
        .to(torch.bfloat16)
        .to(dev)
    )
    logits = torch.nn.functional.linear(x, W).flatten()  # [256] bf16
    out = {"schema": "opt-ops-l0-trio-proto.v1"}
    try:
        wb, ib, yb, tb = baseline_chain(logits, x)
        ids_p = torch.empty(K, device=dev, dtype=torch.int32)
        w_p = torch.empty(K, device=dev, dtype=torch.float32)
        sorted_p = torch.empty(K, HIDDEN, device=dev, dtype=torch.bfloat16)
        rowidx_p = torch.empty(K, device=dev, dtype=torch.int32)
        counts_p = torch.empty(E, device=dev, dtype=torch.int64)
        fused_router_kernel[(1,)](
            logits.contiguous(),
            x.contiguous(),
            ids_p,
            w_p,
            sorted_p,
            rowidx_p,
            counts_p,
            H=HIDDEN,
            E=E,
            K=K,
            num_warps=1,
        )
        torch.npu.synchronize()
        # 契约
        set_ok = torch.equal(torch.sort(ib.flatten())[0], torch.sort(ids_p)[0])
        w_close = bool(
            torch.allclose(wb.flatten().float(), w_p.float(), atol=1e-5, rtol=1e-5)
        )
        counts_ok = bool(torch.equal(tb, counts_p))
        sorted_ok = bool(torch.equal(yb.view(-1), sorted_p.view(-1)))
        doc = {
            "topk_set_ok": set_ok,
            "weights_close": w_close,
            "counts_bitwise": counts_ok,
            "sorted_rows_bitwise": sorted_ok,
            "baseline_ids": ib.flatten().cpu().tolist(),
            "proto_ids": ids_p.cpu().tolist(),
        }

        # 计时: 两臂 graph k16
        def base_arm():
            baseline_chain(logits, x)

        def proto_arm():
            fused_router_kernel[(1,)](
                logits.contiguous(),
                x.contiguous(),
                ids_p,
                w_p,
                sorted_p,
                rowidx_p,
                counts_p,
                H=HIDDEN,
                E=E,
                K=K,
                num_warps=1,
            )

        def t_graph(fn, k=16, iters=40):
            torch.npu.synchronize()
            g = torch.npu.NPUGraph()
            with torch.npu.graph(g):
                for _ in range(k):
                    fn()
            for _ in range(5):
                g.replay()
            torch.npu.synchronize()
            s, e = (
                torch.npu.Event(enable_timing=True),
                torch.npu.Event(enable_timing=True),
            )
            s.record()
            for _ in range(iters):
                g.replay()
            e.record()
            torch.npu.synchronize()
            return round(s.elapsed_time(e) * 1000.0 / (iters * k), 3)

        doc["baseline_us"] = t_graph(base_arm)
        doc["proto_us"] = t_graph(proto_arm)
        doc["savings_pct"] = round((1 - doc["proto_us"] / doc["baseline_us"]) * 100, 2)
        doc["status"] = "OK"
    except Exception as exc:  # noqa: BLE001
        doc = {"status": "FAILED", "error": repr(exc)[:300]}
        print(f"FAILED {exc!r}")
    repo = "/root/vllm-workspace/vllm-ascend-split-batch-hust"
    head = (
        subprocess.run(
            ["git", "-C", repo, "rev-parse", "HEAD"],
            capture_output=True,
            text=True,
            timeout=20,
        ).stdout.strip()
        or None
    )
    doc.update(
        {
            "ts": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
            "hardware": {"device": "npu"},
            "code": {
                "repo_head": head,
                "tool_sha256": hashlib.sha256(
                    open(os.path.abspath(__file__), "rb").read()
                ).hexdigest(),
            },
            "elapsed_s": round(time.monotonic() - t0, 2),
        }
    )
    if args.out:
        with open(args.out, "w") as f:
            json.dump(doc, f, indent=1, ensure_ascii=False)
        print(f"saved -> {args.out}")
    for k, v in doc.items():
        if k not in ("error",):
            print(f"  {k}: {v}")
    return RC_OK if doc.get("status") == "OK" else RC_FAIL


if __name__ == "__main__":
    sys.exit(main())
