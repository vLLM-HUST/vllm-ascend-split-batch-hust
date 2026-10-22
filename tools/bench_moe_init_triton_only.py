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

"""init-routing triton 核单独 L0（验收段门): decode M=1, v2 vs triton-only。

门: triton_init <= v2_init - 10%（graph k16）；契约 bitwise（sorted/counts/rowidx）。
退出码: 0 成功｜1 失败。
"""

import argparse
import hashlib
import json
import os
import statistics
import subprocess
import sys
import time

import torch
import torch_npu
import triton
import triton.language as tl

RC_OK, RC_FAIL = 0, 1
HIDDEN, TOPK, EXPERTS = 2048, 8, 256


@triton.jit
def init_only_kernel(
    x_ptr,
    ids_ptr,
    sorted_ptr,
    rowidx_ptr,
    counts_ptr,
    H: tl.constexpr,
    E: tl.constexpr,
    K: tl.constexpr,
):
    kidx = tl.arange(0, K)
    ids = tl.load(ids_ptr + kidx).to(tl.int32)
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
    for j in tl.static_range(K):
        e_j = tl.sum(tl.where(kidx == j, ids, 0), axis=0)
        counts += (tl.arange(0, E) == e_j.to(tl.int32)).to(tl.int32)
    tl.store(counts_ptr + tl.arange(0, E), counts.to(counts_ptr.dtype.element_ty))


def replay_time(fn, k=16, iters=40):
    torch.npu.synchronize()
    g = torch.npu.NPUGraph()
    with torch.npu.graph(g):
        for _ in range(k):
            fn()
    for _ in range(5):
        g.replay()
    torch.npu.synchronize()
    s, e = torch.npu.Event(enable_timing=True), torch.npu.Event(enable_timing=True)
    s.record()
    for _ in range(iters):
        g.replay()
    e.record()
    torch.npu.synchronize()
    return round(s.elapsed_time(e) * 1000.0 / (iters * k), 3)


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", default="")
    args = ap.parse_args(argv)
    dev = "npu"
    t0 = time.monotonic()
    gen = torch.Generator(device="cpu").manual_seed(args.seed)
    x = (
        torch.randn(1, HIDDEN, generator=gen, dtype=torch.float32)
        .to(torch.bfloat16)
        .to(dev)
    )
    ids = torch.randint(0, EXPERTS, (1, TOPK), generator=gen, dtype=torch.int32).to(dev)

    def v2():
        return torch_npu.npu_moe_init_routing_v2(
            x,
            ids,
            scale=None,
            active_num=TOPK,
            expert_num=EXPERTS,
            expert_tokens_num_type=1,
            expert_tokens_num_flag=True,
            active_expert_range=[0, EXPERTS],
            quant_mode=-1,
        )

    sorted_p = torch.empty(TOPK, HIDDEN, device=dev, dtype=torch.bfloat16)
    rowidx_p = torch.empty(TOPK, device=dev, dtype=torch.int32)
    counts_p = torch.empty(EXPERTS, device=dev, dtype=torch.int64)

    def tri():
        init_only_kernel[(1,)](
            x,
            ids,
            sorted_p,
            rowidx_p,
            counts_p,
            H=HIDDEN,
            E=EXPERTS,
            K=TOPK,
            num_warps=4,
        )

    y2, e2, t2, _ = v2()
    tri()
    torch.npu.synchronize()
    num = {
        "sorted_bitwise": bool(torch.equal(y2, sorted_p)),
        "counts_bitwise": bool(torch.equal(t2, counts_p)),
        "rowidx_bitwise": bool(torch.equal(e2, rowidx_p)),
    }
    for f in (v2, tri):
        for _ in range(10):
            f()
    torch.npu.synchronize()
    # 交错 3 轮配对
    rows = {"v2": [], "tri": []}
    for rep in range(3):
        for name, f in (("v2", v2), ("tri", tri)):
            rows[name].append(replay_time(f))
    perf = {
        "v2_us": round(statistics.median(rows["v2"]), 3),
        "tri_us": round(statistics.median(rows["tri"]), 3),
        "delta_pct": round(
            (statistics.median(rows["tri"]) / statistics.median(rows["v2"]) - 1) * 100,
            2,
        ),
    }
    gate_pass = perf["delta_pct"] <= -10.0 and all(num.values())
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
    doc = {
        "schema": "opt-ops-l0-moe-init-triton-only.v1",
        "ts": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "status": "OK" if gate_pass else "GATE-FAIL",
        "numerics": num,
        "perf": perf,
        "gate": {"rule": "tri <= v2-10% 且 三输出 bitwise", "passed": gate_pass},
        "code": {
            "repo_head": head,
            "tool_sha256": hashlib.sha256(
                open(os.path.abspath(__file__), "rb").read()
            ).hexdigest(),
        },
        "elapsed_s": round(time.monotonic() - t0, 2),
    }
    if args.out:
        with open(args.out, "w") as f:
            json.dump(doc, f, indent=1, ensure_ascii=False)
        print(f"saved -> {args.out}")
    print(json.dumps(doc, ensure_ascii=False)[:400])
    return RC_OK if gate_pass else RC_FAIL


if __name__ == "__main__":
    sys.exit(main())
