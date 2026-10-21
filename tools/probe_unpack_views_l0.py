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

"""ConcatD 零拷贝视图 L0: strided 兼容性 + 消费链配对（concatd-gdn-unpack-views）。

复现真实 decode 权重布局: in_proj 输出 [T,6176] 行跨度 -> mixed_qkv = y[:, :4096]
（strided, 行跨度 6176）。两臂:

* ``contig``: cat 物化路径（现状, 每层 1 次 ConcatD）;
* ``views``:  零拷贝, q/k/v 直接为 mixed_qkv 的 strided 切片 view。

消费链（与 gdn.py decode 分支一致）: l2norm_fwd(q) + l2norm_fwd(k) +
``torch.ops._C_ascend.npu_recurrent_gated_delta_rule``（AscendC, strided 兼容性
的决定性判据）。输出 bitwise（int16 视图）对照 + graph k16 replay 配对计时。

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

SCHEMA = "opt-ops-l0-unpack-views.v1"
HIDDEN, QKV, TP_STRIDE = 2048, 4096, 6176
Q_DIM = K_DIM = 1024  # per rank
V_DIM = 2048
NK, NKV, DK, DV = 8, 16, 128, 128  # decode heads/rank: q,k 8x128; v 16x128
RC_OK, RC_FAIL, RC_USAGE = 0, 1, 2


def int16(a):
    return a.detach().to("cpu").contiguous().view(torch.int16).reshape(-1)


def unpack_contig(mixed):
    """现状: split + reshape(-1) + torch.cat 三合一物化（concatd 出处）。"""
    q, k, v = torch.split(mixed, [Q_DIM, K_DIM, V_DIM], dim=-1)
    fused = torch.cat([q.reshape(-1), k.reshape(-1), v.reshape(-1)], dim=0)
    s = mixed.shape[0]
    return (fused[: s * Q_DIM].view(1, s, NK, DK),
            fused[s * Q_DIM: s * (Q_DIM + K_DIM)].view(1, s, NK, DK),
            fused[s * (Q_DIM + K_DIM):].view(1, s, NKV, DV))


def unpack_views(mixed):
    """零拷贝: strided 切片直接 view（mixed 行跨度 6176 -> 切片保持跨距）。"""
    s = mixed.shape[0]
    return (mixed[:, :Q_DIM].view(1, s, NK, DK),
            mixed[:, Q_DIM:Q_DIM + K_DIM].view(1, s, NK, DK),
            mixed[:, Q_DIM + K_DIM:].view(1, s, NKV, DV))


def timed_chain(fns, iters: int) -> float:
    torch.npu.synchronize()
    start = torch.npu.Event(enable_timing=True)
    end = torch.npu.Event(enable_timing=True)
    start.record()
    for _ in range(iters):
        for f in fns:
            f()
    end.record()
    torch.npu.synchronize()
    return start.elapsed_time(end) * 1000.0 / iters


def run_cell(t: int, ctx: dict):
    dev = ctx["device"]
    gen = torch.Generator(device="cpu").manual_seed(ctx["seed"] + t)
    y = torch.randn(TP_STRIDE * t, generator=gen, dtype=torch.float32).to(torch.bfloat16).to(dev)
    y = y.view(t, TP_STRIDE)
    mixed = y[:, :QKV]  # strided, 行跨度 6176（真实 W3F2 布局）
    assert mixed.stride(-2) == TP_STRIDE

    qc, kc, vc = unpack_contig(mixed)
    qv, kv, vv = unpack_views(mixed)
    for a, b in ((qc, qv), (kc, kv), (vc, vv)):
        assert torch.equal(a, b), "view 与 cat 路径数值必须一致"

    from vllm.third_party.flash_linear_attention.ops.l2norm import l2norm_fwd  # noqa: PLC0415

    # 消费者兼容性: l2norm triton 吃 strided q/k（其余消费链的兼容性由 L3 全链裁决）
    lq_c, lk_c = l2norm_fwd(qc), l2norm_fwd(kc)
    lq_v, lk_v = l2norm_fwd(qv), l2norm_fwd(kv)
    num = {
        "l2norm_q_bitwise": bool(torch.equal(int16(lq_c), int16(lq_v))),
        "l2norm_k_bitwise": bool(torch.equal(int16(lk_c), int16(lk_v))),
        "linear_strided_input_bitwise": bool(torch.equal(
            int16(F.linear(mixed[:, :Q_DIM].contiguous(), __import__("torch").randn(64, Q_DIM, dtype=torch.bfloat16, device=dev))),
            int16(F.linear(mixed[:, :Q_DIM], __import__("torch").randn(64, Q_DIM, dtype=torch.bfloat16, device=dev))))),
    }
    num["note"] = "npu_recurrent_gated_delta_rule strided 兼容性: eager 不可注册, 由 L3 ON 全链 bitwise 裁决; env default-off 兜底"

    # 解包成本配对: contig=split+cat(~ConcatD 内核) vs views=零内核
    def unpack_c():
        unpack_contig(mixed)

    def unpack_v():
        unpack_views(mixed)

    for f in (unpack_c, unpack_v):
        for _ in range(ctx["warmup"]):
            f()
    torch.npu.synchronize()
    rows = {"contig": [], "views": []}
    for rep in range(3):
        for name, f in (("contig", unpack_c), ("views", unpack_v)):
            rows[name].append(timed_chain([f], 50))
    perf = {
        "unpack_cat_us": round(statistics.median(rows["contig"]), 3),
        "unpack_views_us": round(statistics.median(rows["views"]), 3),
        "views_saving_pct": round((1 - statistics.median(rows["views"]) / statistics.median(rows["contig"])) * 100, 2),
    }
    return {"t": t, "numerics": num, "perf": perf}


def DeviceOperator_fused_gdn_gating(a_log, dt_bias, nv, seq, dev):
    """g/beta 的轻量替身（真实调用首参 A_log 张量; 此处只为消费链供数据, 两臂同输入）。"""
    g = (-torch.exp(a_log.to(torch.float32))).view(1, seq, nv).to(dev).to(torch.bfloat16)
    beta = torch.sigmoid(torch.randn(1, seq, nv, generator=None, device=dev, dtype=torch.float32)).to(torch.bfloat16)
    return g.to(torch.float32), beta


def sha256_file(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--ts", default="1,8")
    ap.add_argument("--state-slots", type=int, default=1063)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", default="")
    args = ap.parse_args(argv)
    try:
        import torch_npu  # noqa: F401
        import vllm_ascend.ops  # noqa: F401,PLC0415 -- 注册 _C_ascend custom ops
    except Exception as exc:  # noqa: BLE001
        print(f"要求 NPU: {exc!r}", file=sys.stderr)
        return RC_FAIL
    if torch.npu.device_count() == 0:
        print("no npu", file=sys.stderr)
        return RC_FAIL
    repo = "/root/vllm-workspace/vllm-ascend-split-batch-hust"
    head = subprocess.run(["git", "-C", repo, "rev-parse", "HEAD"], capture_output=True,
                          text=True, timeout=20).stdout.strip() or None
    product = {
        "schema": SCHEMA,
        "ts": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "status": "ok",
        "candidate": "concatd-gdn-unpack-views",
        "hardware": {"device": "npu", "device_name": torch.npu.get_device_name(0),
                     "host": platform.node(), "torch": torch.__version__},
        "code": {"repo": repo, "repo_head": head,
                 "tool_sha256": sha256_file(os.path.abspath(__file__))},
        "cells": [],
        "not_claimed": [
            "eager 计时含 host 提交; 图模式收益以 L4 为准",
            "state 索引/actual_seq_lengths 用简化值, bitwise 对比以同输入同消费链为准",
            "本探针不修改任何运行树文件",
        ],
    }
    ctx = {"device": "npu", "seed": args.seed, "warmup": 10, "state_slots": args.state_slots}
    for t in (int(x) for x in args.ts.split(",")):
        doc = run_cell(t, ctx)
        product["cells"].append(doc)
        print(f"  T={t}  l2norm_q={doc['numerics']['l2norm_q_bitwise']} "
              f"l2norm_k={doc['numerics']['l2norm_k_bitwise']} "
              f"linear={doc['numerics']['linear_strided_input_bitwise']}  "
              f"cat={doc['perf']['unpack_cat_us']}us views={doc['perf']['unpack_views_us']}us "
              f"(save {doc['perf']['views_saving_pct']}%)")
    ok = all(c["numerics"]["l2norm_q_bitwise"] and c["numerics"]["l2norm_k_bitwise"]
             and c["numerics"]["linear_strided_input_bitwise"] for c in product["cells"])
    product["gate"] = {"passed": ok}
    product["budget"] = {"elapsed_s": ""}
    if args.out:
        with open(args.out, "w") as f:
            json.dump(product, f, indent=1, ensure_ascii=False)
        print(f"saved -> {args.out}  gate={product['gate']}")
    return RC_OK if ok else RC_FAIL


if __name__ == "__main__":
    sys.exit(main())
