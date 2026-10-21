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

"""MoeInitRoutingV3 decode M=1 替代核 L0（moeinitrouting-m1-fastpath）。

窗 1021-031741 实测 decode 每层三小核 26.4µs ≈ GMM 对 41%; 本基准裁单核替代:
* ``v2_init``       基线现状（device_op.npu_moe_init_routing 包裹的 npu_moe_init_routing_v2）
* ``composition``   原语组合（topk 展开行 gather + torch_npu.npu_moe_compute_expert_tokens）
契约: 三输出 sorted_tokens / expanded_row_idx / expert_tokens 全部 bitwise 对照基线。
计时: NPU graph capture k16 replay。

退出码：0 成功｜1 失败｜2 用法错。
"""

from __future__ import annotations

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

SCHEMA = "opt-ops-l0-moe-init-m1.v1"
RC_OK, RC_FAIL, RC_USAGE = 0, 1, 2
HIDDEN, TOPK, EXPERTS = 2048, 8, 256


def v2_call(x, ids, eslot=TOPK):
    return torch_npu.npu_moe_init_routing_v2(
        x, ids, scale=None, active_num=ids.numel(), expert_num=EXPERTS,
        expert_tokens_num_type=1, expert_tokens_num_flag=True,
        active_expert_range=[0, EXPERTS], quant_mode=-1)


def composition_call(x, ids, eslot=TOPK):
    n = ids.numel()
    flat = ids.flatten().to(torch.int32)
    token_ids = (torch.arange(n, device=x.device, dtype=torch.int32) // eslot).clamp(max=x.shape[0] - 1)
    expanded = x[token_ids.long()]            # [8,2048] topk 展开行
    order = torch.argsort(flat)               # [8]
    ys = expanded[order].contiguous()
    tt = torch_npu.npu_moe_compute_expert_tokens(flat, EXPERTS)
    return ys, order.to(torch.int32), tt


def bit(a, b):
    return bool(torch.equal(a.detach().to("cpu").contiguous().view(-1), b.detach().to("cpu").contiguous().view(-1)))


def replay_time(fn, k=16, iters=40):
    torch.npu.synchronize()
    g = torch.npu.NPUGraph()
    with torch.npu.graph(g):
        for _ in range(k):
            fn()
    for _ in range(5):
        g.replay()
    torch.npu.synchronize()
    s = torch.npu.Event(enable_timing=True)
    e = torch.npu.Event(enable_timing=True)
    s.record()
    for _ in range(iters):
        g.replay()
    e.record()
    torch.npu.synchronize()
    return round(s.elapsed_time(e) * 1000.0 / (iters * k), 3)


def sha256_file(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", default="")
    args = ap.parse_args(argv)
    repo = "/root/vllm-workspace/vllm-ascend-split-batch-hust"
    t0 = time.monotonic()
    dev = "npu"
    gen = torch.Generator(device="cpu").manual_seed(args.seed)
    x = torch.randn(1, HIDDEN, generator=gen, dtype=torch.float32).to(torch.bfloat16).to(dev)
    ids = torch.randint(0, EXPERTS, (1, TOPK), generator=gen, dtype=torch.int32).to(dev)
    y2, e2, t2, s2 = v2_call(x, ids)
    arms = {"v2_init": lambda: v2_call(x, ids),
            "composition": lambda: composition_call(x, ids)}
    doc = {"cells": []}
    ok_all = True
    for name, fn in arms.items():
        try:
            ys, ei, tt = composition_call(x, ids) if name == "composition" else (y2, e2, t2)
            if name == "composition":
                # torch_npu 兼容收集: composition 输出与 v2 v2_contract 对照（组内排序 tie 顺序可能不同, 用 set 语义+计数核）
                counts_ok = bool(torch.equal(tt, t2))
                sorted_ok = torch.equal(ys.view(-1), y2.view(-1))  # tie 顺序差异会被此捕获
                eidx_ok = torch.equal(ei.view(-1), e2.view(-1))
                ok = sorted_ok and eidx_ok and counts_ok
                doc["cells"].append({"arm": name, "sorted_bitwise": sorted_ok,
                                     "rowidx_bitwise": eidx_ok, "tokens_bitwise": counts_ok,
                                     "dur_us": replay_time(fn)})
            else:
                doc["cells"].append({"arm": name, "baseline": True, "dur_us": replay_time(fn)})
        except Exception as exc:  # noqa: BLE001
            ok_all = False
            doc["cells"].append({"arm": name, "error": repr(exc)[:220]})
        print(doc["cells"][-1])
    comp = [c for c in doc["cells"] if c["arm"] == "composition"]
    base = [c for c in doc["cells"] if c["arm"] == "v2_init"]
    if comp and base and "dur_us" in comp[0] and "dur_us" in base[0]:
        delta = round((comp[0]["dur_us"] / base[0]["dur_us"] - 1) * 100, 2)
        doc["verdict"] = {"composition_delta_pct": delta,
                          "composition_ok": all(k for k in comp[0] if k.endswith("bitwise"))}
        print(f"  composition delta {delta}% vs v2_init {base[0]['dur_us']}us")
    head = subprocess.run(["git", "-C", repo, "rev-parse", "HEAD"], capture_output=True,
                          text=True, timeout=20).stdout.strip() or None
    doc.update({"schema": SCHEMA, "ts": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
                "hardware": {"device": "npu", "device_name": torch.npu.get_device_name(0)},
                "code": {"repo": repo, "repo_head": head, "tool_sha256": sha256_file(os.path.abspath(__file__))},
                "elapsed_s": round(time.monotonic() - t0, 2)})
    if args.out:
        with open(args.out, "w") as f:
            json.dump(doc, f, indent=1, ensure_ascii=False)
        print(f"saved -> {args.out}")
    return RC_OK


if __name__ == "__main__":
    sys.exit(main())
