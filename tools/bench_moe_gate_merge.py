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

"""MoE 双门合并（router + shared_expert_gate）的 L0 配对 A/B。

两臂（同一张 x，单变量 = 是否合并）:

* ``stock_pair``:  ``F.linear(x, W_router)`` + ``F.linear(x, W_shared)``（两次 GEMM）
* ``merged``:      ``F.linear(x, W_merged)`` + 两路 ``narrow().contiguous()``
  （一次 GEMM）

为什么用图捕获回放而不是 eager Event 链: eager 提交地板实测 **34.6–36.6µs**，
大于本候选的 per-call 量级（router 7.56µs / shared_gate 4.77µs,
见 `docs/d7-matmulv2-decode-status.md` §2），eager 判不出差别。这里用
``torch.npu.NPUGraph`` 捕 k 个调用后回放取中位；``--evict-mb`` 做冷态差值法。

判据（预声明，不改）:
  ① ``stock_p50`` / ``merged_p50`` 两臂中位；
     ``gain_pct = (stock - merged) / stock * 100``
  ② ``numerics``: router logits 与 shared gate 与 stock 逐元素 bitwise 相等（int 视图）
  ③ 门: ``gain_pct >= --min-gain``（默认 5.0）且 numerics 全等 → 晋级 plain L3 + compare
     否则不晋级（收益不足或数值不等），证据保留。

用法::

    python3 tools/bench_moe_gate_merge.py --out <dir>/l0-moe-gate-merge.json
    python3 tools/bench_moe_gate_merge.py --m 1 --evict-mb 256 --out ...

退出码：0 成功｜1 失败｜2 用法错。
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import pathlib
import platform
import subprocess
import sys
import time

SCHEMA = "opt-ops-l0-moe-gate-merge.v1"
HIDDEN = 2048
N_ROUTER = 256
RC_OK, RC_FAIL, RC_USAGE = 0, 1, 2
REPO = "/root/vllm-workspace/vllm-ascend-split-batch-hust"


def _median_us(fn, k: int, replays: int, evict_mb: int) -> dict:
    import torch

    def replay_time(graph, n):
        for _ in range(5):
            graph.replay()
        torch.npu.synchronize()
        start = torch.npu.Event(enable_timing=True)
        end = torch.npu.Event(enable_timing=True)
        start.record()
        for _ in range(n):
            graph.replay()
        end.record()
        torch.npu.synchronize()
        return start.elapsed_time(end) * 1000.0 / n

    if evict_mb > 0:
        n_elem = int(evict_mb * 1e6 / 2)
        ev = torch.zeros(n_elem, dtype=torch.bfloat16, device="npu")

        g1 = torch.npu.NPUGraph()
        with torch.npu.graph(g1):
            for _ in range(k):
                ev.fill_(1.0)
                fn()
        g2 = torch.npu.NPUGraph()
        with torch.npu.graph(g2):
            for _ in range(k):
                ev.fill_(1.0)
        t1 = replay_time(g1, replays)
        t2 = replay_time(g2, replays)
        del ev
        return {
            "method": "npu_graph_capture_k_replay_l2cold",
            "kernels_per_graph": k,
            "replays": replays,
            "evict_mb": evict_mb,
            "with_evict_us_per_block": round(t1, 3),
            "evict_only_us_per_block": round(t2, 3),
            "median_us": round(max(t1 - t2, 0.0) / k, 3),
            "caveat": "差值法; 含 evict 与 GEMM 重叠扰动, 为冷态上界近似",
        }

    graph = torch.npu.NPUGraph()
    with torch.npu.graph(graph):
        for _ in range(k):
            fn()
    per = replay_time(graph, replays) / k
    return {
        "method": "npu_graph_capture_k_replay_l2warm",
        "kernels_per_graph": k,
        "replays": replays,
        "evict_mb": 0,
        "median_us": round(per, 3),
        "caveat": "背靠背同权重 L2 热态, 不代表窗口 HBM 冷流",
    }


def run(m: int, seed: int, k: int, replays: int, evict_mb: int) -> dict:
    import torch
    import torch.nn.functional as F

    gen = torch.Generator(device="cpu").manual_seed(seed)
    x = torch.randn(m, HIDDEN, generator=gen, dtype=torch.float32).to(torch.bfloat16)
    x = x.to("npu")
    w_router = (
        torch.randn(N_ROUTER, HIDDEN, generator=gen, dtype=torch.float32)
        .to(torch.bfloat16)
        .to("npu")
    )
    w_shared = (
        torch.randn(1, HIDDEN, generator=gen, dtype=torch.float32)
        .to(torch.bfloat16)
        .to("npu")
    )
    w_merged = torch.cat([w_router, w_shared], dim=0)

    def stock():
        return F.linear(x, w_router), F.linear(x, w_shared)

    def merged():
        out = F.linear(x, w_merged)
        return out.narrow(1, 0, N_ROUTER).contiguous(), out.narrow(
            1, N_ROUTER, 1
        ).contiguous()

    # 数值（int 视图 bitwise）
    ref_logits, ref_gate = stock()
    got_logits, got_gate = merged()
    numerics = {
        "logits_bitwise": bool(
            torch.equal(got_logits.view(torch.int16), ref_logits.view(torch.int16))
        ),
        "gate_bitwise": bool(
            torch.equal(got_gate.view(torch.int16), ref_gate.view(torch.int16))
        ),
        "logits_max_abs_diff": float(
            (got_logits.float() - ref_logits.float()).abs().max()
        ),
        "gate_max_abs_diff": float((got_gate.float() - ref_gate.float()).abs().max()),
        "logits_contiguous": bool(got_logits.is_contiguous()),
        "gate_contiguous": bool(got_gate.is_contiguous()),
    }

    return {
        "m": m,
        "stock_pair": _median_us(stock, k, replays, evict_mb),
        "merged": _median_us(merged, k, replays, evict_mb),
        "numerics": numerics,
        "weights": {
            "router": [N_ROUTER, HIDDEN],
            "shared_gate": [1, HIDDEN],
            "merged": [N_ROUTER + 1, HIDDEN],
        },
    }


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument(
        "--m", type=int, default=1, help="decode=M=1（可加 8 看批量不敏感）"
    )
    ap.add_argument("--k", type=int, default=16, help="每图捕获的调用数")
    ap.add_argument("--replays", type=int, default=40)
    ap.add_argument("--evict-mb", type=int, default=0)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--min-gain", type=float, default=5.0)
    ap.add_argument("--out", default="")
    args = ap.parse_args(argv)
    if args.m <= 0 or args.k <= 0 or args.replays <= 0:
        print("--m/--k/--replays 必须 >0", file=sys.stderr)
        return RC_USAGE
    try:
        import torch  # noqa: F401
        import torch_npu  # noqa: F401
    except Exception as exc:  # noqa: BLE001
        print(f"要求 NPU 但无法 import torch_npu: {exc!r}", file=sys.stderr)
        return RC_FAIL
    if torch.npu.device_count() == 0:
        print("torch.npu.device_count()==0", file=sys.stderr)
        return RC_FAIL

    t0 = time.monotonic()
    try:
        head = subprocess.run(
            ["git", "-C", REPO, "rev-parse", "HEAD"],
            capture_output=True,
            text=True,
            timeout=20,
        ).stdout.strip()
    except Exception:  # noqa: BLE001
        head = None

    cell = run(args.m, args.seed, args.k, args.replays, args.evict_mb)
    stock_us = cell["stock_pair"]["median_us"]
    merged_us = cell["merged"]["median_us"]
    gain = round((stock_us - merged_us) / stock_us * 100, 2) if stock_us else None
    numerics_ok = (
        cell["numerics"]["logits_bitwise"] and cell["numerics"]["gate_bitwise"]
    )
    product = {
        "schema": SCHEMA,
        "ts": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "status": "ok",
        "route": "028/D7-matmulv2-decode-fusion",
        "candidate": "router-shared-gate-merge",
        "command": {
            "argv": list(sys.argv if argv is None else argv),
            "cwd": os.getcwd(),
        },
        "hardware": {
            "device": "npu",
            "device_name": torch.npu.get_device_name(0),
            "host": platform.node(),
            "torch": torch.__version__,
        },
        "code": {
            "repo": REPO,
            "repo_head": head,
            "tool_sha256": hashlib.sha256(
                pathlib.Path(__file__).read_bytes()
            ).hexdigest(),
        },
        "cell": cell,
        "gate": {
            "min_gain_pct": args.min_gain,
            "gain_pct": gain,
            "numerics_ok": numerics_ok,
            "passed": bool(gain is not None and gain >= args.min_gain and numerics_ok),
            "rule": (
                "gain>=min-gain 且两路 bitwise 全等 → 晋级 plain L3 + compare; "
                "否则不晋级"
            ),
        },
        "not_claimed": [
            "图捕获回放是单算子内核时长, 不是窗口 all-kernel 占比; "
            "收益需 light L4 复核",
            "冷态差值法含 evict 扰动, 为上界近似",
            "bitwise 只记录不裁决; L3 8 条 prompt 输出一致才是数值门",
            "本工具不含载波安装路径（env 门在 vllm_ascend_split_batch.moe_gate_merge）",
        ],
        "budget": {"elapsed_s": round(time.monotonic() - t0, 2)},
    }
    verdict = "PASS" if product["gate"]["passed"] else "FAIL"
    print(
        f"  m={args.m}  stock_pair={stock_us:.2f}us  merged={merged_us:.2f}us  "
        f"gain={gain}%  numerics_ok={numerics_ok}  gate={verdict}"
    )
    if args.out:
        with open(args.out, "w") as f:
            json.dump(product, f, indent=1, ensure_ascii=False)
        print(f"saved -> {args.out}")
    return RC_OK


if __name__ == "__main__":
    sys.exit(main())
