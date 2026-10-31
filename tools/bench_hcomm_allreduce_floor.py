#!/usr/bin/env python3
"""L0: hcom allReduce 地板探测 —— TP2 HCCL, 消息规模扫描。

窗内事实(1021-182739, decode FULL graph): 153µs/次 @ ~58KB, transit 占 1.8%,
idle 85% —— 本基准给出裸 eager 链地板, 与窗内图内值对账定机制方向。

用法: torchrun --nproc-per-node=2 tools/bench_hcomm_allreduce_floor.py [--iters 200]
落: bench/results/<ts>-l0-hcomm-floor.json
"""

import argparse
import json
import os
import statistics
import time

import torch
import torch.distributed as dist
import torch_npu  # noqa: F401


def bench_once(x, iters):
    torch.npu.synchronize()
    t0 = time.perf_counter()
    for _ in range(iters):
        dist.all_reduce(x)
    torch.npu.synchronize()
    return (time.perf_counter() - t0) / iters * 1e6


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--iters", type=int, default=200)
    ap.add_argument("--sizes-kb", default="4,16,58,128,256,1024")
    ap.add_argument("--out", default=None)
    args = ap.parse_args()

    rank = int(os.environ["RANK"])
    torch.npu.set_device(rank)
    dist.init_process_group("hccl")

    rows = []
    for kb in (float(s) for s in args.sizes_kb.split(",")):
        n = int(kb * 1024 / 2)
        x = torch.ones(n, dtype=torch.bfloat16, device="npu")
        for _ in range(10):
            dist.all_reduce(x)
        torch.npu.synchronize()
        us = [bench_once(x, args.iters) for _ in range(3)]
        row = {
            "kb": kb,
            "us_med": round(statistics.median(us), 1),
            "us_all": [round(u, 1) for u in us],
        }
        rows.append(row)
        if rank == 0:
            print(f"{kb:>7}KB: {row['us_med']}µs  (runs {row['us_all']})")

    if rank == 0:
        ts = time.strftime("%m%d-%H%M%S")
        out = args.out or f"bench/results/{ts}-l0-hcomm-floor.json"
        os.makedirs(os.path.dirname(out), exist_ok=True)
        doc = {
            "schema": "l0-hcomm-floor.v1",
            "world": int(os.environ["WORLD_SIZE"]),
            "backend": "hccl",
            "dtype": "bf16",
            "iters": args.iters,
            "rows": rows,
            "window_reference": {
                "opstat": "1021-182739-opstat.json",
                "in_graph_us": 153.0,
                "avg_kb": 58.0,
                "note": "窗内为 FULL graph 图内值(无 host launch); "
                "eager 值含 ~35µs host 提交地板",
            },
        }
        with open(out, "w") as f:
            json.dump(doc, f, ensure_ascii=False, indent=1)
        print(f"saved -> {out}")
    dist.destroy_process_group()


if __name__ == "__main__":
    main()
