#!/usr/bin/env python
"""L0 microbenchmark: fused shared-expert epilogue multiply.

Compares the aclnn broadcast multiply (`gate * out`) against the triton kernel
behind `VLLM_HUST_MOE_EPILOGUE=1`. Measurements run inside an NPU graph with
50 repetitions packed per replay, so launch overhead is counted the same way it
is on the serving path.

Usage (single card):

    python ops/moe_epilogue/bench_l0.py

Requires the host fork with `vllm_ascend.ops.triton.moe_epilogue` present.
"""

import os
import time

import torch
import torch_npu  # noqa: F401
from vllm_ascend.ops.triton import moe_epilogue

HIDDEN = 2048
REPEAT = 50
ITERS = 30
ROWS = (1, 8, 16, 32)


def _stock(gate: torch.Tensor, out: torch.Tensor) -> torch.Tensor:
    return gate * out


def _fused(gate: torch.Tensor, out: torch.Tensor) -> torch.Tensor:
    return moe_epilogue.gate_mul(gate, out)


def _bench(fn, gate: torch.Tensor, out: torch.Tensor) -> float:
    for _ in range(10):
        fn(gate, out)
    torch.npu.synchronize()

    graph = torch.npu.NPUGraph()
    with torch.npu.graph(graph):
        for _ in range(REPEAT):
            fn(gate, out)
    torch.npu.synchronize()

    for _ in range(5):
        graph.replay()
    torch.npu.synchronize()

    start = time.perf_counter()
    for _ in range(ITERS):
        graph.replay()
    torch.npu.synchronize()
    return (time.perf_counter() - start) / ITERS * 1e6 / REPEAT


def main() -> None:
    torch.npu.set_device(0)
    device = "npu:0"
    os.environ[moe_epilogue.ENV_MOE_EPILOGUE] = "1"

    header = (
        f"{'M':>3} {'stock_us':>9} {'fused_us':>9} "
        f"{'saved_us':>9} {'saved_%':>8} {'bitwise':>8}"
    )
    print(header)
    for rows in ROWS:
        gate = torch.randn(rows, 1, dtype=torch.bfloat16, device=device)
        out = torch.randn(rows, HIDDEN, dtype=torch.bfloat16, device=device)

        stock_us = _bench(_stock, gate, out)
        fused_us = _bench(_fused, gate, out)

        bitwise = torch.equal(_stock(gate, out), _fused(gate, out))
        saved = stock_us - fused_us
        print(
            f"{rows:3d} {stock_us:9.2f} {fused_us:9.2f} {saved:9.2f} "
            f"{saved / stock_us * 100:7.1f}% {str(bitwise):>8}"
        )


if __name__ == "__main__":
    main()
