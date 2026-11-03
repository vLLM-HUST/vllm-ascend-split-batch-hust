#!/usr/bin/env python3
"""L0 microbenchmark for gdn_qkv_norm (vllm_ascend/ops/gdn.py:480,619-620)."""

from vllm_ascend_split_batch import gdn_qkv_norm


def optimized_op(x):
    return gdn_qkv_norm.unpack_norm(x)


if __name__ == "__main__":
    print("Benchmarking gdn_qkv_norm...")
