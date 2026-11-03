#!/usr/bin/env python3
"""L0 microbenchmark for gdn_input_norm (vllm_ascend/ops/gdn.py:in_proj_norm)."""

from vllm_ascend_split_batch.gdn_input_norm import gdn_input_norm_fused


def optimized_op(x, weight):
    return gdn_input_norm_fused(x, weight)


if __name__ == "__main__":
    print("Benchmarking gdn_input_norm...")
