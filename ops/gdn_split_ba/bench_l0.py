#!/usr/bin/env python3
"""Callable adapter for the deployed host split_ba; no standalone measurement."""

if __name__ == "__main__":
    print("No standalone measurement provided; callable implementation only.")
    raise SystemExit(3)


def optimized_op(ba):
    from vllm_ascend.ops.gdn_input_norm_host import gdn_split_ba, is_split_ba_enabled

    if not is_split_ba_enabled():
        raise ValueError("Host split_ba disabled; refusing to benchmark the fallback")
    return gdn_split_ba(ba)
