# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 vLLM-HUST contributors.

from __future__ import annotations


def validate(x):
    import torch

    if x.shape != (16, 4096) or x.stride() != (4096, 1):
        raise ValueError("GDN fused norm requires T16, width4096, stride4096")
    if x.dtype != torch.bfloat16 or x.requires_grad:
        raise ValueError("GDN fused norm requires inference bf16")
    if x.device.type != "npu":
        raise ValueError("GDN fused norm requires NPU")


def output_views(flat):
    return (
        flat[:16384].view(1, 16, 8, 128),
        flat[16384:32768].view(1, 16, 8, 128),
        flat[32768:].view(1, 16, 16, 128),
    )


def launch_into(x, flat):
    from .gdn_qkv_norm_kernel import unpack_norm_kernel

    unpack_norm_kernel[(3, 16)](x, flat)


def unpack_norm(x):
    import torch

    validate(x)
    flat = torch.empty(65536, dtype=x.dtype, device=x.device)
    launch_into(x, flat)
    return output_views(flat)
