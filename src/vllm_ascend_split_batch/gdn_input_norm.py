# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 vLLM-HUST contributors.

"""Fused input-RMSNorm wrapper.

No module in this package imports this file and it has no ``DELIVERY_INDEX.json``
entry: it is a parked route with no delivery state. Register it in the index before
treating the route as live.
"""

from __future__ import annotations

import os

_NORM_ENABLED = os.environ.get("VLLM_HUST_GDN_INPUT_NORM", "0") == "1"


def gdn_input_norm_fused(x, weight):
    """Fused RMSNorm for Qwen3.5 GDN in_proj decode phase [T=32, H=2048].

    Default-off guarded by VLLM_HUST_GDN_INPUT_NORM=1.

    Precision contract: the enabled path supports **bf16 inputs and bf16
    weights only**.  ``rms_norm_kernel`` rounds its intermediate to bf16 and
    stores bf16 no matter what ``x.dtype`` is (that fixed rounding is what makes
    it match the reference semantics bit for bit), so a non-bf16 call would be
    silently narrowed.  It is therefore refused with ``ValueError`` before the
    launch.  The default-off path keeps native ``x.dtype`` semantics and takes
    any float dtype.
    """
    if not _NORM_ENABLED:
        import torch

        var = x.to(torch.float32).pow(2).mean(-1, keepdim=True)
        rsqrt = torch.rsqrt(var + 1e-6)
        return (x.to(torch.float32) * rsqrt).to(x.dtype) * weight

    import torch

    if x.dtype != torch.bfloat16 or weight.dtype != torch.bfloat16:
        raise ValueError(
            "GDN fused input norm supports bf16 input and weight only "
            f"(got x.dtype={x.dtype}, weight.dtype={weight.dtype}); its "
            "intermediate rounding is fixed to bf16"
        )

    from .gdn_input_norm_kernel import rms_norm_kernel

    # Strided vector loads can exceed Ascend UB after compiler padding.
    if x.stride(1) != 1:
        x = x.contiguous()
    M, N = x.shape
    out = x.new_empty(x.shape)
    rms_norm_kernel[(M,)](
        x,
        weight,
        out,
        x.stride(0),
        x.stride(1),
        out.stride(0),
        out.stride(1),
        N=N,
        BLOCK_SIZE=N,
        eps=1e-6,
    )
    return out
