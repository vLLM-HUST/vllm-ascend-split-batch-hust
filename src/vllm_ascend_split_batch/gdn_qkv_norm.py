# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 vLLM-HUST contributors.

from __future__ import annotations

#: Flat layout of the output buffer, in elements.  ``unpack_norm_kernel`` writes
#: q at ``[0, Q_END)``, k at ``[Q_END, K_END)`` and v at ``[K_END, TOTAL)``; the
#: split is the ``segment < 2`` branch of that kernel (q/k get an L2Norm per
#: 128-wide head, v is copied verbatim).
Q_ELEMENTS = 16 * 8 * 128  # 16 tokens x 8 q heads x 128
K_ELEMENTS = 16 * 8 * 128  # 16 tokens x 8 k heads x 128
V_ELEMENTS = 16 * 16 * 128  # 16 tokens x 16 v heads x 128
Q_END = Q_ELEMENTS
K_END = Q_ELEMENTS + K_ELEMENTS
TOTAL_ELEMENTS = Q_ELEMENTS + K_ELEMENTS + V_ELEMENTS


def validate(x):
    import torch

    if x.shape != (16, 4096) or x.stride() != (4096, 1):
        raise ValueError("GDN fused norm requires T16, width4096, stride4096")
    if x.dtype != torch.bfloat16 or x.requires_grad:
        raise ValueError("GDN fused norm requires inference bf16")
    if x.device.type != "npu":
        raise ValueError("GDN fused norm requires NPU")


def output_views(flat):
    """Split the flat output buffer into ``(query, key, value)`` views.

    Return order and shapes: ``query`` (1, 16, 8, 128) = ``flat[:Q_END]``,
    ``key`` (1, 16, 8, 128) = ``flat[Q_END:K_END]``, ``value`` (1, 16, 16, 128)
    = ``flat[K_END:]``.

    All three views are slices of the *same* ``flat`` storage, i.e. they alias
    each other and stay valid only while ``flat`` is alive; write into one and
    the bytes of the other two are unaffected but the buffer they point at is
    shared.  ``unpack_norm`` allocates one buffer per call, so two calls never
    share storage.
    """
    return (
        flat[:Q_END].view(1, 16, 8, 128),
        flat[Q_END:K_END].view(1, 16, 8, 128),
        flat[K_END:].view(1, 16, 16, 128),
    )


def launch_into(x, flat):
    from .gdn_qkv_norm_kernel import unpack_norm_kernel

    unpack_norm_kernel[(3, 16)](x, flat, multibuffer=True)


def unpack_norm(x):
    """Unpack + L2-normalise a GDN mixed qkv (one call = one fresh buffer).

    Input layout: ``x`` is the ``(16, 4096)`` projection output, one row per
    token, with q in columns ``[0, 1024)``, k in ``[1024, 2048)`` and v in
    ``[2048, 4096)`` -- the per-token offsets the kernel reads.

    Returns ``(query, key, value)``: q and k L2-normalised per 128-wide head, v
    copied verbatim; the three tensors are views of one freshly allocated
    ``TOTAL_ELEMENTS`` buffer (see :func:`output_views` for the exact layout and
    the shared-storage relation).
    """
    import torch

    validate(x)
    flat = torch.empty(TOTAL_ELEMENTS, dtype=x.dtype, device=x.device)
    launch_into(x, flat)
    return output_views(flat)
