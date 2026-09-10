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

"""Host entry point for the fused ``gelu_and_mul`` triton-ascend kernel.

Consumes the vLLM/GeGLU layout: ``input [*, 2*inter]`` bf16 where the gate is
the first half and the up-projection the second half; returns ``[*, inter]``
bf16 with ``out = gelu(gate) * up`` (exact-erf gelu, see ``kernels.py``).

The only public symbol is ``gelu_and_mul``.  grid/block are derived from
``inter``: one program per ``(row, inter-block)`` with ``BLOCK_N`` clamped to the
largest power of two not exceeding ``DEFAULT_BLOCK_N`` (so ``inter`` smaller than
the block still costs one masked program per row, and the tails 13823/13825 fall
out of the same mask).
"""

from __future__ import annotations

import sys

import torch
import triton

from .kernels import DEFAULT_BLOCK_N, DEFAULT_NUM_WARPS, _gelu_and_mul_kernel

__all__ = ["gelu_and_mul"]

# int32 element addressing in the kernel: row * (2*inter) + inter + col < 2**31.
_MAX_ELEMS: int = 2**31

_device_probed = False


def _init_device_properties_triton() -> None:
    """Probe NPU device properties once before the first triton launch.

    Mirrors ``vllm_ascend/ops/triton/triton_utils.py:46-54`` (and the fi_sampling
    helper): on a fresh process the ascend backend wants the driver's device
    properties resolved before a bare kernel launch.  Best-effort -- real errors
    surface at launch time.
    """
    global _device_probed
    if _device_probed:
        return
    try:
        _ = triton.runtime.driver.active.utils.get_device_properties(
            torch.npu.current_device()
        )
    except Exception as exc:  # noqa: BLE001 - probe is best-effort
        print(f"[fi_gelu] device-property probe skipped: {exc!r}", file=sys.stderr)
    _device_probed = True


def _pick_block_n(inter: int) -> int:
    """Largest power of two <= DEFAULT_BLOCK_N, and <= inter (rounded up)."""
    if inter >= DEFAULT_BLOCK_N:
        return DEFAULT_BLOCK_N
    return min(DEFAULT_BLOCK_N, triton.next_power_of_2(inter))


def gelu_and_mul(
    input: torch.Tensor,
    *,
    out: torch.Tensor | None = None,
    block_n: int | None = None,
    num_warps: int = DEFAULT_NUM_WARPS,
) -> torch.Tensor:
    """``out = gelu_exact_erf(input[..., :d]) * input[..., d:]`` on NPU.

    Args:
        input: bf16 tensor with an even last dimension (``2 * inter``); any
            leading dims are flattened over rows.  Non-contiguous inputs are
            copied.
        out: optional pre-allocated bf16 output buffer (shape
            ``input.shape[:-1] + (inter,)``, contiguous, same device).  Worth
            supplying in a hot loop: on this stack an in-loop ``torch.empty`` /
            ``view`` costs ~30-45 us each while kernels are pending, versus
            ~9 us / ~4 us in isolation.
        block_n: optional in-row block override (must be a power of two).
        num_warps: triton launch knob (default ``DEFAULT_NUM_WARPS``).

    Returns:
        bf16 tensor ``input.shape[:-1] + (inter,)`` on the same device (``out``
        when supplied).

    Raises:
        TypeError: ``input`` (or ``out``) is not a bf16 tensor.
        ValueError: fewer than 2 dims, odd last dim, empty gate half, element
            count overflows int32 addressing, or ``out`` is malformed.
    """
    if not isinstance(input, torch.Tensor):
        raise TypeError(f"input must be a torch.Tensor, got {type(input).__name__}")
    if input.dtype != torch.bfloat16:
        raise TypeError(f"input must be bfloat16, got {input.dtype}")
    if input.dim() < 2:
        raise ValueError(
            f"input must be at least 2D [num_tokens, 2*inter], got {tuple(input.shape)}"
        )

    last = input.shape[-1]
    if last % 2 != 0:
        raise ValueError(f"last dim must be even (gate|up concat), got {last}")
    inter = last // 2
    if inter == 0:
        raise ValueError("gate half is empty (last dim == 0)")

    x = input if input.is_contiguous() else input.contiguous()
    num_tokens = x.numel() // last
    if num_tokens * 2 * inter >= _MAX_ELEMS:
        raise ValueError(
            f"num_tokens * 2 * inter = {num_tokens * 2 * inter} "
            "overflows int32 addressing"
        )

    out_shape = input.shape[:-1] + (inter,)
    if out is None:
        if num_tokens == 0:
            return torch.empty(out_shape, dtype=torch.bfloat16, device=input.device)
        out = torch.empty(
            (num_tokens, inter), dtype=torch.bfloat16, device=input.device
        )
    else:
        if not isinstance(out, torch.Tensor) or out.dtype != torch.bfloat16:
            raise TypeError(f"out must be a bfloat16 torch.Tensor, got {out!r}")
        if tuple(out.shape) != tuple(out_shape):
            raise ValueError(
                f"out must have shape {tuple(out_shape)}, got {tuple(out.shape)}"
            )
        if not out.is_contiguous():
            raise ValueError("out must be contiguous")
        if out.device != input.device:
            raise ValueError(f"out device {out.device} != input device {input.device}")
        if num_tokens == 0:
            return out

    x2d = x if x.dim() == 2 else x.reshape(num_tokens, 2 * inter)
    out2d = out if out.dim() == 2 else out.reshape(num_tokens, inter)

    bn = int(block_n) if block_n else _pick_block_n(inter)
    n_col_blocks = (inter + bn - 1) // bn

    _init_device_properties_triton()
    _gelu_and_mul_kernel[(num_tokens * n_col_blocks,)](
        x2d,
        out2d,
        inter,
        n_col_blocks,
        BLOCK_N=bn,
        num_warps=num_warps,
    )
    return out
