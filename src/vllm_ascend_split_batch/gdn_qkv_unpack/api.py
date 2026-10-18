# Copyright (c) 2026 vLLM-HUST contributors.
#
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

"""Host entry for the packed-GDN-qkv single-pass unpack (NPU only).

One launch copies the three row slices ``[T, q]``, ``[T, k]``, ``[T, v]`` of the
packed input into one fresh flat ``q | k | v`` buffer; the returned views share
that storage at the offsets ``torch.cat`` over the three splits produces.

``rearrange_mixed_qkv`` is the only function that allocates or launches: the
device check, the allocation and the launch all live there, in that order.
``_plan`` / ``_views`` are pure (no allocation, no kernel import when ``block``
is given) and are what the CPU tests drive.

``triton`` is imported on the first call only (``.kernels``).
"""

from __future__ import annotations

import importlib
from typing import NamedTuple

import torch

__all__ = ["rearrange_mixed_qkv"]

_SUPPORTED_DTYPES = (torch.bfloat16, torch.float16, torch.float32)
_MAX_ADDRESS: int = 2**31  # int32 element addressing
_MAX_MASK_INDEX: int = 2**24  # the kernel's fp32 column mask is exact below this
_MAX_PROGRAMS: int = 65535  # coreDim is uint16 on this backend


class _Plan(NamedTuple):
    """Launch geometry and view shapes derived from a validated input."""

    n_rows: int
    q_dim: int
    k_dim: int
    v_dim: int
    head_k_dim: int
    head_v_dim: int
    row_stride: int
    block: int
    n_col_tiles: int
    flat_elems: int

    @property
    def grid(self) -> tuple[int, int]:
        return (3, self.n_rows * self.n_col_tiles)


def _positive_int(name: str, value: object) -> int:
    if isinstance(value, bool) or not isinstance(value, int):
        raise TypeError(f"{name} must be a python int, got {type(value).__name__}")
    if value <= 0:
        raise ValueError(f"{name} must be positive, got {value}")
    return value


def _require_npu(mixed_qkv: torch.Tensor) -> None:
    if mixed_qkv.device.type != "npu":
        raise ValueError(
            "mixed_qkv must be on an NPU device (this operator has no CPU path), "
            f"got {mixed_qkv.device}"
        )


def _plan(
    mixed_qkv: torch.Tensor,
    *,
    q_dim: int,
    v_dim: int,
    head_k_dim: int,
    head_v_dim: int,
    k_dim: int | None = None,
    block: int | None = None,
) -> _Plan:
    """Validate the input and derive the launch geometry (no allocation)."""
    if mixed_qkv.dtype not in _SUPPORTED_DTYPES:
        raise TypeError(
            f"mixed_qkv must be bfloat16/float16/float32, got {mixed_qkv.dtype}"
        )
    if mixed_qkv.dim() != 2:
        raise ValueError(
            f"mixed_qkv must be 2D [seq_len, q_dim + k_dim + v_dim], "
            f"got {mixed_qkv.dim()}D {tuple(mixed_qkv.shape)}"
        )

    q_dim = _positive_int("q_dim", q_dim)
    v_dim = _positive_int("v_dim", v_dim)
    head_k_dim = _positive_int("head_k_dim", head_k_dim)
    head_v_dim = _positive_int("head_v_dim", head_v_dim)
    k_dim = q_dim if k_dim is None else _positive_int("k_dim", k_dim)

    packed_width = q_dim + k_dim + v_dim
    n_rows, last = mixed_qkv.shape
    row_stride = int(mixed_qkv.stride(0))
    if last != packed_width:
        raise ValueError(
            f"last dim must be q_dim + k_dim + v_dim = {packed_width}, got {last}"
        )
    if q_dim % head_k_dim != 0 or k_dim % head_k_dim != 0:
        raise ValueError(
            f"q_dim {q_dim} / k_dim {k_dim} not divisible by head_k_dim {head_k_dim}"
        )
    if v_dim % head_v_dim != 0:
        raise ValueError(f"v_dim {v_dim} not divisible by head_v_dim {head_v_dim}")
    if n_rows == 0:
        raise ValueError("seq_len == 0 is unsupported (host view(-1) is ambiguous)")
    if mixed_qkv.stride(-1) != 1:
        raise ValueError(f"last-dim stride must be 1, got {mixed_qkv.stride(-1)}")
    if row_stride < packed_width:
        raise ValueError(f"row stride {row_stride} < packed width {packed_width}")
    if max(q_dim, k_dim, v_dim) >= _MAX_MASK_INDEX:
        raise ValueError(
            f"q_dim/k_dim/v_dim must be < {_MAX_MASK_INDEX} (fp32 column mask)"
        )

    flat_elems = n_rows * packed_width
    last_input_elem = (n_rows - 1) * row_stride + packed_width
    if max(flat_elems, last_input_elem) >= _MAX_ADDRESS:
        raise ValueError(
            f"element span {max(flat_elems, last_input_elem)} >= {_MAX_ADDRESS}: "
            "flat buffer or input row span overflows int32 addressing"
        )

    if block is None:
        block = importlib.import_module(".kernels", __package__).DEFAULT_BLOCK
    block = _positive_int("block", block)
    n_col_tiles = -(-max(q_dim, k_dim, v_dim) // block)
    programs = 3 * n_rows * n_col_tiles
    if programs > _MAX_PROGRAMS:
        raise ValueError(
            f"grid {programs} > {_MAX_PROGRAMS} (coreDim); seq_len must be "
            f"<= {_MAX_PROGRAMS // (3 * n_col_tiles)} for these dims"
        )

    return _Plan(
        n_rows=n_rows,
        q_dim=q_dim,
        k_dim=k_dim,
        v_dim=v_dim,
        head_k_dim=head_k_dim,
        head_v_dim=head_v_dim,
        row_stride=row_stride,
        block=block,
        n_col_tiles=n_col_tiles,
        flat_elems=flat_elems,
    )


def _views(
    flat: torch.Tensor, plan: _Plan
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """The three ``(1, T, heads, head_dim)`` views over ``flat``."""
    q_size = plan.n_rows * plan.q_dim
    k_size = plan.n_rows * plan.k_dim
    query = flat[:q_size].view(
        1, plan.n_rows, plan.q_dim // plan.head_k_dim, plan.head_k_dim
    )
    key = flat[q_size : q_size + k_size].view(
        1, plan.n_rows, plan.k_dim // plan.head_k_dim, plan.head_k_dim
    )
    value = flat[q_size + k_size :].view(
        1, plan.n_rows, plan.v_dim // plan.head_v_dim, plan.head_v_dim
    )
    return query, key, value


def rearrange_mixed_qkv(
    mixed_qkv: torch.Tensor | None,
    *,
    q_dim: int,
    v_dim: int,
    head_k_dim: int,
    head_v_dim: int,
    k_dim: int | None = None,
) -> tuple[torch.Tensor | None, torch.Tensor | None, torch.Tensor | None]:
    """Split packed qkv ``[T, q + k + v]`` into ``(1, T, heads, head_dim)`` views.

    Args:
        mixed_qkv: NPU tensor ``[T, q_dim + k_dim + v_dim]``, bf16/fp16/fp32,
            last-dim stride 1, ``stride(0) >= q_dim + k_dim + v_dim`` (wider rows
            are fine, padding columns are never read); ``None`` returns
            ``(None, None, None)``.
        q_dim: q slice width (``key_dim // tp_size``).
        v_dim: v slice width (``value_dim // tp_size``).
        head_k_dim: q/k head width; divides ``q_dim`` and ``k_dim``.
        head_v_dim: v head width; divides ``v_dim``.
        k_dim: k slice width; defaults to ``q_dim`` (the host splits
            ``[key_dim, key_dim, value_dim]``).

    Returns:
        ``(query, key, value)`` of shape ``(1, T, q_dim // head_k_dim,
        head_k_dim)``, ``(1, T, k_dim // head_k_dim, head_k_dim)`` and
        ``(1, T, v_dim // head_v_dim, head_v_dim)`` over one flat storage.

    Raises:
        TypeError: not a tensor, dtype not bf16/fp16/fp32, or a dimension
            argument is not a python int.
        ValueError: not on an NPU device, unsupported shape/stride/head
            geometry, ``T == 0`` (the host chain raises on ``view(-1)``), or the
            int32 / fp32-mask / coreDim limits are exceeded.
    """
    if mixed_qkv is None:
        return None, None, None
    if not isinstance(mixed_qkv, torch.Tensor):
        raise TypeError(
            f"mixed_qkv must be a torch.Tensor or None, got {type(mixed_qkv).__name__}"
        )
    _require_npu(mixed_qkv)

    plan = _plan(
        mixed_qkv,
        q_dim=q_dim,
        v_dim=v_dim,
        head_k_dim=head_k_dim,
        head_v_dim=head_v_dim,
        k_dim=k_dim,
    )
    flat = torch.empty(plan.flat_elems, dtype=mixed_qkv.dtype, device=mixed_qkv.device)
    kernels = importlib.import_module(".kernels", __package__)
    kernels.init_device_properties_triton()
    kernels._unpack_mixed_qkv_kernel[plan.grid](
        mixed_qkv,
        flat,
        plan.n_rows,
        plan.q_dim,
        plan.k_dim,
        plan.v_dim,
        plan.row_stride,
        plan.n_col_tiles,
        BLOCK=plan.block,
    )
    return _views(flat, plan)
