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

"""Triton-ascend packed-GDN-qkv unpack kernel.

Pure ``load``/``store`` copy: input row ``r`` holds ``[q | k | v]`` contiguously
at ``r * mixed_row_stride`` (last-dim stride 1), output is one flat buffer laid
out ``[n_rows, q]`` ++ ``[n_rows, k]`` ++ ``[n_rows, v]``.

``grid = (3, n_rows * n_col_tiles)``: ``program_id(0)`` is the segment and picks
``(width, src offset, dst offset)`` through scalar integer arithmetic, so each
program issues one contiguous masked load and one contiguous masked store.
``n_col_tiles`` comes from the widest segment; the narrower ones carry
all-false-mask programs (no store).
"""

from __future__ import annotations

import sys

import triton
import triton.language as tl

__all__ = [
    "DEFAULT_BLOCK",
    "init_device_properties_triton",
    "_unpack_mixed_qkv_kernel",
]

#: Single preset, not autotuned.
DEFAULT_BLOCK: int = 4096

_device_probed = False


def init_device_properties_triton() -> None:
    """Resolve ascend device properties once before the first launch.

    Same probe as ``vllm_ascend/ops/triton/triton_utils.py:46-54``; best-effort,
    real errors surface at launch time.
    """
    global _device_probed
    if _device_probed:
        return
    import torch

    try:
        _ = triton.runtime.driver.active.utils.get_device_properties(
            torch.npu.current_device()
        )
    except Exception as exc:  # noqa: BLE001 - probe is best-effort
        print(
            f"[gdn_qkv_unpack] device-property probe skipped: {exc!r}", file=sys.stderr
        )
    _device_probed = True


@triton.jit
def _unpack_mixed_qkv_kernel(
    mixed_ptr,  # [n_rows, >= q + k + v], last-dim stride 1
    flat_ptr,  # [n_rows * (q + k + v)], laid out q | k | v
    n_rows,  # int32
    q_dim,  # int32
    k_dim,  # int32
    v_dim,  # int32
    mixed_row_stride,  # int32: elements between input rows
    n_col_tiles,  # int32: ceil(max(q_dim, k_dim, v_dim) / BLOCK)
    BLOCK: tl.constexpr,
):
    seg = tl.program_id(0)
    is_k = (seg == 1).to(tl.int32)
    is_v = (seg == 2).to(tl.int32)
    is_q = 1 - is_k - is_v

    width = is_q * q_dim + is_k * k_dim + is_v * v_dim
    q_size = n_rows * q_dim
    k_size = n_rows * k_dim
    src_row_base = is_k * q_dim + is_v * (q_dim + k_dim)
    dst_row_base = is_k * q_size + is_v * (q_size + k_size)

    pid = tl.program_id(1)
    row = pid // n_col_tiles
    col = (pid - row * n_col_tiles) * BLOCK + tl.arange(0, BLOCK)
    mask = col < width

    src = mixed_ptr + row * mixed_row_stride + src_row_base + col
    dst = flat_ptr + dst_row_base + row * width + col
    tl.store(dst, tl.load(src, mask=mask, other=0), mask=mask)
