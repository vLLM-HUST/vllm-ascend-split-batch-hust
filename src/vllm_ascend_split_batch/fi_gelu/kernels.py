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

"""Triton-ascend ``gelu_and_mul`` kernel (flashinfer gelu semantics).

Semantics reference: flashinfer ``main@3a4e7052``,
``flashinfer/jit/activation.py:93`` -- the *exact erf* GELU, not the tanh
approximation:

    out = gelu(gate) * up                 gate = x[..., :d], up = x[..., d:]
    gelu(v) = v * 0.5 * (1 + erf(v * sqrt(1/2)))

which is also what vLLM's ``GeluAndMul.forward_native`` documents
(``approximate="none"``).  On this stack ``F.gelu`` is silently degraded to the
tanh form (probe: MARE 3.88e-03, ``knowledge/handoffs/gelu-fusion-kickoff.md``
§2.2), so the erf is computed explicitly here.

Layout: input ``[num_tokens, 2*inter]`` bf16 row-major, output
``[num_tokens, inter]`` bf16; bf16 in / fp32 compute / bf16 out.  The grid is
one program per ``(row, inter-block)`` pair so any ``inter`` is handled by
block masking (including the non-divisible tails 13823/13825).

``tl.erf`` is exact on this backend: the NPU probe (see
``_erf_probe_kernel``) returns ``maxAbs == 0.0`` against ``torch.erf`` in fp32,
i.e. it lowers to the same implementation aten uses.  ``libdevice.erf`` is NOT
usable here (the ascend backend fails to lower it -- verified by probe).
"""

from __future__ import annotations

import triton
import triton.language as tl

__all__ = ["SQRT1_2", "_erf_probe_kernel", "_gelu_and_mul_kernel"]

# sqrt(1/2), the gelu argument scale.  Wrapped as ``tl.constexpr`` so the kernel
# may close over it (triton rejects plain module globals in jitted code; the
# bare annotation form is not honoured by this build).
SQRT1_2 = tl.constexpr(0.7071067811865476)

# Default in-row block for the column (``inter``) dimension.  2048 was the best
# of {256, 512, 1024, 2048, 4096} in the device-bound sweep (large-B effective
# bandwidth 418 vs 320/354 GB/s at 1024/4096); at B=64/256 every block size is
# within launch-overhead noise.
DEFAULT_BLOCK_N: int = 2048
DEFAULT_NUM_WARPS: int = 4


@triton.jit
def _gelu_and_mul_kernel(
    in_ptr,  # [num_tokens, 2 * inter] bf16, row-major
    out_ptr,  # [num_tokens, inter] bf16, row-major
    inter,  # int32: gate/up half width
    n_col_blocks,  # int32: cdiv(inter, BLOCK_N)
    BLOCK_N: tl.constexpr,
):
    pid = tl.program_id(0)
    row = pid // n_col_blocks
    col_block = pid - row * n_col_blocks
    col = col_block * BLOCK_N + tl.arange(0, BLOCK_N)
    mask = col < inter
    base = row * (2 * inter)
    gate = tl.load(in_ptr + base + col, mask=mask, other=0.0).to(tl.float32)
    up = tl.load(in_ptr + base + inter + col, mask=mask, other=0.0).to(tl.float32)
    gelu = gate * 0.5 * (1.0 + tl.erf(gate * SQRT1_2))
    tl.store(out_ptr + row * inter + col, (gelu * up).to(tl.bfloat16), mask=mask)


@triton.jit
def _erf_probe_kernel(x_ptr, y_ptr, n, BLOCK: tl.constexpr):
    """Primitive acceptance probe: elementwise ``tl.erf`` in the fp32 domain.

    The fused kernel's whole numerical contract rests on ``tl.erf`` lowering to
    an exact fp32 erf on the ascend backend; this probe is the first test to run
    (see ``tests/test_npu_w31_gelu.py``) and gates every other NPU case.
    """
    offs = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
    mask = offs < n
    x = tl.load(x_ptr + offs, mask=mask, other=0.0)
    tl.store(y_ptr + offs, tl.erf(x), mask=mask)
