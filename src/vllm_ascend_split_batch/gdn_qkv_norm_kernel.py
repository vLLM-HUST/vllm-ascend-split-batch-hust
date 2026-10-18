# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
# SPDX-FileCopyrightText: Songlin Yang, Yu Zhang
# Copyright (c) 2026 vLLM-HUST contributors.
# L2Norm arithmetic derived from flash-linear-attention (MIT).

import triton
import triton.language as tl


@triton.jit
def unpack_norm_kernel(X, Y):
    segment = tl.program_id(0)
    token = tl.program_id(1)
    if segment < 2:
        head = tl.arange(0, 8)[:, None]
        dim = tl.arange(0, 128)[None, :]
        offset = head * 128 + dim
        x = tl.load(X + token * 4096 + segment * 1024 + offset)
        xf = x.to(tl.float32)
        norm = tl.rsqrt(tl.sum(xf * xf, 1) + 1.0e-6)
        tl.store(Y + segment * 16384 + token * 1024 + offset,
                 xf * norm[:, None])
    else:
        v_offset = tl.arange(0, 2048)
        value = tl.load(X + token * 4096 + 2048 + v_offset)
        tl.store(Y + 32768 + token * 2048 + v_offset, value)

