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

"""Test oracles for the fused ``gelu_and_mul`` kernel.

Two references, deliberately independent of the kernel:

``gelu_erf`` / ``gelu_and_mul_reference``
    The fp32 golden, written as the explicit erf formula
    ``v * 0.5 * (1 + erf(v * sqrt(1/2)))`` (flashinfer
    ``activation.py:93``).  ``torch.erf`` is used directly -- ``F.gelu`` is NOT
    used because on this stack it is degraded to the tanh approximation
    (MARE 3.88e-03 vs the erf form, kickoff §2.2) and would silently bless the
    wrong semantics.

``npu_gelu_reference`` / ``gelu_and_mul_npu_gelu_reference``
    Second oracle: ``torch_npu.npu_gelu``, which the kickoff measured as
    exact-erf (maxAbs 4.77e-07 vs the erf formula) and is therefore a candidate
    for the host chain -- but ``npu_gelu_mul`` is unusable (last dim <= 1024
    limit), hence the two-step chain.  Kept in the fp32 domain so its own error
    budget stays visible.

Both references accept any leading dims and are CPU-safe except the ``npu_gelu``
pair, which requires an NPU (the import is lazy).
"""

from __future__ import annotations

import torch

__all__ = [
    "gelu_and_mul_npu_gelu_reference",
    "gelu_and_mul_reference",
    "gelu_erf",
    "npu_gelu_reference",
    "split_gate_up",
]

_SQRT1_2 = 0.7071067811865476


def split_gate_up(x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """vLLM/flashinfer slicing: gate is the first half, up the second half."""
    inter = x.shape[-1] // 2
    return x[..., :inter], x[..., inter:]


def gelu_erf(x: torch.Tensor) -> torch.Tensor:
    """Exact-erf GELU in fp32: ``v * 0.5 * (1 + erf(v * sqrt(1/2)))``."""
    v = x.to(torch.float32)
    return v * 0.5 * (1.0 + torch.erf(v * _SQRT1_2))


def gelu_and_mul_reference(
    input: torch.Tensor, dtype: torch.dtype = torch.float32
) -> torch.Tensor:
    """fp32 golden: ``gelu_erf(gate) * up`` in fp32, optionally cast to ``dtype``."""
    gate, up = split_gate_up(input)
    out = gelu_erf(gate) * up.to(torch.float32)
    return out.to(dtype)


def npu_gelu_reference(gate: torch.Tensor) -> torch.Tensor:
    """``torch_npu.npu_gelu`` in the fp32 domain (second oracle)."""
    import torch_npu  # noqa: PLC0415 - lazy: keeps this module CPU-importable

    return torch_npu.npu_gelu(gate.to(torch.float32))


def gelu_and_mul_npu_gelu_reference(
    input: torch.Tensor, dtype: torch.dtype = torch.float32
) -> torch.Tensor:
    """Second oracle: ``npu_gelu(gate) * up`` in fp32, optionally cast."""
    gate, up = split_gate_up(input)
    out = npu_gelu_reference(gate) * up.to(torch.float32)
    return out.to(dtype)
