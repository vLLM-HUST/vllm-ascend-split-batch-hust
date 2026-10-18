# Copyright (c) 2026 Huawei Technologies Co., Ltd. All Rights Reserved.
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

"""Golden: the host ``rearrange_mixed_qkv`` chain, verbatim.

Code derived from ``QwenGatedDeltaNetAttention.rearrange_mixed_qkv`` in the
pinned host tree (``vllm-hust``, ``vllm/model_executor/layers/mamba/gdn/
qwen_gdn_linear_attn.py``): ``torch.split`` -> three ``reshape(-1)`` ->
``torch.cat`` -> three ``view``.  Oracle for values and for the storage offsets
and strides of the returned views.  Torch only, CPU safe.
"""

from __future__ import annotations

import torch

__all__ = ["rearrange_mixed_qkv_reference"]


def rearrange_mixed_qkv_reference(
    mixed_qkv: torch.Tensor | None,
    *,
    q_dim: int,
    v_dim: int,
    head_k_dim: int,
    head_v_dim: int,
    k_dim: int | None = None,
) -> tuple[torch.Tensor | None, torch.Tensor | None, torch.Tensor | None]:
    """Host-chain split of packed qkv into ``(1, T, heads, head_dim)`` views."""
    if mixed_qkv is None:
        return None, None, None

    seq_len = mixed_qkv.shape[0]
    k_dim = q_dim if k_dim is None else k_dim

    query, key, value = torch.split(mixed_qkv, [q_dim, k_dim, v_dim], dim=-1)

    fused = torch.cat([query.reshape(-1), key.reshape(-1), value.reshape(-1)], dim=0)

    q_size = seq_len * q_dim
    k_size = seq_len * k_dim

    query = fused[0:q_size].view(1, seq_len, -1, head_k_dim)
    key = fused[q_size : q_size + k_size].view(1, seq_len, -1, head_k_dim)
    value = fused[q_size + k_size :].view(1, seq_len, -1, head_v_dim)
    return query, key, value
