# Copyright (c) 2026 Huawei Technologies Co., Ltd. All Rights Reserved.
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

"""decode M=1 init-routing triton 核（独立模块: 避免 jit 全局扫描污染宿主）。"""

import torch
import triton
import triton.language as tl


@triton.jit
def _moe_init_routing_t1_kernel(
    x_ptr, ids_ptr,
    sorted_ptr, rowidx_ptr, counts_ptr,
    H: tl.constexpr, E: tl.constexpr, K: tl.constexpr,
):
    kidx = tl.arange(0, K)
    ids = tl.load(ids_ptr + kidx).to(tl.int32)
    xrow = tl.load(x_ptr + tl.arange(0, H)).to(x_ptr.dtype.element_ty)
    sblock = tl.broadcast_to(xrow[None, :], (K, H))
    tl.store(sorted_ptr + kidx[:, None] * H + tl.arange(0, H)[None, :], sblock)
    iv = ids[:, None].to(tl.int32)
    less = tl.sum((ids[None, :] < iv).to(tl.int32), axis=1)
    sidx = kidx[None, :]
    ties_before = tl.sum(((ids[None, :] == iv) & (sidx < kidx[:, None])).to(tl.int32), axis=1)
    tl.store(rowidx_ptr + kidx, (less + ties_before).to(tl.int32))
    counts = tl.zeros((E,), tl.int32)
    for j in tl.static_range(K):
        e_j = tl.sum(tl.where(kidx == j, ids, 0), axis=0)
        counts += (tl.arange(0, E) == e_j.to(tl.int32)).to(tl.int32)
    tl.store(counts_ptr + tl.arange(0, E), counts.to(counts_ptr.dtype.element_ty))


def moe_init_routing_t1(hidden_states, topk_ids, expert_num: int):
    """契约等价 npu_moe_init_routing_v2 (M=1): sorted/rowidx/expert_tokens(+scale=None)。

    env 门在函数内: VLLM_HUST_MOE_INIT_TRITON!=1 时走 stock 核（default-off,
    dispatcher 侧分支只按形状路由, 不读 env）。
    """
    import os  # noqa: PLC0415

    if os.environ.get("VLLM_HUST_MOE_INIT_TRITON") != "1":
        from vllm_ascend.device.device_op import DeviceOperator  # noqa: PLC0415

        return DeviceOperator.npu_moe_init_routing(
            hidden_states,
            topk_ids,
            scale=None,
            active_num=topk_ids.numel(),
            expert_num=expert_num,
            expert_tokens_num_type=1,
            expert_tokens_num_flag=True,
            active_expert_range=[0, expert_num],
            quant_mode=-1,
        )
    k = topk_ids.shape[1]
    h = hidden_states.shape[1]
    sorted_t = torch.empty((k, h), dtype=hidden_states.dtype, device=hidden_states.device)
    rowidx = torch.empty((k,), dtype=torch.int32, device=hidden_states.device)
    counts = torch.zeros((expert_num,), dtype=torch.int64, device=hidden_states.device)
    _moe_init_routing_t1_kernel[(1,)](
        hidden_states, topk_ids, sorted_t, rowidx, counts,
        H=h, E=expert_num, K=k, num_warps=4,
    )
    return sorted_t, rowidx, counts, None
