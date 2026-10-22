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

"""MoE 双门合并的运行时载体（router + shared_expert_gate）。

语义依据（`docs/d7-matmulv2-decode-status.md` §4）：MoE 层里
``self.gate``(2048→256) 与 ``self.shared_expert_gate``(2048→1) 消费**同一张**
MoE 输入（`qwen2_moe.py` L121 ``out = F.sigmoid(self.expert_gate(x)[0]) * out``），
因此 ``cat([W_router, W_shared], 0)`` = 257×2048 一次 GEMM + 两路 contiguous
切片在语义上成立，可省一次 kernel 发车。

落地方式（由 `runtime_source_patches` 的锚点变换安装，env 门 default-off）:

* 安装点在 ``Qwen3NextSparseMoeBlock.forward`` 开头（幂等）：把 **router 消费者**
  （vllm_ascend 的 ``expert.gate`` / ``expert._gate``）与 **shared 消费者**
  （``shared_expert.expert_gate``）替换成代理，代理共享同一个 `GateMergeState`；
* 一次 layer 调用内两个消费者拿到的是**同一个张量对象**，状态按对象身份缓存 →
  每层只跑一次合并 GEMM；身份不成立时**重算**（正确性优先，并记 ``recomputes``）；
* 两路输出都 ``contiguous``（router logits 进 top-k、shared gate 进 sigmoid 前的要求）。

数值边界（必须保留）：若宿主 gate 暴露 ``weight_fp32``（vllm_ascend 的
``AscendUnquantizedLinearMethod`` 预铸 fp32 权重），stock 路由走 **fp32** GEMM；
本载体据同一判据选择 fp32 合并权重并返回 fp32 logits，shared 列则转回输入 dtype，
以保持 ``sigmoid(gate) * out`` 的 dtype 不被提升。

回滚：``VLLM_HUST_MOE_GATE_MERGE=0``（默认）→ 变换不安装、代理不创建、宿主原样。

本模块顶层**不 import torch**（CPU CI 可导入）；torch 只在设备侧函数内延迟导入。
"""

from __future__ import annotations

import logging
import os

logger = logging.getLogger(__name__)

ENV = "VLLM_HUST_MOE_GATE_MERGE"
_FP32_HINT = "weight_fp32"


def enabled(environ=None) -> bool:
    """env 门：只有显式 ``=1`` 才开（default-off）。"""
    return (environ if environ is not None else os.environ).get(ENV, "0") == "1"


def split_contiguous(out, n_router: int):
    """按行序契约切两路，**两路都 contiguous**（router 在前、shared gate 在末列）。"""
    total = out.shape[1]
    logits = out.narrow(1, 0, n_router).contiguous()
    shared = out.narrow(1, n_router, total - n_router).contiguous()
    return logits, shared


class TorchOps:
    """设备侧算子集合（可注入，测试用 fake 替换）。"""

    def __init__(self) -> None:
        import torch
        import torch.nn.functional as F

        self.torch = torch
        self._linear = F.linear

    def linear(self, x, weight):
        return self._linear(x, weight)

    def concat(self, tensors):
        return self.torch.cat(list(tensors), dim=0)

    def to_dtype(self, tensor, dtype):
        return tensor.to(dtype)

    def float32(self):
        return self.torch.float32


class GateMergeState:
    """一次合并 GEMM 的两个输出，按输入张量身份缓存。"""

    def __init__(self, router_weight, shared_weight, ops, *, fp32: bool = False):
        self._ops = ops
        self.fp32 = fp32
        self.n_router = int(router_weight.shape[0])
        self.weight = ops.concat([router_weight, shared_weight])
        self._last_input = None
        self._logits = None
        self._shared = None
        self.calls = 0
        self.recomputes = 0

    def outputs(self, hidden_states):
        if self._last_input is hidden_states and self._logits is not None:
            return self._logits, self._shared
        if self._last_input is not None:
            self.recomputes += 1
        x_in = (
            self._ops.to_dtype(hidden_states, self._ops.float32())
            if self.fp32
            else hidden_states
        )
        out = self._ops.linear(x_in, self.weight)
        logits, shared = split_contiguous(out, self.n_router)
        if self.fp32:
            # stock 路由在 fp32 预铸权重下返回 fp32 logits; shared 列转回输入 dtype
            shared = self._ops.to_dtype(shared, hidden_states.dtype)
        self._last_input = hidden_states
        self._logits = logits
        self._shared = shared
        self.calls += 1
        return logits, shared

    def reset(self) -> None:
        self._last_input = None
        self._logits = None
        self._shared = None


class RouterGateProxy:
    """vllm_ascend 消费形态: ``gate(hidden_states)`` → logits。"""

    def __init__(self, state: GateMergeState) -> None:
        self.state = state

    def __call__(self, hidden_states):
        return self.state.outputs(hidden_states)[0]


class SharedGateProxy:
    """core 消费形态: ``expert_gate(x)[0]`` → 标量门（shared 列）。"""

    def __init__(self, state: GateMergeState) -> None:
        self.state = state

    def __call__(self, x):
        return (self.state.outputs(x)[1], None)


def _slot(obj, name):
    return getattr(obj, name, None)


def _collect_consumers(block):
    """定位两个消费者；缺任一路即返回 None（fail-closed）。"""
    experts = _slot(block, "experts")
    router_slots = [
        (experts, attr)
        for attr in ("gate", "_gate")
        if experts is not None and _slot(experts, attr) is not None
    ]
    shared_slots = []
    shared_expert = _slot(block, "shared_expert")
    if shared_expert is not None and _slot(shared_expert, "expert_gate") is not None:
        shared_slots.append((shared_expert, "expert_gate"))
    if experts is not None and _slot(experts, "shared_expert_gate") is not None:
        shared_slots.append((experts, "shared_expert_gate"))
    if not router_slots or not shared_slots:
        return None
    return router_slots, shared_slots


def install_gate_merge(block, *, environ=None, ops=None) -> bool:
    """幂等安装；env 关闭 / 消费者不全 → 不安装并返回 False（回滚即关闭 env）。"""
    if not enabled(environ):
        return False
    if getattr(block, "_hust_gate_merge", None) is not None:
        return False
    consumers = _collect_consumers(block)
    if consumers is None:
        logger.warning(
            "moe_gate_merge: 消费者不全, 不安装（router/shared 需同时可替换）"
        )
        block._hust_gate_merge = False  # 只装一次失败也不再重试
        return False
    router_weight_owner = _slot(block, "gate")
    shared_weight_owner = _slot(block, "shared_expert_gate")
    if router_weight_owner is None or shared_weight_owner is None:
        logger.warning("moe_gate_merge: 缺 gate / shared_expert_gate 权重, 不安装")
        block._hust_gate_merge = False
        return False

    fp32 = hasattr(router_weight_owner, _FP32_HINT) or hasattr(
        shared_weight_owner, _FP32_HINT
    )
    router_weight = (
        router_weight_owner.weight_fp32 if fp32 else router_weight_owner.weight
    )
    shared_weight = (
        shared_weight_owner.weight_fp32 if fp32 else shared_weight_owner.weight
    )
    state = GateMergeState(router_weight, shared_weight, ops or TorchOps(), fp32=fp32)
    router_proxy = RouterGateProxy(state)
    shared_proxy = SharedGateProxy(state)

    router_slots, shared_slots = consumers
    for owner, attr in router_slots:
        setattr(owner, attr, router_proxy)
    for owner, attr in shared_slots:
        setattr(owner, attr, shared_proxy)
    block._hust_gate_merge = state
    logger.info(
        "moe_gate_merge: installed (n_router=%d, fp32=%s, "
        "router_slots=%d, shared_slots=%d)",
        state.n_router,
        fp32,
        len(router_slots),
        len(shared_slots),
    )
    return True


def describe() -> dict:
    """自检/文档用：载体形状、边界、回滚方式。"""
    return {
        "env": ENV,
        "default": "off（未设为 1 时不安装任何代理）",
        "layout": ["router(前 n_router 行)", "shared_gate(末行)"],
        "merged_shape": "[n_router+1, hidden]",
        "single_gemm_per_layer_call": (
            "两个消费者共享同一输入对象 → 只跑一次; 身份不成立则重算"
        ),
        "contiguous": ["router logits", "shared gate"],
        "fp32_rule": (
            "宿主 gate 暴露 weight_fp32 时用 fp32 合并权重; shared 列转回输入 dtype"
        ),
        "rollback": f"{ENV}=0（默认）→ 宿主原样，无残留代理",
        "not_verified": [
            "设备侧收益（L0 配对 A/B）",
            "bf16 重结合下的数值一致（L3 + compare）",
        ],
    }
