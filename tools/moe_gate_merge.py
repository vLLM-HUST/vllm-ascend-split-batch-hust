#!/usr/bin/env python3
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

"""MoE 双门合并（router + shared_expert_gate）的 CPU 参考实现与语义契约。

背景（`docs/d7-matmulv2-decode-status.md` §4）: MoE 层里 `self.gate`
(GateLinear 2048→256) 与 `self.shared_expert_gate` (ReplicatedLinear 2048→1)
消费**同一张** MoE 输入 x（`qwen2_moe.py` L121
``out = F.sigmoid(self.expert_gate(x)[0]) * out``），因此
``cat([W_router, W_shared], 0)`` = 257×2048 一次 GEMM + 两路切片语义成立，
可省一次 kernel 发车。

本模块只做三件事（不 import torch，纯 Python，CPU CI 可跑）:

1. ``merge_gate_weights`` —— 行序契约：前 ``n_router`` 行是 router，最后 1 行是 shared；
2. ``fused_outputs``     —— 单次 matvec 后按行切片，两路都**返回拷贝**
   （对应 device 侧 router logits 进 top-k 前必须 ``.contiguous()``）；
3. ``reference_outputs`` —— 两次独立 matvec 的参考路径，用于等价性核对。

设备侧落地形状（尚未实现；候选 ``router-shared-gate-merge``）: 加载期一次性
构造 merged 权重（**不改 checkpoint 键**，原两个 Linear 参数保留以便回滚），
forward 里一次 ``F.linear(x, merged)`` 再切两路；env 门 default-off。

边界：本模块只证**行序、切片协议与算术等价**；bf16 设备 GEMM 的重结合与实测收益
都不在本模块范围内（属 L0/L3）。
"""

from __future__ import annotations

import math

#: 行序契约。改这个常量等于改契约，生产侧与消费侧必须同时改。
MERGED_LAYOUT = ("router", "shared_gate")


def _rows(weights, n_cols: int, name: str) -> list[list[float]]:
    rows = [list(r) for r in weights]
    if not rows:
        raise ValueError(f"{name}: 权重不能为空")
    for i, row in enumerate(rows):
        if len(row) != n_cols:
            raise ValueError(f"{name}[{i}]: 列数 {len(row)} != {n_cols}")
    return rows


def _dot(x_row, w_row) -> float:
    """定序累加（k 升序）；合并路径与参考路径用同一实现 → 等价可断言到精确相等。"""
    return math.fsum(x_row[k] * w_row[k] for k in range(len(x_row)))


class MergedGateWeights:
    """拼好的门权重（行序由 ``MERGED_LAYOUT`` 决定）。"""

    __slots__ = ("rows", "n_router", "n_shared", "n_cols")

    def __init__(self, rows, n_router: int, n_shared: int, n_cols: int) -> None:
        self.rows = rows
        self.n_router = n_router
        self.n_shared = n_shared
        self.n_cols = n_cols

    @property
    def shape(self) -> tuple[int, int]:
        return (len(self.rows), self.n_cols)

    def router_rows(self) -> list[list[float]]:
        return self.rows[: self.n_router]

    def shared_rows(self) -> list[list[float]]:
        return self.rows[self.n_router :]


def merge_gate_weights(w_router, w_shared) -> MergedGateWeights:
    """按 ``MERGED_LAYOUT`` 拼行；列数必须一致，shared gate 必须恰好 1 行。"""
    if MERGED_LAYOUT != ("router", "shared_gate"):
        raise AssertionError(f"未支持的布局 {MERGED_LAYOUT}")
    n_cols = len(w_router[0]) if w_router else 0
    router = _rows(w_router, n_cols, "w_router")
    shared = _rows(w_shared, n_cols, "w_shared")
    if len(shared) != 1:
        raise ValueError(f"shared gate 应只有 1 行（标量门），收到 {len(shared)}")
    return MergedGateWeights(router + shared, len(router), len(shared), n_cols)


def fused_outputs(x, merged: MergedGateWeights):
    """单次 matvec + 两路切片；返回 ``(logits, shared)``，两者都是拷贝。"""
    logits: list[list[float]] = []
    shared: list[list[float]] = []
    for x_row in x:
        if len(x_row) != merged.n_cols:
            raise ValueError(f"x 列数 {len(x_row)} != 权重列数 {merged.n_cols}")
        out = [_dot(x_row, w) for w in merged.rows]
        logits.append(list(out[: merged.n_router]))
        shared.append(list(out[merged.n_router :]))
    return logits, shared


def reference_outputs(x, w_router, w_shared):
    """两次独立 matvec 的参考路径（未合并）。"""
    n_cols = len(w_router[0]) if w_router else 0
    router = _rows(w_router, n_cols, "w_router")
    shared = _rows(w_shared, n_cols, "w_shared")
    logits: list[list[float]] = []
    gate: list[list[float]] = []
    for x_row in x:
        if len(x_row) != n_cols:
            raise ValueError(f"x 列数 {len(x_row)} != 权重列数 {n_cols}")
        logits.append([_dot(x_row, w) for w in router])
        gate.append([_dot(x_row, w) for w in shared])
    return logits, gate


def max_abs_diff(a, b) -> float:
    """展平后最大绝对差；结构不匹配直接抛错（不能被当成 0）。"""
    if len(a) != len(b):
        raise ValueError(f"行数不同: {len(a)} vs {len(b)}")
    worst = 0.0
    for row_a, row_b in zip(a, b, strict=True):
        if len(row_a) != len(row_b):
            raise ValueError(f"列数不同: {len(row_a)} vs {len(row_b)}")
        for va, vb in zip(row_a, row_b, strict=True):
            worst = max(worst, abs(va - vb))
    return worst


def equivalent(x, w_router, w_shared):
    """合并路径与参考路径是否逐元素相等 → ``(ok, (d_logits, d_shared))``。"""
    merged = merge_gate_weights(w_router, w_shared)
    got_logits, got_shared = fused_outputs(x, merged)
    ref_logits, ref_shared = reference_outputs(x, w_router, w_shared)
    d_logits = max_abs_diff(got_logits, ref_logits)
    d_shared = max_abs_diff(got_shared, ref_shared)
    return (d_logits == 0.0 and d_shared == 0.0), (d_logits, d_shared)


def device_plan(n_router: int, n_cols: int) -> dict:
    """设备侧落地要点（供 core 补丁与 L3 前核对；本函数不含实现）。"""
    return {
        "merged_shape": [n_router + 1, n_cols],
        "layout": list(MERGED_LAYOUT),
        "logits_slice": f"out[:, :{n_router}]",
        "shared_slice": f"out[:, {n_router}:]",
        "requires_contiguous": [
            "router_logits（进 top-k 前）",
            "shared_gate（进 sigmoid 前）",
        ],
        "checkpoint_keys": "不变（保留原两个 Linear 参数，merged 由加载期缓冲构造）",
        "rollback": "env 门 default-off；关闭即走原两次 GEMM",
        "not_verified_by_this_module": [
            "bf16 设备 GEMM 的重结合（属 L0/L3；CPU 参考只证行序与切片协议）",
            "257 行 GEMM 的实测收益（属 L0 配对 A/B）",
        ],
    }


if __name__ == "__main__":  # pragma: no cover - 人工自检入口
    import json

    print(json.dumps(device_plan(256, 2048), ensure_ascii=False, indent=1))
