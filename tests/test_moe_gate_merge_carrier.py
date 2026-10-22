# SPDX-License-Identifier: Apache-2.0

"""MoE 双门合并的**运行时载体**测试（纯 CPU，不 import torch）。

覆盖:
1. env 门 default-off（未设 / 非 "1" 不安装）；
2. 一次 layer 调用只跑一次合并 GEMM（两消费者共享同一输入对象）；
3. 输入对象不同则重算（正确性优先, 并计数）;
4. 两路输出都 contiguous（切片产生新对象, 不是父张量的视图）;
5. fp32 预铸权重时 logits 为 fp32、shared 列转回输入 dtype;
6. fail-closed: 缺 shared 消费者时**不安装**、且 router 槽位保持原样。
"""

import importlib.util
import math
import pathlib

import pytest

HERE = pathlib.Path(__file__).resolve().parent
SPEC = importlib.util.spec_from_file_location(
    "moe_gate_merge_carrier",
    HERE.parent / "src" / "vllm_ascend_split_batch" / "moe_gate_merge.py",
)
carrier = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(carrier)

N_ROUTER = 256
K = 8


class FakeTensor:
    """够用的迷你张量：shape / narrow / contiguous / to / dtype。"""

    def __init__(self, rows, dtype="bf16"):
        self.rows = [list(r) for r in rows]
        self.dtype = dtype

    @property
    def shape(self):
        return (len(self.rows), len(self.rows[0]) if self.rows else 0)

    def narrow(self, dim, start, length):
        assert dim == 1
        return FakeTensor([r[start : start + length] for r in self.rows], self.dtype)

    def contiguous(self):
        return FakeTensor(self.rows, self.dtype)  # 新对象 == 拷贝语义

    def to(self, dtype):
        return FakeTensor(self.rows, dtype)


class FakeOps:
    def __init__(self):
        self.linear_calls = 0
        self.concat_calls = 0

    def linear(self, x, weight):
        self.linear_calls += 1
        out = []
        for x_row in x.rows:
            out.append(
                [
                    math.fsum(x_row[k] * w_row[k] for k in range(len(x_row)))
                    for w_row in weight.rows
                ]
            )
        return FakeTensor(out, x.dtype)

    def concat(self, tensors):
        self.concat_calls += 1
        return FakeTensor([r for t in tensors for r in t.rows])

    def to_dtype(self, tensor, dtype):
        return tensor.to(dtype)

    def float32(self):
        return "fp32"


def _weights(n_rows, cols, value=1.0):
    return FakeTensor([[value] * cols for _ in range(n_rows)])


def _make_block(*, with_shared=True, fp32_hint=False):
    class Mod:
        pass

    block = Mod()
    block.gate = Mod()
    block.gate.weight = _weights(N_ROUTER, K)
    block.shared_expert_gate = Mod()
    block.shared_expert_gate.weight = _weights(1, K, 0.5)
    if fp32_hint:
        block.gate.weight_fp32 = _weights(N_ROUTER, K)
        block.shared_expert_gate.weight_fp32 = _weights(1, K, 0.5)
    block.experts = Mod()
    block.experts.gate = Mod()  # 占位（真实对象是 GateLinear）
    block.experts._gate = block.experts.gate
    block.shared_expert = Mod()
    block.shared_expert.expert_gate = block.shared_expert_gate if with_shared else None
    return block


def test_env_gate_is_default_off():
    assert carrier.enabled({}) is False
    assert carrier.enabled({carrier.ENV: "0"}) is False
    assert carrier.enabled({carrier.ENV: "true"}) is False  # 只认 "1"
    assert carrier.enabled({carrier.ENV: "1"}) is True


def test_install_is_skipped_when_env_off():
    block = _make_block()
    ops = FakeOps()
    assert carrier.install_gate_merge(block, environ={}, ops=ops) is False
    assert not hasattr(block, "_hust_gate_merge")
    assert ops.linear_calls == 0


def test_install_replaces_both_consumers_and_is_idempotent():
    block = _make_block()
    ops = FakeOps()
    assert carrier.install_gate_merge(block, environ={carrier.ENV: "1"}, ops=ops)
    assert isinstance(block.experts.gate, carrier.RouterGateProxy)
    assert isinstance(block.experts._gate, carrier.RouterGateProxy)
    assert isinstance(block.shared_expert.expert_gate, carrier.SharedGateProxy)
    # 权重按契约拼接（257 行）
    assert block._hust_gate_merge.weight.shape == (N_ROUTER + 1, K)
    assert block._hust_gate_merge.n_router == N_ROUTER
    # 幂等
    ok = carrier.install_gate_merge(block, environ={carrier.ENV: "1"}, ops=ops)
    assert ok is False


def test_one_gemm_for_two_consumers_sharing_the_same_input():
    block = _make_block()
    ops = FakeOps()
    carrier.install_gate_merge(block, environ={carrier.ENV: "1"}, ops=ops)
    state = block._hust_gate_merge
    x = FakeTensor([[1.0] * K, [2.0] * K])
    logits = block.experts.gate(x)  # vllm_ascend 形态
    gate = block.shared_expert.expert_gate(x)[0]  # core MLP 形态
    assert ops.linear_calls == 1  # 同一对象 → 只跑一次
    assert logits.shape == (2, N_ROUTER)
    assert gate.shape == (2, 1)
    assert logits.rows[0][0] == K * 1.0
    assert gate.rows[1][0] == 2.0 * K * 0.5
    assert state.calls == 1 and state.recomputes == 0


def test_different_input_object_recomputes():
    block = _make_block()
    ops = FakeOps()
    carrier.install_gate_merge(block, environ={carrier.ENV: "1"}, ops=ops)
    state = block._hust_gate_merge
    x1 = FakeTensor([[1.0] * K])
    x2 = FakeTensor([[1.0] * K])  # 值相同但对象不同
    block.experts.gate(x1)
    block.shared_expert.expert_gate(x2)
    assert ops.linear_calls == 2
    assert state.recomputes == 1


def test_slices_are_contiguous_copies_not_views():
    src = FakeTensor([[float(i) for i in range(K)] for _ in range(3)])
    logits, shared = carrier.split_contiguous(src, K - 1)  # 末 1 列 = shared gate
    assert logits.shape == (3, K - 1) and shared.shape == (3, 1)
    assert shared.rows[0] == [float(K - 1)]  # 末列才属于 shared
    # contiguous 产生新对象: 改切片不影响父张量
    logits.rows[0][0] = 99.0
    assert src.rows[0][0] == 0.0
    assert shared.rows[0] is not src.rows[0]


def test_fp32_hint_keeps_logits_fp32_and_shared_in_input_dtype():
    block = _make_block(fp32_hint=True)
    ops = FakeOps()
    carrier.install_gate_merge(block, environ={carrier.ENV: "1"}, ops=ops)
    state = block._hust_gate_merge
    assert state.fp32 is True
    x = FakeTensor([[1.0] * K], "bf16")
    logits = block.experts.gate(x)
    gate = block.shared_expert.expert_gate(x)[0]
    assert logits.dtype == "fp32"  # stock fp32 路由的 dtype 保持
    assert gate.dtype == "bf16"  # 不把 shared 乘法提升成 fp32


def test_fail_closed_when_shared_consumer_missing():
    block = _make_block(with_shared=False)
    ops = FakeOps()
    original_router = block.experts.gate
    ok = carrier.install_gate_merge(block, environ={carrier.ENV: "1"}, ops=ops)
    assert ok is False
    # router 槽位必须保持原样（不能只装一半）
    assert block.experts.gate is original_router
    assert block.experts._gate is original_router
    assert block._hust_gate_merge is False  # 记一次失败, 不再重试
    ok = carrier.install_gate_merge(block, environ={carrier.ENV: "1"}, ops=ops)
    assert ok is False


def test_describe_pins_env_and_rollback():
    info = carrier.describe()
    assert info["env"] == carrier.ENV
    assert "default-off" in info["default"] or "off" in info["default"]
    assert "0" in info["rollback"]
    assert info["contiguous"] == ["router logits", "shared gate"]


@pytest.mark.parametrize("bad", ["", "yes", "ON"])
def test_env_values_other_than_one_are_off(bad):
    assert carrier.enabled({carrier.ENV: bad}) is False
