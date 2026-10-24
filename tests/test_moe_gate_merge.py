# SPDX-License-Identifier: Apache-2.0

"""MoE 双门合并（router + shared_expert_gate）的 CPU 语义与等价守卫。

证的是三件事（都不需要设备）:
1. 行序契约: 前 n_router 行是 router, 最后 1 行是 shared gate; 列数/行数不合即报错;
2. 切片协议: 两路都返回拷贝（对应 device 侧进 top-k / sigmoid 前必须 contiguous）;
3. 算术等价: 一次 257 行 matvec 与两次（256 + 1）matvec 逐元素相等。
设备 bf16 重结合与收益不在此文件（属 L0/L3）。
"""

import importlib.util
import pathlib
import random

import pytest

HERE = pathlib.Path(__file__).resolve().parent
SPEC = importlib.util.spec_from_file_location(
    "moe_gate_merge", HERE.parent / "tools" / "moe_gate_merge.py"
)
merge = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(merge)

K = 2048
N_ROUTER = 256


def _weights(rng, rows, cols):
    return [[rng.uniform(-1.0, 1.0) for _ in range(cols)] for _ in range(rows)]


def test_layout_constant_is_the_contract():
    assert merge.MERGED_LAYOUT == ("router", "shared_gate")


def test_row_mapping_at_real_dims():
    rng = random.Random(7)
    w_router = [[0.5] * K for _ in range(N_ROUTER)]
    w_shared = [[0.25] * K]
    merged = merge.merge_gate_weights(w_router, w_shared)
    assert merged.shape == (N_ROUTER + 1, K)
    assert merged.router_rows() == w_router
    assert merged.shared_rows() == w_shared
    # 最后一行才是标量门（顺序反了会让 sigmoid 分支吃到 router 的一行）
    assert merged.rows[-1] == w_shared[0]
    assert merged.rows[:N_ROUTER] == w_router
    del rng


def test_fused_equals_reference_at_real_dims_with_onehot_input():
    # one-hot 输入让 matvec 退化成一次行检索: 用真实维度也很快, 且数值可比对
    x = [[0.0] * K]
    x[0][1234] = 2.0
    w_router = [[0.0] * K for _ in range(N_ROUTER)]
    for i in range(N_ROUTER):
        w_router[i][1234] = i * 0.5
    w_shared = [[0.0] * K]
    w_shared[0][1234] = -3.0
    ok, (d_logits, d_shared) = merge.equivalent(x, w_router, w_shared)
    assert ok, (d_logits, d_shared)
    merged = merge.merge_gate_weights(w_router, w_shared)
    logits, shared = merge.fused_outputs(x, merged)
    assert logits[0][7] == 7.0  # w[7][1234] = 7*0.5 = 3.5, x = 2.0
    assert shared[0][0] == -6.0


@pytest.mark.parametrize("rows_x", [1, 3])
def test_fused_equals_reference_on_dense_input(rows_x):
    rng = random.Random(11)
    cols = 64
    x = _weights(rng, rows_x, cols)
    w_router = _weights(rng, 8, cols)
    w_shared = _weights(rng, 1, cols)
    ok, diffs = merge.equivalent(x, w_router, w_shared)
    assert ok, diffs


def test_outputs_are_copies_not_views_of_weights():
    rng = random.Random(3)
    cols = 16
    x = _weights(rng, 1, cols)
    w_router = _weights(rng, 4, cols)
    w_shared = _weights(rng, 1, cols)
    merged = merge.merge_gate_weights(w_router, w_shared)
    logits, shared = merge.fused_outputs(x, merged)
    before_router = [r[:] for r in merged.rows]
    logits[0][0] = 1e9
    shared[0][0] = 1e9
    assert merged.rows == before_router  # 改返回值不影响权重
    # 两路互相独立（shared 不是 logits 的视图）
    assert logits[0][-1] != shared[0][0]


def test_swapped_argument_order_is_rejected():
    rng = random.Random(5)
    w_router = _weights(rng, 4, 8)
    w_shared = _weights(rng, 1, 8)
    with pytest.raises(ValueError, match="只有 1 行"):
        merge.merge_gate_weights(w_shared, w_router)


def test_shared_gate_must_be_scalar_row():
    w_router = [[0.0] * 8 for _ in range(4)]
    with pytest.raises(ValueError, match="只有 1 行"):
        merge.merge_gate_weights(w_router, [[0.0] * 8, [0.0] * 8])


def test_column_mismatch_is_rejected_on_both_paths():
    w_router = [[0.0] * 8 for _ in range(4)]
    w_shared = [[0.0] * 7]
    with pytest.raises(ValueError, match="列数"):
        merge.merge_gate_weights(w_router, w_shared)
    with pytest.raises(ValueError, match="列数"):
        merge.reference_outputs([[0.0] * 8], w_router, w_shared)


def test_input_width_mismatch_is_rejected():
    merged = merge.merge_gate_weights([[0.0] * 8], [[0.0] * 8])
    with pytest.raises(ValueError, match="x 列数"):
        merge.fused_outputs([[0.0] * 7], merged)


def test_max_abs_diff_refuses_structural_mismatch():
    with pytest.raises(ValueError, match="行数不同"):
        merge.max_abs_diff([[0.0]], [[0.0], [0.0]])
    with pytest.raises(ValueError, match="列数不同"):
        merge.max_abs_diff([[0.0, 1.0]], [[0.0]])


def test_device_plan_pins_slices_and_unverified_items():
    plan = merge.device_plan(N_ROUTER, K)
    assert plan["merged_shape"] == [N_ROUTER + 1, K]
    assert plan["logits_slice"] == f"out[:, :{N_ROUTER}]"
    assert plan["shared_slice"] == f"out[:, {N_ROUTER}:]"
    assert plan["checkpoint_keys"].startswith("不变")
    assert any("bf16" in item for item in plan["not_verified_by_this_module"])
    assert any("L0" in item for item in plan["not_verified_by_this_module"])
