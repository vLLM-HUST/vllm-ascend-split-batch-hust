# SPDX-License-Identifier: Apache-2.0

"""标尺纯函数的口径守卫（CPU-only）。

v1 记录的两个口径缺陷，用本文件钉住:
* ``bw_floor`` 把 ``copy_`` 的读+写合计字节当单向用 → 地板恰好小一半;
* 判读量用 eager 提交地板（34.6–36.6µs）当分母 → 大于 FULL 图内 per-call，测不到。
"""

import importlib.util
import pathlib

HERE = pathlib.Path(__file__).resolve().parent
SPEC = importlib.util.spec_from_file_location(
    "ruler", HERE.parent / "tools" / "bench_matmul_m1_ruler.py"
)
ruler = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(ruler)

HIDDEN = 2048


def test_copy_bandwidth_is_halved_to_one_way():
    # gdn_inproj: 6176×2048 bf16 = 25.30MB, copy_ 读+写 50.6MB 用 23.064µs
    # → 单向 1096.8GB/s（v1 记的 2193.6 是读+写合计）
    nbytes = 6176 * 2048 * 2
    assert round(ruler.oneway_bw_gbps(2 * nbytes, 23.064), 1) == 1096.8


def test_load_floor_reproduces_v1_then_doubles():
    # v1 证据: gdn_inproj bw_floor=11.532µs; 单向折算后应恰好为它的 2 倍
    nbytes = 6176 * 2048 * 2
    one_way = ruler.oneway_bw_gbps(2 * nbytes, 23.064)
    assert round(ruler.load_floor_us(nbytes, one_way), 3) == 23.064
    assert round(ruler.load_floor_us(nbytes, 2 * one_way), 3) == 11.532


def test_window_per_call_matches_full_graph_numbers():
    # 窗 1021-031741: gdn_inproj 25.069ms / (30 calls × 32 iters) = 26.11µs
    assert round(ruler.window_us_per_call(25.069, 30, 32), 2) == 26.11
    # lm_head 11.724ms / (1 × 32) = 366.38µs
    assert round(ruler.window_us_per_call(11.724, 1, 32), 2) == 366.38


def test_read_efficiency_puts_lm_head_at_ceiling():
    # lm_head 508.56MB / 366.38µs ≈ 1388GB/s → ≈92% of 1510
    assert round(ruler.read_efficiency_pct(508_560_000, 366.38), 1) == 91.9


def test_reading_classifies_big_buckets_as_unmeasured():
    cells = [
        {"id": "gdn_inproj", "weight_mb": 25.3, "pct_of_read_ceiling": 64.2},
        {"id": "lm_head", "weight_mb": 508.56, "pct_of_read_ceiling": 91.9},
        {"id": "router", "weight_mb": 1.05, "pct_of_read_ceiling": 9.2},
    ]
    reading = ruler.build_reading(cells)
    assert reading["converged"] == ["lm_head"]
    assert "gdn_inproj" in reading["unmeasured_headroom"]
    assert "router" in reading["unmeasured_headroom"]
    assert reading["big_cells"] == ["gdn_inproj", "lm_head"]
    assert reading["big_pct_median"] == round((64.2 + 91.9) / 2, 1)
    # v1 的收敛规则必须显式作废，不能留着当判据
    assert "作废" in reading["withdrawn"]


def test_cells_table_covers_every_bucket_with_calls_per_iter():
    assert set(ruler.CELLS) == set(ruler.CALLS_PER_IT)
    for name, (n, k, window_ms, _callsite) in ruler.CELLS.items():
        assert n > 0 and k > 0 and window_ms > 0, name
        assert ruler.CALLS_PER_IT[name] > 0, name
    assert ruler.CELLS["lm_head"][1] == HIDDEN
