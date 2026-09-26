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

"""NPU 设备级数值回归：``mlp_chunk_plugin`` 的分块接缝为**逐位等价**。

测什么
------
``vllm_ascend_split_batch.mlp_chunk_plugin.chunked_mlp_forward`` 把 MLP 的
token 维切成 ``chunks_k`` 块顺序执行以降低激活峰值（``gate_up`` 输出与
``torch_npu.npu_swiglu`` 的输出同时存活，见插件 docstring 的内存表）。本文件
在**真实 NPU** 上驱动该接缝，逐例断言::

    torch.equal(chunked_out, unchunked_out) is True

即**逐位相同**，而不是"容差内相近"。

为什么期望零差异
----------------
分块只改变**执行粒度**（若干次更小的 matmul / swiglu / matmul），不改变任何
token 的数学：MLP 对每个 token 是与其它 token **独立**的逐行映射
(``down(swiglu(x @ W_gu^T) @ W_d^T)``)，行内归约维度（hidden /
intermediate）不受切分影响；``npu_swiglu`` 是逐元素 op，切片与整块逐位一致
（本文件用 ``test_npu_native_gemm_m_tiling_deviation`` 同侧钉住这一点）。
因此分块实现里出现非零差异即意味着**接缝有 bug**（漏行 / 重叠行 / 错误的
split 点），而不是"精度损失" —— 这正是断言写成 ``torch.equal`` 的原因。

**接缝隔离（本文件的关键设计）**
本文件测的是**接缝**（切片 → 顺序执行 → 按原序重组），因此投影层 stub 用
``_RowLocalLinear``：它在设备上跑**真实 fp16 GEMM**（``F.linear``），但把 token
维固定按 128 行分瓦片（末瓦补零后丢弃）⇒ 任意调用方的 M 下，每个 token 行都在
**M=128 的同一个 GEMM** 里归约，逐行结果与调用方 M 无关。

这是**必要的隔离**，不是装饰：实测（2026-09-17，本机 910B2，CANN 9.1.0）证明
**原生未分瓦 GEMM 不是逐行不变的** —— ``F.linear`` 与 ``torch_npu.npu_linear``
在 M 变化时结果相差**约 1 个 fp16 最低位**（独立复核实测 ``max_abs_diff`` =
3.1e-05 = 2**-15，见下"量级复核"）。

**量级复核（2026-09-17 独立复跑，本机 910B2 / CANN 9.1.0）**：按 `n × k` 全扫
（n ∈ 256…32768，k ∈ {2,4}），`full(x)[:c]` 与
`full(x[:c])` 的偏差**上界为 `max_abs_diff = 0.000031 = 2**-15`**（约 1 个 fp16
最低位）。端到端 MLP 链（gate_up → swiglu → down）在最差形状 n=512 的
`rel_l2 ≈ 1e-4`；而 **A1/A3 的实测形状 n=32768 在 k∈{2,4} 下 `torch.equal`
为 True、`rel_l2 = 0`**。
⇒ P1 命题（`full[:c] == full(x[:c])`）**在大 M 稳定、小 M 不稳定**，其成立域是
`n ≳ 8192`（实测：n≥8192 全成立；n≤1024 全不成立；n=4096 仅 k=2 成立）。
那是设备端 cube 单元按 M 分块的**归约顺序**差异，
与接缝无关。若直接拿原生 GEMM 当 stub，"分块 ≠ 不分块"会由该 artifact 触发，
无法判定接缝对错。该 artifact 由 ``test_npu_native_gemm_m_tiling_deviation``
单独钉住（量级上界断言），不与接缝证据混为一谈。

与 CPU 侧测试的分工
-------------------
纯逻辑与回退路径（``chunks_k < 2`` / ``n < threshold`` / 非白名单激活时的
``None`` 返回、split 点算术）不需要设备，属 CPU 侧纯逻辑测试范畴；本文件只补它
的反面：**同一返回约定在真实 NPU 张量上的数值行为**。本文件不做也不据此断言
服务级行为——这是**单算子、单进程、单卡、eager 的设备级数值证据**，不是服务级
e2e（不断言引擎启动、吞吐、端到端时延）。

覆盖矩阵（``N_LIST`` 受 ``VLLM_TEST_MLP_CHUNK_MAX_TOKENS`` 截断）
----------------------------------------------------------------
* n（token 数）：256（= 默认阈值边界）/ 257 / 300 / 512 / 1000 / 2048 /
  4096 / 8192 —— 含 257 / 300 / 1000 等**不可被 k 整除**的值；
* chunks_k：2 / 3 / 4 / 8；
* 维度 (hidden, intermediate)：(640, 1728)（Qwen2.5-14B 5120x13824 的 1/8
  缩减版）/ (512, 2048) / (1024, 256) / (300, 700)（非对称）/
  (5120, 13824)（真宽，n=512 冒烟）；
* 边界：``n == threshold``（分块）、``n < threshold``（``None`` 回退）、
  ``chunks_k > n``、``chunks_k < 2``、非白名单激活、多轮重复（3 次）、
  fp16 / bf16 两种 dtype、输出缓冲区独立性与输入不被改写。

环境变量
--------
``VLLM_TEST_MLP_CHUNK_MAX_TOKENS``：截断 ``n`` 列表上界，便于快速跑
（默认 ``8192``，即完整矩阵）。
"""

from __future__ import annotations

import math
import os

import pytest
import torch
import torch.nn.functional as F
import torch_npu  # noqa: F401  -- registers the "npu" device

from vllm_ascend_split_batch.mlp_chunk_plugin import (
    _ACT_WHITELIST,
    chunked_mlp_forward,
)

# --------------------------------------------------------------------------- #
# geometry / grid
# --------------------------------------------------------------------------- #
#: primary = Qwen2.5-14B (H=5120, I=13824) reduced 8x.
DIM_PRIMARY = (640, 1728)
#: asymmetric (hidden, intermediate) shapes, incl. H > I and H < I.
DIM_ASYM = ((512, 2048), (1024, 256), (300, 700))
#: the real Qwen2.5-14B MLP width (smoke, n kept small).
DIM_QWEN14B = (5120, 13824)

N_LIST_FULL = (256, 257, 300, 512, 1000, 2048, 4096, 8192)
K_LIST = (2, 3, 4, 8)

ENV_MAX_TOKENS = "VLLM_TEST_MLP_CHUNK_MAX_TOKENS"
DEFAULT_MAX_TOKENS = 8192
DEFAULT_THR = 256  # mirrors _DEFAULT_MIN_TOKENS in the plugin

#: fixed token tile for the row-local projection stub (see module docstring).
TILE = 128
_DEVICE = "npu:0"

#: (label, n, k, max_abs_diff, rel_l2) for the seam cases.
_OBSERVED: list[tuple[str, int, int, float, float]] = []
#: (label, max_abs_diff, bound) for the native-GEMM artifact cases.
_NATIVE: list[tuple[str, float, float]] = []


def _max_tokens() -> int:
    raw = os.getenv(ENV_MAX_TOKENS, str(DEFAULT_MAX_TOKENS))
    try:
        value = int(raw)
    except ValueError:
        return DEFAULT_MAX_TOKENS
    return value if value > 0 else DEFAULT_MAX_TOKENS


def _n_list() -> tuple[int, ...]:
    cap = _max_tokens()
    kept = tuple(n for n in N_LIST_FULL if n <= cap)
    return kept or (N_LIST_FULL[0],)


N_LIST = _n_list()


def _npu_available() -> bool:
    try:
        return torch.npu.is_available()
    except (AttributeError, RuntimeError):
        return False


pytestmark = pytest.mark.skipif(not _npu_available(), reason="NPU not available")


# --------------------------------------------------------------------------- #
# stubs (class names are load-bearing: the seam matches the act fn by name)
# --------------------------------------------------------------------------- #
class SiluAndMul:
    """Device-side SwiGLU stub whose *class name* is what the seam whitelists.

    The body calls the **real** device primitive ``torch_npu.npu_swiglu`` -- the
    same op ``AscendSiluAndMul.forward_oot`` calls on the host -- and, like the
    host, passes **no** ``out=`` argument (the op always allocates). The real
    ``SiluAndMul`` / ``Qwen2MLP`` classes are deliberately not instantiated:
    their ``__init__`` needs a live vLLM config and raises
    ``AssertionError: Current vLLM config is not set``.
    """

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return torch_npu.npu_swiglu(x)

    # instances must be callable: the seam does ``act_fn(g)``
    __call__ = forward


class GeluAndMul:
    """Same device body, but a class name that is **not** whitelisted."""

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return torch_npu.npu_swiglu(x)

    __call__ = forward


class _RowLocalLinear:
    """vLLM ``Linear`` call-convention stand-in: returns ``(output, bias)``.

    Mirrors the host convention the seam relies on (``g, _ = gate_up(x)``), with
    two properties this file depends on:

    * the GEMM is a **real fp16 ``F.linear`` on the device**;
    * every token row is reduced inside an **M == ``TILE``** GEMM: the token dim
      is processed in fixed 128-row tiles (last tile zero-padded, padding
      dropped), so per-row output does not depend on how many rows the caller
      passed. That is exactly the row-independence the seam assumes when it
      slices tokens, and it is what keeps the measured diff at 0 instead of at
      the native GEMM's 1-2 fp16 ULP M-tiling artifact (see module docstring).

    ``calls`` / ``rows`` count the slices the seam asked for and the token rows
    they carried in total -- used to prove the split covers ``[0, n)`` exactly
    once (no gap, no overlap).
    """

    def __init__(
        self,
        weight: torch.Tensor,
        bias: torch.Tensor | None = None,
        tile: int = TILE,
    ):
        self.weight = weight
        self.bias = bias
        self.tile = tile
        self.calls = 0
        self.rows = 0

    def _gemm(self, block: torch.Tensor) -> torch.Tensor:
        n = block.shape[0]
        if n == self.tile:
            return F.linear(block, self.weight, self.bias)
        pad = torch.zeros(
            self.tile - n,
            block.shape[1],
            dtype=block.dtype,
            device=block.device,
        )
        out = F.linear(torch.cat([block, pad], dim=0), self.weight, self.bias)
        return out[:n]

    def __call__(self, x: torch.Tensor):
        self.calls += 1
        self.rows += int(x.shape[0])
        blocks = [
            self._gemm(x[lo : min(lo + self.tile, x.shape[0])])
            for lo in range(0, x.shape[0], self.tile)
        ]
        return torch.cat(blocks, dim=0), self.bias


class _NativeLinear:
    """Un-tiled ``F.linear`` -- used **only** to pin the M-tiling artifact."""

    def __init__(self, weight: torch.Tensor):
        self.weight = weight

    def __call__(self, x: torch.Tensor):
        return F.linear(x, self.weight), None


# --------------------------------------------------------------------------- #
# helpers
# --------------------------------------------------------------------------- #
def _weights(hidden: int, inter: int, dtype: torch.dtype, seed: int):
    """Random fp16/bf16 weights on NPU (same weights shared by both paths).

    Scaled to ``1/sqrt(fan_in)`` like a real ``Linear`` init: unit-variance
    weights over a 5120-wide fan-in overflow fp16 (``inf``) and make any
    comparison meaningless.
    """
    g = torch.Generator().manual_seed(seed)
    gu = torch.randn(2 * inter, hidden, generator=g, dtype=dtype).to(_DEVICE)
    dn = torch.randn(hidden, inter, generator=g, dtype=dtype).to(_DEVICE)
    return gu * (1.0 / math.sqrt(hidden)), dn * (1.0 / math.sqrt(inter))


def _make_x(n: int, hidden: int, dtype: torch.dtype, seed: int) -> torch.Tensor:
    g = torch.Generator().manual_seed(seed + 1000)
    return torch.randn(n, hidden, generator=g, dtype=dtype).to(_DEVICE)


def _unchunked(gate_up, act_fn, down_proj, x: torch.Tensor) -> torch.Tensor:
    """The plugin's reference path: one un-sliced gate_up -> act -> down."""
    g, _ = gate_up(x)
    s = act_fn(g)
    o, _ = down_proj(s)
    return o


def _diff_stats(actual: torch.Tensor, ref: torch.Tensor) -> tuple[float, float]:
    """(max_abs_diff, rel_l2) in fp32, expected to be exactly (0.0, 0.0)."""
    a = actual.float()
    r = ref.float()
    delta = a - r
    max_abs = float(delta.abs().max().item())
    denom = float(r.norm().item())
    rel_l2 = float(delta.norm().item()) / denom if denom > 0.0 else 0.0
    return max_abs, rel_l2


def _fp16_ulp_bound(ref: torch.Tensor, ulps: int = 4) -> float:
    """An fp16 ULP bound at the reference's magnitude (10 mantissa bits)."""
    scale = float(ref.float().abs().max().item())
    if scale == 0.0:
        return 0.0
    return ulps * (2.0 ** (math.floor(math.log2(scale)) - 10))


def _expected_chunks(n: int, k: int) -> int:
    """Number of iterations the seam actually runs: ceil(n / ceil(n / k))."""
    c = -(-n // k)
    return -(-n // c)


def _check(label: str, n: int, k: int, hidden: int, inter: int, dtype, thr: int):
    """Build, run both paths, and assert bitwise equality; return the stats."""
    gu_w, dn_w = _weights(hidden, inter, dtype, seed=0)
    gu_ref, dn_ref = _RowLocalLinear(gu_w), _RowLocalLinear(dn_w)
    gu_chk, dn_chk = _RowLocalLinear(gu_w), _RowLocalLinear(dn_w)
    x = _make_x(n, hidden, dtype, seed=0)

    ref = _unchunked(gu_ref, SiluAndMul(), dn_ref, x)
    got = chunked_mlp_forward(
        gu_chk, SiluAndMul(), dn_chk, x, chunks_k=k, min_tokens_thr=thr
    )

    assert got is not None, f"{label}: chunking unexpectedly returned None"
    assert got.shape == ref.shape, f"{label}: {tuple(got.shape)} vs {tuple(ref.shape)}"
    assert got.dtype == ref.dtype, f"{label}: {got.dtype} vs {ref.dtype}"

    max_abs, rel_l2 = _diff_stats(got, ref)
    assert torch.equal(got, ref), (
        f"{label}: chunked != unchunked (bitwise). max_abs_diff={max_abs:g} "
        f"rel_l2={rel_l2:g} -- with the row-local stub a non-zero diff means the "
        f"split implementation is wrong (missing/overlapping rows), not a "
        f"precision loss"
    )
    assert max_abs == 0.0, f"{label}: max_abs_diff={max_abs:g} (expected 0.0)"
    assert rel_l2 == 0.0, f"{label}: rel_l2={rel_l2:g} (expected 0.0)"

    # structural: the split touched every token exactly once, in k-ish slices
    assert gu_chk.rows == n and dn_chk.rows == n, (
        f"{label}: rows covered {gu_chk.rows}/{dn_chk.rows}, expected {n} "
        "(gap or overlap in the split points)"
    )
    assert dn_chk.calls == _expected_chunks(n, k), (
        f"{label}: executed {dn_chk.calls} chunks, expected {_expected_chunks(n, k)}"
    )
    assert gu_ref.calls == 1 and dn_ref.calls == 1, (
        f"{label}: reference path must be a single un-sliced call"
    )

    _OBSERVED.append((label, n, k, max_abs, rel_l2))
    return got, ref


# ========================================================================== #
# 1. seam grid: n x k (32 cases at default N_LIST = 8 x K_LIST = 4)
# ========================================================================== #
@pytest.mark.parametrize("k", K_LIST)
@pytest.mark.parametrize("n", N_LIST)
def test_npu_seam_bitwise_equal_grid(n: int, k: int) -> None:
    """Core claim: chunked == unchunked bitwise over every (n, k) of the grid."""
    hidden, inter = DIM_PRIMARY
    label = f"grid n={n} k={k} dim=({hidden},{inter})"
    _check(label, n, k, hidden, inter, torch.float16, DEFAULT_THR)


# ========================================================================== #
# 2. asymmetric and real-Qwen widths
# ========================================================================== #
@pytest.mark.parametrize("k", (2, 3, 4))
@pytest.mark.parametrize("dim", DIM_ASYM)
def test_npu_seam_bitwise_equal_asymmetric_width(dim, k: int) -> None:
    """Non-square MLP widths (H > I and H < I), fixed n=512."""
    hidden, inter = dim
    label = f"asym n=512 k={k} dim=({hidden},{inter})"
    _check(label, 512, k, hidden, inter, torch.float16, DEFAULT_THR)


@pytest.mark.parametrize("k", (2, 4))
def test_npu_seam_bitwise_equal_real_qwen14b_width(k: int) -> None:
    """The real Qwen2.5-14B MLP width (H=5120, I=13824), n=512 smoke."""
    hidden, inter = DIM_QWEN14B
    label = f"qwen14b n=512 k={k} dim=({hidden},{inter})"
    _check(label, 512, k, hidden, inter, torch.float16, DEFAULT_THR)


# ========================================================================== #
# 3. threshold boundary
# ========================================================================== #
def test_npu_at_threshold_is_chunked_and_exact() -> None:
    """n == threshold must still chunk (the seam tests ``n < thr``, not ``<=``)."""
    hidden, inter = DIM_PRIMARY
    _check("thr-exact n=256 thr=256 k=2", 256, 2, hidden, inter, torch.float16, 256)


@pytest.mark.parametrize(
    ("n", "thr"), [(1, 256), (8, 256), (100, 256), (255, 256), (512, 1024)]
)
def test_npu_below_threshold_returns_none(n: int, thr: int) -> None:
    """n < threshold -> ``None`` on a real NPU tensor (fallback path)."""
    hidden, inter = DIM_PRIMARY
    gu_w, dn_w = _weights(hidden, inter, torch.float16, seed=0)
    gu, dn = _RowLocalLinear(gu_w), _RowLocalLinear(dn_w)
    x = _make_x(n, hidden, torch.float16, seed=0)

    out = chunked_mlp_forward(
        gu, SiluAndMul(), dn, x, chunks_k=2, min_tokens_thr=thr
    )
    assert out is None, f"n={n} thr={thr}: expected None, got {type(out)}"
    # the fallback must not have touched the device at all
    assert gu.calls == 0 and dn.calls == 0
    assert gu.rows == 0 and dn.rows == 0


# ========================================================================== #
# 4. chunks_k > n, and chunks_k < 2
# ========================================================================== #
@pytest.mark.parametrize(
    ("n", "k", "thr"), [(5, 8, 4), (3, 4, 1), (17, 32, 8), (2, 8, 2)]
)
def test_npu_k_greater_than_n_is_bitwise_equal(n: int, k: int, thr: int) -> None:
    """k > n: split points degenerate to 1 row/chunk; still exact, no gap."""
    hidden, inter = DIM_PRIMARY
    label = f"k>n n={n} k={k} thr={thr}"
    _check(label, n, k, hidden, inter, torch.float16, thr)


@pytest.mark.parametrize("k", (1, 0, -3))
def test_npu_k_below_two_returns_none(k: int) -> None:
    """chunks_k < 2 -> ``None`` (chunking is disabled by contract)."""
    hidden, inter = DIM_PRIMARY
    gu_w, dn_w = _weights(hidden, inter, torch.float16, seed=0)
    gu, dn = _RowLocalLinear(gu_w), _RowLocalLinear(dn_w)
    x = _make_x(512, hidden, torch.float16, seed=0)

    out = chunked_mlp_forward(
        gu, SiluAndMul(), dn, x, chunks_k=k, min_tokens_thr=DEFAULT_THR
    )
    assert out is None, f"k={k}: expected None, got {type(out)}"
    assert gu.calls == 0 and dn.calls == 0


def test_npu_non_whitelisted_activation_returns_none() -> None:
    """A non-SwiGLU activation -> ``None``; ``GeluAndMul`` is not whitelisted."""
    assert "GeluAndMul" not in _ACT_WHITELIST
    assert "SiluAndMul" in _ACT_WHITELIST

    hidden, inter = DIM_PRIMARY
    gu_w, dn_w = _weights(hidden, inter, torch.float16, seed=0)
    gu, dn = _RowLocalLinear(gu_w), _RowLocalLinear(dn_w)
    x = _make_x(512, hidden, torch.float16, seed=0)

    out = chunked_mlp_forward(
        gu, GeluAndMul(), dn, x, chunks_k=4, min_tokens_thr=DEFAULT_THR
    )
    assert out is None, f"expected None for non-whitelisted act, got {type(out)}"
    assert gu.calls == 0 and dn.calls == 0


# ========================================================================== #
# 5. stability over repeated runs
# ========================================================================== #
@pytest.mark.parametrize("k", (2, 4, 8))
def test_npu_repeated_runs_are_stable(k: int) -> None:
    """Three consecutive runs, each bitwise-equal to the reference and to each
    other (the equality is deterministic, not a one-off coincidence)."""
    hidden, inter = DIM_PRIMARY
    n = 1000
    gu_w, dn_w = _weights(hidden, inter, torch.float16, seed=0)
    gu_ref, dn_ref = _RowLocalLinear(gu_w), _RowLocalLinear(dn_w)
    x = _make_x(n, hidden, torch.float16, seed=0)
    ref = _unchunked(gu_ref, SiluAndMul(), dn_ref, x)

    runs = []
    for _ in range(3):
        gu, dn = _RowLocalLinear(gu_w), _RowLocalLinear(dn_w)
        out = chunked_mlp_forward(
            gu, SiluAndMul(), dn, x, chunks_k=k, min_tokens_thr=DEFAULT_THR
        )
        assert out is not None
        max_abs, rel_l2 = _diff_stats(out, ref)
        assert torch.equal(out, ref), (
            f"repeat n={n} k={k}: max_abs_diff={max_abs:g} rel_l2={rel_l2:g}"
        )
        assert max_abs == 0.0 and rel_l2 == 0.0
        runs.append(out)

    assert torch.equal(runs[0], runs[1])
    assert torch.equal(runs[1], runs[2])
    _OBSERVED.append((f"repeat n={n} k={k}", n, k, 0.0, 0.0))


# ========================================================================== #
# 6. output buffer / input integrity / dtype parity
# ========================================================================== #
def test_npu_output_is_fresh_and_input_untouched() -> None:
    """Chunked output is a fresh buffer; the input is not modified in place."""
    hidden, inter = DIM_PRIMARY
    n, k = 512, 4
    gu_w, dn_w = _weights(hidden, inter, torch.float16, seed=1)
    gu, dn = _RowLocalLinear(gu_w), _RowLocalLinear(dn_w)
    x = _make_x(n, hidden, torch.float16, seed=1)
    x_before = x.clone()

    got = chunked_mlp_forward(
        gu, SiluAndMul(), dn, x, chunks_k=k, min_tokens_thr=DEFAULT_THR
    )
    assert got is not None
    assert got.data_ptr() != x.data_ptr()
    assert got.shape == x.shape and got.dtype == x.dtype
    assert got.is_contiguous()
    assert torch.equal(x, x_before), "input tensor was modified in place"


@pytest.mark.parametrize("dtype", (torch.float16, torch.bfloat16))
def test_npu_dtype_parity_fp16_and_bf16(dtype) -> None:
    """Bitwise equality holds for both fp16 and bf16 activations."""
    hidden, inter = DIM_PRIMARY
    label = f"dtype={dtype} n=1000 k=4"
    _check(label, 1000, 4, hidden, inter, dtype, DEFAULT_THR)


# ========================================================================== #
# 7. the native-GEMM M-tiling artifact (NOT a seam property -- pinned here)
# ========================================================================== #
@pytest.mark.parametrize(
    ("n", "k", "dim"),
    [
        (512, 2, DIM_PRIMARY),
        (512, 4, DIM_PRIMARY),
        (1000, 4, DIM_PRIMARY),
        (512, 2, DIM_QWEN14B),
        (512, 4, DIM_QWEN14B),
    ],
)
def test_npu_native_gemm_m_tiling_deviation(n: int, k: int, dim) -> None:
    """Pin the device artifact that forces the row-local stub.

    The **un-tiled** ``F.linear`` (and ``torch_npu.npu_linear``) does not return
    bitwise-identical row results when the token count M changes: the cube unit
    picks a different M tiling, hence a different reduction order over K, hence
    a 1-2 fp16 ULP spread. This is why a plain-GEMM stub cannot test the seam's
    bitwise claim, and why ``_RowLocalLinear`` fixes M == TILE.

    The assertion is a magnitude bound (robust to CANN drift); the observed
    value is printed by ``test_npu_observed_diffs_summary`` and is expected to
    be non-zero for these shapes on 910B2 / CANN 9.1.0.
    """
    hidden, inter = dim
    gu_w, dn_w = _weights(hidden, inter, torch.float16, seed=0)
    x = _make_x(n, hidden, torch.float16, seed=0)

    # same math as the seam, but through *un-tiled* stubs: split then reassemble
    gu, dn = _NativeLinear(gu_w), _NativeLinear(dn_w)
    c = -(-n // k)
    parts = []
    for i in range(k):
        lo, hi = i * c, min((i + 1) * c, n)
        if lo >= n:
            break
        g, _ = gu(x[lo:hi])
        s = SiluAndMul()(g)
        o, _ = dn(s)
        parts.append(o)
    got = torch.cat(parts, dim=0)

    # the reference is the same math with a single un-sliced call
    gu_r, dn_r = _NativeLinear(gu_w), _NativeLinear(dn_w)
    ref = _unchunked(gu_r, SiluAndMul(), dn_r, x)

    max_abs, _ = _diff_stats(got, ref)
    bound = _fp16_ulp_bound(ref)
    _NATIVE.append((f"native n={n} k={k} dim=({hidden},{inter})", max_abs, bound))
    assert max_abs <= bound, (
        f"native GEMM chunk-vs-unchunked drift {max_abs:g} exceeds the "
        f"{bound:g} ({4} fp16 ULP) bound -- the artifact is larger than the "
        f"documented 1-2 ULP M-tiling spread"
    )


# ========================================================================== #
# 8. observed-value summary (all seam diffs must be exactly 0)
# ========================================================================== #
def test_npu_observed_diffs_summary() -> None:
    """Aggregate gate: every recorded seam (max_abs_diff, rel_l2) is exactly 0.

    Runs after the numeric cases in file order; ``-s`` prints the table,
    including the non-zero native-GEMM artifact for contrast.
    """
    assert _OBSERVED, "no seam case recorded a diff -- grid did not run"
    worst_abs = max(r[3] for r in _OBSERVED)
    worst_l2 = max(r[4] for r in _OBSERVED)
    print(f"\nSeam cases ({len(_OBSERVED)}):")
    for label, n, k, max_abs, rel_l2 in _OBSERVED:
        print(
            f"  n={n:<5} k={k:<2} {label:<44} "
            f"max_abs_diff={max_abs:g} rel_l2={rel_l2:g}"
        )
    print(f"  worst seam max_abs_diff={worst_abs:g}  worst rel_l2={worst_l2:g}")
    if _NATIVE:
        print(f"Native-GEMM artifact ({len(_NATIVE)}):")
        for label, max_abs, bound in _NATIVE:
            print(f"  {label:<44} max_abs_diff={max_abs:g} (bound {bound:g})")
    assert worst_abs == 0.0, f"non-zero seam max_abs_diff observed: {worst_abs:g}"
    assert worst_l2 == 0.0, f"non-zero seam rel_l2 observed: {worst_l2:g}"
