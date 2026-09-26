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

"""W3.1 NPU tests: fused ``gelu_and_mul`` kernel (triton-ascend lane).

Structure:

1. ``test_probe_tl_erf_on_npu`` -- the **gating** primitive probe: does
   ``tl.erf`` compile on NPU and reproduce ``torch.erf`` in the fp32 domain
   (tolerance 1e-6)?  Every other numeric case takes the ``erf_ok`` fixture,
   which ``pytest.xfail``s the case when the probe failed, so a broken
   primitive is reported loudly instead of turning into silent numeric noise.
2. golden comparison against the explicit-erf fp32 reference
   (``reference.gelu_and_mul_reference``) over B in {1, 8, 64, 256} x
   inter in {13824, 13823, 13825, 512}.
3. kernel-vs-bf16-rounded-golden: pins that the *compute* is right and the
   residual error is only the final bf16 store rounding.
4. gate/up orientation pin (not a reversed slice).
5. cross-check against ``torch_npu.npu_gelu`` (second oracle, fp32 domain).
6. non-contiguous-input guard + input-shape/dtype guards.

Precision gate: ``ops-precision-standard`` (float_compute.md §3) BFLOAT16 row --
rtol = atol = 2**-6, required_matched_ratio = 0.99, and
max_abs_error_limit = max(1e0, 32 * 2**-7) = 1.0 (32 * bf16 ULP at 1.0, the
looser of "1e-0 or 32*ULP").  The elementwise pass rule is
``|a - g| <= atol + rtol * |g|``.
"""

from __future__ import annotations

import numpy as np
import pytest
import torch
import torch_npu  # noqa: F401
import triton

from vllm_ascend_split_batch.fi_gelu import api, reference
from vllm_ascend_split_batch.fi_gelu.kernels import _erf_probe_kernel

# -- ops-precision-standard, float_compute.md §3, BFLOAT16 row -----------------
BF16_RTOL = 2.0**-6  # 1.5625e-02
BF16_ATOL = 2.0**-6  # 1.5625e-02
BF16_REQUIRED_MATCHED_RATIO = 0.99
BF16_MAX_ABS_LIMIT = max(1e0, 32 * 2.0**-7)  # = 1.0

INTERS = (13824, 13823, 13825, 512)
TOKENS = (1, 8, 64, 256)

_SQRT1_2 = 0.7071067811865476
_PROBE_RESULT: dict = {}


def _npu_available() -> bool:
    try:
        return torch.npu.is_available()
    except (AttributeError, RuntimeError):
        return False


pytestmark = pytest.mark.skipif(not _npu_available(), reason="NPU not available")


def _rand_input(tokens: int, inter: int, seed: int = 0) -> torch.Tensor:
    g = torch.Generator().manual_seed(seed)
    return torch.randn(tokens, 2 * inter, generator=g, dtype=torch.float32).bfloat16()


def _mixed_tolerance(
    actual: np.ndarray, golden: np.ndarray, rtol: float, atol: float
) -> dict:
    """Mixed-tolerance check, verbatim per ``float_compute.md`` §2/§4."""
    a = np.asarray(actual, dtype=np.float64)
    g = np.asarray(golden, dtype=np.float64)
    if a.shape != g.shape:
        raise ValueError(f"shape mismatch: {a.shape} vs {g.shape}")
    abs_err = np.abs(a - g)
    passed = abs_err <= atol + rtol * np.abs(g)
    total = a.size
    matched_ratio = 1.0 if total == 0 else float(passed.sum()) / total
    max_abs_error = 0.0 if total == 0 else float(abs_err.max())
    return {
        "matched_ratio": matched_ratio,
        "max_abs_error": max_abs_error,
        "is_pass": (
            matched_ratio >= BF16_REQUIRED_MATCHED_RATIO
            and max_abs_error <= BF16_MAX_ABS_LIMIT
        ),
    }


def _bf16_check(actual: torch.Tensor, golden: torch.Tensor, note: str = "") -> dict:
    res = _mixed_tolerance(
        actual.float().cpu().numpy(), golden.float().cpu().numpy(), BF16_RTOL, BF16_ATOL
    )
    assert res["is_pass"], (
        f"bf16 mixed-tolerance FAIL{note}: matched_ratio={res['matched_ratio']:.6f} "
        f"(need >= {BF16_REQUIRED_MATCHED_RATIO}), "
        f"max_abs_error={res['max_abs_error']:.6g} (limit {BF16_MAX_ABS_LIMIT})"
    )
    return res


# --------------------------------------------------------------------------- #
# 1. gating primitive probe
# --------------------------------------------------------------------------- #
def _run_erf_probe() -> dict:
    """Run ``tl.erf`` on NPU once and cache (ok, max_abs_err, detail)."""
    if _PROBE_RESULT:
        return _PROBE_RESULT
    try:
        g = torch.Generator().manual_seed(0)
        x = torch.rand(1 << 20, generator=g, dtype=torch.float32) * 12.0 - 6.0
        x[:6] = torch.tensor([0.0, 1e-8, -1e-8, 6.0, -6.0, 3.14159])
        n = x.numel()
        xn = x.npu()
        out = torch.empty_like(xn)
        block = 1024
        _erf_probe_kernel[(triton.cdiv(n, block),)](xn, out, n, BLOCK=block)
        torch.npu.synchronize()
        max_err = float((out.cpu() - torch.erf(x)).abs().max().item())
        result = {
            "ok": bool(max_err <= 1e-6),
            "max_abs_err": max_err,
            "n": n,
            "detail": (
                f"tl.erf vs torch.erf (fp32, n={n}, range [-6, 6]): "
                f"maxAbs={max_err:.3e} (tol 1e-6)"
            ),
        }
    except Exception as exc:  # noqa: BLE001 - report the failure, do not mask it
        result = {
            "ok": False,
            "max_abs_err": float("nan"),
            "n": 0,
            "detail": f"tl.erf probe raised {type(exc).__name__}: {exc}",
        }
    _PROBE_RESULT.update(result)
    return _PROBE_RESULT


@pytest.fixture(scope="module")
def erf_ok() -> dict:
    """Gate every numeric case on the ``tl.erf`` probe (xfail on failure)."""
    probe = _run_erf_probe()
    if not probe["ok"]:
        pytest.xfail(f"tl.erf 前置探针失败 -> {probe['detail']}")
    return probe


def test_probe_tl_erf_on_npu() -> None:
    """GATING: ``tl.erf`` must compile on NPU and match ``torch.erf`` to 1e-6."""
    probe = _run_erf_probe()
    assert probe["ok"], probe["detail"]


# --------------------------------------------------------------------------- #
# 2. golden comparison
# --------------------------------------------------------------------------- #
@pytest.mark.parametrize("inter", INTERS)
@pytest.mark.parametrize("tokens", TOKENS)
def test_matches_fp32_erf_golden(erf_ok: dict, tokens: int, inter: int) -> None:
    x = _rand_input(tokens, inter)
    out = api.gelu_and_mul(x.npu())
    assert out.shape == (tokens, inter)
    assert out.dtype == torch.bfloat16
    golden = reference.gelu_and_mul_reference(x, dtype=torch.float32)
    note = f" B={tokens} inter={inter} [{erf_ok['detail']}]"
    res = _bf16_check(out, golden, note=note)
    assert res["matched_ratio"] > 0.999


# --------------------------------------------------------------------------- #
# 3. the residual error is only the bf16 store rounding
# --------------------------------------------------------------------------- #
@pytest.mark.parametrize("inter", (13824, 13825))
def test_equals_bf16_rounded_golden(erf_ok: dict, inter: int) -> None:
    """tl.erf is bit-exact vs torch.erf, so the kernel must equal the bf16
    rounding of the fp32 golden to within ~1 bf16 ULP."""
    x = _rand_input(64, inter)
    out = api.gelu_and_mul(x.npu()).float().cpu()
    gold_bf16 = reference.gelu_and_mul_reference(x, dtype=torch.bfloat16).float()
    abs_err = (out - gold_bf16).abs()
    # 2**-7 * |g| is one bf16 ULP at |g|; the +1e-6 floor covers exact zeros.
    thr = gold_bf16.abs() * 2.0**-7 + 1e-6
    frac = float((abs_err <= thr).float().mean().item())
    assert frac >= 0.9999, (
        f"{erf_ok['detail']} | bf16-round-trip mismatch frac={frac:.6f} "
        f"maxAbs={abs_err.max().item():.3e}"
    )


# --------------------------------------------------------------------------- #
# 4. gate/up orientation
# --------------------------------------------------------------------------- #
def test_gate_up_orientation(erf_ok: dict) -> None:
    """gate = first half, up = second half -- and the reversed slicing fails."""
    tokens, inter = 4, 512
    g = torch.Generator().manual_seed(7)
    gate = torch.randn(tokens, inter, generator=g, dtype=torch.float32) * 2.0
    up = torch.randn(tokens, inter, generator=g, dtype=torch.float32) * 0.25
    x = torch.cat([gate, up], dim=-1).bfloat16()

    out = api.gelu_and_mul(x.npu())
    forward = reference.gelu_and_mul_reference(x, dtype=torch.float32)
    _bf16_check(out, forward, note=" orientation gate=front")

    # the reversed interpretation must NOT pass the bf16 gate
    reversed_golden = (reference.gelu_erf(up) * gate).float()
    reversed_res = _mixed_tolerance(
        out.float().cpu().numpy(),
        reversed_golden.cpu().numpy(),
        BF16_RTOL,
        BF16_ATOL,
    )
    assert not reversed_res["is_pass"], (
        "reversed gate/up slicing also passes -> orientation is not pinned "
        f"(matched_ratio={reversed_res['matched_ratio']:.6f})"
    )


# --------------------------------------------------------------------------- #
# 5. second oracle: torch_npu.npu_gelu
# --------------------------------------------------------------------------- #
def test_npu_gelu_oracle_is_exact_erf(erf_ok: dict) -> None:
    gate = _rand_input(64, 512, seed=3)[..., :512].float()
    got = reference.npu_gelu_reference(gate.npu()).cpu()
    gold = reference.gelu_erf(gate)
    max_abs = float((got - gold).abs().max().item())
    assert max_abs < 1e-5, f"npu_gelu != exact erf: maxAbs={max_abs:.3e}"


@pytest.mark.parametrize("inter", (13824, 512))
def test_matches_npu_gelu_chain(erf_ok: dict, inter: int) -> None:
    x = _rand_input(8, inter, seed=11)
    out = api.gelu_and_mul(x.npu())
    golden = reference.gelu_and_mul_npu_gelu_reference(x.npu(), dtype=torch.float32)
    _bf16_check(out, golden, note=" vs npu_gelu chain")


# --------------------------------------------------------------------------- #
# 6. non-contiguous input + input guards
# --------------------------------------------------------------------------- #
def test_non_contiguous_input(erf_ok: dict) -> None:
    tokens, inter = 16, 768
    g = torch.Generator().manual_seed(5)
    src = torch.randn(2 * inter, tokens, generator=g, dtype=torch.float32).bfloat16()
    x_nc = src.t()  # [tokens, 2*inter] with strides (1, tokens) -> non-contiguous
    assert not x_nc.is_contiguous()
    out_nc = api.gelu_and_mul(x_nc.npu())
    out_c = api.gelu_and_mul(x_nc.contiguous().npu())
    assert torch.equal(out_nc, out_c)
    _bf16_check(out_nc, reference.gelu_and_mul_reference(x_nc, dtype=torch.float32))


def test_out_parameter_writes_in_place(erf_ok: dict) -> None:
    x = _rand_input(8, 512)
    buf = torch.empty(8, 512, dtype=torch.bfloat16).npu()
    ret = api.gelu_and_mul(x.npu(), out=buf)
    assert ret is buf
    _bf16_check(ret, reference.gelu_and_mul_reference(x, dtype=torch.float32))


def test_rejects_bad_out_shape() -> None:
    x = torch.randn(4, 64, dtype=torch.bfloat16).npu()
    with pytest.raises(ValueError, match="shape"):
        api.gelu_and_mul(x, out=torch.empty(4, 31, dtype=torch.bfloat16).npu())


def test_rejects_non_bf16() -> None:
    with pytest.raises(TypeError, match="bfloat16"):
        api.gelu_and_mul(torch.randn(4, 64, dtype=torch.float32).npu())


def test_rejects_odd_last_dim() -> None:
    with pytest.raises(ValueError, match="even"):
        api.gelu_and_mul(torch.randn(4, 63, dtype=torch.bfloat16).npu())


def test_rejects_1d() -> None:
    with pytest.raises(ValueError, match="at least 2D"):
        api.gelu_and_mul(torch.randn(64, dtype=torch.bfloat16).npu())
