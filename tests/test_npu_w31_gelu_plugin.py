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

"""W3.1c tests: fi_gelu host wiring (``GeluAndMul.forward_oot`` patch).

Two sections in one file:

* **CPU** (top) -- default-off contract and the pure fail-open logic.  The
  subprocess ``sys.modules`` purity test runs anywhere; the ``_make_forward``
  delegation tests only need ``torch`` (importorskip).
* **NPU** (bottom) -- needs the real ``torch_npu`` + vLLM stack: dispatch
  confirmation (which ``forward_*`` the host actually selects), the patch
  matching the kernel / the fp32-erf golden, ACL graph capture+replay equality,
  the fail-open delegation on the real host class, and a live host-signature
  guard (``test_host_signature_drift.py`` style).

Dispatch finding pinned below: on NPU ``CustomOp.dispatch_forward`` stores the
inherited ``CustomOp.forward_oot`` in ``_forward_method`` (``NPUPlatform`` is
``PlatformEnum.OOT`` and ``GeluAndMul.enabled()`` is True); ``GeluAndMul``
defines neither ``forward_oot`` nor ``forward_cuda``.  ``FORWARD_TARGET`` is
therefore ``forward_oot``.

Precision gate: ``ops-precision-standard`` (float_compute.md section 3)
BFLOAT16 row -- rtol = atol = 2**-6, required_matched_ratio = 0.99, and
max_abs_error_limit = max(1e0, 32 * 2**-7) = 1.0.
"""

from __future__ import annotations

import inspect
import subprocess
import sys

import pytest

from vllm_ascend_split_batch import fi_gelu_plugin as plugin

ENV = plugin.ENV_ENABLE

# -- ops-precision-standard, float_compute.md section 3, BFLOAT16 row ---------
BF16_RTOL = 2.0**-6
BF16_ATOL = 2.0**-6
BF16_REQUIRED_MATCHED_RATIO = 0.99
BF16_MAX_ABS_LIMIT = max(1e0, 32 * 2.0**-7)  # = 1.0


def _npu_available() -> bool:
    try:
        import torch
        import torch_npu  # noqa: F401
    except Exception:  # noqa: BLE001 - CPU-only env
        return False
    try:
        return torch.npu.is_available()
    except Exception:  # noqa: BLE001
        return False


needs_npu = pytest.mark.skipif(not _npu_available(), reason="NPU not available")


@pytest.fixture(autouse=True)
def _isolated_plugin_state(monkeypatch):
    """Default-off env, fresh once-guards, and a guaranteed unpatch per test."""
    monkeypatch.delenv(ENV, raising=False)
    plugin._reset_for_tests()
    yield
    plugin._unpatch()
    plugin._reset_for_tests()


# ===================================================================== CPU ==
def test_load_default_off_patches_nothing_and_imports_nothing() -> None:
    """Default-off: no env, no patch, no host/kernel import (subprocess)."""
    code = (
        "import sys;"
        "import vllm_ascend_split_batch.fi_gelu_plugin as p;"
        "assert p.load() is False;"
        "assert p._installed_cls is None;"
        "assert p._orig_forward is None;"
        "bad=[m for m in sys.modules if m=='triton'"
        " or m.split('.')[0] in ('vllm','vllm_ascend')"
        " or m in ('vllm_ascend_split_batch.fi_gelu',)"
        " or m.startswith('vllm_ascend_split_batch.fi_gelu.')];"
        "assert not bad, bad"
    )
    result = subprocess.run(
        [sys.executable, "-c", code],
        cwd="/tmp",
        capture_output=True,
        text=True,
    )
    assert result.returncode == 0, result.stderr


def test_is_enabled_only_accepts_the_exact_token(monkeypatch) -> None:
    monkeypatch.delenv(ENV, raising=False)
    assert plugin.is_enabled() is False
    for value in ("0", "true", "yes", " 1", "1 "):
        monkeypatch.setenv(ENV, value)
        assert plugin.is_enabled() is False, value
    monkeypatch.setenv(ENV, "1")
    assert plugin.is_enabled() is True


def test_kernel_eligible_guards() -> None:
    torch = pytest.importorskip("torch")
    bf16 = torch.bfloat16
    # dtype gate
    assert not plugin._kernel_eligible(
        torch.empty(4, 8, dtype=torch.float32), bf16, "cpu"
    )
    # ndim gate
    assert not plugin._kernel_eligible(torch.empty(8, dtype=bf16), bf16, "cpu")
    # last-dim gates (empty / odd)
    assert not plugin._kernel_eligible(torch.empty(4, 0, dtype=bf16), bf16, "cpu")
    assert not plugin._kernel_eligible(torch.empty(4, 7, dtype=bf16), bf16, "cpu")
    # device gate (NPU tensor required by default)
    cpu_bf16 = torch.empty(4, 8, dtype=bf16)
    assert not plugin._kernel_eligible(cpu_bf16, bf16)
    assert plugin._kernel_eligible(cpu_bf16, bf16, "cpu")
    # a realistic GeGLU row is eligible
    assert plugin._kernel_eligible(torch.empty(4, 2 * 13824, dtype=bf16), bf16, "cpu")


class _FakeGeluAndMul:
    """Minimal stand-in: only ``approximate`` is read by the patch."""

    def __init__(self, approximate: str = "none") -> None:
        self.approximate = approximate


def test_make_forward_delegates_on_ineligible_inputs() -> None:
    torch = pytest.importorskip("torch")
    calls = []

    def orig(self, x):
        calls.append(x)
        return "orig"

    fwd = plugin._make_forward(orig, lambda x: "kernel", torch.bfloat16, "cpu")
    op = _FakeGeluAndMul("none")
    # fp32 -> not eligible
    assert fwd(op, torch.empty(4, 8, dtype=torch.float32)) == "orig"
    # 1D -> not eligible
    assert fwd(op, torch.empty(8, dtype=torch.bfloat16)) == "orig"
    # odd last dim -> not eligible
    assert fwd(op, torch.empty(4, 7, dtype=torch.bfloat16)) == "orig"

    # npu tensor on a host-only fake -> device gate fails
    class _Dev:
        type = "npu"

    class _X:
        dtype = torch.bfloat16
        shape = (4, 8)
        device = _Dev()

        def dim(self):
            return 2

    assert fwd(op, _X()) == "orig"
    assert len(calls) == 4


def test_make_forward_delegates_for_tanh_approximation() -> None:
    torch = pytest.importorskip("torch")
    fwd = plugin._make_forward(
        lambda self, x: "orig", lambda x: "kernel", torch.bfloat16, "cpu"
    )
    op = _FakeGeluAndMul("tanh")
    x = torch.empty(4, 8, dtype=torch.bfloat16)
    assert fwd(op, x) == "orig"


def test_make_forward_kernel_failure_is_fail_open(caplog) -> None:
    torch = pytest.importorskip("torch")
    plugin._reset_for_tests()

    def boom(x):
        raise RuntimeError("kernel boom")

    fwd = plugin._make_forward(lambda self, x: "orig", boom, torch.bfloat16, "cpu")
    with caplog.at_level("WARNING"):
        out = fwd(_FakeGeluAndMul("none"), torch.empty(4, 8, dtype=torch.bfloat16))
    assert out == "orig"
    assert len([r for r in caplog.records if "fi_gelu kernel" in r.getMessage()]) == 1
    # once-only: a second failure does not log again
    with caplog.at_level("WARNING"):
        assert (
            fwd(_FakeGeluAndMul("none"), torch.empty(4, 8, dtype=torch.bfloat16))
            == "orig"
        )
    assert len([r for r in caplog.records if "fi_gelu kernel" in r.getMessage()]) == 1


# ===================================================================== NPU ==
@pytest.fixture(scope="module")
def vllm_config():
    """Set a default VllmConfig so CustomOps can be constructed in isolation."""
    if not _npu_available():
        pytest.skip("NPU not available")
    import torch
    import torch_npu  # noqa: F401

    torch.npu.set_device(0)
    from vllm.config import VllmConfig, set_current_vllm_config

    with set_current_vllm_config(VllmConfig()):
        yield


def _activation_module():
    return pytest.importorskip("vllm.model_executor.layers.activation")


def _rand_input(tokens: int, inter: int, seed: int = 0):
    torch = pytest.importorskip("torch")
    g = torch.Generator().manual_seed(seed)
    return torch.randn(tokens, 2 * inter, generator=g, dtype=torch.float32).bfloat16()


def _bf16_check(actual, golden, note: str = "") -> dict:
    """Mixed-tolerance check, verbatim per ``float_compute.md`` section 2/4."""
    np = pytest.importorskip("numpy")
    a = np.asarray(actual.float().cpu().numpy(), dtype=np.float64)
    g = np.asarray(golden.float().cpu().numpy(), dtype=np.float64)
    assert a.shape == g.shape, f"shape mismatch: {a.shape} vs {g.shape}"
    abs_err = np.abs(a - g)
    passed = abs_err <= BF16_ATOL + BF16_RTOL * np.abs(g)
    total = a.size
    res = {
        "matched_ratio": 1.0 if total == 0 else float(passed.sum()) / total,
        "max_abs_error": 0.0 if total == 0 else float(abs_err.max()),
    }
    res["is_pass"] = (
        res["matched_ratio"] >= BF16_REQUIRED_MATCHED_RATIO
        and res["max_abs_error"] <= BF16_MAX_ABS_LIMIT
    )
    assert res["is_pass"], (
        f"bf16 mixed-tolerance FAIL{note}: matched_ratio="
        f"{res['matched_ratio']:.6f} (need >= {BF16_REQUIRED_MATCHED_RATIO}), "
        f"max_abs_error={res['max_abs_error']:.6g} (limit {BF16_MAX_ABS_LIMIT})"
    )
    return res


@needs_npu
def test_dispatch_selects_forward_oot_and_patch_replaces_it(
    vllm_config, monkeypatch
) -> None:
    """Dispatch confirmation: the patched name IS the dispatched method."""
    act_mod = _activation_module()
    cls = act_mod.GeluAndMul
    # (1) unpatched: the host dispatch selects the inherited forward_oot
    before = cls(approximate="none")
    assert before._forward_method.__name__ == plugin.FORWARD_TARGET, (
        "NPU dispatch no longer selects "
        f"{plugin.FORWARD_TARGET!r} (got {before._forward_method!r}); re-probe "
        "which forward_* the host selects and update FORWARD_TARGET"
    )
    # (2) patched: the same dispatch slot now points at the plugin's function
    monkeypatch.setenv(ENV, "1")
    assert plugin.load() is True
    after = cls(approximate="none")
    fm = after._forward_method
    assert fm.__name__ == plugin.FORWARD_TARGET
    assert getattr(fm, "__func__", None) is getattr(cls, plugin.FORWARD_TARGET)
    # idempotent
    assert plugin.load() is True
    assert getattr(cls, plugin.FORWARD_TARGET) is fm.__func__
    # unpatch restores the inherited method exactly
    plugin._unpatch()
    assert plugin.FORWARD_TARGET not in cls.__dict__
    assert cls(approximate="none")._forward_method.__name__ == "forward_oot"


@needs_npu
def test_patched_forward_equals_fi_gelu_kernel(vllm_config, monkeypatch) -> None:
    torch = pytest.importorskip("torch")
    act_mod = _activation_module()
    from vllm_ascend_split_batch.fi_gelu import api, reference

    monkeypatch.setenv(ENV, "1")
    assert plugin.load() is True

    op = act_mod.GeluAndMul(approximate="none")
    x = _rand_input(64, 1024, seed=1).npu()
    out = op(x)
    assert out.dtype == torch.bfloat16
    assert out.shape == (64, 1024)
    # the patched path is bit-identical to calling the kernel directly
    assert torch.equal(out, api.gelu_and_mul(x))

    golden = reference.gelu_and_mul_reference(x.cpu(), dtype=torch.float32)
    _bf16_check(out, golden, note=" patched forward vs fp32-erf golden")


@needs_npu
def test_patched_forward_matches_qwen_mlp_shape(vllm_config, monkeypatch) -> None:
    """Real Qwen2.5-14B MLP width (inter=13824) end to end."""
    torch = pytest.importorskip("torch")
    act_mod = _activation_module()
    from vllm_ascend_split_batch.fi_gelu import reference

    monkeypatch.setenv(ENV, "1")
    assert plugin.load() is True

    op = act_mod.GeluAndMul(approximate="none")
    x = _rand_input(8, 13824, seed=2).npu()
    out = op(x)
    assert out.shape == (8, 13824)
    golden = reference.gelu_and_mul_reference(x.cpu(), dtype=torch.float32)
    _bf16_check(out, golden, note=" inter=13824")


@needs_npu
def test_graph_capture_replay_matches_eager(vllm_config, monkeypatch) -> None:
    """The patched forward must survive ACL graph capture and replay exactly."""
    torch = pytest.importorskip("torch")
    act_mod = _activation_module()

    monkeypatch.setenv(ENV, "1")
    assert plugin.load() is True

    op = act_mod.GeluAndMul(approximate="none")
    x = _rand_input(64, 1024, seed=3).npu()
    for _ in range(5):
        op(x)
    torch.npu.synchronize()

    handle = torch.npu.NPUGraph()
    with torch.npu.graph(handle):
        graph_out = op(x)
    # capture records but does not execute: replay first, then snapshot
    for _ in range(5):
        handle.replay()
    torch.npu.synchronize()
    replay_snapshot = graph_out.clone()
    eager = op(x).clone()
    torch.npu.synchronize()

    res = _bf16_check(replay_snapshot, eager, note=" graph replay vs eager")
    assert res["matched_ratio"] == 1.0
    assert float((replay_snapshot.float() - eager.float()).abs().max()) == 0.0


@needs_npu
def test_fail_open_non_bf16_delegates_to_host(vllm_config, monkeypatch) -> None:
    torch = pytest.importorskip("torch")
    import torch.nn.functional as F

    act_mod = _activation_module()
    monkeypatch.setenv(ENV, "1")
    assert plugin.load() is True

    op = act_mod.GeluAndMul(approximate="none")
    # fp32 would make the kernel raise TypeError -> must delegate instead
    x = torch.randn(4, 128, dtype=torch.float32).npu()
    out = op(x)
    d = x.shape[-1] // 2
    ref = F.gelu(x[..., :d], approximate="none") * x[..., d:]
    assert out.dtype == torch.float32
    assert torch.equal(out, ref)


@needs_npu
def test_fail_open_tanh_approximation_delegates_to_host(
    vllm_config, monkeypatch
) -> None:
    torch = pytest.importorskip("torch")
    import torch.nn.functional as F

    act_mod = _activation_module()
    monkeypatch.setenv(ENV, "1")
    assert plugin.load() is True

    op = act_mod.GeluAndMul(approximate="tanh")
    x = _rand_input(8, 512, seed=4).npu()
    out = op(x)
    d = x.shape[-1] // 2
    ref = F.gelu(x[..., :d], approximate="tanh") * x[..., d:]
    assert torch.equal(out, ref)


@needs_npu
def test_host_signature_guard(vllm_config) -> None:
    """Live host-signature guard for the patched method (drift => fail loudly).

    The patch assumes: (a) the seam module/class still exist, (b) the dispatched
    method is still ``forward_oot``, (c) the host calls it with a single
    positional tensor (``self.act_fn(gate_up)``), and (d) the host implementation
    of that method is the ``(self, x)`` adapter our replacement mirrors.
    """
    torch = pytest.importorskip("torch")
    act_mod = _activation_module()
    from vllm.model_executor.custom_op import CustomOp

    assert act_mod.__name__ == plugin.HOST_MODULE
    assert hasattr(act_mod, plugin.HOST_CLASS_NAME)
    cls = act_mod.GeluAndMul

    # (b) live: the dispatch selects FORWARD_TARGET on this host/platform
    assert cls(approximate="none")._forward_method.__name__ == plugin.FORWARD_TARGET

    # (c) live: the host call convention is one positional tensor; both the
    #     host's ``forward`` trampoline and the patched method must bind it.
    tensor = torch.empty(4, 2 * 512, dtype=torch.bfloat16)
    host_fn = getattr(cls, plugin.FORWARD_TARGET)
    inspect.signature(host_fn).bind(object(), tensor)
    inspect.signature(CustomOp.forward).bind(object(), tensor)

    # (d) our replacement keeps the (self, x) arity and must not drift
    repl = plugin._make_forward(host_fn, lambda x: x, torch.bfloat16)
    params = list(inspect.signature(repl).parameters.values())
    assert [p.name for p in params] == ["self", "x"]
    inspect.signature(repl).bind(object(), tensor)
