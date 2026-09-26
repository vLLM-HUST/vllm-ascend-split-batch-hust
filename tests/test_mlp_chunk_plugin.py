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

"""CPU-only tests for ``mlp_chunk_plugin`` (default-off token chunking).

What is covered
---------------
``chunked_mlp_forward`` is the plugin's pure, device-free seam.  These tests pin
its three refusal gates (``chunks_k``, token threshold, SwiGLU whitelist), its
boundary arithmetic (``ceil(n/k)`` with a possibly shorter tail), the row
coverage contract (every row written exactly once, no gap / no overlap),
tensor-attribute preservation, and bitwise equivalence against the unchunked
reference.  They also cover env parsing, ``install``/``load`` semantics,
idempotency, fail-open, and the wrapped-forward behaviour.

Why stubs instead of real objects
---------------------------------
Instantiating the real ``SiluAndMul`` / ``Qwen2MLP`` needs a live vLLM config
(``AssertionError: Current vLLM config is not set``), which is unavailable in a
plain CPU test process.  The plugin only inspects ``act_fn.__class__.__name__``,
so a stub whose ``__name__`` is ``"SiluAndMul"`` exercises the very same branch.
The ``install`` tests inject a stub ``vllm.model_executor.models.qwen2`` module
into ``sys.modules`` so the plugin's lazy ``from ... import Qwen2MLP`` binds the
stub without importing the real vLLM tree.

Division of labour with the NPU-side tests
------------------------------------------
These tests prove the *logic*: gates, boundaries, coverage, and equivalence of
the Python scheduling.  They do not run the real Ascend ``npu_swiglu`` kernel, so
the peak-memory win and fp16 numerical fidelity of the underlying op stay the
responsibility of the on-device tests.  No device, no network.
"""

from __future__ import annotations

import sys
import types

import pytest
import torch

from vllm_ascend_split_batch import mlp_chunk_plugin as plugin
from vllm_ascend_split_batch.mlp_chunk_plugin import (
    _ACT_WHITELIST,
    ENV_CHUNKS,
    ENV_MIN_TOKENS,
    chunked_mlp_forward,
    chunks,
    install,
    load,
    min_tokens,
)


class SiluAndMul:
    """Whitelisted act stub: the plugin matches on the class ``__name__``."""

    def __call__(self, x: torch.Tensor) -> torch.Tensor:
        gate, up = x.chunk(2, dim=-1)
        return torch.nn.functional.silu(gate) * up


class AscendSiluAndMul:
    """Second whitelisted name (the device-side SwiGLU variant).

    Like ``SiluAndMul`` it consumes ``2*inter`` columns and returns ``inter``,
    so the stub keeps the same ``gate_up -> act -> down`` dimension contract.
    """

    def __call__(self, x: torch.Tensor) -> torch.Tensor:
        gate, up = x.chunk(2, dim=-1)
        return torch.nn.functional.silu(gate) * up


class GeluAndMul:
    """Not whitelisted: chunking must be refused for non-SwiGLU activations."""

    def __call__(self, x: torch.Tensor) -> torch.Tensor:
        return torch.nn.functional.gelu(x)


class _Linear:
    """Minimal stand-in for a vLLM Linear layer returning ``(output, bias)``."""

    def __init__(
        self,
        in_features: int,
        out_features: int,
        gen: torch.Generator,
        dtype: torch.dtype = torch.float32,
    ):
        self.weight = torch.randn(in_features, out_features, generator=gen, dtype=dtype)

    def __call__(self, x: torch.Tensor):
        return x @ self.weight, None


class _RecordingIdentity:
    """Identity projection that records the row count of every call."""

    def __init__(self) -> None:
        self.sizes: list[int] = []

    def __call__(self, x: torch.Tensor):
        self.sizes.append(x.shape[0])
        return x, None


class _IdentityAct:
    """Identity activation; class name is patched to a whitelisted one below."""

    def __call__(self, x: torch.Tensor) -> torch.Tensor:
        return x


# The plugin checks only ``type(act_fn).__name__``; give the identity act a
# whitelisted name so coverage/boundary tests can observe chunk boundaries
# without a non-linear activation smearing the row markers.
_IdentityAct.__name__ = "SiluAndMul"


class _RecordingAct:
    """SwiGLU activation stub that counts how many chunks ran."""

    def __init__(self) -> None:
        self.calls = 0

    def __call__(self, x: torch.Tensor) -> torch.Tensor:
        self.calls += 1
        gate, up = x.chunk(2, dim=-1)
        return torch.nn.functional.silu(gate) * up


_RecordingAct.__name__ = "AscendSiluAndMul"


def _mlp(
    hidden: int = 8,
    inter: int = 16,
    act=None,
    seed: int = 0,
    dtype: torch.dtype = torch.float32,
):
    """Build ``(gate_up, act, down)`` stub modules with fixed random weights."""
    gen = torch.Generator().manual_seed(seed)
    gate_up = _Linear(hidden, 2 * inter, gen, dtype)
    down = _Linear(inter, hidden, gen, dtype)
    return gate_up, (SiluAndMul() if act is None else act), down


def _reference(gate_up, act, down, x: torch.Tensor) -> torch.Tensor:
    """Unchunked MLP: the semantics chunking must reproduce bit-for-bit."""
    g, _ = gate_up(x)
    s = act(g)
    o, _ = down(s)
    return o


class _StubQwen2MLP:
    """Stand-in for ``Qwen2MLP`` so ``install()`` needs no real vLLM config."""

    def __init__(self, gate_up_proj, act_fn, down_proj) -> None:
        self.gate_up_proj = gate_up_proj
        self.act_fn = act_fn
        self.down_proj = down_proj
        self.orig_calls = 0

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        self.orig_calls += 1
        return _reference(self.gate_up_proj, self.act_fn, self.down_proj, x)


@pytest.fixture
def fake_qwen2(monkeypatch):
    """Inject a stub ``vllm.model_executor.models.qwen2`` for ``install()``.

    ``install()`` binds ``Qwen2MLP`` via a lazy ``from ... import``.  Putting a
    stub module in ``sys.modules`` makes that bind the stub, so no real vLLM
    import (and no vLLM config) is needed.
    """
    module = types.ModuleType("vllm.model_executor.models.qwen2")
    module.Qwen2MLP = _StubQwen2MLP
    monkeypatch.setitem(sys.modules, module.__name__, module)
    # install() writes the class attribute directly, so register the original
    # forward with monkeypatch to guarantee teardown restores it.
    monkeypatch.setattr(_StubQwen2MLP, "forward", _StubQwen2MLP.forward)
    return module


# --------------------------------------------------------------------------- #
# 1. Gate -- token threshold
# --------------------------------------------------------------------------- #
@pytest.mark.parametrize("n,thr", [(0, 1), (1, 2), (7, 8), (255, 256)])
def test_below_threshold_returns_none(n: int, thr: int) -> None:
    """``n < min_tokens_thr`` must refuse chunking (delegate to orig)."""
    gate_up, act, down = _mlp()
    x = torch.randn(n, 8)
    assert (
        chunked_mlp_forward(gate_up, act, down, x, chunks_k=2, min_tokens_thr=thr)
        is None
    )


@pytest.mark.parametrize("n,thr", [(0, 0), (1, 1), (8, 8), (256, 256)])
def test_at_threshold_chunks(n: int, thr: int) -> None:
    """``n == min_tokens_thr`` is inclusive: chunking applies."""
    gate_up, act, down = _mlp()
    x = torch.randn(n, 8)
    out = chunked_mlp_forward(gate_up, act, down, x, chunks_k=2, min_tokens_thr=thr)
    assert out is not None
    assert torch.equal(out, _reference(gate_up, act, down, x))


def test_just_below_threshold_is_none_but_at_threshold_chunks() -> None:
    """The exact boundary step (thr-1 -> None, thr -> chunked) is preserved."""
    gate_up, act, down = _mlp()
    below = torch.randn(255, 8)
    at = torch.randn(256, 8)
    assert (
        chunked_mlp_forward(gate_up, act, down, below, chunks_k=2, min_tokens_thr=256)
        is None
    )
    assert (
        chunked_mlp_forward(gate_up, act, down, at, chunks_k=2, min_tokens_thr=256)
        is not None
    )


# --------------------------------------------------------------------------- #
# 2. Gate -- chunk count
# --------------------------------------------------------------------------- #
@pytest.mark.parametrize("k", [-5, 0, 1])
def test_chunks_k_below_two_returns_none(k: int) -> None:
    """``chunks_k < 2`` must refuse chunking."""
    gate_up, act, down = _mlp()
    x = torch.randn(16, 8)
    assert (
        chunked_mlp_forward(gate_up, act, down, x, chunks_k=k, min_tokens_thr=0) is None
    )


def test_chunks_k_two_chunks() -> None:
    """``chunks_k == 2`` is the smallest enabled granularity."""
    gate_up, act, down = _mlp()
    x = torch.randn(300, 8)
    out = chunked_mlp_forward(gate_up, act, down, x, chunks_k=2, min_tokens_thr=0)
    assert out is not None
    assert torch.equal(out, _reference(gate_up, act, down, x))


def test_chunks_k_greater_than_n_still_complete() -> None:
    """``k > n`` must not crash and must still cover every row exactly once."""
    n, k, d = 256, 512, 3
    gate_up = _RecordingIdentity()
    down = _RecordingIdentity()
    x = torch.arange(1, n * d + 1, dtype=torch.float32).reshape(n, d)
    out = chunked_mlp_forward(
        gate_up, _IdentityAct(), down, x, chunks_k=k, min_tokens_thr=0
    )
    assert out is not None
    assert torch.equal(out, x)
    assert gate_up.sizes == [1] * n
    assert down.sizes == [1] * n


# --------------------------------------------------------------------------- #
# 3. Gate -- act_fn whitelist
# --------------------------------------------------------------------------- #
def test_whitelist_contains_exactly_the_supported_names() -> None:
    """The whitelist is the plugin's sole structural predicate."""
    assert set(_ACT_WHITELIST) == {"SiluAndMul", "AscendSiluAndMul"}


@pytest.mark.parametrize("act_cls", [SiluAndMul, AscendSiluAndMul])
def test_whitelisted_act_chunks(act_cls) -> None:
    """Every whitelisted SwiGLU class name enables chunking."""
    gate_up, _, down = _mlp(act=act_cls())
    act = act_cls()
    x = torch.randn(300, 8)
    out = chunked_mlp_forward(gate_up, act, down, x, chunks_k=2, min_tokens_thr=0)
    assert out is not None
    assert torch.equal(out, _reference(gate_up, act, down, x))


def test_non_whitelisted_act_returns_none() -> None:
    """A non-SwiGLU activation (name off the whitelist) refuses chunking."""
    gate_up, _, down = _mlp(act=GeluAndMul())
    x = torch.randn(300, 8)
    assert (
        chunked_mlp_forward(
            gate_up, GeluAndMul(), down, x, chunks_k=2, min_tokens_thr=0
        )
        is None
    )


# --------------------------------------------------------------------------- #
# 4. Equivalence -- chunked == unchunked, bit for bit
# --------------------------------------------------------------------------- #
@pytest.mark.parametrize("k", [2, 3, 4])
@pytest.mark.parametrize("n", [256, 257, 300, 511, 512, 1000, 4096])
def test_chunked_matches_unchunked_bitwise(n: int, k: int) -> None:
    """Splitting along tokens must be numerically transparent (incl. n%k != 0)."""
    gate_up, act, down = _mlp()
    x = torch.randn(n, 8)
    out = chunked_mlp_forward(gate_up, act, down, x, chunks_k=k, min_tokens_thr=0)
    assert out is not None
    assert torch.equal(out, _reference(gate_up, act, down, x))


@pytest.mark.parametrize("dtype", [torch.float32, torch.float16, torch.float64])
def test_chunked_matches_unchunked_bitwise_across_dtypes(dtype) -> None:
    """Bitwise transparency holds for fp16 / fp32 / fp64."""
    gate_up, act, down = _mlp(dtype=dtype)
    x = torch.randn(511, 8, dtype=dtype)
    out = chunked_mlp_forward(gate_up, act, down, x, chunks_k=3, min_tokens_thr=0)
    assert out is not None
    assert torch.equal(out, _reference(gate_up, act, down, x))


# --------------------------------------------------------------------------- #
# 5. Coverage / no-overlap and tail boundaries
# --------------------------------------------------------------------------- #
def test_every_row_written_exactly_once() -> None:
    """Output rows are fully covered and never double-written.

    An identity pipeline carrying row-index markers makes any gap show up as
    unwritten (garbage) memory, while the recorded call sizes prove there is no
    overlap.
    """
    n, k, d = 1000, 3, 3
    gate_up = _RecordingIdentity()
    down = _RecordingIdentity()
    markers = torch.arange(1, n * d + 1, dtype=torch.float32).reshape(n, d)
    out = chunked_mlp_forward(
        gate_up, _IdentityAct(), down, markers, chunks_k=k, min_tokens_thr=0
    )
    assert out is not None
    assert torch.equal(out, markers)
    assert gate_up.sizes == [334, 334, 332]
    assert down.sizes == [334, 334, 332]
    assert sum(gate_up.sizes) == n


@pytest.mark.parametrize(
    "n,k,expected",
    [
        (1000, 3, [334, 334, 332]),
        (257, 3, [86, 86, 85]),
        (512, 4, [128, 128, 128, 128]),
        (300, 4, [75, 75, 75, 75]),
    ],
)
def test_chunk_sizes_follow_ceil_with_shorter_tail(n, k, expected) -> None:
    """Chunk sizes are ``ceil(n/k)`` with a shortened (possibly empty) tail."""
    gate_up = _RecordingIdentity()
    down = _RecordingIdentity()
    x = torch.arange(1, n * 2 + 1, dtype=torch.float32).reshape(n, 2)
    out = chunked_mlp_forward(
        gate_up, _IdentityAct(), down, x, chunks_k=k, min_tokens_thr=0
    )
    assert out is not None
    assert gate_up.sizes == expected
    assert sum(gate_up.sizes) == n
    assert torch.equal(out, x)


# --------------------------------------------------------------------------- #
# 6. Tensor attributes
# --------------------------------------------------------------------------- #
@pytest.mark.parametrize("dtype", [torch.float32, torch.float16])
def test_output_preserves_shape_dtype_device_and_row_count(dtype) -> None:
    """``empty_like`` semantics: shape/dtype/device and row count are kept."""
    n, k = 300, 3
    gate_up, act, down = _mlp(dtype=dtype)
    x = torch.randn(n, 8, dtype=dtype)
    out = chunked_mlp_forward(gate_up, act, down, x, chunks_k=k, min_tokens_thr=0)
    assert out is not None
    assert out.shape == x.shape
    assert out.dtype == x.dtype
    assert out.device == x.device
    assert out.shape[0] == x.shape[0] == n


# --------------------------------------------------------------------------- #
# 7. Dimension generalization
# --------------------------------------------------------------------------- #
@pytest.mark.parametrize("hidden,inter", [(8, 16), (16, 32), (64, 128)])
def test_equivalence_across_hidden_and_intermediate(hidden, inter) -> None:
    """Equivalence is independent of the hidden / intermediate dimensions."""
    gate_up, act, down = _mlp(hidden=hidden, inter=inter)
    x = torch.randn(300, hidden)
    out = chunked_mlp_forward(gate_up, act, down, x, chunks_k=3, min_tokens_thr=0)
    assert out is not None
    assert torch.equal(out, _reference(gate_up, act, down, x))


# --------------------------------------------------------------------------- #
# 8. Environment parsing
# --------------------------------------------------------------------------- #
@pytest.mark.parametrize(
    "value,expected",
    [(None, 1), ("1", 1), ("0", 1), ("-3", 1), ("abc", 1), ("2", 2), ("4", 4)],
)
def test_chunks_env_parsing(monkeypatch, value, expected) -> None:
    """``chunks()``: unset / <2 / non-numeric all clamp to 1; else the value."""
    monkeypatch.delenv(ENV_CHUNKS, raising=False)
    if value is not None:
        monkeypatch.setenv(ENV_CHUNKS, value)
    assert chunks() == expected


@pytest.mark.parametrize(
    "value,expected",
    [(None, 256), ("abc", 256), ("", 256), ("0", 0), ("128", 128), ("512", 512)],
)
def test_min_tokens_env_parsing(monkeypatch, value, expected) -> None:
    """``min_tokens()``: unset / non-numeric fall back to the default."""
    monkeypatch.delenv(ENV_MIN_TOKENS, raising=False)
    if value is not None:
        monkeypatch.setenv(ENV_MIN_TOKENS, value)
    assert min_tokens() == expected


def test_min_tokens_default_matches_module_constant() -> None:
    """The undocumented default is the documented 256-token threshold."""
    assert plugin._DEFAULT_MIN_TOKENS == 256


# --------------------------------------------------------------------------- #
# 9. install / load semantics
# --------------------------------------------------------------------------- #
def test_load_default_off_when_env_unset(monkeypatch) -> None:
    """Default-off: no env -> ``load()`` is False and patches nothing."""
    monkeypatch.delenv(ENV_CHUNKS, raising=False)
    before = _StubQwen2MLP.forward
    assert load() is False
    assert _StubQwen2MLP.forward is before


@pytest.mark.parametrize("value", ["1", "0", "abc", "-3"])
def test_load_default_off_for_disabled_env(monkeypatch, value) -> None:
    """Any env value that clamps to <2 leaves the host untouched."""
    monkeypatch.setenv(ENV_CHUNKS, value)
    before = _StubQwen2MLP.forward
    assert load() is False
    assert _StubQwen2MLP.forward is before


def test_load_installs_when_env_ge_two(monkeypatch, fake_qwen2) -> None:
    """Env >= 2: ``load()`` is True, forward is replaced and marked."""
    monkeypatch.setenv(ENV_CHUNKS, "2")
    before = _StubQwen2MLP.forward
    assert load() is True
    assert _StubQwen2MLP.forward is not before
    assert getattr(_StubQwen2MLP.forward, plugin._MARKER, False) is True


def test_install_is_idempotent(monkeypatch, fake_qwen2) -> None:
    """A second ``install()`` must not wrap the already-wrapped forward."""
    monkeypatch.setenv(ENV_CHUNKS, "2")
    assert install() is True
    wrapped = _StubQwen2MLP.forward
    assert install() is True
    assert _StubQwen2MLP.forward is wrapped


def test_install_default_off_returns_false(monkeypatch, fake_qwen2) -> None:
    """``install()`` itself honours default-off (no env -> False)."""
    monkeypatch.delenv(ENV_CHUNKS, raising=False)
    before = _StubQwen2MLP.forward
    assert install() is False
    assert _StubQwen2MLP.forward is before


def test_install_fail_open_when_target_import_missing(monkeypatch) -> None:
    """A missing host target must return False, not raise (fail-open)."""
    monkeypatch.setenv(ENV_CHUNKS, "2")
    monkeypatch.setitem(sys.modules, "vllm.model_executor.models.qwen2", None)
    assert install() is False


def test_install_fail_open_when_target_not_callable(monkeypatch, fake_qwen2) -> None:
    """A non-callable ``forward`` must return False, not raise."""
    monkeypatch.setenv(ENV_CHUNKS, "2")
    monkeypatch.setattr(_StubQwen2MLP, "forward", None)
    assert install() is False


# --------------------------------------------------------------------------- #
# 10. Wrapped-forward behaviour
# --------------------------------------------------------------------------- #
def test_wrapped_forward_matches_reference_above_threshold(
    monkeypatch, fake_qwen2
) -> None:
    """Above threshold the wrapper uses the chunked path and skips the orig."""
    monkeypatch.setenv(ENV_CHUNKS, "2")
    monkeypatch.setenv(ENV_MIN_TOKENS, "256")
    assert install() is True
    gate_up, act, down = _mlp()
    mlp = _StubQwen2MLP(gate_up, act, down)
    x = torch.randn(300, 8)
    out = mlp.forward(x)
    assert mlp.orig_calls == 0
    assert torch.equal(out, _reference(gate_up, act, down, x))


def test_wrapped_forward_delegates_to_original_below_threshold(
    monkeypatch, fake_qwen2
) -> None:
    """Below threshold the wrapper falls back to the original forward (spied)."""
    monkeypatch.setenv(ENV_CHUNKS, "2")
    monkeypatch.setenv(ENV_MIN_TOKENS, "256")
    assert install() is True
    gate_up, act, down = _mlp()
    mlp = _StubQwen2MLP(gate_up, act, down)
    x = torch.randn(64, 8)
    out = mlp.forward(x)
    assert mlp.orig_calls == 1
    assert torch.equal(out, _reference(gate_up, act, down, x))


def test_wrapped_forward_honours_env_chunk_count(monkeypatch, fake_qwen2) -> None:
    """The wrapper reads ``chunks_k`` from the env at install time."""
    monkeypatch.setenv(ENV_CHUNKS, "4")
    monkeypatch.setenv(ENV_MIN_TOKENS, "0")
    assert install() is True
    act = _RecordingAct()
    gate_up, _, down = _mlp()
    mlp = _StubQwen2MLP(gate_up, act, down)
    x = torch.randn(1024, 8)
    out = mlp.forward(x)
    assert act.calls == 4
    assert mlp.orig_calls == 0
    assert torch.equal(out, _reference(gate_up, act, down, x))


def test_wrapped_forward_honours_env_min_tokens(monkeypatch, fake_qwen2) -> None:
    """The wrapper reads ``min_tokens_thr`` from the env at install time."""
    monkeypatch.setenv(ENV_CHUNKS, "2")
    monkeypatch.setenv(ENV_MIN_TOKENS, "1024")
    assert install() is True
    act = _RecordingAct()
    gate_up, _, down = _mlp()
    mlp = _StubQwen2MLP(gate_up, act, down)
    x = torch.randn(512, 8)
    mlp.forward(x)
    # 512 < 1024 -> chunked path refused; the original ran exactly once, so the
    # act fires once (not once per chunk as it would with k=2 chunking).
    assert act.calls == 1
    assert mlp.orig_calls == 1
