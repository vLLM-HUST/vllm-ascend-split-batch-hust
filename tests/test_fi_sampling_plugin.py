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

"""CPU-only tests for the fi_sampling plugin glue (default-off contract).

The routing decisions themselves are covered in ``test_fi_sampling_route.py``;
here we test the integration contract:

- default-off: ``load()`` returns False, patches nothing and imports neither
  ``vllm_ascend.sample.sampler`` nor ``triton`` (subprocess assertion);
- on: ``load()`` replaces the module attribute with a subclass, and a
  subclass instance delegates every fallback / unmeasured cell to the base
  ``forward_native`` verbatim;
- the FI routes call the kernel API with fp32 probs and an advancing seed;
- a kernel failure falls back to the base chain instead of raising.
"""

import logging
import os
import subprocess
import sys
import types

import pytest
import torch

from vllm_ascend_split_batch import fi_sampling_plugin as plugin
from vllm_ascend_split_batch import fi_sampling_route as route

ENV_KEYS = (
    plugin.ENV_ENABLE,
    plugin.ENV_SEED,
    plugin.ENV_JOINT_MIN_BATCH,
    plugin.ENV_JOINT_MIN_K,
)


class _RecordingBase:
    """Stand-in for ``AscendTopKTopPSampler`` (records delegation)."""

    def __init__(self, logprobs_mode="raw_logprobs"):
        self.logprobs_mode = logprobs_mode
        self.delegated = []

    def forward_native(self, logits, generators, k, p):
        self.delegated.append((logits, generators, k, p))
        return torch.zeros(logits.shape[0], dtype=torch.long), "fork"


class _StubFiApi:
    def __init__(self):
        self.calls = []

    def sampling_from_probs(self, probs, **kwargs):
        self.calls.append(("api1", probs, kwargs))
        return torch.arange(probs.shape[0], dtype=torch.int32)

    def top_k_top_p_sampling_from_probs(self, probs, **kwargs):
        self.calls.append(("joint", probs, kwargs))
        return torch.arange(probs.shape[0], dtype=torch.int32)


@pytest.fixture
def no_fallbacks(monkeypatch):
    monkeypatch.setenv(plugin.ENV_ENABLE, "1")
    monkeypatch.setattr(plugin, "_fallback_flags", lambda: (False, False, False))


@pytest.fixture
def stub_api(monkeypatch):
    stub = _StubFiApi()
    monkeypatch.setattr(plugin, "_load_fi_api", lambda: stub)
    return stub


@pytest.fixture
def sampler_cls(monkeypatch):
    plugin._reset_for_tests()
    monkeypatch.setenv(plugin.ENV_ENABLE, "1")
    return plugin._build_sampler_class(_RecordingBase, torch)


def _logits(rows: int, vocab: int = 8) -> torch.Tensor:
    return torch.randn(rows, vocab, dtype=torch.float32)


def test_load_default_off_patches_nothing_and_imports_nothing() -> None:
    """Default-off: no env, no patch, no host/kernel import."""
    code = (
        "import sys;"
        "import vllm_ascend_split_batch.fi_sampling_plugin as p;"
        "assert p.load() is False;"
        "assert 'vllm_ascend.sample.sampler' not in sys.modules, 'host imported';"
        "assert 'triton' not in sys.modules, 'kernel imported';"
        "assert p._fi_api is None"
    )
    result = subprocess.run(
        [sys.executable, "-c", code],
        cwd="/tmp",
        capture_output=True,
        text=True,
    )
    assert result.returncode == 0, result.stderr


def test_install_replaces_the_module_attribute_with_a_subclass(monkeypatch) -> None:
    import vllm_ascend.sample.sampler as sampler_mod

    original = sampler_mod.AscendTopKTopPSampler
    monkeypatch.setattr(
        sampler_mod, "_fi_sampling_plugin_patched", False, raising=False
    )
    monkeypatch.setattr(sampler_mod, "AscendTopKTopPSampler", original)
    monkeypatch.setenv(plugin.ENV_ENABLE, "1")

    assert plugin.load() is True
    patched = sampler_mod.AscendTopKTopPSampler
    assert patched is not original
    assert issubclass(patched, original)
    assert patched.__name__ == "FiSamplingTopKTopPSampler"
    # Idempotent: a second load keeps the same class object.
    assert plugin.load() is True
    assert sampler_mod.AscendTopKTopPSampler is patched


def test_install_is_fail_open_when_the_replacement_fails(monkeypatch) -> None:
    import vllm_ascend.sample.sampler as sampler_mod

    monkeypatch.setattr(
        sampler_mod, "_fi_sampling_plugin_patched", False, raising=False
    )
    monkeypatch.setattr(
        plugin,
        "_build_sampler_class",
        lambda *a, **k: (_ for _ in ()).throw(RuntimeError("boom")),
    )
    assert plugin.install() is False


def test_untruncated_route_calls_the_api1_kernel(
    sampler_cls, no_fallbacks, stub_api
) -> None:
    sampler = sampler_cls()
    tokens, logprobs = sampler.forward_native(_logits(4), {}, None, None)

    assert logprobs is None
    # int32 from the kernel; Sampler.forward casts to long right after.
    assert tokens.dtype == torch.int32
    assert tokens.shape == (4,)
    ((kind, probs, kwargs),) = stub_api.calls
    assert kind == "api1"
    assert probs.dtype == torch.float32
    assert kwargs["block_v"] == route.FI_API1_BLOCK_V
    assert sampler.delegated == []


def test_seed_advances_between_calls(sampler_cls, no_fallbacks, stub_api) -> None:
    """The FI seed must not repeat per decode step (review pitfall)."""
    sampler = sampler_cls()
    sampler.forward_native(_logits(2), {}, None, None)
    sampler.forward_native(_logits(2), {}, None, None)
    seeds = [kwargs["seed"] for _, _, kwargs in stub_api.calls]
    assert len(seeds) == 2
    assert seeds[0] != seeds[1]


def test_joint_route_forwards_the_host_top_k_tensor(
    sampler_cls, no_fallbacks, stub_api, monkeypatch
) -> None:
    monkeypatch.setattr(plugin, "_host_top_k_extremes", lambda k: (50, 50))
    sampler = sampler_cls()
    k = torch.full((256,), 50, dtype=torch.int32)
    p = torch.full((256,), 0.95, dtype=torch.float32)
    sampler.forward_native(_logits(256), {}, k, p)

    ((kind, probs, kwargs),) = stub_api.calls
    assert kind == "joint"
    assert kwargs["block_v"] == route.FI_JOINT_BLOCK_V
    assert kwargs["top_k"] is k
    assert kwargs["top_p"] is p
    assert probs.dtype == torch.float32


def test_argmax_route_skips_both_chains(
    sampler_cls, no_fallbacks, stub_api, monkeypatch
) -> None:
    """k=1 rows take argmax where the top-k read is taken (large batch)."""
    monkeypatch.setattr(plugin, "_host_top_k_extremes", lambda k: (1, 1))
    sampler = sampler_cls()
    logits = _logits(256)
    tokens, logprobs = sampler.forward_native(
        logits, {}, torch.ones(256, dtype=torch.int32), None
    )

    assert logprobs is None
    assert torch.equal(tokens, logits.argmax(dim=-1))
    assert stub_api.calls == []
    assert sampler.delegated == []


def test_k1_argmax_can_be_disabled_for_bit_exact_ties(
    sampler_cls, stub_api, monkeypatch
) -> None:
    """``VLLM_HUST_FI_SAMPLING_K1_ARGMAX=0`` keeps k=1 on the fork chain."""
    monkeypatch.setattr(plugin, "_host_top_k_extremes", lambda k: (1, 1))
    monkeypatch.setattr(plugin, "_fallback_flags", lambda: (False, False, False))
    monkeypatch.setenv(plugin.ENV_K1_ARGMAX, "0")
    sampler = sampler_cls()
    sampler.forward_native(_logits(256), {}, torch.ones(256, dtype=torch.int32), None)
    assert len(sampler.delegated) == 1
    assert stub_api.calls == []


@pytest.mark.parametrize(
    "kwargs",
    [
        {"generators": {0: object()}},
        {"logprobs_mode": "processed_logprobs"},
        {"batch_invariant": True},
        {"reduce_sample": True},
        {"async_exponential": True},
    ],
)
def test_fallback_cells_delegate_to_the_base_chain(
    sampler_cls, stub_api, monkeypatch, kwargs
) -> None:
    """Each fallback class must reach the fork chain unchanged."""
    batch_invariant = kwargs.pop("batch_invariant", False)
    reduce_sample = kwargs.pop("reduce_sample", False)
    async_exponential = kwargs.pop("async_exponential", False)
    logprobs_mode = kwargs.pop("logprobs_mode", "raw_logprobs")
    generators = kwargs.pop("generators", {})
    monkeypatch.setattr(
        plugin,
        "_fallback_flags",
        lambda: (batch_invariant, reduce_sample, async_exponential),
    )
    sampler = sampler_cls(logprobs_mode=logprobs_mode)
    logits = _logits(2)
    sampler.forward_native(logits, generators, None, None)

    assert stub_api.calls == []
    assert len(sampler.delegated) == 1
    assert sampler.delegated[0][0] is logits


def test_fallback_cells_never_read_top_k(monkeypatch) -> None:
    """A fallback leg must not pay for a device->host read it will not use."""
    plugin._reset_for_tests()
    monkeypatch.setenv(plugin.ENV_ENABLE, "1")
    cls = plugin._build_sampler_class(_RecordingBase, torch)
    sampler = cls()

    def _boom(_k):
        raise AssertionError("fallback path must not read top-k from the device")

    monkeypatch.setattr(plugin, "_host_top_k_extremes", _boom)
    monkeypatch.setattr(plugin, "_fallback_flags", lambda: (False, False, True))

    k = torch.full((256,), 50, dtype=torch.int32)
    p = torch.full((256,), 0.95, dtype=torch.float32)
    sampler.forward_native(_logits(256), {}, k, p)
    assert len(sampler.delegated) == 1


def test_small_batch_top_k_cell_is_sync_free(monkeypatch) -> None:
    """B < joint_min_batch -> fork chain without any device read (REPORT §4)."""
    plugin._reset_for_tests()
    monkeypatch.setenv(plugin.ENV_ENABLE, "1")
    cls = plugin._build_sampler_class(_RecordingBase, torch)
    sampler = cls()

    def _boom(_k):
        raise AssertionError("this cell resolves without a device read")

    monkeypatch.setattr(plugin, "_host_top_k_extremes", _boom)
    monkeypatch.setattr(plugin, "_fallback_flags", lambda: (False, False, False))

    k = torch.full((8,), 50, dtype=torch.int32)
    p = torch.full((8,), 0.95, dtype=torch.float32)
    sampler.forward_native(_logits(8), {}, k, p)
    assert len(sampler.delegated) == 1


def test_kernel_failure_falls_back_to_the_base_chain(
    sampler_cls, no_fallbacks, monkeypatch
) -> None:
    def boom(*args, **kwargs):
        raise RuntimeError("kernel exploded")

    monkeypatch.setattr(
        plugin,
        "_load_fi_api",
        lambda: types.SimpleNamespace(
            sampling_from_probs=boom,
            top_k_top_p_sampling_from_probs=boom,
        ),
    )
    sampler = sampler_cls()
    sampler.forward_native(_logits(2), {}, None, None)
    assert len(sampler.delegated) == 1


def test_forward_native_is_a_second_gate(monkeypatch) -> None:
    """With the env cleared mid-process, the patched class stops engaging."""
    plugin._reset_for_tests()
    cls = plugin._build_sampler_class(_RecordingBase, torch)
    sampler = cls()
    monkeypatch.delenv(plugin.ENV_ENABLE, raising=False)
    sampler.forward_native(_logits(2), {}, None, None)
    assert len(sampler.delegated) == 1


def test_env_knobs_override_the_bench_thresholds(monkeypatch) -> None:
    monkeypatch.setenv(plugin.ENV_JOINT_MIN_BATCH, "16")
    monkeypatch.setenv(plugin.ENV_JOINT_MIN_K, "64")
    assert plugin._joint_min_batch() == 16
    assert plugin._joint_min_k() == 64
    for key in (plugin.ENV_JOINT_MIN_BATCH, plugin.ENV_JOINT_MIN_K):
        monkeypatch.delenv(key)
    assert plugin._joint_min_batch() == route.FI_JOINT_MIN_BATCH
    assert plugin._joint_min_k() == route.FI_JOINT_MIN_K


def test_host_top_k_extremes_reads_a_cpu_tensor() -> None:
    assert plugin._host_top_k_extremes(torch.tensor([50, 1, 8])) == (1, 50)
    assert plugin._host_top_k_extremes(None) == (None, None)


# --- review F1: the HOST import itself failing must stay fail-open -----------


def test_install_is_fail_open_when_host_module_is_missing(monkeypatch) -> None:
    """A missing/unimportable ``vllm_ascend.sample.sampler`` must not raise.

    ``None`` in ``sys.modules`` makes the next ``import`` raise ImportError —
    exactly what a future fork rename/layout change would do to us.
    """
    monkeypatch.setenv(plugin.ENV_ENABLE, "1")
    monkeypatch.setitem(sys.modules, "vllm_ascend.sample.sampler", None)
    assert plugin.install() is False


def test_load_is_fail_open_when_host_module_is_missing(monkeypatch) -> None:
    """The entry point returns False (never raises) with the switch on."""
    monkeypatch.setenv(plugin.ENV_ENABLE, "1")
    monkeypatch.setitem(sys.modules, "vllm_ascend.sample.sampler", None)
    assert plugin.load() is False


def test_load_process_survives_a_missing_host_module() -> None:
    """Process-level F1 proof: the plugin loader must not die at startup."""
    code = (
        "import sys;"
        "sys.modules['vllm_ascend.sample.sampler'] = None;"
        "import vllm_ascend_split_batch.fi_sampling_plugin as p;"
        "assert p.load() is False"
    )
    env = dict(os.environ)
    env[plugin.ENV_ENABLE] = "1"
    result = subprocess.run(
        [sys.executable, "-c", code],
        cwd="/tmp",
        env=env,
        capture_output=True,
        text=True,
    )
    assert result.returncode == 0, result.stderr


def test_fallback_flags_fail_closed_when_batch_invariant_probe_raises(
    monkeypatch,
) -> None:
    """Runtime host probing failures route to the fork chain, not crash."""

    def _boom() -> bool:
        raise RuntimeError("vllm.envs unavailable")

    monkeypatch.setattr(plugin, "_batch_invariant", _boom)
    assert plugin._fallback_flags() == (True, True, True)


def test_fallback_flags_fail_closed_when_ascend_config_raises(monkeypatch) -> None:
    def _boom():
        raise RuntimeError("ascend config unavailable")

    monkeypatch.setattr(plugin, "_ascend_config", _boom)
    assert plugin._fallback_flags() == (True, True, True)


def test_missing_async_exponential_knob_is_host_drift_not_a_fallback(
    monkeypatch, caplog
) -> None:
    """A knob the host removed must NOT silently kill the FI path.

    New baseline (vllm-ascend 0.25.1rc2): ``AscendConfig`` dropped
    ``enable_async_exponential`` (upstream ``4f0a38a95``).  Plain attribute
    access made ``_fallback_flags`` raise -> fail CLOSED -> 100% fork routing
    while the plugin still logged itself ACTIVE (caught by the 2026-09-10 e2e
    refresh).  The absent knob now reads as "feature gone" = disabled, and the
    drift is warned once.
    """
    plugin._reset_for_tests()
    config = types.SimpleNamespace(enable_reduce_sample=False)
    monkeypatch.setattr(plugin, "_ascend_config", lambda: config)
    monkeypatch.setattr(plugin, "_batch_invariant", lambda: False)
    with caplog.at_level(logging.WARNING):
        assert plugin._fallback_flags() == (False, False, False)
    assert "enable_async_exponential" in caplog.text

    # second call stays quiet (one warning per knob, not per decode step)
    caplog.clear()
    with caplog.at_level(logging.WARNING):
        plugin._fallback_flags()
    assert caplog.text == ""


def test_present_reduce_sample_knob_still_falls_back(monkeypatch) -> None:
    """Tolerating an absent knob must not disable a knob that is really on."""
    plugin._reset_for_tests()
    config = types.SimpleNamespace(enable_reduce_sample=True)
    monkeypatch.setattr(plugin, "_ascend_config", lambda: config)
    monkeypatch.setattr(plugin, "_batch_invariant", lambda: False)
    assert plugin._fallback_flags() == (False, True, False)


# --- review follow-up: garbage env VALUES must never raise either -----------


def test_garbage_enable_value_is_treated_as_off_in_process() -> None:
    """``VLLM_HUST_FI_SAMPLING=true`` must disable, not crash the loader."""
    code = (
        "import vllm_ascend_split_batch.fi_sampling_plugin as p;"
        "assert p.load() is False"
    )
    env = dict(os.environ)
    env[plugin.ENV_ENABLE] = "true"  # not an int
    result = subprocess.run(
        [sys.executable, "-c", code],
        cwd="/tmp",
        env=env,
        capture_output=True,
        text=True,
    )
    assert result.returncode == 0, result.stderr


def test_garbage_threshold_falls_back_to_the_default(monkeypatch) -> None:
    monkeypatch.setenv(plugin.ENV_JOINT_MIN_BATCH, "abc")
    monkeypatch.setenv(plugin.ENV_JOINT_MIN_K, "abc")
    assert plugin._joint_min_batch() == route.FI_JOINT_MIN_BATCH
    assert plugin._joint_min_k() == route.FI_JOINT_MIN_K
    assert plugin._k1_argmax() is True  # garbage K1_ARGMAX keeps the default


def test_install_fully_succeeds_with_garbage_thresholds(
    monkeypatch,
) -> None:
    """A bad knob must not abort (or half-abort) the installation."""
    import vllm_ascend.sample.sampler as sampler_mod

    original = sampler_mod.AscendTopKTopPSampler
    monkeypatch.setattr(
        sampler_mod, "_fi_sampling_plugin_patched", False, raising=False
    )
    monkeypatch.setattr(sampler_mod, "AscendTopKTopPSampler", original)
    monkeypatch.setenv(plugin.ENV_ENABLE, "1")
    monkeypatch.setenv(plugin.ENV_JOINT_MIN_BATCH, "abc")

    assert plugin.install() is True
    patched = sampler_mod.AscendTopKTopPSampler
    assert patched is not original
    assert sampler_mod._fi_sampling_plugin_patched is True


def test_garbage_seed_falls_back_to_torch_seed(monkeypatch) -> None:
    plugin._reset_for_tests()
    monkeypatch.delenv(plugin.ENV_SEED, raising=False)
    monkeypatch.setenv(plugin.ENV_SEED, "not-a-number")
    first = plugin._next_seed()
    second = plugin._next_seed()
    assert first != second  # seed still advances; no raise
