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

"""CPU tests for the GDN ``in_proj`` merge carrier (``gdn_merge_inproj_plugin``).

What is covered
---------------
The carrier migrates the accepted **F2** mechanism (load-time
``in_proj_ba`` -> ``in_proj_qkvz`` weight merge, one ``[6176, 2048]`` matmul per
GDN layer) out of the vllm-ascend runtime tree and into this plugin library.
These tests pin:

* default-off: ``load()`` without ``VLLM_HUST_GDN_MERGE_INPROJ=1`` imports
  nothing and patches nothing (the env token must be exactly ``1``);
* the ``load_model`` seam: wrapping, argument/return pass-through, post-load
  timing only on a *successful* load, idempotency, chaining with another
  plugin's wrapper (``gate_fp32_plugin`` wraps the same method), fail-open
  paths and the ``uninstall()`` recovery;
* the load-time conversion: row order (qkvz first, ba appended), no input
  mutation, and the strict guards (host class, layout, already-merged, dims,
  rows, dtype, device);
* the fail-closed policy: without a forward consumer seam the carrier rewrites
  **no** weight, warns once and leaves the host model bit-identical to stock --
  and it merges properly once such a seam exists (``forward_consumer_available``
  is monkeypatched to prove the merge path is real and not dead code);
* the descriptor wiring (``vllm.general_plugins`` entry point) and the probe
  order.

Why stubs instead of real objects
---------------------------------
Loading the real ``NPUModelRunner``/GDN layer needs a live vLLM config and an
NPU, and wrapping the *real* host class here could leak across tests.  The
carrier resolves the host GDN class by name and the runner by ``getattr``, so a
stub module is injected into ``sys.modules`` for both: that also guarantees the
tests never touch a real host class.  Only ``torch`` (CPU) is required -- no
``torch_npu``, no ``vllm``, no device.
"""

from __future__ import annotations

import inspect
import json
import logging
import subprocess
import sys
import textwrap
import types
from pathlib import Path

import pytest

try:  # Python 3.11+
    import tomllib
except ModuleNotFoundError:  # pragma: no cover -- Python 3.10 (CI matrix leg)
    import tomli as tomllib  # type: ignore[no-redef]

from vllm_ascend_split_batch import gdn_merge_inproj_plugin as plugin

torch = pytest.importorskip("torch", reason="the weight conversion needs torch")

ENV = plugin.ENV_ENABLE
_REPO_ROOT = Path(plugin.__file__).resolve().parents[2]

_EXPECTED_CANDIDATES = (
    ("vllm_ascend.worker.model_runner_v1", "NPUModelRunner"),
    ("vllm_ascend.worker.v2.model_runner", "NPUModelRunner"),
    ("vllm_ascend._310p.worker.v2.model_runner", "NPUModelRunner310V2"),
    ("vllm_ascend._310p.model_runner_310", "NPUModelRunner310"),
    ("vllm.v1.worker.gpu_model_runner", "GPUModelRunner"),
    ("vllm.v1.worker.gpu.model_runner", "GPUModelRunner"),
)


# ------------------------------------------------------------------ stubs
class _InProj(torch.nn.Module):
    """A stand-in for the host's ``MergedColumnParallelLinear`` projection."""

    def __init__(self, rows, hidden=2048, dtype=torch.bfloat16, device="cpu"):
        super().__init__()
        self.weight = torch.nn.Parameter(
            torch.empty(rows, hidden, dtype=dtype, device=device), requires_grad=False
        )


def _stub_host_gdn_class(monkeypatch):
    """Inject a stub host module carrying the confirmed GDN class name."""
    cls = type("QwenGatedDeltaNetAttention", (torch.nn.Module,), {})
    module = types.ModuleType(plugin._HOST_MODULE)
    module.QwenGatedDeltaNetAttention = cls
    monkeypatch.setitem(sys.modules, plugin._HOST_MODULE, module)
    return cls


def _gdn_layer(cls, **overrides):
    """A structurally F2 layer: flat layout, 6144/32 rows, bf16, TP=2."""
    attributes = {
        "key_dim": 2048,
        "value_dim": 4096,
        "tp_size": 2,
        "num_v_heads": 32,
        "gqa_interleaved_layout": False,
        "in_proj_qkvz": _InProj(plugin.QKVZ_OUT_ROWS),
        "in_proj_ba": _InProj(plugin.BA_OUT_ROWS),
    }
    attributes.update(overrides)
    layer = cls()
    for name, value in attributes.items():
        setattr(layer, name, value)
    return layer


def _model(**children: torch.nn.Module) -> torch.nn.Module:
    root = torch.nn.Module()
    for name, child in children.items():
        root.add_module(name, child)
    return root


def _make_runner():
    """A fresh ``load_model`` host class per call (class identity is state)."""

    class _Runner:
        def __init__(self):
            self.args = None
            self.kwargs = None

        def load_model(self, *args, **kwargs):
            self.args, self.kwargs = args, kwargs
            return "ORIGINAL"

    return _Runner


def _make_model_runner(layer):
    """A runner whose ``load_model`` materializes ``self.model`` (real timing)."""

    class _Runner:
        def __init__(self):
            self.model = None

        def load_model(self, *args, **kwargs):
            self.model = _model(block=_model(gdn=layer))
            return "LOADED"

    return _Runner


def _make_failing_runner(layer):
    """A runner whose ``load_model`` fails after building the model."""

    class _Runner:
        def __init__(self):
            self.model = None

        def load_model(self, *args, **kwargs):
            self.model = _model(gdn=layer)
            raise RuntimeError("loader exploded")

    return _Runner


def _stub_candidates(monkeypatch, target_cls=None, target_module=None):
    """Replace every probed candidate module with an inert stub.

    ``install()`` early-continues on ``AttributeError``, so only the module named
    ``target_module`` receives ``target_cls``; every other probe misses cleanly
    instead of importing a real host module.
    """
    for module_name, class_name in plugin._CANDIDATES:
        module = types.ModuleType(module_name)
        if target_cls is not None and module_name == target_module:
            setattr(module, class_name, target_cls)
        monkeypatch.setitem(sys.modules, module_name, module)


@pytest.fixture(autouse=True)
def _reset_plugin_state(monkeypatch):
    monkeypatch.delenv(ENV, raising=False)
    monkeypatch.setattr(plugin, "_WARNED_FAIL_CLOSED", False)
    plugin._LOCAL.installing = False
    yield
    plugin._LOCAL.installing = False
    plugin.uninstall()
    assert plugin._INSTALLED == []


@pytest.fixture
def consumer_available(monkeypatch):
    """Pretend the host exposes a safe forward consumer seam."""
    monkeypatch.setattr(plugin, "forward_consumer_available", lambda: True)


# ------------------------------------------------------------- env / default-off
@pytest.mark.parametrize(
    "value,expected",
    [
        (None, False),
        ("1", True),
        ("0", False),
        ("true", False),
        ("YES", False),
        ("", False),
        (" 1", False),
        ("1 ", False),
        ("2", False),
    ],
)
def test_is_enabled_only_accepts_the_exact_token(monkeypatch, value, expected):
    """Only an explicit ``1`` opts in; anything else is stock mode."""
    monkeypatch.delenv(ENV, raising=False)
    if value is not None:
        monkeypatch.setenv(ENV, value)
    assert plugin.is_enabled() is expected


def test_load_default_off_patches_nothing(monkeypatch):
    """No env: ``load()`` is False and the host class is left stock."""
    runner = _make_runner()
    _stub_candidates(monkeypatch, runner, plugin._CANDIDATES[0][0])
    assert plugin.load() is False
    assert getattr(runner.load_model, plugin._MARKER, False) is False
    assert plugin._INSTALLED == []


def test_load_default_off_imports_nothing():
    """Default-off must not even pull torch / vllm / vllm_ascend into memory."""
    code = (
        "import sys;"
        "import vllm_ascend_split_batch.gdn_merge_inproj_plugin as p;"
        "assert p.load() is False;"
        "bad=[m for m in sys.modules"
        " if m.split('.')[0] in ('torch','torch_npu','vllm','vllm_ascend')];"
        "assert not bad, bad"
    )
    result = subprocess.run(
        [sys.executable, "-c", code],
        cwd="/tmp",
        capture_output=True,
        text=True,
    )
    assert result.returncode == 0, result.stderr


# ------------------------------------------------------- the forward-consumer seam
def test_the_carrier_is_fail_closed_and_documented():
    """The merge is refused until a safe consumer seam exists (one constant)."""
    assert plugin.forward_consumer_available() is False
    assert isinstance(plugin.FORWARD_CONSUMER_BLOCKER, str)
    assert "in_proj_qkvz" in plugin.FORWARD_CONSUMER_BLOCKER
    assert "AscendGatedDeltaNetAttention.forward" in plugin.FORWARD_CONSUMER_BLOCKER


def test_process_model_refuses_to_rewrite_weights_without_a_consumer(
    monkeypatch, caplog
):
    """Fail-closed: env on, no seam -> no rewrite, one warning, stock weight."""
    cls = _stub_host_gdn_class(monkeypatch)
    layer = _gdn_layer(cls)
    before = layer.in_proj_qkvz.weight
    # The serve logger is vLLM's (propagate=False); route it to the module logger
    # so caplog sees the record.  The transport itself is not what is under test.
    monkeypatch.setattr(plugin, "_SERVE_LOGGER", plugin.logger)

    with caplog.at_level(logging.WARNING):
        assert plugin._process_model(_model(gdn=layer)) == (0, 0)

    assert layer.in_proj_qkvz.weight is before
    assert layer.in_proj_qkvz.weight.shape == (plugin.QKVZ_OUT_ROWS, 2048)
    assert getattr(layer, plugin._LAYER_MARK, False) is False
    warnings = [r.getMessage() for r in caplog.records if r.levelno == logging.WARNING]
    assert any("fail-closed" in message for message in warnings), warnings

    # ... and the warning is logged once, not per load.
    caplog.clear()
    with caplog.at_level(logging.WARNING):
        plugin._process_model(_model(gdn=layer))
    assert not [r for r in caplog.records if r.levelno == logging.WARNING]


def test_process_model_merges_once_a_consumer_seam_exists(
    monkeypatch, consumer_available
):
    """The merge path is live: it is what the fail-closed gate guards."""
    cls = _stub_host_gdn_class(monkeypatch)
    layer = _gdn_layer(cls)
    layer.in_proj_qkvz.weight.data.fill_(1.0)
    layer.in_proj_ba.weight.data.fill_(2.0)

    assert plugin._process_model(_model(gdn=layer)) == (1, 0)

    weight = layer.in_proj_qkvz.weight
    assert isinstance(weight, torch.nn.Parameter)
    assert weight.requires_grad is False
    assert tuple(weight.shape) == (plugin.MERGED_OUT_ROWS, 2048)
    # Row order is the accepted one: qkvz first, the ba shard appended.
    assert bool((weight.data[: plugin.QKVZ_OUT_ROWS] == 1.0).all())
    assert bool((weight.data[plugin.QKVZ_OUT_ROWS :] == 2.0).all())
    assert getattr(layer, plugin._LAYER_MARK, False) is True
    # ``in_proj_ba`` is left in place: only the qkvz weight carries the merge.
    assert tuple(layer.in_proj_ba.weight.shape) == (plugin.BA_OUT_ROWS, 2048)

    # Idempotent: a second pass refuses (already merged) instead of re-concatenating.
    assert plugin._process_model(_model(gdn=layer)) == (0, 1)
    assert tuple(layer.in_proj_qkvz.weight.shape) == (plugin.MERGED_OUT_ROWS, 2048)


def test_process_model_does_not_rewrite_a_same_named_foreign_module(
    monkeypatch, consumer_available
):
    """A non-host module with ``in_proj_qkvz``/``in_proj_ba`` must stay stock."""
    cls = _stub_host_gdn_class(monkeypatch)
    host_layer = _gdn_layer(cls)
    foreign = torch.nn.Module()
    foreign.in_proj_qkvz = _InProj(plugin.QKVZ_OUT_ROWS)
    foreign.in_proj_ba = _InProj(plugin.BA_OUT_ROWS)

    assert plugin._process_model(_model(host=host_layer, other=foreign)) == (1, 0)

    assert tuple(foreign.in_proj_qkvz.weight.shape) == (plugin.QKVZ_OUT_ROWS, 2048)
    assert getattr(foreign, plugin._LAYER_MARK, False) is False


def test_process_model_skips_a_layer_that_is_not_f2(monkeypatch, consumer_available):
    cls = _stub_host_gdn_class(monkeypatch)
    good = _gdn_layer(cls)
    interleaved = _gdn_layer(cls, gqa_interleaved_layout=True)

    assert plugin._process_model(_model(a=good, b=interleaved)) == (1, 1)
    assert getattr(interleaved, plugin._LAYER_MARK, False) is False


def test_process_model_warns_when_the_host_class_is_missing(monkeypatch, caplog):
    """No importable host class: nothing is rewritten, and it says so."""
    monkeypatch.setitem(sys.modules, plugin._HOST_MODULE, None)
    monkeypatch.setattr(plugin, "_SERVE_LOGGER", plugin.logger)
    with caplog.at_level(logging.WARNING):
        assert plugin._process_model(_model()) == (0, 0)

    warnings = [r.getMessage() for r in caplog.records if r.levelno == logging.WARNING]
    assert any("not importable" in message for message in warnings), warnings


# ------------------------------------------------------------- the conversion
def test_merge_concatenates_qkvz_first_then_ba():
    qkvz = torch.full((6, 4), 1.0, dtype=torch.bfloat16)
    ba = torch.full((2, 4), 2.0, dtype=torch.bfloat16)

    merged = plugin.merge_inproj_weight(qkvz, ba)

    assert tuple(merged.shape) == (8, 4)
    assert merged.dtype == torch.bfloat16
    assert bool((merged[:6] == 1.0).all())
    assert bool((merged[6:] == 2.0).all())


def test_merge_does_not_mutate_its_inputs():
    qkvz = torch.full((6, 4), 1.0, dtype=torch.bfloat16)
    ba = torch.full((2, 4), 2.0, dtype=torch.bfloat16)

    merged = plugin.merge_inproj_weight(qkvz, ba)

    assert tuple(qkvz.shape) == (6, 4)
    assert tuple(ba.shape) == (2, 4)
    assert merged.data_ptr() not in (qkvz.data_ptr(), ba.data_ptr())


def test_merge_guards_are_explicit():
    qkvz = torch.zeros(6, 4, dtype=torch.bfloat16)
    with pytest.raises(TypeError):
        plugin.merge_inproj_weight([1, 2], qkvz)
    with pytest.raises(ValueError):
        plugin.merge_inproj_weight(torch.zeros(6), qkvz)
    with pytest.raises(ValueError):  # hidden dims differ
        plugin.merge_inproj_weight(qkvz, torch.zeros(2, 5, dtype=torch.bfloat16))
    with pytest.raises(ValueError):  # dtypes differ
        plugin.merge_inproj_weight(qkvz, torch.zeros(2, 4, dtype=torch.float16))
    with pytest.raises(ValueError):  # devices differ
        plugin.merge_inproj_weight(
            qkvz, torch.zeros(2, 4, dtype=torch.bfloat16).to("meta")
        )


# ------------------------------------------------------------- plan_merge guards
def test_plan_merge_accepts_an_f2_layer(monkeypatch):
    cls = _stub_host_gdn_class(monkeypatch)
    assert plugin.plan_merge(_gdn_layer(cls)) is None


def _weightless_proj(value):
    """A projection-like module whose ``weight`` is an arbitrary value."""
    proj = torch.nn.Module()
    proj.weight = value
    return proj


@pytest.mark.parametrize(
    "overrides,removed,invalid",
    [
        ({}, ("in_proj_qkvz",), "in_proj_qkvz"),
        ({}, ("in_proj_ba",), "in_proj_ba"),
        ({"in_proj_qkvz": _weightless_proj(3)}, (), "tensor"),
        ({"in_proj_qkvz": _weightless_proj(torch.zeros(8))}, (), "2-d"),
        (
            {"in_proj_ba": _InProj(plugin.BA_OUT_ROWS, dtype=torch.float16)},
            (),
            "bfloat16",
        ),
        ({"in_proj_ba": _InProj(plugin.BA_OUT_ROWS, hidden=4096)}, (), "differ"),
        ({"in_proj_qkvz": _InProj(plugin.QKVZ_OUT_ROWS, device="meta")}, (), "meta"),
        ({"tp_size": 4}, (), "layer dims imply"),
        ({"num_v_heads": 64}, (), "layer dims imply"),
        ({}, (), "already merged"),
        ({"in_proj_qkv": torch.nn.Module()}, (), "LoRA"),
        ({"gqa_interleaved_layout": True}, (), "layout"),
        ({"key_dim": None}, (), "dims"),
    ],
)
def test_plan_merge_refuses_every_off_contract_shape(
    monkeypatch, overrides, removed, invalid
):
    """Each refusal is explicit -- nothing is ever silently rewritten."""
    cls = _stub_host_gdn_class(monkeypatch)
    layer = _gdn_layer(cls)
    if invalid == "already merged":
        setattr(layer, plugin._LAYER_MARK, True)
    for name in removed:
        delattr(layer, name)
    for name, value in overrides.items():
        setattr(layer, name, value)

    reason = plugin.plan_merge(layer)

    assert reason is not None
    assert invalid.lower() in reason.lower(), reason


def test_plan_merge_refuses_a_foreign_module(monkeypatch):
    _stub_host_gdn_class(monkeypatch)
    foreign = torch.nn.Module()
    foreign.in_proj_qkvz = _InProj(plugin.QKVZ_OUT_ROWS)
    foreign.in_proj_ba = _InProj(plugin.BA_OUT_ROWS)

    assert plugin.plan_merge(foreign) == "not the host GDN attention class"


def test_plan_merge_refuses_wrong_row_counts(monkeypatch):
    """Rows must match both the F2 shape and the layer's own dims."""
    cls = _stub_host_gdn_class(monkeypatch)
    layer = _gdn_layer(cls, in_proj_qkvz=_InProj(plugin.QKVZ_OUT_ROWS + 32))

    reason = plugin.plan_merge(layer)

    assert reason is not None and "rows" in reason


def test_layer_dims_are_derived_from_the_f2_arithmetic(monkeypatch):
    cls = _stub_host_gdn_class(monkeypatch)
    assert plugin._layer_dims(_gdn_layer(cls)) == (
        plugin.QKVZ_OUT_ROWS,
        plugin.BA_OUT_ROWS,
    )
    assert plugin._layer_dims(_gdn_layer(cls, tp_size=0)) is None
    assert plugin._layer_dims(torch.nn.Module()) is None


# ------------------------------------------------------------- load_model seam
def test_install_on_class_wraps_and_marks():
    runner = _make_runner()
    original = runner.load_model

    assert plugin._install_on_class(runner) is True
    assert runner.load_model is not original
    assert getattr(runner.load_model, plugin._MARKER, False) is True
    assert runner.load_model.__name__ == original.__name__
    assert plugin._INSTALLED[0] == (runner, original)


def test_install_on_class_is_idempotent():
    runner = _make_runner()
    assert plugin._install_on_class(runner) is True
    wrapped = runner.load_model

    assert plugin._install_on_class(runner) is False
    assert runner.load_model is wrapped
    assert len(plugin._INSTALLED) == 1


def test_install_on_class_without_a_callable_seam_returns_false():
    class NoSeam:
        pass

    class BadSeam:
        load_model = None

    assert plugin._install_on_class(NoSeam) is False
    assert plugin._install_on_class(BadSeam) is False
    assert plugin._INSTALLED == []


def test_wrapper_forwards_arguments_and_result():
    runner = _make_runner()
    assert plugin._install_on_class(runner) is True
    instance = runner()

    assert instance.load_model("a", b=2) == "ORIGINAL"
    assert instance.args == ("a",)
    assert instance.kwargs == {"b": 2}


def test_wrapper_processes_after_a_successful_load(monkeypatch, consumer_available):
    """The merge runs *after* the loader materialized the model."""
    cls = _stub_host_gdn_class(monkeypatch)
    layer = _gdn_layer(cls)
    runner = _make_model_runner(layer)
    assert plugin._install_on_class(runner) is True
    instance = runner()

    assert instance.load_model() == "LOADED"

    loaded = instance.model.block.gdn
    assert tuple(loaded.in_proj_qkvz.weight.shape) == (plugin.MERGED_OUT_ROWS, 2048)
    assert getattr(loaded, plugin._LAYER_MARK, False) is True


def test_wrapper_skips_the_merge_when_the_load_fails(monkeypatch, consumer_available):
    """A raising loader must not be post-processed, and must stay raising."""
    cls = _stub_host_gdn_class(monkeypatch)
    layer = _gdn_layer(cls)
    runner = _make_failing_runner(layer)
    assert plugin._install_on_class(runner) is True

    instance = runner()
    with pytest.raises(RuntimeError, match="loader exploded"):
        instance.load_model()

    assert tuple(layer.in_proj_qkvz.weight.shape) == (plugin.QKVZ_OUT_ROWS, 2048)


def test_wrapper_skips_the_merge_when_no_model_was_loaded():
    runner = _make_runner()
    assert plugin._install_on_class(runner) is True

    # No ``model`` attribute at all: the wrapper must not raise.
    assert runner().load_model() == "ORIGINAL"


def test_wrapper_chains_with_another_plugins_wrapper(monkeypatch, consumer_available):
    """``gate_fp32_plugin`` wraps the same method; both hooks must survive."""
    cls = _stub_host_gdn_class(monkeypatch)
    layer = _gdn_layer(cls)
    runner = _make_model_runner(layer)
    calls = []

    original = runner.load_model

    def gate_fp32_wrapper(self, *args, **kwargs):
        calls.append("gate-fp32")
        return original(self, *args, **kwargs)

    gate_fp32_wrapper.gate_marker = True
    runner.load_model = gate_fp32_wrapper

    assert plugin._install_on_class(runner) is True
    assert runner().load_model() == "LOADED"

    assert calls == ["gate-fp32"]
    assert tuple(layer.in_proj_qkvz.weight.shape) == (plugin.MERGED_OUT_ROWS, 2048)

    # Uninstall puts gate-fp32's wrapper back, not the pristine method.
    assert plugin.uninstall() == 1
    assert runner.load_model is gate_fp32_wrapper


def test_install_wraps_the_first_available_candidate(monkeypatch):
    runner = _make_runner()
    _stub_candidates(monkeypatch, runner, plugin._CANDIDATES[0][0])

    assert plugin.install() is True
    assert getattr(runner.load_model, plugin._MARKER, False) is True


def test_install_returns_false_when_no_candidate_matches(monkeypatch):
    _stub_candidates(monkeypatch)  # every probe misses cleanly

    assert plugin.install() is False
    assert plugin._INSTALLED == []


def test_install_keeps_probing_after_a_missing_class(monkeypatch):
    runner = _make_runner()
    _stub_candidates(monkeypatch, runner, plugin._CANDIDATES[-1][0])

    assert plugin.install() is True
    assert getattr(runner.load_model, plugin._MARKER, False) is True


def test_install_does_not_double_wrap_on_a_second_call(monkeypatch):
    runner = _make_runner()
    _stub_candidates(monkeypatch, runner, plugin._CANDIDATES[0][0])

    assert plugin.install() is True
    wrapped = runner.load_model

    assert plugin.install() is False
    assert runner.load_model is wrapped
    assert getattr(runner.load_model, plugin._MARKER, False) is True


def test_install_fails_open_on_an_unexpected_error(monkeypatch, caplog):
    runner = _make_runner()
    _stub_candidates(monkeypatch, runner, plugin._CANDIDATES[0][0])

    def _boom(_cls):
        raise RuntimeError("injected class build failure")

    monkeypatch.setattr(plugin, "_install_on_class", _boom)
    with caplog.at_level(logging.ERROR):
        assert plugin.install() is False

    messages = [record.getMessage() for record in caplog.records]
    assert any("failed open" in message for message in messages), messages


def test_install_reentrancy_guard_returns_false(monkeypatch):
    runner = _make_runner()
    _stub_candidates(monkeypatch, runner, plugin._CANDIDATES[0][0])

    plugin._LOCAL.installing = True
    try:
        assert plugin.install() is False
    finally:
        plugin._LOCAL.installing = False

    assert getattr(runner.load_model, plugin._MARKER, False) is False


def test_load_enabled_installs_the_carrier(monkeypatch):
    monkeypatch.setenv(ENV, "1")
    runner = _make_runner()
    _stub_candidates(monkeypatch, runner, plugin._CANDIDATES[0][0])

    assert plugin.load() is True
    assert getattr(runner.load_model, plugin._MARKER, False) is True


def test_uninstall_restores_the_pristine_seam(monkeypatch):
    runner = _make_runner()
    original = runner.load_model
    _stub_candidates(monkeypatch, runner, plugin._CANDIDATES[0][0])

    assert plugin.install() is True
    assert plugin.uninstall() == 1
    assert runner.load_model is original
    assert plugin._INSTALLED == []

    # Recoverable and idempotent: nothing left to undo.
    assert plugin.uninstall() == 0
    assert plugin.install() is True  # a fresh install still works


def test_post_load_hook_fails_open_on_an_unexpected_error(monkeypatch, caplog):
    """A broken post-load hook must not break engine startup."""
    cls = _stub_host_gdn_class(monkeypatch)
    runner = _make_model_runner(_gdn_layer(cls))
    assert plugin._install_on_class(runner) is True
    # The serve logger is vLLM's (``propagate=False``) whenever vllm is
    # importable -- i.e. exactly on the host this carrier ships to, and *not* on
    # the dependency-free CI legs.  Route it to the module logger so the
    # assertion is about the fail-open contract and not about which of the two
    # transports happened to be installed (same as the fail-closed test above).
    monkeypatch.setattr(plugin, "_SERVE_LOGGER", plugin.logger)

    def _boom(_model):
        raise RuntimeError("injected post-load failure")

    monkeypatch.setattr(plugin, "_process_model", _boom)
    with caplog.at_level(logging.WARNING):
        assert runner().load_model() == "LOADED"

    messages = [record.getMessage() for record in caplog.records]
    assert any("failed open" in message for message in messages), messages


# ------------------------------------------------------------- descriptor / drift
def test_probe_order_matches_the_expected_host_candidates():
    """``install()``'s probe list is part of the host contract; pin it.

    The probe order lives in the module-level ``_CANDIDATES`` constant, so the
    test pins both the constant and the fact that ``install()`` consumes it (a
    future second, hand-rolled list would otherwise go unnoticed).
    """
    install_source = textwrap.dedent(inspect.getsource(plugin.install))
    assert "for module_name, class_name in _CANDIDATES:" in install_source
    assert plugin._CANDIDATES == _EXPECTED_CANDIDATES


def test_entry_point_is_declared_in_pyproject():
    """The general-plugin entry point is what the runtime loader consumes."""
    data = tomllib.loads((_REPO_ROOT / "pyproject.toml").read_text())
    general = data["project"]["entry-points"]["vllm.general_plugins"]
    assert general["gdn-merge-inproj"] == (
        "vllm_ascend_split_batch.gdn_merge_inproj_plugin:load"
    )
    # The neighbours must survive the addition.
    assert general["gate-fp32"] == "vllm_ascend_split_batch.gate_fp32_plugin:load"


def test_entry_point_target_is_callable():
    assert callable(plugin.load)


GDN_MERGE_MANIFEST = (
    Path(plugin.__file__).resolve().parent
    / "gdn_merge_inproj"
    / "vllm-hust-extension-v0.2.json"
)


def test_bundle_manifest_is_import_only_with_its_own_injection_key() -> None:
    """The bundle descriptor must not advertise the carrier as enablable.

    The carrier is default-off *and* fail-closed (no safe merged-consumption
    seam on the host), so a flipped ``active`` status would let the extension
    manager inject a switch whose only effect is a warning.  Pin the
    ``import_only`` status, the exact env key and the bounded host range.
    """
    raw = json.loads(GDN_MERGE_MANIFEST.read_text(encoding="utf-8"))
    assert raw["extension_id"] == "org.vllm-hust.gdn-merge-inproj"
    assert raw["activation"]["environment"] == {plugin.ENV_ENABLE: "1"}
    carriers = {item["module"]: item["status"] for item in raw["implementation"]}
    assert carriers["vllm_ascend_split_batch.gdn_merge_inproj_plugin"] == "import_only"
    assert raw["components"][0]["implementation_ref"] == (
        "vllm_ascend_split_batch.gdn_merge_inproj_plugin:load"
    )
    version_range = raw["host"]["version_range"]
    assert version_range == ">=0.25.1rc2.dev125,<0.25.2"
    assert not version_range.strip().startswith(">=") or "," in version_range


def test_descriptor_declares_both_entry_point_groups() -> None:
    """The two wiring lines (general plugins + bundle) are declared in pyproject.

    Entry-point discovery reads installed distribution metadata, so a *new* entry
    point only shows up after the editable install is refreshed; this test pins
    the declaration itself, which is what a refresh consumes.
    """
    data = tomllib.loads((_REPO_ROOT / "pyproject.toml").read_text())
    groups = data["project"]["entry-points"]
    assert (
        groups["vllm_hust.extension_bundles"]["org.vllm-hust.gdn-merge-inproj"]
        == "vllm_ascend_split_batch.gdn_merge_inproj"
    )
    assert (
        groups["vllm.general_plugins"]["gdn-merge-inproj"]
        == "vllm_ascend_split_batch.gdn_merge_inproj_plugin:load"
    )


def test_bundle_package_carries_only_the_manifest() -> None:
    """The bundle subpackage is a marker: no import side effects, no carrier copy.

    Importing it must not pull torch / vllm / vllm_ascend, and it must not
    re-export the carrier (a second copy of the merge logic is exactly what the
    migration out of the runtime tree removed).
    """
    code = (
        "import sys;"
        "import vllm_ascend_split_batch.gdn_merge_inproj as b;"
        "bad=[m for m in sys.modules"
        " if m.split('.')[0] in ('torch','torch_npu','vllm','vllm_ascend')];"
        "assert not bad, bad;"
        "assert not hasattr(b, 'load'), 'the bundle package must not re-export load'"
    )
    result = subprocess.run(
        [sys.executable, "-c", code],
        cwd="/tmp",
        capture_output=True,
        text=True,
    )
    assert result.returncode == 0, result.stderr


def test_host_seams_are_still_present():
    """Read-only drift guard: the probed host names must still resolve.

    This never wraps anything -- it only checks that the host GDN class and at
    least one runner ``load_model`` attribute are importable, so a host rename
    shows up as a red test instead of a silently inert carrier.
    """
    import importlib

    found = []
    for module_name, class_name in plugin._CANDIDATES:
        try:
            module = importlib.import_module(module_name)
        except Exception:  # noqa: BLE001 -- absence is the point of this guard
            continue
        cls = getattr(module, class_name, None)
        if cls is not None and callable(getattr(cls, "load_model", None)):
            found.append(f"{module_name}.{class_name}")

    if not found:
        pytest.skip("no GDN-merge runner candidate is importable here")

    try:
        host_module = importlib.import_module(plugin._HOST_MODULE)
    except Exception:  # noqa: BLE001
        pytest.skip("the host GDN module is not importable here")
    host_class = getattr(host_module, plugin._HOST_CLASS, None)
    if host_class is None:
        pytest.skip("the host GDN class is not importable here")

    assert found
    assert isinstance(host_class, type)
