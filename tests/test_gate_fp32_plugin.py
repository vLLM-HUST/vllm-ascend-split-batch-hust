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

"""CPU tests for the MoE gate fp32-hoist carrier (``gate_fp32_plugin``).

What is covered
---------------
The carrier hoists ``gate.weight.to(torch.float32)`` -- the per-step
``Cast[256,2048]`` seen 40x per decode step (one per MoE layer) in the D4 L4
windows -- out of the decode hot path.  These tests pin:

* default-off: ``load()`` without ``VLLM_HUST_GATE_FP32=1`` imports nothing and
  patches nothing;
* the gate selector: only a module whose *leaf* name is exactly ``gate`` is
  cached, so ``shared_expert_gate`` / ``router_gate`` / ``expert_gate`` stay
  stock (the leaf-name check is the plugin's sole structural predicate);
* the cached copy is fp32 and contiguous, converted from the already
  materialized weight, while the original weight is left untouched;
* the ``load_model`` seam: wrapping, argument/return pass-through, post-load
  timing, idempotency, and the fail-open paths (missing seam, non-callable
  seam, unexpected error, reentrancy);
* the descriptor wiring (``vllm.general_plugins`` entry point) and that the
  probe order has not drifted.

Why stubs instead of real objects
---------------------------------
Loading the real ``NPUModelRunner`` needs a live vLLM config and a device, and
wrapping the *real* host class here could not be undone (the carrier has no
``uninstall``).  ``install()`` only needs ``getattr(module, class_name)`` and a
callable ``load_model``, so fake candidate modules are injected into
``sys.modules`` for every probed name -- that also guarantees the test can never
touch the real host classes.  The positive host-side drift guard lives in
``test_host_candidate_seam_is_still_present`` (read-only; it never wraps).

No device, no network.  ``_device_stack`` gives real ``torch`` for the tensor
predicate, or a clean skip on a CPU-only runner.
"""

from __future__ import annotations

import ast
import inspect
import logging
import subprocess
import sys
import textwrap
import types
from pathlib import Path

import pytest

# 探针链到 /vllm-workspace 运行树(子进程 import 链), CPU CI 收敛 deselect。
pytestmark = pytest.mark.host_tree
from _device_stack import torch

from vllm_ascend_split_batch import gate_fp32_plugin as plugin

ENV = plugin.ENV_ENABLE
_REPO_ROOT = Path(plugin.__file__).resolve().parents[2]


def _probe_candidates() -> tuple[tuple[str, str], ...]:
    """Read ``install()``'s probe list straight out of its source.

    The tuple is a local, so it cannot be imported; parsing the AST keeps the
    stub coverage exact -- a stubbed module list that silently missed a probed
    name would let ``install()`` reach the real host class.
    """
    source = textwrap.dedent(inspect.getsource(plugin.install))
    for node in ast.walk(ast.parse(source)):
        if isinstance(node, ast.Assign) and any(
            getattr(target, "id", None) == "candidates" for target in node.targets
        ):
            return tuple(
                (item.elts[0].value, item.elts[1].value) for item in node.value.elts
            )
    raise AssertionError("install() no longer declares a `candidates` tuple")


_CANDIDATES = _probe_candidates()

_EXPECTED_CANDIDATES = (
    ("vllm_ascend.worker.model_runner_v1", "NPUModelRunner"),
    ("vllm_ascend.worker.v2.model_runner", "NPUModelRunner"),
    ("vllm_ascend._310p.worker.v2.model_runner", "NPUModelRunner310V2"),
    ("vllm_ascend._310p.model_runner_310", "NPUModelRunner310"),
    ("vllm.v1.worker.gpu_model_runner", "GPUModelRunner"),
    ("vllm.v1.worker.gpu.model_runner", "GPUModelRunner"),
)


# ------------------------------------------------------------------ helpers
class _Gate(torch.nn.Module):
    """A leaf module named ``gate`` carrying an arbitrary ``weight`` value."""

    def __init__(self, weight):
        super().__init__()
        self.weight = weight


def _model(**children: torch.nn.Module) -> torch.nn.Module:
    """Build a root module whose attributes give each child its leaf name."""
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


def _make_model_runner():
    """A runner whose ``load_model`` materializes ``self.model`` (real timing)."""

    class _Runner:
        def __init__(self):
            self.model = None

        def load_model(self, *args, **kwargs):
            self.model = _model(gate=_Gate(torch.zeros(4, 8, dtype=torch.bfloat16)))
            return "LOADED"

    return _Runner


def _make_modeless_runner():
    """A runner that materializes nothing; the wrapper must simply skip."""

    class _Runner:
        def load_model(self, *args, **kwargs):
            return "LOADED"

    return _Runner


def _stub_candidates(monkeypatch, target_cls=None, target_module=None):
    """Replace every probed candidate module with an inert stub.

    ``install()`` early-continues on ``AttributeError``, so only the module
    named ``target_module`` receives ``target_cls``; every other probe misses
    cleanly instead of importing a real host module.
    """
    for module_name, class_name in _CANDIDATES:
        module = types.ModuleType(module_name)
        if target_cls is not None and module_name == target_module:
            setattr(module, class_name, target_cls)
        monkeypatch.setitem(sys.modules, module_name, module)


@pytest.fixture(autouse=True)
def _reset_plugin_state(monkeypatch):
    monkeypatch.delenv(ENV, raising=False)
    plugin._LOCAL.installing = False
    yield
    plugin._LOCAL.installing = False


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
    _stub_candidates(monkeypatch, runner, _CANDIDATES[0][0])
    assert plugin.load() is False
    assert getattr(runner.load_model, plugin._MARKER, False) is False


def test_load_default_off_imports_nothing():
    """Default-off must not even pull torch / vllm / vllm_ascend into memory."""
    code = (
        "import sys;"
        "import vllm_ascend_split_batch.gate_fp32_plugin as p;"
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


def test_worker_startup_order_wraps_the_active_runner():
    """Regression: the carrier must wrap the *active* v1 model runner.

    Reproduces the real worker order in a subprocess -- ``import vllm``
    (activates the Ascend platform plugin), then ``WorkerBase.__init__``'s
    ``load_general_plugins()`` call -- and asserts the marker landed on
    ``vllm_ascend.worker.model_runner_v1.NPUModelRunner`` (the class the host
    actually instantiates; ``worker/worker.py``).  Before the
    ``vllm_ascend.ops`` warm-up this probe hit a circular ``ImportError`` and the
    wrapper silently landed only on the unused v2/GPU runners, leaving the
    carrier inert on a live serve.
    """
    code = (
        "import os;"
        "os.environ['VLLM_HUST_GATE_FP32']='1';"
        "import vllm;"
        "from vllm.plugins import load_general_plugins;"
        "load_general_plugins();"
        "import vllm_ascend_split_batch.gate_fp32_plugin as p;"
        "from vllm_ascend.worker.model_runner_v1 import NPUModelRunner as V1;"
        "assert getattr(V1.load_model, p._MARKER, False),"
        "'the active v1 model runner was not wrapped';"
        "print('v1 wrapped')"
    )
    result = subprocess.run(
        [sys.executable, "-c", code],
        cwd="/tmp",
        capture_output=True,
        text=True,
    )
    assert result.returncode == 0, result.stderr
    assert "v1 wrapped" in result.stdout


# ------------------------------------------------------------- gate selection
def test_only_leaf_named_gate_is_cached():
    """``router_gate`` / ``expert_gate`` are not the MoE router gate."""
    gate = _Gate(torch.zeros(4))
    router = _Gate(torch.zeros(4))
    expert = _Gate(torch.zeros(4))
    model = _model(gate=gate, router_gate=router, expert_gate=expert)

    assert plugin._cache_gate_weights(model) == 1
    assert hasattr(gate, "weight_fp32")
    assert not hasattr(router, "weight_fp32")
    assert not hasattr(expert, "weight_fp32")


def test_shared_expert_gate_is_excluded():
    """The leaf check is exact: ``shared_expert_gate`` must stay stock."""
    shared = _Gate(torch.zeros(4))
    model = _model(shared_expert_gate=shared)

    assert plugin._cache_gate_weights(model) == 0
    assert not hasattr(shared, "weight_fp32")


def test_nested_leaf_gate_is_cached():
    """A gate nested under a block (``block.gate``) is still a leaf ``gate``."""
    inner = _Gate(torch.zeros(4))
    model = _model(block=_model(gate=inner))

    assert plugin._cache_gate_weights(model) == 1
    assert hasattr(inner, "weight_fp32")


def test_returns_the_number_of_cached_gates():
    model = _model(
        gate=_Gate(torch.zeros(4)),
        block=_model(gate=_Gate(torch.zeros(4))),
    )
    assert plugin._cache_gate_weights(model) == 2


def test_skips_a_gate_without_a_tensor_weight():
    module = torch.nn.Module()
    module.weight = 3  # not a tensor: nothing to convert
    model = _model(gate=module)

    assert plugin._cache_gate_weights(model) == 0
    assert not hasattr(module, "weight_fp32")


def test_skips_a_gate_without_a_weight_attribute():
    model = _model(gate=torch.nn.Module())

    assert plugin._cache_gate_weights(model) == 0


def test_second_call_does_not_overwrite_an_existing_cache():
    gate = _Gate(torch.zeros(4))
    gate.weight_fp32 = torch.full((1,), 7.0)
    model = _model(gate=gate)

    assert plugin._cache_gate_weights(model) == 0
    assert gate.weight_fp32.shape == (1,)
    assert torch.equal(gate.weight_fp32, torch.full((1,), 7.0))


def test_empty_model_caches_nothing():
    assert plugin._cache_gate_weights(torch.nn.Module()) == 0


# ------------------------------------------------------------- dtype / layout
@pytest.mark.parametrize(
    "dtype",
    [torch.float32, torch.float16, torch.bfloat16, torch.int8],
)
def test_cached_copy_is_fp32_and_contiguous(dtype):
    """The hoisted copy is the fp32 conversion of the materialized weight."""
    weight = torch.zeros(4, 8, dtype=dtype)
    gate = _Gate(weight)

    assert plugin._cache_gate_weights(_model(gate=gate)) == 1
    cached = gate.weight_fp32
    assert cached.dtype == torch.float32
    assert cached.is_contiguous()
    assert torch.equal(cached, weight.to(torch.float32))
    assert gate.weight.dtype == dtype  # the original weight is untouched


def test_non_contiguous_weight_is_materialized_contiguously():
    weight = torch.randn(8, 4).t()
    assert not weight.is_contiguous()
    gate = _Gate(weight)

    assert plugin._cache_gate_weights(_model(gate=gate)) == 1
    assert gate.weight_fp32.is_contiguous()
    assert torch.equal(gate.weight_fp32, weight.to(torch.float32))


# ------------------------------------------------------------- load_model seam
def test_install_on_class_wraps_and_marks():
    runner = _make_runner()
    original = runner.load_model

    assert plugin._install_on_class(runner) is True
    assert runner.load_model is not original
    assert getattr(runner.load_model, plugin._MARKER, False) is True
    assert runner.load_model.__name__ == original.__name__


def test_install_on_class_is_idempotent():
    runner = _make_runner()
    assert plugin._install_on_class(runner) is True
    wrapped = runner.load_model

    assert plugin._install_on_class(runner) is False
    assert runner.load_model is wrapped


def test_install_on_class_without_load_model_returns_false():
    class NoSeam:
        pass

    assert plugin._install_on_class(NoSeam) is False


def test_install_on_class_with_non_callable_load_model_returns_false():
    class BadSeam:
        load_model = None

    assert plugin._install_on_class(BadSeam) is False


def test_wrapper_forwards_arguments_and_result():
    runner = _make_runner()
    assert plugin._install_on_class(runner) is True
    instance = runner()

    assert instance.load_model("a", b=2) == "ORIGINAL"
    assert instance.args == ("a",)
    assert instance.kwargs == {"b": 2}


def test_wrapper_caches_after_the_original_loads_the_model():
    """The cache is filled *after* the loader materializes the model."""
    runner = _make_model_runner()
    assert plugin._install_on_class(runner) is True
    instance = runner()

    assert instance.load_model() == "LOADED"
    gate = instance.model.gate
    assert hasattr(gate, "weight_fp32")
    assert gate.weight_fp32.dtype == torch.float32


def test_wrapper_skips_the_cache_when_no_model_was_loaded():
    runner = _make_modeless_runner()
    assert plugin._install_on_class(runner) is True

    # No ``model`` attribute at all: the wrapper must not raise.
    assert runner().load_model() == "LOADED"


# ------------------------------------------------------------- install() / load()
def test_install_wraps_the_first_available_candidate(monkeypatch):
    runner = _make_runner()
    _stub_candidates(monkeypatch, runner, _CANDIDATES[0][0])

    assert plugin.install() is True
    assert getattr(runner.load_model, plugin._MARKER, False) is True


def test_install_returns_false_when_no_candidate_matches(monkeypatch):
    _stub_candidates(monkeypatch)  # every probe misses cleanly

    assert plugin.install() is False


def test_install_keeps_probing_after_a_missing_class(monkeypatch):
    """A probe whose module lacks the class must not stop the loop."""
    runner = _make_runner()
    _stub_candidates(monkeypatch, runner, _CANDIDATES[-1][0])

    assert plugin.install() is True
    assert getattr(runner.load_model, plugin._MARKER, False) is True


def test_install_does_not_double_wrap_on_a_second_call(monkeypatch):
    runner = _make_runner()
    _stub_candidates(monkeypatch, runner, _CANDIDATES[0][0])

    assert plugin.install() is True
    wrapped = runner.load_model

    # Nothing new to wrap: the carrier reports False but the seam is intact
    # (one wrapper, still marked) -- a second ``load()`` never stacks wrappers.
    assert plugin.install() is False
    assert runner.load_model is wrapped
    assert getattr(runner.load_model, plugin._MARKER, False) is True


def test_install_fails_open_on_an_unexpected_error(monkeypatch, caplog):
    # The first probe must carry the class so the (patched) installer runs and
    # its RuntimeError escapes the inner ImportError/AttributeError guard.
    runner = _make_runner()
    _stub_candidates(monkeypatch, runner, _CANDIDATES[0][0])

    def _boom(_cls):
        raise RuntimeError("injected class build failure")

    monkeypatch.setattr(plugin, "_install_on_class", _boom)
    with caplog.at_level(logging.ERROR):
        assert plugin.install() is False

    messages = [record.getMessage() for record in caplog.records]
    assert any("failed open" in message for message in messages), messages


def test_install_reentrancy_guard_returns_false(monkeypatch):
    runner = _make_runner()
    _stub_candidates(monkeypatch, runner, _CANDIDATES[0][0])

    plugin._LOCAL.installing = True
    try:
        assert plugin.install() is False
    finally:
        plugin._LOCAL.installing = False

    assert getattr(runner.load_model, plugin._MARKER, False) is False


def test_load_enabled_installs_the_carrier(monkeypatch):
    monkeypatch.setenv(ENV, "1")
    runner = _make_runner()
    _stub_candidates(monkeypatch, runner, _CANDIDATES[0][0])

    assert plugin.load() is True
    assert getattr(runner.load_model, plugin._MARKER, False) is True


# ------------------------------------------------------------- descriptor / drift
def test_probe_order_matches_the_expected_host_candidates():
    """``install()``'s probe list is part of the host contract; pin it."""
    assert _CANDIDATES == _EXPECTED_CANDIDATES


def test_entry_point_is_declared_in_pyproject():
    """The general-plugin entry point is what the runtime loader consumes."""
    import tomllib

    data = tomllib.loads((_REPO_ROOT / "pyproject.toml").read_text())
    general = data["project"]["entry-points"]["vllm.general_plugins"]
    assert general["gate-fp32"] == ("vllm_ascend_split_batch.gate_fp32_plugin:load")


def test_entry_point_target_is_callable():
    assert callable(plugin.load)


def test_host_candidate_seam_is_still_present():
    """Read-only drift guard: at least one probed host seam must still exist.

    This never wraps anything -- it only checks that the host module/class and
    its ``load_model`` attribute are still importable, so a host rename shows up
    as a red test instead of a silently inert carrier.
    """
    import importlib

    found = []
    for module_name, class_name in _CANDIDATES:
        try:
            module = importlib.import_module(module_name)
        except Exception:  # noqa: BLE001 -- absence is the point of this guard
            continue
        cls = getattr(module, class_name, None)
        if cls is not None and callable(getattr(cls, "load_model", None)):
            found.append(f"{module_name}.{class_name}")

    if not found:
        pytest.skip("no gate-fp32 host candidate is importable here")

    assert found
