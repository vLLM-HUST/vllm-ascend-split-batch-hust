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

"""CPU tests for the RoPE variant defect carrier (``rope_fix_plugin``).

Everything here is mock-based: a fake ``vllm`` / ``vllm_ascend`` module tree is
put into ``sys.modules`` so the override mechanics (registry entries, wrapper
ordering, idempotency, fail-open warnings) can be exercised without torch,
torch_npu or a device.  The reverse drift guard (does the *host* still carry the
defects, do the seams still match) lives in ``test_rope_fix_drift.py``.

Covered contract:

* default-off: ``load()`` imports nothing, patches nothing and never logs;
* the wrapper runs the host registration FIRST and only then repoints
  ``op_registry_oot`` (a pre-planted key would trip the host's duplicate-name
  assert, so the ordering is not cosmetic);
* the three keys are repointed at the plugin subclasses, idempotently;
* a foreign registry entry (310P-style) or a missing seam only warns ONCE and
  leaves the host behaviour in place (fail-open).
"""

from __future__ import annotations

import inspect
import json
import logging
import subprocess
import sys
import types
from pathlib import Path

import pytest
from _device_stack import vllm_ascend  # noqa: F401  -- host stack, or skip

from vllm_ascend_split_batch import rope_fix_plugin as plugin

ENV = plugin.ENV_ENABLE


# --------------------------------------------------------------------- stubs


class _FakeTensor:
    """Minimal tensor stand-in for the mirrored ``forward_triton`` body."""

    def __init__(self, shape=(4, 8), chunks=None):
        self.shape = shape
        self.ndim = len(shape)
        self._chunks = chunks
        self.reshaped = None

    def __getitem__(self, _key):
        return _FakeCosSin(self)

    def chunk(self, _n, dim=-1):
        if self._chunks is not None:
            return self._chunks
        return (_FakeTensor(), _FakeTensor())

    def contiguous(self):
        return self

    def reshape(self, shape):
        self.reshaped = shape
        return self


class _FakeCosSin:
    """What ``cos_sin_cache[positions]`` returns."""

    def __init__(self, parent):
        self._parent = parent

    def chunk(self, n, dim=-1):
        return self._parent.chunk(n, dim)


def _fake_module(name, **attrs):
    module = types.ModuleType(name)
    for key, value in attrs.items():
        setattr(module, key, value)
    return module


class _HostTree:
    """The fake host surface the carrier leans on."""

    def __init__(
        self,
        monkeypatch,
        register: bool = True,
        variants: dict | None = None,
        has_triton: bool = True,
        base_rope_310p: bool = False,
    ):
        variants = variants or {}
        self.registry = {}
        self.registered = False
        self.registration_calls = []
        self.registry_during_registration = None
        self.recorded_cache = []
        self.triton_calls = []
        self.delegate_calls = []

        tree = self

        class AscendRotaryEmbedding:
            def forward_oot(self, *args, **kwargs):
                tree.delegate_calls.append(("910", args, kwargs))
                return ("ascend-forward-oot", args, kwargs)

        class AscendRotaryEmbedding310(AscendRotaryEmbedding):
            """Compatibility-build variant: the host swaps the base key for this."""

            def forward_oot(self, *args, **kwargs):
                tree.delegate_calls.append(("310p", args, kwargs))
                return ("ascend-forward-oot-310p", args, kwargs)

        base_rope_cls = (
            AscendRotaryEmbedding310 if base_rope_310p else AscendRotaryEmbedding
        )
        self.base_rope_cls = base_rope_cls

        class AscendMRotaryEmbedding:
            _ASCEND_TRITON_GRID_LIMIT = 65535

            def _match_cos_sin_cache_dtype(self, _query):
                return None

            def forward_triton(self, *args, **kwargs):
                return "host-forward-triton"

        class AscendYaRNRotaryEmbedding:
            def __init__(self, *args, truncate=False, **kwargs):
                self.args = args
                self.truncate = truncate
                self.kwargs = kwargs

        class Llama3RotaryEmbedding:
            def __init__(
                self,
                head_size,
                rotary_dim,
                max_position_embeddings,
                base,
                is_neox_style,
                dtype,
                scaling_factor,
                low_freq_factor,
                high_freq_factor,
                orig_max_position,
            ):
                self.cos_sin_cache = "llama3-cache"
                self.head_size = head_size

        self.AscendRotaryEmbedding = AscendRotaryEmbedding
        self.AscendMRotaryEmbedding = AscendMRotaryEmbedding
        self.AscendYaRNRotaryEmbedding = AscendYaRNRotaryEmbedding
        self.Llama3RotaryEmbedding = Llama3RotaryEmbedding

        def register_ascend_customop(vllm_config=None):
            tree.registration_calls.append(vllm_config)
            if tree.registered:
                # The real function early-returns on
                # ``_ASCEND_CUSTOMOP_IS_REIGISTERED``; the wrapper still runs its
                # post-pass step because it wraps the whole call.
                return "already-registered"
            # Simulate the host pass: build the dict it is about to register
            # (``REGISTERED_ASCEND_OPS``, where the 310P compatibility branch
            # swaps entries) and then plant it, asserting each name is still
            # absent -- exactly like ``CustomOp.register_oot`` does.
            registered = {
                "RotaryEmbedding": base_rope_cls,
                "MRotaryEmbedding": AscendMRotaryEmbedding,
                "YaRNScalingRotaryEmbedding": AscendYaRNRotaryEmbedding,
            }
            registered.update(variants)
            tree.registry_during_registration = dict(tree.registry)
            for key, cls in registered.items():
                assert key not in tree.registry, f"Duplicate op name: {key}"
                tree.registry[key] = cls
            tree.registered = True
            return "registered"

        self.register_ascend_customop = register_ascend_customop

        def triton_mrope(*args):
            tree.triton_calls.append(args)
            return (_FakeTensor(), _FakeTensor())

        def record_cos_sin_cache(cache):
            tree.recorded_cache.append(cache)

        def get_current_vllm_config():
            return types.SimpleNamespace(speculative_config=None)

        fake_tree = {
            "vllm": {},
            "vllm.config": {"get_current_vllm_config": get_current_vllm_config},
            "vllm.model_executor": {},
            "vllm.model_executor.custom_op": {"op_registry_oot": self.registry},
            "vllm.model_executor.layers": {},
            "vllm.model_executor.layers.rotary_embedding": {
                "Llama3RotaryEmbedding": Llama3RotaryEmbedding
            },
            "vllm.model_executor.layers.rotary_embedding.mrope": {
                "triton_mrope": triton_mrope
            },
            "vllm_ascend": {},
            "vllm_ascend.utils": (
                {"register_ascend_customop": register_ascend_customop}
                if register
                else {}
            ),
            "vllm_ascend.ops": {},
            "vllm_ascend.ops.rotary_embedding": {
                "AscendRotaryEmbedding": AscendRotaryEmbedding,
                "AscendMRotaryEmbedding": AscendMRotaryEmbedding,
                "AscendYaRNRotaryEmbedding": AscendYaRNRotaryEmbedding,
                "_record_cos_sin_cache": record_cos_sin_cache,
                # The fork module binds the kernel at module level under
                # HAS_TRITON; the carrier mirrors defect ② from *this* binding.
                "HAS_TRITON": has_triton,
            },
        }
        if has_triton:
            fake_tree["vllm_ascend.ops.rotary_embedding"]["triton_mrope"] = triton_mrope
        self.modules = {}
        for name, attrs in fake_tree.items():
            module = _fake_module(name, **attrs)
            self.modules[name] = module
            monkeypatch.setitem(sys.modules, name, module)
        # Wire parent -> child attributes so both ``import a.b as x`` and
        # ``from a.b import y`` resolve against the fakes.
        for name, module in self.modules.items():
            if "." not in name:
                continue
            parent, _, leaf = name.rpartition(".")
            setattr(self.modules[parent], leaf, module)

    # -- helpers ----------------------------------------------------------
    def call_registration(self):
        module = self.modules["vllm_ascend.utils"]
        return module.register_ascend_customop("cfg")

    def wrapper(self):
        return self.modules["vllm_ascend.utils"].register_ascend_customop


@pytest.fixture(autouse=True)
def _isolated_plugin_state(monkeypatch):
    monkeypatch.delenv(ENV, raising=False)
    plugin._reset_for_tests()
    yield
    plugin.uninstall()
    plugin._OVERRIDE_CLASSES.clear()
    plugin._ORIG.clear()
    plugin._reset_for_tests()
    monkeypatch.setattr(plugin, "triton_mrope", None)


# ===================================================================== CPU ==
def test_load_default_off_patches_nothing_and_imports_nothing() -> None:
    """Default-off: no env, no import of the host stack, no patch, no log."""
    code = (
        "import sys;"
        "import vllm_ascend_split_batch.rope_fix_plugin as p;"
        "assert p.load() is False;"
        "assert p.stats()['applied'] == {} and p.stats()['wrapped'] == [];"
        "bad=[m for m in sys.modules"
        " if m.split('.')[0] in ('vllm','vllm_ascend','torch','torch_npu')"
        " or m in ('triton','triton_ascend')];"
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
    for value in ("0", "true", "yes", " 1", "1 ", "TRUE"):
        monkeypatch.setenv(ENV, value)
        assert plugin.is_enabled() is False, value
    monkeypatch.setenv(ENV, "1")
    assert plugin.is_enabled() is True


def test_wrapper_registers_first_and_overrides_afterwards(monkeypatch) -> None:
    """The overlap order: host registration first, overrides on the way out.

    A key planted before ``register_ascend_customop`` would hit the host's
    ``assert reg_name not in op_registry_oot``; a key planted after the whole
    registration pass is what ``CustomOp.__new__`` reads at instantiation.
    """
    tree = _HostTree(monkeypatch)
    monkeypatch.setenv(ENV, "1")
    assert plugin.load() is True

    # Installing is not registering, and it is not overriding either: nothing may
    # be written into op_registry_oot before the host's own pass has run (a
    # pre-planted key would trip ``CustomOp.register_oot``'s duplicate-name
    # assert -- and would collide the day upstream wires defect ① itself).
    assert tree.registry == {}

    assert tree.call_registration() == "registered"
    # During the pass the registry was still pristine.
    assert tree.registry_during_registration == {}
    assert plugin._OVERRIDE_CLASSES[plugin.KEY_MROPE] not in (
        tree.registry_during_registration.values()
    )
    assert plugin._OVERRIDE_CLASSES[plugin.KEY_YARN] not in (
        tree.registry_during_registration.values()
    )

    assert (
        tree.registry[plugin.KEY_LLAMA3] is plugin._OVERRIDE_CLASSES[plugin.KEY_LLAMA3]
    )
    assert tree.registry[plugin.KEY_MROPE] is plugin._OVERRIDE_CLASSES[plugin.KEY_MROPE]
    assert tree.registry[plugin.KEY_YARN] is plugin._OVERRIDE_CLASSES[plugin.KEY_YARN]
    applied = plugin.stats()["applied"]
    assert "created" in applied[plugin.KEY_LLAMA3]
    assert applied[plugin.KEY_MROPE] == "replaced AscendMRotaryEmbedding"
    assert applied[plugin.KEY_YARN] == "replaced AscendYaRNRotaryEmbedding"
    assert plugin.stats()["skipped"] == {}


def test_repeated_load_and_registration_stay_idempotent(monkeypatch, caplog) -> None:
    tree = _HostTree(monkeypatch)
    monkeypatch.setenv(ENV, "1")
    with caplog.at_level(logging.WARNING):
        assert plugin.load() is True
        assert plugin.load() is True
        tree.call_registration()
        first = dict(tree.registry)
        tree.call_registration()
        second = dict(tree.registry)
        plugin._reset_for_tests()  # once-guards alone must not re-apply
        tree.call_registration()

    assert first == second
    assert tree.registry[plugin.KEY_MROPE] is first[plugin.KEY_MROPE]
    # One wrapper warning, one ACTIVE warning and exactly one "overrides in
    # place" line: everything is applied in the single post-registration pass.
    messages = [record.getMessage() for record in caplog.records]
    assert sum("is wrapped" in m for m in messages) == 1
    assert sum("carrier is ACTIVE" in m for m in messages) == 1
    assert sum("rope overrides in place" in m for m in messages) == 1


def test_wrapper_is_installed_on_every_module_holding_a_direct_reference(
    monkeypatch,
) -> None:
    """``from vllm_ascend.utils import register_ascend_customop`` call sites.

    ``vllm_ascend/worker/worker.py`` binds the function at import time, so
    patching only the ``vllm_ascend.utils`` attribute would miss the live call
    site.  Every module that already holds the original is rebound.
    """
    tree = _HostTree(monkeypatch)
    binder = _fake_module(
        "vllm_ascend_fake_worker",
        register_ascend_customop=tree.register_ascend_customop,
    )
    monkeypatch.setitem(sys.modules, "vllm_ascend_fake_worker", binder)
    monkeypatch.setenv(ENV, "1")
    assert plugin.load() is True

    assert binder.register_ascend_customop is tree.wrapper()
    assert "vllm_ascend_fake_worker" in plugin.stats()["wrapped"]
    # The wrapped call still reaches the original exactly once.
    binder.register_ascend_customop("cfg")
    assert tree.registration_calls == ["cfg"]
    assert tree.registry[plugin.KEY_MROPE] is plugin._OVERRIDE_CLASSES[plugin.KEY_MROPE]


def test_foreign_registry_entry_is_left_alone_with_one_warning(
    monkeypatch, caplog
) -> None:
    """310P-style variants (or a future upstream rework) are never clobbered."""

    class ForeignMRotary:
        pass

    tree = _HostTree(monkeypatch, variants={plugin.KEY_MROPE: ForeignMRotary})
    monkeypatch.setenv(ENV, "1")
    assert plugin.load() is True
    with caplog.at_level(logging.WARNING):
        tree.call_registration()
        tree.call_registration()

    assert tree.registry[plugin.KEY_MROPE] is ForeignMRotary
    assert tree.registry[plugin.KEY_YARN] is plugin._OVERRIDE_CLASSES[plugin.KEY_YARN]
    warnings = [
        r.getMessage()
        for r in caplog.records
        if "was left on the host" in r.getMessage()
    ]
    assert len(warnings) == 1, warnings
    assert "ForeignMRotary" in warnings[0]
    assert plugin.stats()["skipped"][plugin.KEY_MROPE].startswith(
        "expected the fork class AscendMRotaryEmbedding"
    )


def test_missing_registration_seam_warns_once_and_returns_false(
    monkeypatch, caplog
) -> None:
    tree = _HostTree(monkeypatch, register=False)
    monkeypatch.setenv(ENV, "1")
    with caplog.at_level(logging.WARNING):
        assert plugin.load() is False
        assert plugin.load() is False
    assert tree.registry == {}
    messages = [r.getMessage() for r in caplog.records]
    assert sum("is missing" in m for m in messages) == 1, messages
    assert not any("carrier is ACTIVE" in m for m in messages)


def test_install_failure_fails_open_with_one_warning(monkeypatch, caplog) -> None:
    _HostTree(monkeypatch)
    monkeypatch.setenv(ENV, "1")

    def _boom():
        raise RuntimeError("injected class build failure")

    monkeypatch.setattr(plugin, "_make_override_classes", _boom)
    with caplog.at_level(logging.WARNING):
        assert plugin.load() is False
        assert plugin.load() is False

    messages = [r.getMessage() for r in caplog.records]
    assert sum("installation failed" in m for m in messages) == 1, messages
    assert plugin.stats()["applied"] == {}


def test_half_installation_is_rolled_back(monkeypatch, caplog) -> None:
    """A failure *after* the wrap must not leave a live downgrading wrapper.

    The wrapper is installed first (its whole point is to run after the host's
    registration pass), so an override failure there used to leave the wrapper in
    place while the log claimed the process "stays on the fork's stock rope
    classes" -- the next registration call would then downgrade silently.  The
    rollback makes the log line and ``stats()`` true.
    """

    tree = _HostTree(monkeypatch)
    monkeypatch.setenv(ENV, "1")

    def _boom(*_args, **_kwargs):
        raise RuntimeError("injected eager override failure")

    monkeypatch.setattr(plugin, "_apply_overrides", _boom)
    with caplog.at_level(logging.WARNING):
        assert plugin.load() is False

    # No half state: the host reference is restored, the state is empty.
    assert plugin.stats()["installed"] is False
    assert plugin.stats()["wrapped"] == []
    assert (
        tree.modules["vllm_ascend.utils"].register_ascend_customop
        is tree.register_ascend_customop
    )
    messages = [r.getMessage() for r in caplog.records]
    assert sum("installation failed" in m for m in messages) == 1, messages
    assert not any("carrier is ACTIVE" in m for m in messages)

    # A later registration call therefore keeps the fork classes untouched.
    monkeypatch.undo()
    tree.call_registration()
    assert tree.registry[plugin.KEY_MROPE] is tree.AscendMRotaryEmbedding
    assert plugin.KEY_LLAMA3 not in tree.registry


def test_load_after_registration_applies_eagerly(monkeypatch) -> None:
    """The real worker ordering: registration first, ``load()`` after it.

    ``NPUWorker.__init__`` calls ``register_ascend_customop`` before
    ``super().__init__()``, and vLLM loads general plugins after that, so the
    wrapper will never fire in that process.  ``install()`` must notice and apply
    the overrides immediately.
    """
    tree = _HostTree(monkeypatch)
    monkeypatch.setenv(ENV, "1")

    # 1) the host pass runs before the plugin is ever loaded
    assert tree.call_registration() == "registered"
    assert plugin.KEY_LLAMA3 not in tree.registry

    # 2) the plugin loads afterwards
    assert plugin.load() is True

    assert (
        tree.registry[plugin.KEY_LLAMA3] is plugin._OVERRIDE_CLASSES[plugin.KEY_LLAMA3]
    )
    assert tree.registry[plugin.KEY_MROPE] is plugin._OVERRIDE_CLASSES[plugin.KEY_MROPE]
    assert tree.registry[plugin.KEY_YARN] is plugin._OVERRIDE_CLASSES[plugin.KEY_YARN]
    assert plugin.stats()["installed"] is True
    assert plugin.stats()["applied"][plugin.KEY_LLAMA3].startswith("created")


def test_missing_triton_skips_the_mrope_override_only(monkeypatch, caplog) -> None:
    """Defect ② needs the fork's own ``triton_mrope`` binding.

    Without triton the fork's ``forward_triton`` is unreachable as well, and the
    mirrored body would call a ``None`` global at *model forward* time -- a hard
    runtime failure instead of a load-time fail-open.  So key ② is not installed
    while ① and ③ still are.
    """
    tree = _HostTree(monkeypatch, has_triton=False)
    monkeypatch.setenv(ENV, "1")
    with caplog.at_level(logging.WARNING):
        assert plugin.load() is True
        tree.call_registration()

    assert plugin.KEY_MROPE not in plugin._OVERRIDE_CLASSES
    assert tree.registry[plugin.KEY_MROPE] is tree.AscendMRotaryEmbedding
    assert tree.registry[plugin.KEY_YARN] is plugin._OVERRIDE_CLASSES[plugin.KEY_YARN]
    assert (
        tree.registry[plugin.KEY_LLAMA3] is plugin._OVERRIDE_CLASSES[plugin.KEY_LLAMA3]
    )
    assert "not built" in plugin.stats()["skipped"][plugin.KEY_MROPE]
    messages = [r.getMessage() for r in caplog.records]
    assert sum("triton_mrope is missing" in m for m in messages) == 1, messages


def test_override_application_error_never_escapes_the_wrapper(
    monkeypatch, caplog
) -> None:
    """A failing override pass must not break the host's registration call."""

    tree = _HostTree(monkeypatch)
    monkeypatch.setenv(ENV, "1")
    assert plugin.load() is True

    def _boom(*_args, **_kwargs):
        raise RuntimeError("injected override failure")

    monkeypatch.setattr(plugin, "_apply_overrides", _boom)
    with caplog.at_level(logging.WARNING):
        assert tree.call_registration() == "registered"

    # The host pass kept running and the registry kept the host classes.
    assert tree.registry[plugin.KEY_MROPE] is tree.AscendMRotaryEmbedding
    messages = [r.getMessage() for r in caplog.records]
    assert len([m for m in messages if "applying the rope overrides" in m]) == 1


def test_uninstall_restores_the_registry_and_the_wrapper(monkeypatch) -> None:
    tree = _HostTree(monkeypatch)
    monkeypatch.setenv(ENV, "1")
    assert plugin.load() is True
    tree.call_registration()

    plugin.uninstall()

    assert tree.registry[plugin.KEY_MROPE] is tree.AscendMRotaryEmbedding
    assert tree.registry[plugin.KEY_YARN] is tree.AscendYaRNRotaryEmbedding
    assert plugin.KEY_LLAMA3 not in tree.registry
    assert (
        tree.modules["vllm_ascend.utils"].register_ascend_customop
        is tree.register_ascend_customop
    )


# ------------------------------------------------- the three override classes
def _installed_tree(monkeypatch):
    tree = _HostTree(monkeypatch)
    monkeypatch.setenv(ENV, "1")
    assert plugin.load() is True
    return tree


def test_llama3_override_keeps_the_host_construction_signature(monkeypatch) -> None:
    tree = _installed_tree(monkeypatch)
    cls = plugin._OVERRIDE_CLASSES[plugin.KEY_LLAMA3]
    host_params = list(
        inspect.signature(tree.Llama3RotaryEmbedding.__init__).parameters
    )
    own_params = list(inspect.signature(cls.__init__).parameters)
    assert own_params == host_params
    assert len(own_params) == 11  # self + the host's 10 arguments
    assert issubclass(cls, tree.Llama3RotaryEmbedding)


def test_llama3_override_records_the_cache_and_delegates_forward(monkeypatch) -> None:
    tree = _installed_tree(monkeypatch)
    cls = plugin._OVERRIDE_CLASSES[plugin.KEY_LLAMA3]
    rope = cls(128, 128, 131072, 1e6, True, "bf16", 8.0, 1.0, 4.0, 8192)

    assert rope.use_mtp is False
    assert tree.recorded_cache == ["llama3-cache"]
    assert rope.forward_oot("pos", "q", "k")[0] == "ascend-forward-oot"


def test_yarn_override_defaults_truncate_true_and_forwards_verbatim(
    monkeypatch,
) -> None:
    tree = _installed_tree(monkeypatch)
    cls = plugin._OVERRIDE_CLASSES[plugin.KEY_YARN]

    default = cls(128, 128, 32768, 1e6, True, 4.0, "bf16")
    explicit = cls(128, 128, 32768, 1e6, True, 4.0, "bf16", truncate=False)

    assert default.truncate is True  # vLLM/HF default (defect ③)
    assert explicit.truncate is False  # escape hatch preserved
    assert default.args == (128, 128, 32768, 1e6, True, 4.0, "bf16")
    assert issubclass(cls, tree.AscendYaRNRotaryEmbedding)


def test_mrope_override_passes_the_missing_is_neox_style(monkeypatch) -> None:
    """The mirrored ``forward_triton`` must call the 9-argument kernel."""
    tree = _installed_tree(monkeypatch)
    cls = plugin._OVERRIDE_CLASSES[plugin.KEY_MROPE]
    rope = cls.__new__(cls)
    cos_stub, sin_stub = _FakeTensor(), _FakeTensor()
    rope.cos_sin_cache = _FakeTensor((16, 8), chunks=(cos_stub, sin_stub))
    rope.mrope_section = [16, 24, 24]
    rope.head_size = 128
    rope.rotary_dim = 128
    rope.mrope_interleaved = True
    rope.is_neox_style = False
    query, key = _FakeTensor((4, 512)), _FakeTensor((4, 128))

    out = rope.forward_triton(_FakeTensor((4, 512), chunks=("cos", "sin")), query, key)

    assert len(tree.triton_calls) == 1
    call = tree.triton_calls[0]
    assert len(call) == 9
    assert call[-1] is False  # self.is_neox_style, the argument the fork drops
    assert call[0] is query and call[1] is key
    assert call[4] == [16, 24, 24]
    assert rope.cos is cos_stub and rope.sin is sin_stub
    assert out[0].reshaped == (4, 512) and out[1].reshaped == (4, 128)


def test_llama3_delegate_follows_the_registered_base_rope_class(monkeypatch) -> None:
    """The delegate is resolved from the *live* base-rope entry.

    A compatibility build (310P) swaps ``op_registry_oot["RotaryEmbedding"]`` for
    a class with its own ``forward_oot``.  Delegating to a hard-coded 910 class
    would hand such a build an unimplemented/incorrect kernel, so the override
    must follow whatever the build registered.
    """
    tree = _installed_tree(monkeypatch)
    tree.call_registration()
    cls = plugin._OVERRIDE_CLASSES[plugin.KEY_LLAMA3]

    rope = cls(128, 128, 131072, 1e6, True, "bf16", 8.0, 1.0, 4.0, 8192)
    rope.forward_oot("pos", "q", "k")

    assert [call[0] for call in tree.delegate_calls] == ["910"]
    assert rope._rope_delegate is tree.base_rope_cls.forward_oot
    assert tree.base_rope_cls is tree.AscendRotaryEmbedding


def test_llama3_delegate_prefers_the_compatibility_class(monkeypatch) -> None:
    """310P ordering: register first (base key swapped), then load the carrier."""
    tree = _HostTree(monkeypatch, base_rope_310p=True)
    monkeypatch.setenv(ENV, "1")
    assert tree.call_registration() == "registered"
    assert plugin.load() is True

    cls = plugin._OVERRIDE_CLASSES[plugin.KEY_LLAMA3]
    rope = cls(128, 128, 131072, 1e6, True, "bf16", 8.0, 1.0, 4.0, 8192)
    rope.forward_oot("pos", "q", "k")

    assert [call[0] for call in tree.delegate_calls] == ["310p"]
    assert rope._rope_delegate is tree.base_rope_cls.forward_oot
    assert tree.base_rope_cls is not tree.AscendRotaryEmbedding


def test_llama3_delegate_falls_back_when_the_base_key_is_not_a_class(
    monkeypatch,
) -> None:
    """A non-class registry entry must not break instantiation (fail-open)."""
    tree = _HostTree(monkeypatch, variants={plugin.KEY_BASE_ROTARY: object()})
    monkeypatch.setenv(ENV, "1")
    assert plugin.load() is True
    tree.call_registration()

    cls = plugin._OVERRIDE_CLASSES[plugin.KEY_LLAMA3]
    rope = cls(128, 128, 131072, 1e6, True, "bf16", 8.0, 1.0, 4.0, 8192)
    rope.forward_oot("pos", "q", "k")
    assert [call[0] for call in tree.delegate_calls] == ["910"]  # module fallback


# --------------------------------------------------- live host integration
def test_official_registration_path_reaches_the_overrides(monkeypatch) -> None:
    """The wrapper must catch the *real* ``register_ascend_customop`` call.

    ``vllm_ascend/worker/worker.py`` binds the function at import time
    (``from vllm_ascend.utils import register_ascend_customop``), so this drives
    the live host module: fresh registry, fork guard reset, real rope classes.
    Skipped when the vllm-ascend stack is not importable.
    """
    import importlib
    from types import SimpleNamespace

    importlib.import_module("vllm_ascend.ops")
    host_utils = pytest.importorskip(
        "vllm_ascend.utils",
        reason="the live registration path needs the vllm-ascend stack",
    )
    custom_op = pytest.importorskip("vllm.model_executor.custom_op")
    monkeypatch.setenv(ENV, "1")

    saved_registry = dict(custom_op.op_registry_oot)
    saved_flag = host_utils._ASCEND_CUSTOMOP_IS_REIGISTERED
    try:
        assert plugin.load() is True
        monkeypatch.setattr(host_utils, "_ASCEND_CUSTOMOP_IS_REIGISTERED", False)
        host_utils.register_ascend_customop(
            SimpleNamespace(model_config=SimpleNamespace(is_deepseek_mla=False))
        )
        for key in plugin.OVERRIDE_KEYS:
            assert custom_op.op_registry_oot[key] is plugin._OVERRIDE_CLASSES[key], key
        assert plugin.stats()["skipped"] == {}
    finally:
        plugin.uninstall()
        custom_op.op_registry_oot.clear()
        custom_op.op_registry_oot.update(saved_registry)
        host_utils._ASCEND_CUSTOMOP_IS_REIGISTERED = saved_flag


# ------------------------------------------------------------------ manifest
ROPE_FIX_MANIFEST = (
    Path(plugin.__file__).resolve().parent
    / "rope_fix"
    / "vllm-hust-extension-v0.2.json"
)


def test_bundle_manifest_is_import_only_with_its_own_injection_key() -> None:
    raw = json.loads(ROPE_FIX_MANIFEST.read_text(encoding="utf-8"))
    assert raw["extension_id"] == "org.vllm-hust.rope-fix"
    assert raw["activation"]["environment"] == {"VLLM_HUST_ROPE_FIX": "1"}
    carriers = {item["module"]: item["status"] for item in raw["implementation"]}
    assert carriers["vllm_ascend_split_batch.rope_fix_plugin"] == "import_only"
    assert raw["components"][0]["implementation_ref"] == (
        "vllm_ascend_split_batch.rope_fix_plugin:load"
    )


def test_descriptor_declares_both_entry_point_groups() -> None:
    """The two wiring lines (general plugins + bundle) are declared in pyproject.

    Entry-point discovery reads installed distribution metadata, so a *new*
    entry point only shows up after the editable install is refreshed; this test
    pins the declaration itself, which is what a refresh consumes.
    """
    import tomllib

    repo_root = Path(plugin.__file__).resolve().parents[2]
    groups = tomllib.loads((repo_root / "pyproject.toml").read_text())["project"][
        "entry-points"
    ]
    assert (
        groups["vllm_hust.extension_bundles"]["org.vllm-hust.rope-fix"]
        == "vllm_ascend_split_batch.rope_fix"
    )
    assert (
        groups["vllm.general_plugins"]["rope-fix"]
        == "vllm_ascend_split_batch.rope_fix_plugin:load"
    )
