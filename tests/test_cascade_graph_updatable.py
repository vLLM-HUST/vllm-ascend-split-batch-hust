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

"""UpdatableGraph replay seam (cascade graph params on host >= fbe4911bb).

Why this file exists: the host generation that replays FULL graphs through
``UpdatableGraph`` never calls ``update_full_graph_params`` for them (it
early-returns; ``vllm_ascend/compilation/acl_graph.py``), so the wrapper the
plugin installs on ``impl_cls.update_graph_params`` is not entered.  Measured
on a locally rebuilt target stack (2026-09-26): ``replay update`` fired **0**
times and the cascade twin replayed with capture-time (dummy-prefix)
parameters, which hung the first decode step (three requests, zero throughput,
engine spinning).  The plugin therefore hooks the wrapper's own
``_updatable_graph_replay`` (seam #2) and re-parameterizes the cascade task
groups right after the replay.

These tests are CPU-only: the hook is exercised through a fake wrapper class and
the host ``get_graph_params`` import happens lazily, inside the function under
test (monkeypatched here).  The two cascade gate flags are read with
``getattr(..., False)`` in the hook so the pinned host (whose ``envs`` module has
no such attributes before ``load()`` injects them) can be exercised too.
"""

import logging

import _device_stack  # noqa: F401  -- device stack present, or skip
import pytest

from vllm_ascend_split_batch import cascade_graph_plugin as gp


class _FakeWrapper:
    """Duck-typed stand-in for ``ACLGraphWrapper``."""

    def __init__(self, update_stream="upstream"):
        self.update_stream = update_stream
        self.replayed: list[str] = []

    def _updatable_graph_replay(self, forward_context, graph):
        self.replayed.append(getattr(graph, "name", "?"))
        return "host-replay-result"


class _FakeDescriptor:
    def __init__(self, num_tokens):
        self.num_tokens = num_tokens


class _FakeForwardContext:
    def __init__(self, num_tokens=64):
        self.batch_descriptor = _FakeDescriptor(num_tokens)
        self.attn_metadata = {"layer0": object()}


@pytest.fixture()
def hooked(monkeypatch):
    """Fresh fake wrapper class with the plugin hook installed."""
    monkeypatch.setattr(gp, "_cascade_updatable_patched", False, raising=False)
    wrapper_cls = type("FakeACLGraphWrapper", (_FakeWrapper,), {})
    assert gp._wrap_updatable_graph_replay(wrapper_cls) is True
    return wrapper_cls


def _set_env_flag(monkeypatch, key: str, value: bool) -> None:
    """Turn a plugin gate flag on through the module's **env_variables** table.

    Do NOT use ``monkeypatch.setattr(envs_mod, KEY, …)``: ``vllm_ascend.envs``
    resolves unknown attributes through ``__getattr__`` + ``env_variables``, and
    pytest's undo then writes the *resolved old value* into the module
    ``__dict__`` as a real attribute -- which sticks for the rest of the process
    and silently defeats lazy resolution for every later test (measured: 4
    downstream failures in ``tests/test_cascade_plugin.py``).  Patching the
    table entry is exactly the surface the plugin injects into.
    """
    from vllm_ascend import envs as envs_mod

    monkeypatch.setitem(envs_mod.env_variables, key, lambda: value)


@pytest.fixture()
def cascade_step(monkeypatch):
    """Cascade graph gate on + 'this step selected the twin' flag set."""
    _set_env_flag(monkeypatch, "VLLM_ASCEND_ENABLE_CASCADE_DECODE", True)
    _set_env_flag(monkeypatch, "VLLM_ASCEND_ENABLE_CASCADE_GRAPH", True)
    monkeypatch.setattr(gp._replay_ctx, "cascade_selected", True)
    yield
    gp._replay_ctx.cascade_selected = False


def _install_fake_graph_params(monkeypatch, recorded):
    """Patch the **real** host ``get_graph_params`` with a recorder.

    Must import the real module: inserting a stub into ``sys.modules`` would
    survive monkeypatch undo and make every later test that imports
    ``vllm_ascend.compilation.acl_graph`` fail (measured: 18 downstream failures
    in ``tests/test_cascade_plugin.py`` when this helper stubbed the module).
    """
    acl_graph = pytest.importorskip(
        "vllm_ascend.compilation.acl_graph",
        reason="the updatable-graph seam lives in the vllm-ascend compilation stack",
    )

    def get_graph_params():
        recorded["graph_params_calls"] += 1
        return "graph-params"

    monkeypatch.setattr(acl_graph, "get_graph_params", get_graph_params, raising=False)


def test_seam_absent_is_a_noop(monkeypatch) -> None:
    """Hosts without the seam must be left alone (seam #1 stays in charge)."""

    class _NoSeamWrapper:
        """Same shape as the old host: no ``_updatable_graph_replay``."""

        def __init__(self):
            self.update_stream = "upstream"

    monkeypatch.setattr(gp, "_cascade_updatable_patched", False, raising=False)
    assert gp._wrap_updatable_graph_replay(_NoSeamWrapper) is False
    assert "_cascade_updatable_patched" not in _NoSeamWrapper.__dict__


def test_hook_is_idempotent(hooked, monkeypatch) -> None:
    monkeypatch.setattr(gp, "_cascade_updatable_patched", True, raising=False)
    assert gp._wrap_updatable_graph_replay(hooked) is False


def test_cascade_step_updates_the_captured_groups(
    hooked, cascade_step, monkeypatch
) -> None:
    """Cascade step: the host replays, then our re-parameterization runs."""
    recorded = {"graph_params_calls": 0, "update_args": None}
    _install_fake_graph_params(monkeypatch, recorded)

    def fake_update(update_stream, forward_context, graph_params, num_tokens):
        recorded["update_args"] = (
            update_stream,
            forward_context,
            graph_params,
            num_tokens,
        )

    monkeypatch.setattr(gp, "_update_cascade_graph_params", fake_update)

    wrapper = hooked()
    forward_context = _FakeForwardContext(num_tokens=64)
    result = wrapper._updatable_graph_replay(forward_context, object())

    assert result == "host-replay-result"
    assert wrapper.replayed == ["?"]
    assert recorded["update_args"] == (
        "upstream",  # the wrapper's update stream
        forward_context,
        "graph-params",
        64,  # descriptor num_tokens -> ("cascade", 64) key
    )


def test_non_cascade_step_does_not_touch_cascade_groups(
    hooked, cascade_step, monkeypatch
) -> None:
    recorded = {"graph_params_calls": 0, "update_args": None}
    _install_fake_graph_params(monkeypatch, recorded)
    monkeypatch.setattr(
        gp,
        "_update_cascade_graph_params",
        lambda *a, **k: recorded.update(update_args=a),
    )

    gp._replay_ctx.cascade_selected = False  # standard step (no twin swap)
    wrapper = hooked()
    wrapper._updatable_graph_replay(_FakeForwardContext(), object())

    assert recorded["update_args"] is None
    assert recorded["graph_params_calls"] == 0


def test_default_off_graph_gate_skips_the_update(
    hooked, cascade_step, monkeypatch
) -> None:
    """Graph gate off (or decode gate off) => no cascade update at all."""

    recorded = {"graph_params_calls": 0, "update_args": None}
    _install_fake_graph_params(monkeypatch, recorded)
    monkeypatch.setattr(
        gp,
        "_update_cascade_graph_params",
        lambda *a, **k: recorded.update(update_args=a),
    )
    _set_env_flag(monkeypatch, "VLLM_ASCEND_ENABLE_CASCADE_GRAPH", False)
    _set_env_flag(monkeypatch, "VLLM_ASCEND_ENABLE_CASCADE_DECODE", True)

    hooked()._updatable_graph_replay(_FakeForwardContext(), object())
    assert recorded["update_args"] is None
    assert recorded["graph_params_calls"] == 0


def test_update_failure_is_fail_open(hooked, cascade_step, monkeypatch, caplog) -> None:
    """Our update must never break the step; the host result is returned."""
    recorded = {"graph_params_calls": 0}
    _install_fake_graph_params(monkeypatch, recorded)

    def boom(*args, **kwargs):
        raise RuntimeError("synthetic update failure")

    monkeypatch.setattr(gp, "_update_cascade_graph_params", boom)

    with caplog.at_level(logging.WARNING):
        result = hooked()._updatable_graph_replay(_FakeForwardContext(), object())

    assert result == "host-replay-result"
    assert any(
        "cascade update on the updatable-graph replay path failed" in r.getMessage()
        for r in caplog.records
    )


def test_missing_update_stream_is_reported_not_raised(
    hooked, cascade_step, monkeypatch, caplog
) -> None:
    recorded = {"graph_params_calls": 0}
    _install_fake_graph_params(monkeypatch, recorded)
    monkeypatch.setattr(
        gp, "_update_cascade_graph_params", lambda *a: pytest.fail("must not run")
    )

    wrapper = hooked(update_stream=None)
    with caplog.at_level(logging.WARNING):
        assert (
            gp._update_cascade_on_updatable_replay(wrapper, _FakeForwardContext())
            is False
        )
    assert any("update_stream" in r.getMessage() for r in caplog.records)
    assert recorded["graph_params_calls"] == 0


def test_cascade_update_runs_before_the_host_replay(
    hooked, cascade_step, monkeypatch
) -> None:
    """ORDER IS LOAD-BEARING: update (which records the in-graph events) first.

    The twin graph waits on ``ExternalEvent``s that only our update records.  If
    the hook updated *after* ``orig``, the host replay would block forever on
    events nobody records -- measured 2026-09-26/27 on a locally rebuilt target
    stack: a 4-concurrent request hung > 90 s (engine spinning) with the update
    placed after the replay, while the same code passed with it placed before.
    """
    order: list[str] = []
    _install_fake_graph_params(monkeypatch, {"graph_params_calls": 0})
    monkeypatch.setattr(
        gp, "_update_cascade_graph_params", lambda *a, **k: order.append("update")
    )

    class _Wrapper(_FakeWrapper):
        def _updatable_graph_replay(self, forward_context, graph):
            order.append("host-replay")
            return "host-replay-result"

    wrapper_cls = type("FakeWrapper4", (_Wrapper,), {})
    monkeypatch.setattr(gp, "_cascade_updatable_patched", False, raising=False)
    assert gp._wrap_updatable_graph_replay(wrapper_cls) is True

    assert wrapper_cls()._updatable_graph_replay(_FakeForwardContext(), object()) == (
        "host-replay-result"
    )
    assert order == ["update", "host-replay"], (
        f"the cascade update must be issued before the host replay (got {order})"
    )


def test_replay_context_flag_is_set_only_around_the_twin_call(monkeypatch) -> None:
    """``_replay_ctx.cascade_selected`` must be True *inside* the swapped call."""
    descriptor = _FakeDescriptor(num_tokens=64)
    forward_context = _FakeForwardContext(num_tokens=64)
    forward_context.batch_descriptor = descriptor

    class _Wrapper:
        def __init__(self):
            self.concrete_aclgraph_entries = {"standard": 1}
            # The twin exists for this descriptor, so the swap path is taken.
            self._cascade_aclgraph_entries = {descriptor: 2}
            self._cascade_variants_ready = True

    seen = {}

    def fake_orig_call(self, *args, **kwargs):
        seen["inside"] = gp._replay_ctx.cascade_selected
        seen["entries"] = dict(self.concrete_aclgraph_entries)
        return "ok"

    wrapper_cls = type("FakeWrapper2", (_Wrapper,), {})
    wrapper_cls.__call__ = fake_orig_call

    # Cascade replay: step flag on, capture window off, twin present.

    _set_env_flag(monkeypatch, "VLLM_ASCEND_ENABLE_CASCADE_DECODE", True)
    monkeypatch.setattr(gp, "_capture_ctx", gp._CaptureContext())
    import vllm_ascend_split_batch.cascade_runner_patch as rp

    # NOTE: the wrapper binds ``_step_is_cascade`` when it is installed, so the
    # patch has to land before ``_wrap_aclgraph_wrapper`` runs.
    monkeypatch.setattr(rp, "_step_is_cascade", lambda: True)
    gp._wrap_aclgraph_wrapper(wrapper_cls)

    monkeypatch.setattr(
        "vllm.forward_context.get_forward_context", lambda: forward_context
    )
    import vllm_ascend_split_batch.cascade_gate as gate

    monkeypatch.setattr(gate, "decision_for", lambda *a, **k: True)

    wrapper = wrapper_cls()
    out = wrapper.__call__(object())

    assert out == "ok"
    assert seen["inside"] is True, "flag must be live inside the swapped call"
    assert seen["entries"] == {descriptor: 2}, "the cascade table must be active"
    assert gp._replay_ctx.cascade_selected is False  # restored after the call
    assert wrapper.concrete_aclgraph_entries == {"standard": 1}  # table restored


def test_capture_window_does_not_set_the_replay_flag(monkeypatch) -> None:
    """During twin *capture* the table is swapped too, but no replay happens."""
    descriptor = _FakeDescriptor(num_tokens=32)
    forward_context = _FakeForwardContext(num_tokens=32)
    forward_context.batch_descriptor = descriptor

    class _Wrapper:
        def __init__(self):
            self.concrete_aclgraph_entries = {"standard": 1}
            self._cascade_aclgraph_entries = {}
            self._cascade_variants_ready = True

    seen = {}

    def fake_orig_call(self, *args, **kwargs):
        seen["inside"] = gp._replay_ctx.cascade_selected
        return "ok"

    wrapper_cls = type("FakeWrapper3", (_Wrapper,), {})
    wrapper_cls.__call__ = fake_orig_call

    _set_env_flag(monkeypatch, "VLLM_ASCEND_ENABLE_CASCADE_DECODE", True)
    ctx = gp._CaptureContext()
    ctx.active = True  # capture window
    monkeypatch.setattr(gp, "_capture_ctx", ctx)
    import vllm_ascend_split_batch.cascade_runner_patch as rp

    monkeypatch.setattr(rp, "_step_is_cascade", lambda: False)
    gp._wrap_aclgraph_wrapper(wrapper_cls)

    assert wrapper_cls().__call__(object()) == "ok"
    assert seen["inside"] is False, "capture must not claim a cascade replay"
