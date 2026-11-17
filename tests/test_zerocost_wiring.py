# Copyright (c) 2026 Huawei Technologies Co., Ltd. All Rights Reserved.
# Licensed under the Apache License, Version 2.0 (the "License").
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

"""CPU tests for the zero-new-kernel wiring carrier (default-off, guarded).

Covers four things, none of which needs an NPU, ``torch_npu`` or a serving
engine:

* the default-off contract: with both env vars unset :func:`load` returns
  before importing ``torch``/``vllm_ascend`` and leaves every seam untouched;
* the pure identity predicate (the only numeric-relevant piece of ①);
* the AST rewrite itself, exercised against a synthetic host class that has the
  same statement shapes as ``attention_v1`` -- so the guard/injection behaviour
  is asserted *behaviourally* (the copy is really skipped / really kept) and not
  just by string matching;
* a drift guard on the real host source: the anchors ①/② lean on must still be
  there, worded so a host bump that moves them fails loudly.

It also covers the closure traversal that resolves the host body behind other
plugins' wrappers: stacked (one / two / three layers) wrappers must resolve, and
an ambiguous or broken chain must refuse (amended 2026-09-13 after the
cascade-graph double-wrapper blocker).
"""

from __future__ import annotations

import ast
import importlib.util
import json
import linecache
import os
import subprocess
import sys
import textwrap
import types

import pytest

from vllm_ascend_split_batch import zerocost_fi_out as fi_out
from vllm_ascend_split_batch import zerocost_skip_cos_sin as skip_cos_sin
from vllm_ascend_split_batch import zerocost_state as state
from vllm_ascend_split_batch import zerocost_wiring as wiring


@pytest.fixture
def torch():
    return pytest.importorskip("torch", reason="CPU tensor tests need torch")


HOST_METHODS = (
    ("AscendAttentionBackendImpl", "forward"),
    ("AscendAttentionBackendImpl", "forward_fused_infer_attention"),
)

SYNTHETIC_HOST = '''
class DeviceOperator:
    """Stand-in for vllm_ascend.device.device_op.DeviceOperator."""

    calls = []

    @classmethod
    def npu_fused_infer_attention_score(cls, query=None, scale=1.0, **kwargs):
        cls.calls.append(kwargs)
        dest = kwargs.get("_zc_fi_out")
        if dest is None:
            return query.new_zeros(query.shape[0], 2, 2), None
        dest.zero_()
        return dest, None


class FakeImpl:
    def forward(self, attn_output, output, num_tokens):
        output[:num_tokens] = attn_output[:num_tokens]
        return output

    def forward_fused_infer_attention(self, output, num_tokens, query):
        attn_output, _ = DeviceOperator.npu_fused_infer_attention_score(
            query=query, scale=1.0
        )
        output[:num_tokens] = attn_output[:num_tokens]
        return output
'''


AMBIGUOUS_HOST = """
class DeviceOperator:
    @classmethod
    def npu_fused_infer_attention_score(cls, **kwargs):
        return None, None


class Ambiguous:
    def forward(self, attn_output, output, num_tokens):
        output[:num_tokens] = attn_output[:num_tokens]
        output[:num_tokens] = attn_output[:num_tokens]

    def forward_fused_infer_attention(self, output, num_tokens):
        attn_output, _ = DeviceOperator.npu_fused_infer_attention_score()
        output[:num_tokens] = attn_output[:num_tokens]
        output[:num_tokens] = attn_output[:num_tokens]
"""


#: a *plugin* stand-in: wrappers defined in a different file than the host, so
#: their code objects never satisfy the "lives in the host file" criterion and
#: the closure walk has to descend through them to the host body underneath.
STACKED_WRAPPER_PLUGIN = """
def make_forward_wrapper(orig):
    def forward_fused_infer_attention(self, output, num_tokens, query):
        return orig(self, output, num_tokens, query)
    return forward_fused_infer_attention


def make_ambiguous_wrapper(orig, decoy):
    def forward_fused_infer_attention(self, output, num_tokens, query):
        decoy()
        return orig(self, output, num_tokens, query)
    return forward_fused_infer_attention
"""


#: host stand-in whose file holds *two* functions, so a wrapper that closes over
#: both of them is ambiguous (the file criterion alone cannot pick one).
AMBIGUOUS_CHAIN_HOST = """
def _decoy_host_helper():
    return None


class FakeImpl:
    def forward_fused_infer_attention(self, output, num_tokens, query):
        return output
"""


def _synthetic_module(name: str = "zc_synthetic_host", source: str = SYNTHETIC_HOST):
    """A host stand-in whose methods ``inspect.getsource`` can still read."""
    filename = f"<{name}.py>"
    linecache.cache[filename] = (
        len(source),
        None,
        source.splitlines(keepends=True),
        filename,
    )
    module = types.ModuleType(name)
    module.__dict__["__file__"] = filename
    module.__dict__[state.IDENTITY_HELPER] = fi_out._identity_helper
    exec(compile(source, filename, "exec"), module.__dict__)
    return module


@pytest.fixture(autouse=True)
def _clean_stats():
    state._reset_for_tests()
    yield
    state._reset_for_tests()


# --------------------------------------------------------------- default-off


def test_default_off_imports_nothing_and_returns_false():
    """The unset-env path must not touch torch/torch_npu/vllm_ascend at all."""
    script = textwrap.dedent(
        """
        import os, sys
        for key in (
            "VLLM_HUST_FI_PREFILL_OUT",
            "VLLM_HUST_SKIP_COS_SIN",
            "VLLM_HUST_ZC_STATS_FILE",
        ):
            os.environ.pop(key, None)
        from vllm_ascend_split_batch import zerocost_wiring as zc
        assert zc.load() is False
        assert zc._installed == set()
        assert zc._orig == {}
        assert zc.rewritten_sources() == {}
        leaked = [
            name
            for name in ("torch", "torch_npu", "vllm_ascend")
            if name in sys.modules
        ]
        assert leaked == [], f"default-off imported {leaked}"
        print("clean")
        """
    )
    env = dict(os.environ)
    for key in ("VLLM_HUST_FI_PREFILL_OUT", "VLLM_HUST_SKIP_COS_SIN"):
        env.pop(key, None)
    proc = subprocess.run(
        [sys.executable, "-c", script], env=env, capture_output=True, text=True
    )
    assert proc.returncode == 0, proc.stderr
    assert "clean" in proc.stdout


def test_enable_tokens_are_exact(monkeypatch):
    monkeypatch.setenv(state.ENV_FI_OUT, "1")
    monkeypatch.setenv(state.ENV_SKIP_COS_SIN, "1")
    assert state.is_fi_out_enabled() and state.is_skip_cos_sin_enabled()
    for value in ("0", "true", "yes", " 1", ""):
        monkeypatch.setenv(state.ENV_FI_OUT, value)
        assert not state.is_fi_out_enabled(), value


# ------------------------------------------------------- identity predicate


def test_identity_copy_predicate(torch):
    output = torch.zeros(4, 2, 3)
    other = torch.ones(4, 2, 3)
    ident = fi_out._identity_copy
    # identical object / identical region
    assert ident(output, output, 4)
    assert ident(output[:4], output, 4)
    assert ident(output[:2], output, 2)
    # different buffer, different region
    assert not ident(other, output, 4)
    assert not ident(other[:2], output, 2)
    # a shorter source must still take the copy
    assert not ident(other[:2], output, 4)
    # vacuous and degenerate inputs are "nothing to do"
    assert ident(other, output, 0)
    assert not ident(None, output, 4)
    assert not ident(other, output, None)


# ------------------------------------------------- rollback bookkeeping (①)
#
# ``install_fi_out`` needs the real host tree, so the adaptor patch -- the piece
# that carries a rollback record -- is driven directly.  The record must name the
# descriptor that was there *before* the ``setattr`` (recording it afterwards
# stores the wrapper as "the original"), and uninstall must write a class-owned
# descriptor back while deleting an override that only shadowed a base class.


def _owned_adaptor():
    """Adaptor whose FIA entry is its own class attribute."""

    class Adaptor:
        @classmethod
        def npu_fused_infer_attention_score(cls, query, key, value, attn_metadata):
            return ("host-owned", query, key, value, attn_metadata)

    return Adaptor


def _inherited_adaptor():
    """Adaptor that inherits the FIA entry from a base class."""

    class Base:
        @classmethod
        def npu_fused_infer_attention_score(cls, query, key, value, attn_metadata):
            return ("host-inherited", query, key, value, attn_metadata)

    class Adaptor(Base):
        pass

    return Adaptor


def test_adaptor_patch_records_and_restores_a_class_owned_descriptor():
    adaptor = _owned_adaptor()
    original = adaptor.__dict__[state.FIA_SYMBOL]
    fi_out._patch_adaptor(adaptor)

    assert adaptor.__dict__[state.FIA_SYMBOL] is not original
    # the rollback record must be the pre-patch descriptor, not the wrapper
    key = f"method:{adaptor.__name__}.{state.FIA_SYMBOL}"
    assert state._orig[key][2] is original
    # without the private kwarg the wrapper delegates to the host call
    assert adaptor.npu_fused_infer_attention_score("q", "k", "v", "m") == (
        "host-owned",
        "q",
        "k",
        "v",
        "m",
    )

    wiring._uninstall_locked()
    assert adaptor.__dict__[state.FIA_SYMBOL] is original
    assert adaptor.npu_fused_infer_attention_score("q", "k", "v", "m") == (
        "host-owned",
        "q",
        "k",
        "v",
        "m",
    )


def test_adaptor_patch_on_an_inherited_entry_is_deleted_on_rollback():
    adaptor = _inherited_adaptor()
    assert state.FIA_SYMBOL not in adaptor.__dict__

    fi_out._patch_adaptor(adaptor)
    assert state.FIA_SYMBOL in adaptor.__dict__
    # inherited attribute -> nothing to write back
    key = f"method:{adaptor.__name__}.{state.FIA_SYMBOL}"
    assert state._orig[key][2] is None

    wiring._uninstall_locked()
    assert state.FIA_SYMBOL not in adaptor.__dict__, "class override not removed"
    assert adaptor.npu_fused_infer_attention_score("q", "k", "v", "m") == (
        "host-inherited",
        "q",
        "k",
        "v",
        "m",
    )


def test_adaptor_patch_is_idempotent():
    adaptor = _owned_adaptor()
    original = adaptor.__dict__[state.FIA_SYMBOL]
    fi_out._patch_adaptor(adaptor)
    first = adaptor.__dict__[state.FIA_SYMBOL]
    fi_out._patch_adaptor(adaptor)

    assert adaptor.__dict__[state.FIA_SYMBOL] is first, "wrapped twice"
    key = f"method:{adaptor.__name__}.{state.FIA_SYMBOL}"
    assert state._orig[key][2] is original, "record overwritten by the wrapper"

    wiring._uninstall_locked()
    assert adaptor.__dict__[state.FIA_SYMBOL] is original


def test_adaptor_patch_refuses_a_missing_seam():
    class Adaptor:
        pass

    with pytest.raises(RuntimeError):
        fi_out._patch_adaptor(Adaptor)
    assert f"method:Adaptor.{state.FIA_SYMBOL}" not in state._orig


# ------------------------------------------------------- the AST rewrite


def test_inject_fi_out_rewrites_the_destination(torch):
    module = _synthetic_module()
    state._rewrite_method(
        module,
        module.FakeImpl,
        "forward_fused_infer_attention",
        fi_out._transform_fia_method,
    )
    impl = module.FakeImpl()
    output = torch.zeros(2, 2, 2)
    query = torch.randn(2, 2, 2)
    result = impl.forward_fused_infer_attention(output, 2, query)
    assert result is output
    assert module.DeviceOperator.calls, "FIA stand-in was never called"
    received = module.DeviceOperator.calls[-1]
    assert "_zc_fi_out" in received, "destination kwarg not injected"
    assert received["_zc_fi_out"].shape == (2, 2, 2)


def test_guard_skips_only_identity_copies(torch):
    module = _synthetic_module()
    state._rewrite_method(module, module.FakeImpl, "forward", fi_out._guard_tail_copy)
    impl = module.FakeImpl()
    output = torch.zeros(3, 2, 2)

    # attn_output IS output -> the statement cannot change a byte: skipped
    impl.forward(output, output, 3)
    assert state.stats()["selfcopy_identity"] == 1
    assert state.stats()["selfcopy_copied"] == 0

    # a distinct producer -> the copy still happens and really lands
    producer = torch.arange(12, dtype=torch.float32).reshape(3, 2, 2)
    target = torch.zeros(3, 2, 2)
    impl.forward(producer, target, 3)
    assert state.stats()["selfcopy_copied"] == 1
    assert torch.equal(target, producer)


def test_rewrite_refuses_ambiguous_anchors():
    module = _synthetic_module("zc_synthetic_ambiguous", AMBIGUOUS_HOST)
    with pytest.raises(RuntimeError):
        state._rewrite_method(
            module, module.Ambiguous, "forward", fi_out._guard_tail_copy
        )
    with pytest.raises(RuntimeError):
        state._rewrite_method(
            module,
            module.Ambiguous,
            "forward_fused_infer_attention",
            fi_out._transform_fia_method,
        )


# ------------------------------------------- closure traversal (any depth)


def test_host_func_traverses_stacked_plugin_wrappers():
    """The host body resolves through one *and* two plugin wrapper layers.

    This is the regression test for the blocker that kept the bundle
    ``import_only``: with ``VLLM_ASCEND_ENABLE_CASCADE_GRAPH=1`` the class
    attribute is two plugin wrappers deep, and the old one-hop resolver broke on
    the first (plugin-file) cell instead of following it down to the host file.
    """
    host = _synthetic_module("zc_host_stacked")
    plugin = _synthetic_module("zc_plugin_stacked", STACKED_WRAPPER_PLUGIN)
    host_fn = host.FakeImpl.forward_fused_infer_attention
    once = plugin.make_forward_wrapper(host_fn)
    twice = plugin.make_forward_wrapper(once)
    assert once.__code__.co_filename != host.__file__
    assert twice.__code__.co_filename != host.__file__
    assert state._host_func(once, host.__file__) is host_fn
    assert state._host_func(twice, host.__file__) is host_fn


def test_host_func_traverses_three_wrapper_layers():
    host = _synthetic_module("zc_host_stacked3")
    plugin = _synthetic_module("zc_plugin_stacked3", STACKED_WRAPPER_PLUGIN)
    host_fn = host.FakeImpl.forward_fused_infer_attention
    func = host_fn
    for _ in range(3):
        func = plugin.make_forward_wrapper(func)
    assert state._host_func(func, host.__file__) is host_fn


def test_rewrite_through_stacked_wrappers_engages_fi_out(torch):
    """End-to-end: the double-wrapped class attribute still gets ① installed.

    ``_zc_fi_out`` must reach the adaptor and the (now-identity) tail copy must
    be skipped, exactly as in the single-wrapper leg -- proving the in-place
    ``host_func.__code__`` replacement is seen through both wrappers.
    """
    host = _synthetic_module("zc_host_stacked_rw")
    plugin = _synthetic_module("zc_plugin_stacked_rw", STACKED_WRAPPER_PLUGIN)
    host_fn = host.FakeImpl.forward_fused_infer_attention
    host.FakeImpl.forward_fused_infer_attention = plugin.make_forward_wrapper(
        plugin.make_forward_wrapper(host_fn)
    )
    state._rewrite_method(
        host,
        host.FakeImpl,
        "forward_fused_infer_attention",
        fi_out._transform_fia_method,
    )
    impl = host.FakeImpl()
    output = torch.zeros(2, 2, 2)
    query = torch.randn(2, 2, 2)
    result = impl.forward_fused_infer_attention(output, 2, query)
    assert result is output
    assert host.DeviceOperator.calls, "FIA stand-in was never called"
    assert "_zc_fi_out" in host.DeviceOperator.calls[-1], "destination not injected"
    # the .out stand-in wrote into the caller's region -> the copy is identity
    assert state.stats()["selfcopy_identity"] == 1
    assert state.stats()["selfcopy_copied"] == 0


def test_host_func_refuses_an_ambiguous_wrapper_chain():
    """Two host-file functions behind one wrapper -> refuse (fail-open)."""
    host = _synthetic_module("zc_host_ambiguous", AMBIGUOUS_CHAIN_HOST)
    plugin = _synthetic_module("zc_plugin_ambiguous", STACKED_WRAPPER_PLUGIN)
    func = plugin.make_ambiguous_wrapper(
        host.FakeImpl.forward_fused_infer_attention,
        host._decoy_host_helper,
    )
    with pytest.raises(RuntimeError):
        state._host_func(func, host.__file__)


def test_host_func_refuses_a_broken_chain():
    """A wrapper that delegates to nothing host-defined must raise, not guess."""
    host = _synthetic_module("zc_host_broken")
    plugin = _synthetic_module("zc_plugin_broken", STACKED_WRAPPER_PLUGIN)
    func = plugin.make_forward_wrapper(lambda *args, **kwargs: None)
    with pytest.raises(RuntimeError):
        state._host_func(func, host.__file__)


def test_rewrite_source_is_the_host_body_plus_the_guard():
    """The ported body must be the host source with exactly the guard added."""
    module = _synthetic_module("zc_synthetic_host_b")
    state._rewrite_method(module, module.FakeImpl, "forward", fi_out._guard_tail_copy)
    try:
        ported = state.rewritten_sources()["FakeImpl.forward"]
    finally:
        state._rewritten.clear()
    assert ported.count(state.IDENTITY_HELPER) == 1
    original_fn = ast.parse(textwrap.dedent(SYNTHETIC_HOST)).body[1].body[0]
    ported_fn = ast.parse(ported).body[0]
    tail = ported_fn.body[-2]
    assert isinstance(tail, ast.If)
    assert ast.unparse(tail.test) == (
        f"not {state.IDENTITY_HELPER}(attn_output, output, num_tokens)"
    )
    assert len(tail.body) == 1
    assert ast.unparse(tail.body[0]) == "output[:num_tokens] = attn_output[:num_tokens]"
    # every other statement is untouched (same source text, same order)
    assert [ast.unparse(node) for node in ported_fn.body[:-2]] == [
        ast.unparse(node) for node in original_fn.body[:-2]
    ]
    assert ast.unparse(ported_fn.body[-1]) == "return output"


# ------------------------------------------------------------ drift anchors


def _host_root() -> str:
    override = os.getenv("VLLM_ASCEND_HUST_ROOT")
    if override:
        package = os.path.join(override, "vllm_ascend")
        return package if os.path.isdir(package) else override
    spec = importlib.util.find_spec("vllm_ascend")
    if spec is None or not spec.origin:
        pytest.fail("host source not found; set VLLM_ASCEND_HUST_ROOT", pytrace=False)
    return os.path.dirname(spec.origin)


def _fn_source(path: str, class_name: str, fn_name: str) -> ast.FunctionDef:
    with open(path, encoding="utf-8") as handle:
        tree = ast.parse(handle.read())
    for node in ast.walk(tree):
        if isinstance(node, ast.ClassDef) and node.name == class_name:
            for child in node.body:
                if isinstance(child, ast.FunctionDef) and child.name == fn_name:
                    return child
    raise AssertionError(f"{class_name}.{fn_name} not found in {path}")


def _count_tail_copies(fn: ast.FunctionDef) -> int:
    return sum(
        1
        for stmt in fn.body
        if isinstance(stmt, ast.Assign)
        and len(stmt.targets) == 1
        and isinstance(stmt.targets[0], ast.Subscript)
        and isinstance(stmt.targets[0].value, ast.Name)
        and stmt.targets[0].value.id == "output"
        and isinstance(stmt.value, ast.Subscript)
        and isinstance(stmt.value.value, ast.Name)
        and stmt.value.value.id == "attn_output"
    )


@pytest.mark.host_tree
def test_attention_anchors_still_present():
    """host drift guard for ①: re-audit HOST_CONTRACT if this fails."""
    root = _host_root()
    path = os.path.join(root, "attention", "attention_v1.py")
    if not os.path.exists(path):
        pytest.fail(f"host source not present: {path}", pytrace=False)
    forward = _fn_source(path, state.ATTENTION_CLASS, state.METHOD_FORWARD)
    assert _count_tail_copies(forward) == 1, "anchor drifted → re-audit HOST_CONTRACT"
    fia = _fn_source(path, state.ATTENTION_CLASS, state.METHOD_FIA)
    assert _count_tail_copies(fia) == 1, "anchor drifted → re-audit HOST_CONTRACT"
    adaptor_calls = [
        node
        for node in ast.walk(fia)
        if isinstance(node, ast.Call)
        and isinstance(node.func, ast.Attribute)
        and node.func.attr == state.FIA_SYMBOL
        and isinstance(node.func.value, ast.Name)
        and node.func.value.id == state.ADAPTOR_ATTR
    ]
    assert len(adaptor_calls) == 1, "anchor drifted → re-audit HOST_CONTRACT"


@pytest.mark.host_tree
def test_rope_anchors_still_present():
    """host drift guard for ②: re-audit HOST_CONTRACT if this fails."""
    root = _host_root()
    path = os.path.join(root, "ops", "rotary_embedding.py")
    if not os.path.exists(path):
        pytest.fail(f"host source not present: {path}", pytrace=False)
    with open(path, encoding="utf-8") as handle:
        tree = ast.parse(handle.read())
    names = {node.name for node in tree.body if isinstance(node, ast.FunctionDef)}
    assert state.ROTARY_UPDATE in names
    assert state.ROTARY_READ in names
    body = next(
        node
        for node in tree.body
        if isinstance(node, ast.FunctionDef) and node.name == state.ROTARY_UPDATE
    )
    flat = ast.unparse(body)
    assert "_cos[:, :num_tokens] =" in flat, "anchor drifted → re-audit HOST_CONTRACT"
    assert "_cos_slice = _cos[:, :num_tokens]" in flat


@pytest.mark.host_tree
def test_rope_readers_are_the_known_pair():
    """② is only sound because exactly these two modules read the buffers."""
    root = _host_root()
    expected = {
        "vllm_ascend._310p.ops.rotary_embedding": os.path.join(
            root, "_310p", "ops", "rotary_embedding.py"
        ),
        "vllm_ascend.patch.worker.patch_minimax_m2": os.path.join(
            root, "patch", "worker", "patch_minimax_m2.py"
        ),
    }
    assert set(skip_cos_sin.ROPE_READER_MODULES) == set(expected)
    for module_name, path in expected.items():
        assert os.path.exists(path), f"{module_name} moved away"
        with open(path, encoding="utf-8") as handle:
            assert state.ROTARY_READ in handle.read()


# ------------------------------------------------- phase snapshots (2026-09-13)
#
# The counters are cumulative, so a phase split (startup vs serving) can only be
# obtained from a dump taken *between* the phases.  These tests pin the three
# properties the serving-side measurement relies on: the file name/shape of an
# on-demand snapshot, that the SIGUSR1 hook is installed exactly when the stats
# file is configured (and restored on uninstall), and that a ① call is
# attributed to its num_tokens.


def _signals_unavailable() -> str | None:
    """Reason the SIGUSR1 tests cannot run here (empty string = they can)."""
    import signal as _signal

    if not hasattr(_signal, "SIGUSR1"):
        return "platform has no SIGUSR1"
    try:
        _signal.getsignal(_signal.SIGUSR1)
    except ValueError:  # not the main thread
        return "tests are not running in the main thread"
    return None


def _read_json(path) -> dict:
    with open(path, encoding="utf-8") as handle:
        return json.load(handle)


def test_snapshot_file_name_shape_and_env_gate(torch, tmp_path, monkeypatch):
    base = str(tmp_path / "zc.json")
    monkeypatch.delenv(state.ENV_STATS_FILE, raising=False)
    assert state.snapshot() is None, "unset env must not write anything"
    assert list(tmp_path.iterdir()) == []

    monkeypatch.setenv(state.ENV_STATS_FILE, base)
    state._stats["fi_out_calls"] = 7
    state._tally_tokens(torch.zeros(2048, 4, 8))
    state._tally_tokens(torch.zeros(2048, 4, 8))
    state._tally_tokens(torch.zeros(128, 4, 8))

    first = state.snapshot()
    second = state.snapshot()
    assert first == f"{base}.snap1.json" and os.path.exists(first)
    assert second == f"{base}.snap2.json" and os.path.exists(second)
    assert not os.path.exists(f"{base}.tmp"), "atomic write left a temp file"

    payload = _read_json(second)
    assert set(payload) == {
        "pid",
        "utc",
        "stats",
        "fi_out_tokens",
        "fi_out_states",
        "installed",
    }
    assert payload["pid"] == os.getpid()
    assert payload["stats"]["fi_out_calls"] == 7
    assert payload["fi_out_tokens"] == {"128": 1, "2048": 2}
    assert payload["utc"].endswith("Z")


def test_state_histogram_tallies_by_state_and_survives_odd_input():
    class _State:
        name = "DecodeOnly"

    class _Metadata:
        attn_state = _State()

    state._tally_state(_Metadata())
    state._tally_state(_Metadata())
    state._tally_state(object())  # no attn_state at all
    assert state.state_histogram() == {"DecodeOnly": 2, "none": 1}


def test_exit_dump_carries_the_same_payload(torch, tmp_path, monkeypatch):
    """The exit dump and the snapshots must be comparable by subtraction."""
    base = str(tmp_path / "zc_exit.json")
    monkeypatch.setenv(state.ENV_STATS_FILE, base)
    state._tally_tokens(torch.zeros(64, 4, 8))
    state._dump_stats_on_exit()
    payload = _read_json(base)
    assert payload["fi_out_tokens"] == {"64": 1}
    assert "stats" in payload and "installed" in payload
    # pid-tagged copy: several processes share the env var, so the shared path
    # alone cannot say which process wrote it (measured collision 2026-09-13).
    tagged = _read_json(f"{base}.{os.getpid()}.json")
    assert tagged == payload
    assert f"zc_exit.json.{os.getpid()}.json" in os.listdir(tmp_path)


def test_token_histogram_ignores_non_tensors(torch):
    """A diagnostics hook must never break the host call it measures."""
    state._tally_tokens(object())  # no .shape
    state._tally_tokens(None)
    assert state.token_histogram() == {}
    state._tally_tokens(torch.zeros(3, 2))
    assert state.token_histogram() == {"3": 1}


def test_snapshot_hook_installed_only_with_env_and_restored(tmp_path, monkeypatch):
    import signal as _signal

    reason = _signals_unavailable()
    if reason:
        pytest.skip(reason)
    pristine = _signal.getsignal(_signal.SIGUSR1)

    # env unset -> the hook must not touch the process
    monkeypatch.delenv(state.ENV_STATS_FILE, raising=False)
    state._install_stats_snapshot()
    assert _signal.getsignal(_signal.SIGUSR1) is pristine
    assert state._prev_snapshot_handler is state._UNSET

    monkeypatch.setenv(state.ENV_STATS_FILE, str(tmp_path / "zc_sig.json"))
    state._install_stats_snapshot()
    assert _signal.getsignal(_signal.SIGUSR1) is state._snapshot_handler

    # the hook really writes the phase file
    state._stats["fi_out_calls"] = 3
    _signal.raise_signal(_signal.SIGUSR1)
    written = tmp_path / "zc_sig.json.snap1.json"
    assert written.exists(), "SIGUSR1 did not produce a snapshot"
    assert _read_json(written)["stats"]["fi_out_calls"] == 3

    state._restore_stats_snapshot()
    assert _signal.getsignal(_signal.SIGUSR1) is pristine
    # idempotent + no-op once restored
    state._restore_stats_snapshot()
    assert _signal.getsignal(_signal.SIGUSR1) is pristine
