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

"""Host-signature drift guards for the cascade graph wrappers.

The graph cascade plugin monkeypatches *private* host methods (unversioned
seams, see ``docs/pitfalls.md`` 2.1).  The 2026-09-09 baseline exposed four
signature drifts that the previous symbol-existence audit missed:

* **D1** ``_capture_cudagraphs`` gained ``profiler=None`` and the host call
  site passes it -> engine init died with ``unexpected keyword argument``;
* **D2** ``_update_full_graph_params_if_needed`` dropped ``positions`` and the
  plugin still passed it positionally;
* **D3** ``AscendAttentionBackendImpl.update_graph_params`` dropped
  ``num_dcp_pcp_tokens`` and the plugin forwarded seven positional arguments;
* **D4** ``_model_forward``'s upstream call convention is keyword-only (no
  ``num_tokens_padded``), while the plugin's wrapper required it.

These tests compare the plugin's wrappers against the host's *real* signatures
(``inspect.signature`` of the host callables) and the host's *real* call
conventions (keyword sets parsed from the host call sites with ``ast``), then
assert the documented fail-open contract: an internal failure logs ONE warning,
delegates to the original implementation and never raises out of a patched host
method.  A future host bump that changes one of these signatures fails here
loudly instead of silently breaking engine startup.
"""

import ast
import importlib
import inspect
import logging
import types
from pathlib import Path

import pytest

# Resolve the device_op <-> ops circular import exactly as the plugin's
# ``load()`` does, otherwise importing the worker module fails.
importlib.import_module("vllm_ascend.ops")

cascade_plugin = pytest.importorskip(
    "vllm_ascend_split_batch.cascade_plugin",
    reason="signature drift tests run on the NPU host with the full "
    "vllm-ascend stack installed",
)
runner_patch = pytest.importorskip(
    "vllm_ascend_split_batch.cascade_runner_patch",
    reason="signature drift tests run on the NPU host",
)
graph_plugin = pytest.importorskip(
    "vllm_ascend_split_batch.cascade_graph_plugin",
    reason="signature drift tests run on the NPU host",
)
cascade_gate = pytest.importorskip("vllm_ascend_split_batch.cascade_gate")
acl_graph = pytest.importorskip("vllm_ascend.compilation.acl_graph")
gpu_model_runner = pytest.importorskip("vllm.v1.worker.gpu_model_runner")
model_runner_v1 = pytest.importorskip("vllm_ascend.worker.model_runner_v1")
attention_v1 = pytest.importorskip("vllm_ascend.attention.attention_v1")
forward_context_mod = pytest.importorskip("vllm.forward_context")

GPUModelRunner = gpu_model_runner.GPUModelRunner
NPUModelRunner = model_runner_v1.NPUModelRunner
AscendAttentionBackendImpl = attention_v1.AscendAttentionBackendImpl

# Host signatures, read from the live host classes (not hardcoded).
_CAPTURE_SIG = inspect.signature(GPUModelRunner._capture_cudagraphs)
_MODEL_FORWARD_UPSTREAM_SIG = inspect.signature(GPUModelRunner._model_forward)
_MODEL_FORWARD_ASCEND_SIG = inspect.signature(NPUModelRunner._model_forward)
_UPDATE_FULL_SIG = inspect.signature(NPUModelRunner._update_full_graph_params_if_needed)
_UPDATE_GRAPH_SIG = inspect.signature(AscendAttentionBackendImpl.update_graph_params)
_DETERMINE_SIG = inspect.signature(
    GPUModelRunner._determine_batch_execution_and_padding
)

ENV_KEYS = (
    "VLLM_ASCEND_ENABLE_CASCADE_DECODE",
    "VLLM_ASCEND_ENABLE_CASCADE_GRAPH",
)


def _boom(*_args, **_kwargs):
    raise RuntimeError("injected cascade failure")


class _SigRecorder:
    """Callable that records calls and flags host-signature violations.

    ``signature`` is the *host* callable's signature.  A call that does not
    bind to it (extra positional, dropped/renamed keyword) is recorded in
    ``signature_errors`` and re-raised, mirroring what the real host method
    would do.  ``prepend_self`` is used when the recorder is installed as an
    instance attribute (so the host method's ``self`` is implicit).
    """

    def __init__(self, signature, result=None, prepend_self=False):
        self.signature = signature
        self.result = result
        self.prepend_self = prepend_self
        self.calls = []
        self.signature_errors = []

    def __call__(self, *args, **kwargs):
        bind_args = ((object(),) + args) if self.prepend_self else args
        try:
            self.signature.bind(*bind_args, **kwargs)
        except TypeError as exc:
            self.signature_errors.append(str(exc))
            raise
        self.calls.append((args, kwargs))
        return self.result


class _FakeForwardContext:
    def __init__(self, num_tokens=64, shared_len=4096):
        self.batch_descriptor = types.SimpleNamespace(num_tokens=num_tokens)
        self.attn_metadata = {
            "layer.0": types.SimpleNamespace(cascade_shared_len=shared_len)
        }


class _FakeRunner:
    def __init__(self, update_recorder=None):
        if update_recorder is not None:
            self._update_full_graph_params_if_needed = update_recorder


def _host_call_keywords(host_module_file, func_name):
    """Keyword names passed at every call site of ``func_name`` in the host.

    Source/AST level on purpose: the assertion must fail when the host bumps a
    call convention, not only when the runtime import happens to break.
    """
    tree = ast.parse(Path(host_module_file).read_text(encoding="utf-8"))
    out = []
    for node in ast.walk(tree):
        if not isinstance(node, ast.Call):
            continue
        func = node.func
        name = (
            func.attr
            if isinstance(func, ast.Attribute)
            else (func.id if isinstance(func, ast.Name) else None)
        )
        if name == func_name:
            out.append([kw.arg for kw in node.keywords if kw.arg])
    return out


def _warned(caplog, fragment):
    return [r for r in caplog.records if fragment in r.getMessage()]


@pytest.fixture(autouse=True)
def _isolated_plugin_state(monkeypatch):
    """Default-off env, fresh once-guards and per-step flags for each test."""
    for key in ENV_KEYS:
        monkeypatch.delenv(key, raising=False)
    graph_plugin._ONCE_SEEN.clear()
    runner_patch._reset_drift_warnings_for_tests()
    monkeypatch.setattr(runner_patch, "_step_cascade", False)
    monkeypatch.setattr(runner_patch, "_step_update_done", False)
    yield
    graph_plugin._ONCE_SEEN.clear()
    runner_patch._reset_drift_warnings_for_tests()
    runner_patch._step_cascade = False
    runner_patch._step_update_done = False


# --------------------------------------------------------------- D1 capture


def test_capture_wrapper_accepts_every_keyword_the_host_call_site_passes():
    """D1: the host passes ``profiler=``; the wrapper must accept it.

    The old wrapper declared ``(self, batch_descriptors, cudagraph_runtime_mode)``
    and raised ``TypeError: unexpected keyword argument 'profiler'`` during
    engine capture, killing engine init.
    """
    call_keywords = _host_call_keywords(
        gpu_model_runner.__file__, "_capture_cudagraphs"
    )
    assert call_keywords, "host _capture_cudagraphs call site not found"
    assert any("profiler" in names for names in call_keywords), (
        "host no longer passes profiler= ; re-audit D1 and this guard"
    )
    wrapper = runner_patch._make_capture_cudagraphs_wrapper(_SigRecorder(_CAPTURE_SIG))
    for names in call_keywords:
        recorder = _SigRecorder(_CAPTURE_SIG)
        wrapper_under_test = runner_patch._make_capture_cudagraphs_wrapper(recorder)
        assert wrapper_under_test(None, **{name: None for name in names}) is None
        assert recorder.signature_errors == []
        assert len(recorder.calls) == 1
    # Signature-agnostic by construction.
    params = inspect.signature(wrapper).parameters.values()
    assert any(p.kind is inspect.Parameter.VAR_POSITIONAL for p in params)
    assert any(p.kind is inspect.Parameter.VAR_KEYWORD for p in params)


def test_capture_wrapper_failure_falls_back_to_orig(monkeypatch, caplog):
    """Fail-open: a broken cascade twin capture must not break capture."""
    monkeypatch.setattr(runner_patch, "_capture_cascade_twins", _boom)
    recorder = _SigRecorder(_CAPTURE_SIG, result="capture-ok")
    wrapper = runner_patch._make_capture_cudagraphs_wrapper(recorder)
    with caplog.at_level(logging.WARNING):
        result = wrapper(
            None,
            batch_descriptors=[],
            cudagraph_runtime_mode=None,
            profiler=None,
        )
    assert result == "capture-ok"
    assert len(recorder.calls) == 1
    assert len(_warned(caplog, "cascade twin capture")) == 1


# ------------------------------------------------- D2 / D4 model forward


def test_model_forward_pre_update_passes_only_accepted_params(monkeypatch):
    """D2: the pre-model update must not pass the dropped ``positions``."""
    monkeypatch.setattr(runner_patch, "_step_is_cascade", lambda: True)
    monkeypatch.setattr(forward_context_mod, "get_forward_context", _FakeForwardContext)
    update_recorder = _SigRecorder(_UPDATE_FULL_SIG, prepend_self=True)
    fake_runner = _FakeRunner(update_recorder)
    orig = _SigRecorder(_MODEL_FORWARD_ASCEND_SIG, result="model-output")
    wrapper = runner_patch._make_model_forward_wrapper(orig)

    assert wrapper(fake_runner, 128) == "model-output"

    assert update_recorder.signature_errors == []
    assert len(update_recorder.calls) == 1
    args, kwargs = update_recorder.calls[0]
    assert args == ()  # keyword-only forwarding, no positional pinning
    assert "positions" not in kwargs
    assert kwargs["num_tokens_padded"] == 128
    assert orig.calls == [((fake_runner, 128), {})]
    assert runner_patch._step_update_done is True


def test_model_forward_wrapper_supports_host_keyword_only_convention(
    monkeypatch,
):
    """D4: upstream calls ``_model_forward`` with keywords only.

    The old wrapper required ``num_tokens_padded`` as its first positional and
    raised ``TypeError: missing 1 required positional argument``.  The padded
    token count must instead be recovered from the forward context descriptor.
    """
    monkeypatch.setattr(runner_patch, "_step_is_cascade", lambda: True)
    fake_context = _FakeForwardContext(num_tokens=64)
    monkeypatch.setattr(
        forward_context_mod, "get_forward_context", lambda: fake_context
    )
    update_recorder = _SigRecorder(_UPDATE_FULL_SIG, prepend_self=True)
    fake_runner = _FakeRunner(update_recorder)
    orig = _SigRecorder(_MODEL_FORWARD_UPSTREAM_SIG, result="ok")
    wrapper = runner_patch._make_model_forward_wrapper(orig)

    result = wrapper(
        fake_runner,
        input_ids=None,
        positions=None,
        intermediate_tensors=None,
        inputs_embeds=None,
    )

    assert result == "ok"
    assert orig.signature_errors == []
    assert len(orig.calls) == 1
    assert set(orig.calls[0][1]) == {
        "input_ids",
        "positions",
        "intermediate_tensors",
        "inputs_embeds",
    }
    assert update_recorder.calls[0][1]["num_tokens_padded"] == 64
    assert update_recorder.calls[0][1]["forward_context"] is fake_context


def test_model_forward_failure_falls_back_to_orig(monkeypatch, caplog):
    """Fail-open: a failing pre-model update must still run the model."""
    monkeypatch.setattr(runner_patch, "_step_is_cascade", lambda: True)
    monkeypatch.setattr(forward_context_mod, "get_forward_context", _FakeForwardContext)
    fake_runner = _FakeRunner(_SigRecorder(_UPDATE_FULL_SIG, prepend_self=True))
    fake_runner._update_full_graph_params_if_needed = _boom
    orig = _SigRecorder(_MODEL_FORWARD_ASCEND_SIG, result="model-output")
    wrapper = runner_patch._make_model_forward_wrapper(orig)
    with caplog.at_level(logging.WARNING):
        result = wrapper(fake_runner, 128)
    assert result == "model-output"
    assert len(orig.calls) == 1
    assert runner_patch._step_update_done is False
    assert len(_warned(caplog, "pre-model graph param update")) == 1


# ----------------------------------------------------- D3 update_graph_params


def test_update_graph_params_forwards_host_convention_verbatim():
    """D3: the host call convention must be forwarded unchanged.

    The old wrapper declared an extra ``num_dcp_pcp_tokens`` parameter and
    always forwarded seven positional arguments, which the host signature
    (six parameters, five positional at its own call site) rejects.
    """
    call_keywords = _host_call_keywords(acl_graph.__file__, "update_graph_params")
    assert call_keywords, "host update_graph_params call site not found"
    assert any("draft_attn_metadatas" in names for names in call_keywords)

    recorder = _SigRecorder(_UPDATE_GRAPH_SIG, result="updated")
    wrapper = graph_plugin._make_update_graph_params_wrapper(recorder)
    args = ("stream", object(), 64, "vllm-config", None)
    assert wrapper(*args, draft_attn_metadatas=None) == "updated"
    assert recorder.signature_errors == []
    assert recorder.calls == [(args, {"draft_attn_metadatas": None})]
    params = inspect.signature(wrapper).parameters.values()
    assert any(p.kind is inspect.Parameter.VAR_POSITIONAL for p in params)
    assert any(p.kind is inspect.Parameter.VAR_KEYWORD for p in params)


def test_update_graph_params_gate_off_forwards_verbatim(monkeypatch):
    """D3: the gate-off branch must delegate, not rebuild the argument list."""
    monkeypatch.setattr(runner_patch, "_step_is_cascade", lambda: True)
    monkeypatch.setattr(
        acl_graph,
        "get_graph_params",
        lambda: types.SimpleNamespace(attn_params={("cascade", 64): [object()]}),
    )
    monkeypatch.setattr(cascade_gate, "decision_for", lambda _shared, _n: False)
    recorder = _SigRecorder(_UPDATE_GRAPH_SIG, result="updated")
    wrapper = graph_plugin._make_update_graph_params_wrapper(recorder)
    context = _FakeForwardContext(num_tokens=64)
    args = ("stream", context, 64, "vllm-config", None)
    assert wrapper(*args, draft_attn_metadatas=None) == "updated"
    assert recorder.signature_errors == []
    assert recorder.calls == [(args, {"draft_attn_metadatas": None})]


def test_update_graph_params_failure_falls_back_to_orig(monkeypatch, caplog):
    """Fail-open: a failing cascade re-parameterization delegates to orig."""
    monkeypatch.setattr(runner_patch, "_step_is_cascade", lambda: True)
    monkeypatch.setattr(
        acl_graph,
        "get_graph_params",
        lambda: types.SimpleNamespace(attn_params={("cascade", 64): [object()]}),
    )
    monkeypatch.setattr(cascade_gate, "decision_for", lambda _shared, _n: True)
    monkeypatch.setattr(graph_plugin, "_update_cascade_graph_params", _boom)
    recorder = _SigRecorder(_UPDATE_GRAPH_SIG, result="updated")
    wrapper = graph_plugin._make_update_graph_params_wrapper(recorder)
    context = _FakeForwardContext(num_tokens=64)
    args = ("stream", context, 64, "vllm-config", None)
    with caplog.at_level(logging.WARNING):
        assert wrapper(*args, draft_attn_metadatas=None) == "updated"
    assert recorder.signature_errors == []
    assert recorder.calls == [(args, {"draft_attn_metadatas": None})]
    assert len(_warned(caplog, "delegating to the original update pass")) == 1


# ------------------------------------------------------- dispatch re-dispatch


def test_determine_batch_wrapper_accepts_host_keyword_only_convention():
    """The host calls the dispatch with keywords only; keep that working."""
    call_keywords = _host_call_keywords(
        gpu_model_runner.__file__, "_determine_batch_execution_and_padding"
    )
    assert call_keywords, "host dispatch call site not found"
    for names in call_keywords:
        assert names, "host dispatch is no longer called keyword-only"
        recorder = _SigRecorder(_DETERMINE_SIG, result="descriptor")
        wrapper = runner_patch._make_determine_batch_wrapper(recorder)
        kwargs = {
            name: (False if name == "use_cascade_attn" else None) for name in names
        }
        assert wrapper(object(), **kwargs) == "descriptor"
        assert recorder.signature_errors == []
        assert len(recorder.calls) == 1


def test_determine_batch_wrapper_redispatch_keeps_host_convention():
    recorder = _SigRecorder(_DETERMINE_SIG, result="descriptor")
    wrapper = runner_patch._make_determine_batch_wrapper(recorder)
    kwargs = dict(
        num_tokens=64,
        num_reqs=2,
        num_scheduled_tokens_np=None,
        max_num_scheduled_tokens=64,
        use_cascade_attn=True,
        allow_microbatching=False,
    )
    assert wrapper(object(), **kwargs) == "descriptor"
    assert recorder.signature_errors == []
    assert recorder.calls[0][1]["use_cascade_attn"] is False
    assert runner_patch._step_is_cascade() is True


def test_determine_batch_wrapper_fails_open_on_unknown_signature(caplog):
    """Future drift: no ``use_cascade_attn`` -> stock dispatch, one warning."""
    recorder = _SigRecorder(
        inspect.signature(lambda self, num_tokens, num_reqs: None),
        result="descriptor",
    )
    wrapper = runner_patch._make_determine_batch_wrapper(recorder)
    with caplog.at_level(logging.WARNING):
        assert wrapper(object(), 64, 2) == "descriptor"
    assert recorder.signature_errors == []
    assert len(recorder.calls) == 1
    assert runner_patch._step_is_cascade() is False
    assert len(_warned(caplog, "batch dispatch re-dispatch")) == 1


# --------------------------------------------------------- source-level guard


def _find_functions(path, names):
    tree = ast.parse(Path(path).read_text(encoding="utf-8"))
    found = {}
    for node in ast.walk(tree):
        if isinstance(node, ast.FunctionDef) and node.name in names:
            found.setdefault(node.name, []).append(node)
    return found


def _called_name(node):
    func = node.func
    if isinstance(func, ast.Name):
        return func.id
    if isinstance(func, ast.Attribute):
        return func.attr
    return None


def test_wrappers_forward_to_orig_via_star_args():
    """Every cascade wrapper must be signature-agnostic and forward verbatim.

    A wrapper that rebuilds the host argument list by position is exactly the
    D2/D3 failure mode; this guard fails if a future wrapper reintroduces it.
    """
    targets = (
        (
            runner_patch.__file__,
            (
                "_determine_batch_execution_and_padding",
                "_capture_cudagraphs",
                "_model_forward",
            ),
        ),
        (graph_plugin.__file__, ("update_graph_params",)),
    )
    for path, names in targets:
        found = _find_functions(path, names)
        for name in names:
            assert name in found, f"{name} wrapper not found in {path}"
            for func in found[name]:
                assert func.args.vararg is not None, f"{name} must accept *args"
                assert func.args.kwarg is not None, f"{name} must accept **kwargs"
                orig_calls = [
                    node
                    for node in ast.walk(func)
                    if isinstance(node, ast.Call) and _called_name(node) == "orig"
                ]
                assert orig_calls, f"{name} never forwards to orig"
                for call in orig_calls:
                    assert any(isinstance(arg, ast.Starred) for arg in call.args), (
                        f"{name} forwards positional args to orig"
                    )
                    assert any(kw.arg is None for kw in call.keywords), (
                        f"{name} does not forward **kwargs to orig"
                    )


def test_host_method_forwards_use_keywords_not_positions():
    """No host private method may be pinned to argument positions."""
    targets = (
        (cascade_plugin.__file__, "_make_build_wrapper", "orig_build"),
        (
            cascade_plugin.__file__,
            "_make_forward_wrapper",
            "orig_forward_fused_infer_attention",
        ),
        (graph_plugin.__file__, "_wrap_build_for_capture", "orig"),
    )
    for path, func_name, callee in targets:
        found = _find_functions(path, (func_name,))
        assert func_name in found, f"{func_name} not found in {path}"
        for func in found[func_name]:
            calls = [
                node
                for node in ast.walk(func)
                if isinstance(node, ast.Call) and _called_name(node) == callee
            ]
            assert calls, f"{callee} not called in {func_name}"
            for call in calls:
                # only ``self`` may be positional; host args go by name
                assert len(call.args) == 1, (
                    f"{func_name} passes host args by position to {callee}"
                )
                assert any(kw.arg for kw in call.keywords), (
                    f"{func_name} does not forward host args by name to {callee}"
                )
