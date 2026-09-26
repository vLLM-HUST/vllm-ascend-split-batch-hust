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

"""Runner-side patches for graph-mode cascade decode (default-off).

Extends the official ``NPUModelRunner`` (vllm-ascend) with the HUST fork's
cascade graph scheduling, WITHOUT touching ``BatchDescriptor`` (the official
frozen descriptor has no cascade field; HUST patched core to add one):

  1. **Runtime dispatch**: the official vllm core disables the FULL graph for
     cascade steps (``disable_full=use_cascade_attn``).  On the Ascend
     backend the two-stage task groups are re-parameterized before every
     replay, so the plugin calls the original dispatch with
     ``use_cascade_attn=False`` (FULL stays enabled) and records the cascade
     step in a plugin-side per-step flag consumed by the patched
     ``ACLGraphWrapper`` and update pass.  The returned descriptor is
     unchanged, so non-cascade behavior is bit-identical.
  2. **Capture scheduling**: after the standard FULL capture, capture a
     cascade "twin" per uniform-decode bucket through the runner's own
     ``_warmup_and_capture`` with a cascade-legal dummy prefix.  The twin is
     stored in a per-instance entry table on the ``ACLGraphWrapper`` (keyed
     by the same standard descriptor), so runtime cascade steps replay the
     twin while standard steps keep using the standard graph.

Fail-safe: with the cascade-graph env off every patched method delegates to
the original; a runtime cascade step whose twin is missing fails open to
eager execution (which still runs the two-stage eager path).

Signature discipline (2026-09-09): every wrapper is signature-agnostic --
``(self, *args, **kwargs)`` forwarding to the original verbatim -- and every
cascade-specific step runs inside a ``try`` that logs ONE warning and falls
back to the original implementation on any failure.  Monkeypatched host
methods are private, unversioned seams (see ``docs/pitfalls.md`` 2.1); a
host bump that renames/reorders/removes a parameter must degrade the feature,
never raise out of a patched host method (it used to kill engine init:
``_capture_cudagraphs`` missing ``profiler``).
"""

from __future__ import annotations

import logging
import threading

logger = logging.getLogger(__name__)

_installed = False
_lock = threading.Lock()

# Per-step flag: the current execute_model step is a cascade decode step.
# Set by the patched _determine_batch_execution_and_padding, consumed by the
# patched ACLGraphWrapper (graph-variant selection) and the patched
# update_graph_params (replay re-parameterization).  Cleared on every call,
# including the warmup/capture dummy runs.
_step_cascade = False
# Set by the pre-model update in _patch_update_order; consumed (and cleared)
# by the update wrapper so the official post-model update call short-circuits
# instead of re-running the cascade re-parameterization on the same metadata.
_step_update_done = False

# Once-per-process guard for the fail-open drift warnings (one warning per
# cascade sub-path, never one per step).
_drift_warned: set = set()


def _step_is_cascade() -> bool:
    return _step_cascade


def _cascade_wheel_ready() -> bool:
    """True when the ascend_kernel wheel is importable and registered.

    The W2 gate bench probes and the twin captures both exercise the
    kernel-wheel custom ops; when the wheel is missing or failed to register
    the whole cascade feature is disabled (fail-open), so capture scheduling
    must not bench or capture unusable cascade twins.  Logs the plugin's
    once-per-process fail-open warning on the disabled path.
    """
    from vllm_ascend_split_batch import cascade_plugin

    if cascade_plugin.kernel_wheel_available():
        return True
    cascade_plugin._warn_wheel_missing()
    return False


def _warn_drift_once(where: str, exc: BaseException) -> None:
    """Log ONE warning per sub-path and keep the host method working."""
    if where in _drift_warned:
        return
    _drift_warned.add(where)
    logger.warning(
        "cascade graph: %s could not run on this host build (%s: %s); "
        "delegating to the original implementation (fail-open, stock behavior "
        "for this path)",
        where,
        type(exc).__name__,
        exc,
    )


def _reset_drift_warnings_for_tests() -> None:
    """Clear the once-only drift warning state (unit tests only)."""
    _drift_warned.clear()


def _pick_arg(args, kwargs, index, name):
    """Fetch a host-method argument by name, falling back to its old position.

    Host call sites use both conventions (vllm core passes keyword-only,
    vllm-ascend passes positionally), and both must keep working.  Returns
    ``None`` when the argument is absent from both, which callers turn into a
    fail-open fallback.
    """
    if name in kwargs:
        return kwargs[name]
    if index is not None and len(args) > index:
        return args[index]
    return None


def _use_ubatching(runner) -> bool:
    """True when the runner splits batches (cascade is mutually exclusive)."""
    parallel_config = getattr(
        getattr(runner, "vllm_config", None), "parallel_config", None
    )
    return bool(getattr(parallel_config, "use_ubatching", False))


def _spec_decode_active(runner) -> bool:
    """True when speculative decoding is configured (skip the twin capture).

    Cascade admits exactly one query row per request; MTP and the other
    speculative methods feed ``k + 1`` rows, so the cascade twin must not be
    captured for them -- the standard FULL graph keeps serving those steps.
    Mirrors the dispatch-gate guard in ``cascade_plugin._use_cascade_attention``
    (see README "Speculative decoding boundary").
    """
    vllm_config = getattr(runner, "vllm_config", None)
    return getattr(vllm_config, "speculative_config", None) is not None


def _make_determine_batch_wrapper(orig):
    """Wrap the host dispatch so cascade steps keep the FULL graph.

    The wrapper accepts the host's call convention verbatim (``*args``,
    ``**kwargs``) and only re-dispatches when ``use_cascade_attn`` is set.
    Fail-open: if the host no longer exposes that argument (renamed, moved,
    removed) the original call is forwarded untouched after ONE warning and
    the cascade step flag stays False, i.e. the stock path runs.
    """

    def _determine_batch_execution_and_padding(self, *args, **kwargs):
        global _step_cascade, _step_update_done
        cascade_step = False
        kwargs_override = None
        args_override = None
        try:
            use_cascade_attn = _pick_arg(args, kwargs, 4, "use_cascade_attn")
            if not isinstance(use_cascade_attn, bool):
                raise RuntimeError(
                    "use_cascade_attn is neither passed by keyword nor the "
                    "5th positional argument of "
                    "_determine_batch_execution_and_padding"
                )
            if use_cascade_attn:
                # Mirror the official vllm core gate: cascade attention is
                # disabled under ANY microbatching (enable_dbo OR
                # ubatch_size > 1).  The asc runner only checks enable_dbo in
                # its own execute_model guard, so without this the two-stage
                # path would still be engaged while the batch is split by
                # UBatchWrapper (known loss at BS>=threshold).
                cascade_step = not _use_ubatching(self)
                # The official dispatch disables FULL for cascade steps; the
                # Ascend backend re-parameterizes the cascade task groups per
                # replay, so FULL remains valid.  Re-dispatch with the flag
                # off keeps every other dispatch input (and the returned
                # descriptor) unchanged.
                if "use_cascade_attn" in kwargs:
                    kwargs_override = {**kwargs, "use_cascade_attn": False}
                else:
                    args_override = (*args[:4], False, *args[5:])
        except Exception as exc:
            _warn_drift_once("batch dispatch re-dispatch", exc)
            cascade_step = False
            kwargs_override = None
            args_override = None
        _step_cascade = cascade_step
        _step_update_done = False
        if kwargs_override is not None:
            return orig(self, *args, **kwargs_override)
        if args_override is not None:
            return orig(self, *args_override, **kwargs)
        return orig(self, *args, **kwargs)

    return _determine_batch_execution_and_padding


def _patch_determine_batch_execution() -> None:
    """Keep the FULL graph for cascade steps and record the step state."""
    from vllm_ascend.worker.model_runner_v1 import NPUModelRunner

    if getattr(NPUModelRunner, "_cascade_graph_patched", False):
        return

    NPUModelRunner._determine_batch_execution_and_padding = (
        _make_determine_batch_wrapper(
            NPUModelRunner._determine_batch_execution_and_padding
        )
    )
    NPUModelRunner._cascade_graph_patched = True


def _capture_cascade_twins(self, args, kwargs) -> None:
    """Capture the cascade twin of every uniform-decode FULL bucket.

    HUST ``_capture_cudagraphs`` post-loop: after the standard capture, each
    uniform-decode bucket is re-captured with a cascade-legal dummy prefix
    (block-aligned, strictly below max_model_len so stage 2 keeps a positive
    suffix).  The capture window (``gp._capture_ctx``) routes the attention
    capture body and the wrapper entry table to the cascade variant.
    """
    from vllm.config import CUDAGraphMode
    from vllm_ascend import envs as envs_mod

    if not (
        getattr(envs_mod, "VLLM_ASCEND_ENABLE_CASCADE_DECODE", False)
        and getattr(envs_mod, "VLLM_ASCEND_ENABLE_CASCADE_GRAPH", False)
    ):
        return
    batch_descriptors = _pick_arg(args, kwargs, 0, "batch_descriptors")
    cudagraph_runtime_mode = _pick_arg(
        args, kwargs, 1, "cudagraph_runtime_mode"
    )
    if batch_descriptors is None or cudagraph_runtime_mode is None:
        raise RuntimeError(
            "host _capture_cudagraphs no longer takes batch_descriptors / "
            "cudagraph_runtime_mode"
        )
    if (
        cudagraph_runtime_mode != CUDAGraphMode.FULL
        or getattr(self, "use_sparse", False)
        or getattr(self, "use_compress", False)
        or _use_ubatching(self)
        or _spec_decode_active(self)
    ):
        return
    if not _cascade_wheel_ready():
        return
    from vllm_ascend_split_batch import cascade_graph_plugin as gp

    blk = 0
    kv_groups = getattr(self, "kv_cache_config", None)
    if kv_groups is not None and getattr(kv_groups, "kv_cache_groups", None):
        blk = kv_groups.kv_cache_groups[0].kv_cache_spec.block_size
    if not blk:
        return
    max_model_len = getattr(self, "max_model_len", 0)
    dummy_prefix = min(
        getattr(envs_mod, "VLLM_ASCEND_CASCADE_MIN_PREFIX", 8192),
        (max_model_len - 1) // blk * blk,
    )
    if dummy_prefix < blk:
        return
    # --- W2 gate: micro-bench BEFORE the twin captures ----------------
    # Ordering is load-bearing: the full-FIA probes must run before the
    # process' first fa_fp32_stage1 call (W0 coexistence hazard: stage-1
    # poisons subsequent >=8k eager FIA in the same process).
    from vllm_ascend_split_batch import cascade_gate as gate

    if gate.enabled() and gate.override() is None:
        try:
            gate.bench_all(self, batch_descriptors, blk)
        except Exception:
            logger.exception(
                "cascade gate bench failed; gate stays neutral "
                "(cascade-on everywhere)"
            )
    for batch_desc in batch_descriptors:
        if not batch_desc.uniform:
            continue
        # The dummy batch must have sequences strictly longer than the
        # dummy shared prefix (profile_seq_lens = 2x prefix), otherwise
        # the stage-2 suffix is empty and the capture falls back to the
        # standard graph body.
        profile_seq_lens = min(
            2 * getattr(envs_mod, "VLLM_ASCEND_CASCADE_MIN_PREFIX", 8192),
            self.max_model_len,
        )
        gp._capture_ctx.active = True
        gp._capture_ctx.shared_len = dummy_prefix
        try:
            self._warmup_and_capture(
                batch_desc,
                cudagraph_runtime_mode=cudagraph_runtime_mode,
                num_warmups=1,
                profile_seq_lens=profile_seq_lens,
            )
        finally:
            gp._capture_ctx.active = False
            gp._capture_ctx.shared_len = 0
    # Freshly captured twin graphs are bound to DUMMY capture-time
    # parameters; any stage-1 stability signature cached from a previous
    # serving step is stale by construction.
    gp.invalidate_stage1_cache()


def _make_capture_cudagraphs_wrapper(orig):
    """Wrap the host capture so cascade twins are captured after it.

    Accepts the host's call convention verbatim (the v1 host calls with
    ``batch_descriptors=`` / ``cudagraph_runtime_mode=`` / ``profiler=``
    keywords) and always runs the original first.  Any failure of the
    cascade-specific post-step logs ONE warning and is dropped: capture
    proceeds with the standard graphs only.
    """

    def _capture_cudagraphs(self, *args, **kwargs):
        result = orig(self, *args, **kwargs)
        try:
            _capture_cascade_twins(self, args, kwargs)
        except Exception as exc:
            _warn_drift_once("cascade twin capture", exc)
        return result

    return _capture_cudagraphs


def _patch_capture_scheduling() -> None:
    """Capture the cascade twin of every uniform-decode FULL bucket."""
    from vllm_ascend.worker.model_runner_v1 import NPUModelRunner

    if getattr(NPUModelRunner, "_cascade_graph_capture_patched", False):
        return

    NPUModelRunner._capture_cudagraphs = _make_capture_cudagraphs_wrapper(
        NPUModelRunner._capture_cudagraphs
    )
    NPUModelRunner._cascade_graph_capture_patched = True


def _make_model_forward_wrapper(orig):
    """Run the replay parameter update BEFORE the model call on cascade steps.

    Official ``_model_forward`` runs the model (which replays the captured
    aclgraph) first and the parameter update afterwards; that update serves
    the NEXT step's replay.  A cascade replay must not run with the
    CAPTURE-time dummy parameters (kv ~3072, dummy blocks) against the real
    step (kv ~1152): the captured task-group tiling reads out of the real
    KV cache range and the device aborts with 507011 on the first event
    sync.  The HUST fork calls ``_update_full_graph_params_if_needed``
    BEFORE ``run_model()`` unconditionally
    (cascade-e2e model_runner_v1.py:3044-3047).  The plugin re-orders the
    same way for cascade steps only; non-cascade steps keep the official
    order bit-for-bit.

    Signature discipline: the wrapper takes ``*args, **kwargs`` so both host
    call conventions work (vllm-ascend calls positionally with
    ``num_tokens_padded`` first; vllm core calls keyword-only without it).
    When the padded token count is not in the call, it is read from
    ``forward_context.batch_descriptor.num_tokens`` (the same value the graph
    dispatch recorded).  Any failure logs ONE warning and the original runs
    unchanged -- never raise out of a patched host method.
    """

    def _model_forward(self, *args, **kwargs):
        global _step_update_done
        try:
            if _step_is_cascade():
                from vllm.forward_context import get_forward_context

                forward_context = get_forward_context()
                num_tokens_padded = _pick_arg(
                    args, kwargs, 0, "num_tokens_padded"
                )
                if not isinstance(num_tokens_padded, int):
                    descriptor = getattr(
                        forward_context, "batch_descriptor", None
                    )
                    num_tokens_padded = getattr(descriptor, "num_tokens", None)
                if not isinstance(num_tokens_padded, int):
                    raise RuntimeError(
                        "num_tokens_padded is unavailable (host "
                        "_model_forward signature/convention changed)"
                    )
                self._update_full_graph_params_if_needed(
                    forward_context=forward_context,
                    num_tokens_padded=num_tokens_padded,
                )
                # The official _model_forward calls the update again AFTER
                # run_model(); that second pass would re-run the cascade
                # re-parameterization on the SAME step metadata (pure host/NPU
                # overhead).  Mark the step as updated so the post-call
                # short-circuits (the flag is per-step, re-armed above).
                _step_update_done = True
        except Exception as exc:
            _warn_drift_once("pre-model graph param update", exc)
        return orig(self, *args, **kwargs)

    return _model_forward


def _patch_update_order() -> None:
    """Patch ``_model_forward`` for the pre-model replay update."""
    from vllm_ascend.worker.model_runner_v1 import NPUModelRunner

    if getattr(NPUModelRunner, "_cascade_graph_update_order_patched", False):
        return

    NPUModelRunner._model_forward = _make_model_forward_wrapper(
        NPUModelRunner._model_forward
    )
    NPUModelRunner._cascade_graph_update_order_patched = True


def install() -> bool:
    """Patch the NPUModelRunner for graph-mode cascade (idempotent)."""
    global _installed
    if _installed:
        return True
    with _lock:
        if _installed:
            return True
        try:
            _patch_determine_batch_execution()
            _patch_capture_scheduling()
            _patch_update_order()
        except Exception:
            logger.exception(
                "cascade graph runner patch failed; graph cascade stays "
                "disabled (eager cascade unaffected)"
            )
            return False
        _installed = True
        return True
