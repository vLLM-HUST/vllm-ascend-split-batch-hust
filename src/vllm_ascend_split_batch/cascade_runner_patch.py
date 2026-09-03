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


def _step_is_cascade() -> bool:
    return _step_cascade


def _patch_determine_batch_execution() -> None:
    """Keep the FULL graph for cascade steps and record the step state."""
    from vllm_ascend.worker.model_runner_v1 import NPUModelRunner

    if getattr(NPUModelRunner, "_cascade_graph_patched", False):
        return

    orig = NPUModelRunner._determine_batch_execution_and_padding

    def _determine_batch_execution_and_padding(
        self,
        num_tokens,
        num_reqs,
        num_scheduled_tokens_np,
        max_num_scheduled_tokens,
        use_cascade_attn,
        allow_microbatching=False,
        force_eager=False,
        force_uniform_decode=None,
        force_has_lora=None,
        force_num_active_loras=None,
        num_encoder_reqs=0,
    ):
        global _step_cascade, _step_update_done
        _step_cascade = bool(use_cascade_attn)
        _step_update_done = False
        # The official dispatch disables FULL for cascade steps; the Ascend
        # backend re-parameterizes the cascade task groups per replay, so
        # FULL remains valid.  Re-dispatch with the flag off keeps every
        # other dispatch input (and the returned descriptor) unchanged.
        return orig(
            self,
            num_tokens,
            num_reqs,
            num_scheduled_tokens_np,
            max_num_scheduled_tokens,
            False,
            allow_microbatching=allow_microbatching,
            force_eager=force_eager,
            force_uniform_decode=force_uniform_decode,
            force_has_lora=force_has_lora,
            force_num_active_loras=force_num_active_loras,
            num_encoder_reqs=num_encoder_reqs,
        )

    NPUModelRunner._determine_batch_execution_and_padding = (
        _determine_batch_execution_and_padding
    )
    NPUModelRunner._cascade_graph_patched = True


def _patch_capture_scheduling() -> None:
    """Capture the cascade twin of every uniform-decode FULL bucket.

    HUST ``_capture_cudagraphs`` post-loop: after the standard capture, each
    uniform-decode bucket is re-captured with a cascade-legal dummy prefix
    (block-aligned, strictly below max_model_len so stage 2 keeps a positive
    suffix).  The capture window (``gp._capture_ctx``) routes the attention
    capture body and the wrapper entry table to the cascade variant.
    """
    from vllm.config import CUDAGraphMode
    from vllm_ascend import envs as envs_mod
    from vllm_ascend.worker.model_runner_v1 import NPUModelRunner

    if getattr(NPUModelRunner, "_cascade_graph_capture_patched", False):
        return

    orig = NPUModelRunner._capture_cudagraphs

    def _capture_cudagraphs(self, batch_descriptors, cudagraph_runtime_mode):
        orig(self, batch_descriptors, cudagraph_runtime_mode)
        if (
            not getattr(envs_mod, "VLLM_ASCEND_ENABLE_CASCADE_DECODE", False)
            or not getattr(envs_mod, "VLLM_ASCEND_ENABLE_CASCADE_GRAPH", False)
            or cudagraph_runtime_mode != CUDAGraphMode.FULL
            or getattr(self, "use_sparse", False)
            or getattr(self, "use_compress", False)
        ):
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

    NPUModelRunner._capture_cudagraphs = _capture_cudagraphs
    NPUModelRunner._cascade_graph_capture_patched = True


def _patch_update_order() -> None:
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
    """
    from vllm_ascend.worker.model_runner_v1 import NPUModelRunner

    if getattr(NPUModelRunner, "_cascade_graph_update_order_patched", False):
        return

    orig = NPUModelRunner._model_forward

    def _model_forward(
        self,
        num_tokens_padded,
        input_ids=None,
        positions=None,
        intermediate_tensors=None,
        inputs_embeds=None,
        **model_kwargs,
    ):
        from vllm.forward_context import get_forward_context

        global _step_update_done
        if _step_is_cascade():
            forward_context = get_forward_context()
            self._update_full_graph_params_if_needed(
                forward_context, num_tokens_padded, positions
            )
            # The official _model_forward calls the update again AFTER
            # run_model(); that second pass would re-run the cascade
            # re-parameterization on the SAME step metadata (pure host/NPU
            # overhead).  Mark the step as updated so the post-call
            # short-circuits (the flag is per-step, re-armed above).
            _step_update_done = True
        return orig(
            self,
            num_tokens_padded,
            input_ids,
            positions,
            intermediate_tensors,
            inputs_embeds,
            **model_kwargs,
        )

    NPUModelRunner._model_forward = _model_forward
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
