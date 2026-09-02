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

"""Graph-mode two-stage cascade decode plugin (default-off).

Plugin-side port of the HUST fork cascade *graph* integration
(``plan811/lo-dual-stream-0826`` @ ``1407d2a12``).  It extends the eager
cascade plugin (``cascade_plugin.py``) with:

1. **Capture**: one cascade "twin" aclgraph per uniform-decode FULL bucket,
   registered on the official ``CudagraphDispatcher`` via its public
   ``add_cudagraph_key`` and captured through the runner's own warmup/capture
   flow with a cascade-legal dummy prefix.
2. **Dispatch**: runtime cascade steps dispatch to the cascade descriptor
   (``dataclasses.replace(desc, cascade=True)``) so the wrapper selects the
   cascade graph variant instead of the standard decode graph.
3. **Capture path**: during cascade-twin capture the attention impl calls the
   two-stage capture body (``full_graph_fia_cascade`` port) so the two-stage
   FIA task groups land in the graph under the ``("cascade", num_tokens)``
   param key.
4. **Replay**: the official per-replay parameter update pass re-parameterizes
   the captured stage-1/stage-2 task groups from the step metadata (shared
   prefix length, per-request suffix lengths, block-table slices).

Variant table instead of a descriptor flag: the official ``BatchDescriptor``
is frozen and has no cascade field (HUST patched core to add one).  The
plugin keeps an entry table on each ``ACLGraphWrapper`` instance keyed by
the *standard* descriptor; when a cascade step replays, the wrapper swaps
the ``concrete_aclgraph_entries`` mapping to the cascade table for the
duration of the call, so descriptor equality selects the cascade twin graph
without altering the host dataclass.

Host source trees stay untouched; everything rides the official public
surface: the runner's public capture methods and vllm-ascend's
``update_full_graph_params`` entry.
"""

from __future__ import annotations

import logging
import threading

import torch
import torch_npu  # noqa: F401
import vllm_ascend.envs as envs_mod

from vllm_ascend_split_batch.cascade_plugin import (
    _cascade_has_short_real_request,
    _cascade_stage2_kv_lens,
)

logger = logging.getLogger(__name__)

# vllm's logger adds *_once helpers; a plain logging.Logger does not, so the
# plugin keeps its own once-set and uses plain logging calls.
_ONCE_SEEN: set = set()


def _warn_once(msg, *args):
    key = "warn:" + msg
    if key not in _ONCE_SEEN:
        _ONCE_SEEN.add(key)
        logger.warning(msg, *args)


def _info_once(msg, *args):
    key = "info:" + msg
    if key not in _ONCE_SEEN:
        _ONCE_SEEN.add(key)
        logger.info(msg, *args)


def _trace(msg, *args):
    """Diagnostic trace (enabled with VLLM_ASCEND_CASCADE_TRACE=1)."""
    import os as _os

    if _os.getenv("VLLM_ASCEND_CASCADE_TRACE") == "1":
        print("[cas-trace] " + (msg % args if args else msg), flush=True)

# ---------------------------------------------------------------- bookkeeping

# Set by install().  Guards against double installation.
_installed = False
_lock = threading.Lock()

# Original (unwrapped) methods kept for uninstall-free fail-safe behavior.
_orig = {}

# Tier-1 fp32 stage-1 device inputs per ("cascade", num_tokens) key:
# (q_seqlens, kv_seqlens, bt_shared) bucket-static graph-pool tensors whose
# CONTENT the replay update pass refreshes in place.
_S1_SEQLENS: dict = {}


def _ensure_graph_param_key(graph_params, key) -> bool:
    """Ensure ``("cascade", num_tokens)`` lists exist on official GraphParams.

    Official GraphParams pre-creates only integer bucket keys; the cascade
    variant needs its own key.  Mirrors the HUST ``ensure_graph_param_key``.
    """
    if graph_params is None:
        return False
    try:
        graph_params.events.setdefault(key, [])
        graph_params.handles.setdefault(key, [])
        graph_params.attn_params.setdefault(key, [])
        graph_params.workspaces.setdefault(key, None)
    except Exception:
        return False
    return True


# ------------------------------------------------------------ capture helpers


def _build_cascade_capture_metadata(self, common_attn_metadata):
    """Cascade-variant capture metadata (HUST build_for_cascade_graph_capture).

    Runs on the patched builder inside the cascade-twin capture window: the
    wrapper layer (installed by install()) has stashed the shared prefix to
    use for this build call, so build() produces metadata that satisfies the
    cascade gate.  Runtime replays overwrite every dynamic parameter, so the
    capture value only needs to be legal.
    """
    shared_len = getattr(common_attn_metadata, "_cascade_capture_shared_len", 0)
    return self._cascade_orig_build(
        common_prefix_len=shared_len, common_attn_metadata=common_attn_metadata
    )


def _wrap_build_for_capture(builder_cls):
    """Patch build_for_cudagraph_capture for the cascade capture window.

    The official capture path calls ``build_for_cudagraph_capture`` (which
    builds with common_prefix_len=0 and would never satisfy the cascade gate).
    During cascade-twin capture we translate it into a cascade build with the
    stashed dummy prefix; outside the window the original runs unchanged.
    """
    orig = builder_cls.build_for_cudagraph_capture

    def build_for_cudagraph_capture(self, common_attn_metadata):
        from vllm_ascend_split_batch import cascade_graph_plugin as gp

        shared = gp._capture_ctx.shared_len
        if gp._capture_ctx.active and shared > 0:
            return self._cascade_orig_build(
                common_prefix_len=shared,
                common_attn_metadata=common_attn_metadata,
            )
        return orig(self, common_attn_metadata)

    builder_cls.build_for_cudagraph_capture = build_for_cudagraph_capture


# --------------------------------------------------------------- capture body


def _full_graph_fia_cascade(
    self,
    query,
    key,
    value,
    attn_metadata,
    output,
    kv_cache=None,
):
    """Capture-side two-stage cascade body (HUST full_graph_fia_cascade).

    Called (via the patched ``forward_fused_infer_attention``) while the
    cascade twin aclgraph is being captured.  Registers the two-stage FIA
    task groups under the ("cascade", num_tokens) graph param key; every
    dynamic parameter is re-derived at replay from the step metadata.
    """
    from vllm_ascend.compilation.acl_graph import (
        get_graph_params,
        update_graph_params_workspaces,
    )
    from vllm_ascend.utils import weak_ref_tensors

    key_t, value_t, block_size, block_table, _ = self._get_fia_params(
        key, value, attn_metadata, kv_cache
    )
    shared_len = attn_metadata.cascade_shared_len
    seq_lens_list = attn_metadata.seq_lens_list
    num_tokens = len(attn_metadata.actual_seq_lengths_q)
    shared_blocks = shared_len // block_size
    indep_kv_lens = _cascade_stage2_kv_lens(seq_lens_list, num_tokens, shared_len)
    if (
        shared_blocks < 1
        or _cascade_has_short_real_request(seq_lens_list, num_tokens, shared_len)
    ):
        _warn_once(
            "cascade graph capture fell back to the standard path "
            "(shared_len=%s num_tokens=%s)",
            shared_len,
            num_tokens,
        )
        # Capture-time sanity: the captured graph must match a non-cascade
        # step, so fall back to the standard full-graph body.
        return self._cascade_orig_forward_capture(
            self, query, key, value, attn_metadata, output, kv_cache
        )

    _trace("capture body: num_tokens=%s shared_len=%s", num_tokens, shared_len)
    graph_params = get_graph_params()
    param_key = ("cascade", num_tokens)
    if not _ensure_graph_param_key(graph_params, param_key):
        _warn_once(
            "cascade capture: GraphParams unavailable; standard path used"
        )
        return self._cascade_orig_forward_capture(
            self, query, key, value, attn_metadata, output, kv_cache
        )

    num_kv_heads, num_heads = self.num_kv_heads, self.num_heads
    q_lens_cs = attn_metadata.actual_seq_lengths_q[:num_tokens]

    # Graph-mode Tier-1 (fp32 stage-1 in-graph) stays disabled: the HUST
    # in-graph fp32 stage-1 e2e integration is not green (capture_end
    # 507903 / replay 507011 hazards; see plan-20260903-mc-integration C2).
    # Graph cascade runs Tier-0 (bf16) two-stage task groups in both
    # precision modes; fp32 remains an eager-only capability.

    # Block-table slices (graph-pool copies; re-sliced per step in the update
    # pass with fresh pointers).
    bt_shared = block_table[:1, :shared_blocks].contiguous()
    bt_rest = block_table[:, shared_blocks:].contiguous()

    # Bucket-static workspaces, allocated ONCE per (cascade, num_tokens)
    # bucket and shared by every cascade layer (the official FIA update pass
    # shares one workspace the same way).  Sizing uses the block-table
    # capacity, an upper bound for any shared/indep split this bucket can
    # ever see, so replays never outgrow the captured workspace.  Stage 1
    # and stage 2 get SEPARATE buffers: task groups may execute
    # concurrently and sharing one buffer races (HUST 14B repro: 9/64 arms
    # diverged).
    bt_cols = block_table.shape[1]
    kv_bound = bt_cols * block_size
    if graph_params.workspaces.get(param_key) is None:
        ws_stage1 = (
            torch_npu._npu_fused_infer_attention_score_v2_get_max_workspace(
                query=query,
                key=key_t,
                value=value_t,
                block_table=block_table[:1],
                block_size=block_size,
                actual_seq_qlen=[num_tokens],
                actual_seq_kvlen=[kv_bound],
                num_query_heads=num_heads,
                num_key_value_heads=num_kv_heads,
                input_layout="TND",
                softmax_scale=self.scale,
                return_softmax_lse=True,
            )
        )
        ws_stage2 = (
            torch_npu._npu_fused_infer_attention_score_v2_get_max_workspace(
                query=query,
                key=key_t,
                value=value_t,
                block_table=block_table,
                block_size=block_size,
                actual_seq_qlen=q_lens_cs,
                actual_seq_kvlen=[kv_bound] * num_tokens,
                num_query_heads=num_heads,
                num_key_value_heads=num_kv_heads,
                input_layout="TND",
                softmax_scale=self.scale,
                return_softmax_lse=True,
            )
        )
        update_graph_params_workspaces(param_key, (ws_stage1, ws_stage2))
    ws_stage1, ws_stage2 = graph_params.workspaces.get(param_key)

    stream = torch_npu.npu.current_stream()
    event_pre = torch.npu.ExternalEvent()
    event_pre.wait(stream)
    event_pre.reset(stream)
    graph_params.events[param_key].append(event_pre)
    _info_once(
        "cascade aclgraph captured: num_tokens=%s shared_len=%s kv_bound=%s",
        num_tokens,
        shared_len,
        kv_bound,
    )

    # Persistent graph-pool intermediates, shared by all cascade layers of
    # this bucket (layers execute serially inside the graph).
    o1 = torch.empty(
        num_tokens, num_heads, self.head_size, dtype=query.dtype, device=query.device
    )
    l1 = torch.empty(
        num_tokens, num_heads, 1, dtype=torch.float32, device=query.device
    )
    # Stage 1 = FIA task group (re-parameterized per step).
    torch.npu.graph_task_group_begin(stream)
    torch_npu.npu_fused_infer_attention_score_v2.out(
        query=query,
        key=key_t,
        value=value_t,
        block_table=bt_shared,
        block_size=block_size,
        actual_seq_qlen=[num_tokens],
        actual_seq_kvlen=[shared_len],
        num_query_heads=num_heads,
        num_key_value_heads=num_kv_heads,
        input_layout="TND",
        softmax_scale=self.scale,
        return_softmax_lse=True,
        workspace=ws_stage1,
        out=[o1, l1],
    )
    handle_pre = torch.npu.graph_task_group_end(stream)
    graph_params.handles[param_key].append(handle_pre)

    event_suf = torch.npu.ExternalEvent()
    event_suf.wait(stream)
    event_suf.reset(stream)
    graph_params.events[param_key].append(event_suf)
    # Stage 2 = FIA task group (per-request suffix params change every step).
    o2 = torch.empty(
        num_tokens, num_heads, self.head_size, dtype=query.dtype, device=query.device
    )
    l2 = torch.empty(
        num_tokens, num_heads, 1, dtype=torch.float32, device=query.device
    )
    torch.npu.graph_task_group_begin(stream)
    torch_npu.npu_fused_infer_attention_score_v2.out(
        query=query,
        key=key_t,
        value=value_t,
        block_table=bt_rest,
        block_size=block_size,
        actual_seq_qlen=q_lens_cs,
        actual_seq_kvlen=indep_kv_lens,
        num_query_heads=num_heads,
        num_key_value_heads=num_kv_heads,
        input_layout="TND",
        softmax_scale=self.scale,
        return_softmax_lse=True,
        workspace=ws_stage2,
        out=[o2, l2],
    )
    handle_suf = torch.npu.graph_task_group_end(stream)
    graph_params.handles[param_key].append(handle_suf)

    l1_v = l1.reshape(num_tokens, num_heads)
    l2_v = l2.reshape(num_tokens, num_heads)
    merged = None
    from vllm_ascend_split_batch.cascade_plugin import (
        _HAS_LSE_MERGE_OP as has_lse_merge,
    )

    if has_lse_merge:
        try:
            merged = torch.ops.npu.lse_merge(o1, o2, l1_v, l2_v)
        except Exception:
            if not getattr(self, "_lse_merge_fallback_logged", False):
                logger.exception(
                    "lse_merge kernel failed during capture, falling back to "
                    "torch merge"
                )
                self._lse_merge_fallback_logged = True
            merged = None
    if merged is None:
        m = torch.maximum(l1_v, l2_v)
        w_pre = torch.exp(l1_v - m)
        w_suf = torch.exp(l2_v - m)
        merged = (
            (o1.float() * w_pre.unsqueeze(-1) + o2.float() * w_suf.unsqueeze(-1))
            / (w_pre + w_suf).unsqueeze(-1)
        ).to(o1.dtype)
    output[:num_tokens] = merged.reshape(o1.shape)

    # One attn_params entry per layer; handles/events are two-per-layer
    # (Tier-0 FIA task groups).
    layer_name = (
        self._graph_metadata_layer_name(None)
        if getattr(self, "_use_layer_aware_fia_graph_replay", False)
        else None
    )
    graph_params.attn_params[param_key].append(
        (
            "cascade",
            weak_ref_tensors(query),
            weak_ref_tensors(key_t),
            weak_ref_tensors(value_t),
            weak_ref_tensors(output),
            weak_ref_tensors(o1),
            weak_ref_tensors(l1),
            weak_ref_tensors(o2),
            weak_ref_tensors(l2),
            block_size,
            num_kv_heads,
            num_heads,
            self.scale,
            num_tokens,
            layer_name,
        )
    )
    # Keep the graph-pool intermediates alive for the lifetime of this impl:
    # weak refs in attn_params do NOT own the buffers, and a freed buffer is
    # reused by later allocations while the replayed nodes still write to its
    # old address (observed as 507011 MTE out-of-range on first replay).
    # All layers of the bucket accumulate here, mirroring the HUST fork.
    bufs = getattr(self, "_cascade_graph_buffers", None)
    if bufs is None:
        bufs = self._cascade_graph_buffers = {}
    bufs.setdefault(param_key, []).append(
        (o1, l1, o2, l2, merged.reshape(o1.shape), bt_shared, bt_rest)
    )
    _trace(
        "capture body SUCCESS: num_tokens=%s layers-so-far=%s",
        num_tokens,
        len(graph_params.attn_params[param_key]),
    )
    return output, num_tokens


# ----------------------------------------------------------- replay update


def _update_cascade_graph_params(
    update_stream, forward_context, graph_params, num_tokens
):
    """Re-parameterize the captured cascade two-stage FIA task groups.

    Called before every cascade-graph replay.  All cascade layers of a
    bucket share one metadata instance, so the per-step dynamic parameters
    (shared prefix length, per-request suffix lengths, block-table slices)
    are derived once and reused for every layer's task update.
    """
    cascade_key = ("cascade", num_tokens)
    captured = graph_params.attn_params.get(cascade_key) or []
    handles = graph_params.handles.get(cascade_key) or []
    events = graph_params.events.get(cascade_key) or []
    if not captured:
        return
    attn_metadata = forward_context.attn_metadata

    first_param = captured[0]
    layer_name = first_param[14]
    metadata = attn_metadata.get(layer_name) if layer_name else None
    if metadata is None:
        metadata = next(iter(attn_metadata.values()))
    shared_len = getattr(metadata, "cascade_shared_len", 0)
    if shared_len <= 0:
        # Step metadata lost the cascade decision.  The captured graph's
        # in-graph ExternalEvent waits must be released or the replay
        # dead-locks: record all events, then skip re-parameterization.
        _warn_once(
            "cascade replay without cascade_shared_len in metadata "
            "(layer=%s keys=%s)",
            layer_name,
            list(attn_metadata.keys())[:2],
        )
        with torch.npu.stream(update_stream):
            for ev in events:
                ev.record(update_stream)
        return

    (_kind, query, key_t, value_t, _output, o1, l1, o2, l2, block_size,
     num_kv_heads, num_heads, scale, num_tokens_cap, _layer_name_unused) = first_param

    groups_per_layer = 2  # Tier-0: stage-1 + stage-2 FIA task groups
    events_per_layer = 2
    if (
        len(handles) < groups_per_layer * len(captured)
        or len(events) < events_per_layer * len(captured)
    ):
        _warn_once(
            "cascade graph param lists misaligned (%d params, %d handles, %d "
            "events); skipping update",
            len(captured),
            len(handles),
            len(events),
        )
        with torch.npu.stream(update_stream):
            for ev in events:
                ev.record(update_stream)
        return

    num_tokens_i = len(metadata.actual_seq_lengths_q)
    indep_kv_lens = _cascade_stage2_kv_lens(
        metadata.seq_lens_list, num_tokens_i, shared_len
    )
    if _cascade_has_short_real_request(
        metadata.seq_lens_list, num_tokens_i, shared_len
    ):
        _warn_once(
            "cascade replay with short real request (shared=%s seq=%s)",
            shared_len,
            str(metadata.seq_lens_list[:4]),
        )
        with torch.npu.stream(update_stream):
            for ev in events:
                ev.record(update_stream)
        return
    bt = metadata.block_tables
    sb = shared_len // block_size
    bt_shared = bt[:1, :sb].contiguous()
    bt_rest = bt[:, sb:].contiguous()
    q_lens_cs = metadata.actual_seq_lengths_q[:num_tokens_i]
    workspaces = graph_params.workspaces.get(cascade_key)
    ws_stage1, ws_stage2 = (
        workspaces if isinstance(workspaces, tuple) else (workspaces, workspaces)
    )

    with torch.npu.stream(update_stream):
        for i, param in enumerate(captured):
            query_i = param[1]
            key_i = param[2]
            value_i = param[3]
            o1_i, l1_i = param[5], param[6]
            o2_i, l2_i = param[7], param[8]

            torch.npu.graph_task_update_begin(update_stream, handles[2 * i])
            torch_npu.npu_fused_infer_attention_score_v2.out(
                query=query_i,
                key=key_i,
                value=value_i,
                block_table=bt_shared,
                block_size=block_size,
                actual_seq_qlen=[num_tokens_i],
                actual_seq_kvlen=[shared_len],
                num_query_heads=num_heads,
                num_key_value_heads=num_kv_heads,
                input_layout="TND",
                softmax_scale=scale,
                return_softmax_lse=True,
                workspace=ws_stage1,
                out=[o1_i, l1_i],
            )
            torch.npu.graph_task_update_end(update_stream)
            events[2 * i].record(update_stream)

            torch.npu.graph_task_update_begin(update_stream, handles[2 * i + 1])
            torch_npu.npu_fused_infer_attention_score_v2.out(
                query=query_i,
                key=key_i,
                value=value_i,
                block_table=bt_rest,
                block_size=block_size,
                actual_seq_qlen=q_lens_cs,
                actual_seq_kvlen=indep_kv_lens,
                num_query_heads=num_heads,
                num_key_value_heads=num_kv_heads,
                input_layout="TND",
                softmax_scale=scale,
                return_softmax_lse=True,
                workspace=ws_stage2,
                out=[o2_i, l2_i],
            )
            torch.npu.graph_task_update_end(update_stream)
            events[2 * i + 1].record(update_stream)




def _wrap_aclgraph_wrapper(ACLGraphWrapper) -> None:
    """Patch ACLGraphWrapper.__call__ for cascade variant selection.

    Entry tables: standard captures fill ``concrete_aclgraph_entries``;
    cascade-twin captures (during the plugin capture window) are routed into
    the per-instance ``_cascade_aclgraph_entries`` table under the same
    standard descriptor key.  During a cascade replay the table mapping is
    swapped for the duration of the call so descriptor equality resolves to
    the cascade twin; everything else (capture detection, eager fail-open,
    output handling) is the official code path.
    """
    import vllm_ascend.envs as envs_mod

    from vllm_ascend_split_batch.cascade_runner_patch import _step_is_cascade

    orig_call = ACLGraphWrapper.__call__

    def __call__(self, *args, **kwargs):
        from vllm_ascend_split_batch import cascade_graph_plugin as gp

        if not getattr(envs_mod, "VLLM_ASCEND_ENABLE_CASCADE_DECODE", False):
            return orig_call(self, *args, **kwargs)

        capture_window = bool(getattr(gp._capture_ctx, "active", False))
        cascade_replay = _step_is_cascade() and not capture_window
        _trace(
            "wrapper: cascade_flag=%s capture_window=%s replay_swap=%s",
            _step_is_cascade(), capture_window, cascade_replay,
        )
        if capture_window or cascade_replay:
            # Cascade twin capture: route the new graph into the cascade
            # table (a standard descriptor key would otherwise overwrite the
            # standard graph).  Cascade replay: select the cascade table so
            # descriptor equality resolves to the twin graph.
            _wrap_aclgraph_entries(self)
            orig_entries = self.concrete_aclgraph_entries
            self.concrete_aclgraph_entries = self._cascade_aclgraph_entries
            try:
                return orig_call(self, *args, **kwargs)
            finally:
                self.concrete_aclgraph_entries = orig_entries
        return orig_call(self, *args, **kwargs)

    ACLGraphWrapper.__call__ = __call__

# ---------------------------------------------------------------- install()


class _CaptureContext:
    """Per-process cascade capture window state."""

    def __init__(self):
        self.active = False
        self.shared_len = 0


_capture_ctx = _CaptureContext()


def _wrap_aclgraph_entries(wrapper):
    """Give an ACLGraphWrapper instance a cascade variant entry table.

    Official ACLGraphWrapper stores captured graphs in
    ``concrete_aclgraph_entries[batch_descriptor]``.  The cascade twin is
    captured for the same standard descriptor, so it would overwrite the
    standard graph.  The plugin therefore keeps a second table
    (``_cascade_aclgraph_entries``); during a cascade replay the wrapper's
    ``__call__`` temporarily points ``concrete_aclgraph_entries`` at the
    cascade table (see the patched __call__ below).
    """
    if getattr(wrapper, "_cascade_variants_ready", False):
        return wrapper
    wrapper._cascade_aclgraph_entries = {}
    wrapper._cascade_variants_ready = True
    return wrapper


def install(attn_mod, builder_cls, impl_cls):
    """Patch the host stack for graph-mode cascade (idempotent).

    Patch surfaces (all on the plugin's own import graph, none on host
    source):
      - ``AscendAttentionMetadataBuilder.build``          (already patched by
        the eager plugin for cascade_shared_len; reused for capture builds)
      - ``AscendAttentionMetadataBuilder.build_for_cudagraph_capture``
        (capture-window translation to cascade metadata)
      - ``AscendAttentionBackendImpl.forward_fused_infer_attention``
        (capture-window dispatch into the two-stage capture body)
      - ``AscendAttentionBackendImpl.update_graph_params``
        (replay cascade re-parameterization)
    """
    global _installed
    if _installed:
        return True
    with _lock:
        if _installed:
            return True

        # --- ACLGraphWrapper: cascade variant table + replay-time routing ---
        from vllm_ascend.compilation.acl_graph import ACLGraphWrapper

        if not getattr(ACLGraphWrapper, "_cascade_graph_patched", False):
            _wrap_aclgraph_wrapper(ACLGraphWrapper)
            ACLGraphWrapper._cascade_graph_patched = True

        # --- builder: capture-window metadata translation -------------------
        orig_build = builder_cls.build  # eager wrapper (sets shared_len)
        if not getattr(builder_cls, "_cascade_graph_build_stored", False):
            builder_cls._cascade_orig_build = orig_build
            builder_cls._cascade_graph_build_stored = True
        _wrap_build_for_capture(builder_cls)

        # --- impl: capture-window forward dispatch --------------------------
        orig_forward = impl_cls.forward_fused_infer_attention

        def forward_fused_infer_attention(
            self, query, key, value, attn_metadata, output, kv_cache=None
        ):
            from vllm_ascend_split_batch import cascade_graph_plugin as gp
            from vllm_ascend_split_batch.cascade_runner_patch import _step_is_cascade

            if gp._capture_ctx.active and getattr(
                attn_mod._EXTRA_CTX, "capturing", False
            ):
                try:
                    # The capture body returns (output, num_tokens) like the
                    # HUST full_graph_fia_cascade; the official caller
                    # convention (forward_impl -> forward) expects the
                    # output TENSOR, so unpack here (the body already wrote
                    # the merge result into output[:num_tokens]).
                    attn_output, num_tokens = _full_graph_fia_cascade(
                        self, query, key, value, attn_metadata, output, kv_cache
                    )
                    output[:num_tokens] = attn_output[:num_tokens]
                    return output
                except Exception:
                    # Capture-time failure falls back to the standard graph
                    # capture body so the twin graph still matches a
                    # non-cascade step (fail-safe, mirrors HUST).
                    logger.exception(
                        "cascade graph capture failed; capturing the "
                        "standard graph instead"
                    )
            if (
                _step_is_cascade()
                and not getattr(attn_mod._EXTRA_CTX, "capturing", False)
                and getattr(envs_mod, "VLLM_ASCEND_ENABLE_CASCADE_DECODE", False)
                and attn_metadata.attn_state
                == attn_mod.AscendAttentionState.DecodeOnly
                and getattr(attn_metadata, "cascade_shared_len", 0) > 0
                and not getattr(self, "enable_hamming_sparse", False)
                and getattr(self, "sliding_window", None) is None
            ):
                # Runtime cascade step that reached the impl outside graph
                # replay (e.g. graph dispatch failed): run the eager
                # two-stage body instead of the standard full-KV FIA.
                # _forward_cascade_decode returns True when it filled
                # `output`, False to fail open to the standard path.
                try:
                    if self._forward_cascade_decode(
                        query, key, value, attn_metadata, output, kv_cache
                    ):
                        return output
                except Exception:
                    logger.exception(
                        "cascade eager fallback failed; standard FIA path used"
                    )
            return orig_forward(
                self, query, key, value, attn_metadata, output, kv_cache
            )

        # The eager plugin wrapped this method first; chain on top of it.
        impl_cls.forward_fused_infer_attention = forward_fused_infer_attention
        impl_cls._cascade_orig_forward_capture = orig_forward

        # --- impl: replay re-parameterization -------------------------------
        orig_update = impl_cls.update_graph_params

        def update_graph_params(
            update_stream,
            forward_context,
            num_tokens,
            vllm_config,
            speculative_config=None,
            num_dcp_pcp_tokens=None,
            draft_attn_metadatas=None,
        ):
            from vllm_ascend.compilation.acl_graph import get_graph_params

            from vllm_ascend_split_batch.cascade_runner_patch import _step_is_cascade

            if _step_is_cascade():
                graph_params = get_graph_params()
                cascade_key = ("cascade", num_tokens)
                if graph_params is not None and graph_params.attn_params.get(
                    cascade_key
                ):
                    _trace("replay update: cascade key hit num_tokens=%s", num_tokens)
                    _update_cascade_graph_params(
                        update_stream, forward_context, graph_params, num_tokens
                    )
                    return
            return orig_update(
                update_stream,
                forward_context,
                num_tokens,
                vllm_config,
                speculative_config,
                num_dcp_pcp_tokens,
                draft_attn_metadatas,
            )

        # update_graph_params is a @staticmethod on the impl class; assign
        # the plain function wrapped in staticmethod (matches the original's
        # usage through update_full_graph_params -> get_impl_cls()).
        impl_cls.update_graph_params = staticmethod(update_graph_params)

        _installed = True
        return True