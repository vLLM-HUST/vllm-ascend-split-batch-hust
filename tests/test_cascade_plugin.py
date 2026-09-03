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

"""CPU-only unit tests for the cascade decode plugin (default-off semantics).

``load()`` imports the real vllm-ascend stack, which is expected to succeed on
this host.  The tests exercise gate and bookkeeping logic through small
stand-in objects; the NPU kernels themselves are never invoked here.
"""

import types

import pytest

cascade_plugin = pytest.importorskip(
    "vllm_ascend_split_batch.cascade_plugin",
    reason="cascade plugin unit tests run on the NPU host with the full "
    "vllm-ascend stack installed",
)

ENV_KEYS = (
    "VLLM_ASCEND_ENABLE_CASCADE_DECODE",
    "VLLM_ASCEND_CASCADE_MIN_PREFIX",
    "VLLM_ASCEND_CASCADE_MIN_REQS",
    "VLLM_ASCEND_CASCADE_STRICT",
    "VLLM_ASCEND_CASCADE_PRECISION",
)


def _reset_env(monkeypatch):
    """Clear cascade env keys so lazy resolution falls back to defaults."""
    for key in ENV_KEYS:
        monkeypatch.delenv(key, raising=False)


def _gate(monkeypatch):
    """Return the patched (gate, builder_cls, impl_cls) triple."""
    _reset_env(monkeypatch)
    cascade_plugin.load()
    import vllm_ascend.attention.attention_v1 as attn_mod

    return (
        attn_mod.AscendAttentionMetadataBuilder.use_cascade_attention,
        attn_mod.AscendAttentionMetadataBuilder,
        attn_mod.AscendAttentionBackendImpl,
    )


def test_load_is_idempotent_and_sets_patch_markers(monkeypatch) -> None:
    _, builder_cls, impl_cls = _gate(monkeypatch)

    assert builder_cls._cascade_plugin_patched is True
    assert impl_cls._cascade_plugin_patched is True

    # Repeated load() calls must not stack another wrapper layer.
    build_before = builder_cls.build
    forward_before = impl_cls.forward_fused_infer_attention
    cascade_plugin.load()
    cascade_plugin.load()
    assert builder_cls.build is build_before
    assert impl_cls.forward_fused_infer_attention is forward_before


def test_env_vars_are_injected_and_resolve_lazily(monkeypatch) -> None:
    _gate(monkeypatch)
    from vllm_ascend import envs as envs_mod

    for key in ENV_KEYS:
        assert key in envs_mod.env_variables, key

    # Values resolve lazily from os.environ at attribute access time.
    monkeypatch.setenv("VLLM_ASCEND_CASCADE_MIN_PREFIX", "123")
    assert envs_mod.VLLM_ASCEND_CASCADE_MIN_PREFIX == 123
    monkeypatch.setenv("VLLM_ASCEND_CASCADE_MIN_REQS", "7")
    assert envs_mod.VLLM_ASCEND_CASCADE_MIN_REQS == 7
    monkeypatch.setenv("VLLM_ASCEND_ENABLE_CASCADE_DECODE", "1")
    assert envs_mod.VLLM_ASCEND_ENABLE_CASCADE_DECODE is True
    monkeypatch.setenv("VLLM_ASCEND_CASCADE_STRICT", "1")
    assert envs_mod.VLLM_ASCEND_CASCADE_STRICT is True
    monkeypatch.setenv("VLLM_ASCEND_CASCADE_PRECISION", "fp32")
    assert envs_mod.VLLM_ASCEND_CASCADE_PRECISION == "fp32"


def test_gate_default_off(monkeypatch) -> None:
    gate, _, _ = _gate(monkeypatch)
    kwargs = dict(
        common_prefix_len=8192,
        query_lens=[1] * 64,
        num_query_heads=8,
        num_kv_heads=1,
        use_alibi=False,
        use_sliding_window=False,
        use_local_attention=False,
        num_sms=0,
        dcp_world_size=1,
    )
    # Default: VLLM_ASCEND_ENABLE_CASCADE_DECODE unset -> gate stays closed
    # even for batches that would otherwise qualify.
    assert gate(None, **kwargs) is False


def test_gate_stays_closed_below_thresholds(monkeypatch) -> None:
    gate, _, _ = _gate(monkeypatch)
    monkeypatch.setenv("VLLM_ASCEND_ENABLE_CASCADE_DECODE", "1")
    kwargs = dict(
        common_prefix_len=100,
        query_lens=[1] * 4,
        num_query_heads=8,
        num_kv_heads=1,
        use_alibi=False,
        use_sliding_window=False,
        use_local_attention=False,
        num_sms=0,
        dcp_world_size=1,
    )
    # prefix 100 < 8192 and 4 reqs < 32 -> gate must stay closed.
    assert gate(None, **kwargs) is False


def test_strict_overrides_enable(monkeypatch, caplog) -> None:
    gate, _, _ = _gate(monkeypatch)
    monkeypatch.setenv("VLLM_ASCEND_ENABLE_CASCADE_DECODE", "1")
    monkeypatch.setenv("VLLM_ASCEND_CASCADE_STRICT", "1")
    kwargs = dict(
        common_prefix_len=8192,
        query_lens=[1] * 64,
        num_query_heads=8,
        num_kv_heads=1,
        use_alibi=False,
        use_sliding_window=False,
        use_local_attention=False,
        num_sms=0,
        dcp_world_size=1,
    )
    assert gate(None, **kwargs) is False
    assert any("STRICT" in rec.message for rec in caplog.records)


def test_rejects_unsupported_attention_variants(monkeypatch) -> None:
    gate, _, _ = _gate(monkeypatch)
    monkeypatch.setenv("VLLM_ASCEND_ENABLE_CASCADE_DECODE", "1")
    base = dict(
        common_prefix_len=8192,
        query_lens=[1] * 64,
        num_query_heads=8,
        num_kv_heads=1,
        use_alibi=False,
        use_sliding_window=False,
        use_local_attention=False,
        num_sms=0,
        dcp_world_size=1,
    )
    assert gate(None, **{**base, "use_alibi": True}) is False
    assert gate(None, **{**base, "use_sliding_window": True}) is False
    assert gate(None, **{**base, "use_local_attention": True}) is False
    assert gate(None, **{**base, "dcp_world_size": 2}) is False


def test_precision_whitelist(monkeypatch) -> None:
    gate, _, _ = _gate(monkeypatch)
    monkeypatch.setenv("VLLM_ASCEND_ENABLE_CASCADE_DECODE", "1")
    kwargs = dict(
        common_prefix_len=8192,
        query_lens=[1] * 64,
        num_query_heads=8,
        num_kv_heads=1,
        use_alibi=False,
        use_sliding_window=False,
        use_local_attention=False,
        num_sms=0,
        dcp_world_size=1,
    )
    monkeypatch.setenv("VLLM_ASCEND_CASCADE_PRECISION", "int8")
    assert gate(None, **kwargs) is False
    monkeypatch.setenv("VLLM_ASCEND_CASCADE_PRECISION", "fp32")
    # fp32 is accepted at the gate (kernel-capability check happens later).
    assert gate(None, **kwargs) is True


def test_build_wrapper_sets_cascade_shared_len(monkeypatch) -> None:
    _, builder_cls, _ = _gate(monkeypatch)

    class FakeMetadata:
        pass

    def fake_build(self, common_prefix_len, common_attn_metadata, fast_build=False):
        return FakeMetadata()

    wrapper = cascade_plugin._make_build_wrapper(fake_build)
    fake_self = object()
    metadata = wrapper(fake_self, 1152, object())
    assert metadata.cascade_shared_len == 1152

    # A metadata type without the dynamic attribute support must not break
    # the standard build path.
    class SlotsOnly:
        __slots__ = ()

    captured = {}

    def fake_build_slots(
        self, common_prefix_len, common_attn_metadata, fast_build=False
    ):
        captured["prefix"] = common_prefix_len
        return SlotsOnly()

    wrapper_slots = cascade_plugin._make_build_wrapper(fake_build_slots)
    assert wrapper_slots(fake_self, 0, object()) is not None
    assert captured["prefix"] == 0


def test_stage2_kv_lens_helpers(monkeypatch) -> None:
    _gate(monkeypatch)
    assert cascade_plugin._cascade_stage2_kv_lens([100, 60, 1], 3, 48) == [52, 12, 1]
    assert cascade_plugin._cascade_has_short_real_request([100, 60, 1], 3, 48) is False
    assert cascade_plugin._cascade_has_short_real_request([48, 60, 1], 3, 48) is True


def test_env_module_fork_safety(monkeypatch) -> None:
    """env_variables must stay a plain dict of env lambdas (no captured state)."""
    _gate(monkeypatch)
    import copy
    import os

    from vllm_ascend import envs as envs_mod

    snapshot = copy.copy(envs_mod.env_variables)
    try:
        os.environ["VLLM_ASCEND_CASCADE_MIN_PREFIX"] = "4321"
        assert envs_mod.VLLM_ASCEND_CASCADE_MIN_PREFIX == 4321
        del os.environ["VLLM_ASCEND_CASCADE_MIN_PREFIX"]
        assert envs_mod.VLLM_ASCEND_CASCADE_MIN_PREFIX == 8192
    finally:
        envs_mod.env_variables.clear()
        envs_mod.env_variables.update(snapshot)
        os.environ.pop("VLLM_ASCEND_CASCADE_MIN_PREFIX", None)


# ------------------------------------------------------- stage-1 stable skip


def _stage1_update_env(monkeypatch):
    _reset_env(monkeypatch)
    monkeypatch.setenv("VLLM_ASCEND_ENABLE_CASCADE_DECODE", "1")
    monkeypatch.setenv("VLLM_ASCEND_ENABLE_CASCADE_GRAPH", "1")
    from vllm_ascend_split_batch import cascade_graph_plugin as gp

    gp.invalidate_stage1_cache()
    return gp


class _FakeEvent:
    def __init__(self, log):
        self._log = log

    def record(self, stream):
        self._log.append("record")


def _fake_graph_params(block_tables, seq_lens, shared_len):
    """One-layer stand-in for the graph-pool param tables."""
    log = []
    entry = (
        "cascade",          # 0 kind
        "q", "k", "v",      # 1-3 tensors (opaque)
        "out",              # 4
        "o1", "l1",         # 5-6 stage-1 outs
        "o2", "l2",         # 7-8 stage-2 outs
        128,                # 9 block_size
        1,                  # 10 num_kv_heads
        8,                  # 11 num_heads
        0.1,                # 12 scale
        64,                 # 13 num_tokens_cap
        "L0",               # 14 layer_name
    )

    class GP:
        attn_params = {("cascade", 64): [entry]}
        handles = {("cascade", 64): [object(), object()]}
        events = {
            ("cascade", 64): [_FakeEvent(log), _FakeEvent(log)]
        }
        workspaces = {("cascade", 64): ("ws1", "ws2")}

    meta = types.SimpleNamespace(
        cascade_shared_len=shared_len,
        actual_seq_lengths_q=list(range(1, len(seq_lens) + 1)),
        seq_lens_list=seq_lens,
        block_tables=block_tables,
    )
    fwd_ctx = types.SimpleNamespace(attn_metadata={"L0": meta})
    return GP(), fwd_ctx, log


@pytest.fixture()
def _patch_npu_graph_apis(monkeypatch):
    """Neutralize the NPU-side graph task update APIs for CPU tests."""
    import contextlib

    import torch

    calls = {"stage1": 0, "stage2": 0}

    def fake_out(**kwargs):
        calls["stage1" if kwargs.get("workspace") == "ws1" else "stage2"] += 1

    fake_fia = types.SimpleNamespace(out=fake_out)
    monkeypatch.setattr(
        "torch_npu.npu_fused_infer_attention_score_v2", fake_fia, raising=False
    )
    monkeypatch.setattr(
        "vllm_ascend_split_batch.cascade_graph_plugin.torch_npu",
        types.SimpleNamespace(npu_fused_infer_attention_score_v2=fake_fia),
        raising=False,
    )
    monkeypatch.setattr(
        torch.npu, "stream", lambda s: contextlib.nullcontext(), raising=False
    )
    monkeypatch.setattr(
        torch.npu, "graph_task_update_begin", lambda *a, **k: None, raising=False
    )
    monkeypatch.setattr(
        torch.npu, "graph_task_update_end", lambda *a, **k: None, raising=False
    )
    return calls


def test_skip_stable_env_is_injected_and_defaults_on(monkeypatch) -> None:
    _gate(monkeypatch)
    from vllm_ascend import envs as envs_mod

    assert "VLLM_ASCEND_CASCADE_UPDATE_SKIP_STABLE" in envs_mod.env_variables
    assert envs_mod.VLLM_ASCEND_CASCADE_UPDATE_SKIP_STABLE is True
    monkeypatch.setenv("VLLM_ASCEND_CASCADE_UPDATE_SKIP_STABLE", "0")
    assert envs_mod.VLLM_ASCEND_CASCADE_UPDATE_SKIP_STABLE is False


def test_stage1_signature_tracks_all_dynamic_inputs(monkeypatch) -> None:
    gp = _stage1_update_env(monkeypatch)
    import torch

    bt = torch.zeros((4, 8), dtype=torch.int32)
    sig = gp._stage1_signature(bt[:1, :2].contiguous(), 256, 4)
    assert sig is not None
    # Same content -> equal signature.
    assert gp._stage1_signature(bt[:1, :2].contiguous(), 256, 4) == sig
    # Any dynamic input change must change the signature.
    bt2 = bt.clone()
    bt2[0, 0] = 7
    assert gp._stage1_signature(bt2[:1, :2].contiguous(), 256, 4) != sig
    assert gp._stage1_signature(bt[:1, :3].contiguous(), 256, 4) != sig
    assert gp._stage1_signature(bt[:1, :2].contiguous(), 384, 4) != sig
    assert gp._stage1_signature(bt[:1, :2].contiguous(), 256, 8) != sig


def test_update_skips_stage1_rebind_when_stable(
    monkeypatch, _patch_npu_graph_apis
) -> None:
    import torch

    gp = _stage1_update_env(monkeypatch)
    calls = _patch_npu_graph_apis
    bt = torch.zeros((4, 8), dtype=torch.int32)
    graph_params, fwd_ctx, log = _fake_graph_params(
        bt, [300, 260, 260, 260], 256
    )

    # Step 1: cold cache -> full re-bind (1 stage-1 + 1 stage-2 per layer),
    # both in-graph events recorded.
    gp._update_cascade_graph_params(None, fwd_ctx, graph_params, 64)
    assert calls == {"stage1": 1, "stage2": 1}
    assert log.count("record") == 2

    # Step 2: identical shared region -> stage-1 re-bind skipped, stage-2
    # still re-bound, events still recorded (replay waits on them).
    gp._update_cascade_graph_params(None, fwd_ctx, graph_params, 64)
    assert calls == {"stage1": 1, "stage2": 2}
    assert log.count("record") == 4

    # Step 3: shared-prefix block ids changed -> full re-bind again.
    bt[0, 0] = 42
    gp._update_cascade_graph_params(None, fwd_ctx, graph_params, 64)
    assert calls == {"stage1": 2, "stage2": 3}

    # invalidate_stage1_cache (post-capture) forces a re-bind.
    bt[0, 0] = 0
    gp.invalidate_stage1_cache()
    gp._update_cascade_graph_params(None, fwd_ctx, graph_params, 64)
    assert calls == {"stage1": 3, "stage2": 4}


def test_update_never_skips_when_knob_off(
    monkeypatch, _patch_npu_graph_apis
) -> None:
    import torch

    gp = _stage1_update_env(monkeypatch)
    monkeypatch.setenv("VLLM_ASCEND_CASCADE_UPDATE_SKIP_STABLE", "0")
    calls = _patch_npu_graph_apis
    bt = torch.zeros((4, 8), dtype=torch.int32)
    graph_params, fwd_ctx, log = _fake_graph_params(
        bt, [300, 260, 260, 260], 256
    )

    gp._update_cascade_graph_params(None, fwd_ctx, graph_params, 64)
    gp._update_cascade_graph_params(None, fwd_ctx, graph_params, 64)
    assert calls == {"stage1": 2, "stage2": 2}
    assert log.count("record") == 4
