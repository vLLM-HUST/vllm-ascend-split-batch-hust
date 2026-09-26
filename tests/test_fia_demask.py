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

"""CPU tests for the decode FIA de-mask plugin (default-off, guarded).

Covers: the pure decode-domain planner, the two-state patch (off == stock /
on == rewritten) at the op surface, the signature-drift fail-closed path and
the per-call fail-open path.  No NPU or vllm-ascend import is required.
"""

import logging
import subprocess
import sys
import types

import pytest

from vllm_ascend_split_batch import fia_demask_plugin as plug

INT_MAX = 2147483647


# ------------------------------------------------------------------ fakes


class _Dtype:
    def __str__(self):
        return self._name

    def __init__(self, name):
        self._name = name


class _Tensor:
    def __init__(self, shape, dtype):
        self.shape = tuple(shape)
        self.dtype = _Dtype(dtype)

    def tolist(self):
        raise AssertionError("only called for actual_seq_lengths")


def _mask(side=2048, dtype="torch.int8"):
    return _Tensor((side, side), dtype)


def _production_decode_kwargs(batch=32, dtype="torch.int8", **overrides):
    """The exact decode signature observed in production (see the design note)."""
    kwargs = {
        "query": _Tensor((batch, 40, 128), "torch.bfloat16"),
        "key": _Tensor((13, 128, 1024), "torch.bfloat16"),
        "value": _Tensor((13, 128, 1024), "torch.bfloat16"),
        "atten_mask": _mask(2048, dtype),
        "block_table": _Tensor((batch, 13), "torch.int32"),
        "input_layout": "TND",
        "block_size": 128,
        "actual_seq_lengths": list(range(1, batch + 1)),
        "actual_seq_lengths_kv": [1600] * batch,
        "num_key_value_heads": 8,
        "num_heads": 40,
        "scale": 128**-0.5,
        "sparse_mode": 3,
        "pre_tokens": INT_MAX,
        "next_tokens": INT_MAX,
    }
    kwargs.update(overrides)
    return kwargs


class _RecordingOp:
    """Stands in for ``torch_npu.npu_fused_infer_attention_score``."""

    def __init__(self):
        self.calls = []
        self.out_calls = []

    def __call__(self, *args, **kwargs):
        self.calls.append(kwargs)
        return "direct"

    def out(self, *args, **kwargs):
        self.out_calls.append(kwargs)
        return "out"


class _RecordingWorkspace:
    def __init__(self):
        self.calls = []

    def __call__(self, *args, **kwargs):
        self.calls.append(kwargs)
        return "workspace"


@pytest.fixture(autouse=True)
def _clean_state(monkeypatch):
    monkeypatch.delenv(plug.ENV_ENABLE, raising=False)
    monkeypatch.delenv(plug.ENV_TRACE, raising=False)
    plug._reset_for_tests()
    yield
    plug._reset_for_tests()
    plug._orig.clear()
    plug._installed = False
    plug._mode["installed"] = False


def _install_fake_torch_npu(monkeypatch, op=None, workspace=None):
    """Install a fake ``torch_npu`` module the plugin will patch."""
    op = op if op is not None else _RecordingOp()
    workspace = workspace if workspace is not None else _RecordingWorkspace()
    fake = types.ModuleType("torch_npu")
    fake.npu_fused_infer_attention_score = op
    fake._npu_fused_infer_attention_score_get_max_workspace = workspace
    monkeypatch.setitem(sys.modules, "torch_npu", fake)
    return fake, op, workspace


# ------------------------------------------------------- planner: positive


def test_plan_applies_to_the_production_decode_signature():
    kwargs = _production_decode_kwargs()
    new, reason = plug.plan_transform(kwargs)
    assert reason == "applied"
    assert new is not None
    assert new["atten_mask"] is None
    assert new["sparse_mode"] == 0
    # everything else is byte-identical, and the input dict is untouched.
    assert kwargs["atten_mask"] is not None
    assert kwargs["sparse_mode"] == 3
    for key, value in kwargs.items():
        if key in {"atten_mask", "sparse_mode"}:
            continue
        assert new[key] is value


def test_plan_applies_to_a_single_request_decode():
    new, reason = plug.plan_transform(_production_decode_kwargs(batch=1))
    assert reason == "applied" and new["sparse_mode"] == 0


# ------------------------------------------------------- planner: negatives


def _q_layout(seq_lengths, total):
    """Override the query side of the production signature."""
    return {
        "actual_seq_lengths": seq_lengths,
        "query": _Tensor((total, 40, 128), "torch.bfloat16"),
    }


@pytest.mark.parametrize(
    ("overrides", "reason"),
    [
        # prefill (chunked / prompt): >1 query token per request keeps the mask
        (_q_layout([5, 10], 10), "not-decode"),
        # mixed decode+prefill batch: also >1 token for some request
        (_q_layout([1, 2, 7], 7), "not-decode"),
        # padded query (T > len(actual_seq_lengths)) is not pure decode
        (_q_layout([1, 2], 4), "not-decode"),
        # speculative decode: Q_S = k > 1
        (_q_layout([3], 3), "not-decode"),
        ({"input_layout": "BNSD"}, "layout"),
        ({"input_layout": "BSH"}, "layout"),
        # sliding window (sparse_mode=4 / pre_tokens window) must never be touched
        ({"sparse_mode": 4, "pre_tokens": 4096, "next_tokens": 0}, "sparse-mode"),
        ({"pre_tokens": 4096}, "pre-tokens"),
        ({"next_tokens": 0}, "next-tokens"),
        ({"sparse_mode": 0}, "sparse-mode"),
        ({"atten_mask": None}, "no-mask"),
        ({"atten_mask": _mask(2048, "torch.bfloat16")}, "mask-dtype"),
        ({"atten_mask": _Tensor((2048, 1024), "torch.int8")}, "mask-nonsquare"),
        ({"atten_mask": _Tensor((2048,), "torch.int8")}, "mask-ndim"),
        ({"actual_seq_lengths": None}, "not-decode"),
        ({"query": None}, "not-decode"),
        ({"actual_seq_lengths": []}, "not-decode"),
    ],
)
def test_plan_declines_outside_the_decode_domain(overrides, reason):
    new, observed = plug.plan_transform(_production_decode_kwargs(**overrides))
    assert new is None
    assert observed == reason


def test_plan_never_raises_on_garbage():
    for kwargs in (
        {},
        {"query": object(), "actual_seq_lengths": object()},
        {"sparse_mode": "three"},
        {"query": _Tensor((1, 2), "torch.bfloat16"), "actual_seq_lengths": [1]},
    ):
        new, reason = plug.plan_transform(kwargs)
        assert new is None
        assert isinstance(reason, str)


# ------------------------------------------------ op surface: two-state patch


def test_default_off_does_not_patch_or_log(monkeypatch, caplog):
    fake, op, _ = _install_fake_torch_npu(monkeypatch)
    with caplog.at_level(logging.DEBUG):
        assert plug.load() is False
    assert fake.npu_fused_infer_attention_score is op  # untouched
    assert caplog.records == []


def test_enabled_rewrites_only_the_decode_call(monkeypatch):
    fake, op, workspace = _install_fake_torch_npu(monkeypatch)
    monkeypatch.setenv(plug.ENV_ENABLE, "1")
    assert plug.load() is True

    patched = fake.npu_fused_infer_attention_score
    assert patched is not op
    assert patched.out.__self__ is patched  # .out is the proxy's own overload

    decode = _production_decode_kwargs()
    prefill = _production_decode_kwargs(**_q_layout([5, 10], 10))
    assert patched.out(**decode) == "out"
    assert patched.out(**prefill) == "out"

    assert len(op.out_calls) == 2
    # decode: mask dropped, sparse_mode flipped
    assert op.out_calls[0]["atten_mask"] is None
    assert op.out_calls[0]["sparse_mode"] == 0
    # prefill: byte-identical to what the host passed
    assert op.out_calls[1]["atten_mask"] is not None
    assert op.out_calls[1]["sparse_mode"] == 3
    stats = plug.stats()
    assert stats["applied"] == 1
    assert stats["reasons"] == {"applied": 1, "not-decode": 1}


def test_enabled_keeps_the_workspace_query_consistent(monkeypatch):
    fake, op, workspace = _install_fake_torch_npu(monkeypatch)
    monkeypatch.setenv(plug.ENV_ENABLE, "1")
    assert plug.load() is True

    patched_ws = fake._npu_fused_infer_attention_score_get_max_workspace
    assert patched_ws is not workspace

    decode = _production_decode_kwargs()
    assert patched_ws(**decode) == "workspace"
    assert workspace.calls[0]["atten_mask"] is None
    assert workspace.calls[0]["sparse_mode"] == 0

    prefill = _production_decode_kwargs(**_q_layout([5, 10], 10))
    assert patched_ws(**prefill) == "workspace"
    assert workspace.calls[1]["atten_mask"] is not None
    assert workspace.calls[1]["sparse_mode"] == 3


def test_proxy_forwards_attributes_to_the_wrapped_op(monkeypatch):
    fake, op, _ = _install_fake_torch_npu(monkeypatch)
    op.op = "packet-marker"
    op.overloads = ("out", "default")
    monkeypatch.setenv(plug.ENV_ENABLE, "1")
    assert plug.load() is True
    proxy = fake.npu_fused_infer_attention_score
    assert proxy.op == "packet-marker"
    assert proxy.overloads == ("out", "default")
    # the proxy is not the packet, but it is indistinguishable for callers
    assert proxy is not op


def test_direct_call_overload_is_also_guarded(monkeypatch):
    fake, op, _ = _install_fake_torch_npu(monkeypatch)
    monkeypatch.setenv(plug.ENV_ENABLE, "1")
    assert plug.load() is True
    direct = fake.npu_fused_infer_attention_score(**_production_decode_kwargs())
    assert direct == "direct"
    assert op.calls[0]["atten_mask"] is None
    assert op.calls[0]["sparse_mode"] == 0


def test_install_is_idempotent(monkeypatch):
    fake, op, _ = _install_fake_torch_npu(monkeypatch)
    monkeypatch.setenv(plug.ENV_ENABLE, "1")
    assert plug.load() is True
    first = fake.npu_fused_infer_attention_score
    assert plug.load() is True
    assert fake.npu_fused_infer_attention_score is first


def test_uninstall_restores_the_stock_surface(monkeypatch):
    fake, op, workspace = _install_fake_torch_npu(monkeypatch)
    monkeypatch.setenv(plug.ENV_ENABLE, "1")
    assert plug.load() is True
    plug.uninstall()
    assert fake.npu_fused_infer_attention_score is op
    assert fake._npu_fused_infer_attention_score_get_max_workspace is workspace
    assert plug.load() is True  # re-installable after uninstall
    assert fake.npu_fused_infer_attention_score is not op


# ------------------------------------------------------------- observe mode


def test_trace_only_observes_and_never_rewrites(monkeypatch, caplog):
    fake, op, _ = _install_fake_torch_npu(monkeypatch)
    monkeypatch.setenv(plug.ENV_TRACE, "1")
    with caplog.at_level(logging.INFO):
        assert plug.load() is True
        fake.npu_fused_infer_attention_score.out(**_production_decode_kwargs())
    # the OFF arm still sees the real (masked) call ...
    assert op.out_calls[0]["atten_mask"] is not None
    assert op.out_calls[0]["sparse_mode"] == 3
    assert plug.stats()["applied"] == 0
    # ... and the evidence line names the mask + sparse_mode
    messages = [r.getMessage() for r in caplog.records]
    assert any(
        "decode FIA call carries the redundant mask" in m
        and "sparse_mode=3" in m
        and "VLLM_HUST_FIA_DEMASK is OFF" in m
        for m in messages
    )


def test_enabled_logs_the_demask_evidence_once(monkeypatch, caplog):
    fake, _, _ = _install_fake_torch_npu(monkeypatch)
    monkeypatch.setenv(plug.ENV_ENABLE, "1")
    with caplog.at_level(logging.INFO):
        assert plug.load() is True
        for _ in range(3):
            fake.npu_fused_infer_attention_score.out(**_production_decode_kwargs())
    applied = [
        r for r in caplog.records
        if "FIA decode de-mask applied" in r.getMessage()
    ]
    assert len(applied) == 1
    assert plug.stats()["applied"] == 3


# ------------------------------------------------- signature drift / fail-open


def test_missing_out_overload_fails_closed(monkeypatch, caplog):
    class _NoOut:
        def __call__(self, *args, **kwargs):  # pragma: no cover
            return None

    fake, _, _ = _install_fake_torch_npu(monkeypatch, op=_NoOut())
    monkeypatch.setenv(plug.ENV_ENABLE, "1")
    with caplog.at_level(logging.WARNING):
        assert plug.load() is False
    assert isinstance(fake.npu_fused_infer_attention_score, _NoOut)
    assert any(
        "refused" in r.getMessage() for r in caplog.records
    )


def test_missing_symbol_fails_closed(monkeypatch, caplog):
    fake = types.ModuleType("torch_npu")
    monkeypatch.setitem(sys.modules, "torch_npu", fake)
    monkeypatch.setenv(plug.ENV_ENABLE, "1")
    with caplog.at_level(logging.WARNING):
        assert plug.load() is False
    assert any("refused" in r.getMessage() for r in caplog.records)


def test_workspace_symbol_missing_still_installs_the_op(monkeypatch, caplog):
    fake = types.ModuleType("torch_npu")
    op = _RecordingOp()
    fake.npu_fused_infer_attention_score = op
    monkeypatch.setitem(sys.modules, "torch_npu", fake)
    monkeypatch.setenv(plug.ENV_ENABLE, "1")
    with caplog.at_level(logging.WARNING):
        assert plug.load() is True
    assert fake.npu_fused_infer_attention_score is not op
    assert any(
        "_npu_fused_infer_attention_score_get_max_workspace" in r.getMessage()
        for r in caplog.records
    )


def test_planner_failure_fails_open(monkeypatch, caplog):
    fake, op, _ = _install_fake_torch_npu(monkeypatch)
    monkeypatch.setenv(plug.ENV_ENABLE, "1")
    assert plug.load() is True
    monkeypatch.setattr(
        plug, "_dispatch", lambda kwargs: (_ for _ in ()).throw(RuntimeError("boom"))
    )
    with caplog.at_level(logging.WARNING):
        assert fake.npu_fused_infer_attention_score.out(
            **_production_decode_kwargs()
        ) == "out"
    # the ORIGINAL kwargs reached the op, i.e. stock behaviour
    assert op.out_calls[0]["atten_mask"] is not None
    assert op.out_calls[0]["sparse_mode"] == 3
    assert plug.stats()["errors"] >= 1
    assert any("dispatch failed" in r.getMessage() for r in caplog.records)


def test_raising_planner_is_contained(monkeypatch):
    fake, op, _ = _install_fake_torch_npu(monkeypatch)
    monkeypatch.setenv(plug.ENV_ENABLE, "1")
    assert plug.load() is True
    real_plan = plug.plan_transform

    def _boom(kwargs):
        raise RuntimeError("planner exploded")

    monkeypatch.setattr(plug, "plan_transform", _boom)
    # _dispatch itself is defensive too: no exception may escape the op proxy
    assert fake.npu_fused_infer_attention_score.out(
        **_production_decode_kwargs()
    ) == "out"
    monkeypatch.setattr(plug, "plan_transform", real_plan)
    assert op.out_calls[0]["atten_mask"] is not None


# ------------------------------------------------------- import purity guard


def test_default_off_imports_no_torch_or_torch_npu():
    """Default-off must not pull torch/torch_npu into the process."""
    code = (
        "import sys;"
        "from vllm_ascend_split_batch import fia_demask_plugin as p;"
        "assert p.load() is False;"
        "mods=[m for m in ('torch','torch_npu') if m in sys.modules];"
        "assert mods==[], mods;"
        "print('purity-ok')"
    )
    result = subprocess.run(
        [sys.executable, "-c", code],
        capture_output=True,
        text=True,
        check=False,
    )
    assert result.returncode == 0, result.stderr
    assert "purity-ok" in result.stdout
