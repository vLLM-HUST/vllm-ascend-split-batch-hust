# Copyright (c) 2025-2026 Huawei Technologies Co., Ltd. All Rights Reserved.
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

"""Fail-open tests for the ascend_kernel wheel soft dependency.

Contract under test: a missing or registration-defective kernel wheel must

1. never raise out of ``load()`` (host import chain stays healthy);
2. disable the cascade feature AS A WHOLE (gate stays closed even with
   ``VLLM_ASCEND_ENABLE_CASCADE_DECODE=1``);
3. log exactly ONE clear warning per process;
4. keep the default-off semantics byte-identical (no extra noise when the
   feature is not enabled, apart from the single wheel warning).

The import/registration failure paths are simulated by patching the
``_import_kernel_module`` hook and the ``torch.ops.npu`` namespace; the
subprocess test shadows the real wheel through PYTHONPATH (the "wheel was
never installed" equivalent).
"""

import logging
import os
import subprocess
import sys
import types

import pytest
import torch

cascade_plugin = pytest.importorskip(
    "vllm_ascend_split_batch.cascade_plugin",
    reason="fail-open tests run on the NPU host with the full vllm-ascend "
    "stack installed",
)

ENV_KEYS = (
    "VLLM_ASCEND_ENABLE_CASCADE_DECODE",
    "VLLM_ASCEND_CASCADE_MIN_PREFIX",
    "VLLM_ASCEND_CASCADE_MIN_REQS",
    "VLLM_ASCEND_CASCADE_STRICT",
    "VLLM_ASCEND_CASCADE_PRECISION",
)

WARNING_FRAGMENT = "ascend_kernel wheel unavailable"

_QUALIFYING_KWARGS = dict(
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


@pytest.fixture(autouse=True)
def _healthy_state_after_test(monkeypatch):
    """Leave a re-probed (healthy) plugin state behind for later modules.

    The re-probe needs the master gate set: disabled discovery returns before
    the wheel probe (see ``cascade_plugin.load``).
    """
    cascade_plugin._reset_fail_open_for_tests()
    yield
    cascade_plugin._reset_fail_open_for_tests()
    monkeypatch.setenv("VLLM_ASCEND_ENABLE_CASCADE_DECODE", "1")
    cascade_plugin.load()  # re-probes with the real import; restores globals


def _reset_env(monkeypatch):
    for key in ENV_KEYS:
        monkeypatch.delenv(key, raising=False)


def _poison_import(monkeypatch, exc=None):
    """Make the kernel-wheel import fail (missing wheel simulation)."""
    if exc is None:
        exc = ModuleNotFoundError("No module named 'ascend_kernel'")

    def _boom():
        raise exc

    monkeypatch.setattr(cascade_plugin, "_import_kernel_module", _boom)


def _gate():
    import vllm_ascend.attention.attention_v1 as attn_mod

    return attn_mod.AscendAttentionMetadataBuilder.use_cascade_attention


def _wheel_warnings(caplog):
    return [r for r in caplog.records if WARNING_FRAGMENT in r.getMessage()]


# ----------------------------------------------------------------- probe()


def test_probe_reports_missing_wheel(monkeypatch) -> None:
    _poison_import(monkeypatch)
    ok, reason = cascade_plugin._probe_kernel_wheel()
    assert ok is False
    assert "import failed" in reason
    assert "ascend_kernel" in reason


def test_probe_reports_unregistered_ops(monkeypatch) -> None:
    fake_module = types.ModuleType("ascend_kernel")
    monkeypatch.setattr(cascade_plugin, "_import_kernel_module", lambda: fake_module)
    # Wheel "installed" but torch.ops registration failed: no cascade ops.
    monkeypatch.setattr(torch.ops, "npu", types.SimpleNamespace())
    ok, reason = cascade_plugin._probe_kernel_wheel()
    assert ok is False
    assert "unregistered" in reason
    assert "fa_fp32_stage1" in reason
    assert "lse_merge" in reason


def test_probe_succeeds_on_healthy_host() -> None:
    ok, reason = cascade_plugin._probe_kernel_wheel()
    assert ok is True, reason
    assert reason == ""


# ----------------------------------------------------------------- load()


def test_load_with_missing_wheel_warns_once_and_disables(monkeypatch, caplog) -> None:
    _reset_env(monkeypatch)
    _poison_import(monkeypatch)
    monkeypatch.setenv("VLLM_ASCEND_ENABLE_CASCADE_DECODE", "1")
    with caplog.at_level(logging.WARNING):
        cascade_plugin.load()
        cascade_plugin.load()  # idempotent load must not stack warnings
    assert cascade_plugin._KERNEL_WHEEL_OK is False
    assert cascade_plugin._HAS_LSE_MERGE_OP is False
    assert cascade_plugin._HAS_FA_FP32_STAGE1_OP is False
    assert len(_wheel_warnings(caplog)) == 1


def test_load_reports_registration_failure_reason(monkeypatch, caplog) -> None:
    _reset_env(monkeypatch)
    fake_module = types.ModuleType("ascend_kernel")
    monkeypatch.setattr(cascade_plugin, "_import_kernel_module", lambda: fake_module)
    monkeypatch.setattr(torch.ops, "npu", types.SimpleNamespace())
    monkeypatch.setenv("VLLM_ASCEND_ENABLE_CASCADE_DECODE", "1")
    with caplog.at_level(logging.WARNING):
        cascade_plugin.load()
    assert cascade_plugin._KERNEL_WHEEL_OK is False
    warnings = _wheel_warnings(caplog)
    assert len(warnings) == 1
    assert "unregistered" in warnings[0].getMessage()


def test_disabled_load_does_not_probe_the_wheel(monkeypatch, caplog) -> None:
    """Disabled discovery must not import the kernel wheel (no warning)."""
    _reset_env(monkeypatch)
    _poison_import(monkeypatch)
    with caplog.at_level(logging.WARNING):
        cascade_plugin.load()
    assert _wheel_warnings(caplog) == []
    assert cascade_plugin._KERNEL_WHEEL_PROBED is False
    assert cascade_plugin._KERNEL_WHEEL_OK is False


# ----------------------------------------------------------------- gate


def test_gate_closed_when_wheel_missing_despite_enable(monkeypatch) -> None:
    _reset_env(monkeypatch)
    _poison_import(monkeypatch)
    monkeypatch.setenv("VLLM_ASCEND_ENABLE_CASCADE_DECODE", "1")
    cascade_plugin.load()
    gate = _gate()
    # Qualifying batch (prefix/batch above both thresholds) but no wheel:
    # the whole cascade feature must stay disabled.
    assert gate(None, **_QUALIFYING_KWARGS) is False


def test_gate_default_off_stays_silent_shape_without_wheel(monkeypatch, caplog) -> None:
    _reset_env(monkeypatch)
    _poison_import(monkeypatch)
    monkeypatch.setenv("VLLM_ASCEND_ENABLE_CASCADE_DECODE", "1")
    with caplog.at_level(logging.WARNING):
        cascade_plugin.load()
    before = len(_wheel_warnings(caplog))
    assert before == 1
    monkeypatch.delenv("VLLM_ASCEND_ENABLE_CASCADE_DECODE", raising=False)
    gate = _gate()
    assert gate(None, **_QUALIFYING_KWARGS) is False
    # Default-off: the disabled wheel warning is NOT repeated per step.
    assert len(_wheel_warnings(caplog)) == before


def test_gate_still_opens_with_wheel_present(monkeypatch) -> None:
    _reset_env(monkeypatch)
    monkeypatch.setenv("VLLM_ASCEND_ENABLE_CASCADE_DECODE", "1")
    monkeypatch.setenv("VLLM_ASCEND_CASCADE_PRECISION", "fp32")
    cascade_plugin.load()  # real wheel installed on this host
    gate = _gate()
    assert cascade_plugin._KERNEL_WHEEL_OK is True
    assert gate(None, **_QUALIFYING_KWARGS) is True


def test_wheel_warning_logged_once_across_paths(monkeypatch, caplog) -> None:
    _reset_env(monkeypatch)
    _poison_import(monkeypatch)
    monkeypatch.setenv("VLLM_ASCEND_ENABLE_CASCADE_DECODE", "1")
    with caplog.at_level(logging.WARNING):
        cascade_plugin.load()
        assert _gate()(None, **_QUALIFYING_KWARGS) is False
        assert _gate()(None, **_QUALIFYING_KWARGS) is False
    # load() + two gate calls -> still exactly one process-level warning.
    assert len(_wheel_warnings(caplog)) == 1


# ------------------------------------------------- graph runner patch side


def test_runner_wheel_gate_blocks_capture_scheduling(monkeypatch, caplog) -> None:
    runner_patch = pytest.importorskip("vllm_ascend_split_batch.cascade_runner_patch")
    monkeypatch.setattr(cascade_plugin, "_KERNEL_WHEEL_OK", False)
    cascade_plugin._reset_fail_open_for_tests()
    with caplog.at_level(logging.WARNING):
        assert runner_patch._cascade_wheel_ready() is False
        assert runner_patch._cascade_wheel_ready() is False
    assert len(_wheel_warnings(caplog)) == 1

    monkeypatch.setattr(cascade_plugin, "_KERNEL_WHEEL_OK", True)
    assert runner_patch._cascade_wheel_ready() is True


def test_spec_decode_skips_the_twin_capture(monkeypatch) -> None:
    """2026-09-26: MTP / other speculative configs keep the standard graph.

    The cascade twin admits one query row per request only, so capture
    scheduling must skip those runners (mirrors the dispatch gate guard).
    """
    runner_patch = pytest.importorskip("vllm_ascend_split_batch.cascade_runner_patch")

    def _runner(spec_config):
        return types.SimpleNamespace(
            vllm_config=types.SimpleNamespace(speculative_config=spec_config)
        )

    assert runner_patch._spec_decode_active(_runner(None)) is False
    assert runner_patch._spec_decode_active(_runner(object())) is True
    # Missing config objects must not raise (host seam discipline).
    assert runner_patch._spec_decode_active(types.SimpleNamespace()) is False


# ------------------------------------------------- gate bench child process


def test_gate_self_child_fails_open_without_wheel(tmp_path) -> None:
    """Simulated missing wheel: the bench child exits cleanly, gate neutral.

    A poison ``ascend_kernel.py`` on PYTHONPATH shadows the installed wheel
    and raises at import time -- the "wheel absent / broken install" scenario
    exercised against the real subprocess entry point.
    """
    poison = tmp_path / "poison"
    poison.mkdir()
    (poison / "ascend_kernel.py").write_text(
        "raise ModuleNotFoundError(\"No module named 'ascend_kernel'\")\n",
        encoding="utf-8",
    )
    env = dict(os.environ)
    env["PYTHONPATH"] = os.pathsep.join(
        [str(poison), env.get("PYTHONPATH", "")]
    ).rstrip(os.pathsep)
    proc = subprocess.run(
        [
            sys.executable,
            "-m",
            "vllm_ascend_split_batch.cascade_gate_self",
            "spec.json",
            "out.json",
        ],
        capture_output=True,
        text=True,
        env=env,
        timeout=300,
    )
    assert proc.returncode == 3, proc.stderr[-500:]
    assert WARNING_FRAGMENT in proc.stdout + proc.stderr
    assert "gate stays neutral" in proc.stdout + proc.stderr
