# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 vLLM-HUST contributors.

import json
from types import SimpleNamespace

import pytest

torch = pytest.importorskip("torch")

from vllm_ascend_split_batch.gdn_qkv_norm_diagnostic import (  # noqa: E402
    NormDiagnostic,
    attention_class_for_model,
    tensor_difference,
)


def values():
    return tuple(torch.zeros(1, 16, h, 128, dtype=torch.bfloat16) for h in (8, 8, 16))


def observe(probe, name, different=False):
    probe.current_layer = SimpleNamespace(prefix=name)
    actual, expected = values(), values()
    if different:
        actual[0][0, 0, 0, 1] = 0.5
    x = torch.zeros(16, 4096, dtype=torch.bfloat16)
    probe.observe(x, actual, expected)
    return actual


def test_difference_preserves_nonfinite_and_signed_zero():
    a = torch.tensor([0.0, float("nan"), float("inf")], dtype=torch.bfloat16)
    b = a.clone()
    b[0] = -0.0
    result = tensor_difference(a, b)
    assert result["different"] == 0
    assert not result["bitwise_equal"]
    b[2] = -float("inf")
    assert tensor_difference(a, b)["first_index"] == [2]


def test_first_active_layer_and_immutable_capture(tmp_path):
    probe = NormDiagnostic(tmp_path, rank=1)
    observe(probe, "stale", different=True)
    probe.begin_step()
    observe(probe, "layer0")
    observe(probe, "layer1", different=True)
    observe(probe, "layer2", different=True)
    probe.end_step({"req": 16})
    report = json.loads((tmp_path / "rank1-summary.json").read_text())
    assert report["layer"] == "layer1"
    assert report["active_layers"] == ["layer0", "layer1", "layer2"]
    assert report["counts"] == [1, 0, 0]
    assert report["differences"]["q"]["first_index"] == [0, 0, 0, 1]
    payload = torch.load(tmp_path / "rank1-first.pt", weights_only=True)
    assert payload["input"].shape == (16, 4096)
    assert payload["actual"][0][0, 0, 0, 1] == 0.5
    before = (tmp_path / "rank1-summary.json").read_bytes()
    probe.end_step({})
    assert (tmp_path / "rank1-summary.json").read_bytes() == before
    second = NormDiagnostic(tmp_path, rank=1)
    observe(second, "layer1", different=True)
    with pytest.raises(FileExistsError):
        second.end_step({})


@pytest.mark.parametrize(
    "covered,status", [(True, "NO_DIVERGENCE"), (False, "NO_COVERAGE")]
)
def test_bounded_no_divergence_is_not_acceptance(tmp_path, covered, status):
    probe = NormDiagnostic(tmp_path, rank=0, max_steps=1)
    if covered:
        observe(probe, "layer0")
    probe.end_step({})
    report = json.loads((tmp_path / "rank0-summary.json").read_text())
    assert report["status"] == status
    assert "not L3 PASS" in report["scope"]
    assert not (tmp_path / "rank0-first.pt").exists()


def test_install_returns_original_results_and_restores_context(tmp_path):
    result = values()
    api = SimpleNamespace(unpack_norm=lambda x: result)

    class Attention:
        prefix = "layer"

        def rearrange_mixed_qkv(self, x):
            return values()

        def _forward_core(self, x, fail=False):
            out = api.unpack_norm(x)
            if fail:
                raise ValueError("injected")
            return out

    probe = NormDiagnostic(tmp_path, rank=0)
    probe.install(api, Attention, lambda t: t)
    x = torch.zeros(16, 4096, dtype=torch.bfloat16)
    assert Attention()._forward_core(x) is result
    assert probe.current_layer is None
    with pytest.raises(ValueError, match="injected"):
        Attention()._forward_core(x, fail=True)
    assert probe.current_layer is None


def test_invalid_budget_and_missing_context(tmp_path):
    with pytest.raises(ValueError):
        NormDiagnostic(tmp_path, 0, max_steps=0)
    probe = NormDiagnostic(tmp_path, 0)
    with pytest.raises(RuntimeError, match="context"):
        probe.observe(torch.zeros(1), values(), values())


def test_rebound_host_class_not_definition_class(tmp_path):
    api = SimpleNamespace(unpack_norm=lambda x: values())

    class Carrier:
        def _forward_core(self, x):
            return api.unpack_norm(x)

    class Bound:
        prefix = "bound-layer"
        _forward_core = Carrier._forward_core

        def rearrange_mixed_qkv(self, x):
            return values()

    model = SimpleNamespace(modules=lambda: iter([Bound(), Bound()]))
    cls = attention_class_for_model(model, Carrier._forward_core)
    assert cls is Bound
    probe = NormDiagnostic(tmp_path, 0)
    probe.install(api, cls, lambda t: t)
    Bound()._forward_core(torch.zeros(16, 4096, dtype=torch.bfloat16))
    assert list(probe.slots) == ["bound-layer"]
    assert probe.current_layer is None


@pytest.mark.parametrize("ambiguous", [False, True])
def test_model_binding_fails_closed(ambiguous):
    def core(self):
        pass

    class A:
        _forward_core = core

    class B:
        _forward_core = core

    layers = [A(), B()] if ambiguous else []
    with pytest.raises(RuntimeError, match="bound GDN class"):
        attention_class_for_model(SimpleNamespace(modules=lambda: iter(layers)), core)


def test_v_is_bitwise_and_metadata_is_embedded(tmp_path):
    scope = {"model": "test-model", "precision": "bf16"}
    probe = NormDiagnostic(tmp_path, 0, metadata=scope)
    probe.current_layer = SimpleNamespace(prefix="layer")
    actual, expected = values(), values()
    actual[2][0, 0, 0, 0] = -0.0
    probe.observe(torch.zeros(16, 4096, dtype=torch.bfloat16), actual, expected)
    probe.end_step({})
    report = json.loads((tmp_path / "rank0-summary.json").read_text())
    assert report["counts"] == [0, 0, 1]
    assert report["differences"]["v"]["different"] == 1
    payload = torch.load(tmp_path / "rank0-first.pt", weights_only=True)
    assert payload["metadata"]["model_scope"] == scope


def test_coverage_survives_inactive_last_step(tmp_path):
    probe = NormDiagnostic(tmp_path, 0, max_steps=2)
    observe(probe, "layer")
    probe.end_step({})
    probe.begin_step()
    probe.end_step({})
    report = json.loads((tmp_path / "rank0-summary.json").read_text())
    assert report["status"] == "NO_DIVERGENCE"
    assert report["covered_steps"] == 1
    assert report["seen_layers"] == ["layer"]


def test_graph_warmup_inference_buffers_reset_outside_inference(tmp_path):
    probe = NormDiagnostic(tmp_path, 0)
    with torch.inference_mode():
        observe(probe, "layer", different=True)
    assert probe.slots["layer"]["active"].is_inference()
    probe.begin_step()
    assert probe.slots["layer"]["active"].item() == 0
    with torch.inference_mode():
        observe(probe, "layer", different=True)
    probe.end_step({"request": 1})
    report = json.loads((tmp_path / "rank0-summary.json").read_text())
    assert report["status"] == "DIVERGENCE"
