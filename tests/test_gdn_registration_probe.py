# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 vLLM-HUST contributors.

import importlib.util
import sys
from pathlib import Path
from types import SimpleNamespace


def load_probe():
    path = Path(__file__).resolve().parents[1] / "tools/probe_gdn_registration.py"
    spec = importlib.util.spec_from_file_location("gdn_registration_probe", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_probe_import_does_not_load_device_stack():
    before = set(sys.modules)
    load_probe()
    assert not (set(sys.modules) - before) & {"torch", "torch_npu", "npugraph_ex"}


def test_opaque_core_is_not_reported_as_a_match():
    module = load_probe()
    nodes = [SimpleNamespace(op="placeholder", target="x"),
             SimpleNamespace(op="call_function",
                             target="vllm.qwen_gdn_attention_core.default")]
    result = module.graph_boundary(SimpleNamespace(nodes=nodes))
    assert result["opaque_gdn_core"] is True
    assert result["decision"] == "outer_fx_cannot_match_core_internals"
    assert len(result["call_function_targets"]) == 1


def test_absent_core_does_not_claim_subgraph_visibility():
    result = load_probe().graph_boundary(SimpleNamespace(nodes=[]))
    assert result["opaque_gdn_core"] is False
    assert result["decision"] == "inspect"
