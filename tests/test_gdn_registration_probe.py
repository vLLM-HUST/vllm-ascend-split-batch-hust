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


def test_inner_core_call_inventory_is_scoped():
    source = '''
class AscendGatedDeltaNetAttention:
    def other(self, x):
        return unrelated_hook(x)
    def _forward_core(self, x):
        q, k, v = self.rearrange_mixed_qkv(x)
        return l2norm_fwd(q), l2norm_fwd(k), v
'''
    result = load_probe().core_calls(source)
    assert result['direct_l2norm'] is True
    assert result['instance_unpack'] is True
    assert 'unrelated_hook' not in result['calls']


def test_gdn_bundle_entrypoints_remain_absent():
    import json

    root = Path(__file__).resolve().parents[1]
    try:
        import tomllib
    except ModuleNotFoundError:
        import tomli as tomllib
    entries = tomllib.loads((root / 'pyproject.toml').read_text())[
        'project']['entry-points']
    assert 'gdn-qkv-norm' not in entries['vllm.general_plugins']
    assert 'org.vllm-hust.gdn-qkv-norm' not in entries[
        'vllm_hust.extension_bundles']
    assert not (root / 'src/vllm_ascend_split_batch/gdn_qkv_norm_bundle').exists()
    decision = json.loads((root / 'host-proposals/GDN-qkv-norm/'
                           'registration-closure.json').read_text())
    assert decision['bundle_registration'] == 'withdrawn'
    assert decision['kernel_changed'] is False
