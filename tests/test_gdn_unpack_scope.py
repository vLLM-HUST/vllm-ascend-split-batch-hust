import ast
import importlib.util
from pathlib import Path
from types import SimpleNamespace as NS

import pytest

torch = pytest.importorskip("torch")
ROOT = Path(__file__).resolve().parents[1]


def module(name):
    spec = importlib.util.spec_from_file_location(
        name, ROOT / "src/vllm_ascend_split_batch" / f"{name}.py"
    )
    result = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(result)
    return result


@pytest.fixture
def layer(monkeypatch):
    monkeypatch.delenv("VLLM_HUST_GDN_UNPACK_VIEWS", raising=False)
    return NS(
        model_config=NS(
            model="/models/Qwen3.5-35B-A3B",
            hf_text_config=NS(
                model_type="qwen3_5_moe_text",
                hidden_size=2048,
                num_hidden_layers=40,
                num_experts=256,
            ),
        ),
        _hust_unpack_full_graph=True,
        speculative_config=None,
        quant_config=None,
        tp_size=2,
        key_dim=2048,
        value_dim=4096,
        head_k_dim=128,
        head_v_dim=128,
    )


@pytest.mark.parametrize("rows", [0, 1, 2, 8, 14, 16, 32])
@pytest.mark.parametrize("prefills", [0, 1])
@pytest.mark.parametrize("enabled", [False, True])
def test_only_accepted_shapes_are_eligible(layer, rows, prefills, enabled, monkeypatch):
    carrier = module("gdn_unpack")
    monkeypatch.setattr(carrier, "ENABLED", enabled)
    mixed = NS(shape=(rows, 4096), ndim=2, dtype=torch.bfloat16, device=NS(type="npu"))
    assert carrier.eligible(layer, mixed, NS(num_prefills=prefills)) == (
        enabled and not prefills and rows in (1, 16)
    )


@pytest.mark.parametrize(
    "change",
    ["model", "config", "tp", "dtype", "device", "spec", "quant", "width", "heads"],
)
def test_uncovered_domains_fall_back(layer, change):
    carrier = module("gdn_unpack")
    mixed = NS(shape=(16, 4096), ndim=2, dtype=torch.bfloat16, device=NS(type="npu"))
    if change == "model":
        layer.model_config.model = "other"
    elif change == "config":
        layer.model_config.hf_text_config.num_experts = 128
    elif change == "tp":
        layer.tp_size = 1
    elif change == "dtype":
        mixed.dtype = torch.float16
    elif change == "device":
        mixed.device.type = "cpu"
    elif change == "spec":
        layer.speculative_config = object()
    elif change == "quant":
        layer.quant_config = object()
    elif change == "width":
        mixed.shape = (16, 8192)
    else:
        layer.head_k_dim = 64
    assert not carrier.eligible(layer, mixed, NS(num_prefills=0))


@pytest.mark.parametrize("rows", [1, 16])
@pytest.mark.parametrize("dtype", [torch.bfloat16, torch.float16, torch.float32])
def test_shipping_helper_exact_values_strides_and_input(layer, rows, dtype):
    carrier = module("gdn_unpack")
    x = torch.randn(rows, 4096).to(dtype)
    before = x.clone()
    for got, part, heads in zip(
        carrier.unpack(x, layer),
        x.split([1024, 1024, 2048], -1),
        [8, 8, 16],
        strict=True,
    ):
        expected = part.reshape(-1).view(1, rows, heads, 128)
        assert torch.equal(got, expected)
        assert got.stride() == expected.stride()
        assert got.dtype == expected.dtype and got.is_contiguous()
    assert torch.equal(x, before)


@pytest.mark.parametrize(
    "mode", [None, "NONE", "PIECEWISE", "FULL", "FULL_AND_PIECEWISE"]
)
def test_graph_scope_is_frozen_during_weight_loading(layer, mode, monkeypatch):
    import sys
    import types

    parent = types.ModuleType("vllm")
    parent.__path__ = []
    config = types.ModuleType("vllm.config")
    config.get_current_vllm_config_or_none = lambda: (
        NS(compilation_config=NS(cudagraph_mode=mode)) if mode is not None else None
    )
    monkeypatch.setitem(sys.modules, "vllm", parent)
    monkeypatch.setitem(sys.modules, "vllm.config", config)
    carrier = module("gdn_unpack")
    carrier.configure(layer)
    assert layer._hust_unpack_full_graph == (mode == "FULL")
    mixed = NS(shape=(16, 4096), ndim=2, dtype=torch.bfloat16, device=NS(type="npu"))
    assert carrier.eligible(layer, mixed, NS(num_prefills=0)) == (mode == "FULL")


def fixture_source(rsp):
    return (
        "from vllm_ascend.ops.triton.mamba.causal_conv1d import extract_last_width\n"
        "class Layer:\n"
        "    def process_weights_after_loading(self, model):\n"
        "        pass\n"
        "    def run(self, attn_metadata, mixed_qkv_spec, mixed_qkv_non_spec):\n"
        "        if mixed_qkv_non_spec is not None:\n"
        "            output_non_spec = torch.empty_like(mixed_qkv_non_spec)\n"
        + rsp._VIEWS_TRANSFORMS[2][0]
        + "\n        return query_non_spec, key_non_spec, value_non_spec\n"
    )


def test_normalized_stock_dispatch_uses_scope_and_contiguity(layer, monkeypatch):
    rsp = module("runtime_source_patches")
    source = rsp.scoped_gdn_source(fixture_source(rsp))
    node = next(
        n
        for n in ast.walk(ast.parse(source))
        if isinstance(n, ast.FunctionDef) and n.name == "run"
    )
    calls = []
    layer.rearrange_mixed_qkv = lambda x: (
        (None, None, None) if x is None else calls.append("stock") or ("stock",) * 3
    )
    namespace = {
        "torch": torch,
        "_hust_unpack_eligible": lambda *a: True,
        "_unpack_mixed_qkv_views": lambda *a: calls.append("views") or ("views",) * 3,
    }
    exec(
        compile(ast.Module(body=[node], type_ignores=[]), "<dispatch>", "exec"),
        namespace,
    )
    for contiguous in (True, False):
        calls.clear()
        x = torch.zeros(16, 4096) if contiguous else torch.zeros(16, 8192)[:, ::2]
        namespace["run"](layer, NS(num_prefills=0), None, x)
        assert calls == ["views" if contiguous else "stock"]


@pytest.mark.host_tree
def test_stock_and_both_deployed_trees_normalize_to_same_entry():
    rsp = module("runtime_source_patches")
    paths = [
        Path(
            "/root/包体恢复/records/tree-patches/20261021T1355Z_d4_conv_hoist_site2.orig-gdn.py"
        ),
        Path("/root/host-trees/vllm-ascend-base/vllm_ascend/ops/gdn.py"),
        Path("/vllm-workspace/vllm-ascend/vllm_ascend/ops/gdn.py"),
    ]
    for p in paths:
        source = rsp.scoped_gdn_source(p.read_text())
        compile(source, str(p), "exec")
        assert source.count("if (_hust_unpack_eligible(") == 1
        assert source.count("if _hust_unpack_eligible(") == 1
        assert "_UNPACK_VIEWS_ON =" not in source
        assert "def _unpack_mixed_qkv_views(" not in source


def test_unknown_legacy_helper_is_not_executed():
    rsp = module("runtime_source_patches")
    source = (
        fixture_source(rsp) + "\ndef _unpack_mixed_qkv_views(x, y):\n    return x\n"
    )
    with pytest.raises(rsp.AnchorError, match="unrecognized deployed unpack helper"):
        rsp.scoped_gdn_source(source)
