import ast
import importlib.util
from pathlib import Path
from types import SimpleNamespace

import pytest

torch = pytest.importorskip("torch")
ROOT = Path(__file__).resolve().parents[1]
SPEC = importlib.util.spec_from_file_location(
    "concatd_rsp", ROOT / "src/vllm_ascend_split_batch/runtime_source_patches.py"
)
rsp = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(rsp)


def _helper(source):
    node = next(
        n
        for n in ast.parse(source).body
        if isinstance(n, ast.FunctionDef) and n.name == "_unpack_mixed_qkv_views"
    )
    namespace = {}
    exec(
        compile(ast.Module(body=[node], type_ignores=[]), "<helper>", "exec"), namespace
    )
    return namespace[node.name]


def _carrier_helper():
    source = next(
        new
        for _, new, _ in rsp._VIEWS_TRANSFORMS
        if "def _unpack_mixed_qkv_views" in new
    )
    return _helper(source)


@pytest.mark.parametrize("rows", [1, 2, 8, 32])
@pytest.mark.parametrize("layout", ["contiguous", "row-padded", "transposed"])
@pytest.mark.parametrize("dtype", [torch.bfloat16, torch.float32])
def test_carrier_slices_are_contiguous_and_match_stock(rows, layout, dtype):
    width = 4096
    if layout == "transposed":
        mixed = torch.arange(rows * width).reshape(width, rows).t().to(dtype)
    else:
        stride = width if layout == "contiguous" else 6176
        mixed = torch.arange(rows * stride).reshape(rows, stride).to(dtype)[:, :width]
    layer = SimpleNamespace(
        key_dim=2048, value_dim=4096, tp_size=2, head_k_dim=128, head_v_dim=128
    )
    got = _carrier_helper()(mixed, layer)
    for actual, part, heads in zip(
        got, torch.split(mixed, [1024, 1024, 2048], -1), [8, 8, 16], strict=True
    ):
        expected = part.reshape(-1).view(1, rows, heads, 128)
        assert actual.is_contiguous()
        assert actual.shape == expected.shape
        assert torch.equal(actual, expected)
        assert torch.equal(actual.view(-1, 128), expected.view(-1, 128))


def test_old_helper_fails_at_l2norm_flatten_for_multirow_input():
    source = next(
        new
        for _, new, _ in rsp._VIEWS_TRANSFORMS
        if "def _unpack_mixed_qkv_views" in new
    )
    old_helper = _helper(source.replace(".contiguous().view(", ".view("))
    mixed = torch.zeros(8, 4096)
    layer = SimpleNamespace(
        key_dim=2048, value_dim=4096, tp_size=2, head_k_dim=128, head_v_dim=128
    )
    q, _, _ = old_helper(mixed, layer)
    with pytest.raises(RuntimeError, match="view size is not compatible"):
        q.view(-1, q.shape[-1])


@pytest.mark.parametrize("rows", [1, 8])
def test_benchmark_adapter_matches_carrier(rows):
    spec = importlib.util.spec_from_file_location(
        "concatd_adapter", ROOT / "tools/bench_concatd_views_adapter.py"
    )
    adapter = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(adapter)
    mixed = torch.randn(rows, 4096)
    want = _carrier_helper()(mixed, adapter.layer)
    for actual, expected in zip(adapter.unpack(mixed), want, strict=True):
        assert torch.equal(actual, expected)
        assert actual.stride() == expected.stride()


def test_runtime_patch_matches_carrier():
    patch = (
        ROOT
        / "host-proposals/ConcatD-zero-copy-unpack/patches"
        / "20261024_contiguous.gdn.py.patch"
    )
    added = [
        line[1:] for line in patch.read_text().splitlines() if line.startswith("+    ")
    ]
    source = next(
        new
        for _, new, _ in rsp._VIEWS_TRANSFORMS
        if "def _unpack_mixed_qkv_views" in new
    )
    assert len(added) == 3
    assert all(line in source for line in added)
    assert all(".contiguous().view(" in line for line in added)


@pytest.mark.parametrize("rows", [1, 2, 8, 16, 32])
@pytest.mark.parametrize("prefills", [0, 1])
@pytest.mark.parametrize("enabled", [False, True])
@pytest.mark.parametrize("contiguous", [False, True])
def test_production_dispatch_scope(rows, prefills, enabled, contiguous):
    import textwrap

    source = next(
        new
        for _, new, _ in rsp._VIEWS_TRANSFORMS
        if "elif mixed_qkv_non_spec.is_contiguous()" in new
    )
    calls = []

    def stock(x):
        if x is None:
            return None, None, None
        calls.append("stock")
        return "stock", "stock", "stock"

    def views(x, layer):
        calls.append("views")
        return "views", "views", "views"

    namespace = {"_UNPACK_VIEWS_ON": enabled, "_unpack_mixed_qkv_views": views}
    body = textwrap.indent(textwrap.dedent(source), "    ")
    exec(
        "def dispatch(self, attn_metadata, mixed_qkv_spec, mixed_qkv_non_spec):\n"
        + body
        + "\n    return query_non_spec, key_non_spec, value_non_spec\n",
        namespace,
    )
    mixed = torch.zeros(rows, 4096) if contiguous else torch.zeros(rows, 8192)[:, ::2]
    got = namespace["dispatch"](
        SimpleNamespace(rearrange_mixed_qkv=stock),
        SimpleNamespace(num_prefills=prefills),
        None,
        mixed,
    )
    expected = (
        "views" if enabled and not prefills and rows == 1 and contiguous else "stock"
    )
    assert calls == [expected]
    assert got == (expected,) * 3
