"""Input transfer must retain the layouts exercised by operator tests."""

import importlib.util
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]


def load_utils():
    path = ROOT / ".openbitfun/skills/triton-op-verifier/scripts/_common_utils.py"
    spec = importlib.util.spec_from_file_location("verifier_layout_utils", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


@pytest.mark.parametrize("layout", ["slice", "transpose", "expand", "empty"])
def test_transfer_preserves_layout_and_values(layout):
    torch = pytest.importorskip("torch")
    base = torch.arange(48).reshape(6, 8)
    source = {
        "slice": base[1:5, 1::2],
        "transpose": base.T,
        "expand": base[:1].expand(4, 8),
        "empty": base[6:],
    }[layout]
    utils = load_utils()
    nested = utils.move_to_device({"unused": 1}, "cpu")
    assert nested == {"unused": 1}
    actual = utils.move_to_device([source, (source, None)], "cpu")
    for result in (actual[0], actual[1][0]):
        assert result.stride() == source.stride()
        assert result.storage_offset() == source.storage_offset()
        torch.testing.assert_close(result, source, rtol=0, atol=0)
        assert (
            result.untyped_storage().data_ptr() != source.untyped_storage().data_ptr()
        )
    description = utils.describe_input([actual[0]])[0]
    assert description["stride"] == list(source.stride())
    assert description["storage_offset"] == source.storage_offset()


@pytest.mark.npu
@pytest.mark.parametrize("rows", [1, 7, 32, 97, 266])
@pytest.mark.parametrize("stride", [1, 2])
def test_input_norm_npu_strides(rows, stride, monkeypatch):
    torch = pytest.importorskip("torch")
    pytest.importorskip("torch_npu")
    pytest.importorskip("triton")
    if not torch.npu.is_available():
        pytest.skip("NPU unavailable")
    from vllm_ascend_split_batch import gdn_input_norm

    monkeypatch.setattr(gdn_input_norm, "_NORM_ENABLED", True)
    torch.manual_seed(42)
    source = torch.randn(rows, 2048 * stride, dtype=torch.bfloat16)[:, ::stride]
    x = load_utils().move_to_device(source, "npu:0")
    assert x.stride() == source.stride()
    weight = torch.randn(2048, dtype=torch.bfloat16, device="npu:0")
    var = x.float().square().mean(-1, keepdim=True)
    expected = (x.float() * torch.rsqrt(var + 1e-6)).to(x.dtype) * weight
    actual = gdn_input_norm.gdn_input_norm_fused(x, weight)
    torch.testing.assert_close(actual, expected, atol=1e-3, rtol=1e-3)
