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
