import importlib.util
import sys
from pathlib import Path

import pytest


def load_probe():
    path = Path(__file__).resolve().parents[1] / "tools/probe_gelu_registration.py"
    spec = importlib.util.spec_from_file_location("gelu_registration_probe", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_probe_import_is_device_independent():
    before = set(sys.modules)
    load_probe()
    assert not (set(sys.modules) - before) & {"torch", "torch_npu", "vllm"}


def test_definition_executes_only_requested_class(tmp_path):
    path = tmp_path / "host.py"
    path.write_text("""
raise RuntimeError("unrelated module side effect")
class Target:
    def forward(self, x: MissingDependency):
        return x + 1
class Unrelated:
    raise RuntimeError("unrelated class")
""")
    namespace = {}
    exec(load_probe().definition(path, "Target"), namespace)
    assert namespace["Target"]().forward(4) == 5
    assert "Unrelated" not in namespace


def test_missing_host_definition_fails_loudly(tmp_path):
    path = tmp_path / "host.py"
    path.write_text("class Other: pass\n")
    with pytest.raises(StopIteration):
        load_probe().definition(path, "Target")
