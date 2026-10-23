# SPDX-License-Identifier: Apache-2.0
import os
import subprocess
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
SOURCE = '''from functools import wraps
from vllm.logger import init_logger


def decorated(fn):
    @wraps(fn)
    def wrapper(*args, **kwargs):
        return fn(*args, **kwargs)
    return wrapper


class AutoWeightsLoader:
    def __init__(self, module):
        self.module = module

    def _check_skipped_aliases(self, weights):
        pass

    @decorated
    def load_weights(self, weights):
        autoloaded_weights = set(weights)
        self._check_skipped_aliases(autoloaded_weights)
        return autoloaded_weights
'''


@pytest.mark.parametrize("enabled", ["0", "1"])
def test_fresh_probe_then_import_and_install(tmp_path, enabled):
    site_dir = tmp_path / "site"
    site_dir.mkdir()
    (site_dir / "_editable.pth").write_text(str(ROOT / "src") + "\n")
    pth = ROOT / "packaging/zz_husk_runtime_patches.pth"
    (site_dir / pth.name).write_bytes(pth.read_bytes())
    for package in ("vllm", "vllm/model_executor", "vllm/model_executor/models"):
        directory = tmp_path / package
        directory.mkdir(exist_ok=True)
        (directory / "__init__.py").write_text("")
    (tmp_path / "vllm/logger.py").write_text("init_logger = None\n")
    (tmp_path / "vllm/model_executor/models/utils.py").write_text(SOURCE)
    env = {k: v for k, v in os.environ.items() if not k.startswith("VLLM_HUST_")}
    env["VLLM_HUST_MOE_GATE_MERGE"] = enabled
    env["VLLM_HUST_MOE_GATE_MERGE_TRACE"] = str(tmp_path / "trace.jsonl")
    result = subprocess.run(
        [sys.executable, "-S", "-c", r'''
import importlib
import importlib.util
import inspect
import json
import os
import pathlib
import site
import sys
sys.path.insert(0, sys.argv[1])
site.addsitedir(sys.argv[2])
from vllm_ascend_split_batch import runtime_source_patches as rsp
on = os.environ["VLLM_HUST_MOE_GATE_MERGE"] == "1"
assert rsp._INSTALLED == on
for _ in range(2):
    spec = importlib.util.find_spec(rsp.MODELS_UTILS)
    assert isinstance(spec.loader, rsp._PatchingLoader) == on
    assert rsp.MODELS_UTILS not in sys.modules
module = importlib.import_module(rsp.MODELS_UTILS)
assert isinstance(module.__loader__, rsp._PatchingLoader) == on
fn = module.AutoWeightsLoader.load_weights
assert "_hust_install_gate_merge_all" not in fn.__code__.co_names
assert ("_hust_install_gate_merge_all" in inspect.unwrap(fn).__code__.co_names) == on
if on:
    from types import SimpleNamespace as NS
    from vllm_ascend_split_batch import moe_gate_merge as carrier
    class FakeOps:
        def concat(self, tensors):
            return NS(shape=(sum(t.shape[0] for t in tensors), 8))
    carrier.TorchOps = FakeOps
    blocks = []
    for _ in range(40):
        gate = NS(weight=NS(shape=(256, 8)))
        shared = NS(weight=NS(shape=(1, 8)))
        blocks.append(NS(gate=gate, shared_expert_gate=shared,
                         experts=NS(gate=gate),
                         shared_expert=NS(expert_gate=shared)))
    class Model:
        def modules(self):
            return iter(blocks)
    loader = module.AutoWeightsLoader(Model())
    assert loader.load_weights(["loaded"]) == {"loaded"}
    assert all(isinstance(b.experts.gate, carrier.RouterGateProxy) for b in blocks)
    assert loader.load_weights([]) == set()
    events = [json.loads(s) for s in pathlib.Path(
        os.environ["VLLM_HUST_MOE_GATE_MERGE_TRACE"]).read_text().splitlines()]
    counts = [e["installed"] for e in events if e["event"] == "install_all"]
    assert counts == [40, 0], events
    print("synthetic installed=40; repeated install=0; decorated hook verified")
else:
    assert "vllm_ascend_split_batch.moe_gate_merge" not in sys.modules
    assert not pathlib.Path(os.environ["VLLM_HUST_MOE_GATE_MERGE_TRACE"]).exists()
assert "torch" not in sys.modules
''', str(tmp_path), str(site_dir), str(ROOT / "tests")],
        env=env,
        capture_output=True,
        text=True,
        timeout=30,
    )
    assert result.returncode == 0, result.stdout + result.stderr
