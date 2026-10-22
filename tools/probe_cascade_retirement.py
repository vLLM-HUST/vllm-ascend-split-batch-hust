"""Check installed plugin isolation without importing the device stack."""

import argparse
import importlib.abc
import importlib.metadata
import importlib.util
import json
import os
import sys
from pathlib import Path


class ImportGuard(importlib.abc.MetaPathFinder):
    def __init__(self):
        self.attempts = []

    def find_spec(self, fullname, path=None, target=None):
        if fullname.split(".")[0] in {
            "vllm",
            "vllm_ascend",
            "torch",
            "torch_npu",
            "triton",
            "ascend_kernel",
        } or fullname.startswith("vllm_ascend_split_batch.cascade"):
            self.attempts.append(fullname)
            raise ImportError(
                f"device/retired import forbidden in offline probe: {fullname}"
            )
        return None


def probe(stale_cascade_env=False):
    for key in list(os.environ):
        if key.startswith(("VLLM_HUST_", "VLLM_ASCEND_")):
            del os.environ[key]
    os.environ["VLLM_HUST_AOT_CACHE_GUARD"] = "0"
    if stale_cascade_env:
        os.environ["VLLM_ASCEND_ENABLE_CASCADE_DECODE"] = "1"
        os.environ["VLLM_ASCEND_ENABLE_CASCADE_GRAPH"] = "1"
    removed_modules = [
        f"vllm_ascend_split_batch.{name}"
        for name in (
            "cascade_plugin",
            "cascade_graph_plugin",
            "cascade_runner_patch",
            "cascade_gate",
            "cascade_gate_self",
        )
    ]
    for module in removed_modules:
        assert importlib.util.find_spec(module) is None, module
    guard = ImportGuard()
    sys.meta_path.insert(0, guard)
    try:
        dist = importlib.metadata.distribution("vllm-ascend-split-batch")
        entries = [e for e in dist.entry_points if e.group == "vllm.general_plugins"]
        assert len(entries) == 9
        assert all("cascade" not in e.name + e.value for e in entries)
        assert not any("ascend-kernel" in r.lower() for r in (dist.requires or []))
        loaded = {}
        for entry in entries:
            loaded[entry.name] = entry.load()()
            assert loaded[entry.name] is False, entry.name
        package_dir = Path(
            importlib.util.find_spec(
                "vllm_ascend_split_batch"
            ).submodule_search_locations[0]
        )
        root = json.loads((package_dir / "vllm-hust-extension-v0.2.json").read_text())
        assert root["activation"]["environment"] == {}
        assert [item["status"] for item in root["implementation"]] == ["import_only"]
        assert not guard.attempts, guard.attempts
        return {
            "status": "PASS",
            "distribution": str(dist.locate_file("")),
            "stale_cascade_env": stale_cascade_env,
            "aot_guard": "explicitly disabled for offline isolation only",
            "plugins": loaded,
            "forbidden_import_attempts": guard.attempts,
            "root_bundle": "planner-only/import_only",
            "removed_modules": removed_modules,
            "npu_validation": "NOT_RUN",
            "performance_claim": False,
        }
    finally:
        sys.meta_path.remove(guard)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--stale-cascade-env", action="store_true")
    args = parser.parse_args()
    result = probe(args.stale_cascade_env)
    with args.output.open("x", encoding="utf-8") as stream:
        json.dump(result, stream, indent=2)
        stream.write("\n")
    print(json.dumps(result, sort_keys=True))


if __name__ == "__main__":
    main()
