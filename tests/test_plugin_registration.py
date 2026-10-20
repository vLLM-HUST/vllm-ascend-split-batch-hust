import json
import subprocess
import sys
from pathlib import Path

import pytest

try:
    import tomllib
except ModuleNotFoundError:
    import tomli as tomllib

ROOT = Path(__file__).resolve().parents[1]
PROJECT = tomllib.loads((ROOT / "pyproject.toml").read_text())["project"]


def test_cascade_is_not_a_general_plugin():
    plugins = PROJECT["entry-points"]["vllm.general_plugins"]
    assert "cascade-attention" not in plugins
    assert all("cascade" not in target for target in plugins.values())
    assert set(plugins) == {
        "fi-sampling",
        "fi-gelu",
        "fia-demask",
        "aot-cache-guard",
        "mlp-chunk",
        "gate-fp32",
        "gdn-merge-inproj",
        "rope-fix",
        "zerocost-wiring",
    }
    for target in plugins.values():
        module, entry = target.split(":")
        assert entry == "load"
        assert (ROOT / "src" / (module.replace(".", "/") + ".py")).is_file()


def test_external_cascade_kernel_dependency_is_disconnected():
    assert "kernels" not in PROJECT["optional-dependencies"]
    dependencies = list(PROJECT["dependencies"])
    for extra in PROJECT["optional-dependencies"].values():
        dependencies.extend(extra)
    assert all("ascend-kernel" not in item.lower() for item in dependencies)


def test_no_bundle_can_inject_cascade_or_register_its_carriers():
    for path in (ROOT / "src").rglob("vllm-hust-extension-v0.2.json"):
        raw = json.loads(path.read_text())
        assert not any("CASCADE" in key for key in raw["activation"]["environment"])
        assert not any("cascade" in item["module"] for item in raw["implementation"])


def test_host_plugin_contract_is_preserved():
    entries = PROJECT["entry-points"]
    assert "vllm.platform_plugins" not in entries
    assert len(entries["vllm_hust.extension_bundles"]) == 6
    assert PROJECT["dependencies"] == []


@pytest.mark.parametrize("stale_env", [False, True])
def test_installed_plugins_are_isolated_from_cascade(tmp_path, stale_env):
    output = tmp_path / "probe.json"
    command = [
        sys.executable,
        str(ROOT / "tools/probe_cascade_retirement.py"),
        "--output",
        str(output),
    ]
    if stale_env:
        command.append("--stale-cascade-env")
    result = subprocess.run(command, capture_output=True, text=True, check=False)
    assert result.returncode == 0, result.stdout + result.stderr
    report = json.loads(output.read_text())
    assert report["status"] == "PASS"
    assert report["stale_cascade_env"] is stale_env
    assert report["forbidden_import_attempts"] == []
    assert len(report["plugins"]) == 9
