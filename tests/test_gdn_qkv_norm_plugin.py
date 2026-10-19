# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 vLLM-HUST contributors.

import importlib.util
import json
import os
import subprocess
import sys
from pathlib import Path
from types import ModuleType, SimpleNamespace

import pytest

from vllm_ascend_split_batch import gdn_qkv_norm_plugin as plugin

ROOT = Path(__file__).resolve().parents[1]


@pytest.fixture
def host(tmp_path, monkeypatch):
    proposal = ROOT / "host-proposals/GDN-qkv-norm"
    spec = importlib.util.spec_from_file_location("gdn_gen", proposal / "gen_patch.py")
    gen = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(gen)
    path = tmp_path / "gdn.py"
    path.write_text(gen.build_patched((proposal / "host_orig/gdn.py.orig").read_text()))
    monkeypatch.setattr(plugin, "host_source", lambda: path)
    monkeypatch.setenv(plugin.ENV_ENABLE, "1")
    monkeypatch.delenv("USE_DEFAULT_FLA_NORM", raising=False)
    monkeypatch.delenv("VLLM_PLUGINS", raising=False)
    monkeypatch.delitem(sys.modules, plugin.HOST_MODULE, raising=False)
    return path


def test_reviewed_patch_is_ready_and_load_is_read_only(host):
    before = host.read_bytes()
    env = dict(os.environ)
    assert plugin.preflight()["status"] == "ready"
    assert plugin.load() is True
    assert plugin.load() is True
    assert host.read_bytes() == before
    assert dict(os.environ) == env
    assert plugin.HOST_MODULE not in sys.modules


@pytest.mark.parametrize("flag", [None, "0", "yes", "true"])
def test_disabled_does_not_probe_host(monkeypatch, flag):
    if flag is None:
        monkeypatch.delenv(plugin.ENV_ENABLE, raising=False)
    else:
        monkeypatch.setenv(plugin.ENV_ENABLE, flag)
    monkeypatch.setattr(plugin, "host_source", lambda: pytest.fail("host probe"))
    assert plugin.load() is False
    assert plugin.preflight() == {"status": "disabled", "enabled": False}


@pytest.mark.parametrize("norm", ["1", "true", ""])
def test_norm_conflict_is_not_silently_overridden(host, monkeypatch, norm):
    monkeypatch.setenv("USE_DEFAULT_FLA_NORM", norm)
    with pytest.raises(RuntimeError, match="requires USE_DEFAULT_FLA_NORM=0"):
        plugin.load()
    assert os.environ["USE_DEFAULT_FLA_NORM"] == norm


@pytest.mark.parametrize("allowlist", ["", "gate-fp32", "gdn-merge-inproj"])
def test_excluded_entry_is_blocked(host, monkeypatch, allowlist):
    monkeypatch.setenv("VLLM_PLUGINS", allowlist)
    assert plugin.preflight()["status"] == "blocked"


def test_included_entry(host, monkeypatch):
    monkeypatch.setenv("VLLM_PLUGINS", "gate-fp32,gdn-qkv-norm")
    assert plugin.load() is True


def test_missing_and_drifted_host(host):
    host.write_text(host.read_text() + "\n")
    with pytest.raises(RuntimeError, match="seam mismatch"):
        plugin.load()
    host.unlink()
    assert plugin.preflight()["status"] == "blocked"


@pytest.mark.parametrize("flag", [False, None, True])
def test_already_imported_host_requires_enabled_flag(host, monkeypatch, flag):
    module = ModuleType(plugin.HOST_MODULE)
    module._FUSE_GDN_QKV_NORM = flag
    monkeypatch.setitem(sys.modules, plugin.HOST_MODULE, module)
    assert plugin.preflight()["status"] == ("ready" if flag is True else "blocked")


def test_source_resolution_does_not_import_host(tmp_path, monkeypatch):
    root = tmp_path / "vllm_ascend"
    ops = root / "ops"
    ops.mkdir(parents=True)
    for path in (root / "__init__.py", ops / "__init__.py", ops / "gdn.py"):
        path.write_text("raise AssertionError('must not import host')\n")
    for name in ("vllm_ascend", "vllm_ascend.ops", plugin.HOST_MODULE):
        monkeypatch.delitem(sys.modules, name, raising=False)
    spec = SimpleNamespace(submodule_search_locations=[str(root)])
    monkeypatch.setattr(plugin.importlib.util, "find_spec", lambda name: spec)
    assert plugin.host_source() == ops / "gdn.py"
    assert "vllm_ascend.ops" not in sys.modules


def test_loaded_host_uses_loaded_module_path(host, monkeypatch):
    module = ModuleType(plugin.HOST_MODULE)
    module.__file__ = str(host)
    module._FUSE_GDN_QKV_NORM = True
    monkeypatch.setitem(sys.modules, plugin.HOST_MODULE, module)
    monkeypatch.setattr(plugin.importlib.util, "find_spec",
                        lambda name: pytest.fail("already-loaded host lookup"))
    assert plugin.preflight()["status"] == "ready"



def test_missing_package(monkeypatch):
    monkeypatch.setenv(plugin.ENV_ENABLE, "1")
    monkeypatch.delenv("USE_DEFAULT_FLA_NORM", raising=False)
    monkeypatch.delenv("VLLM_PLUGINS", raising=False)
    monkeypatch.setattr(plugin.importlib.util, "find_spec", lambda name: None)
    assert "unavailable" in plugin.preflight()["reason"]


def test_fresh_process_disabled_has_no_heavy_imports():
    script = '''
import sys
from vllm_ascend_split_batch.gdn_qkv_norm_plugin import load
assert load() is False
for name in ('torch', 'torch_npu', 'triton', 'vllm', 'vllm_ascend',
             'vllm_ascend_split_batch.gdn_qkv_norm_kernel'):
    assert name not in sys.modules, name
'''
    env = dict(os.environ, VLLM_HUST_GDN_QKV_NORM="0")
    result = subprocess.run([sys.executable, "-c", script], env=env,
                            capture_output=True, text=True, timeout=30)
    assert result.returncode == 0, result.stderr


def test_bundle_is_discoverable_but_not_released():
    from vllm_hust_ext.manifest import activation_blocker, load_manifest

    try:
        import tomllib
    except ModuleNotFoundError:
        import tomli as tomllib

    path = ROOT / (
        "src/vllm_ascend_split_batch/gdn_qkv_norm_bundle/"
        "vllm-hust-extension-v0.2.json"
    )
    raw = json.loads(path.read_text())
    manifest = load_manifest(path)
    assert manifest.bundle_id == "org.vllm-hust.gdn-qkv-norm"
    assert "import_only" in activation_blocker(manifest)
    assert raw["activation"]["environment"] == {
        plugin.ENV_ENABLE: "1", "USE_DEFAULT_FLA_NORM": "0",
    }
    assert raw["host"]["version_range"] == "==0.25.1rc2.dev125"
    project = tomllib.loads((ROOT / "pyproject.toml").read_text())["project"]
    assert raw["extension_version"] == project["version"]
    entries = project["entry-points"]
    assert entries["vllm.general_plugins"]["gdn-qkv-norm"] == (
        "vllm_ascend_split_batch.gdn_qkv_norm_plugin:load"
    )
    assert entries["vllm_hust.extension_bundles"][manifest.bundle_id] == (
        "vllm_ascend_split_batch.gdn_qkv_norm_bundle"
    )
