# SPDX-License-Identifier: Apache-2.0

"""Guard the candidate serving interpreter against implicit inheritance.

A candidate deployment may only see its own packages: no ``--system-site-packages``,
no user site, no editable install pointing at a legacy tree, and no ``.pth`` that
reaches outside the interpreter. Module attribution is read with ``find_spec``
rather than import, because importing the plugin extension aborts on a host
without an Ascend device.
"""

import json
import os
import subprocess
from pathlib import Path

import pytest
from _deployment import deployed_file

REPO = Path(__file__).resolve().parent.parent
CANDIDATE = Path(
    os.environ.get(
        "HUST_CANDIDATE_PYTHON", "/root/venvs/hust-serving-candidate/bin/python"
    )
)
LEGACY_TREES = (
    "/vllm-workspace/vllm",
    "/root/vllm-workspace/vllm-hust",
    ".stack/",
    "rv1",
)


def _run(code, check=True):
    proc = subprocess.run(
        [str(CANDIDATE), "-c", code],
        capture_output=True,
        text=True,
        env={k: v for k, v in os.environ.items() if k != "PYTHONPATH"},
    )
    if check:
        assert proc.returncode == 0, proc.stderr
    return proc


requires_candidate = pytest.mark.skipif(
    not deployed_file(CANDIDATE), reason=f"candidate interpreter absent: {CANDIDATE}"
)


@requires_candidate
def test_candidate_disables_system_and_user_site():
    cfg = (CANDIDATE.parent.parent / "pyvenv.cfg").read_text()
    assert "include-system-site-packages = false" in cfg
    out = _run("import site; print(site.ENABLE_USER_SITE)").stdout.strip()
    assert out == "False"


@requires_candidate
def test_candidate_site_packages_does_not_reach_outside():
    code = """
import sys, sysconfig, pathlib
site_packages = pathlib.Path(sysconfig.get_paths()["purelib"]).resolve()
prefix = pathlib.Path(sys.prefix).resolve()
for pth in site_packages.glob("*.pth"):
    for line in pth.read_text().splitlines():
        entry = line.strip()
        if not entry or entry.startswith("#") or entry.startswith("import "):
            continue
        target = (site_packages / entry).resolve()
        allowed = target in (prefix, site_packages) or prefix in target.parents
        assert allowed, (pth.name, entry)
print("ok")
"""
    assert _run(code).stdout.strip() == "ok"


@requires_candidate
def test_candidate_distributions_are_not_editable():
    code = """
import json, pathlib
from importlib import metadata
prefix = pathlib.Path(__import__("sysconfig").get_paths()["purelib"]).resolve()
out = {}
for name in ("vllm", "vllm_ascend"):
    try:
        dist = metadata.distribution(name)
    except metadata.PackageNotFoundError:
        continue
    path = pathlib.Path(str(dist._path)).resolve()
    direct = path / "direct_url.json"
    editable = False
    if direct.is_file():
        info = json.loads(direct.read_text()).get("dir_info", {})
        editable = bool(info.get("editable"))
    inside = prefix in path.parents or path == prefix
    out[name] = {"path": str(path), "inside": inside, "editable": editable}
print(json.dumps(out))
"""
    found = json.loads(_run(code).stdout)
    for name, info in found.items():
        assert info["inside"], (name, info["path"])
        assert not info["editable"], (name, info["path"])


@requires_candidate
def test_candidate_package_location_resolves_inside_prefix():
    code = """
import json, sysconfig, importlib.util
from pathlib import Path
prefix = Path(sysconfig.get_paths()["purelib"]).resolve()
out = {}
for name in ("vllm", "vllm_ascend"):
    spec = importlib.util.find_spec(name)
    out[name] = None if spec is None or spec.origin is None else spec.origin
print(json.dumps(out))
"""
    origins = json.loads(_run(code).stdout)
    for name, origin in origins.items():
        if origin is None:
            continue
        assert name in origin or f"{name.replace('_', '/')}" in origin, (name, origin)


def _serve_env_stack_present():
    """serve-env.sh sources the declared CANN set_env.sh; without it rc!=0."""
    try:
        cann = json.loads((REPO / "stack-versions.json").read_text(encoding="utf-8"))[
            "locations"
        ]["cann_home"]
    except (OSError, ValueError, KeyError):
        return False
    return deployed_file(Path(cann) / "set_env.sh")


def test_serve_env_exports_contain_no_legacy_tree():
    """Neither PYTHONPATH nor LD_LIBRARY_PATH may re-add a retired tree."""
    if not _serve_env_stack_present():
        pytest.skip("declared CANN stack not installed on this host")
    body = (
        '. "$1/tools/serve-env.sh" >/dev/null 2>&1; rc=$?; '
        'printf "RC=%s\\nPP=%s\\nLLP=%s\\n" "$rc" '
        '"${PYTHONPATH-}" "${LD_LIBRARY_PATH-}"'
    )
    env = {
        k: v
        for k, v in os.environ.items()
        if k not in {"PYTHONPATH", "LD_LIBRARY_PATH"}
    }
    env.pop("STACK_MANIFEST", None)
    proc = subprocess.run(
        ["bash", "-c", body, "_", str(REPO)],
        capture_output=True,
        text=True,
        env=env,
        cwd=str(REPO),
    )
    parsed = dict(
        line.split("=", 1) for line in proc.stdout.splitlines() if "=" in line
    )
    assert parsed.get("RC") == "0", proc.stderr
    for line in (parsed.get("PP", ""), parsed.get("LLP", "")):
        for entry in line.split(":"):
            for legacy in LEGACY_TREES:
                assert legacy not in entry, (legacy, entry)
