"""Fail closed unless installed host packages identify the requested commits."""

from __future__ import annotations

import importlib.metadata
import importlib.util
import json
import re
import subprocess
from pathlib import Path
from urllib.parse import unquote, urlparse


def distribution_revision(distribution):
    direct = json.loads(distribution.read_text("direct_url.json") or "{}")
    revision = direct.get("vcs_info", {}).get("commit_id")
    if direct.get("dir_info", {}).get("editable"):
        parsed = urlparse(direct["url"])
        if parsed.scheme != "file":
            raise ValueError("Editable host must use a local source checkout")
        revision = subprocess.check_output(
            ["git", "-C", unquote(parsed.path), "rev-parse", "HEAD"], text=True
        ).strip()
        dirty = subprocess.check_output(
            ["git", "-C", unquote(parsed.path), "status", "--porcelain"], text=True
        )
        if dirty:
            raise ValueError("Host source checkout has uncommitted changes")
    return revision


def check_host_runtime(expected):
    if set(expected) != {"vllm", "vllm-ascend"}:
        raise ValueError("Specify exact revisions for both HUST host packages")
    result = {}
    for name, commit in expected.items():
        if not isinstance(commit, str) or not re.fullmatch(r"[a-f0-9]{40}", commit):
            raise ValueError(f"Expected full commit for {name}")
        dist = importlib.metadata.distribution(name)
        actual = distribution_revision(dist)
        if actual != commit:
            raise ValueError(
                f"Host mismatch: {name} version={dist.version}, "
                f"revision={actual!r}, expected={commit}; locate the existing HUST "
                "venv and verify its interpreter before installing or rebuilding"
            )
        module = name.replace("-", "_")
        spec = importlib.util.find_spec(module)
        if spec is None or spec.origin is None:
            raise ValueError(f"Host module missing: {module}")
        origin = Path(spec.origin).resolve()
        direct = json.loads(dist.read_text("direct_url.json") or "{}")
        if direct.get("dir_info", {}).get("editable"):
            source = Path(unquote(urlparse(direct["url"]).path)).resolve()
        else:
            source = Path(dist.locate_file("")).resolve()
        if origin != source / module / "__init__.py":
            raise ValueError(f"Host module shadowed: {module} loaded from {origin}")
        result[name] = {
            "module_origin": str(origin),
            "version": dist.version,
            "revision": actual,
            "location": str(Path(dist.locate_file("")).resolve()),
        }
    return result
