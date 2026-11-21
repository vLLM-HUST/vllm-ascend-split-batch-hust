"""Fail closed unless installed host packages identify the requested commits."""

from __future__ import annotations

import ast
import hashlib
import importlib.metadata
import importlib.util
import json
import os
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


def check_gate_reader(package, relative, gate):
    """Check the declared reader in the installed host, not examples or proposals."""
    path = (package / relative).resolve()
    if not path.is_relative_to(package) or path.suffix != ".py":
        raise ValueError(f"Invalid host gate reader: {relative!r}")
    tree = ast.parse(path.read_text(), filename=str(path))
    aliases = {
        target.id
        for node in tree.body
        if isinstance(node, ast.Assign)
        and isinstance(node.value, ast.Constant)
        and node.value.value == gate
        for target in node.targets
        if isinstance(target, ast.Name)
    }
    for call in ast.walk(tree):
        if not isinstance(call, ast.Call) or not call.args:
            continue
        if ast.unparse(call.func) not in ("os.environ.get", "os.getenv"):
            continue
        key = call.args[0]
        if (isinstance(key, ast.Constant) and key.value == gate) or (
            isinstance(key, ast.Name) and key.id in aliases
        ):
            return
    raise ValueError(f"No environment reader for {gate} in {relative}")


def route_identity(entry):
    return {
        **entry["service_route"],
        "operator": entry["id"],
        "layer": entry["layer"],
        "gate_env": entry["gate_env"],
    }


def check_service_route(route, host_package):
    """Static prerequisites only. Paired per-rank L4 is still required."""
    plugin = route["layer"] == "plugin_carrier"
    provider = "vllm_ascend_split_batch" if plugin else "vllm_ascend"
    reader = Path(route["reader"])
    consumer = Path(route["consumer"])
    if (
        route["layer"] not in {"plugin_carrier", "host_fork"}
        or reader.parts[0] != provider
        or consumer.parts[0] != "vllm_ascend"
        or ".." in reader.parts + consumer.parts
        or consumer.suffix != ".py"
    ):
        raise ValueError(f"Invalid provider or consumer in {route}")
    package = host_package
    if plugin:
        ep = route["entry_point"]
        value = ".".join(reader.with_suffix("").parts) + ":load"
        dist = importlib.metadata.distribution("vllm-ascend-split-batch")
        entries = dist.entry_points.select(group="vllm.general_plugins", name=ep)
        allowed = os.environ.get("VLLM_PLUGINS")
        if [e.value for e in entries] != [value] or (
            allowed is not None and ep not in allowed.split(",")
        ):
            raise ValueError(f"Plugin entry point missing or disabled: {ep}")
        spec = importlib.util.find_spec(provider)
        if spec is None or not spec.origin:
            raise ValueError(f"Missing plugin module: {provider}")
        package = Path(spec.origin).resolve().parent
        direct = json.loads(dist.read_text("direct_url.json") or "{}")
        if direct.get("dir_info", {}).get("editable"):
            base = Path(unquote(urlparse(direct["url"]).path)).resolve() / "src"
        else:
            base = Path(dist.locate_file("")).resolve()
        if package != base / provider:
            raise ValueError(f"Plugin package shadowed: {package}")
    rel = reader.relative_to(provider)
    check_gate_reader(package, rel, route["gate_env"].removesuffix("=1"))
    source = (host_package / consumer.relative_to("vllm_ascend")).resolve()
    if not source.is_relative_to(host_package):
        raise ValueError(f"Consumer outside host: {source}")
    tree = ast.parse(source.read_bytes())
    scope = tree
    for name in route["scope"].split("."):
        scope = next(
            (
                node
                for node in scope.body
                if isinstance(node, (ast.ClassDef, ast.FunctionDef))
                and node.name == name
            ),
            None,
        )
        if scope is None:
            raise ValueError(f"Missing consumer scope: {route['scope']}")
    target = ast.dump(ast.parse(route["expression"], mode="eval").body)
    if not any(ast.dump(node) == target for node in ast.walk(scope)):
        raise ValueError(
            f"Consumer no longer uses {route['expression']} in {route['scope']}"
        )
    hashes = {
        key: hashlib.sha256(path.read_bytes()).hexdigest()
        for key, path in ((route["reader"], package / rel), (route["consumer"], source))
    }
    if plugin and hashes[route["reader"]] != route["reviewed_plugin_sha256"]:
        raise ValueError(f"Installed plugin differs from reviewed source: {reader}")
    return {"identity": route, "sha256": hashes}


def check_host_runtime(expected, routes=()):
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
    if routes:
        host = Path(result["vllm-ascend"]["module_origin"]).parent
        result["service_routes"] = {
            r["operator"]: check_service_route(r, host) for r in routes
        }
    return result
