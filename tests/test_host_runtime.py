"""Runtime identity must be established before accepting serving evidence."""

import importlib.util
from pathlib import Path
from types import SimpleNamespace

import pytest

PATH = Path(__file__).resolve().parents[1] / "tools/operator_tools/host_runtime.py"
spec = importlib.util.spec_from_file_location("host_runtime", PATH)
host = importlib.util.module_from_spec(spec)
spec.loader.exec_module(host)
COMMITS = {"vllm": "a" * 40, "vllm-ascend": "b" * 40}


def test_reject_community_host(monkeypatch):
    dist = SimpleNamespace(version="0.23.0+empty", read_text=lambda _: None)
    monkeypatch.setattr(host.importlib.metadata, "distribution", lambda _: dist)
    with pytest.raises(ValueError, match="Host mismatch"):
        host.check_host_runtime(COMMITS)


def test_accept_exact_host_revisions(monkeypatch):
    import json

    def distribution(name):
        return SimpleNamespace(
            version="hust",
            read_text=lambda _: json.dumps({"vcs_info": {"commit_id": COMMITS[name]}}),
            locate_file=lambda _: "/installed",
        )

    monkeypatch.setattr(host.importlib.metadata, "distribution", distribution)
    monkeypatch.setattr(
        host.importlib.util,
        "find_spec",
        lambda name: SimpleNamespace(origin=f"/installed/{name}/__init__.py"),
    )
    assert host.check_host_runtime(COMMITS)["vllm"]["revision"] == COMMITS["vllm"]
    monkeypatch.setattr(
        host.importlib.util,
        "find_spec",
        lambda name: SimpleNamespace(origin=f"/wrong-host/{name}/__init__.py"),
    )
    with pytest.raises(ValueError, match="Host module shadowed"):
        host.check_host_runtime(COMMITS)


@pytest.mark.parametrize("dirty", [False, True])
def test_editable_host_uses_checkout_and_actual_module(tmp_path, monkeypatch, dirty):
    import json

    source = tmp_path / "hust"
    dist = SimpleNamespace(
        version="stale-install-version",
        read_text=lambda _: json.dumps(
            {"url": source.as_uri(), "dir_info": {"editable": True}}
        ),
        locate_file=lambda _: tmp_path / "venv",
    )
    monkeypatch.setattr(host.importlib.metadata, "distribution", lambda _: dist)
    monkeypatch.setattr(
        host.subprocess,
        "check_output",
        lambda argv, **kw: (
            (" M file.py" if dirty else "") if "status" in argv else "a" * 40
        ),
    )
    monkeypatch.setattr(
        host.importlib.util,
        "find_spec",
        lambda name: SimpleNamespace(origin=str(source / name / "__init__.py")),
    )
    expected = dict.fromkeys(COMMITS, "a" * 40)
    if dirty:
        with pytest.raises(ValueError, match="uncommitted"):
            host.check_host_runtime(expected)
    else:
        result = host.check_host_runtime(expected)
        assert result["vllm"]["module_origin"] == str(source / "vllm/__init__.py")


@pytest.mark.parametrize("expected", [{}, {"vllm": "main", "vllm-ascend": "b" * 40}])
def test_reject_missing_or_ambiguous_identity(expected):
    with pytest.raises(ValueError):
        host.check_host_runtime(expected)


@pytest.mark.parametrize(
    "expr",
    [
        'os.environ.get("VLLM_HUST_TEST", "0")',
        "os.getenv(NAME)",
        '"VLLM_HUST_TEST"',
        "other.get(NAME)",
    ],
)
def test_gate_must_have_an_environment_reader(tmp_path, expr):
    (tmp_path / "reader.py").write_text(f'NAME = "VLLM_HUST_TEST"\nvalue = {expr}\n')
    if expr.startswith("os."):
        host.check_gate_reader(tmp_path, "reader.py", "VLLM_HUST_TEST")
    else:
        with pytest.raises(ValueError, match="No environment reader"):
            host.check_gate_reader(tmp_path, "reader.py", "VLLM_HUST_TEST")


def test_gate_with_removed_consumer_or_wrong_owner_is_not_a_service_route(tmp_path):
    module = tmp_path / "ops.py"
    module.write_text(
        'import os\nFLAG = os.getenv("GATE")\ndef forward(x):\n    return native(x)\n'
    )
    route = dict(
        operator="demo",
        layer="host_fork",
        gate_env="GATE=1",
        reader="vllm_ascend/ops.py",
        consumer="vllm_ascend/ops.py",
        scope="forward",
        expression="fused(x)",
    )
    host.check_gate_reader(tmp_path, "ops.py", "GATE")
    with pytest.raises(ValueError, match="Consumer no longer uses"):
        host.check_service_route(route, tmp_path)
    module.write_text(module.read_text().replace("return native(x)", "return fused(x)"))
    receipt = host.check_service_route(route, tmp_path)
    assert set(receipt["sha256"]) == {"vllm_ascend/ops.py"}
    from tools.operator_tools.serving_sync import validate_route

    entry = {
        "id": "demo",
        "layer": route["layer"],
        "gate_env": route["gate_env"],
        "service_route": {
            k: route[k] for k in ("reader", "consumer", "scope", "expression")
        },
    }
    observed = {"service_routes": {"demo": receipt}}
    assert validate_route(entry, observed) == receipt
    with pytest.raises(ValueError, match="owner or consumer"):
        validate_route(entry, {})
    entry["layer"] = "plugin_carrier"
    with pytest.raises(ValueError, match="owner or consumer"):
        validate_route(entry, observed)
    with pytest.raises(ValueError, match="Invalid provider"):
        host.check_service_route(dict(route, layer="plugin_carrier"), tmp_path)
