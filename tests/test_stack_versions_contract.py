# SPDX-License-Identifier: Apache-2.0

"""Guard the single declaration source for the pinned build stack.

The contract is: ``stack-versions.json`` is the only place the
torch / torch_npu / CANN / ATB / SOC four-tuple is declared, ``--check`` fails
on drift instead of adapting to it, and capability probes stay out of the
contract entirely.
"""

import importlib.util
import json
import os
import re
import subprocess
import sys
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parent.parent
RESOLVER = REPO / "tools" / "stack_versions.py"
CANDIDATE_TOOL = REPO / "tools" / "candidate_manifest.py"

# Pre-existing hardcoded install paths that predate the manifest.  These are out
# of the build path, so rewiring them is separate work; they are listed here so
# the exemption is visible and cannot grow silently.
KNOWN_EXEMPT = {
    "tools/bench-ladder.sh": "runtime bench harness; LADDER_ATB_HOME overrides",
    "tools/repair-ascend-login-env.sh": "legacy login repair, pre-manifest env",
    "tests/test_bench_ladder_env_guard.py": "guards bench-ladder.sh ATB default",
    "tests/test_repair_ascend_login_env.py": "guards repair-ascend-login-env.sh",
}

PIN_KEYS = re.compile(
    r"\b(SOC_VERSION|ASCEND_HOME_PATH|ATB_HOME_PATH|TORCH_NPU_PATH|CANN_HOME|VIRTUAL_ENV)\b\s*[=:]"
)
# Only versioned CANN roots and the ATB home are contract-relevant.  The driver
# path carries no version and is not part of the four-tuple.
ASCEND_PATH = re.compile(r"/usr/local/Ascend/(cann-\d|nnal/atb)")


@pytest.fixture(scope="module")
def sv():
    spec = importlib.util.spec_from_file_location("stack_versions_under_test", RESOLVER)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


@pytest.fixture(scope="module")
def manifest(sv):
    return sv.load_manifest()


@pytest.fixture(scope="module")
def candidate(sv):
    spec = importlib.util.spec_from_file_location(
        "candidate_manifest_under_test", CANDIDATE_TOOL
    )
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _source_serve_env(stack_manifest=None, python_exe=None):
    """Source serve-env.sh in a child shell and report its exit status."""
    env = dict(os.environ)
    env.pop("STACK_MANIFEST", None)
    env.pop("STACK_PYTHON", None)
    env.pop("VLLM_SERVING_PYTHON", None)
    if stack_manifest is not None:
        env["STACK_MANIFEST"] = str(stack_manifest)
    if python_exe is not None:
        env["STACK_PYTHON"] = str(python_exe)
    body = (
        '. "$1/tools/serve-env.sh"; rc=$?; '
        'printf "RC=%s\\nSERVING=%s\\n" "$rc" "${VLLM_SERVING_PYTHON-}"'
    )
    proc = subprocess.run(
        ["bash", "-c", body, "_", str(REPO)],
        capture_output=True,
        text=True,
        env=env,
        cwd=str(REPO),
    )
    parsed = {}
    for line in proc.stdout.splitlines():
        if "=" in line:
            key, _, value = line.partition("=")
            parsed[key] = value
    return int(parsed.get("RC", "-1")), parsed.get("SERVING", ""), proc.stderr


def _synthetic(sv, **overrides):
    base = {
        "schema": sv.SCHEMA,
        "contract": {
            "python": "3.12.13",
            "torch": "2.13.0+cpu",
            "torch_npu": "2.13.0rc1",
            "numpy": "1.26.4",
            "cann": "9.1.0",
            "atb": "9.1.0",
            "soc_version": "ascend910b1",
            "cxx11_abi": True,
        },
        "locations": {
            "venv": "/nonexistent/venv",
            "serving_python": "/nonexistent/serving/python",
            "cann_home": "/nonexistent/cann",
            "atb_home": "/nonexistent/atb",
        },
        "plugin_tree": "/nonexistent/plugin",
    }
    base["contract"].update(overrides)
    return base


def _actual(sv, manifest_doc, **overrides):
    actual = dict(manifest_doc["contract"])
    actual["_torch_npu_include"] = "/nonexistent/include"
    actual.update(overrides)
    return actual


def test_manifest_declares_contract_and_locations(manifest):
    assert set(manifest["contract"]) == set(
        (
            "python",
            "torch",
            "torch_npu",
            "numpy",
            "cann",
            "atb",
            "soc_version",
            "cxx11_abi",
        )
    )
    assert set(manifest["locations"]) == {
        "venv",
        "serving_python",
        "cann_home",
        "atb_home",
    }
    assert isinstance(manifest["contract"]["cxx11_abi"], bool)


def test_missing_serving_python_is_drift(sv):
    """A declared serving interpreter that is absent must not pass silently."""
    doc = _synthetic(sv)
    fields = {field for field, _, _ in sv.serving_drift(doc)}
    assert fields == {"serving_python"}


def test_serve_env_excludes_build_venv(sv, manifest):
    """Serving must not inherit the build venv's torch alongside its own vllm."""
    joined = "\n".join(sv.serve_env_exports(manifest))
    assert "VIRTUAL_ENV" not in joined
    assert f"{manifest['locations']['venv']}/bin" not in joined
    assert (
        f"export VLLM_SERVING_PYTHON={manifest['locations']['serving_python']}"
        in joined
    )
    for key in ("ASCEND_HOME_PATH", "ATB_HOME_PATH", "ATB_ENV", "SOC_VERSION"):
        assert f"export {key}=" in joined, key


def test_pep440_rc_spelling_is_equivalent(sv, manifest):
    doc = _synthetic(sv)
    for spelling in ("2.13.0.rc1", "2.13.0rc1"):
        assert sv.drift(doc, _actual(sv, doc, torch_npu=spelling)) == []


def test_local_segment_is_contract_relevant(sv, manifest):
    doc = _synthetic(sv)
    # A build without the +cpu local segment is a different wheel identity.
    assert sv.drift(doc, _actual(sv, doc, torch="2.13.0")) != []


def test_absent_soc_is_not_drift(sv):
    doc = _synthetic(sv)
    assert sv.drift(doc, _actual(sv, doc, soc_version=None)) == []


def test_drift_reports_every_mismatched_field(sv):
    doc = _synthetic(sv)
    actual = _actual(sv, doc, cann="9.2.0", torch="2.12.0+cpu", cxx11_abi=False)
    fields = {field for field, _, _ in sv.drift(doc, actual)}
    assert fields == {"cann", "torch", "cxx11_abi"}


def test_env_exports_cover_build_inputs(sv, manifest):
    lines = sv.env_exports(manifest)
    joined = "\n".join(lines)
    for key in (
        "ASCEND_HOME_PATH",
        "ATB_HOME_PATH",
        "ATB_ENV",
        "ATB_CXX_ABI",
        "SOC_VERSION",
        "TORCH_NPU_PATH",
        "VIRTUAL_ENV",
        "PATH",
    ):
        assert f"export {key}=" in joined, key
    assert manifest["contract"]["soc_version"] in joined


def test_cmake_args_cover_build_inputs(sv, manifest):
    venv_python = Path(manifest["locations"]["venv"]) / "bin" / "python"
    if not venv_python.is_file():
        pytest.skip("declared build venv is not installed on this host")
    joined = " ".join(sv.cmake_args(manifest))
    for flag in (
        "-DASCEND_HOME_PATH=",
        "-DTORCH_NPU_PATH=",
        "-DSOC_VERSION=",
        "-DPYTHON_EXECUTABLE=",
        "-DPYTHON_INCLUDE_PATH=",
        "-DCMAKE_PREFIX_PATH=",
    ):
        assert flag in joined, flag


def test_resolver_emits_no_capability_probe_flags(sv, manifest):
    """Probes ask what the headers support; they must not be pinned here."""
    venv_python = Path(manifest["locations"]["venv"]) / "bin" / "python"
    if not venv_python.is_file():
        pytest.skip("declared build venv is not installed on this host")
    joined = " ".join(sv.cmake_args(manifest))
    assert "HAVE_" not in joined
    assert "CMAKE_REQUIRED" not in joined


def test_env_scripts_restate_no_pin():
    for name in ("build-env.sh", "serve-env.sh"):
        text = (REPO / "tools" / name).read_text()
        assert ASCEND_PATH.search(text) is None, name
        for literal in ("2.13.0", "9.1.0", "ascend910", "python3.12"):
            assert literal not in text, (name, literal)


def test_no_second_declaration_site():
    offenders = []
    for root in ("src", "tools", "tests"):
        for path in sorted((REPO / root).rglob("*")):
            if path.suffix not in {".py", ".sh", ".json", ".toml", ".cfg"}:
                continue
            rel = path.relative_to(REPO).as_posix()
            if rel in KNOWN_EXEMPT or path.name in {
                "stack-versions.json",
                "stack_versions.py",
                "build-env.sh",
                "serve-env.sh",
            }:
                continue
            if rel == "tests/test_stack_versions_contract.py":
                continue
            for lineno, line in enumerate(
                path.read_text(errors="replace").splitlines(), 1
            ):
                if line.lstrip().startswith("#"):
                    continue
                if PIN_KEYS.search(line) or ASCEND_PATH.search(line):
                    offenders.append(f"{rel}:{lineno}: {line.strip()[:110]}")
    assert not offenders, (
        "build-stack pins must live only in stack-versions.json:\n"
        + "\n".join(offenders)
    )


def test_serve_env_without_selection_uses_repo_manifest(manifest):
    rc, serving, stderr = _source_serve_env()
    assert rc == 0, stderr
    assert serving == manifest["locations"]["serving_python"]


def test_serve_env_selects_declared_manifest(tmp_path, candidate):
    """STACK_MANIFEST must change which interpreter serving resolves to."""
    snapshot = tmp_path / "candidate.json"
    relocated = str(tmp_path / "candidate-venv" / "bin" / "python")
    candidate.main(["--serving-python", relocated, "--out", str(snapshot)])
    rc, serving, stderr = _source_serve_env(stack_manifest=snapshot)
    assert rc == 0, stderr
    assert serving == relocated


def test_serve_env_fails_closed_when_manifest_cannot_be_resolved(tmp_path):
    """A failing resolver must abort the source, not apply an empty export set."""
    rc, serving, stderr = _source_serve_env(stack_manifest=tmp_path / "absent.json")
    assert rc != 0
    assert serving == ""
    assert "manifest resolution failed" in stderr


def test_candidate_snapshot_only_relocates_interpreter(
    sv, manifest, candidate, tmp_path
):
    relocated = str(tmp_path / "candidate-venv" / "bin" / "python")
    snapshot = candidate.build_snapshot(sv.DEFAULT_MANIFEST, relocated)
    assert snapshot["contract"] == manifest["contract"]
    assert snapshot["plugin_tree"] == manifest["plugin_tree"]
    assert snapshot["locations"]["serving_python"] == relocated
    assert snapshot["locations"]["venv"] == manifest["locations"]["venv"]
    assert snapshot["snapshot"]["schema"] == candidate.SNAPSHOT_SCHEMA
    assert snapshot["snapshot"]["source_manifest_sha256"] == candidate.sha256_file(
        sv.DEFAULT_MANIFEST
    )
    assert snapshot["schema"] == sv.SCHEMA


def test_candidate_snapshot_is_loadable_by_the_resolver(sv, candidate, tmp_path):
    out = tmp_path / "candidate.json"
    candidate.main(["--serving-python", str(tmp_path / "p"), "--out", str(out)])
    loaded = sv.load_manifest(out)
    assert loaded["locations"]["serving_python"] == str(tmp_path / "p")


def test_candidate_snapshot_verify_reports_stale_source(candidate, tmp_path):
    out = tmp_path / "candidate.json"
    candidate.main(["--serving-python", str(tmp_path / "p"), "--out", str(out)])
    assert candidate.verify(out, candidate.stack_versions.DEFAULT_MANIFEST) == 0

    doc = json.loads(out.read_text())
    doc["snapshot"]["source_manifest_sha256"] = "0" * 64
    out.write_text(json.dumps(doc))
    assert candidate.verify(out, candidate.stack_versions.DEFAULT_MANIFEST) == 1


def test_candidate_snapshot_exposes_no_contract_option(sv):
    """The snapshot tool offers no flag that could rewrite a contract value."""
    proc = subprocess.run(
        [sys.executable, str(CANDIDATE_TOOL), "--help"],
        capture_output=True,
        text=True,
    )
    assert proc.returncode == 0, proc.stderr
    for key in sv.CONTRACT_KEYS:
        assert f"--{key}" not in proc.stdout, key
