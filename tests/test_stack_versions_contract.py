# SPDX-License-Identifier: Apache-2.0

"""Guard the single declaration source for the pinned build stack.

The contract is: ``stack-versions.json`` is the only place the
torch / torch_npu / CANN / ATB / SOC four-tuple is declared, ``--check`` fails
on drift instead of adapting to it, and capability probes stay out of the
contract entirely.
"""

import importlib.util
import re
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parent.parent
RESOLVER = REPO / "tools" / "stack_versions.py"

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


def _synthetic(sv, **overrides):
    base = {
        "schema": sv.SCHEMA,
        "contract": {
            "python": "3.12.13",
            "torch": "2.13.0+cpu",
            "torch_npu": "2.13.0rc1",
            "cann": "9.1.0",
            "atb": "9.1.0",
            "soc_version": "ascend910b1",
            "cxx11_abi": True,
        },
        "locations": {
            "venv": "/nonexistent/venv",
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
        ("python", "torch", "torch_npu", "cann", "atb", "soc_version", "cxx11_abi")
    )
    assert set(manifest["locations"]) == {"venv", "cann_home", "atb_home"}
    assert isinstance(manifest["contract"]["cxx11_abi"], bool)


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


def test_build_env_restates_no_pin():
    text = (REPO / "tools" / "build-env.sh").read_text()
    assert ASCEND_PATH.search(text) is None
    for literal in ("2.13.0", "9.1.0", "ascend910"):
        assert literal not in text, literal


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
