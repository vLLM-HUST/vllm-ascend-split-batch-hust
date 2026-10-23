#!/usr/bin/env python3
"""Resolve the pinned build stack from ``stack-versions.json``.

``stack-versions.json`` is the only place the torch / torch_npu / CANN / ATB /
SOC four-tuple is declared.  Everything that needs those values reads them from
here: ``--env`` for shell, ``--cmake-args`` for CMake.  ``--check`` compares the
manifest against what is actually installed and exits non-zero on drift.

Capability probes (``HAVE_ACLRT_*`` and friends inside CANN's ascendc.cmake) are
deliberately *not* represented here -- they stay adaptive, because they ask what
the installed headers support rather than what version the build contract pins.

Stdlib only, and no ``import torch``: probing versions through
``importlib.metadata`` keeps ``--check`` fast enough to run on every build.
"""

import argparse
import importlib.metadata
import json
import re
import subprocess
import sys
from pathlib import Path

SCHEMA = "vllm-hust.stack-versions.v1"
REPO_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_MANIFEST = REPO_ROOT / "stack-versions.json"

CONTRACT_KEYS = (
    "python",
    "torch",
    "torch_npu",
    "cann",
    "atb",
    "soc_version",
    "cxx11_abi",
)
LOCATION_KEYS = ("venv", "cann_home", "atb_home")


def load_manifest(path=None):
    path = Path(path) if path is not None else DEFAULT_MANIFEST
    manifest = json.loads(path.read_text())
    if manifest.get("schema") != SCHEMA:
        raise ValueError(
            f"{path}: schema must be {SCHEMA!r}, got {manifest.get('schema')!r}"
        )
    missing = [k for k in CONTRACT_KEYS if k not in manifest.get("contract", {})]
    missing += [k for k in LOCATION_KEYS if k not in manifest.get("locations", {})]
    if missing:
        raise ValueError(f"{path}: missing keys: {', '.join(sorted(missing))}")
    return manifest


def torch_base(version):
    """``2.13.0+cpu`` -> ``2.13.0``; the local segment is distilled, not re-pinned."""
    return version.split("+", 1)[0]


def canon(version):
    """PEP 440 spelling equivalence only: ``2.13.0.rc1`` == ``2.13.0rc1``.

    The local segment (``+cpu``) is part of the contract and is never dropped --
    a CPU wheel and a CUDA wheel must not compare equal.
    """
    return re.sub(r"\.(rc|alpha|beta|a|b|post|dev)(\d)", r"\1\2", version.lower())


def python_series(version):
    """``3.12.13`` -> ``3.12`` for site-packages paths."""
    return ".".join(version.split(".")[:2])


def _read_kv(path, key):
    for line in Path(path).read_text().splitlines():
        if line.startswith(f"{key}="):
            return line.split("=", 1)[1].strip()
    return None


def probe_cann(cann_home):
    for info in sorted(Path(cann_home).glob("*/ascend_toolkit_install.info")):
        value = _read_kv(info, "version")
        if value:
            return value
    return None


def probe_atb(atb_home):
    version_info = Path(atb_home) / "latest" / "version.info"
    if not version_info.is_file():
        return None
    match = re.search(r"Ascend-cann-atb\s*:\s*(\S+)", version_info.read_text())
    return match.group(1) if match else None


def probe_python_executable(python_exe):
    out = subprocess.run(
        [python_exe, "-c", "import sys; print('%d.%d.%d' % sys.version_info[:3])"],
        capture_output=True,
        text=True,
        check=True,
    )
    return out.stdout.strip()


def probe(manifest, python_exe=None, device_soc=None):
    """Return the values actually observable on this host."""
    contract = manifest["contract"]
    locations = manifest["locations"]
    venv = Path(locations["venv"])
    site_packages = (
        venv / "lib" / f"python{python_series(contract['python'])}" / "site-packages"
    )

    def dist(name):
        try:
            return importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError:
            return None

    atb_abi_dir = (
        Path(locations["atb_home"])
        / "latest"
        / "atb"
        / f"cxx_abi_{1 if contract['cxx11_abi'] else 0}"
    )
    return {
        "python": probe_python_executable(python_exe or (venv / "bin" / "python")),
        "torch": dist("torch"),
        "torch_npu": dist("torch_npu"),
        "cann": probe_cann(locations["cann_home"]),
        "atb": probe_atb(locations["atb_home"]),
        "soc_version": device_soc,
        "cxx11_abi": atb_abi_dir.is_dir(),
        "_torch_npu_include": str(site_packages / "torch_npu"),
    }


def drift(manifest, actual):
    """Fields whose installed value contradicts the manifest."""
    contract = manifest["contract"]
    found = []
    for key in CONTRACT_KEYS:
        expected = contract[key]
        seen = actual.get(key)
        if key == "soc_version" and seen is None:
            # Card-less hosts cannot report a SOC.  The manifest still pins the
            # build target; absence is not drift.
            continue
        if seen is None:
            found.append((key, expected, seen))
        elif isinstance(expected, bool):
            if bool(seen) is not expected:
                found.append((key, expected, seen))
        elif canon(str(seen)) != canon(str(expected)):
            found.append((key, expected, seen))
    return found


def plugin_torch_pin(plugin_tree):
    """Read (never rewrite) the torch pin the plugin's CMakeLists enforces."""
    cmakelists = Path(plugin_tree) / "CMakeLists.txt"
    if not cmakelists.is_file():
        return None
    match = re.search(r'VERSION_EQUAL\s+"([^"]+)"', cmakelists.read_text())
    return match.group(1) if match else None


def env_exports(manifest):
    contract = manifest["contract"]
    locations = manifest["locations"]
    venv = locations["venv"]
    site_packages = (
        f"{venv}/lib/python{python_series(contract['python'])}/site-packages"
    )
    abi = 1 if contract["cxx11_abi"] else 0
    return [
        f"export VIRTUAL_ENV={venv}",
        f"export PATH={venv}/bin:$PATH",
        f"export CANN_HOME={locations['cann_home']}",
        f"export ASCEND_HOME_PATH={locations['cann_home']}",
        f"export ATB_HOME_PATH={locations['atb_home']}/latest/atb/cxx_abi_{abi}",
        f"export ATB_ENV={locations['atb_home']}/set_env.sh",
        f"export ATB_CXX_ABI={abi}",
        f"export SOC_VERSION={contract['soc_version']}",
        f"export TORCH_NPU_PATH={site_packages}/torch_npu",
    ]


def cmake_args(manifest):
    contract = manifest["contract"]
    locations = manifest["locations"]
    venv = Path(locations["venv"])
    site_packages = (
        venv / "lib" / f"python{python_series(contract['python'])}" / "site-packages"
    )
    include = subprocess.run(
        [
            str(venv / "bin" / "python"),
            "-c",
            "import sysconfig; print(sysconfig.get_path('include'))",
        ],
        capture_output=True,
        text=True,
        check=True,
    ).stdout.strip()
    return [
        f"-DASCEND_HOME_PATH={locations['cann_home']}",
        f"-DTORCH_NPU_PATH={site_packages}/torch_npu",
        f"-DSOC_VERSION={contract['soc_version']}",
        f"-DPYTHON_EXECUTABLE={venv}/bin/python",
        f"-DPYTHON_INCLUDE_PATH={include}",
        "-DCMAKE_PREFIX_PATH="
        + ";".join(
            [
                str(site_packages / "pybind11" / "share" / "cmake" / "pybind11"),
                str(site_packages / "torch" / "share" / "cmake"),
            ]
        ),
    ]


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", default=str(DEFAULT_MANIFEST))
    parser.add_argument("--env", action="store_true", help="emit shell exports")
    parser.add_argument("--cmake-args", action="store_true", help="emit CMake -D flags")
    parser.add_argument("--check", action="store_true", help="exit non-zero on drift")
    parser.add_argument(
        "--json", action="store_true", help="emit manifest + live + drift"
    )
    args = parser.parse_args(argv)

    manifest = load_manifest(args.manifest)

    if args.env:
        print("\n".join(env_exports(manifest)))
        return 0
    if args.cmake_args:
        print(" ".join(cmake_args(manifest)))
        return 0

    actual = probe(manifest)
    found = drift(manifest, actual)
    pin = plugin_torch_pin(manifest["plugin_tree"])

    if args.json:
        print(
            json.dumps(
                {
                    "schema": manifest["schema"],
                    "contract": manifest["contract"],
                    "locations": manifest["locations"],
                    "live": actual,
                    "drift": [
                        {"field": f, "expected": e, "actual": a} for f, e, a in found
                    ],
                    "plugin_torch_pin": pin,
                    "plugin_pin_matches": pin
                    == torch_base(manifest["contract"]["torch"]),
                },
                indent=2,
            )
        )
        return 0 if not found else 1

    if not args.check:
        parser.error("one of --check, --env, --cmake-args, --json is required")

    for field, expected, seen in found:
        print(f"DRIFT {field}: contract={expected!r} installed={seen!r}")
    if pin is not None and pin != torch_base(manifest["contract"]["torch"]):
        print(
            "DRIFT plugin CMakeLists torch pin: "
            f"contract={torch_base(manifest['contract']['torch'])!r} plugin={pin!r}"
        )
        found = list(found) + [
            ("plugin_torch_pin", torch_base(manifest["contract"]["torch"]), pin)
        ]

    if found:
        print(f"FAIL: {len(found)} field(s) drifted from {args.manifest}")
        return 1
    print(f"PASS: installed stack matches {Path(args.manifest).name}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
