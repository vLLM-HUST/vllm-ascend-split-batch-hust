#!/usr/bin/env python3
"""Install one candidate serving environment from a reviewed wheelhouse.

The stack cannot be installed in a single pip resolution: the core wheel wants
``opencv-python-headless>=4.13`` (which requires ``numpy>=2``), while the
contract pins ``numpy`` and triton-ascend pins it exactly.  The procedure is
therefore staged, and the one resulting metadata conflict is asserted by name
instead of being ignored:

1. resolve the core wheel's own closure (numpy may leave the contract value);
2. install the plugin wheel without its dependency closure;
3. install the plugin's declared dependencies, with the contract pins applied;
4. re-apply the contract ``numpy`` pin;
5. gate on ``pip check``: exactly the registered line, nothing else.

Pass ``--expected-conflict`` to change the registered line; any other output is
a failure.  The wheelhouse must already hold the wheels; this tool never builds.
"""

import argparse
import json
import os
import subprocess
import sys
from pathlib import Path

DEFAULT_EXPECTED = "opencv-python-headless {opencv} has requirement numpy>=2"
# torchaudio on PyPI is not the +cpu build and fails to load its native library
# against a cpu-only torch; the pinned stack must come from the torch cpu index.
DEFAULT_TORCH_INDEX = "https://download.pytorch.org/whl/cpu"


def clean_env():
    """Drop PYTHONPATH: CANN's site-packages would answer for the candidate."""
    return {k: v for k, v in os.environ.items() if k != "PYTHONPATH"}


def run(cmd, *, check=True, quiet=False, env=None):
    if not quiet:
        print("+", " ".join(str(c) for c in cmd), flush=True)
    proc = subprocess.run(
        [str(c) for c in cmd], capture_output=True, text=True, env=env
    )
    if proc.stdout and not quiet:
        print(proc.stdout, end="", flush=True)
    if proc.stderr and not quiet:
        print(proc.stderr, end="", file=sys.stderr, flush=True)
    if check and proc.returncode != 0:
        raise SystemExit(f"failed ({proc.returncode}): {' '.join(str(c) for c in cmd)}")
    return proc


def manifest_contract(manifest):
    return json.loads(Path(manifest).read_text())["contract"]


def wheel_by_prefix(wheelhouse, prefix):
    hits = sorted(Path(wheelhouse).glob(f"{prefix}-*.whl"))
    if not hits:
        raise SystemExit(f"no {prefix} wheel in {wheelhouse}")
    return hits[-1]


def requires_dist(wheel):
    import zipfile

    with zipfile.ZipFile(wheel) as archive:
        meta = next(n for n in archive.namelist() if n.endswith(".dist-info/METADATA"))
        text = archive.read(meta).decode()
    out = []
    for line in text.splitlines():
        if not line.startswith("Requires-Dist:"):
            continue
        spec = line.split(":", 1)[1].strip()
        if "extra ==" in spec:
            continue
        out.append(spec)
    return out


def watched_env(paths):
    keep = {"PATH", "HOME", "LANG", "LC_ALL", "TMPDIR", "VIRTUAL_ENV"}
    return {k: v for k, v in paths.items() if k in keep}


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--venv", required=True)
    parser.add_argument("--wheelhouse", required=True)
    parser.add_argument(
        "--manifest",
        default=str(Path(__file__).resolve().parent.parent / "stack-versions.json"),
    )
    parser.add_argument("--core-wheel")
    parser.add_argument("--plugin-wheel")
    parser.add_argument("--extra-find-links", action="append", default=[])
    parser.add_argument(
        "--torch-index",
        default=DEFAULT_TORCH_INDEX,
        help="extra index that carries the +cpu torch stack; empty string disables it",
    )
    parser.add_argument(
        "--expected-conflict",
        default=DEFAULT_EXPECTED,
        help="registered pip-check line; {opencv} is filled from the installed version",
    )
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args(argv)

    venv = Path(args.venv)
    python = venv / "bin" / "python"
    if not python.is_file():
        if args.dry_run:
            raise SystemExit(f"{venv} missing (dry-run does not create environments)")
        run([sys.executable, "-m", "venv", str(venv)])
    pip = [python, "-m", "pip"]
    contract = manifest_contract(args.manifest)

    core = (
        Path(args.core_wheel)
        if args.core_wheel
        else wheel_by_prefix(args.wheelhouse, "vllm")
    )
    plugin = (
        Path(args.plugin_wheel)
        if args.plugin_wheel
        else wheel_by_prefix(args.wheelhouse, "vllm_ascend")
    )
    find_links = ["--find-links", str(args.wheelhouse)]
    for extra in args.extra_find_links:
        find_links += ["--find-links", extra]
    if args.torch_index:
        find_links += ["--extra-index-url", args.torch_index]

    # 1. plugin's declared dependencies, contract pins on top.  Doing this before
    #    the core closure keeps the resolver off the opencv/numpy contradiction.
    pins = [
        f"numpy=={contract['numpy']}",
        f"torch=={contract['torch']}",
        f"torch_npu=={contract['torch_npu']}",
    ]
    run(pip + ["install", *find_links, *pins, *requires_dist(plugin)])
    # 2. core closure (the resolver is free to move numpy here)
    run(pip + ["install", *find_links, str(core)])
    # 3. plugin payload only
    run(pip + ["install", "--no-deps", str(plugin)])
    # 4. contract pin re-applied after the closure resolved the numpy question
    run(pip + ["install", "--no-deps", f"numpy=={contract['numpy']}"])

    # 5. gate
    check = run(pip + ["check"], check=False, quiet=True, env=clean_env())
    lines = [
        line for line in (check.stdout + check.stderr).splitlines() if line.strip()
    ]
    version_code = (
        "import importlib.metadata as m; print(m.version('opencv-python-headless'))"
    )
    opencv_version = run(
        [python, "-c", version_code], quiet=True, env=clean_env()
    ).stdout.strip()
    expected = args.expected_conflict.format(opencv=opencv_version)
    unexpected = [line for line in lines if expected not in line]
    if unexpected or len(lines) != 1:
        print("pip check output:", json.dumps(lines, indent=1), file=sys.stderr)
        raise SystemExit(
            f"dependency gate FAILED: expected exactly one line matching {expected!r}"
        )
    print(f"dependency gate PASS: only the registered line is present\n  {lines[0]}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
