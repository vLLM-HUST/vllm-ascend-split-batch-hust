#!/usr/bin/env python3
"""Snapshot the repo manifest for one deployment instance.

A candidate manifest is a *deployment instance snapshot*, not a second version
declaration: it may only relocate the interpreter that serves and runs the
contract check.  Every contract value is copied from ``stack-versions.json``,
and the source manifest hash is recorded so ``--verify`` can report the snapshot
as stale once the contract moves.

``--verify`` is a staleness check only; it does not re-answer whether the
declared interpreter actually satisfies the contract.  Run
``tools/stack_versions.py --manifest <snapshot> --check`` for that.
"""

import argparse
import hashlib
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

import stack_versions

SNAPSHOT_SCHEMA = "vllm-hust.stack-versions.candidate.v1"
LOCATION_KEYS = ("serving_python", "venv")


def sha256_file(path):
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def build_snapshot(source, serving_python, venv=None):
    manifest = stack_versions.load_manifest(source)
    snapshot = json.loads(json.dumps(manifest))
    if serving_python is not None:
        snapshot["locations"]["serving_python"] = serving_python
    if venv is not None:
        snapshot["locations"]["venv"] = venv
    snapshot["snapshot"] = {
        "schema": SNAPSHOT_SCHEMA,
        "source_manifest": str(Path(source).resolve()),
        "source_manifest_sha256": sha256_file(source),
        "overridden": sorted(
            k
            for k in LOCATION_KEYS
            if snapshot["locations"][k] != manifest["locations"][k]
        ),
    }
    if snapshot["contract"] != manifest["contract"]:
        raise SystemExit(
            "refusing to emit a snapshot whose contract differs from the source"
        )
    return snapshot


def verify(snapshot_path, source):
    snapshot = json.loads(Path(snapshot_path).read_text())
    recorded = (snapshot.get("snapshot") or {}).get("source_manifest_sha256")
    current = sha256_file(source)
    if recorded is None:
        print(f"FAIL {snapshot_path}: no recorded source manifest hash")
        return 1
    if recorded != current:
        print(
            f"FAIL {snapshot_path}: snapshot is stale, source manifest sha256 "
            f"{recorded} -> {current}; regenerate the snapshot"
        )
        return 1
    print(f"PASS {snapshot_path}: snapshot matches source manifest {current[:16]}...")
    return 0


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", default=str(stack_versions.DEFAULT_MANIFEST))
    parser.add_argument("--serving-python")
    parser.add_argument("--venv")
    parser.add_argument("--out")
    parser.add_argument("--verify")
    args = parser.parse_args(argv)

    if args.verify:
        return verify(args.verify, args.source)

    if not args.out:
        parser.error("--out is required unless --verify is given")
    if args.serving_python is None and args.venv is None:
        parser.error("nothing to snapshot: pass --serving-python and/or --venv")

    snapshot = build_snapshot(args.source, args.serving_python, args.venv)
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(snapshot, indent=2) + "\n")
    provenance = snapshot["snapshot"]
    print(f"wrote {out}")
    print(
        f"  source {provenance['source_manifest']} "
        f"sha256={provenance['source_manifest_sha256'][:16]}..."
    )
    for key in provenance["overridden"]:
        print(f"  {key} -> {snapshot['locations'][key]}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
