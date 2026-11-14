#!/usr/bin/env python3
"""Lightweight custody and anti-drift fingerprinting for operator delivery."""

from __future__ import annotations

import hashlib
import os
import re


def calculate_dir_fingerprint(dir_path: str) -> dict[str, str]:
    """Calculate sha256 checksums of all evidence files in an operator directory."""
    fingerprints: dict[str, str] = {}
    if os.path.islink(dir_path):
        raise ValueError("Symlink delivery directory")
    if not os.path.isdir(dir_path):
        return fingerprints

    def fail(error):
        raise error

    for root, directories, files in os.walk(dir_path, onerror=fail):
        for name in directories + files:
            if os.path.islink(os.path.join(root, name)):
                raise ValueError(f"Symlink in delivery directory: {name}")
        directories[:] = sorted(d for d in directories if d != "__pycache__")
        for f in sorted(files):
            full_p = os.path.join(root, f)
            rel_p = os.path.relpath(full_p, dir_path)
            if rel_p == "checksum.sha256" or f.endswith((".pyc", ".pyo")):
                continue
            with open(full_p, "rb") as fp:
                fingerprints[rel_p] = hashlib.sha256(fp.read()).hexdigest()

    return fingerprints


def write_dir_checksum(dir_path: str) -> str:
    """Generate and write a checksum.sha256 file inside the operator directory."""
    fps = calculate_dir_fingerprint(dir_path)
    if not fps:
        raise ValueError("Cannot fingerprint an empty delivery directory")
    checksum_file = os.path.join(dir_path, "checksum.sha256")
    lines = [f"{sha}  {fname}" for fname, sha in sorted(fps.items())]
    with open(checksum_file, "w", encoding="utf-8") as f:
        f.write("\n".join(lines) + "\n")
    return checksum_file


def verify_dir_checksum(dir_path: str) -> tuple[bool, str]:
    """Verify that current files match recorded checksums (anti-drift gate)."""
    checksum_file = os.path.join(dir_path, "checksum.sha256")
    if not os.path.isfile(checksum_file):
        return False, f"Missing checksum file: {checksum_file}"

    recorded: dict[str, str] = {}
    try:
        with open(checksum_file, encoding="utf-8") as f:
            for line in f:
                match = re.fullmatch(r"([a-f0-9]{64})  (.+)\n?", line)
                if not match:
                    return False, "Malformed checksum entry"
                digest, name = match.groups()
                if name in recorded or os.path.isabs(name) or ".." in name.split("/"):
                    return False, f"Duplicate or unsafe checksum entry: {name}"
                recorded[name] = digest
        if not recorded:
            return False, "Empty checksum manifest"
        current = calculate_dir_fingerprint(dir_path)
    except (OSError, UnicodeError, ValueError) as error:
        return False, f"Cannot read delivery files: {error}"
    if set(recorded) != set(current):
        return False, (
            f"Evidence drift: missing files {sorted(set(recorded) - set(current))}; "
            f"unlisted files {sorted(set(current) - set(recorded))}"
        )

    modified = [k for k in recorded if recorded[k] != current.get(k)]
    if modified:
        return False, f"Evidence drift: modified files {sorted(modified)}"

    return True, "All files matched checksums cleanly."
