# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 vLLM-HUST contributors.

"""Test layering: ``mirror`` vs ``contract`` (labels only, no files move).

Two kinds of test live in this directory and they fail for different reasons:

``contract``
    Runs this repository's code and asserts its behaviour.  A failure means the
    code changed behaviour.

``mirror``
    Reads in-repo artefacts (source, JSON, Markdown, workflow YAML) and asserts
    that a declaration still mirrors the thing it claims to describe -- plugin
    counts against ``pyproject.toml``, index labels against evidence, anchors
    against host source.  A failure means two places disagree and one of them
    must be updated.

Run one layer at a time with ``pytest -m mirror`` or ``pytest -m contract``.

The mirror list is explicit rather than pattern-matched so the classification
can be reviewed.  A file that is not listed is a ``contract`` test, and
``pytest_configure`` fails loudly if a listed file disappears -- a stale list
would otherwise silently shrink the static layer.
"""

from __future__ import annotations

import pathlib

import pytest

#: Files that only compare in-repo declarations against the artefacts they mirror.
MIRROR_TESTS = frozenset(
    {
        "test_ascend910b_static_audit.py",
        "test_benchmark_standard.py",
        "test_delivery_status_consistency.py",
        "test_distribution.py",
        "test_doc_consistency.py",
        "test_document_custody.py",
        "test_evidence_custody.py",
        "test_json_hygiene.py",
        "test_manifest.py",
        "test_ops_delivery_policy.py",
        "test_pitfall_guards.py",
        "test_provenance_hashes.py",
        "test_release_workflow.py",
        "test_repo_audit.py",
        "test_runtime_patch_manifest_guard.py",
        "test_verifier_layout.py",
        "test_zerocost_wiring_boundary.py",
    }
)


def pytest_configure(config: pytest.Config) -> None:
    """Refuse to run with a mirror list that no longer matches the tree."""
    tests_dir = pathlib.Path(__file__).resolve().parent
    missing = sorted(name for name in MIRROR_TESTS if not (tests_dir / name).is_file())
    if missing:
        raise pytest.UsageError(
            f"tests/conftest.py MIRROR_TESTS lists files that do not exist: {missing}"
        )


def pytest_collection_modifyitems(
    config: pytest.Config, items: list[pytest.Item]
) -> None:
    """Label every collected item with exactly one layer marker."""
    for item in items:
        name = pathlib.Path(str(item.path)).name
        marker = pytest.mark.mirror if name in MIRROR_TESTS else pytest.mark.contract
        item.add_marker(marker)
