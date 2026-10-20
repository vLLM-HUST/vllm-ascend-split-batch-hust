# SPDX-License-Identifier: Apache-2.0

import importlib.util
import json
from pathlib import Path
from types import SimpleNamespace

import pytest

OPT = Path("/root/opt-ops/scripts/opt.py")


@pytest.fixture
def opt(monkeypatch):
    if not OPT.exists():
        pytest.skip("opt-ops deployment not installed")
    monkeypatch.syspath_prepend(str(OPT.parent))
    spec = importlib.util.spec_from_file_location("opt_delivery_test", OPT)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    if not hasattr(mod, "delivery_view"):
        pytest.skip("delivery integration patch not deployed")
    return mod


def test_missing_helper_is_explicit_error(opt, monkeypatch, tmp_path):
    monkeypatch.setitem(opt.TOOLS, "bench", str(tmp_path / "bench-ladder.sh"))
    assert opt.delivery_view()["state"] == "ERROR"
    assert opt.dispatch(opt.build_parser().parse_args(["delivery"])) == 4


def test_invalid_helper_payload_is_error(opt, monkeypatch):
    monkeypatch.setattr(
        opt.subprocess,
        "run",
        lambda *a, **kw: SimpleNamespace(
            returncode=0,
            stdout="[]",
            stderr="",
        ),
    )
    assert opt.delivery_view()["state"] == "ERROR"


def test_delivery_dispatch_preserves_scope_and_rc(opt, monkeypatch):
    calls = []

    def run(argv, **kwargs):
        calls.append(argv)
        return SimpleNamespace(returncode=3)

    monkeypatch.setattr(opt, "run", run)
    args = opt.build_parser().parse_args(
        [
            "delivery",
            "--base",
            "a" * 40,
            "--path",
            "src",
            "--path",
            "tests",
            "--check",
            "--json",
        ]
    )
    assert opt.dispatch(args) == 3
    assert calls[0][2:] == [
        "--base",
        "a" * 40,
        "--path",
        "src",
        "--path",
        "tests",
        "--check",
        "--json",
    ]


def test_dirty_baseline_is_explicitly_forwarded(opt, monkeypatch):
    calls = []
    monkeypatch.setattr(
        opt,
        "run",
        lambda argv, **kw: calls.append(argv) or SimpleNamespace(returncode=0),
    )
    args = opt.build_parser().parse_args(
        [
            "delivery",
            "--base",
            "b" * 40,
            "--path",
            "shared.py",
            "--dirty-baseline",
            "/tmp/start.json",
            "--check",
        ]
    )
    assert opt.dispatch(args) == 0
    assert calls[0][-3:] == ["--dirty-baseline", "/tmp/start.json", "--check"]


def test_note_omitted_body_is_not_forwarded_as_empty(opt, monkeypatch):
    calls = []
    monkeypatch.setattr(
        opt,
        "run",
        lambda argv, **kw: calls.append(argv) or SimpleNamespace(returncode=0),
    )
    args = opt.build_parser().parse_args(
        [
            "note",
            "update",
            "--id",
            "test_card",
            "--route",
            "028/test",
        ]
    )
    opt.dispatch(args)
    assert "--body" not in calls[-1]
    args = opt.build_parser().parse_args(
        [
            "note",
            "update",
            "--id",
            "test_card",
            "--route",
            "028/test",
            "--body",
            "",
        ]
    )
    opt.dispatch(args)
    assert calls[-1][-2:] == ["--body", ""]


def test_status_payload_reads_versioned_helper(opt):
    report = opt.delivery_view()
    assert report["state"] == "UNSCOPED"
    assert report["remote"] == "NOT_VERIFIED"
    assert report["repo"] == str(Path(__file__).resolve().parents[1])
    assert len(report["contract"]["separate_states"]) == 4
    assert json.loads(json.dumps(report)) == report
