# SPDX-License-Identifier: Apache-2.0

import hashlib
import importlib.util
import json
import subprocess
from pathlib import Path

import pytest
from _deployment import deployed_file


@pytest.fixture
def kb(tmp_path, monkeypatch):
    source = Path("/root/opt-ops/scripts/kbctl.py")
    if not deployed_file(source):
        pytest.skip("opt-ops not deployed")
    spec = importlib.util.spec_from_file_location("kb_repair_test", source)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    for key, path in {
        "ROOT": tmp_path,
        "KB": tmp_path / "kb",
        "CARD_BASE": tmp_path / "cards",
        "HIST": tmp_path / "history",
        "ROUTES": tmp_path / "routes",
        "NOTES_INDEX": tmp_path / "kb/INDEX.json",
        "PIN": tmp_path / "pin.json",
    }.items():
        monkeypatch.setattr(mod, key, str(path))
    (tmp_path / "routes/028/test").mkdir(parents=True)
    return mod


def note(kb, action="new", *extra):
    args = kb.build_parser().parse_args(
        [
            "note",
            action,
            "--id",
            "test_card",
            "--route",
            "028/test",
            *extra,
        ]
    )
    return kb.cmd_note(args)


@pytest.mark.parametrize("action", ["new", "update", "supersede"])
def test_missing_body_rejected_before_any_write(kb, tmp_path, action):
    before = set(tmp_path.rglob("*"))
    with pytest.raises(SystemExit) as error:
        note(kb, action)
    assert error.value.code == 2
    assert set(tmp_path.rglob("*")) == before


def test_update_preserves_body_and_rejects_blank(kb):
    assert note(kb, "new", "--body", "Evidence-backed body") == 0
    path = Path(kb.card_path("test_card", "028/test"))
    assert note(kb, "update", "--confidence", "0.9") == 0
    assert "Evidence-backed body" in path.read_text()
    before = path.read_bytes()
    with pytest.raises(SystemExit):
        note(kb, "update", "--body", "  ")
    assert path.read_bytes() == before
    assert (
        json.loads(Path(kb.sidecar_path("test_card")).read_text())["confidence"] == 0.9
    )


def test_wrong_route_does_not_create_empty_card(kb):
    note(kb, "new", "--body", "Keep this body")
    args = kb.build_parser().parse_args(
        [
            "note",
            "update",
            "--id",
            "test_card",
            "--route",
            "028/typo",
            "--body",
            "Should not move the card",
        ]
    )
    with pytest.raises(SystemExit):
        kb.cmd_note(args)
    assert not Path(kb.card_path("test_card", "028/typo")).exists()


def test_pin_rejects_stdout_inside_archive_before_writing(kb, tmp_path, monkeypatch):
    monkeypatch.setattr(kb, "ARCHIVE_ROOTS", [str(tmp_path / "archive")])
    monkeypatch.setattr(
        kb.os, "readlink", lambda path: str(tmp_path / "archive/pin.log")
    )
    with pytest.raises(SystemExit) as error:
        kb.cmd_pin(None)
    assert error.value.code == 2
    assert not Path(kb.PIN).exists()


def test_deployed_patch_roundtrip(tmp_path):
    proposal = (
        Path(__file__).resolve().parents[1] / "host-proposals/OPT-toolchain-repair"
    )
    metadata = json.loads((proposal / "proposal_meta.json").read_text())
    for row in metadata["files"]:
        source = Path("/root") / row["target"]
        if not deployed_file(source):
            pytest.skip("opt-ops deployment not installed")
        data = source.read_bytes()
        assert hashlib.sha256(data).hexdigest() == row["after_sha256"]
        target = tmp_path / row["target"]
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_bytes(data)
    patch = (proposal / "patch.diff").read_bytes()
    for reverse, key in [(True, "before_sha256"), (False, "after_sha256")]:
        command = ["patch", "-p1", "-d", str(tmp_path)]
        if reverse:
            command.append("-R")
        subprocess.run(command, input=patch, check=True, capture_output=True)
        for row in metadata["files"]:
            data = (tmp_path / row["target"]).read_bytes()
            assert hashlib.sha256(data).hexdigest() == row[key]
