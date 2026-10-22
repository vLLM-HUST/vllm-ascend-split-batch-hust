# SPDX-License-Identifier: Apache-2.0

"""The workbench probe classifies "unreachable" as absent, never as an error.

Regression guard for run 39437993180: a skip guard written as
``Path('/root/...').exists()`` raised ``PermissionError`` on the unprivileged
runner, so 38 items errored on the 3.10/3.12 legs and skipped on 3.14.
"""

from pathlib import Path

from _deployment import deployed_dir, deployed_file


def test_unreadable_parent_is_absent_not_an_error(monkeypatch):
    def denied(self, *args, **kwargs):
        raise PermissionError(13, "Permission denied", str(self))

    monkeypatch.setattr(Path, "is_file", denied)
    assert deployed_file("/root/opt-ops/scripts/opt.py") is False


def test_untraversable_directory_is_absent_not_an_error(monkeypatch):
    def denied(self, *args, **kwargs):
        raise PermissionError(13, "Permission denied", str(self))

    monkeypatch.setattr(Path, "is_dir", denied)
    assert deployed_dir("/root/包体恢复/records/tree-patches") is False


def test_readable_file_is_deployed(tmp_path):
    target = tmp_path / "opt.py"
    target.write_text("")
    assert deployed_file(target) is True


def test_missing_file_is_absent(tmp_path):
    assert deployed_file(tmp_path / "opt.py") is False
