# SPDX-License-Identifier: Apache-2.0

import os
import subprocess
from pathlib import Path

import pytest

SCRIPT = Path(__file__).resolve().parents[1] / "tools/bench-ladder.sh"


def select(window, rank=None):
    args = [str(window)] + ([] if rank is None else [str(rank)])
    return subprocess.run(
        [
            "bash",
            "-c",
            'source "$1" __src >/dev/null 2>&1; shift; profile_rank_dir "$@"',
            "bash",
            str(SCRIPT),
            *args,
        ],
        capture_output=True,
        text=True,
        timeout=30,
    )


def test_default_is_rank_zero_not_newest(tmp_path):
    zero = tmp_path / "tp0_rank0_123_ascend_pt"
    one = tmp_path / "tp1_rank1_456_ascend_pt"
    zero.mkdir()
    one.mkdir()
    os.utime(zero, (1, 1))
    os.utime(one, (2, 2))
    assert select(tmp_path).stdout.strip() == str(zero)
    assert select(tmp_path, 1).stdout.strip() == str(one)


@pytest.mark.parametrize("rank", ["bad", "-1", "1; echo unsafe"])
def test_invalid_rank_refused(tmp_path, rank):
    proc = select(tmp_path, rank)
    assert proc.returncode == 2
    assert not proc.stdout


def test_missing_rank_never_falls_back(tmp_path):
    (tmp_path / "tp1_rank1_123_ascend_pt").mkdir()
    proc = select(tmp_path)
    assert proc.returncode == 1
    assert "不按 mtime 回退" in proc.stderr


def test_duplicate_rank_is_ambiguous(tmp_path):
    for pid in (123, 456):
        (tmp_path / f"tp0_rank0_{pid}_ascend_pt").mkdir()
    proc = select(tmp_path)
    assert proc.returncode == 1
    assert not proc.stdout


def test_multirow_l0_dispatch_and_live_rank_wiring():
    source = SCRIPT.read_text()
    assert 'if [ "$op" = "ConcatDContiguous" ]' in source
    assert '"$concat_dir/bench_concatd_contiguous.py" "$@"' in source
    assert (
        'asc_dir=$(profile_rank_dir "$prof_dir" "${LADDER_PROFILE_RANK:-0}")' in source
    )
