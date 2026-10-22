"""档② 原件守卫: D4 site-2 patch 必须仍可作用于其 orig 快照基线（CPU-only）。"""

import pathlib
import shutil
import subprocess

HERE = pathlib.Path(__file__).resolve().parent
PATCHES = HERE.parent / "host-proposals" / "D4-conv-hoist-site2" / "patches"

PAIR = (
    "20261021T1355Z_d4_conv_hoist_site2.gdn.py.orig",
    "20261021T1355Z_d4_conv_hoist_site2.gdn.py.patch",
)


def test_patch_applies_onto_its_orig_snapshot(tmp_path):
    work = tmp_path / PAIR[0]
    shutil.copy(PATCHES / PAIR[0], work)
    proc = subprocess.run(
        ["patch", "--dry-run", str(work), str(PATCHES / PAIR[1])],
        capture_output=True,
        text=True,
        timeout=60,
    )
    assert proc.returncode == 0, proc.stdout + proc.stderr
    assert "FAILED" not in proc.stdout


def test_patch_contents_anchor_expected_transforms():
    body = (PATCHES / PAIR[1]).read_text()
    assert "_PACKED_CONV_WEIGHT_NAME" in body
    removed = sum(
        1
        for line in body.splitlines()
        if (line.startswith("-")
                and "conv_weights_T = conv_weights.transpose(0, 1)" in line)
    )
    assert removed == 4
    assert "VLLM_HUST_D4_CONV_PACK" in body
