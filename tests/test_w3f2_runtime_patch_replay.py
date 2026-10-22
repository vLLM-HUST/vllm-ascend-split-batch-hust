"""档② 原件守卫: W3F2/W3F3 patch 必须仍可作用于其 orig 快照基线（CPU-only）。"""

import pathlib
import shutil
import subprocess

HERE = pathlib.Path(__file__).resolve().parent
PATCHES = HERE.parent / "host-proposals" / "W3F2-runtime-tree" / "patches"

PAIRS = [
    (
        "20261015T1720_W3F2-runtime-tree.gdn.py.orig",
        "20261015T1720_W3F2-runtime-tree.gdn.py.patch",
    ),
    (
        "20261015T1730_W3F2-runtime-tree.model_runner_v1.py.orig",
        "20261015T1730_W3F2-runtime-tree.model_runner_v1.py.patch",
    ),
    (
        "20261015T1731_W3F2-runtime-tree.patch_qwen3_5.py.orig",
        "20261015T1731_W3F2-runtime-tree.patch_qwen3_5.py.patch",
    ),
    (
        "20261016T1130_W3F3-runtime-tree.shared_experts.py.orig",
        "20261016T1130_W3F3-runtime-tree.shared_experts.py.patch",
    ),
]


def test_patch_applies_onto_its_orig_snapshot(tmp_path):
    for orig, patch in PAIRS:
        work = tmp_path / orig
        shutil.copy(PATCHES / orig, work)
        proc = subprocess.run(
            ["patch", "--dry-run", str(work), str(PATCHES / patch)],
            capture_output=True,
            text=True,
            timeout=60,
        )
        assert proc.returncode == 0, f"{patch}: {proc.stdout}{proc.stderr}"
