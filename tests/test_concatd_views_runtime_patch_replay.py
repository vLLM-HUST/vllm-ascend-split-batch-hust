"""档② 原件守卫: ConcatD views 组合 patch 仍可作用于其 orig 快照（CPU-only）。"""

import pathlib
import shutil
import subprocess

HERE = pathlib.Path(__file__).resolve().parent
PATCHES = HERE.parent / "host-proposals" / "ConcatD-zero-copy-unpack" / "patches"
PAIR = ("20261021T1500Z_concatdViews.gdn.py.orig",
        "20261021T1500Z_concatdViews.gdn.py.patch")


def test_patch_applies_onto_its_orig_snapshot(tmp_path):
    work = tmp_path / PAIR[0]
    shutil.copy(PATCHES / PAIR[0], work)
    proc = subprocess.run(
        ["patch", "--dry-run", str(work), str(PATCHES / PAIR[1])],
        capture_output=True, text=True, timeout=60,
    )
    assert proc.returncode == 0, proc.stdout + proc.stderr
    assert "FAILED" not in proc.stdout


def test_patch_contents_anchor_expected_transforms():
    body = (PATCHES / PAIR[1]).read_text()
    assert "VLLM_HUST_GDN_UNPACK_VIEWS" in body
    assert "_unpack_mixed_qkv_views" in body
    assert body.count("torch.empty_like(mixed_qkv_non_spec)") == 3  # decode fallback + prefill 两分支
