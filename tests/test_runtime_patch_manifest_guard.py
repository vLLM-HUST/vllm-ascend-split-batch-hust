"""运行树补丁栈守卫: MANIFEST 三新层 + seam 指纹 + patch 文件存在性。"""

import json
import pathlib
import subprocess

HERE = pathlib.Path(__file__).resolve().parent
RT = HERE.parent / "tools" / "runtime-tree"
PDIR = pathlib.Path("/root/包体恢复/records/tree-patches")
TREE = pathlib.Path("/vllm-workspace/vllm-ascend/vllm_ascend")

NEW_LAYERS = [
    ("d4site2-views-combined", "ops/gdn.py",
     "20261021T1355Z_d4_conv_hoist_site2.orig-gdn.py",
     "20261021T1459Z_concatdViews.gdn.py.patch"),
    ("moe-init-triton", "ops/fused_moe/token_dispatcher.py",
     "20261021T1710Z_moeInitTriton.token_dispatcher.py.orig",
     "20261021T1710Z_moeInitTriton.token_dispatcher.py.patch"),
    ("nz-skip-gate", "ops/linear.py",
     "20261021T1545Z_nzSkipGate.linear.py.orig",
     "20261021T1545Z_nzSkipGate.linear.py.patch"),
]
FINGERPRINTS = [
    ("ops/gdn.py", "_PACKED_CONV_WEIGHT_NAME"),
    ("ops/gdn.py", "_UNPACK_VIEWS_ON"),
    ("ops/fused_moe/token_dispatcher.py", "_moe_init_routing_t1_kernel"),
    ("ops/linear.py", "VLLM_HUST_NZ_SKIP_GATE"),
]


def test_manifest_contains_new_layers():
    text = (RT / "apply-runtime-patches.sh").read_text()
    for name, rel, orig, patch in NEW_LAYERS:
        assert f"{name}|{rel}|{orig}|{patch}" in text, name


def test_seam_table_contains_fingerprints():
    d = json.loads((RT / "tree-seam-assertions.json").read_text())
    ids = {s["id"] for s in d["seams"]}
    for fid in ("d4site2-pack-fingerprint", "concatd-views-fingerprint",
                "moe-init-triton-fingerprint", "nz-skip-gate-fingerprint"):
        assert fid in ids, fid


def test_patch_files_exist_in_pdir():
    for _, _, orig, patch in NEW_LAYERS:
        assert (PDIR / orig).is_file(), orig
        assert (PDIR / patch).is_file(), patch
        assert (PDIR / patch).stat().st_size > 200, patch


def test_live_tree_carries_fingerprints():
    """活守卫: 并发重放/重灌抹层时此测试立刻红（fingerprints 缺失即 DRIFT）。"""
    missing = []
    for rel, token in FINGERPRINTS:
        src = (TREE / rel).read_text(encoding="utf-8", errors="replace")
        if token not in src:
            missing.append(f"{rel}:{token}")
    assert not missing, f"运行树补丁指纹缺失(被重灌/并发重放抹除): {missing}; 修复: bash /root/包体恢复/scripts/apply-runtime-patches.sh"
