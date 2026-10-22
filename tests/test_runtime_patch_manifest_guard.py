"""运行树补丁栈守卫: MANIFEST 三新层 + seam 指纹 + patch 文件存在性。"""

import json
import pathlib

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



def test_manifest_retired_layers_absent():
    """2026-10-22 退役: 三层不得再进自动重放(并发重放曾反复还原危险 vintage)。
    载波唯一通道 = husk; 手动回滚件在 tree-patches/。"""
    text = (RT / "apply-runtime-patches.sh").read_text()
    entries = [ln for ln in text.splitlines()
               if ln.strip() and not ln.lstrip().startswith("#")]
    for retired in ("d4site2-views-combined|", "moe-init-triton|", "nz-skip-gate|"):
        assert not any(retired in ln for ln in entries), (
            f"退役层仍在 MANIFEST: {retired}")


def test_seam_table_contains_husk_fingerprints():
    """C 形态: seam 表断言 husk 变换表存在（树应 stock）。"""
    d = json.loads((RT / "tree-seam-assertions.json").read_text())
    ids = {s["id"] for s in d["seams"]}
    husk_ids = ("husk-rsp-VLLM_HUST_D4_CONV_PACK",
                "husk-rsp-VLLM_HUST_GDN_UNPACK_VIEWS",
                "husk-rsp-VLLM_HUST_MOE_INIT_TRITON",
                "husk-rsp-VLLM_HUST_NZ_SKIP_GATE")
    for fid in husk_ids:
        assert fid in ids, fid


def test_patch_files_exist_in_pdir():
    for _, _, orig, patch in NEW_LAYERS:
        assert (PDIR / orig).is_file(), orig
        assert (PDIR / patch).is_file(), patch
        assert (PDIR / patch).stat().st_size > 200, patch


def test_live_tree_is_anchor_matchable_for_husk():
    """活守卫(C 形态严格版): 运行树必须与 husk 变换锚点逐条匹配（loader 可施变换）。

    2026-10-22 教训: 重启同步源带回旧 vintage 树内补丁（views 无 contiguity 守卫），
    env 开启即 capture 期崩（l2norm strided view）。树带补丁≠安全态——唯一安全态
    是锚点可匹配（即 loader 语义下的 stock），任何锚点失配都会让载波静默失效。"""
    from vllm_ascend_split_batch import runtime_source_patches as rsp
    ascend = "/vllm-workspace/vllm-ascend/vllm_ascend/"
    mod2rel = {
        rsp.GDN: "ops/gdn.py",
        rsp.TD: "ops/fused_moe/token_dispatcher.py",
        rsp.LIN: "ops/linear.py",
        getattr(rsp, "PLATFORM", ""): ascend + "platform.py",
        getattr(rsp, "CORE_RUNNER", ""):
            "/vllm-workspace/vllm/vllm/v1/worker/gpu_model_runner.py",
        getattr(rsp, "NPU_RUNNER", ""): ascend + "worker/model_runner_v1.py",
    }
    for tok in ("VLLM_HUST_D4_CONV_PACK", "VLLM_HUST_GDN_UNPACK_VIEWS",
                "VLLM_HUST_MOE_INIT_TRITON", "VLLM_HUST_NZ_SKIP_GATE"):
        assert tok in (HERE.parent / "src" / "vllm_ascend_split_batch" /
                       "runtime_source_patches.py").read_text(), f"husk 变换表缺 {tok}"
    mismatches = []
    merged: dict[str, list] = {}
    for _, module, transforms in rsp.TABLES:
        merged.setdefault(mod2rel[module], []).extend(transforms)
    for rel, transforms in merged.items():
        src = (TREE / rel).read_text(encoding="utf-8", errors="replace")
        try:
            rsp.apply_transforms(src, transforms)
        except rsp.AnchorError as exc:
            mismatches.append(f"{rel}: {exc}")
    # C-ON 验证态钉子(1021-182645 L3 8/8): views 变换的 empty_like 恒 2
    gdn_src = (TREE / "ops/gdn.py").read_text(encoding="utf-8", errors="replace")
    assert "ops/gdn.py" in merged
    out_gdn, _ = rsp.apply_transforms(gdn_src, merged["ops/gdn.py"])
    assert out_gdn.count("torch.empty_like(mixed_qkv_non_spec)") == 3, (
        "views 验证态漂移(C-ON 运行时: stock prefill 回退 1 + views 2)")
    assert not mismatches, (
        "运行树锚点失配 -> loader fail-closed 回 stock, 载波失效"
        "（旧 vintage 树内补丁残留? 恢复: 用 tree-patches 对应 .orig 覆盖）: "
        + "; ".join(mismatches[:4]))
