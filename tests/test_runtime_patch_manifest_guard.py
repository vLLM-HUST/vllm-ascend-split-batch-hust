"""运行树补丁退役守卫: 仓内不保留运行时打补丁工具与可执行层。"""

from __future__ import annotations

import json
import pathlib

import pytest
from _deployment import deployed_file

HERE = pathlib.Path(__file__).resolve().parent
RT = HERE.parent / "tools" / "runtime-tree"
TREE = pathlib.Path("/vllm-workspace/vllm-ascend/vllm_ascend")


def test_runtime_patch_tools_are_absent_from_repository():
    """运行时打补丁工具全部废弃，宿主修改下沉至宿主 fork (work/issue20-w3f2-host)。"""
    for name in (
        "apply-runtime-patches.sh",
        "portable_runtime.py",
        "patch-manifest.json",
    ):
        assert not (RT / name).exists(), f"{name} should be retired from plugin repo"


def test_seam_table_contains_husk_fingerprints():
    """C 形态: seam 表断言 husk 变换表存在（树应 stock）。"""
    d = json.loads((RT / "tree-seam-assertions.json").read_text())
    ids = {s["id"] for s in d["seams"]}
    husk_ids = (
        "husk-rsp-VLLM_HUST_D4_CONV_PACK",
        "husk-rsp-VLLM_HUST_GDN_UNPACK_VIEWS",
        "husk-rsp-VLLM_HUST_MOE_INIT_TRITON",
        "husk-rsp-VLLM_HUST_NZ_SKIP_GATE",
    )
    for fid in husk_ids:
        assert fid in ids, fid


@pytest.mark.host_tree
def test_live_tree_is_anchor_matchable_for_husk():
    """活守卫(C 形态严格版): 运行树必须与 husk 变换锚点逐条匹配（loader 可施变换）。"""
    from vllm_ascend_split_batch import runtime_source_patches as rsp

    ascend = "/vllm-workspace/vllm-ascend/vllm_ascend/"
    mod2rel = {
        rsp.GDN: "ops/gdn.py",
        rsp.TD: "ops/fused_moe/token_dispatcher.py",
        rsp.LIN: "ops/linear.py",
        getattr(rsp, "PLATFORM", ""): ascend + "platform.py",
        getattr(
            rsp, "CORE_RUNNER", ""
        ): "/vllm-workspace/vllm/vllm/v1/worker/gpu_model_runner.py",
        getattr(rsp, "NPU_RUNNER", ""): ascend + "worker/model_runner_v1.py",
    }
    mismatches = []
    merged: dict[str, list] = {}
    for _, module, transforms in rsp.TABLES:
        merged.setdefault(mod2rel[module], []).extend(transforms)
    if not all(deployed_file(TREE / rel) for rel in merged):
        pytest.skip("vllm-ascend runtime tree is not deployed")
    for rel, transforms in merged.items():
        src = (TREE / rel).read_text(encoding="utf-8", errors="replace")
        try:
            rsp.apply_transforms(src, transforms)
        except rsp.AnchorError as exc:
            mismatches.append(f"{rel}: {exc}")
    gdn_src = (TREE / "ops/gdn.py").read_text(encoding="utf-8", errors="replace")
    assert "ops/gdn.py" in merged
    out_gdn, _ = rsp.apply_transforms(gdn_src, merged["ops/gdn.py"])
    assert "empty_like" in out_gdn
