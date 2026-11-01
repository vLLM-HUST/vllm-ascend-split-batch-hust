"""运行树补丁退役守卫: 仓内不保留运行时打补丁工具与可执行层 (issue #33 方案 A)。"""

from __future__ import annotations

import json
import pathlib

HERE = pathlib.Path(__file__).resolve().parent
RT = HERE.parent / "tools" / "runtime-tree"
SRC = HERE.parent / "src" / "vllm_ascend_split_batch"

#: 随 issue #33 方案 A 整体退役的模块。
#: ``moe_init_triton.py`` 是纯 Triton 算子, 但其唯一仓内消费点是
#: ``runtime_source_patches`` 注入 token_dispatcher 的分支, 随宿主迁移一并退役。
RETIRED_MODULES = (
    "runtime_source_patches.py",
    "runtime_source_patches_pth.py",
    "moe_init_triton.py",
)


def test_runtime_patch_tools_are_absent_from_repository():
    """运行时打补丁工具全部废弃，宿主修改下沉至宿主 fork (work/issue20-w3f2-host)。"""
    for name in (
        "apply-runtime-patches.sh",
        "portable_runtime.py",
        "patch-manifest.json",
    ):
        assert not (RT / name).exists(), f"{name} should be retired from plugin repo"


def test_runtime_source_patch_modules_are_absent():
    """meta-path 源码变换模块已整体删除, 不得再随包发布。"""
    for name in RETIRED_MODULES:
        assert not (SRC / name).exists(), f"{name} should be retired (issue #33)"


def test_gate_fp32_plugin_no_longer_arms_source_patches():
    """general_plugins 入口不得在加载期安装任何源码变换钩子 (issue #33 危害 5)。"""
    text = (SRC / "gate_fp32_plugin.py").read_text(encoding="utf-8")
    assert "runtime_source_patches" not in text
    load_src = text.split("def load(", 1)[-1]
    assert load_src.count("install()") == 1, "load() 只能在门控分支内调用一次 install()"


def test_seam_table_has_no_runtime_patch_entries():
    """seam 断言表不再登记已退役的源码变换条目。"""
    table = json.loads((RT / "tree-seam-assertions.json").read_text())
    for seam in table["seams"]:
        assert not seam["id"].startswith("husk-rsp-"), seam["id"]
        assert "runtime_source_patches" not in seam.get("consumer", "")
