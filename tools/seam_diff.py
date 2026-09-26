#!/usr/bin/env python3
# Copyright (c) 2025-2026 Huawei Technologies Co., Ltd. All Rights Reserved.
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""宿主升级核对清单（docs/release.md §4）的可执行前半程：两棵树对拍接缝。

**为什么需要它**：本插件的能力都挂在宿主**私有方法**上（`docs/pitfalls.md` §2.1），
历史上签名漂移真炸过一次（`ddc0120` 修的 D1–D4）。`release.md` §4 那份清单以前靠人眼
逐项读；本脚本把它变成机械判定，而且**不需要安装宿主**（纯 AST，不 import），
因此可以在只有一份已安装宿主的环境里，对拍"已钉版本"与"待升版本"两棵树。

对拍三件事：

1. **签名**：接缝符号是否仍存在、参数表（含 `*args`/`**kwargs`/默认值个数）是否一致；
2. **调用点**：宿主是否仍在调这些方法。文件集合层面比对；`_capture_cudagraphs`
   这类由父类调用的，在 vllm-ascend 树里为 0 属正常，要拿 core 树再对一次；
3. **结构面**：新树独有的顶层目录（提示大重构）。

用法::

    git worktree add --detach /tmp/new-host <待升 commit>   # 不动主树
    python3 tools/seam_diff.py <已钉的宿主树> <待升的宿主树>
    # core 与 vllm-ascend 各跑一次（接缝分属两个仓）

退出码：0 = 全部一致；1 = 有签名变化/符号缺失/调用点消失（升级需要处理）。
"""

from __future__ import annotations

import argparse
import ast
import pathlib
import sys

#: 接缝清单 = docs/release.md §4 的逐项。键是仓内相对路径，值是必须存在的符号。
#: 两个仓共用本表；不属于该仓的路径会被判为"文件缺失"并跳过（见 --repo 过滤）。
SEAMS: dict[str, tuple[str, ...]] = {
    "vllm/v1/cudagraph_dispatcher.py": (
        "add_cudagraph_key",
        "dispatch",
    ),
    "vllm/v1/worker/gpu_model_runner.py": (
        "_capture_cudagraphs",
        "_warmup_and_capture",
        "_determine_batch_execution_and_padding",
        "_model_forward",
        "_update_full_graph_params_if_needed",
    ),
    "vllm_ascend/worker/model_runner_v1.py": (
        "_capture_cudagraphs",
        "_warmup_and_capture",
        "_determine_batch_execution_and_padding",
        "_model_forward",
        "_update_full_graph_params_if_needed",
    ),
    "vllm_ascend/attention/attention_v1.py": (
        "update_full_graph_params",
        "forward_fused_infer_attention",
        "AscendAttentionMetadataBuilder",
        "AscendAttentionBackendImpl",
    ),
    "vllm_ascend/compilation/acl_graph.py": (
        "get_graph_params",
        "update_graph_params_workspaces",
        "update_full_graph_params",
        "ACLGraphWrapper",
    ),
    "vllm_ascend/compilation/compiler_interface.py": (
        "_disable_pytorch_aot_cache_for_npugraph_ex",
        "AscendCompiler",
    ),
    "vllm_ascend/ops/rotary_embedding.py": ("triton_mrope",),
}

#: 调用点检查的目标（与 SEAMS 的符号并集，去掉类名）。
CALL_TARGETS = tuple(
    sorted(
        {name for names in SEAMS.values() for name in names if not name[:1].isupper()}
    )
)

#: 名字太通用（`dispatch` 之类）→ 只在**接收者**表达式含该子串时才算作接缝调用。
#: 实测假阳性：`dispatch` 在 core 树里从 22 涨到 35 个调用点，多出来的全是
#: `model_executor/kernels/mhc/*` 与 `models/deepseek_v4/**` 里同名的**无关**方法
#: （接收者也叫 `self.dispatch`，所以只比接收者还不够）。
CALL_RECEIVER_FILTER: dict[str, str] = {
    "dispatch": "cudagraph",
    "add_cudagraph_key": "cudagraph",
    "update_graph_params": "acl_graph",
}


def _signature(node: ast.AST) -> str:
    """把一个函数/类节点渲染成可比对的规范串。"""
    if isinstance(node, ast.ClassDef):
        return f"class {node.name}"
    if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
        a = node.args
        parts = list(x.arg for x in a.posonlyargs)
        if a.posonlyargs:
            parts.append("/")
        parts += [x.arg for x in a.args]
        if a.vararg:
            parts.append("*" + a.vararg.arg)
        elif a.kwonlyargs:
            parts.append("*")
        parts += [x.arg for x in a.kwonlyargs]
        if a.kwarg:
            parts.append("**" + a.kwarg.arg)
        return f"{node.name}({', '.join(parts)})  [默认值 {len(a.defaults)}]"
    return type(node).__name__


def _symbols(root: pathlib.Path, rel: str, names: tuple[str, ...]) -> dict[str, str]:
    path = root / rel
    if not path.is_file():
        return dict.fromkeys(names, "<文件缺失>")
    tree = ast.parse(path.read_text(encoding="utf-8", errors="replace"))
    found: dict[str, str] = {}
    for parent in ast.walk(tree):
        for child in ast.iter_child_nodes(parent):
            if (
                isinstance(child, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef))
                and child.name in names
                and child.name not in found
            ):
                found[child.name] = _signature(child)
    return {n: found.get(n, "<未找到>") for n in names}


def _calls(root: pathlib.Path) -> list[tuple[str, str, str, int, tuple[str, ...]]]:
    """扫出 (文件, 方法名, 接收者, 位置参数个数, 关键字参数名元组)。

    **关键字参数是重点**：`ddc0120` 修的 D1–D4 就是这一类漂移
    （`_capture_cudagraphs` 少了 `profiler`、`update_full_graph_params` 多了
    `num_dcp_pcp_tokens`、`_update_full_graph_params_if_needed` 多了 `positions`）——
    宿主的**方法签名**可能没变，但**调用点传的关键字**变了，插件包装器就接不上。
    这类变化只有比对调用点才看得出来。

    **接收者**计入键是为了消歧：`self.cudagraph_dispatcher.dispatch` 与
    `kernels.mhc.dispatch` 不是一回事。
    """
    found: list[tuple[str, str, str, int, tuple[str, ...]]] = []
    for path in root.rglob("*.py"):
        parts = path.parts
        if "tests" in parts or path.name.startswith("test_") or "csrc" in parts:
            continue
        try:
            tree = ast.parse(path.read_text(encoding="utf-8", errors="replace"))
        except SyntaxError:
            continue
        rel = str(path.relative_to(root))
        for node in ast.walk(tree):
            if not isinstance(node, ast.Call):
                continue
            func = node.func
            if isinstance(func, ast.Attribute):
                name, receiver = func.attr, ast.unparse(func.value)
            elif isinstance(func, ast.Name):
                name, receiver = func.id, "<bare>"
            else:
                continue
            if name in CALL_TARGETS:
                needle = CALL_RECEIVER_FILTER.get(name)
                if needle and needle not in (receiver + rel):
                    continue
                kw = tuple(sorted(k.arg for k in node.keywords if k.arg))
                found.append((rel, name, receiver, len(node.args), kw))
    return found


def _selftest() -> int:
    """判别力自证：合成一对小树，注入一次 D1 类漂移，断言工具能抓到。

    与 `tests/test_rope_fix_drift.py` 的反向断言同一思路：**守卫必须先证明它抓得住**，
    否则"全一致"可能只是因为脚本什么都没检查。
    """
    import tempfile

    body_ok = (
        "def call(self, descs, mode, profiler):\n"
        "    self._capture_cudagraphs(\n"
        "        batch_descriptors=descs,\n"
        "        cudagraph_runtime_mode=mode,\n"
        "        profiler=profiler,\n"
        "    )\n\n"
        "def _capture_cudagraphs(\n"
        "    self, batch_descriptors, cudagraph_runtime_mode, profiler\n"
        "):\n"
        "    pass\n"
    )
    body_drift = body_ok.replace("        profiler=profiler,\n", "")

    with tempfile.TemporaryDirectory() as tmp:
        base = pathlib.Path(tmp)
        for name, text in (
            ("old", body_ok),
            ("new_ok", body_ok),
            ("new_bad", body_drift),
        ):
            d = base / name / "vllm" / "v1"
            d.mkdir(parents=True)
            (d / "worker").mkdir()
            (d / "worker" / "gpu_model_runner.py").write_text(text, encoding="utf-8")

        pick = lambda root: (  # noqa: E731 -- 只在本函数里用一次
            root / "vllm" / "v1" / "worker" / "gpu_model_runner.py"
        ).read_text(encoding="utf-8")

        def run(a: pathlib.Path, b: pathlib.Path) -> int:
            import contextlib
            import io

            buf = io.StringIO()
            with contextlib.redirect_stdout(buf):
                rc = main([str(a), str(b), "--repo", "core"])
            return rc

        same = run(base / "old", base / "new_ok")
        drifted = run(base / "old", base / "new_bad")
        print(f"  [selftest] 未漂移 -> rc={same}（期望 0）")
        print(f"  [selftest] 删掉 profiler 关键字 -> rc={drifted}（期望 1）")
        assert pick(base / "old") != pick(base / "new_bad"), "合成树没造对"
        return 0 if (same == 0 and drifted == 1) else 1


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("old", nargs="?", type=pathlib.Path, help="已钉版本的宿主树")
    parser.add_argument("new", nargs="?", type=pathlib.Path, help="待升版本的宿主树")
    parser.add_argument(
        "--repo",
        choices=["auto", "core", "ascend"],
        default="auto",
        help="只对属于该仓的路径做判定（auto=按文件是否存在推断）",
    )
    parser.add_argument(
        "--selftest",
        action="store_true",
        help="用合成小树证明本脚本抓得住一类已知漂移（不读真实宿主树）",
    )
    args = parser.parse_args(argv)
    if args.selftest:
        print("== seam_diff 判别力自证 ==")
        return _selftest()
    if args.old is None or args.new is None:
        parser.error("需要两个位置参数（或改用 --selftest）")

    old_root, new_root = args.old.resolve(), args.new.resolve()
    if not old_root.is_dir() or not new_root.is_dir():
        parser.error("两个参数都必须是存在的目录")
    print(f"旧树 {old_root}\n新树 {new_root}\n")

    changed = missing = 0
    print("== 1. 签名 ==")
    for rel, names in SEAMS.items():
        if args.repo != "auto" and args.repo == "core" and not rel.startswith("vllm/"):
            continue
        if args.repo == "ascend" and not rel.startswith("vllm_ascend/"):
            continue
        a, b = _symbols(old_root, rel, names), _symbols(new_root, rel, names)
        if a == b:
            if b[names[0]] == "<文件缺失>":
                print(f"[跳过] {rel}（不在本仓）")
                continue
            print(f"[一致] {rel}")
            for n in names:
                print(f"        {b[n]}")
            continue
        print(f"[差分] {rel}")
        for n in names:
            if a[n] == b[n]:
                continue
            print(f"        {n}\n          旧: {a[n]}\n          新: {b[n]}")
            if b[n] in ("<文件缺失>", "<未找到>"):
                missing += 1
            else:
                changed += 1

    print("\n== 2. 调用点（接收者 + 位置参数个数 + 关键字参数） ==")
    old_rows = _calls(old_root)
    new_rows = _calls(new_root)
    # 键含接收者与参数形状，所以"同一处调用改了参数"与"新增/消失一处调用"都能看出来。
    old_keys = {(rel, name, recv, nargs, kw) for rel, name, recv, nargs, kw in old_rows}
    new_keys = {(rel, name, recv, nargs, kw) for rel, name, recv, nargs, kw in new_rows}
    for target in CALL_TARGETS:
        old_t = {k for k in old_keys if k[1] == target}
        new_t = {k for k in new_keys if k[1] == target}
        if old_t == new_t:
            print(f"[一致] {target}: {len(old_t)} 个调用点")
            continue
        print(f"[变化] {target}: {len(old_t)} -> {len(new_t)} 点")
        for key in sorted(old_t - new_t):
            rel, _name, recv, nargs, kw = key
            print(f"        仅旧树: {rel}  {recv}.{target}(pos={nargs}, kw={list(kw)})")
        for key in sorted(new_t - old_t):
            rel, _name, recv, nargs, kw = key
            print(f"        仅新树: {rel}  {recv}.{target}(pos={nargs}, kw={list(kw)})")
        changed += len(old_t ^ new_t)

    print("\n== 3. 结构面（新树独有目录） ==")
    old_dirs = {str(d.relative_to(old_root)) for d in old_root.rglob("*") if d.is_dir()}
    new_dirs = {str(d.relative_to(new_root)) for d in new_root.rglob("*") if d.is_dir()}
    fresh = sorted(
        d
        for d in new_dirs - old_dirs
        if d.count("/") <= 2 and ".git" not in d and "__pycache__" not in d
    )
    if not fresh:
        print("（无）")
    for d in fresh:
        print(f"    + {d}")

    print(f"\n合计：签名变化 {changed} 项、符号缺失 {missing} 项")
    print(
        "注意：本脚本只证明**静态面**一致。方法体语义（对象如何被构造、描述符如何被填充）"
        "必须靠真机冒烟，见 docs/release.md §4 后半段。"
    )
    return 1 if (changed or missing) else 0


if __name__ == "__main__":
    sys.exit(main())
