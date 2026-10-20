#!/usr/bin/env python3
"""check-tree-seams.py —— 运行树「接口面」起服前置校验（纯静态 / 不 import / 不占卡 / 秒级）

背景（2026-10-16 11:53 事故）: 运行树里「调用点已改、定义点未改」时, `py_compile` 一律通过,
错误只能在起服后 2-5 min 由 Worker 抛出（AttributeError），一轮约 40min 浪费；
另一类是「载体依赖的宿主契约被上游改掉」——不报错、只静默失效（fail-open），更难发现。

两条判据（表见同目录 tree-seam-assertions.json）:
  kind=symbol : definer 文件里必须存在该符号（支持 `Class.method` 与顶层 `func`），
                且 consumer 文件里必须出现 call_token（调用点仍在调）。
  kind=text   : consumer 文件里必须出现 call_token（契约文案，用于静默失效防线）。

用法:
  check-tree-seams.py                       # 默认 tree=/vllm-workspace/vllm-ascend/vllm_ascend
  check-tree-seams.py --tree <dir> --table <json>
  check-tree-seams.py --check-import-path    # 另核 editable 安装映射是否指向被补丁的那棵树
  check-tree-seams.py --selftest             # 自证: 合成一次已知漂移必须报错(退出码 1)
退出码: 0 = 全过; 1 = 有断言失败; 2 = 用法/表错误。
"""
from __future__ import annotations

import argparse
import ast
import json
import os
import pathlib
import sys

DEFAULT_TREE = "/vllm-workspace/vllm-ascend/vllm_ascend"
DEFAULT_TABLE = os.path.join(os.path.dirname(os.path.abspath(__file__)), "tree-seam-assertions.json")
SITE = "/usr/local/python3.12.13/lib/python3.12/site-packages"


def _defined_symbols(path: pathlib.Path) -> tuple[set[str], set[str], bool]:
    """返回 (顶层函数名, 'Class.method' 名, 文件是否可解析)。"""
    src = path.read_text(encoding="utf-8", errors="replace")
    tree = ast.parse(src)          # 语法错误 → 抛给调用方，按失败处理
    funcs: set[str] = set()
    methods: set[str] = set()
    for node in ast.walk(tree):
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
            funcs.add(node.name)
            for parent in ast.walk(tree):
                if isinstance(parent, ast.ClassDef) and node in parent.body:
                    methods.add(f"{parent.name}.{node.name}")
    return funcs, methods, True


def _resolve(tree: pathlib.Path, rel: str) -> pathlib.Path:
    p = pathlib.Path(rel)
    return p if p.is_absolute() else tree / rel


def check(table: dict, tree: pathlib.Path) -> list[str]:
    fails: list[str] = []
    for seam in table.get("seams", []):
        sid = seam.get("id", "?")
        kind = seam.get("kind", "symbol")
        token = seam.get("call_token") or seam.get("symbol") or ""
        consumer = seam.get("consumer")
        if kind == "symbol":
            definer = _resolve(tree, seam["definer"])
            if not definer.is_file():
                fails.append(f"[{sid}] 定义点文件缺失: {definer}")
            else:
                try:
                    funcs, methods, _ = _defined_symbols(definer)
                except SyntaxError as ex:
                    fails.append(f"[{sid}] 定义点文件语法错误: {definer} ({ex})")
                    funcs, methods = set(), set()
                sym = seam["symbol"]
                ok = (sym in methods) if "." in sym else (sym in funcs)
                if not ok:
                    fails.append(f"[{sid}] 定义点缺符号 {sym} @ {seam['definer']}")
        if consumer:
            cpath = _resolve(tree, consumer)
            if not cpath.is_file():
                fails.append(f"[{sid}] 调用点/消费方文件缺失: {cpath}")
            elif token and token not in cpath.read_text(encoding="utf-8", errors="replace"):
                fails.append(f"[{sid}] 调用点未命中 {token!r} @ {consumer}")
    return fails


def check_import_path() -> list[str]:
    """editable 安装映射必须指向被打补丁的那棵树，否则补丁静默不生效。"""
    fails: list[str] = []
    if not os.path.isdir(SITE):
        return [f"site-packages 不存在: {SITE}"]
    finders = [f for f in os.listdir(SITE) if f.startswith("__editable___vllm_ascend") and f.endswith("_finder.py")]
    if not finders:
        return ["未找到 vllm_ascend 的 editable finder（安装方式变了？需人工核对 import 路径）"]
    for f in finders:
        txt = (pathlib.Path(SITE) / f).read_text(encoding="utf-8", errors="replace")
        if "'vllm_ascend': '/vllm-workspace/vllm-ascend/vllm_ascend'" not in txt:
            fails.append(f"editable 映射异常 @ {f}: 未指向 /vllm-workspace/vllm-ascend/vllm_ascend")
    return fails


def selftest(verbose: bool = True) -> int:
    """自证：工具必须抓得住「调用点有、定义点无」这一类已知漂移。"""
    import tempfile

    with tempfile.TemporaryDirectory() as td:
        root = pathlib.Path(td) / "tree"
        (root / "ops").mkdir(parents=True)
        (root / "worker").mkdir(parents=True)
        (root / "ops" / "shared.py").write_text("class A:\n    def foo(self):\n        return 1\n", encoding="utf-8")
        (root / "worker" / "caller.py").write_text("def go(self):\n    self.sep.foo()\n", encoding="utf-8")
        tbl = {"seams": [{"id": "self-ok", "kind": "symbol", "definer": "ops/shared.py",
                          "symbol": "A.foo", "consumer": "worker/caller.py", "call_token": ".foo("}]}
        ok_fails = check(tbl, root)
        # 注入漂移：定义点改名（调用点不动）
        (root / "ops" / "shared.py").write_text("class A:\n    def renamed(self):\n        return 1\n", encoding="utf-8")
        drift_fails = check(tbl, root)
        good = (not ok_fails) and bool(drift_fails)
        if verbose:
            print(f"[selftest] 一致态失败数={len(ok_fails)}（应 0）; 注入漂移后失败数={len(drift_fails)}（应 >0）")
            for f in drift_fails:
                print(f"[selftest]   抓到: {f}")
            print(f"[selftest] {'PASS' if good else 'FAIL'}")
        return 0 if good else 1


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--tree", default=os.environ.get("SEAM_TREE", DEFAULT_TREE))
    ap.add_argument("--table", default=DEFAULT_TABLE)
    ap.add_argument("--check-import-path", action="store_true")
    ap.add_argument("--selftest", action="store_true")
    ap.add_argument("--quiet", action="store_true")
    a = ap.parse_args()

    if a.selftest:
        return selftest(verbose=not a.quiet)

    try:
        table = json.loads(pathlib.Path(a.table).read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as ex:
        print(f"断言表读取失败: {a.table} ({ex})", file=sys.stderr)
        return 2

    tree = pathlib.Path(a.tree)
    fails = check(table, tree)
    if a.check_import_path:
        fails += check_import_path()

    n = len(table.get("seams", []))
    if fails:
        print(f"接口面校验 FAIL（{len(fails)}/{n} 条）  tree={tree}")
        for f in fails:
            print(f"  - {f}")
        print("  → 起服前先修: 定义点/调用点必须同一状态（补丁原件+patch 成对, 勿手工单侧改）")
        return 1
    if not a.quiet:
        print(f"接口面校验 PASS（{n} 条）  tree={tree}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
