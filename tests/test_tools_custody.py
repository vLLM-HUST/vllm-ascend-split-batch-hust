"""`tools/` 根与 `tools/analysis/` 的每个脚本必须被仓内至少一处引用。

背景：一次性探针/基准脚本曾以零引用状态散落在 `tools/` 根下，成为无人认领的
仓内产物。本测试把"工具必须有人认领"落成机械判据：删除后新增的散落脚本直接
报红。引用可以是文档、测试、skill、`pyproject.toml` 的 ruff exclude 登记
（方向记录脚本走这条）——任何被跟踪文件里的全路径、文件名或模块名提及都算数。
"""

from __future__ import annotations

import subprocess
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]

#: 方向记录脚本在 pyproject.toml 的 ruff exclude 里显式登记，属合法认领；
#: 目录自述文件不算待认领对象。
EXEMPT_NAMES = {"README.md"}


def tracked_tools() -> list[Path]:
    out = subprocess.run(
        ["git", "ls-files", "tools/"],
        cwd=ROOT,
        capture_output=True,
        text=True,
        check=True,
    ).stdout.splitlines()
    return [
        ROOT / rel
        for rel in out
        if rel.count("/") <= 1 or rel.startswith("tools/analysis/")
    ]


def is_referenced(path: Path) -> bool:
    rel = path.relative_to(ROOT).as_posix()
    name = path.name
    stem = path.stem
    for token in (rel, name, stem):
        proc = subprocess.run(
            ["git", "grep", "-l", "-F", "--", token],
            cwd=ROOT,
            capture_output=True,
            text=True,
            check=False,
        )
        # git grep 退出码：0 = 有命中，1 = 无命中，>1 = 真错误。
        if proc.returncode > 1:
            raise RuntimeError(f"git grep 失败: {proc.stderr.strip()}")
        referencing = [
            line for line in proc.stdout.splitlines() if line and line != rel
        ]
        if referencing:
            return True
    return False


def test_every_tool_script_is_claimed_by_a_reference():
    orphans = [
        path.relative_to(ROOT).as_posix()
        for path in tracked_tools()
        if path.name not in EXEMPT_NAMES and not is_referenced(path)
    ]
    assert not orphans, (
        "tools/ 下存在零引用脚本（散落产物，删除或先登记引用）: " + "; ".join(orphans)
    )
