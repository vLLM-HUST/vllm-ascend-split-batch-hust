"""测试全仓 JSON 文件合规与治理分类。

规则：
1. 根目录下禁止出现未登记的散落运行数据（如 *-l0-hcomm-floor.json）。
2. 根目录下禁止存在已撤除路线的死约束文件（如 constraints.json）。
3. 仓内所有现存 .json 文件必须有明确的功能归属。
"""

from __future__ import annotations

import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]

# 允许存在于根目录的白名单 JSON
ALLOWED_ROOT_JSONS = {
    "EVIDENCE_AVAILABILITY.json",
    "EVIDENCE_SOURCES.json",
    "MOD_METADATA.json",
}


def test_root_directory_has_no_loose_bench_dumps_or_retired_constraints():
    """根目录下禁止散落临时性能 dump 或过时 cascade 启动约束。"""
    root_jsons = {p.name for p in ROOT.glob("*.json")}
    disallowed = root_jsons - ALLOWED_ROOT_JSONS
    assert not disallowed, f"根目录下存在未登记或散落的 JSON 文件: {disallowed}"


def test_all_repository_jsons_belong_to_governed_categories():
    """仓内全部 .json 文件必须属于清晰的分类目录。"""
    valid_prefixes = (
        "src/",
        "tools/runtime-tree/",
        "tools/workbench-archive.json",
        "host-proposals/",
        "docs/evidence/",
        "bench/results/",
    )
    for p in ROOT.rglob("*.json"):
        rel = p.relative_to(ROOT).as_posix()
        # 跳过构建产物与临时目录
        if (
            rel.startswith(".git/")
            or rel.startswith(".openbitfun/")
            or rel.startswith("build/")
            or rel.startswith("dist/")
            or ".egg-info" in rel
        ):
            continue
        if rel in ALLOWED_ROOT_JSONS:
            continue
        assert any(rel.startswith(prefix) for prefix in valid_prefixes), (
            f"未分类的 JSON 文件: {rel}"
        )


def test_packaging_extension_manifests_are_valid():
    """验证 6 份 extension bundle json 规范合法。"""
    manifest_paths = list(ROOT.glob("src/**/vllm-hust-extension-v0.2.json"))
    assert len(manifest_paths) == 6, (
        f"预期 6 份扩展 manifest，实际 {len(manifest_paths)}"
    )
    for path in manifest_paths:
        data = json.loads(path.read_text(encoding="utf-8"))
        assert data.get("extension_version") == "0.2.0"
        assert "extension_id" in data
        assert "host" in data
        assert data["host"]["name"] == "vllm-ascend"
