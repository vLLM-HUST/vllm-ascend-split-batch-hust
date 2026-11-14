"""文档行文与仓库事实的一致性门禁（纯文本、无设备依赖）。

四类容易漂移的文字描述在这里落成机械检查：

1. **数量陈述**：所有 Markdown 里"N 个插件 / N 个 bundle"必须等于
   `pyproject.toml` 里 `vllm.general_plugins` 与 `vllm_hust.extension_bundles`
   的实际条目数；
2. **算子目录契约**：`ops/README.md` 约定的必备文件（README、REPORT、
   evidence.json、bench_l0.py）在每个算子子目录里真实存在；
3. **术语表质量**：`GLOSSARY.md` 章节齐全、词条四列非空、术语不重复、
   锚点里的文件路径真实存在，且被 `AGENTS.md` 与 `docs/README.md` 引用；
4. **行文纪律**：对外规范文档不得出现"关键词 + 数字"同段（镜像组织基准
   数据政策判据），也不得出现已废止的隐晦暗语。

新增对外规范文档时，把它加进 `PROSE_SURFACE` 才会被检查；这里刻意用显式
清单，避免把历史证据目录（`docs/evidence/`、`provenance/`）卷入新判据。
"""

from __future__ import annotations

import re
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]

try:
    import tomllib
except ModuleNotFoundError:  # Python 3.10
    import tomli as tomllib

#: 受"关键词 + 数字"与禁语判据约束的对外规范行文。
PROSE_SURFACE = (
    "README.md",
    "AGENTS.md",
    "GLOSSARY.md",
    "docs/README.md",
    "docs/architecture.md",
    "docs/agent-guide.md",
    "docs/support-matrix.md",
    "docs/operator-benchmark-standard.md",
    "docs/mod-acceptance-standard.md",
    "docs/mod-acceptance-audit.md",
    ".openbitfun/skills/mod-acceptance/SKILL.md",
    "docs/npu-benchmark-execution-guide.md",
    ".openbitfun/skills/operator-sol-optimizer/SKILL.md",
    "docs/container-and-plugin-boundary-spec.md",
    "docs/HOST_CONTRACT.md",
    "docs/cohort-and-ownership.md",
    "docs/release.md",
    "docs/delivery-inventory.md",
    "docs/pitfalls.md",
    "ops/README.md",
)

#: 记录禁语本身的文件（墓碑与禁令原文），不参与禁语扫描。
BANNED_WORD_SOURCES = ("AGENTS.md", "GLOSSARY.md")

#: 与组织基准数据政策逐字一致的度量关键词判据。
MEASURED = re.compile(
    r"(?:ttft|tpot|throughput|latency|tokens/s|实测|吞吐|延迟).{0,60}\d",
    re.IGNORECASE,
)

#: `AGENTS.md` §3.2 明令禁止的隐晦暗语。
BANNED_WORDS = ("载波", "领地", "点钉", "husk", "档2 载体", "短跑", "活守卫")

#: "N 个插件 / N 个 bundle" 的数量陈述；组 1 是数字，键是实际数量来源。
COUNT_CLAIMS = (
    (re.compile(r"(\d+)\s*`vllm\.general_plugins`\s*carriers"), "plugin"),
    (re.compile(r"(\d+)\s*个\s*`vllm\.general_plugins`"), "plugin"),
    (re.compile(r"(\d+)\s*个\s*general plugins"), "plugin"),
    (re.compile(r"(\d+)\s*个\s*插件(?![包])"), "plugin"),
    (re.compile(r"(\d+)\s*`vllm_hust\.extension_bundles`"), "bundle"),
    (re.compile(r"(\d+)\s*个\s*`vllm_hust\.extension_bundles`"), "bundle"),
    (re.compile(r"(\d+)\s*(?:个\s*)?extension bundles?"), "bundle"),
    (re.compile(r"(\d+)\s*个\s*bundles?"), "bundle"),
)

#: 算子目录必备文件（`ops/README.md` 的目录契约）。
OPERATOR_REQUIRED = ("README.md", "REPORT.md", "bench_l0.py", "evidence.json")

#: 术语表必备章节前缀，顺序即文档结构。
GLOSSARY_SECTIONS = ("## 0.", "## 1.", "## 2.", "## 3.", "## 4.", "## 5.", "## 6.")

#: 锚点里可以按仓库相对路径校验的后缀。
ANCHOR_SUFFIXES = (".py", ".md", ".json", ".toml", ".sh", ".sha256", ".yaml", ".yml")

#: 锚点中表示"不是固定路径"的字符（通配、占位、变量、家目录）。
ANCHOR_NON_PATH = ("*", "<", ">", "$", "~")


def _actual_counts() -> dict[str, int]:
    pyproject = tomllib.loads((ROOT / "pyproject.toml").read_text(encoding="utf-8"))
    entry_points = pyproject["project"]["entry-points"]
    return {
        "plugin": len(entry_points["vllm.general_plugins"]),
        "bundle": len(entry_points["vllm_hust.extension_bundles"]),
    }


def _markdown_files() -> list[Path]:
    files = list(ROOT.glob("*.md")) + list((ROOT / "docs").glob("*.md"))
    files += list((ROOT / "ops").glob("*.md"))
    return [path for path in files if path.is_file()]


def test_plugin_and_bundle_counts_in_prose_match_pyproject():
    """Markdown 里的"N 个插件 / N 个 bundle"必须与 entry points 实际数量一致。"""
    actual = _actual_counts()
    violations = []
    for path in sorted(_markdown_files()):
        rel = path.relative_to(ROOT).as_posix()
        for lineno, line in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
            for pattern, key in COUNT_CLAIMS:
                for match in pattern.finditer(line):
                    if int(match.group(1)) != actual[key]:
                        violations.append(
                            f"{rel}:{lineno} 写的是 {match.group(1)} 个 "
                            f"{key}，实际 {actual[key]} 个：{match.group(0)!r}"
                        )
    assert not violations, "数量陈述与 pyproject.toml 不一致: " + "; ".join(
        violations[:10]
    )


def test_every_operator_directory_has_the_contract_files():
    """`ops/<operator>/` 必须齐备 README、REPORT、bench_l0、evidence。"""
    missing = []
    for op_dir in sorted(p for p in (ROOT / "ops").iterdir() if p.is_dir()):
        for name in OPERATOR_REQUIRED:
            if not (op_dir / name).is_file():
                missing.append(f"{op_dir.relative_to(ROOT).as_posix()}/{name}")
    assert not missing, "算子目录缺少契约文件: " + "; ".join(missing)


def test_glossary_structure_and_term_uniqueness():
    """术语表章节齐全、词条四列非空、同一术语不重复登记。"""
    text = (ROOT / "GLOSSARY.md").read_text(encoding="utf-8")
    assert text.startswith("# GLOSSARY"), "GLOSSARY.md 首行必须是标题"
    for section in GLOSSARY_SECTIONS:
        assert section in text, f"GLOSSARY.md 缺少章节 {section}"

    rows = []
    inside_terms = False
    for line in text.splitlines():
        if line.startswith("## "):
            inside_terms = False
        if line.startswith("### 3."):
            inside_terms = True
            continue
        if inside_terms and line.startswith("|"):
            cells = [cell.strip() for cell in line.strip().strip("|").split("|")]
            if len(cells) < 4 or set(cells[0]) <= set("-: "):
                continue
            if cells[0] == "术语":
                continue
            rows.append(cells)

    assert len(rows) >= 50, f"词条数量异常（{len(rows)} 条），疑似结构被破坏"
    for cells in rows:
        for index, cell in enumerate(cells[:4]):
            assert cell, f"词条 {cells[0]!r} 第 {index + 1} 列为空"
    names = [cells[0] for cells in rows]
    duplicates = sorted({name for name in names if names.count(name) > 1})
    assert not duplicates, f"术语重复登记: {duplicates}"


def test_glossary_anchors_point_at_real_paths():
    """词条锚点里的仓库相对路径必须真实存在（通配与占位符跳过）。"""
    text = (ROOT / "GLOSSARY.md").read_text(encoding="utf-8")
    anchors = set()
    for line in text.splitlines():
        if not line.startswith("|"):
            continue
        cells = [cell.strip() for cell in line.strip().strip("|").split("|")]
        if len(cells) < 4:
            continue
        for token in re.findall(r"`([^`]+)`", cells[2]):
            anchors.add(token)

    missing = []
    for anchor in sorted(anchors):
        if any(char in anchor for char in ANCHOR_NON_PATH):
            continue
        if anchor.startswith(("http", "#")):
            continue
        if not anchor.endswith(ANCHOR_SUFFIXES) and not anchor.endswith("/"):
            continue
        if not (ROOT / anchor).exists():
            missing.append(anchor)
    assert not missing, "术语锚点指向不存在的路径: " + "; ".join(missing)


def test_glossary_is_reachable_from_the_entry_documents():
    """术语表必须出现在 `AGENTS.md` 与 `docs/README.md` 的导航里。"""
    for name in ("AGENTS.md", "docs/README.md"):
        text = (ROOT / name).read_text(encoding="utf-8")
        assert "GLOSSARY.md" in text, f"{name} 未链接 GLOSSARY.md"
    agents = (ROOT / "AGENTS.md").read_text(encoding="utf-8")
    assert "统一语言" in agents, "AGENTS.md 缺少统一语言约束条目"


def test_prose_surface_has_no_measured_metric_pairs():
    """对外规范行文不得出现"关键词 + 数字"同段（组织基准数据政策判据）。"""
    violations = []
    for name in PROSE_SURFACE:
        path = ROOT / name
        assert path.is_file(), f"PROSE_SURFACE 里的文件不存在: {name}"
        for lineno, line in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
            if MEASURED.search(line):
                violations.append(f"{name}:{lineno}")
    assert not violations, "命中度量关键词判据: " + "; ".join(violations)


def test_prose_surface_has_no_retired_slang():
    """对外规范行文不得出现 `AGENTS.md` §3.2 禁用的隐晦暗语。"""
    violations = []
    for name in PROSE_SURFACE:
        if name in BANNED_WORD_SOURCES:
            continue
        text = (ROOT / name).read_text(encoding="utf-8")
        for word in BANNED_WORDS:
            if word in text:
                violations.append(f"{name} -> {word}")
    assert not violations, "命中禁用暗语: " + "; ".join(violations)
