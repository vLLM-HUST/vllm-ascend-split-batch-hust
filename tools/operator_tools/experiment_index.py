"""Generate a balanced directory from canonical experiment records."""

import argparse
import json
from pathlib import Path
from urllib.parse import quote

ROOT = Path(__file__).resolve().parents[2]


def entries(root):
    links = {}

    def add(key, path):
        if key in links or not path.exists() or not path.resolve().is_relative_to(root):
            raise ValueError(f"duplicate key or invalid target: {key}: {path}")
        links[key] = path.relative_to(root).as_posix()

    for op in json.loads((root / "DELIVERY_INDEX.json").read_bytes())["operators"]:
        add("op/" + op["id"], root / op["evidence"])
    for path in sorted((root / "docs/optimization-history").glob("*.json")):
        add("history/" + path.stem, path)
    for path in sorted(root.glob("host-proposals/*/proposal_meta.json")):
        add("proposal/" + path.parent.name, path)
    for path in sorted(root.glob("experiments/*/EXPERIMENT_PLAN.md")):
        add("plan/" + path.parent.name, path)
    for path in sorted(root.glob("experiments/*/*.patch")):
        add("patch/" + path.parent.name + "/" + path.stem, path)
    for path in sorted((root / "docs/evidence").iterdir()):
        if path.is_dir() and not path.name.startswith((".", "_")):
            add("archive/" + path.name, path)
    for path in sorted(
        {p.parent for p in root.glob("provenance/legacy-patches/*/*.patch")}
    ):
        add("legacy/" + path.name, path)
    return sorted(links.items())


def balanced(items, lo, hi, depth=0):
    """Yield (depth, item) in preorder of the median-split search tree."""
    if lo < hi:
        mid = (lo + hi) // 2
        yield depth, items[mid]
        yield from balanced(items, lo, mid, depth + 1)
        yield from balanced(items, mid + 1, hi, depth + 1)


def render(root):
    items = entries(root)
    lines = [
        "# 实验索引",
        "",
        "**下一步做什么：** [可勾选计划](实验计划.md) · "
        "[当前候选与停止条件](docs/optimization-next.json)",
        "",
        "**哪些结论能用：** [交付索引](DELIVERY_INDEX.json) · "
        "[验收标准](docs/operator-benchmark-standard.md) · "
        "[资料可用性](EVIDENCE_AVAILABILITY.json) · "
        "[规范导航](docs/README.md)",
        "",
        "按主题名在本页查找即可。`history` 是研发记录，`op` 是交付证据，"
        "`proposal` 是宿主提案；`patch`、`legacy` 和 `archive` 不代表当前生效。",
        "",
        "按关键词查找：`python -m tools.operator_tools.experiment_index --find gdn`。",
        "",
        "以下为自动生成的高度平衡搜索树。键小的进左子树，大的进右子树；缩进表示子节点，不表示技术依赖。",
        "只改源记录，随后运行 "
        "`python -m tools.operator_tools.experiment_index --write`。"
        "不带参数只检查，不写文件。",
        "",
    ]
    for depth, (key, path) in balanced(items, 0, len(items)):
        lines.append(f"{'  ' * depth}- [`{key}`]({quote(path)})")
    return "\n".join(lines) + "\n"


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    mode = parser.add_mutually_exclusive_group()
    mode.add_argument("--write", action="store_true")
    mode.add_argument("--find", metavar="TOPIC", help="只在已生成的索引中查找主题")
    args = parser.parse_args()
    index = ROOT / "INDEX.md"
    if args.find is not None:
        query = args.find.strip().casefold()
        if not query or not index.is_file():
            parser.exit(1, "需要主题和已生成的 INDEX.md。\n")
        matches = [
            line.strip()
            for line in index.read_text(encoding="utf-8").splitlines()
            if line.lstrip().startswith("- [`") and query in line.casefold()
        ]
        if not matches:
            parser.exit(1, "没有匹配的实验资料。\n")
        print("\n".join(matches))
        return
    text = render(ROOT)
    if args.write:
        index.write_text(text, encoding="utf-8")
    elif not index.is_file() or index.read_text(encoding="utf-8") != text:
        parser.exit(
            1, "INDEX.md 与源记录不同；运行 experiment_index --write 后提交。\n"
        )


if __name__ == "__main__":
    main()
