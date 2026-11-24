"""Generate a balanced directory from canonical experiment records."""

import argparse
import hashlib
import json
from pathlib import Path
from urllib.parse import quote

ROOT = Path(__file__).resolve().parents[2]


def _has_conclusion(value):
    parts = value.values() if isinstance(value, dict) else (value,)
    return bool(parts) and all(isinstance(v, str) and v.strip() for v in parts)


def entries(root):
    links = {}

    def add(key, path):
        if key in links or not path.exists() or not path.resolve().is_relative_to(root):
            raise ValueError(f"duplicate key or invalid target: {key}: {path}")
        links[key] = path.relative_to(root).as_posix()

    for op in json.loads((root / "DELIVERY_INDEX.json").read_bytes())["operators"]:
        add("op/" + op["id"], root / op["evidence"])
    patches = {}
    for path in root.glob("experiments/*/*.patch"):
        add("patch/" + path.parent.name + "/" + path.stem, path)
        patches[path.relative_to(root).as_posix()] = path
    referenced = set()
    for path in sorted((root / "docs/optimization-history").glob("*.json")):
        add("history/" + path.stem, path)
        record = json.loads(path.read_bytes())
        if not any(
            _has_conclusion(record.get(key))
            for key in ("decision", "verdict", "performance_verdict")
        ):
            raise ValueError(f"研发记录缺少结论（未测也须明示）：{path.name}")
        if "patch" in record or "patch_sha256" in record:
            name = record.get("patch")
            if not isinstance(name, str) or name not in patches:
                raise ValueError(f"研发记录引用的补丁不存在：{path.name}: {name}")
            digest = hashlib.sha256(patches[name].read_bytes()).hexdigest()
            if digest != record.get("patch_sha256"):
                raise ValueError(f"补丁与研发记录指纹不符：{name}")
            referenced.add(name)
    if missing := patches.keys() - referenced:
        raise ValueError(f"补丁缺少研发记录：{', '.join(sorted(missing))}")
    for path in sorted(root.glob("host-proposals/*/proposal_meta.json")):
        add("proposal/" + path.parent.name, path)
    for path in sorted(root.glob("experiments/*/EXPERIMENT_PLAN.md")):
        add("plan/" + path.parent.name, path)
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


GROUP_ORDER = (
    ("plan", "实验计划（进行中或参考）"),
    ("op", "算子交付证据（以 DELIVERY_INDEX.json 为准）"),
    ("history", "研发记录（阶段性结论）"),
    ("proposal", "宿主修改提案"),
    ("patch", "历史补丁（不代表当前生效）"),
    ("archive", "历史证据归档（只读溯源）"),
    ("legacy", "早期上游 PR 留痕（只读溯源）"),
)


def render_grouped(root):
    """Render entries grouped by category for human-readable terminal viewing."""
    buckets = {}
    for key, path in entries(root):
        prefix = key.split("/", 1)[0]
        buckets.setdefault(prefix, []).append((key, path))
    lines = ["# 实验索引（按类别分组视图，只读）", ""]
    seen = set()
    for prefix, title in GROUP_ORDER:
        items = buckets.get(prefix)
        if not items:
            continue
        seen.add(prefix)
        lines.append(f"## {title}")
        for key, path in items:
            lines.append(f"- [`{key}`]({quote(path)})")
        lines.append("")
    for prefix in sorted(buckets.keys() - seen):
        lines.append(f"## 其他（{prefix}）")
        for key, path in buckets[prefix]:
            lines.append(f"- [`{key}`]({quote(path)})")
        lines.append("")
    return "\n".join(lines).rstrip() + "\n"


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    mode = parser.add_mutually_exclusive_group()
    mode.add_argument("--write", action="store_true")
    mode.add_argument("--find", metavar="TOPIC", help="只在已生成的索引中查找主题")
    mode.add_argument(
        "--group", action="store_true", help="按类别分组打印人类可读视图（只读）"
    )
    args = parser.parse_args()
    index = ROOT / "INDEX.md"
    if args.group:
        print(render_grouped(ROOT), end="")
        return
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
