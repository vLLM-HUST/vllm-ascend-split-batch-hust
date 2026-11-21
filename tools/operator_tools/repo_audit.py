"""Audit delivery index, evidence files and document path references.

Reuses `serving_sync.validate` and `custody` so the check follows the same
rules as collection and promotion; never mutates files.
"""

from __future__ import annotations

import json
import re
import sys
from pathlib import Path
from urllib.parse import unquote, urlparse

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from tools.operator_tools import custody  # noqa: E402

INDEX = ROOT / "DELIVERY_INDEX.json"
MEASURED = {
    "measured_no_gain",
    "verified_faster",
    "within_noise",
    "verified_on_par_with_baseline",
}
NO_CLAIM = {"pending_outer_loop", "negative_or_pending", "not_applicable"}
ATTR_PENDING = {
    "pending_outer_loop",
    "negative_or_pending",
    "superseded_by_host_fork",
    "l3_failed_default_off",
    "historical_only",
}

LINK = re.compile(r"\[[^\]]*\]\(([^)]+)\)")
CODE_PATH = re.compile(r"`([^`\n]+)`")
SKIP_PREFIX = ("http", "#", "mailto:")


def load(path: Path):
    return json.loads(path.read_text(encoding="utf-8"))


def uri_path(uri: str) -> Path | None:
    parsed = urlparse(uri)
    if parsed.scheme != "file" or parsed.netloc not in ("", "localhost"):
        return None
    return Path(unquote(parsed.path))


def check_index(problems: list[str]) -> dict[str, dict]:
    index = load(INDEX)
    entries: dict[str, dict] = {}
    for entry in index["operators"]:
        op_id = entry["id"]
        entries[op_id] = entry
        if entry.get("evidence") != f"ops/{op_id}/evidence.json":
            problems.append(
                f"{op_id}: evidence 路径与算子名不匹配: {entry.get('evidence')}"
            )
        path = ROOT / entry["evidence"]
        if not path.is_file():
            problems.append(f"{op_id}: 证据文件不存在: {path}")
            continue
        evidence = load(path)
        if evidence.get("$schema") != "operator-benchmark-evidence/v1":
            continue
        if evidence.get("operator_name") != op_id:
            problems.append(
                f"{op_id}: evidence.operator_name={evidence.get('operator_name')!r}"
            )
        ok, reason = custody.verify_dir_checksum(str(path.parent))
        if not ok:
            problems.append(f"{op_id}: 指纹校验失败: {reason}")
        l3 = (evidence.get("l3_numerical_guard") or {}).get("passed")
        expected_l3 = entry.get("l3_numerical_guard")
        if (l3 is True) != (expected_l3 == "passed") and (l3 is False) != (
            expected_l3 == "failed"
        ):
            problems.append(
                f"{op_id}: 索引 l3={expected_l3!r} 与证据 passed={l3!r} 不一致"
            )
        serving = evidence.get("serving_end_to_end") or {}
        status = entry.get("serving_status")
        if status in MEASURED and serving.get("tested") is not True:
            problems.append(
                f"{op_id}: serving_status={status} 但 tested={serving.get('tested')!r}"
            )
        if status == "not_applicable" and not (
            entry.get("gate_env") == "unexposed"
            and serving.get("tested") is False
            and serving.get("status") == status
            and (evidence.get("l4_attribution") or {}).get("status") == status
        ):
            problems.append(f"{op_id}: 已关闭的外环与门控或证据不一致")
        if status in NO_CLAIM and serving.get("tested") is True:
            problems.append(f"{op_id}: serving_status={status} 但 tested=true")
        if (l3 is not True) and status in MEASURED:
            problems.append(f"{op_id}: 数值护栏未过却有服务结论 {status}")
        attribution = evidence.get("l4_attribution") or {}
        if (
            attribution.get("status") == "pending_outer_loop"
            and status not in ATTR_PENDING
        ):
            problems.append(f"{op_id}: 归因未完成却有服务结论 {status}")
        if status == "historical_only" and (
            serving.get("tested") is not True
            or serving.get("applicability") != "historical"
        ):
            problems.append(f"{op_id}: historical_only 证据缺少历史服务测量")
        for field in (
            "l0_microbenchmark",
            "l3_numerical_guard",
            "l4_attribution",
            "serving_end_to_end",
        ):
            if field not in evidence:
                problems.append(f"{op_id}: v1 证据缺少 {field}")
        if entry.get("layer") == "host_fork" and status in MEASURED:
            has_attribution_sources = bool(
                attribution.get("sources") or attribution.get("profiles")
            )
            if attribution.get("status") != "completed" and not (
                attribution.get("passed") is True and has_attribution_sources
            ):
                problems.append(f"{op_id}: 宿主侧已测算子缺少逐卡归因来源")
            if not evidence.get("serving_sync"):
                problems.append(f"{op_id}: 宿主侧已测算子缺少服务配对归档")
    return entries


def check_archives(entries: dict[str, dict], problems: list[str]) -> int:
    try:
        from tools.operator_tools import serving_sync
    except ImportError:
        return 0
    checked = 0
    for op_id, entry in entries.items():
        evidence = load(ROOT / entry["evidence"])
        sync = evidence.get("serving_sync")
        if not sync:
            continue
        directory = uri_path(sync.get("archive", ""))
        if directory is None or not directory.is_dir():
            problems.append(f"{op_id}: 归档目录不存在: {sync.get('archive')}")
            continue
        try:
            result = serving_sync.validate(
                directory, entry, legacy_replay="service_route" not in sync
            )
            checked += 1
        except Exception as exc:
            problems.append(f"{op_id}: 归档重放校验失败: {type(exc).__name__}: {exc}")
            continue
        for key in ("l3_numerical_guard", "l4_attribution", "serving_end_to_end"):
            if result[key] != evidence.get(key):
                problems.append(f"{op_id}: 归档重放的 {key} 与仓内证据不一致")
        if result["serving_sync"] != sync:
            problems.append(f"{op_id}: 归档文件哈希与 serving_sync 不一致")
    return checked


def check_docs(entries: dict[str, dict], problems: list[str]) -> None:
    paths = [ROOT / "README.md", ROOT / "AGENTS.md", ROOT / "GLOSSARY.md"]
    paths += sorted((ROOT / "docs").rglob("*.md"))
    paths += sorted((ROOT / "ops").glob("*.md"))
    for path in paths:
        text = path.read_text(encoding="utf-8")
        rel = path.relative_to(ROOT).as_posix()
        for lineno, line in enumerate(text.splitlines(), 1):
            references = [(t, True) for t in LINK.findall(line)]
            references += [(t, False) for t in CODE_PATH.findall(line)]
            for target, is_link in references:
                target = target.strip()
                if not target or target.startswith(SKIP_PREFIX):
                    continue
                if "<" in target or "*" in target or "$" in target or "~" in target:
                    continue
                target = target.split("#", 1)[0].split(" ", 1)[0].rstrip(":,")
                # 只校验指向当前索引算子目录的引用；历史记录、废止词与
                # 宿主源码路径不在本仓交付面，不按存在性判错。
                match = re.match(r"(?:\.\./)*ops/([^/]+)/", target)
                if not match or match.group(1) not in entries:
                    continue
                base = path.parent if is_link else ROOT
                resolved = (base / target).resolve()
                if not resolved.exists():
                    problems.append(f"{rel}:{lineno}: 算子路径不存在: {target}")


def main() -> int:
    problems: list[str] = []
    entries = check_index(problems)
    archived = check_archives(entries, problems)
    check_docs(entries, problems)
    synced = sum(
        1
        for e in entries.values()
        if (ROOT / e["evidence"]).is_file()
        and load(ROOT / e["evidence"]).get("serving_sync")
    )
    report = {
        "indexed_operators": len(entries),
        "synced_operators": synced,
        "archive_replays": archived,
        "problems": problems,
    }
    print(json.dumps(report, indent=2, ensure_ascii=False))
    return 1 if problems else 0


if __name__ == "__main__":
    raise SystemExit(main())
