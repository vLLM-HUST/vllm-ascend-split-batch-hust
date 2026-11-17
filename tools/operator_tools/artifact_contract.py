"""Name the artifacts a collection archive may hold, and check them.

A collection run writes a directory tree that is archived outside the
repository. Every consumer of that tree — hotspot ranking, SOL accounting,
wall-clock breakdown — has so far re-derived the layout by hand, which is why
the same question costs a new script each time.

This module declares the artifact kinds that exist, the fields each one must
carry, and two checking levels:

``strict``
    Applies to output produced from now on: every JSON artifact names its
    ``schema`` and ``produced_at``, and carries the required body fields.

``legacy``
    Applies to archives written before the contract existed. Only the body is
    checked, so history stays readable without being rewritten.

Usage::

    python -m tools.operator_tools.artifact_contract <archive-dir> [--strict]
"""

from __future__ import annotations

import argparse
import csv
import json
import re
from dataclasses import dataclass, field
from pathlib import Path

SCHEMA_PREFIX = "vllm-hust/collection"
SCHEMA_VERSION = "v1"


@dataclass(frozen=True)
class Kind:
    """One artifact kind: how to recognise it and what its body must carry.

    ``item_field`` names the key that holds the record list. ``None`` means the
    body itself is that list. ``item_required`` of ``None`` means the record
    list is not inspected at all; an empty tuple means it must be a list but its
    item fields are not constrained.
    """

    name: str
    pattern: re.Pattern[str]
    required: tuple[str, ...] = ()
    item_required: tuple[str, ...] | None = None
    item_field: str | None = None
    note: str = ""


KINDS: tuple[Kind, ...] = (
    Kind(
        "hotspot-summary",
        re.compile(r"^hotspots\.json$"),
        item_required=("op", "count", "total_us", "share_pct"),
        note="per-op time ranking, one object per operator",
    ),
    Kind(
        "hotspot-pipes",
        re.compile(r"^pipes_.*\.json$"),
        required=("compute_total_us", "ops"),
        item_required=(
            "op",
            "calls",
            "total_us",
            "share_pct",
            "median_us",
            "busy_pipe_pct",
            "bottleneck_form",
        ),
        item_field="ops",
        note="per-op pipe composition; median_us is what separates prefill from decode",
    ),
    Kind(
        "hotspot-statistic",
        re.compile(r"^op_statistic_.*\.csv$"),
        required=("OP Type", "Count", "Total Time(us)"),
        note="msprof operator statistic export",
    ),
    Kind(
        "step-trace",
        re.compile(r"^step_trace_.*\.csv$"),
        required=("Computing", "Free", "Communication(Not Overlapped)", "Stage"),
        note="the only artifact that records wall-clock time",
    ),
    Kind(
        "sol-plan",
        re.compile(r"^sol_list\.json$"),
        required=(
            "peak_source",
            "peak_bandwidth_gb_s",
            "decidable",
            "over_capacity",
            "not_modelable",
        ),
        note="roofline inputs resolved against a calibration window",
    ),
    Kind(
        "sol-checklist",
        re.compile(r"^checklist\.json$"),
        item_required=("op", "share_pct", "measured_us", "calls", "section"),
        note="per-op verdict table; section separates decidable from not",
    ),
    Kind(
        "sol-gap-bench",
        re.compile(r"^sol_gap_bench\.json$"),
        required=("device", "peaks"),
        note="L0 micro-bench that fills roofline inputs the profile cannot express",
    ),
    Kind(
        "peak-calibration",
        re.compile(r"^(window|\d{2}_[a-z0-9_]+)\.json$"),
        note="measured bandwidth / cube / granularity for one device window",
    ),
    Kind(
        "round-invocation",
        re.compile(r"^invocation\.json$"),
        required=("command", "gate"),
        note="what a long-run round asked the server to do",
    ),
    Kind(
        "round-exit",
        re.compile(r"^exit\.json$"),
        required=("returncode",),
        note="how a long-run round ended",
    ),
    Kind(
        "round-owner",
        re.compile(r"^owner\.json$"),
        required=("pid",),
        note="which process owned the round",
    ),
    Kind(
        "round-response",
        re.compile(r"^(profile_response|responses)\.json$"),
        note="server replies kept per round; body is the API payload verbatim",
    ),
    Kind(
        "round-output",
        re.compile(r"^greedy\.json$"),
        item_required=(),
        note="sampled completions per round; a plain list of strings",
    ),
)

# Directory names that hold raw profiler output rather than summarised
# artifacts. Their payload is device trace data, not a contract subject.
RAW_DIRS = (
    "ASCEND_PROFILER_OUTPUT",
    "profiler_output",
    "mindstudio_profiler_output",
    "profile",
)


@dataclass
class Finding:
    path: str
    kind: str
    level: str
    problem: str


@dataclass
class Report:
    archive: str
    tolerated: int = 0
    skipped: int = 0
    findings: list[Finding] = field(default_factory=list)
    kinds_seen: set[str] = field(default_factory=set)

    @property
    def ok(self) -> bool:
        return not [f for f in self.findings if f.level == "error"]

    def as_dict(self) -> dict:
        return {
            "archive": self.archive,
            "ok": self.ok,
            "artifacts_checked": self.tolerated,
            "paths_skipped": self.skipped,
            "kinds_seen": sorted(self.kinds_seen),
            "findings": [f.__dict__ for f in self.findings],
        }


def classify(name: str) -> Kind | None:
    for kind in KINDS:
        if kind.pattern.match(name):
            return kind
    return None


def _load(path: Path):
    if path.suffix == ".csv":
        with path.open(newline="") as handle:
            reader = csv.DictReader(handle)
            return list(reader), list(reader.fieldnames or ())
    return json.loads(path.read_text()), None


def _missing(body: dict, required: tuple[str, ...]) -> list[str]:
    return [key for key in required if key not in body]


def check_file(path: Path, kind: Kind, strict: bool) -> list[Finding]:
    findings: list[Finding] = []
    try:
        data, header = _load(path)
    except (json.JSONDecodeError, UnicodeDecodeError, OSError) as exc:
        return [
            Finding(str(path), kind.name, "error", f"unreadable: {type(exc).__name__}")
        ]

    if kind.required and header is not None:
        absent = [key for key in kind.required if key not in header]
        if absent:
            findings.append(
                Finding(str(path), kind.name, "error", f"csv columns missing: {absent}")
            )
        return findings

    if strict:
        body = data if isinstance(data, dict) else {}
        if "schema" not in body:
            findings.append(Finding(str(path), kind.name, "error", "no schema field"))
        else:
            want = f"{SCHEMA_PREFIX}/{kind.name}/{SCHEMA_VERSION}"
            if body.get("schema") != want:
                findings.append(
                    Finding(
                        str(path),
                        kind.name,
                        "error",
                        f"schema is {body.get('schema')!r}, expected {want!r}",
                    )
                )
        if "produced_at" not in body:
            findings.append(
                Finding(str(path), kind.name, "error", "no produced_at field")
            )

    if kind.required:
        if not isinstance(data, dict):
            findings.append(
                Finding(str(path), kind.name, "error", "expected an object body")
            )
            return findings
        absent = _missing(data, kind.required)
        if absent:
            findings.append(
                Finding(str(path), kind.name, "error", f"fields missing: {absent}")
            )

    if kind.item_required is not None:
        if kind.item_field is None:
            rows = data
            where = "body"
        else:
            rows = data.get(kind.item_field) if isinstance(data, dict) else None
            where = kind.item_field
        if not isinstance(rows, list) or not rows:
            findings.append(
                Finding(
                    str(path),
                    kind.name,
                    "error",
                    f"expected a non-empty list at {where}",
                )
            )
            return findings
        if not kind.item_required:
            return findings
        first = rows[0]
        if not isinstance(first, dict):
            findings.append(
                Finding(
                    str(path), kind.name, "error", f"items at {where} are not objects"
                )
            )
            return findings
        absent = _missing(first, kind.item_required)
        if absent:
            findings.append(
                Finding(str(path), kind.name, "error", f"item fields missing: {absent}")
            )
    return findings


def _is_raw(path: Path, root: Path) -> bool:
    return any(part in RAW_DIRS for part in path.relative_to(root).parts)


def inspect(archive: Path, strict: bool = False) -> Report:
    report = Report(archive=str(archive))
    if not archive.is_dir():
        report.findings.append(
            Finding(str(archive), "-", "error", "archive path is not a directory")
        )
        return report

    for path in sorted(archive.rglob("*")):
        if not path.is_file():
            continue
        kind = classify(path.name)
        if kind is None:
            report.skipped += 1
            continue
        if _is_raw(path, archive):
            report.skipped += 1
            continue
        report.tolerated += 1
        report.kinds_seen.add(kind.name)
        report.findings.extend(check_file(path, kind, strict))
    return report


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("archive", type=Path, help="archived collection directory")
    parser.add_argument(
        "--strict",
        action="store_true",
        help="require schema and produced_at (for output made after the contract)",
    )
    parser.add_argument("--json-out", type=Path, default=None)
    args = parser.parse_args(argv)

    report = inspect(args.archive, strict=args.strict)
    payload = report.as_dict()
    if args.json_out:
        args.json_out.write_text(
            json.dumps(payload, indent=2, ensure_ascii=False) + "\n"
        )

    mode = "strict" if args.strict else "legacy"
    print(f"archive: {report.archive}")
    print(f"mode: {mode}  artifacts: {report.tolerated}  skipped: {report.skipped}")
    print(f"kinds: {', '.join(sorted(report.kinds_seen)) or '(none)'}")
    if not report.findings:
        print("result: ok")
        return 0
    for item in report.findings:
        print(f"  [{item.level}] {item.kind}: {Path(item.path).name} — {item.problem}")
    print(f"result: {len(report.findings)} finding(s)")
    return 0 if report.ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
