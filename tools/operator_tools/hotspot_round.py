"""Build a device-time ranking for one collection round, never across rounds."""

import csv
import json
import math
import re
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path

from tools.operator_tools.artifact_contract import SCHEMA_PREFIX, SCHEMA_VERSION

SCHEMA = f"{SCHEMA_PREFIX}/hotspot-summary/{SCHEMA_VERSION}"


def invocation_size(root: Path):
    path = root / "invocation.json"
    if path.is_file():
        cmd = json.loads(path.read_bytes())["command"]
        if "--tensor-parallel-size" in cmd:
            return int(cmd[cmd.index("--tensor-parallel-size") + 1])
    return None


def check_ranks(sources, expected):
    if not sources or (
        expected is not None and set(sources) != {str(i) for i in range(expected)}
    ):
        raise ValueError(
            f"incomplete rank sources: expected={expected}, found={sorted(sources)}"
        )


def summarize(profile: Path, expected: int | None):
    sources = {}
    counts = defaultdict(int)
    times = defaultdict(float)
    for path in sorted(profile.rglob("op_statistic.csv")):
        rel = path.relative_to(profile.parent).as_posix()
        match = re.search(r"(?:^|[_/])rank(\d+)(?:[_/]|$)", rel)
        if not match or match[1] in sources:
            raise ValueError(f"ambiguous rank source: {rel}")
        sources[match[1]] = rel
        device_us = 0.0
        with path.open(newline="", encoding="utf-8") as handle:
            for row in csv.DictReader(handle):
                op = row["OP Type"].strip()
                calls = int(row["Count"])
                total = float(row["Total Time(us)"])
                if not op or calls < 0 or total < 0 or not math.isfinite(total):
                    raise ValueError(f"invalid operator statistic: {rel}: {op}")
                counts[op] += calls
                times[op] += total
                device_us += total
        if not device_us > 0:
            raise ValueError(f"no device time: {rel}")
    check_ranks(sources, expected)
    grand = sum(times.values())
    return {
        "schema": SCHEMA,
        "produced_at": datetime.now(timezone.utc).isoformat(),
        "rank_sources": sources,
        "ops": [
            {
                "op": op,
                "count": counts[op],
                "total_us": round(times[op], 3),
                "share_pct": round(times[op] * 100 / grand, 3),
            }
            for op in sorted(times, key=lambda op: (-times[op], op))
        ],
    }
