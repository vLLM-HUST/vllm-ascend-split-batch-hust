"""算子交付面(ops/** 与 DELIVERY_INDEX.json)的组织基准数据政策合规门禁。

镜像 vLLM-HUST/.github@POLICY_REF scripts/benchmark_data_policy.py 的 fail-closed
判据:组织工作流 `Benchmark-only data (organization)` 审查 PR 的每个变更文件,
命中即拒绝。docs/evidence/** 是有审查基线的存量债,由组织级 audit 看管,
不在本测试范围;新增文件严禁落入该类路径。
"""

import importlib.util
import re
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SPEC = importlib.util.spec_from_file_location(
    "delivery_custody", ROOT / "tools/op_agent/custody.py"
)
custody = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(custody)
# 与 .github/workflows/organization-benchmark-custody.yml 的固定审查版本一致。
POLICY_REF = "0cfe5109551e9b1c30ff3a75a2d93ee6ba8c9aa4"

# 与 scripts/benchmark_data_policy.py@POLICY_REF 逐字对齐。
DATA_PARTS = {"data", "datasets", "results", "evidence", "receipts", "runs", "traces"}
DATA_SUFFIXES = (
    ".csv",
    ".tsv",
    ".jsonl",
    ".ndjson",
    ".parquet",
    ".arrow",
    ".log",
    ".log.gz",
    ".prom",
)
MEASURED = re.compile(
    r"(?:ttft|tpot|throughput|latency|tokens/s|实测|吞吐|延迟).{0,60}\d",
    re.IGNORECASE,
)
PROSE_SUFFIXES = (".md", ".rst", ".txt", ".json", ".yaml", ".yml")


def delivery_surface():
    return [ROOT / "DELIVERY_INDEX.json", *sorted((ROOT / "ops").rglob("*"))]


def test_policy_revision_matches_custody_workflow():
    workflow = (
        ROOT / ".github/workflows/organization-benchmark-custody.yml"
    ).read_text(encoding="utf-8")
    assert POLICY_REF in workflow


def test_delivery_surface_has_no_data_candidate_paths():
    violations = []
    for path in delivery_surface():
        rel = path.relative_to(ROOT).as_posix().lower()
        parts = rel.split("/")[:-1]
        if DATA_PARTS.intersection(parts) or rel.endswith(DATA_SUFFIXES):
            violations.append(rel)
    assert violations == []


def test_delivery_surface_prose_has_no_measured_metric_pairs():
    violations = []
    for path in delivery_surface():
        if path.is_file() and path.suffix.lower() in PROSE_SUFFIXES:
            for lineno, line in enumerate(
                path.read_text(encoding="utf-8").splitlines(), 1
            ):
                if MEASURED.search(line):
                    violations.append(f"{path.relative_to(ROOT)}:{lineno}")
    assert violations == []


def test_delivery_checksum_covers_untampered_evidence():
    for checksum in sorted((ROOT / "ops").glob("*/checksum.sha256")):
        ok, message = custody.verify_dir_checksum(str(checksum.parent))
        assert ok, message
