# SPDX-License-Identifier: Apache-2.0

import csv
import importlib.util
import json
import sqlite3
from pathlib import Path

import pytest

SPEC = importlib.util.spec_from_file_location(
    "attribute_comm_free",
    Path(__file__).resolve().parents[1] / "tools/attribute_comm_free.py",
)
forensics = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(forensics)


def test_intersection_does_not_double_count():
    assert forensics.intersection_ns([(0, 10), (5, 15)], [(2, 4), (3, 8)]) == 6
    assert forensics.intersection_ns([(1, 5)], [(10, 20)]) == 0


def test_free_pair_clips_and_tracks_unpaired_edges():
    kernels = [(100, 110, "10", "a"), (150, 160, "COMM", "b")]
    pairs, unpaired, total = forensics.stream_pairs(
        [(50, 60), (110, 150), (160, 180)], kernels, (100, 170)
    )
    assert pairs == {"10_to_COMM": 0.00004}
    assert unpaired == 0.00001
    assert total == 0.00005


def csv_fixture(path, names, durations):
    with path.open("w", newline="") as out:
        writer = csv.DictWriter(
            out,
            fieldnames=[
                "Start Time(us)",
                "Duration(us)",
                "Name",
                "Stream ID",
                "Accelerator Core",
            ],
        )
        writer.writeheader()
        for index, (name, duration) in enumerate(zip(names, durations, strict=True)):
            writer.writerow(
                {
                    "Start Time(us)": str(index * 2 + 1),
                    "Duration(us)": duration,
                    "Name": name,
                    "Stream ID": "N/A",
                    "Accelerator Core": "COMMUNICATION",
                }
            )


def test_duplicate_communication_name_rejected(tmp_path):
    path = tmp_path / "kernel.csv"
    csv_fixture(path, ["hcom_allReduce_a", "hcom_allReduce_a"], ["1", "1"])
    with pytest.raises(ValueError, match="duplicate communication"):
        forensics.read_kernels(path)


def test_pairing_mismatch_fails_closed(tmp_path):
    path0 = tmp_path / "rank0.csv"
    path1 = tmp_path / "rank1.csv"
    csv_fixture(path0, ["hcom_allReduce_a", "hcom_allGather_b"], ["1", "1"])
    csv_fixture(path1, ["hcom_allGather_b", "hcom_allReduce_a"], ["1", "1"])
    db = tmp_path / "profile.db"
    sqlite3.connect(db).close()
    cohort = tmp_path / "cohort.json"
    cohort.write_text(
        json.dumps(
            {"model": "Qwen3.5-35B-A3B", "precision": "bf16", "serve": {"tp": 2}}
        )
    )
    config = tmp_path / "config.json"
    config.write_text(json.dumps({"text_config": {"hidden_size": 2048}}))
    for rank in (0, 1):
        (tmp_path / f"opstat{rank}.json").write_text(
            json.dumps(
                {"device_ms": {"all_kernel_ms": 0.002}, "step_budget": {"free_ms": 0}}
            )
        )
    with pytest.raises(ValueError, match="rank communication name/order"):
        forensics.analyze(
            db,
            path0,
            path1,
            cohort,
            tmp_path / "opstat0.json",
            tmp_path / "opstat1.json",
            config,
        )


def test_repo_withdrawal_and_paired_evidence():
    root = (
        Path(__file__).resolve().parents[1]
        / "host-proposals"
        / "GDN-qkv-norm/validation/l4-multibuffer-20261019"
    )
    record = json.loads((root / "free-attribution-v1-withdrawal.json").read_text())
    for name, (_, expected) in record["opstat_sha256_before_after"].items():
        path = root / name
        assert forensics.digest(path) == expected
        stat = json.loads(path.read_text())
        assert "free_attribution" not in stat
        assert stat["step_budget"]["free_ms"] > 0
    paired = root / "comm-free-paired-forensics.json"
    assert forensics.digest(paired) == (
        "13212eccec831dbb7ba8c0cc2dafb43967118a890f5b02f01e35b6e534f06ce3"
    )
    data = json.loads(paired.read_text())
    assert data["rank_pair"]["all_comm_calls"] == 2624
    assert data["rank_pair"]["allreduce_calls"] == 2592
    assert "recover-copy-1019-0240_ascend_pt" in data["sources"][0]["path"]
