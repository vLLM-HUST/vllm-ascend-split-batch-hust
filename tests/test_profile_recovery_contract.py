import json
import os
import subprocess
import sys
from pathlib import Path

import pytest

SCRIPT = Path(__file__).resolve().parents[1] / "tools/bench-ladder.sh"


def run_shell(tmp_path, command, *args):
    records = tmp_path / "records"
    records.mkdir(exist_ok=True)
    return subprocess.run(
        [
            "bash", "-c",
            'source "$1" __src >/dev/null 2>&1\n'
            'PY="$2"; RECORDS="$3/records"; SCRIPTS="$3/scripts"; MODEL="$3/model"\n'
            'shift 3\n' + command,
            "bash", str(SCRIPT), sys.executable, str(tmp_path), *map(str, args),
        ],
        env={k: v for k, v in os.environ.items() if not k.startswith("LADDER_")},
        capture_output=True, text=True, timeout=30,
    )


def document(rank, volume=353):
    return {
        "asc_dir": str(rank),
        "load_caliber": {"window_coverage": {
            "schema": "opt-ops-window-coverage.v1",
            "declared": True, "prefill_token_volume": volume,
        }},
        "cohort": {"id": "test", "model": "Qwen3.5-35B-A3B", "precision": "bf16"},
    }


@pytest.fixture
def window(tmp_path):
    win = tmp_path / "window with spaces"
    rank = win / "tp0_rank0_ascend_pt"
    out = rank / "ASCEND_PROFILER_OUTPUT"
    out.mkdir(parents=True)
    for name in ("op_statistic.csv", "kernel_details.csv"):
        (out / name).write_text("test input\n")
    (win / "profile-plan.json").write_text(json.dumps({"require_shapes": False}))
    (win / "cohort.json").write_text(json.dumps(document(rank)["cohort"]))
    (win / "load-manifest.json").write_text("{}")
    scripts = tmp_path / "scripts"
    scripts.mkdir()
    (scripts / "ladder-opstat.py").write_text(
        'import json, sys\nfrom pathlib import Path\n'
        'args = dict(zip(sys.argv[1::2], sys.argv[2::2]))\n'
        'cohort = json.loads(Path(args["--cohort-file"]).read_text()) '
        'if "--cohort-file" in args else {"model": "unknown"}\n'
        'doc = {"asc_dir": args["--asc-dir"], "cohort": cohort, '
        '"load_caliber": {"window_coverage": {"schema": "opt-ops-window-coverage.v1", '
        '"declared": True, "prefill_token_volume": 353}}}\n'
        'Path(args["--out"]).write_text(json.dumps(doc))\n'
        'Path(args["--out"] + ".argv").write_text(json.dumps(args))\n'
    )
    return win, rank


@pytest.mark.parametrize("volume", [0, 353])
def test_scope_accepts_facts_not_caliber_id(tmp_path, volume):
    rank = tmp_path / "rank0_ascend_pt"
    rank.mkdir()
    p = tmp_path / "evidence.json"
    p.write_text(json.dumps(document(rank, volume)))
    proc = run_shell(tmp_path, 'window_opstat_ok "$1" "$2"', rank, p)
    assert proc.returncode == 0, proc.stderr
    assert proc.stdout.strip() == str(p)


@pytest.mark.parametrize("field", ["model", "precision"])
@pytest.mark.parametrize(
    "value", [None, "", "unknown", " UNMARKED ", "unspecified", []]
)
def test_scope_rejects_missing_or_placeholder_weights(tmp_path, field, value):
    doc = document(tmp_path / "rank0")
    doc["cohort"][field] = value
    p = tmp_path / "bad.json"
    p.write_text(json.dumps(doc))
    proc = run_shell(tmp_path, 'window_opstat_ok "$1" "$2"', tmp_path / "rank0", p)
    assert proc.returncode == 1, proc.stderr
    assert not proc.stdout.strip()


@pytest.mark.parametrize("facts", [
    {"schema": "opt-ops-window-coverage.v1"},
    {"prefill_token_volume": 353, "declared": False},
    {"prefill_token_volume": True},
    {"prefill_token_volume": -1},
    {"prefill_token_volume": float("nan")},
    {"prefill_token_volume": float("inf")},
    [353],
])
def test_scope_rejects_placeholder_or_invalid_shapes(tmp_path, facts):
    doc = document(tmp_path / "rank0")
    doc["load_caliber"]["window_coverage"] = facts
    p = tmp_path / "bad.json"
    p.write_text(json.dumps(doc))
    proc = run_shell(tmp_path, 'window_opstat_ok "$1" "$2"', tmp_path / "rank0", p)
    assert proc.returncode == 1, proc.stderr


def test_explicit_bad_hint_never_falls_back_to_other_rank(tmp_path):
    records = tmp_path / "records"
    records.mkdir()
    (records / "valid-opstat.json").write_text(json.dumps(document(tmp_path / "rank0")))
    bad = tmp_path / "other-rank.json"
    bad.write_text(json.dumps(document(tmp_path / "rank1")))
    proc = run_shell(tmp_path, 'window_opstat_ok "$1" "$2"', tmp_path / "rank0", bad)
    assert proc.returncode == 1


def test_recovery_preserves_window_cohort_and_append_only(tmp_path, window):
    win, rank = window
    command = 'cmd_recover_profile "$1" same-ts --caliber synthetic-fixed-protocol'
    for _ in range(2):
        proc = run_shell(tmp_path, command, win)
        assert proc.returncode == 0, proc.stdout + proc.stderr
    outputs = sorted((tmp_path / "records").glob("*.json"))
    assert len(outputs) == 2
    for p in outputs:
        doc = json.loads(p.read_text())
        assert doc["cohort"] == document(rank)["cohort"]
        argv = json.loads(Path(str(p) + ".argv").read_text())
        assert argv["--cohort-file"] == str(win / "cohort.json")
        assert argv["--window-manifest"] == str(win / "load-manifest.json")
        assert argv["--asc-dir"] == str(rank)


def test_recovery_ambiguous_window_requires_explicit_rank(tmp_path, window):
    win, rank = window
    (win / "tp1_rank1_ascend_pt").mkdir()
    proc = run_shell(tmp_path, 'cmd_recover_profile "$1"', win)
    assert proc.returncode == 1
    assert "不按 mtime" in proc.stderr
    assert not list((tmp_path / "records").iterdir())
    explicit = run_shell(tmp_path, 'cmd_recover_profile "$1"', rank)
    assert explicit.returncode == 0, explicit.stdout + explicit.stderr


def test_recovery_missing_cohort_writes_nothing(tmp_path, window):
    win, rank = window
    (win / "cohort.json").unlink()
    proc = run_shell(tmp_path, 'cmd_recover_profile "$1"', rank)
    assert proc.returncode == 1
    assert "缺同窗权重标签" in proc.stderr
    assert not list((tmp_path / "records").iterdir())


def test_recovery_unknown_model_does_not_report_success(tmp_path, window):
    win, rank = window
    (win / "cohort.json").write_text(json.dumps({"id": "test", "model": "unknown"}))
    proc = run_shell(tmp_path, 'cmd_recover_profile "$1"', rank)
    assert proc.returncode == 1
    assert "恢复完成;" not in proc.stdout
    assert "范围无效" in proc.stdout


def test_recovery_incomplete_rank_does_not_parse_or_write(tmp_path, window):
    _, rank = window
    (rank / "ASCEND_PROFILER_OUTPUT/kernel_details.csv").unlink()
    proc = run_shell(tmp_path, 'cmd_recover_profile "$1"', rank)
    assert proc.returncode == 1
    assert "另行受控解析" in proc.stdout
    assert not list((tmp_path / "records").iterdir())


def test_recovery_help_does_not_inspect_window(tmp_path):
    proc = run_shell(tmp_path, 'cmd_recover_profile --help')
    assert proc.returncode == 0
    assert "多 rank" in proc.stdout
    assert not list((tmp_path / "records").iterdir())


@pytest.mark.parametrize("field", ["load_caliber", "cohort", "asc_dir"])
def test_scope_malformed_blocks_without_traceback(tmp_path, field):
    doc = document(tmp_path / "rank0")
    doc[field] = ["invalid"]
    p = tmp_path / "bad.json"
    p.write_text(json.dumps(doc))
    proc = run_shell(tmp_path, 'window_opstat_ok "$1" "$2"', tmp_path / "rank0", p)
    assert proc.returncode == 1
    assert "Traceback" not in proc.stderr


def test_scope_same_rank_symlink_remains_valid(tmp_path):
    rank = tmp_path / "rank0"
    rank.mkdir()
    alias = tmp_path / "alias"
    alias.symlink_to(rank, target_is_directory=True)
    p = tmp_path / "evidence.json"
    p.write_text(json.dumps(document(alias)))
    proc = run_shell(tmp_path, 'window_opstat_ok "$1" "$2"', rank, p)
    assert proc.returncode == 0, proc.stderr


def test_slim_dryrun_rejects_unknown_weights(tmp_path):
    win = tmp_path / "window"
    rank = win / "rank0_ascend_pt"
    out = rank / "ASCEND_PROFILER_OUTPUT"
    out.mkdir(parents=True)
    for name in ("op_statistic.csv", "kernel_details.csv"):
        (out / name).write_text("input\n")
    (win / "profile-plan.json").write_text(json.dumps({"require_shapes": False}))
    raw = rank / "FRAMEWORK/torch.op_range"
    raw.parent.mkdir()
    raw.write_bytes(b"preserve this raw input")
    doc = document(rank)
    doc["cohort"]["model"] = "unknown"
    p = tmp_path / "bad.json"
    p.write_text(json.dumps(doc))
    proc = run_shell(tmp_path, 'cmd_slim "$1" --dryrun --opstat "$2"', rank, p)
    assert proc.returncode == 1, proc.stdout + proc.stderr
    assert "无任何可安全瘦身" in proc.stdout
    assert raw.read_bytes() == b"preserve this raw input"
    assert not list((tmp_path / "records").iterdir())
