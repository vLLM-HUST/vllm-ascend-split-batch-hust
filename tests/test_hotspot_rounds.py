import json
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from tools.operator_tools import artifact_view  # noqa: E402


def source(root, rank, op):
    path = root / "profile" / f"worker_rank{rank}_capture" / "op_statistic.csv"
    path.parent.mkdir(parents=True)
    path.write_text(f"Device_id,OP Type,Count,Total Time(us)\n{rank},{op},2,12.5\n")
    return path


def view(path):
    return artifact_view.main([str(path), "--view", "hotspot", "--top", "1"])


def test_old_multi_round_archive_reads_each_rank_without_writing(tmp_path, capsys):
    for i, op in enumerate(["FirstKernel", "SecondKernel"]):
        for rank in [0, 1]:
            source(tmp_path / f"round_{i:02d}_off", rank, op)
    files = set(tmp_path.rglob("*"))
    assert view(tmp_path) == 0
    out = capsys.readouterr().out
    for i, op in enumerate(["FirstKernel", "SecondKernel"]):
        assert f"round_{i:02d}_off" in out and f"{op}" in out
        assert any(
            op in line and line.split()[1:4] == ["4", "100.00", "25"]
            for line in out.splitlines()
        )
    assert "rank0" in out and "rank1" in out
    assert set(tmp_path.rglob("*")) == files


@pytest.mark.parametrize(
    "bad", ["empty_archive", "missing_rank", "empty_rank", "missing_round"]
)
def test_explicit_hotspot_fails_on_incomplete_window(tmp_path, bad):
    if bad != "empty_archive":
        root = tmp_path / "round_00_off"
        path = source(root, 0, "FirstKernel")
        if bad == "missing_rank":
            (root / "invocation.json").write_text(
                json.dumps({"command": ["--tensor-parallel-size", "2"]})
            )
        elif bad == "empty_rank":
            path.write_text(path.read_text().splitlines()[0] + "\n")
        else:
            (tmp_path / "collection_manifest.json").write_text('{"rounds":2}')
    assert view(tmp_path) != 0


def test_new_summary_independently_readable_and_contract_valid(tmp_path, capsys):
    from tools.operator_tools.artifact_contract import check_file, classify
    from tools.operator_tools.hotspot_round import summarize

    root = tmp_path / "round_00_off"
    source(root, 0, "NewKernel")
    body = summarize(root / "profile", 1)
    path = root / "hotspots.json"
    path.write_text(json.dumps(body))
    assert not check_file(path, classify(path.name), strict=True)
    assert view(tmp_path) == 0
    assert "NewKernel" in capsys.readouterr().out


@pytest.mark.parametrize("exported", [True, False])
def test_background_collector_and_reader_agree(tmp_path, exported):
    """Run the actual launcher with a CPU-only collect_guard in an isolated tree."""
    import os
    import shutil
    import subprocess

    tree = tmp_path / "tree"
    script = Path("workloads/profile_hotspot_longrun.sh")
    for rel in [
        script,
        Path("tools/operator_tools/artifact_contract.py"),
        Path("tools/operator_tools/hotspot_round.py"),
    ]:
        (tree / rel.parent).mkdir(parents=True, exist_ok=True)
        shutil.copyfile(ROOT / rel, tree / rel)
    (tree / "tools/operator_tools/serving_collect.py").write_text(
        "import os\n"
        "def collect_guard(config, output, gate):\n"
        '    p = output / "profile/test_rank0_capture/op_statistic.csv"\n'
        "    p.parent.mkdir(parents=True)\n"
        '    if os.environ["EXPORT"] == "True":\n'
        '        p.write_text("OP Type,Count,Total Time(us)\\nSentinel,2,25\\n")\n'
    )
    plan = tmp_path / "template.json"
    plan.write_text('{"operators":[{"config":{"environment":{"test":"1"}}}]}')
    run, archive = tmp_path / "run", tmp_path / "archive"
    proc = subprocess.run(
        [
            "bash",
            "-c",
            'source "$1"; wait "$(cat "$RUN_DIR/runner.pid")"',
            "_",
            tree / script,
        ],
        env={
            **os.environ,
            "RUN_DIR": str(run),
            "ARCHIVE_ROOT": str(archive),
            "PY": sys.executable,
            "RUNTIME_PY": sys.executable,
            "BASE_PLAN": str(plan),
            "TP_SIZE": "1",
            "PROFILE_ROUNDS": "1",
            "EXPORT": str(exported),
        },
        capture_output=True,
        text=True,
        timeout=20,
    )
    assert (proc.returncode == 0) == exported, (
        proc.stderr,
        (run / "nohup.log").read_text(),
    )
    state = json.loads((run / "status.json").read_text())["state"]
    assert state == ("done" if exported else "failed")
    assert (archive / "round_00_off/hotspots.json").is_file() == exported
    assert (view(archive) == 0) == exported
