from __future__ import annotations

import subprocess
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "tools" / "repair-ascend-login-env.sh"
VALUE = (
    "/usr/local/Ascend/nnal/atb/latest/atb/cxx_abi_1/lib"
    ":/usr/local/Ascend/ascend-toolkit/latest/lib64"
)


def run_repair(*args: str) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        ["bash", str(SCRIPT), *args],
        capture_output=True,
        text=True,
        check=False,
        timeout=60,
    )


def test_repair_adds_missing_line(tmp_path: Path) -> None:
    env_file = tmp_path / "environment"
    env_file.write_text("PATH=/usr/bin:/bin\n")
    proc = run_repair("--env-file", str(env_file), "--value", VALUE)
    assert proc.returncode == 0, proc.stderr
    lines = env_file.read_text().splitlines()
    assert lines == ["PATH=/usr/bin:/bin", f"LD_LIBRARY_PATH={VALUE}"]


def test_repair_replaces_stale_line_and_keeps_extras(tmp_path: Path) -> None:
    env_file = tmp_path / "environment"
    env_file.write_text(
        "PATH=/usr/bin:/bin\n"
        "LD_LIBRARY_PATH=/usr/local/Ascend/driver/lib64:/usr/custom/extra\n"
        "PYTHONPATH=/srv/pkg\n"
    )
    proc = run_repair("--env-file", str(env_file), "--value", VALUE)
    assert proc.returncode == 0, proc.stderr
    lines = env_file.read_text().splitlines()
    assert lines == [
        "PATH=/usr/bin:/bin",
        f"LD_LIBRARY_PATH={VALUE}:/usr/local/Ascend/driver/lib64:/usr/custom/extra",
        "PYTHONPATH=/srv/pkg",
    ]


def test_repair_is_idempotent(tmp_path: Path) -> None:
    env_file = tmp_path / "environment"
    env_file.write_text(f"LD_LIBRARY_PATH={VALUE}\n")
    proc = run_repair("--env-file", str(env_file), "--value", VALUE)
    assert proc.returncode == 0, proc.stderr
    assert "已一致" in proc.stdout
    assert env_file.read_text() == f"LD_LIBRARY_PATH={VALUE}\n"
    assert not list(tmp_path.glob("environment.bak-*"))


def test_repair_dry_run_leaves_file_untouched(tmp_path: Path) -> None:
    env_file = tmp_path / "environment"
    env_file.write_text("LD_LIBRARY_PATH=/usr/local/Ascend/driver/lib64\n")
    proc = run_repair("--env-file", str(env_file), "--value", VALUE, "--dry-run")
    assert proc.returncode == 0, proc.stderr
    assert env_file.read_text() == "LD_LIBRARY_PATH=/usr/local/Ascend/driver/lib64\n"


def test_repair_writes_backup(tmp_path: Path) -> None:
    env_file = tmp_path / "environment"
    env_file.write_text(f"LD_LIBRARY_PATH={VALUE}\nPATH=/bin\n")
    proc = run_repair("--env-file", str(env_file), "--value", f"{VALUE}:/opt/new")
    assert proc.returncode == 0, proc.stderr
    assert env_file.read_text() == f"LD_LIBRARY_PATH={VALUE}:/opt/new\nPATH=/bin\n"
    baks = list(tmp_path.glob("environment.bak-*"))
    assert len(baks) == 1
    assert baks[0].read_text() == f"LD_LIBRARY_PATH={VALUE}\nPATH=/bin\n"
