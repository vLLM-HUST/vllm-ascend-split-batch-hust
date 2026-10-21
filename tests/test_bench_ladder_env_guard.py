from __future__ import annotations

import subprocess
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "tools" / "bench-ladder.sh"


def run_guard(
    *, ld_library_path: str, atb_home: str, skip: str = ""
) -> subprocess.CompletedProcess[str]:
    script = 'source "$1" __src >/dev/null 2>&1\nserve_env_preflight\n'
    env = {
        "PATH": "/usr/bin:/bin",
        "HOME": "/root",
        "LD_LIBRARY_PATH": ld_library_path,
        "LADDER_ATB_HOME": atb_home,
    }
    if skip:
        env["LADDER_SKIP_ENV_CHECK"] = skip
    return subprocess.run(
        ["bash", "-c", script, "bash", str(SCRIPT)],
        capture_output=True,
        text=True,
        check=False,
        env=env,
        timeout=60,
    )


def test_guard_rejects_missing_atb_entry(tmp_path: Path) -> None:
    proc = run_guard(
        ld_library_path="/usr/local/Ascend/driver/lib64", atb_home=str(tmp_path)
    )
    assert proc.returncode == 1
    assert "libatb.so" in proc.stderr


def test_guard_accepts_atb_entry(tmp_path: Path) -> None:
    proc = run_guard(
        ld_library_path=f"{tmp_path}/lib:/usr/local/Ascend/driver/lib64",
        atb_home=str(tmp_path),
    )
    assert proc.returncode == 0, proc.stderr


def test_guard_skip_env_check(tmp_path: Path) -> None:
    proc = run_guard(
        ld_library_path="/usr/local/Ascend/driver/lib64",
        atb_home=str(tmp_path),
        skip="1",
    )
    assert proc.returncode == 0, proc.stderr


def test_guard_absent_atb_home_is_noop(tmp_path: Path) -> None:
    proc = run_guard(
        ld_library_path="/usr/local/Ascend/driver/lib64",
        atb_home=str(tmp_path / "nonexistent"),
    )
    assert proc.returncode == 0, proc.stderr
