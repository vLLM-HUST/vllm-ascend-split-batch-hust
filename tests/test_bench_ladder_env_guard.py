from __future__ import annotations

import ctypes
import subprocess
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "tools" / "bench-ladder.sh"
ATB_HOME = Path("/usr/local/Ascend/nnal/atb")


def run_guard(
    *,
    ld_library_path: str,
    atb_home: str,
    skip: str = "",
    extra_env: dict[str, str] | None = None,
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
    if extra_env:
        env.update(extra_env)
    return subprocess.run(
        ["bash", "-c", script, "bash", str(SCRIPT)],
        capture_output=True,
        text=True,
        check=False,
        env=env,
        timeout=60,
    )


@pytest.fixture(name="fake_atb_lib")
def fixture_fake_atb_lib(tmp_path: Path) -> Path:
    atb_home = tmp_path / "atb"
    lib_dir = atb_home / "latest" / "lib"
    lib_dir.mkdir(parents=True)
    src = tmp_path / "empty.c"
    src.write_text("int atb_fake_symbol;\n")
    subprocess.run(
        ["gcc", "-shared", "-fPIC", "-o", str(lib_dir / "libatb.so"), str(src)],
        check=True,
        capture_output=True,
        timeout=60,
    )
    return atb_home


def test_guard_rejects_missing_atb_entry(tmp_path: Path) -> None:
    proc = run_guard(
        ld_library_path="/usr/local/Ascend/driver/lib64", atb_home=str(tmp_path)
    )
    assert proc.returncode == 1
    assert "缺 nnal/atb 路径" in proc.stderr


def test_guard_rejects_unresolvable_libatb(tmp_path: Path) -> None:
    atb_home = tmp_path / "atb"
    lib_dir = atb_home / "latest" / "lib"
    lib_dir.mkdir(parents=True)
    proc = run_guard(
        ld_library_path=f"{lib_dir}:/usr/local/Ascend/driver/lib64",
        atb_home=str(atb_home),
    )
    assert proc.returncode == 1
    assert "dlopen libatb.so 失败" in proc.stderr


def test_guard_accepts_resolvable_libatb(fake_atb_lib: Path) -> None:
    proc = run_guard(
        ld_library_path=f"{fake_atb_lib}/latest/lib:/usr/local/Ascend/driver/lib64",
        atb_home=str(fake_atb_lib),
    )
    assert proc.returncode == 0, proc.stderr


def test_guard_survives_absent_host_interpreter(fake_atb_lib: Path) -> None:
    # 宿主无 0.28 栈解释器（CPU CI）时回退 PATH 的 python3，不报 dlopen 失败。
    proc = run_guard(
        ld_library_path=f"{fake_atb_lib}/latest/lib:/usr/local/Ascend/driver/lib64",
        atb_home=str(fake_atb_lib),
        extra_env={"LADDER_PYTHON": "/nonexistent/bin/python"},
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


@pytest.mark.skipif(not ATB_HOME.is_dir(), reason="nnal/atb 不在该容器")
def test_login_env_canary() -> None:
    ctypes.CDLL("libatb.so")
