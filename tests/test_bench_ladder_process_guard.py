from __future__ import annotations

import subprocess
from pathlib import Path

import pytest

SCRIPT = Path(__file__).resolve().parents[1] / "tools/bench-ladder.sh"


def run(tmp_path, body):
    setup = """source "$1" __src >/dev/null 2>&1
LADDER_PIDFILE="$2/pid"
TEST_TMP="$2"
printf '%s\\n' "$$" > "$LADDER_PIDFILE"
health_ok() { return 0; }
"""
    return subprocess.run(
        ["bash", "-c", setup + body, "bash", str(SCRIPT), str(tmp_path)],
        capture_output=True,
        text=True,
        check=False,
        timeout=10,
    )


def test_live_owned_pid_and_health_pass(tmp_path):
    result = run(tmp_path, 'cmd_wait 1 "$$"')
    assert result.returncode == 0
    assert "serve 就绪" in result.stdout


def test_dead_pid_cannot_borrow_unrelated_health(tmp_path):
    result = run(
        tmp_path,
        """
printf '%s\\n' 99999999 > "$LADDER_PIDFILE"
health_ok() { echo called > "$TEST_TMP/health-called"; return 0; }
cmd_wait 1 99999999
""",
    )
    assert result.returncode == 1
    assert "PID 已退出或替换" in result.stderr
    assert "serve 就绪" not in result.stdout
    assert not (tmp_path / "health-called").exists()


def test_replaced_pid_fails_even_when_expected_pid_lives(tmp_path):
    result = run(
        tmp_path,
        """
printf '%s\\n' 99999999 > "$LADDER_PIDFILE"
cmd_wait 1 "$$"
""",
    )
    assert result.returncode == 1
    assert "PID 已退出或替换" in result.stderr


def test_replacement_during_health_probe_fails(tmp_path):
    result = run(
        tmp_path,
        """
health_ok() { printf '%s\\n' 99999999 > "$LADDER_PIDFILE"; return 0; }
cmd_wait 1 "$$"
""",
    )
    assert result.returncode == 1
    assert "serve 就绪" not in result.stdout


def test_process_loss_between_poll_iterations_fails(tmp_path):
    result = run(
        tmp_path,
        """
health_ok() { return 1; }
sleep() { printf '%s\\n' 99999999 > "$LADDER_PIDFILE"; }
cmd_wait 30 "$$"
""",
    )
    assert result.returncode == 1
    assert "PID 已退出或替换" in result.stderr


def test_owned_live_process_still_times_out_without_health(tmp_path):
    result = run(
        tmp_path,
        """
health_ok() { return 1; }
sleep() { :; }
cmd_wait 1 "$$"
""",
    )
    assert result.returncode == 1
    assert "超时(1s)" in result.stdout


def test_public_wait_keeps_endpoint_only_contract(tmp_path):
    result = run(tmp_path, 'rm "$LADDER_PIDFILE"; cmd_wait 1')
    assert result.returncode == 0
    assert "serve 就绪" in result.stdout


@pytest.mark.parametrize("state", ["Z", "Z+", "X", ""])
def test_zombie_dead_or_missing_process_state_fails(tmp_path, state):
    result = run(
        tmp_path,
        f"""ps() {{ printf '%s' '{state}'; }}
cmd_wait 1 "$$"
""",
    )
    assert result.returncode == 1


@pytest.mark.parametrize("pid", ["0", "-1", "not-a-pid"])
def test_invalid_pid_never_reaches_kill(tmp_path, pid):
    result = run(
        tmp_path,
        f"""kill() {{ echo unsafe-kill; return 0; }}
startup_pid_ok '{pid}'
""",
    )
    assert result.returncode == 1
    assert "unsafe-kill" not in result.stdout


def test_serve_binds_wait_and_mode_publication_to_captured_pid():
    source = SCRIPT.read_text()
    serve = source.split("cmd_serve() {", 1)[1].split("\ncmd_stop()", 1)[0]
    assert "local started_pid=$!" in serve
    assert 'cmd_wait 1800 "$started_pid"' in serve
    tail = serve.split('carrier_summary "$(carrier_probe_json "$log")"', 1)[1]
    assert tail.index('startup_pid_ok "$started_pid"') < tail.index("SERVE_MODE_FILE")
