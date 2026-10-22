from __future__ import annotations

import json
import os
import subprocess
import sys
import threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

import pytest

# 运行树真身由 bench-ladder.sh 的 SCRIPTS 指向; 测试允许环境覆盖以便移植。
SCRIPT = Path(os.environ.get("LADDER_L1_SCRIPT", "/root/包体恢复/scripts/ladder-l1.py"))


def sse(*objs: dict) -> bytes:
    out = b""
    for o in objs:
        out += b"data: " + json.dumps(o).encode() + b"\n\n"
    return out + b"data: [DONE]\n\n"


def chat_stream(completion_tokens: int, prompt_tokens: int, no_space: bool) -> bytes:
    body = sse(
        {"choices": [{"delta": {"role": "assistant", "content": ""}}]},
        {"choices": [{"delta": {"content": "Hello"}}]},
        {"choices": [{"delta": {"content": " world"}}]},
        {
            "choices": [],
            "usage": {
                "prompt_tokens": prompt_tokens,
                "completion_tokens": completion_tokens,
            },
        },
    )
    if no_space:
        return body.replace(b"data: ", b"data:")
    return body


def make_handler(behave: dict) -> type[BaseHTTPRequestHandler]:
    class Handler(BaseHTTPRequestHandler):
        def log_message(self, *args: object) -> None:
            pass

        def do_POST(self) -> None:  # noqa: N802
            self.rfile.read(int(self.headers.get("Content-Length", 0)))
            behave["requests"] += 1
            n = behave["requests"]
            if n in behave["fail_at"]:
                body = b'{"error": "context too long"}'
                self.send_response(500)
            elif n in behave["bad_line_at"]:
                body = b"data: {broken json\n\n"
                self.send_response(200)
            else:
                body = chat_stream(
                    behave["completion_tokens"], 1700, behave["no_space"]
                )
                self.send_response(200)
            self.send_header("Content-Type", "text/event-stream")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)

    return Handler


@pytest.fixture(name="fake_serve")
def fixture_fake_serve():
    behave = {
        "fail_at": set(),
        "bad_line_at": set(),
        "completion_tokens": 256,
        "no_space": False,
        "requests": 0,
    }
    server = ThreadingHTTPServer(("127.0.0.1", 0), make_handler(behave))
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    yield behave, server.server_address[1]
    server.shutdown()
    server.server_close()


def run_l1(port: int, iters: int, out_path: Path) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        [
            sys.executable,
            str(SCRIPT),
            "--port",
            str(port),
            "--iters",
            str(iters),
            "--out",
            str(out_path),
        ],
        capture_output=True,
        text=True,
        check=False,
        timeout=120,
    )


def test_success_records_usage_annotations(fake_serve, tmp_path: Path) -> None:
    behave, port = fake_serve
    out = tmp_path / "l1.json"
    proc = run_l1(port, iters=3, out_path=out)
    assert proc.returncode == 0, proc.stderr
    assert "L1 median:" in proc.stdout
    data = json.loads(out.read_text())
    assert data["failures"] == []
    assert data["truncated"] is False
    assert data["out_tokens_min"] == data["out_tokens_max"] == 256
    assert data["prompt_tokens"] == 1700
    assert data["max_tokens"] == 256
    for row in data["rows"]:
        assert row["out_tokens"] == 256
        assert row["ttft_ms"] is not None
        assert row["prompt_tokens"] == 1700
    assert set(data["median"]) == {"ttft_ms", "e2e_ms", "tpot_ms"}


def test_midloop_failure_keeps_rows_and_exits_nonzero(
    fake_serve, tmp_path: Path
) -> None:
    behave, port = fake_serve
    behave["fail_at"] = {2}
    out = tmp_path / "l1.json"
    proc = run_l1(port, iters=3, out_path=out)
    assert proc.returncode == 1
    assert "iter02 FAIL" in proc.stdout
    assert "L1 FAIL: 1/3" in proc.stdout
    data = json.loads(out.read_text())
    assert len(data["failures"]) == 1
    assert data["failures"][0]["iter"] == 2
    assert "HTTP 500" in data["failures"][0]["error"]
    assert "context too long" in data["failures"][0]["error"]
    assert len(data["rows"]) == 3
    assert set(data["rows"][1]) == {"error"}
    assert data["rows"][0]["out_tokens"] == 256
    assert data["median"]["ttft_ms"] is not None
    assert data["truncated"] is False


def test_short_output_flagged_truncated(fake_serve, tmp_path: Path) -> None:
    behave, port = fake_serve
    behave["completion_tokens"] = 100
    out = tmp_path / "l1.json"
    proc = run_l1(port, iters=2, out_path=out)
    assert proc.returncode == 0, proc.stderr
    data = json.loads(out.read_text())
    assert data["truncated"] is True
    assert data["out_tokens_max"] == 100


def test_bad_sse_line_recorded_loop_continues(fake_serve, tmp_path: Path) -> None:
    behave, port = fake_serve
    behave["bad_line_at"] = {1}
    out = tmp_path / "l1.json"
    proc = run_l1(port, iters=2, out_path=out)
    assert proc.returncode == 1
    assert "iter01 FAIL" in proc.stdout
    data = json.loads(out.read_text())
    assert len(data["rows"]) == 2
    assert "JSONDecodeError" in data["rows"][0]["error"]
    assert data["rows"][1]["out_tokens"] == 256
    assert len(data["failures"]) == 1


def test_data_without_space_still_parsed(fake_serve, tmp_path: Path) -> None:
    behave, port = fake_serve
    behave["no_space"] = True
    out = tmp_path / "l1.json"
    proc = run_l1(port, iters=1, out_path=out)
    assert proc.returncode == 0, proc.stderr
    data = json.loads(out.read_text())
    assert data["rows"][0]["out_tokens"] == 256
    assert data["rows"][0]["ttft_ms"] is not None
