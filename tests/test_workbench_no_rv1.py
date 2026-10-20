from __future__ import annotations

from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def test_workbench_scripts_do_not_reference_rv1():
    for name in ("bench-ladder.sh", "collect_comm_mechanism.sh"):
        text = (ROOT / "tools" / name).read_text()
        assert "/root/venvs/rv1" not in text, name
        assert "RV1_PYTHON" not in text, name


def test_parse_python_defaults_to_serving_stack():
    text = (ROOT / "tools" / "bench-ladder.sh").read_text()
    assert "PARSE_PYTHON=${LADDER_PARSE_PYTHON:-$PY}" in text
