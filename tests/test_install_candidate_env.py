# SPDX-License-Identifier: Apache-2.0

"""Guard the candidate install entry: index selection, wheel lookup, deps."""

import importlib.util
import subprocess
import sys
import zipfile
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parent.parent
TOOL = REPO / "tools" / "install_candidate_env.py"


@pytest.fixture(scope="module")
def installer():
    spec = importlib.util.spec_from_file_location(
        "install_candidate_env_under_test", TOOL
    )
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_cpu_torch_index_is_the_default(installer):
    """A PyPI torchaudio wheel cannot load against cpu-only torch."""
    assert installer.DEFAULT_TORCH_INDEX == "https://download.pytorch.org/whl/cpu"


def test_help_exposes_the_torch_index_switch():
    proc = subprocess.run(
        [sys.executable, str(TOOL), "--help"], capture_output=True, text=True
    )
    assert proc.returncode == 0, proc.stderr
    assert "--torch-index" in proc.stdout


def test_registered_conflict_is_named_not_ignored(installer):
    assert "opencv-python-headless" in installer.DEFAULT_EXPECTED
    assert "numpy>=2" in installer.DEFAULT_EXPECTED


def _fake_wheel(path: Path, name: str, metadata: str) -> Path:
    with zipfile.ZipFile(path, "w") as archive:
        archive.writestr(f"{name}-1.0.dist-info/METADATA", metadata)
    return path


def test_requires_dist_skips_extras(installer, tmp_path):
    wheel = _fake_wheel(
        tmp_path / "demo-1.0-py3-none-any.whl",
        "demo",
        "Name: demo\n"
        "Version: 1.0\n"
        "Requires-Dist: torch==2.13.0\n"
        "Requires-Dist: torchvision==0.28.0\n"
        'Requires-Dist: sentencepiece; extra == "audio"\n',
    )
    assert installer.requires_dist(wheel) == ["torch==2.13.0", "torchvision==0.28.0"]


def test_wheel_by_prefix_prefers_the_last_match(installer, tmp_path):
    for version in ("0.1", "0.2"):
        _fake_wheel(
            tmp_path / f"vllm-{version}-py3-none-any.whl", "vllm", "Name: vllm\n"
        )
    assert (
        installer.wheel_by_prefix(tmp_path, "vllm").name == "vllm-0.2-py3-none-any.whl"
    )


def test_wheel_by_prefix_fails_loudly(installer, tmp_path):
    with pytest.raises(SystemExit):
        installer.wheel_by_prefix(tmp_path, "vllm")
