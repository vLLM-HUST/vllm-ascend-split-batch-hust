import importlib.util
import io
import tarfile
import zipfile
from pathlib import Path

import pytest

SPEC = importlib.util.spec_from_file_location(
    "check_distribution",
    Path(__file__).resolve().parents[1] / "tools/check_distribution.py",
)
MODULE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(MODULE)


def artifacts(tmp_path, extra=None, missing=False):
    names = [
        f"vllm_ascend_split_batch/b{i}/vllm-hust-extension-v0.2.json" for i in range(6)
    ]
    with zipfile.ZipFile(tmp_path / "test.whl", "w") as archive:
        for name in names:
            archive.writestr(name, "{}")
        archive.writestr("test.dist-info/entry_points.txt", "plugins")
    with tarfile.open(tmp_path / "test.tar.gz", "w:gz") as archive:
        for name in names[1:] if missing else names:
            archive.addfile(tarfile.TarInfo("test/src/" + name), io.BytesIO())
        if extra:
            archive.addfile(tarfile.TarInfo("test/" + extra), io.BytesIO())


def test_clean(tmp_path):
    artifacts(tmp_path)
    MODULE.check(tmp_path)


@pytest.mark.parametrize(
    "path",
    ["bench/results/run.json", "docs/evidence/report.md", "run.log", "../escape"],
)
def test_reject_evidence(tmp_path, path):
    artifacts(tmp_path, extra=path)
    with pytest.raises(ValueError):
        MODULE.check(tmp_path)


def test_missing_runtime(tmp_path):
    artifacts(tmp_path, missing=True)
    with pytest.raises(ValueError, match="runtime"):
        MODULE.check(tmp_path)
