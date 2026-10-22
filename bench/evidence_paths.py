import json
import os
from pathlib import Path, PurePosixPath

ROOT = Path(__file__).resolve().parents[1]


def output_directory():
    value = os.environ.get("SPLIT_BATCH_RESULTS_DIR")
    if not value:
        raise ValueError("Set SPLIT_BATCH_RESULTS_DIR to an external output directory")
    path = Path(value)
    if not path.is_absolute():
        raise ValueError("SPLIT_BATCH_RESULTS_DIR must be absolute")
    resolved = path.resolve()
    if path.is_relative_to(ROOT) or resolved.is_relative_to(ROOT):
        raise ValueError("Results must stay outside the MOD repository")
    return resolved


def archived_url(path):
    name = PurePosixPath(path)
    if name.is_absolute() or ".." in name.parts or str(name) != path:
        raise ValueError("Unsafe source path")
    source = json.loads((ROOT / "EVIDENCE_SOURCES.json").read_text())
    return (
        f"https://github.com/{source['repository']}/blob/{source['commit']}/"
        f"{source['snapshot']}/{path}"
    )
