import json
from pathlib import Path

import tomllib
from vllm_hust_ext.manifest import activation_blocker, load_manifest

import vllm_ascend_split_batch

REPO_ROOT = Path(__file__).resolve().parents[1]
MANIFEST_PATH = Path(vllm_ascend_split_batch.__file__).with_name(
    "vllm-hust-extension-v0.2.json"
)


def _manifest_json() -> dict:
    return json.loads(MANIFEST_PATH.read_text(encoding="utf-8"))


def test_descriptor_is_discoverable_but_not_activatable() -> None:
    manifest = load_manifest(MANIFEST_PATH)
    assert manifest.bundle_id == "org.vllm-hust.split-batch-full-graph"
    assert activation_blocker(manifest) is not None


def test_extension_version_matches_distribution_version() -> None:
    manifest = _manifest_json()
    pyproject = tomllib.loads((REPO_ROOT / "pyproject.toml").read_text())
    assert manifest["extension_version"] == pyproject["project"]["version"]


def test_activation_environment_values_are_injectable_flags() -> None:
    environment = _manifest_json()["activation"]["environment"]
    for key, value in environment.items():
        assert value in {"0", "1"}, (
            f"{key}={value!r} is not an injectable flag; "
            "activation.environment holds values injected on enable, "
            "not documentation strings"
        )