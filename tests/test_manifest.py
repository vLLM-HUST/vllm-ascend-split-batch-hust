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


CASCADE_CARRIERS = (
    "vllm_ascend_split_batch.cascade_plugin",
    "vllm_ascend_split_batch.cascade_graph_plugin",
)
PLANNER_CARRIER = "vllm_ascend_split_batch.planner"


def test_descriptor_is_discoverable_and_blocked_pending_rerun() -> None:
    """Discoverable, but activation is blocked while carriers are import_only.

    The W4 active flip was reverted when the host baseline moved (see
    test_all_carriers_are_import_only_after_host_change); the blocker coming
    back is the guard working as designed.
    """
    manifest = load_manifest(MANIFEST_PATH)
    assert manifest.bundle_id == "org.vllm-hust.split-batch-full-graph"
    blocker = activation_blocker(manifest)
    assert blocker is not None
    assert "import_only" in blocker


def test_all_carriers_are_import_only_after_host_change() -> None:
    """W4 flipped the cascade carriers to active on 0.23.0rc1 evidence.

    The working baseline moved to vllm-ascend-hust main
    (0.25.1rc2.dev125+hust, worker/model_runner_v1 + compilation/acl_graph
    anchors).  The three acceptance evidences have NOT been re-run on that
    host, so per release.md discipline the carriers are demoted back to
    ``import_only`` until the re-run passes.  The planner stays inert
    regardless (review F7: no acceptance evidence at all).
    """
    carriers = {
        item["module"]: item["status"] for item in _manifest_json()["implementation"]
    }
    for module in (*CASCADE_CARRIERS, PLANNER_CARRIER):
        assert carriers[module] == "import_only", module


def test_host_version_range_pins_the_verified_baseline() -> None:
    """host.version_range must pin the exact verified host build.

    packaging rejects local-version labels in ordered comparators, so the
    single verified point is pinned with an ``==`` (local label included).
    """
    import packaging.specifiers
    import packaging.version

    host = _manifest_json()["host"]
    range_ = packaging.specifiers.SpecifierSet(host["version_range"])
    verified = "0.25.1rc2.dev125+hust.20260903.4.g74f0c0a27"
    assert range_.contains(packaging.version.Version(verified), prereleases=True)
    # A neighbouring build must NOT satisfy a point pin.
    assert not range_.contains(
        packaging.version.Version("0.25.1rc2.dev126"), prereleases=True
    )


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