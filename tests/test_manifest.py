import json
from pathlib import Path

import tomllib
from vllm_hust_ext.manifest import activation_blocker, load_manifest

import vllm_ascend_split_batch
import vllm_ascend_split_batch.fia_demask as fia_demask_module
import vllm_ascend_split_batch.rope_fix as rope_fix_module
import vllm_ascend_split_batch.zerocost as zerocost_module

REPO_ROOT = Path(__file__).resolve().parents[1]
MANIFEST_PATH = Path(vllm_ascend_split_batch.__file__).with_name(
    "vllm-hust-extension-v0.2.json"
)
FIA_DEMASK_MANIFEST_PATH = Path(fia_demask_module.__file__).with_name(
    "vllm-hust-extension-v0.2.json"
)
ROPE_FIX_MANIFEST_PATH = Path(rope_fix_module.__file__).with_name(
    "vllm-hust-extension-v0.2.json"
)
ZEROCOST_MANIFEST_PATH = Path(zerocost_module.__file__).with_name(
    "vllm-hust-extension-v0.2.json"
)


def _manifest_json() -> dict:
    return json.loads(MANIFEST_PATH.read_text(encoding="utf-8"))


CASCADE_CARRIERS = (
    "vllm_ascend_split_batch.cascade_plugin",
    "vllm_ascend_split_batch.cascade_graph_plugin",
)
PLANNER_CARRIER = "vllm_ascend_split_batch.planner"


def test_descriptor_is_discoverable_and_activatable() -> None:
    """Discoverable and activatable: the cascade carriers now carry evidence.

    The three acceptance evidences were re-run on the pinned host baseline
    (default-off smoke / real-model correctness 6/64 / gatefix performance
    margin), so the cascade carrier guard is lifted and an active carrier
    makes the bundle enable-able.
    """
    manifest = load_manifest(MANIFEST_PATH)
    assert manifest.bundle_id == "org.vllm-hust.split-batch-full-graph"
    assert activation_blocker(manifest) is None


def test_only_cascade_carriers_are_active() -> None:
    """Flip the two cascade carriers to active; the planner stays inert.

    The evidence re-run on the pinned host baseline closed the flip blockers,
    so both cascade carriers (``cascade_plugin:load`` /
    ``cascade_graph_plugin:install``) are ``active``.  The planner has no
    acceptance evidence (review F7), so it must keep ``import_only`` -- an
    accidental flip would silently enable an unimplemented host contract.
    """
    carriers = {
        item["module"]: item["status"] for item in _manifest_json()["implementation"]
    }
    for module in CASCADE_CARRIERS:
        assert carriers[module] == "active", module
    assert carriers[PLANNER_CARRIER] == "import_only"
    # The demask carrier moved to its own bundle; it must not linger here.
    assert "vllm_ascend_split_batch.fia_demask_plugin" not in carriers


def test_fia_demask_bundle_is_activatable() -> None:
    """The standalone demask bundle is active with its own injection key.

    Acceptance evidence (probe-fia/E2E-mask-removal.md: 8/8 token parity,
    six-grid TPOT +3.3~+10.1% geomean +6.43%, e2e throughput +3.52%) cleared
    the flip blockers, so the carrier is ``active`` -- same ladder the cascade
    carriers went through (commit c968c73).
    """
    manifest = load_manifest(FIA_DEMASK_MANIFEST_PATH)
    assert manifest.bundle_id == "org.vllm-hust.fia-demask"
    assert activation_blocker(manifest) is None
    raw = json.loads(FIA_DEMASK_MANIFEST_PATH.read_text(encoding="utf-8"))
    carriers = {i["module"]: i["status"] for i in raw["implementation"]}
    assert carriers["vllm_ascend_split_batch.fia_demask_plugin"] == "active"
    env = raw["activation"]["environment"]
    assert env == {"VLLM_HUST_FIA_DEMASK": "1"}
    for key, value in env.items():
        assert value in {"0", "1"}, key
    assert raw["host"]["version_range"] == _manifest_json()["host"]["version_range"]


def test_rope_fix_bundle_is_import_only_with_its_own_key() -> None:
    """The rope-fix carrier is env-gated and stays ``import_only``.

    The three defects it carries are fixed plugin-side, but the acceptance
    ladder (default-off smoke / correctness / performance) is not closed for it
    yet, so the manifest must NOT advertise it as ``active``.
    """
    manifest = load_manifest(ROPE_FIX_MANIFEST_PATH)
    assert manifest.bundle_id == "org.vllm-hust.rope-fix"
    raw = json.loads(ROPE_FIX_MANIFEST_PATH.read_text(encoding="utf-8"))
    carriers = {i["module"]: i["status"] for i in raw["implementation"]}
    assert carriers["vllm_ascend_split_batch.rope_fix_plugin"] == "import_only"
    assert raw["activation"]["environment"] == {"VLLM_HUST_ROPE_FIX": "1"}
    assert raw["host"]["version_range"] == _manifest_json()["host"]["version_range"]


def test_zerocost_bundle_is_import_only_with_its_own_keys() -> None:
    """The zero-cost host-wiring carrier is env-gated and stays ``import_only``.

    It has the three acceptance items gathered (default-off smoke / bitwise
    correctness / prefill TTFT -1.09% vs a 0.30% band), but the release.md §3
    enablement ladder is blocked: under the manager launch (``vllm-hust-ext
    run``, which co-enables cascade graph) ``cascade_graph_plugin.install()``
    stacks a second wrapper over the host method, which
    ``zerocost_wiring._host_func`` cannot traverse, so capability ① silently
    fail-opens (evidence: ``docs/evidence/zerocost-activation-20260913.md``).
    An earlier, still-standing reason applies too: flipping ``active`` injects
    the two env keys for every deployment that enables the bundle, so the
    opt-in contract must be deliberate.  Either way the manifest must NOT
    advertise it as ``active``.
    """
    manifest = load_manifest(ZEROCOST_MANIFEST_PATH)
    assert manifest.bundle_id == "org.vllm-hust.zerocost-wiring"
    raw = json.loads(ZEROCOST_MANIFEST_PATH.read_text(encoding="utf-8"))
    carriers = {i["module"]: i["status"] for i in raw["implementation"]}
    assert carriers["vllm_ascend_split_batch.zerocost_wiring"] == "import_only"
    assert raw["activation"]["environment"] == {
        "VLLM_HUST_FI_PREFILL_OUT": "1",
        "VLLM_HUST_SKIP_COS_SIN": "1",
    }
    assert raw["host"]["version_range"] == _manifest_json()["host"]["version_range"]


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
    """Every bundle manifest must carry the distribution's version.

    The root bundle is not the only one shipped in the wheel: the standalone
    ``org.vllm-hust.fia-demask`` and ``org.vllm-hust.rope-fix`` manifests are
    discovered separately, and a stale ``extension_version`` there would survive
    a release bump unnoticed.
    """
    pyproject = tomllib.loads((REPO_ROOT / "pyproject.toml").read_text())
    version = pyproject["project"]["version"]
    for path in (
        MANIFEST_PATH,
        FIA_DEMASK_MANIFEST_PATH,
        ROPE_FIX_MANIFEST_PATH,
    ):
        raw = json.loads(path.read_text(encoding="utf-8"))
        assert raw["extension_version"] == version, path


def test_activation_environment_values_are_injectable_flags() -> None:
    environment = _manifest_json()["activation"]["environment"]
    for key, value in environment.items():
        assert value in {"0", "1"}, (
            f"{key}={value!r} is not an injectable flag; "
            "activation.environment holds values injected on enable, "
            "not documentation strings"
        )
