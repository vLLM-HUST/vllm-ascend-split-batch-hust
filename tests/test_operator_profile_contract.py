# SPDX-License-Identifier: Apache-2.0

import hashlib
import importlib.util
import json
from pathlib import Path
from types import SimpleNamespace

import pytest

OPT = Path("/root/opt-ops/scripts/opt.py")


@pytest.fixture
def gate(monkeypatch, tmp_path):
    if not OPT.exists():
        pytest.skip("opt-ops deployment not installed")
    monkeypatch.syspath_prepend(str(OPT.parent))
    spec = importlib.util.spec_from_file_location("operator_gate_test", OPT)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    if not hasattr(mod, "_operator_profile_check"):
        pytest.skip("operator profile contract not deployed")
    evidence = tmp_path / "evidence.json"
    evidence.write_text("{}")
    digest = hashlib.sha256(evidence.read_bytes()).hexdigest()
    cand = {
        "id": "operator-test",
        "route": "test/operator",
        "stage": "ready",
        "checks": {
            key: {
                "status": "PASS",
                "evidence": [
                    {"path": str(evidence), "sha256": digest},
                ],
            }
            for key in ("l0", "l3")
        },
    }
    routes = [{"route": "test/operator", "layer": "operator"}]
    ws = mod._ws()
    monkeypatch.setattr(
        mod,
        "_ws",
        lambda: SimpleNamespace(
            routes=lambda: routes,
            hash_file=ws.hash_file,
            counters=ws.counters,
        ),
    )
    monkeypatch.setattr(mod, "read_candidate_scope", lambda cid: (True, cand, ""))
    calls = []

    def run(argv, **kwargs):
        calls.append(argv)
        return SimpleNamespace(returncode=0, stdout='{"ok":true}', stderr="")

    monkeypatch.setattr(mod, "run", run)
    return mod, cand, routes, calls, evidence


def args(reason="candidate-ab", tier="light"):
    return [
        "--candidate",
        "operator-test",
        "--trace-tier",
        tier,
        "--collection-reason",
        reason,
        "--gap",
        "target call attribution",
        "--reuse-evidence",
        "/existing/evidence.json",
    ]


@pytest.mark.parametrize("entry", ["bench", "raw"])
def test_operator_without_screen_only_dispatches_profile(gate, entry):
    mod, _, _, calls, _ = gate
    prefix = ["bench", "profile"] if entry == "bench" else ["raw", "bench", "profile"]
    assert mod.main(prefix + args()) == 0
    assert calls[-1] == [mod.TOOLS["bench"], "profile", "--tier", "light"]
    assert len([c for c in calls if c[0] == mod.TOOLS["bench"]]) == 1
    assert any("collect-check" in c for c in calls)


def test_check_only_does_not_launch(gate, monkeypatch):
    mod, _, _, calls, _ = gate
    monkeypatch.delattr(hashlib, "file_digest", raising=False)
    before = mod._ws().counters["hash_calls"]
    assert mod.profile_gate(args() + ["--check-only"]) == 0
    assert mod._ws().counters["hash_calls"] - before == 1
    assert all(c[0] != mod.TOOLS["bench"] for c in calls)


@pytest.mark.parametrize("key", ["l0", "l3"])
@pytest.mark.parametrize("status", ["FAIL", "PENDING", None])
def test_normal_profile_requires_both_pass(gate, key, status):
    mod, cand, _, calls, _ = gate
    cand["checks"][key]["status"] = status
    assert mod.profile_gate(args()) == 3
    assert not calls


@pytest.mark.parametrize("layer", ["graph", "system", "meta", None])
def test_no_screen_is_operator_only(gate, layer):
    mod, cand, routes, calls, _ = gate
    routes[0]["layer"] = layer
    cand["layer"] = "operator"
    assert mod.profile_gate(args()) == 3
    assert not calls


@pytest.mark.parametrize("stage", ["rejected", "inconclusive", "accepted"])
def test_terminal_cannot_enter_diagnostic(gate, stage):
    mod, cand, _, calls, _ = gate
    cand["stage"] = stage
    assert mod.profile_gate(args("mechanism-gap")) == 3
    assert not calls


@pytest.mark.parametrize("tier", ["light", "deep"])
def test_failed_l3_allows_only_evidenced_diagnostic(gate, tier, capsys):
    mod, cand, _, _, _ = gate
    cand["stage"] = "blocked"
    cand["checks"]["l3"]["status"] = "FAIL"
    assert mod.profile_gate(args("mechanism-gap", tier)) == 0
    assert cand["checks"]["l3"]["status"] == "FAIL"
    assert "禁止据此晋级" in capsys.readouterr().out


def test_deep_is_not_normal_ab(gate):
    mod, _, _, calls, _ = gate
    assert mod.profile_gate(args(tier="deep")) == 3
    assert not calls


@pytest.mark.parametrize("bad", ["missing", "hash", "empty", "malformed"])
def test_bad_evidence_blocks_before_collect(gate, bad):
    mod, cand, _, calls, evidence = gate
    if bad == "missing":
        cand["checks"]["l3"]["evidence"][0]["path"] = str(evidence) + ".absent"
    elif bad == "hash":
        cand["checks"]["l3"]["evidence"][0]["sha256"] = "0" * 64
    elif bad == "empty":
        cand["checks"]["l3"]["evidence"] = []
    else:
        cand["checks"] = []
    assert mod.profile_gate(args()) == 3
    assert not calls


def test_explicit_bad_screen_cannot_fall_back(gate):
    mod, _, _, calls, evidence = gate
    assert mod.profile_gate(args() + ["--screen-report", str(evidence)]) == 3
    assert not calls


def test_legacy_screen_unchanged_for_graph(gate, tmp_path):
    mod, cand, routes, _, evidence = gate
    routes[0]["layer"] = "graph"
    cand.pop("checks")
    report = tmp_path / "screen.json"
    report.write_text(
        json.dumps(
            {
                "schema": "opt-ops-screen.v1",
                "mode": "plain",
                "repetitions": 3,
                "artifacts": {
                    k: [str(evidence)] * n for k, n in [("l1", 3), ("l2", 3), ("l3", 1)]
                },
            }
        )
    )
    before = report.read_bytes()
    assert mod.profile_gate(args() + ["--screen-report", str(report)]) == 0
    assert report.read_bytes() == before
    cand["checks"] = {"l3": {"status": "FAIL"}}
    assert mod.profile_gate(args() + ["--screen-report", str(report)]) == 3
    assert report.read_bytes() == before
