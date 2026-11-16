"""Offline arithmetic and command construction, without device claims."""

import importlib.util
import math
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
SCRIPTS = ROOT / ".openbitfun/skills/operator-sol-optimizer/scripts"


def load(name):
    spec = importlib.util.spec_from_file_location(name, SCRIPTS / (name + ".py"))
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_memory_bound_and_optimization_ceiling():
    result = load("sol").calculate(20, {"cube": [1e8, 1e14], "hbm": [1e7, 1e12]})
    assert result["lower_bound_us"] == pytest.approx(10)
    assert result["sol_pct"] == pytest.approx(50)
    assert result["max_speedup"] == pytest.approx(2)
    assert result["reducible_fraction"] == pytest.approx(0.5)


def test_inconsistent_bound_is_reported_without_clamping():
    result = load("sol").calculate(5, {"hbm": [1e7, 1e12]})
    assert result["sol_pct"] == pytest.approx(200)
    assert result["consistent"] is False
    assert result["max_speedup"] is None


@pytest.mark.parametrize("value", [0, -1, math.inf, math.nan, True])
def test_invalid_measurement(value):
    with pytest.raises(ValueError):
        load("sol").calculate(value, {"hbm": [10, 100]})


@pytest.mark.parametrize(
    "resources",
    [{}, {"x": [0, 1]}, {"x": [-1, 1]}, {"x": [1, 0]}, {"x": [math.inf, 1]}],
)
def test_invalid_resource_budget(resources):
    with pytest.raises(ValueError):
        load("sol").calculate(10, resources)


def test_profile_dry_run_does_not_execute(tmp_path, capsys):
    profile = load("profile")
    output = tmp_path / "trace"
    assert (
        profile.main(
            [
                "timeline",
                "--application",
                "python case.py",
                "--output",
                str(output),
                "--dry-run",
            ]
        )
        == 0
    )
    assert not output.exists()
    assert "--application=python case.py" in capsys.readouterr().out


def test_profile_rejects_repository_output():
    with pytest.raises(ValueError, match="outside"):
        load("profile").main(
            [
                "timeline",
                "--application",
                "python case.py",
                "--output",
                str(ROOT / "build/trace"),
                "--dry-run",
            ]
        )


def test_profile_preserves_failure_exit(tmp_path, monkeypatch):
    import subprocess

    profile = load("profile")
    monkeypatch.setattr(
        profile.subprocess, "run", lambda *a, **k: subprocess.CompletedProcess(a[0], 7)
    )
    assert (
        profile.main(
            [
                "timeline",
                "--application",
                "python case.py",
                "--output",
                str(tmp_path / "trace"),
            ]
        )
        == 7
    )


@pytest.mark.parametrize("mode", ["metrics", "simulator"])
def test_profile_presets(mode, tmp_path, capsys):
    argv = [
        mode,
        "--application",
        "python case.py",
        "--output",
        str(tmp_path / "new"),
        "--dry-run",
    ]
    if mode == "simulator":
        argv += ["--kernel", "kernel_name", "--soc", "test_soc"]
    assert load("profile").main(argv) == 0
    import json

    command = json.loads(capsys.readouterr().out)["command"]
    if mode == "simulator":
        assert command[1:3] == ["op", "simulator"]
        assert "--soc-version=test_soc" in command
    else:
        assert "--aic-metrics=PipeUtilization" in command


def test_skill_references_and_evaluation_scope():
    import json
    import re

    skill = SCRIPTS.parent / "SKILL.md"
    text = skill.read_text()
    assert text.startswith("---\nname: operator-sol-optimizer\ndescription: Use when")
    links = re.findall(r"\]\(([^)]+)\)", text)
    assert links
    for relative in links:
        assert (skill.parent / relative).is_file(), relative
    evaluations = json.loads((skill.parent / "evals/cases.json").read_text())
    assert evaluations["baseline"] == evaluations["with_skill"] == "not_run"


def test_optimizer_workflow_reference_exists_and_covers_current_flow():
    skill = SCRIPTS.parent / "SKILL.md"
    text = skill.read_text()
    workflow_link = "references/optimization-workflow.md"
    assert workflow_link in text
    workflow = SCRIPTS.parent / workflow_link
    assert workflow.is_file()
    body = workflow.read_text()
    assert "单侧候选" in body
    assert "端侧长测" in body
    assert "fail-closed" in body
    assert "pending_longrun.py" in body
    assert "sync_mode: none" in body
    assert "serving_sync.validate" in body


def test_active_entrypoints_do_not_reference_removed_optimizer():
    removed = (
        "tools/" + "op_agent",
        "tools." + "op_agent",
        "op-agent-" + "harness-spec.md",
        "collect_" + "profile.sh",
        "--update-" + "op-dir",
    )
    files = [
        ROOT / name
        for name in (
            "AGENTS.md",
            "GLOSSARY.md",
            "docs/README.md",
            "docs/operator-benchmark-standard.md",
            "docs/outer-loop-test-plan.md",
            "docs/npu-benchmark-execution-guide.md",
        )
    ]
    for folder in ("tools/operator_tools", "workloads", "experiments/ascend910b-2card"):
        files.extend(
            p
            for p in (ROOT / folder).rglob("*")
            if p.suffix in {".py", ".md", ".sh"} and p.is_file()
        )
    files.extend((ROOT / ".openbitfun/skills").glob("*/SKILL.md"))
    for path in files:
        text = path.read_text()
        assert not any(old in text for old in removed), path.relative_to(ROOT)
