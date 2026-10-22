import os
import re
import subprocess
import textwrap
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
WORKFLOWS = ROOT / ".github/workflows"
POLICY_REF = "0cfe5109551e9b1c30ff3a75a2d93ee6ba8c9aa4"


def workflow(name):
    return (WORKFLOWS / name).read_text()


def test_policy_jobs_use_same_trusted_revision():
    for name in ("organization-benchmark-custody.yml", "custody-policy-tests.yml"):
        source = workflow(name)
        assert re.findall(r"^ +repository: (.+)$", source, re.M) == [
            "vLLM-HUST/.github"
        ]
        assert re.findall(r"^ +ref: (.+)$", source, re.M) == [POLICY_REF]
        assert "persist-credentials: false" in source
        assert "contents: write" not in source
        assert "github.event.pull_request.head" not in source
    assert "python -m unittest discover -s tests" in workflow(
        "custody-policy-tests.yml"
    )


def test_custody_preserves_manual_and_pr_entry_points():
    source = workflow("organization-benchmark-custody.yml")
    assert "  workflow_dispatch:\n" in source
    assert "  pull_request_target:\n" in source
    assert "types: [opened, reopened, synchronize, ready_for_review]" in source
    assert (
        "PR_NUMBER: ${{ github.event.pull_request.number || inputs.pr_number }}"
        in source
    )
    assert "--paginate --slurp" in source
    assert '--expected "$EXPECTED"' in source


@pytest.mark.parametrize("number", ["", "abc", "9/../10", "9;echo unsafe", "9\n10"])
def test_custody_rejects_invalid_number_before_api(number, tmp_path):
    result = run_custody(number, tmp_path)
    assert result.returncode == 1
    assert result.stdout == ""


@pytest.mark.parametrize("failure", ["none", "count", "pages", "policy"])
def test_custody_api_and_policy_failure_propagation(failure, tmp_path):
    result = run_custody("9", tmp_path, failure)
    assert result.returncode == (0 if failure == "none" else 7)
    if failure == "none":
        assert result.stdout.splitlines() == [
            "api repos/example/mod/pulls/9 --jq .changed_files",
            "api --paginate --slurp repos/example/mod/pulls/9/files?per_page=100",
            "policy:2",
        ]


def run_custody(number, tmp_path, failure="none"):
    source = workflow("organization-benchmark-custody.yml")
    script = textwrap.dedent(source.split("        run: |\n", 1)[1])
    script = script.replace("/tmp/custody-pr-files.json", str(tmp_path / "files.json"))
    helpers = """
    gh() {
        printf '%s\n' "$*" >&2
        case "$*" in
          *--jq*) [ "$FAILURE" != count ] || return 7; printf '2\n';;
          *) [ "$FAILURE" != pages ] || return 7; printf '[[]]\n';;
        esac
    }
    python() {
        [ "$FAILURE" != policy ] || return 7
        [ "$5" = --expected ] && [ "$6" = 2 ] || return 8
        printf 'policy:%s\n' "$6" >&2
    }
    """
    return subprocess.run(
        ["bash", "-e", "-o", "pipefail", "-c", textwrap.dedent(helpers) + script],
        env={
            **os.environ,
            "PR_NUMBER": number,
            "REPOSITORY": "example/mod",
            "FAILURE": failure,
        },
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        check=False,
    )
