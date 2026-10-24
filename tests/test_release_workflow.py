import os
import subprocess
import textwrap
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
RELEASE = ROOT / ".github/workflows/release.yml"
STEP = (
    "      - name: Assemble the release body from the annotated tag\n        run: |\n"
)


def git(repo, *args):
    return subprocess.run(
        ["git", "-C", str(repo), *args],
        check=True,
        capture_output=True,
        text=True,
    )


def body_script():
    source = RELEASE.read_text()
    assert STEP in source, "release body step renamed; update this guard"
    return textwrap.dedent(source.split(STEP, 1)[1].split("\n      - ", 1)[0])


def test_release_body_carries_tag_annotation_not_commit_message(tmp_path):
    origin = tmp_path / "origin.git"
    subprocess.run(["git", "init", "-q", "--bare", str(origin)], check=True)
    work = tmp_path / "work"
    work.mkdir()
    git(work, "init", "-q", "--initial-branch", "main")
    git(work, "config", "user.email", "release@example.com")
    git(work, "config", "user.name", "release")
    (work / "runtime.py").write_text("VALUE = 1\n")
    git(work, "add", "runtime.py")
    git(work, "commit", "-qm", "commit subject that must not leak")
    git(work, "tag", "-a", "v9.9.9", "-m", "annotation that must appear")
    git(work, "remote", "add", "origin", str(origin))
    git(work, "push", "-q", "origin", "main", "v9.9.9")
    commit = git(work, "rev-parse", "HEAD").stdout.strip()

    git(work, "tag", "-f", "v9.9.9", "HEAD")
    assert git(work, "cat-file", "-t", "v9.9.9").stdout.strip() == "commit"

    subprocess.run(
        ["bash", "-e", "-o", "pipefail", "-c", body_script()],
        cwd=work,
        env={**os.environ, "GITHUB_REF_NAME": "v9.9.9"},
        check=True,
        capture_output=True,
        text=True,
    )

    body = (work / "release-body.md").read_text()
    assert "annotation that must appear" in body
    assert "commit subject that must not leak" not in body
    assert "pypi.org/project/vllm-ascend-split-batch/9.9.9/" in body
    assert git(work, "cat-file", "-t", "v9.9.9").stdout.strip() == "tag"
    assert git(work, "rev-parse", "v9.9.9^{}").stdout.strip() == commit
