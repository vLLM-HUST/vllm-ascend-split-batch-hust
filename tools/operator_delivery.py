# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 vLLM-HUST contributors.

import argparse
import base64
import json
import subprocess
import tempfile
from pathlib import Path, PurePosixPath

REPO = Path(__file__).resolve().parents[1]
CONTRACT = {
    "objective": "交付到目标插件仓：实现 + 测试 + 可回滚接入 + 证据与边界",
    "separate_states": ["实验完成", "优化验收", "代码提交", "远端发布"],
    "failure_action": "数值失败保持 default-off；提交复现、测试和负结果，不宣称验收",
    "stop_rule": "L2 回执或 L4 结束不等于已交付；先核验本轮路径和 commit",
    "push_rule": "push 须有用户授权，并以实际 push 和远端 hash 核验；不信缓存",
    "scope_rule": "只暂存本轮明确路径；无关 dirty 文件不是停止任务或批量提交的理由",
}


def git(repo, *args):
    result = subprocess.run(
        ["git", "--no-optional-locks", "-C", str(repo), *args],
        capture_output=True,
        text=True,
        timeout=15,
    )
    if result.returncode:
        raise ValueError(result.stderr.strip() or "git command failed")
    return result.stdout.rstrip("\n")


def validate_scope(repo, paths):
    if Path(git(repo, "rev-parse", "--show-toplevel")).resolve() != repo:
        raise ValueError("target must be a repository root")
    for path in paths:
        parts = PurePosixPath(path).parts
        if not parts or path.startswith(("/", ":", "-")) or ".." in parts:
            raise ValueError("scope requires repository-relative literal paths")
        if path == "." or any(c in path for c in "*?[\\"):
            raise ValueError("scope must not be the entire repository or a glob")


def capture_dirty_baseline(repo, paths):
    repo = Path(repo).resolve()
    validate_scope(repo, paths)
    if not paths:
        raise ValueError("dirty baseline requires explicit file paths")
    files = {}
    for path in paths:
        literal = ":(literal)" + path
        if git(repo, "status", "--porcelain=v1", "--", literal) != " M " + path:
            raise ValueError("baseline accepts only unstaged modified tracked files")
        file = repo / path
        if not file.is_file() or file.is_symlink() or file.resolve() != file.absolute():
            raise ValueError("baseline requires regular non-symlink files")
        files[path] = {
            "content": base64.b64encode(file.read_bytes()).decode("ascii"),
            "mode": file.stat().st_mode & 0o777,
        }
    return {
        "schema": "operator-dirty-baseline.v1",
        "repo": str(repo),
        "base": git(repo, "rev-parse", "HEAD"),
        "files": files,
    }


def preserved_dirty(repo, path, baseline):
    entry = baseline.get("files", {}).get(path)
    file = repo / path
    if (
        not entry
        or not file.is_file()
        or file.is_symlink()
        or file.resolve() != file.absolute()
        or file.stat().st_mode & 0o777 != entry["mode"]
    ):
        return False
    literal = ":(literal)" + path
    if git(repo, "status", "--porcelain=v1", "--", literal) != " M " + path:
        return False
    trees = [
        git(repo, "ls-tree", rev, "--", literal) for rev in (baseline["base"], "HEAD")
    ]
    if not all(trees) or trees[0].split()[0] != trees[1].split()[0]:
        return False
    with tempfile.TemporaryDirectory(prefix="delivery-merge-") as tmp:
        paths = [Path(tmp) / name for name in ("head", "base", "dirty")]
        for dest, rev in zip(paths[:2], ("HEAD", baseline["base"]), strict=True):
            result = subprocess.run(
                ["git", "-C", str(repo), "show", f"{rev}:{path}"],
                capture_output=True,
                timeout=15,
                check=True,
            )
            dest.write_bytes(result.stdout)
        paths[2].write_bytes(base64.b64decode(entry["content"], validate=True))
        merged = subprocess.run(
            ["git", "merge-file", "-p", *map(str, paths)],
            capture_output=True,
            timeout=15,
        )
    return merged.returncode == 0 and merged.stdout == file.read_bytes()


def collect(repo=REPO, base=None, paths=(), dirty_baseline=None):
    repo = Path(repo).resolve()
    validate_scope(repo, paths)
    report = {
        "schema": "operator-delivery.v1",
        "repo": str(repo),
        "branch": git(repo, "branch", "--show-current"),
        "head": git(repo, "rev-parse", "HEAD"),
        "base": base,
        "paths": list(paths),
        "contract": CONTRACT,
        "state": "UNSCOPED",
        "acceptance": "NOT_CHECKED",
        "remote": "NOT_VERIFIED",
        "checks": [],
        "next_action": "记录起点 HEAD 和本轮路径；收尾用 opt delivery --check 核验",
    }
    if not base or not paths:
        return report
    base = git(repo, "rev-parse", "--verify", base + "^{commit}")
    git(repo, "merge-base", "--is-ancestor", base, "HEAD")
    report["base"] = base
    if dirty_baseline is not None:
        if (
            not isinstance(dirty_baseline, dict)
            or dirty_baseline.get("schema") != "operator-dirty-baseline.v1"
            or dirty_baseline.get("repo") != str(repo)
            or dirty_baseline.get("base") != base
            or not isinstance(dirty_baseline.get("files"), dict)
        ):
            raise ValueError("dirty baseline must match repository and starting HEAD")
        validate_scope(repo, dirty_baseline["files"])
    for path in paths:
        literal = ":(literal)" + path
        dirty = bool(git(repo, "status", "--porcelain=v1", "--", literal))
        changed = bool(git(repo, "diff", "--name-only", base, "HEAD", "--", literal))
        preserved = (
            dirty
            and dirty_baseline is not None
            and preserved_dirty(repo, path, dirty_baseline)
        )
        report["checks"].append(
            {
                "path": path,
                "dirty": dirty,
                "preexisting_dirty_preserved": preserved,
                "changed_since_base": changed,
            }
        )
    committed = all(
        (not c["dirty"] or c["preexisting_dirty_preserved"]) and c["changed_since_base"]
        for c in report["checks"]
    )
    report["state"] = "COMMITTED_LOCAL" if committed else "INCOMPLETE"
    report["next_action"] = (
        "本轮路径已落本地提交；另核验测试、证据、验收状态及获授权的远端 push"
        if committed
        else "核对未提交或未相对起点变更的路径；只处理本轮文件"
    )
    return report


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--repo", type=Path, default=REPO)
    parser.add_argument("--base")
    parser.add_argument("--path", action="append", default=[])
    parser.add_argument("--check", action="store_true")
    parser.add_argument("--json", action="store_true")
    group = parser.add_mutually_exclusive_group()
    group.add_argument("--capture-dirty-baseline", type=Path)
    group.add_argument("--dirty-baseline", type=Path)
    args = parser.parse_args()
    try:
        if args.capture_dirty_baseline:
            if args.base or args.check:
                raise ValueError("capture is a startup operation, not a delivery check")
            baseline = capture_dirty_baseline(args.repo, args.path)
            with args.capture_dirty_baseline.open("x") as stream:
                json.dump(baseline, stream, ensure_ascii=False, indent=2)
                stream.write("\n")
            print(
                json.dumps(
                    {
                        "state": "BASELINE_CAPTURED",
                        "base": baseline["base"],
                        "paths": args.path,
                    }
                )
            )
            return 0
        baseline = (
            json.loads(args.dirty_baseline.read_text()) if args.dirty_baseline else None
        )
        report = collect(args.repo, args.base, args.path, baseline)
    except (
        ValueError,
        TypeError,
        KeyError,
        OSError,
        subprocess.SubprocessError,
    ) as exc:
        print(json.dumps({"state": "ERROR", "error": str(exc)}, ensure_ascii=False))
        return 4
    if args.json:
        print(json.dumps(report, ensure_ascii=False, indent=2))
    else:
        print(f"交付: {report['repo']} | {report['branch']} | {report['state']}")
        print(f"HEAD: {report['head']} | remote: {report['remote']}")
        print(report["next_action"])
        for check in report["checks"]:
            print(json.dumps(check, ensure_ascii=False))
    return 3 if args.check and report["state"] != "COMMITTED_LOCAL" else 0


if __name__ == "__main__":
    raise SystemExit(main())
