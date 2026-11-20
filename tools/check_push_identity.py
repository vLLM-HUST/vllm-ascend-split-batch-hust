#!/usr/bin/env python3
"""push 前提交身份检查。

CI 的 auto-merge 要求 PR 内每个提交的 committer.login 是 Raing5Days。
login 由 GitHub 按提交者邮箱映射，本地等价判据是：committer 邮箱必须在
允许名单内。本脚本在 push 前拦下不合格提交，避免 PR 检查全绿却无法合入。

默认名单只有主邮箱。新增账号邮箱时用环境变量扩展，不改编码：

    PUSH_COMMITTER_EMAILS=a@x.com,b@x.com pre-commit run --hook-stage pre-push

由 pre-push 钩子调用时从参数读取待推引用；手工运行时检查所有未推送提交。
"""

import os
import subprocess
import sys

DEFAULT_EMAILS = "M202574102@hust.edu.cn"
ZERO_SHA = "0" * 40


def git(*args: str) -> str:
    proc = subprocess.run(["git", *args], capture_output=True, text=True, check=False)
    if proc.returncode != 0:
        raise SystemExit(f"git {' '.join(args)} 失败: {proc.stderr.strip()}")
    return proc.stdout


def allowed_emails() -> set[str]:
    raw = os.environ.get("PUSH_COMMITTER_EMAILS", DEFAULT_EMAILS)
    return {e.strip().lower() for e in raw.split(",") if e.strip()}


def pushed_ranges(argv: list[str]) -> list[tuple[str, str]]:
    """从 pre-push 参数解析 (remote_sha, local_sha) 列表。"""
    tokens = " ".join(argv).split()
    ranges = []
    for i in range(0, len(tokens) - 3, 4):
        local_sha, remote_sha = tokens[i + 1], tokens[i + 3]
        if local_sha.startswith("0") or not local_sha:
            continue  # 删除远端分支，无新提交
        ranges.append((remote_sha, local_sha))
    if not ranges:
        # 手工运行：检查所有未推送提交
        out = git("rev-list", "HEAD", "--not", "--remotes")
        shas = out.split()
        return [("NONE", sha) for sha in shas] if shas else []
    return ranges


def committers(remote_sha: str, local_sha: str) -> list[tuple[str, str]]:
    """返回范围内每个提交的 (sha, committer email)。"""
    if remote_sha.startswith("0") or remote_sha == "NONE":
        out = git("rev-list", local_sha, "--not", "--remotes")
        shas = out.split()
    else:
        shas = git("rev-list", f"{remote_sha}..{local_sha}").split()
    rows = []
    for sha in shas:
        email = git("log", "-1", "--format=%ce", sha).strip()
        rows.append((sha, email))
    return rows


def main() -> int:
    allow = allowed_emails()
    offenders = []
    for remote_sha, local_sha in pushed_ranges(sys.argv[1:]):
        for sha, email in committers(remote_sha, local_sha):
            if email.lower() not in allow:
                offenders.append((sha, email))

    if not offenders:
        return 0

    print("以下提交的 committer 邮箱不在允许名单内，CI auto-merge 会拒绝整条 PR：")
    for sha, email in offenders:
        subject = git("log", "-1", "--format=%s", sha).strip()
        print(f"  {sha[:8]} <{email}> {subject}")
    print()
    print(f"允许名单: {', '.join(sorted(allow))}")
    print("修复方式：git config user.email 设为名单内邮箱，然后只重写本分支")
    print("未推送的提交（git rebase --committer-date-is-author-date --fork-point）。")
    print("不要用 filter-branch 扫全仓：它会剥掉 main 上签名提交的")
    print("gpgsig，整条历史换 SHA。")
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
