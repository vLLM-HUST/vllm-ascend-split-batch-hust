---
name: github-env-auth
description: Use when 已获授权的 GitHub API 或 PR 操作缺少进程认证变量，而用户说明 Token 位于指定 .env 文件中。
---

# GitHub 环境文件认证

只补齐已有 GitHub 操作的认证，不扩大仓库、权限或操作范围。提交与 PR 流程沿用 commit-push-pr。

1. 先在进程内检查 `GITHUB_TOKEN`，其次 `GH_TOKEN`；只报告是否可用，不输出值。
2. 缺失时只使用用户明确指定的环境文件；用户只说 `.env` 时，指当前仓库根目录的 `.env`。文件不存在则报告缺失，不递归搜索其他目录、不读取其他凭据文件。
3. 不用 Read、cat、grep、env、printenv 或 shell 调试输出文件内容及认证变量。先确认环境文件未被 Git 跟踪；仓内 `.env` 应被忽略，绝不暂存。已被跟踪则停止使用并报告。
4. 在发起请求的同一进程内解析文件，只接受 `GITHUB_TOKEN` 和 `GH_TOKEN`。支持空行、注释、可选 `export` 和单行引号值；不执行文件，不使用 source、eval、命令替换或变量插值。遇到歧义格式只报变量名与格式问题，不打印原行。已有进程变量优先，文件中优先 `GITHUB_TOKEN`。
5. Token 只留内存，通过 HTTPS 请求头发送给固定的 `https://api.github.com`；拒绝重定向。不把 Token 放入命令行参数、URL、Git remote、临时文件、日志、PR 或异常回显。只加载到单次操作需要的进程，不写入 shell 配置。
6. 对已推送分支先查询准确 head/base 是否已有 PR，存在则沿用，没有则按已授权内容创建草稿。写请求超时先查询远端结果，再决定重试，避免重复创建。
7. 只输出 HTTP 状态、PR 编号和 URL。认证失败不尝试其他凭据来源；保留已完成的提交和推送结果，如实报告阻塞。

验证使用合成环境文件，覆盖环境变量优先、文件回退、缺文件和错误格式，确认输出不含测试值。结构检查通过不等于 GitHub 认证成功；真实 PR 必须回读其 head、base、draft 和 URL。没有独立 Agent 对照时明确标注未做行为评测。
