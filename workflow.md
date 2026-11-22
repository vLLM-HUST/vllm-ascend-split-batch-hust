# workflow.md — 参考性执行流程

本文件记录本仓常用操作顺序。它只作参考。冲突时以 `AGENTS.md` 和权威文档为准。

## 1. 开始工作

1. 先执行 `git fetch origin main --prune`。
2. 比较本地 `main` 与 `origin/main`。本地落后且无分叉时，快进本地 `main`。
3. 不切换有未提交改动的工作区。用 `git worktree` 建独立目录。
4. 新分支基于当前 `main`。分支名使用 `openbitfun/<short-name>`。
5. 开工前确认 `git status --short --branch` 为 clean。

查看[实验计划](实验计划.md)的未完成项；找旧资料直接查[实验索引](INDEX.md)，不从工作树名称推断状态。研发结论只写在 `docs/optimization-history/` 对应记录里，未测也写明。索引仅生成链接，不改原件或指纹。增删资料后运行 `python -m tools.operator_tools.experiment_index --write`；如果补丁指纹不符，先核实原件，不用重建索引覆盖差异。

**新会话收到“继续优化”**：从仓根运行只读入口，不用复制本机队列或恢复聊天历史。

```bash
python -B -m tools.operator_tools.next
```

[当前研发任务](docs/optimization-next.json) 是选题的唯一记录，换方向时在同一 PR 更新范围、依据、跳过项、下一步、停止条件和归档指纹。它不授予执行或回填资格。归档换了挂载点时加 `--archive <目录>`。退出码 `0` 表示已核对记录及其归档输入，不表示当前宿主已通过预检；`2` 表示缺输入，只可静态设计；`1` 表示记录或身份核验失败，应停下。原件路径来自回执和结果清单，不从忽略目录补找。整网热点的读取方法见[热点判读](docs/hotspot-profile-method.md)。

## 2. 提交与 PR

1. 只暂存明确属于本任务的路径。不默认暂存整个工作区。
2. 提交前运行风格检查、离线检查和 `repo_audit.py`。
3. 修改 `ops/<operator>/` 覆盖文件后，重新生成 `checksum.sha256`。
4. 推送使用普通 push。不改写历史，不自动强推。pre-push 钩子会检查待推
   提交的 committer 邮箱，名单外邮箱先修复再推。
5. 创建 PR 前先按 head 分支查询现有 PR。存在时更新，不重复创建。
6. PR 创建后回读 head、base、draft、state 和 URL。
7. CI 失败时先读日志和失败上下文。修复用新提交，不改写已发布提交。
8. CI 全绿并合并后，再次 fetch，快进本地 `main`，新建验证分支。

## 3. 认证

1. 进程里优先读取 `GITHUB_TOKEN`，其次读取 `GH_TOKEN`。
2. 进程变量缺失时，只读取用户指定的环境文件。
3. Token 只留在单次进程内存。不打印，不写仓库，不写日志。
4. GitHub API 请求发往固定地址。不接受重定向。
5. 认证失败时停止远端写操作，不改用其他凭据来源。

## 4. 证据与审计

1. 原始日志、Trace、checkpoint 和临时计划放仓外或 `.openbitfun/`。
2. `.openbitfun/` 默认忽略。只跟踪 skill 白名单文件。
3. 仓内只放结构化证据、报告和索引。
4. 证据结论必须与 `DELIVERY_INDEX.json` 同向。
5. 数值失败时保持 default-off，不写服务收益标签。
6. 历史结论必须标明适用范围。不得复用为当前版本结论。
7. 清理过时证据前先搜查，再备份，后删除。删除后重算指纹。
8. `docs/evidence/` 是登记档案。不要按“过期”整体删除。

## 5. NPU 长测

1. 申请 NPU 前，在干净分支跑完整离线检查和 `repo_audit.py`。
2. 只有已评审、明确点名的服务复测才用以下命令；它不是选题入口：

```bash
SERVING_BASE_PLAN=/path/to/reviewed-plan.json bash workloads/run_pending_longrun.sh "$OPERATOR"
```

3. 脚本会生成计划，并用 nohup 启动 `serving_e2e_auto.py`。
4. 仓外目录保存原始归档。`.openbitfun/` 保存日志和 checkpoint。
5. 值守计划使用 `sync_mode: none`，不写单算子或组合证据。通用采集器未指定模式时也只归档。已评审的回填计划须显式指定 `auto`；任一层指定 `none` 均禁止写回。
6. 没有调用点或门控读者时必须 skip。调用点已接入但缺少归因规则时才可只做数值护栏。
7. 服务结论按主测负载判定。默认噪声底线为 `1.5%`。
8. 有效噪声取底线、关闭组波动、启用组波动中的最大值。
9. 差异不超过有效噪声时记 `within_noise`。记录组内波动和原始结果。
10. 不把微基准结果或算子耗时外推为服务收益。

## 6. 失败算子处理

1. 数值失败或归因失败不得进入当前交付面。
2. 删除交付目录前先备份目录和引用清单。
3. 同步清理索引、文档、工具登记和测试引用。
4. 清理后运行离线检查、风格检查和 `repo_audit.py`。
5. 修复后必须重新核验数值、调用和服务结论，才能恢复交付资格。

## 7. 合并后验证

1. 拉取最新 `main`。
2. 确认合并提交已进入 `origin/main`。
3. 快进本地 `main`。
4. 从新 `main` 新建验证分支和 worktree。
5. 再次跑完整离线检查和 `repo_audit.py`。
6. 保持旧 worktree 和主工作区不动，便于回溯。
