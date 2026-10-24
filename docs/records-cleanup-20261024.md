# 证据库清理事故（2026-10-24）——仓内指针

正文在 L2，本仓只放指针，避免两份真相：

- 事故与恢复核查：`/root/operator-experiments/records/notes/20261024T0255Z_raw-cleanup-incident.md`
- 重测计划：`/root/operator-experiments/records/notes/20261024T0310Z_remeasure-plan.md`
- 删除清单（含批准范围）：`/root/operator-experiments/records/notes/20261024T0250Z_raw-cleanup.json`
- 工作台裁决：candidate `records-profile-metadata-loss-20261024`（route `028/global`，stage `validating`）

## 一句话结论

清理 `records/raw`（1.18 G）与 `profile/` 14 个与 `ladder/raw` 重名的窗口（1.58 G）时，
引用匹配漏了解析软链别名 `/root/prof`（→ `profile/`），把**被引用**的 profile 侧元数据误判为零引用并删除。
`records/ladder/raw`（49 GB / 37 窗口）、全部解析结果、`probes`、`tree-patches` 未受影响。

## 影响与处置

| 项 | 状态 |
|---|---|
| profiler 转储路径（`/root/prof/<win>/…`） | 已用软链 `profile/<win> → ../records/ladder/raw/<win>` 恢复 |
| `cohort.json` | 原件丢；内容可从 34 份解析产物的 `cohort` 字段读取（不得回填为原件） |
| `profile-plan.json` | 原件丢；参数可推导 |
| `load-manifest.json` / `load-exec.json` | 原件丢，无副本 |
| `metrics-before/after.txt` | 原件丢，属时点快照，重测不可复原 |
| `ladder-1017-093130` tp0 转储 | 原件丢，需重新采集 |
| `CURRENT.json` 4 条指针（`028/D3-gdn-high-concurrency-longctx`） | 仍失效，未改动，待 owner 裁决 |

处置决定：无可用快照，不回滚，**之后重测**；重测窗口按新时间戳登记，不回填原目录。

## 防再发

删除任何证据目录前：① 先 `realpath` 解析软链别名（至少 `/root/prof`、`/root/包体恢复/{records,logs}`）；
② 以引用路径的**文件级**存在性为判据，不接受目录名匹配；③ 单窗口存在任一未解析引用即整窗保留。
