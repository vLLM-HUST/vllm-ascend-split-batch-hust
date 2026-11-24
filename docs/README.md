# docs/ — 知识库导航

本目录收录插件交付规范、架构契约、支持边界与发布检查。根目录 README 保留仓库状态、安装与证据入口；当前状态以明确的权威清单为准，历史原件只用于溯源。

查实验材料直接用[实验索引](../INDEX.md)，不在本页另列实验清单。排期见[实验计划](../实验计划.md)。

## 日常核心入口速查表（现役 vs 只读溯源区）

| 分类 | 核心文件 | 用途说明 |
|---|---|---|
| **现役：下一步与排期** | [optimization-next.json](optimization-next.json)、[../实验计划.md](../实验计划.md) | 唯一选题依据、停止条件与当前推进待办（命令行只读入口：`python -B -m tools.operator_tools.next`） |
| **现役：算子交付总账** | [../DELIVERY_INDEX.json](../DELIVERY_INDEX.json)、[../ops/README.md](../ops/README.md) | 全部已登记算子的归属层、门控开关、数值护栏与服务结论 |
| **现役：规范与流程** | [../AGENTS.md](../AGENTS.md)、[../workflow.md](../workflow.md)、[../GLOSSARY.md](../GLOSSARY.md) | 工作契约、参考执行流程与统一术语表 |
| **现役：资料检索** | [../INDEX.md](../INDEX.md) | 自动生成的全仓资料索引（关键词查找：`--find <词>`；按分组查看：`--group`） |
| **只读溯源区** | [../EVIDENCE_AVAILABILITY.json](../EVIDENCE_AVAILABILITY.json)、[../EVIDENCE_SOURCES.json](../EVIDENCE_SOURCES.json)、[../MOD_METADATA.json](../MOD_METADATA.json)、[evidence/](evidence/)、[../provenance/legacy-patches/](../provenance/legacy-patches/) | 历史归档与组织审计留痕，仅作溯源，不代表当前能力或收益 |

## 按任务的最短路径(改第一个算子)

只改一个算子时按这个顺序读,不用通读全目录:

1. [ops/README.md](../ops/README.md) — 一个算子目录里必须放哪几个文件、目录指纹怎么来;
2. [ops/gdn_norm_gate/README.md](../ops/gdn_norm_gate/README.md) 配同目录 `evidence.json` — 照着现成交付目录填结构;
3. [operator-benchmark-standard.md](operator-benchmark-standard.md) — 四级阶梯、形状事实合同、报告模板与交付合规门禁;
4. [npu-benchmark-execution-guide.md](npu-benchmark-execution-guide.md) — 带卡怎么跑、结果怎么回填;
5. [delivery-inventory.md](delivery-inventory.md) 的「性能证据最小字段与评测阶梯」节与 [../DELIVERY_INDEX.json](../DELIVERY_INDEX.json) — 结论标签怎么写、状态怎么登记;
6. [agent-guide.md](agent-guide.md) — 编码与测试纪律,改完按 [../AGENTS.md](../AGENTS.md) 的「提交前检查」节跑验证命令。

## 阅读顺序(新人 15 分钟路径)

| 顺序 | 文档 | 回答的问题 |
|---|---|---|
| 1 | [../README.md](../README.md) | 这个仓库是什么、装完怎么验证 |
| 2 | [../GLOSSARY.md](../GLOSSARY.md) | 术语怎么读：一个词指什么、同名异义怎么分、哪些词已废止 |
| 3 | [architecture.md](architecture.md) | 插件壳/kernel 库/宿主三者什么关系,manifest 各字段什么语义 |
| 4 | [container-and-plugin-boundary-spec.md](container-and-plugin-boundary-spec.md) | 容器存储与生命周期、宿主Fork与插件边界、知识库隔离规范（权威 Spec） |
| 5 | [agent-guide.md](agent-guide.md) | 在这个仓库写代码必须遵守什么(人机通用) |
| 6 | [support-matrix.md](support-matrix.md) | 能力的激活边界和宿主条件 |
| 7 | [delivery-inventory.md](delivery-inventory.md) | 当前组件、补丁、工具归属和收益证据状态 |
| 8 | [operator-benchmark-standard.md](operator-benchmark-standard.md) | 算子优化评测四级阶梯、形状事实合同、报告标准与交付合规门禁 |
| 9 | [npu-benchmark-execution-guide.md](npu-benchmark-execution-guide.md) | NPU 实测执行步骤、各算子测试命令与结果回填流程 |
| 10 | [release.md](release.md) | 构建、安装、发布、启用、停止和回滚判据 |
| 11 | [pitfalls.md](pitfalls.md) | 仍适用的运行约束与故障判据 |
| 12 | [算子优化 Skill](../.openbitfun/skills/operator-sol-optimizer/SKILL.md) | SOL 计算、单点优化与证据交接 |
| 13 | [hotspot-profile-method.md](hotspot-profile-method.md) | 整网 profile 怎么采、怎么判读，哪些结论不能从里面得出 |

## 单一事实源分工

与仓库既有约定一致,每个主题只在一处维护,其余地方只放指针:

| 主题 | 权威位置 |
|---|---|
| 术语唯一写法、定义判据、代码锚点、同名异义与废止词 | [../GLOSSARY.md](../GLOSSARY.md) |
| 继续优化：当前候选、依据和停止条件 | [当前研发任务](optimization-next.json)；只读入口见 [workflow.md](../workflow.md) |
| 参考性执行流程 | [../workflow.md](../workflow.md) |
| 使用方法、env 开关、验证证据 | [../README.md](../README.md) |
| **容器运行时分层、宿主Fork边界与证据隔离规范** | [container-and-plugin-boundary-spec.md](container-and-plugin-boundary-spec.md) |
| 一个算子目录放什么（必备文件、证据契约、目录指纹） | [ops/README.md](../ops/README.md)（原件）+ [delivery-inventory.md](delivery-inventory.md) §5 的字段合同 |
| 能力激活边界 × 代码位置 | [support-matrix.md](support-matrix.md) |
| 组件、补丁、工具与收益状态 | [delivery-inventory.md](delivery-inventory.md) |
| MOD 最终验收、冻结协议与四项目标 | [mod-acceptance-standard.md](mod-acceptance-standard.md) |
| MOD 当前缺项与证据核查 | [mod-acceptance-audit.md](mod-acceptance-audit.md)、[结构化记录](mod-acceptance/evidence.json) |
| MOD 验收 Agent 流程 | [mod-acceptance skill](../.openbitfun/skills/mod-acceptance/SKILL.md) |
| 算子测试标准、形状合同与验收报告 | [operator-benchmark-standard.md](operator-benchmark-standard.md) |
| **热点 profile 采集口径、pipe 判读与结论边界** | [hotspot-profile-method.md](hotspot-profile-method.md) |
| **冻结请求范围、参考服务配置与本地诊断的适用范围** | [acceptance-shape-alignment.md](acceptance-shape-alignment.md) + [结构化对照](acceptance-shape.json) |
| **遍测遍验模式（NPU 持卡保活循环的规则与队列 schema）** | [sweep-verify-plan.md](sweep-verify-plan.md)（实时队列在 `.openbitfun/sweep-verify-queue.json`，不进 git） |
| **NPU 带卡环境实测与评测指南** | [npu-benchmark-execution-guide.md](npu-benchmark-execution-guide.md) |
| **cohort 边界（我们能声称什么）× 四项归属** | [cohort-and-ownership.md](cohort-and-ownership.md)（约束入口：[../AGENTS.md](../AGENTS.md) 的「证据与结论」节） |
| 宿主提案补丁（host proposal，逐项记录提交与部署状态） | [`host-proposals/README.md`](../host-proposals/README.md)（章程）+ `host-proposals/<id>/`（`proposal_meta.json` = env 语法/哈希/版本精确锁定/上游状态）；`docs/upstream/` 只是路径消歧 stub |
| 宿主 seam 与 monkeypatch 面清单 | [HOST_CONTRACT.md](HOST_CONTRACT.md) |
| 历史实现溯源（索引 + 逐条核对结论） | [PROVENANCE.md](PROVENANCE.md) / [provenance-verification.md](provenance-verification.md) / [../provenance/](../provenance/)(只读) |
| 历史证据与原始度量保管规范 | [EVIDENCE.md](EVIDENCE.md) |
| 算子优化与证据交接 | [算子优化 Skill](../.openbitfun/skills/operator-sol-optimizer/SKILL.md) |
| 算子交付形态：编译型实现独立交付；**Triton/纯 Python 算子实现由本仓承载** | [architecture.md](architecture.md) §5 + 本仓 `src/vllm_ascend_split_batch/` |
| 算子调度(选择)接口 | [design/operator-dispatch.md](design/operator-dispatch.md)(骨架,未接线,实现暂存 `experiments/`) |
| 架构语义、编程规范、发布、踩坑、进度 | 本目录 |
| 设计稿与历史记录（只读，不是仓库现状） | [design/fia-decode-demask.md](design/fia-decode-demask.md)（未实施设计稿：仓库内无实现、无入口、无开关）、[design/rope-variant-defects.md](design/rope-variant-defects.md)（宿主 fork 三个缺陷的读码结论）、[方向分支证据批次](EVIDENCE.md#direction-branch-batch) |

改行为必须同步改对应权威文档;本目录只放指针,不复制内容。
