# docs/ — 知识库导航

本目录收录插件交付规范、架构契约、支持边界与发布检查。根目录 README 保留仓库状态、安装与证据入口；当前状态以明确的权威清单为准，历史原件只用于溯源。

## 阅读顺序(新人 15 分钟路径)

| 顺序 | 文档 | 回答的问题 |
|---|---|---|
| 1 | [../README.md](../README.md) | 这个仓库是什么、装完怎么验证 |
| 2 | [architecture.md](architecture.md) | 插件壳/kernel 库/宿主三者什么关系,manifest 各字段什么语义 |
| 3 | [container-and-plugin-boundary-spec.md](container-and-plugin-boundary-spec.md) | 容器存储与生命周期、宿主Fork与插件边界、知识库隔离规范（权威 Spec） |
| 4 | [agent-guide.md](agent-guide.md) | 在这个仓库写代码必须遵守什么(人机通用) |
| 5 | [support-matrix.md](support-matrix.md) | 能力的激活边界和宿主条件 |
| 6 | [delivery-inventory.md](delivery-inventory.md) | 当前组件、补丁、工具归属和收益证据状态 |
| 7 | [operator-benchmark-standard.md](operator-benchmark-standard.md) | 算子优化评测四级阶梯、形状事实合同与报告标准 |
| 8 | [npu-benchmark-execution-guide.md](npu-benchmark-execution-guide.md) | NPU 实测执行步骤、各算子测试命令与结果回填流程 |
| 9 | [release.md](release.md) | 构建、安装、发布、启用、停止和回滚判据 |
| 10 | [pitfalls.md](pitfalls.md) | 仍适用的运行约束与故障判据 |
| 11 | [progress.md](progress.md) | 项目当前状态 |

## 单一事实源分工

与仓库既有约定一致,每个主题只在一处维护,其余地方只放指针:

| 主题 | 权威位置 |
|---|---|
| 使用方法、env 开关、验证证据 | [../README.md](../README.md) |
| **容器运行时分层、宿主Fork边界与证据隔离规范** | [container-and-plugin-boundary-spec.md](container-and-plugin-boundary-spec.md) |
| 能力激活边界 × 代码位置 | [support-matrix.md](support-matrix.md) |
| 组件、补丁、工具与收益状态 | [delivery-inventory.md](delivery-inventory.md) |
| 算子测试标准、形状合同与验收报告 | [operator-benchmark-standard.md](operator-benchmark-standard.md) |
| **NPU 带卡环境实测与评测指南** | [npu-benchmark-execution-guide.md](npu-benchmark-execution-guide.md) |
| **cohort 边界（我们能声称什么）× 四项归属** | [cohort-and-ownership.md](cohort-and-ownership.md)（约束入口 [../AGENTS.md](../AGENTS.md) 硬约束 6） |
| 宿主提案补丁（tier-2，未提交上游） | [`host-proposals/README.md`](../host-proposals/README.md)（章程）+ `host-proposals/<id>/`（`proposal_meta.json` = env 语法/哈希/版本精确锁定/上游状态）；`docs/upstream/` 只是路径消歧 stub |
| 宿主 seam 与 monkeypatch 面清单 | [../HOST_CONTRACT.md](../HOST_CONTRACT.md) |
| 历史实现溯源（索引 + 逐条核对结论） | [../PROVENANCE.md](../PROVENANCE.md) / [provenance-verification.md](provenance-verification.md) / [../provenance/](../provenance/)(只读) |
| 算子交付形态：编译型实现独立交付；**Triton/纯 Python 算子实现由本仓承载** | [architecture.md](architecture.md) §2/§5 + 本仓 `src/vllm_ascend_split_batch/` |
| 算子调度(选择)接口 | [design/operator-dispatch.md](design/operator-dispatch.md)(骨架,未接线) |
| 架构语义、编程规范、发布、踩坑、进度 | 本目录 |

改行为必须同步改对应权威文档;本目录只放指针,不复制内容。
