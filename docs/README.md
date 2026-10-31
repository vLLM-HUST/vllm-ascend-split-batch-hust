# docs/ — 知识库导航

本目录收录插件交付规范、架构契约、支持边界与发布检查。根目录 README 保留仓库状态、安装与证据入口；当前状态以明确的权威清单为准，历史原件只用于溯源。

## 阅读顺序 (新人 10 分钟路径)

| 顺序 | 文档 | 回答的问题 |
|---|---|---|
| 1 | [../README.md](../README.md) | 仓库定位、安装方法与验证入口 |
| 2 | [container-and-plugin-boundary-spec.md](container-and-plugin-boundary-spec.md) | 容器生命周期、宿主 Fork 与插件边界（权威 Spec） |
| 3 | [delivery-inventory.md](delivery-inventory.md) | 当前发布组件、宿主下沉算子及收益状态 |
| 4 | [operator-benchmark-standard.md](operator-benchmark-standard.md) | 算子优化四级评测阶梯、形状合同与报告标准 |
| 5 | [npu-benchmark-execution-guide.md](npu-benchmark-execution-guide.md) | NPU 真实环境运行指南（含 Qwen3.5-35B TP=2 双卡状态与测试步骤） |
| 6 | [cohort-and-ownership.md](cohort-and-ownership.md) | 适用基准 (Qwen3.5-35B-A3B TP=2) 与归属原则 |
| 7 | [support-matrix.md](support-matrix.md) | 插件与 Planner 的激活边界 |
| 8 | [agent-guide.md](agent-guide.md) | 开发者与 Agent 必须遵守的行为准则 |
| 9 | [release.md](release.md) | 构建、打包、合规检查与发布流程 |
| 10 | [pitfalls.md](pitfalls.md) | 现行环境的运行约束与故障避坑指南 |

## 单一事实源分工

| 主题 | 权威位置 |
|---|---|
| **容器分层、宿主 Fork 边界与隔离规范** | [container-and-plugin-boundary-spec.md](container-and-plugin-boundary-spec.md) |
| **组件清单与状态** | [delivery-inventory.md](delivery-inventory.md) |
| **评测四级阶梯与数据合同** | [operator-benchmark-standard.md](operator-benchmark-standard.md) |
| **NPU 运行操作手册 (Qwen3.5-35B TP=2 双卡)** | [npu-benchmark-execution-guide.md](npu-benchmark-execution-guide.md) |
| **适用域 Cohort 约束** | [cohort-and-ownership.md](cohort-and-ownership.md) |
| **代码与开发规范** | [agent-guide.md](agent-guide.md) |
| **发布与检查表** | [release.md](release.md) |
