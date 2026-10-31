# 项目状态

本页不维护历史时间线，也不把旧记录当作当前验收。历史进度原件保存在 [benchmark 归档](https://github.com/vLLM-HUST/vllm-hust-benchmark/tree/67505eee3dc71fa3dbbc0fc365620f0a2e5658f4/archive/mod-git-20261022/split-current-git/snapshot/docs/)；只用于溯源。

## 当前状态

- 当前发布面、补丁状态、工具归属及收益证据边界见[交付清单](delivery-inventory.md)。
- root split-batch planner 为 `import_only`，Cascade 已从当前运行时和注册面撤除；历史结果不构成当前能力声明。
- CPU 可移植测试、仓内补丁重放和包检查结果属于工程验证，不替代 NPU 数值、实际 dispatch、图回放或端到端服务收益。
- 当前宿主 cohort、数值与性能主张的范围见[归属与 cohort](cohort-and-ownership.md)及[证据索引](EVIDENCE.md)。
- 新的组件、补丁或基准变更须同步更新权威清单，不在本页另建一套进度表。
