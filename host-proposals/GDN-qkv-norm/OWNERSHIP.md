# GDN QKV 归一化：宿主接入与验证条件

- **宿主接入**由 `vLLM-HUST/vllm-ascend-hust` 维护。[PR #82](https://github.com/vLLM-HUST/vllm-ascend-hust/pull/82) 的目标是 `based/split-batch-baseline`。提案未进入所测基线 `320eced8c`，不能在该基线上用提案的门控起服。
- **算子与证据**由本仓维护。合入后，先部署对应版本，在服务解释器核对两个 `host_commits`，再按[外环测试计划](../../docs/outer-loop-test-plan.md)验证启停、数值、图回放、逐卡归因和服务 A/B。历史算子测量不验证新基线。
- **插件注册**仍已撤回。只有证明目标调用可通过窄接口到达，并明确设备验证预算，才可重开注册路线。见[注册关闭记录](registration-closure.json)和[重开条件](release-candidate.json)。宿主 PR 不是重开插件注册的依据。
