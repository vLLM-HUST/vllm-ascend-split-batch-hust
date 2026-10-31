# 支持矩阵

## 当前发布面

逐组件的可发现、可激活、真实调用、算子收益和服务收益状态，以[交付清单](delivery-inventory.md)为唯一事实源。这里不复制历史收益数字或重复 plugin 状态。

- 根 bundle `org.vllm-hust.split-batch-full-graph` 只含 `planner`，状态为 `import_only`，没有宿主执行面，**不可激活**。
- Cascade 实现、注册入口及执行路径均已撤除，不属于当前发布面。
- 当前包的 9 个 `vllm.general_plugins` 与 6 个 `vllm_hust.extension_bundles` 以 `pyproject.toml` 和对应 manifest 为准；源码、补丁、工具、提案不计作插件。
- `active`/可发现不等于每个请求真实调用；真实调用与性能结论须分别满足[证据字段合同](delivery-inventory.md#性能证据最小字段)。

## Planner 边界

`planner.plan_dual_pad` 和预检函数是纯 Python 规划逻辑，不接管宿主调度、worker 执行或 graph replay。规划成功只表示描述符可生成，不表示存在可运行的 Split-Batch/full-graph 服务功能。当前 root bundle 无 activation entry、environment 或 additional-config 通道，不能通过安装旧 wheel 或设置环境变量启用。

planner 的 fail-closed 预检包括非 uniform decode、非 full/piecewise graph、投机解码、LoRA/MLA/M-RoPE 冲突及 batch 过小。具体返回码和纯函数约定见 `tests/test_planner.py`；这些预检不构成当前服务支持矩阵或收益证据。

## 环境和证据边界

- 当前宿主兼容范围和发布操作见 [release.md](release.md)。
- 正确性/性能声明的 cohort 与权重归属见 [cohort-and-ownership.md](cohort-and-ownership.md)。不同模型、权重、形状或 cohort 只可比较方向，不得拼接数字。
- 旧 Cascade 与 Qwen2.5/0.25 记录属于历史证据，不能推出当前 Cascade 能力或 Qwen3.5/0.28 收益。
- MTP 属于 planner 的 fail-closed 边界，不是支持声明。配置被拒后回原生路径时，不得将其记为优化实际启用。
