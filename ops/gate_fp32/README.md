# MoE Gate 权重 FP32 外提插件 (`gate-fp32`)

## 1. 背景与优化点
- **注册入口**：`vllm.general_plugins` -> `gate-fp32 = vllm_ascend_split_batch.gate_fp32_plugin:load`
- **优化原理**：在 MoE 模型的 Decode 路径中，每个 MoE 层每一步前向都会执行一次 `gate.weight.to(torch.float32)`，在 40 层 MoE 模型中每步产生 40 次高频 Cast 调用。本插件在模型加载期将叶子名为 `gate` 的权重外提并缓存为连续的 FP32 张量，从而消除 Decode 热路径上的高频类型转换开销。
- **门控策略**：环境变量 `VLLM_HUST_GATE_FP32=1`，默认关闭（default-off）；未开启时零加载、零打补丁。

## 2. 评测现状
- **L3 数值**：外提缓存的权重与原生转换数值 bitwise 一致。
- **L4 归因**：每 Decode 步预期消除 40 次 `Cast[256, 2048]`。
- **有卡评测任务**：待进入 NPU 环境后，通过 `bench-ladder.sh` 或定向 L0/L3 测试更新实际执行开销与延迟收益。
