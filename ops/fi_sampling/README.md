# FlashInfer Sampling Triton Port (`fi-sampling`)

## 1. 背景与优化点
- **注册入口**：`vllm.general_plugins` -> `fi-sampling = vllm_ascend_split_batch.fi_sampling_plugin:load`
- **优化原理**：提供基于 Triton-Ascend 的采样 kernel 实现（包含 `argmax`、`top-k/top-p` 联合采样核），加速生成步的采样计算。
- **主开关**：`VLLM_HUST_FI_SAMPLING=1`，关闭时零 import、零侵入。

## 2. 评测现状与适用域约束
- 历史数据源于 Qwen2.5-Coder-14B 与早期 0.25 宿主环境，端到端差异在测量噪声内；
- 联合核（joint route）要求 $B \ge 256$ 且 $k \ge 32$，在典型长序列单卡下较难触发；
- **适用域严守**：历史 Qwen2.5 证据不得外推至当前 0.28 / Qwen3.5 基线栈；当前基线下的实际收益待重测。
