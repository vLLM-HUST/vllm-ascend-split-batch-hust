# FlashInfer Attention Demask (`fia-demask`)

## 1. 背景与优化点
- **注册入口**：`vllm.general_plugins` -> `fia-demask = vllm_ascend_split_batch.fia_demask:load`；同时包含 bundle `org.vllm-hust.fia-demask`
- **优化原理**：在纯 decode 阶段，当每个请求仅有 1 个 query token 时，移除多余的 attention mask 构造与处理开销。
- **激活条件**：配置环境变量 `VLLM_HUST_FIA_DEMASK=1` 激活。

## 2. 评测现状与边界
仅适用于纯 decode 且 $Q_S=1$ 的极窄场景；历史 A/B 对比结果处于噪声范围内（inconclusive），暂不主张确定性端到端加速，开关默认保持关闭。
