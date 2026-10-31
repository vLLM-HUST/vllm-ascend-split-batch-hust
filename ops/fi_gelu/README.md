# GELU and Mul Triton Fused Kernel (`fi-gelu`)

## 1. 背景与优化点
- **注册入口**：`vllm.general_plugins` -> `fi-gelu = vllm_ascend_split_batch.fi_gelu_plugin:load`
- **优化原理**：使用 vendored 的 Triton-Ascend 融合算子替换原生的分立 `gelu` 与乘法调用。
- **门控策略**：主开关为 `VLLM_HUST_FI_GELU=1`，仅对 NPU bf16、偶数末维且 `approximate="none"` 生效，其余输入自动回退宿主。

## 2. 设备核验现状
在 910B2 设备上的微基准核验表明，在所测独立形状下，融合 kernel 耗时比原生宿主计算链更长；且当前主模型 Qwen3.5 采用 SwiGLU 而不走 GELU 路径。因此该插件严格保持 **default-off**，不主张端到端加速收益。
