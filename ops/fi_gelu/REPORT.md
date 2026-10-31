# 算子验收报告: GELU and Mul Triton Fused Kernel (`fi-gelu`)

## 1. 评测环境与形状事实
- **硬件拓扑**: Atlas 800T A2 (1×910B2 / 2×910B2)
- **软件栈版本**: CANN 9.1.0 / Triton-Ascend 3.2.2 / vllm-ascend 0.25.1rc2.dev125+hust
- **探测形状**: (64, 2048), (8, 27648)
- **数据类型**: bf16 (approximate="none")

## 2. L0 纯算子对比 (Microbenchmark)
- **测量结果**: 融合 kernel 在测试独立形状下慢于原生宿主实现。
- **评测定性**: **负性能（Negative Speedup）**。保留测量事实，不删除负收益记录。

## 3. L3 数值一致性核验
- **测试工具**: `tools/probe_gelu_device.py`
- **容差标准**: atol=2.0^-6, rtol=2.0^-6, 匹配率 >= 99%
- **核验结论**: PASS (数值在容差内满足精度要求)。

## 4. L4 调用链路与图归因
- **执行面**: `CustomOp.register_oot(name="GeluAndMul")`
- **模型匹配**: 当前主基准模型（Qwen3.5-35B-A3B）采用 SwiGLU，真实推理中不走 GELU 算子，调用计数为 0。

## 5. 服务端到端验证 (Serving L3/L4)
- **结论**: 未完成，且无端到端服务收益预期。

## 6. 结论与交付标签
- [x] Triton 实现与 default-off 插件已落地
- [x] L3 数值精度核验通过
- [x] 负收益事实如实记录，主开关保持关闭 (`VLLM_HUST_FI_GELU=0`)
- [x] 不主张任何端到端服务加速
