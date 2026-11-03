# GDN in_proj RMSNorm Fusion (`gdn_input_norm`)

## 1. 背景与优化原理
在 Qwen3.5-35B-A3B (TP=2) 架构中，GDN（Gated DeltaNet）层解码阶段的输入投影在每步都需要对 batch tokens 进行逐行 RMSNorm 规约计算。原生 PyTorch 实现涉及平方累加、求均值、求倒数平方根和逐元素相乘的多次分立发射，中间张量反复出入 HBM。

本优化使用 Triton-Ascend 编写单 pass 融合算子：
- 一次性将隐藏维度搬运进 Unified Buffer (UB)；
- 在片内寄存器直接完成平方累加和 RMS 规约；
- 保留高精度浮点累加与严格的 bfloat16 语义截断，原样写回连续视图，消除多余 HBM 访存。

## 2. 硬件环境与单算子收益 (2×Ascend 910B2)
- **基线耗时**: 45.34 µs (分立算子链)
- **融合优化耗时**: 6.48 µs
- **纯算子加速比**: **7.00x**
- **L3 数值验证**: 严格通过（误差在容差范围内与 reference 保持位级一致）

## 3. 开关与控制
- 环境变量门控：`VLLM_HUST_GDN_INPUT_NORM=1`（默认关闭）
- 未启用时回退原生路径，零开销零副作用。
