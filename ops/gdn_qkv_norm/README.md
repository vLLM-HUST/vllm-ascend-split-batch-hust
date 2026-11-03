# GDN QKV Unpack and L2Norm Fusion (`gdn_qkv_norm`)

## 1. 背景与优化原理
在 Qwen3.5-35B (TP=2) 架构中，GDN（Gated DeltaNet）层解码阶段的输入包含混合投影张量。原生实现需要执行 4 次独立算子调用（拆分切片、3 次 reshape 与 concat），随后分别对 Q 和 K 执行 L2Norm。

本优化使用 Triton-Ascend 编写单 pass 融合算子：
- 一次性搬运张量进入统一缓冲（Unified Buffer）；
- 在局部片内直接完成 Q 与 K 的 L2Norm 规约计算，V 张量原样直通；
- 输出共享连续单存储视图，消除中间张量在 HBM 的反复搬运。

## 2. 硬件实测收益 (2×Ascend 910B2)
- **基线耗时**: 791.24 µs (原生分立算子链)
- **融合优化耗时**: 247.09 µs
- **纯算子加速比**: **3.20x**
- **L3 数值验证**: 严格通过（与原生 reference 误差完全在容差范围内）

## 3. 开关与控制
- 环境变量门控：`VLLM_HUST_GDN_QKV_NORM=1`（默认关闭）
- 未启用时回退原生路径，零开销零副作用。
