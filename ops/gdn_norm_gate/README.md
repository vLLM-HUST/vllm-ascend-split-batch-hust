# GDN Gated RMSNorm Fusion (`gdn_norm_gate`)

## 1. 背景与优化原理
在 Qwen3.5-35B-A3B (TP=2) 架构中，GDN（Gated DeltaNet）层解码阶段的注意力输出投影前需要执行带门控分支的规约计算：
$y = \text{RMSNorm}(x, \text{weight}) \times \text{SiLU}(z)$
其中 $x$ 为注意力输出张量，$z$ 为并行前向提取的门控张量。原生实现涉及二次访存、均方差平方和规约、求倒数平方根和逐元素激活相乘的多步算子链。

本优化使用 Triton-Ascend 编写单 pass 2D 分块融合算子：
- 按 2D 网格将多行多列直接载入片上 Unified Buffer；
- 在片内寄存器一次性完成平方累加规约与 SiLU 门控计算；
- 消除两张大输入张量与中间输出反复在 HBM 与 UB 之间的读写往返。

## 2. 硬件环境与内环验证结论
- 已通过内环多桶形状与未见动态行数（`do_not_specialize`）零重编译护栏；
- 各批大小分桶耗时与加速比详见 `evidence.json`；
- 外环主测负载：`random`，压力负载：`sharegpt`。

## 3. 开关与控制
- 环境变量门控：`VLLM_HUST_GDN_NORM_GATE=1`（默认关闭）
- 未启用时回退原生路径，零开销零副作用。
