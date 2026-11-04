# GDN Split BA Fusion (`gdn_split_ba`)

## 1. 背景与优化原理
在 Qwen3.5-35B-A3B (TP=2) 架构中，GDN（Gated DeltaNet）层解码阶段的 $ba$ 投影输出为合并张量 $[T=32, 2 \times H_v]$。原生实现调用两次切片并分别执行 `.contiguous()` 强制开辟连续显存，触发多步小核调用与多次离散 HBM 搬运。

本优化使用 Triton-Ascend 编写单 pass 2D 分块融合算子：
- 一次性将整块 $ba$ 张量按行载入片上 Unified Buffer；
- 在片上寄存器同时提取高半区与低半区，分别对齐写入输出目标张量 $b$ 与 $a$；
- 消除额外的 chunk 步进开销与多余的 `.contiguous()` 独立内存分配调用。

## 2. 硬件环境与内环验证结论
- 已通过内环多桶形状与未见动态行数（`do_not_specialize`）零重编译护栏；
- 各批大小分桶耗时与加速比详见 `evidence.json`；
- 外环主测负载：`random`，压力负载：`sharegpt`。

## 3. 开关与控制
- 环境变量门控：`VLLM_HUST_GDN_SPLIT_BA=1`（默认关闭）
- 未启用时回退原生路径，零开销零副作用。
