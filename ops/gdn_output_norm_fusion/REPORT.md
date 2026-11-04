# 算子验收报告: GDN Part 3 Output Norm and Rearrange Fusion (`gdn_output_norm_fusion`)

## 1. 评测环境与形状事实

- **硬件拓扑**: Atlas 800T A2 (2×910B2, TP=2)
- **软件栈版本**: CANN 9.1.0 / vllm `0.28.1.post1.dev143+gf18cf803c` / vllm-ascend `0.25.1rc2.dev125+hust`
- **宿主 Fork 归属**: `vLLM-HUST/vllm-ascend-hust` @ `work/gdn-output-norm-fusion` (Commit `68c0790818dd15ff28b55d95713c2f0f4b3df66e`)，Base 分支为 `based/split-batch-baseline`
- **模型与权重**: Qwen3.5-35B-A3B (bf16, ND 格式)
- **执行相位**: Decode (Tokens 1, 4, 8, 16, 32, 64, 128)
- **门控开关**: `VLLM_HUST_GDN_OUTPUT_NORM_FUSED=1` (默认 `0`)

## 2. L0 纯微基准

在 910B2 硬件上通过 NPUGraph 图模式（PACKED REPEAT 20 次重放）对比原生离散链路与 Triton 融合链路：
- 完整度量指标与加速比详见结构化证据文件 `evidence.json`；
- 在 Decode 典型尺寸（Tokens 1~128）下，单 Kernel 融合执行消除了多次视图重塑、均方差平方规约中间写回与 einops 重排开销；
- 评测定性为显著正收益（Positive Speedup）。

## 3. L3 数值正确性护栏

- **核验标准**: 对比原生分立链路输出与 Triton 融合算子输出。
- **核验结果**: PASS。
  - 各 Batch Token 尺寸下输出张量与原生链路绝对偏差为绝对零；
  - 达到完全位级一致（Bitwise Identical），严格通过 L3 精度护栏。

## 4. 结论与交付标签

- [x] 宿主层源码落地完成 (`vllm-ascend-hust` @ `work/gdn-output-norm-fusion`)
- [x] L3 数值正确性护栏通过 (Max Diff = 0.0)
- [x] default-off 门控就绪 (`VLLM_HUST_GDN_OUTPUT_NORM_FUSED=1`)
