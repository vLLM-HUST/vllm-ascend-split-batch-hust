# 算子验收报告: Qwen Text Attention Split-QKV RoPE Fusion (`qwen_text_split_qkv_fuse`)

## 1. 评测环境与形状事实

- **硬件拓扑**: Atlas 800T A2 (2×910B2, TP=2)
- **软件栈版本**: CANN 9.1.0 / vllm `0.28.1.post1.dev143+gf18cf803c` / vllm-ascend `0.25.1rc2.dev125+hust`
- **宿主 Fork 归属**: `vLLM-HUST/vllm-ascend-hust` @ `work/qwen-text-split-qkv-fuse` (Commit `493ec93d243816217e15daaab480fbe073051981`，宿主 PR #67，合并提交 `d1014aadb`)，Base 分支为 `based/split-batch-baseline`
- **模型与权重**: Qwen3.5-35B-A3B (bf16, ND 格式)
- **执行相位**: Decode (Tokens 1, 8, 16, 32, 64, 128)
- **门控开关**: `VLLM_HUST_SPLIT_QKV_ROPE_FUSED=1` (默认 `0`)

## 2. L0 纯微基准

在 910B2 硬件上通过 NPUGraph 图模式（PACKED REPEAT 50 次重放）对比原生离散链路与 Triton 融合链路：
- 完整度量指标与加速比详见结构化证据文件 `evidence.json`；
- 在 Decode 典型尺寸（Tokens 1~32）下，单 Kernel 融合执行耗时相比原生离散链路消除约 12 ~ 32 µs；
- 评测定性为正收益（Positive Speedup）。

## 3. L3 数值正确性护栏

- **核验标准**: 对比原生离散链路输出与 Triton 融合算子输出。
- **核验结果**: PASS。
  - `v` 与 `gate` 张量 bitwise 逐字节一致（最大绝对偏差为 0.0）；
  - `q` 与 `k` 张量受硬件 RMSNorm 中间舍入与 Triton 内 FP32 方差累加微小差异影响，最大绝对偏差为 0.03125（在有效数值范围内相对误差约为 2%），符合 bf16 精度规范。

## 4. 结论与交付标签

- [x] 宿主层源码落地完成 (`vllm-ascend-hust` @ `work/qwen-text-split-qkv-fuse`)
- [x] L3 数值正确性护栏通过
- [x] default-off 门控就绪 (`VLLM_HUST_SPLIT_QKV_ROPE_FUSED=1`)
