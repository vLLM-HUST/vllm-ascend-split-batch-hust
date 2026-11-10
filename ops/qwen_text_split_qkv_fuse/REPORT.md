# 算子验收报告: Qwen Text Attention Split-QKV RoPE Fusion (`qwen_text_split_qkv_fuse`)

## 1. 评测环境与形状事实

- **硬件拓扑**: Atlas 800T A2 (2×910B2, TP=2)
- **软件栈版本**: CANN 9.1.0 / vllm `0.28.1.post1.dev89+gf18cf80` / vllm-ascend `0.25.1rc2.dev140+hust`
- **宿主 Fork 归属**: `vLLM-HUST/vllm-ascend-hust` @ `work/qwen-text-split-qkv-fuse` (Commit `493ec93d243816217e15daaab480fbe073051981`，宿主 PR #67，合并提交 `d1014aadb`)，Base 分支为 `based/split-batch-baseline`
- **模型与权重**: Qwen3.5-35B-A3B (bf16, ND 格式)，纯文本 RoPE 配置（`rope_parameters` 不含 `mrope_section`）
- **执行相位**: Decode (Tokens 1, 8, 16, 32, 64, 128)
- **门控开关**: `VLLM_HUST_SPLIT_QKV_ROPE_FUSED=1` (默认 `0`)

## 2. L0 纯微基准

在 910B2 硬件上通过 NPUGraph 图模式（PACKED REPEAT 50 次重放）对比原生离散链路与 Triton 融合链路：
- 完整度量指标与加速比详见结构化证据文件 `evidence.json`；
- 在 Decode 典型尺寸（Tokens 1~32）下，单 Kernel 融合执行耗时相比原生离散链路消除约 12 ~ 32 µs；
- 评测定性为正收益（Positive Speedup）。

## 3. L3 数值正确性护栏

- **核验标准**: 对比原生离散链路输出与 Triton 融合算子输出，判据为 `allclose(atol=1e-3, rtol=1e-3)`。
- **纯算子层面**（真实形状：每卡 Q 头 8 / KV 头 1 / head 256 / RoPE 维 64）:

| Tokens | q / k | v / gate |
| :--- | :--- | :--- |
| 1 | 位级一致 | 位级一致 |
| 16 | 最大绝对偏差 0.015625，门限不满足 | 位级一致 |
| 32 | 最大绝对偏差 0.03125，门限不满足 | 位级一致 |

- **根因**: 融合内核以 Triton FP32 累加 RMSNorm 方差，与 `npu_rms_norm` 的舍入方式不同；偏差随单步 token 数增长。
- **判定**: 单 token 位级一致，多 token 不满足门限。

## 4. L4 图内调用归因

- 独立起服采集逐卡 `op_statistic.csv`，按两卡分别核对目标内核命中数。
- 门控关闭时两卡融合内核命中均为零，原生 `RmsNorm` 承担全部注意力层归一化。
- 门控开启（原判据）时两卡各命中 160 次，覆盖 10 个 full attention 层。
- 归因结论：融合路径确实被调度，非静默回退。

## 5. 服务端验收

- **方法**: 独立起服（互斥占用端口与设备），8 条固定提示词、`temperature=0.0`、`seed=42`，开启与关闭门控逐字对拍。
- **原判据结果**: 8 条中仅 1 条完全一致，其余 7 条自第 114~325 字符起分叉。prefill 阶段偏差经自回归放大。
- **收紧为单 token 后的结果**:
  - 默认 cudagraph 模式：融合内核命中为零，对拍 8 条一致。该结果不采信，因为它等价于关闭门控，未覆盖融合路径；原因是 cudagraph 将输入填充至捕获尺寸（1/2/4/8/16/24/32），单 token 判据不成立。
  - `--enforce-eager` 模式：融合内核命中 150 次，对拍 8 条中 3 条一致，仍不满足要求。
- **未定位项**: 纯算子单 token 位级一致，端到端仍分叉，指向算子输出张量布局/stride 或后续 kernel 选择差异，尚未定位。

## 6. 结论与交付标签

- [x] 宿主层源码落地完成 (`vllm-ascend-hust` @ `work/qwen-text-split-qkv-fuse`)
- [ ] L3 数值正确性护栏通过 —— **未通过**（多 token 不满足门限，服务端逐字对拍不成立）
- [x] default-off 门控就绪 (`VLLM_HUST_SPLIT_QKV_ROPE_FUSED=1`)
- **交付判定**: 不纳入收益清单，保持默认关闭。单 token 收紧方案已评审否定，理由见 `evidence.json` 的 `serving_end_to_end`。
