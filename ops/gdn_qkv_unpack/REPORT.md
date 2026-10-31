# 算子验收报告: GDN QKV Single-pass Unpack Kernel (`gdn_qkv_unpack`)

## 1. 评测环境与形状事实
- **硬件拓扑**: Atlas 800T A2 (Ascend910B2 单卡实测)
- **软件栈版本**: CANN 9.1.0 / Triton-Ascend 3.2.2 / vllm 0.28.1.post1.dev143+gf18cf803c / vllm-ascend 0.25.1rc2.dev125+hust (Commit `74f0c0a27`)
- **模型规格**: Qwen3.5-35B-A3B (GDN 层参数: `q_dim=1024, k_dim=1024, v_dim=2048, head_k_dim=128, head_v_dim=128, row_stride=4096`)
- **评测形状**: Token 数量 `T ∈ {1, 16, 64}`，输入张量 `[T, 4096]`，连续跨距 `row_stride=4096`
- **数据类型**: bfloat16 (主性能对比)；float16 / float32 (正确性校验)

## 2. L0 纯算子对比 (Microbenchmark 数据)
在 910B2 硬件上通过 `torch.npu.Event`（warmup=5, iters=20, 3 轮测量取配对中位数，使用 `tools/bench_gdn_qkv_unpack.py`）对比：

| 评估档位 (Tokens) | 测量场景 | 输入形状与跨距 | 原生基线耗时 (Baseline) | 优化候选耗时 (Candidate) | 配对中位数变化率 (Paired Median) |
|---|---|---|---|---|---|
| **T = 1** | 合成覆盖档 | `[1, 4096]`, stride=[4096, 1] | **147.51 us** | **291.04 us** | **-49.32% (慢 1.97x)** |
| **T = 16** | 真实 Decode 档 | `[16, 4096]`, stride=[4096, 1] | **286.69 us** | **303.45 us** | **-5.55% (慢 1.06x)** |
| **T = 64** | 合成覆盖档 | `[64, 4096]`, stride=[4096, 1] | **283.58 us** | **292.04 us** | **-3.14% (慢 1.03x)** |

- **基线定义**：`split + 3×reshape(-1) + cat`（4 次算子 Launch，逐字照抄模型原实现 `vllm-hust/vllm/model_executor/layers/mamba/gdn/qwen_gdn_linear_attn.py:890-924`）。
- **候选定义**：`rearrange_mixed_qkv`（同输入输出契约的 Triton 单 pass 扁平搬运核）。
- **瓶颈归因**：Triton 内核在 910B2 上的 Launch 发射开销（~110-140 us）占主导，导致单核实现无法抵消发射时延，性能整体跑输原生 PyTorch 链。
- **评测定性**：**明确为负性能（Negative Speedup）**。实事求是记录测试数据，绝不掩盖。

## 3. L3 数值一致性核验
- **测试用例**: `tests/test_gdn_qkv_unpack.py`（66 项用例在 910B2 上 **全部 PASSED**）
- **核验范围**:
  - `test_values_layout_and_offsets_match_the_host_chain`：bf16/fp16/fp32 在不同 token 长度下位图（bitwise）级与原生链完全一致；
  - `test_views_span_one_flat_buffer_with_disjoint_slices`：输出各 view 共享单连续 buffer 且无数据重叠覆盖；
  - 非法维度、跨距过短、溢出越界及 CPU 非法输入拒绝测试全部通过。
- **核验结论**: **PASS**（数值与内存布局契约完全满足）。

## 4. L4 调用链路与图归因
- **目标 Callsite**: `vllm-hust/.../qwen_gdn_linear_attn.py:rearrange_mixed_qkv`
- **实际调用数**: 0（未接入发布插件与模型执行面）。
- **下沉替代说明**: GDN 的核心性能瓶颈聚焦在卷积权重未连续化导致的每步重复转置，已由宿主层卷积权重加载期预打包优化（`ops/gdn_conv_pack`）在权重加载期消除，不再依赖运行时单 pass 解包。

## 5. 服务端到端验证 (Serving L3/L4)
- **结论**: 算子级为负性能且未接入生产调用链，无端到端服务收益预期。

## 6. 结论与交付标签
- [x] Triton 单 pass 解包实现与延迟加载包结构已就绪
- [x] L3 数值与切片契约核验通过 (66/66 passed)
- [x] 负性能事实如实记录，保持未激活状态，不接管模型调用链
- [x] 不主张任何端到端服务加速
