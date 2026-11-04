# 算子验收报告: GDN Part 3 Output Norm and Rearrange Fusion (`gdn_output_norm_fusion`)

## 1. 评测环境与形状事实

- **硬件拓扑**: Atlas 800T A2 (双卡 Ascend 910B2, TP=2)
- **软件栈版本**: CANN 9.1.0 / vllm `0.28.1.post1.dev143+gf18cf803c` / vllm-ascend `0.25.1rc2.dev125+hust`
- **宿主 Fork 归属**: `vLLM-HUST/vllm-ascend-hust` @ `work/gdn-output-norm-fusion`（初始提交 `68c079081` 经 PR #69 合入 `based/split-batch-baseline` `0ee020516`；非连续 `z` 与动态行数免重编修复提交 `8499a55aa`）
- **模型与权重**: Qwen3.5-35B-A3B (bf16, ND 格式, 含 GDN 层与全注意力层)
- **执行相位与图模式**: Decode 与 Chunked Prefill，`cudagraph_mode=FULL`
- **门控开关**: `VLLM_HUST_GDN_OUTPUT_NORM_FUSED=1` (默认 `0`)

## 2. L0 纯算子微基准

在 Ascend 910B2 上通过 NPUGraph 图重放模式对比原生分立链路与 Triton 融合算子（同时覆盖连续 `z` 与列切片非连续 `z` 两种内存布局）：
- 完整分档度量数据与加速比记录于 `evidence.json`；
- 在各典型批次尺寸下，单核融合消除了多次 Python 视图重塑、非连续张量隐式拷贝、方差与倒数标准差中间张量写回以及 `einops.rearrange` 调度开销；
- 评测定性为正收益（Positive Speedup）。

## 3. L3 数值正确性护栏

- **算子级核验**: 在连续与非连续列切片 `z` 输入下，融合算子输出与原生分立参考实现的绝对偏差恒为零，达到位级一致（Bitwise Identical）；宿主单元测试 `tests/ut/ops/test_gdn_epilogue.py` 全量通过。
- **服务级贪心核验**: 在双卡 TP=2、`cudagraph_mode=FULL` 服务配置下，使用固定八条提示词、`temperature=0.0` 独立启动关闭组（`VLLM_HUST_GDN_OUTPUT_NORM_FUSED=0`）与开启组（`VLLM_HUST_GDN_OUTPUT_NORM_FUSED=1`），两组生成文本逐字完全一致（详见 `evidence.json`）。

## 4. L4 运行期调用链路归因

通过服务期性能采集窗口（单请求固定解码窗口，`op_statistic.csv`）进行调用计数与链路核验：
- **初始合入版本缺陷定位与修复**: PR #69 初始合入版本在 `output_norm_fusable()` 中要求 `z.is_contiguous()`，而 Qwen3.5 非交错投影路径下 `z` 为 `mixed_qkvz` 列切片重塑视图（非连续），导致运行期全部回退原生链路。修复提交 `8499a55aa` 将准入条件放宽至末维单位步长的二维跨距视图，并将行数与跨距加入 `do_not_specialize` 消除分块预填充下的动态重编译。
- **修复后调用计数核验**: 开启组在固定解码窗口内，融合内核 `_output_norm_kernel` 实际触发次数与模型 GDN 层数乘以解码步数严格相等，回退次数为零；具体计数与耗时对比存入 `evidence.json`。

## 5. 服务端到端对照评测

在双卡 TP=2、`cudagraph_mode=FULL` 下，固定其余五个 `VLLM_HUST_*` 门控开关为零，对关闭组与开启组各执行两轮独立起服与平衡负载测试（完整指标见 `evidence.json`）：
- 关闭组两轮测试与开启组两轮测试内部一致性良好；
- 开启组相比关闭组在整体完成耗时、输出生成速率及单步生成耗时上均呈正向改善，服务状态定性为正向有效（`verified_faster`），默认保持关闭（`default_enabled: false`）。

## 6. 结论与交付清单

- [x] 宿主层源码落地与非连续输入修复完成 (`vllm-ascend-hust` @ `work/gdn-output-norm-fusion`, commit `8499a55aa`)
- [x] L3 算子级位级一致与服务级八条贪心提示词逐字一致核验通过
- [x] L4 运行期调用计数与零回退验证通过
- [x] 服务端到端对照评测完成，数据落盘于 `evidence.json`
- [x] default-off 门控就绪 (`VLLM_HUST_GDN_OUTPUT_NORM_FUSED=1`)
