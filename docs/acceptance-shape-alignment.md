# 验收形态对齐口径

本文件定义三个容易混淆的形态口径、它们与本仓采集/测量形态的差距，以及结论能跨形态引用到什么程度。全部参数值与来源指针在结构化记录 [acceptance-shape.json](acceptance-shape.json) 里，本文只写判据。

## 1. 三个口径，三个权威来源

| 口径 | 权威来源 | 性质 |
|---|---|---|
| 冻结验收服务形态（B0 35B） | `vLLM-HUST/hust-eval-client-mmlu-pro` 的 `examples/profile.b0_35b.json`（frozen） | documented command，不是运行回执 |
| 冻结请求形状标准（五数据集） | `vllm-hust-benchmark` 发布的 `dataset-standard-v1/standard.json`（five-datasets-20261113-v1） | 冻结协议，不是实测记录 |
| 引擎 B1 r6 组合形态 | `vLLM-HUST/vllm-hust-engine` 的 `engines/b1-r4.json` + r6 overlay | 组合验收形态，与 B0 客户端期望配置不是同一口径 |

第三行单独点名：引擎组合形态的服务参数（上下文、并发槽数等）与 B0 客户端期望配置不同。引用"验收形态"前先说清楚指哪一行，禁止用一行佐证另一行。

## 2. 采集脚本的对齐状态

`workloads/profile_hotspot_longrun.sh` 的默认服务形态已逐项对齐冻结验收服务形态：上下文、并发槽数、批 token 预算、KV block、显存水位、expert parallel、图捕获尺寸、prefix caching 关闭，以及本轮补齐的 chunked prefill、fcfs 调度、关闭 custom all-reduce、seed 与 `FULL_DECODE_ONLY` 编译配置。每一项都显式下发并写进 `collection_shape`，有自检拦"manifest 说开了、服务没开"。

仍未对齐的是请求构造路径：MMLU 客户端用冻结 token id 走 `/v1/completions`，本仓脚本用文本合成请求；trace 数据集的输入上限也未复刻。这三点差距使热点对比只能比方向。

## 3. 并发的三个声明位置

同一数据集的并发在三个位置有三个数：标准预算（`standard.json` 的 concurrency 字段）、客户端默认值、冻结 plan 契约。三处不一致不是文档错误，是三份不同的东西；MMLU 客户端在偏离契约运行时会显式把偏离写进当次运行记录。

判据：任何采集配置必须把所用并发写进 `collection_shape` 并注明口径来源；引用并发相关结论时写清楚用的是哪个口径，不得把标准预算当成实际并发。

## 4. 结论适用范围

- 索引内全部 `within_noise` 服务结论的测量形态是：上下文 4096、enforce-eager、无前缀缓存、合成 random 负载按并发 32。这些结论只在该形态内成立，验收形态下未复测。
- 跨形态只比方向，不比数字。batch 维度的方向推演只能用已记录的 shape_buckets 桶数据。
- 图回放形态下 host 下发开销被摊薄，eager 形态里由下发与同步主导的小算子排名会下降。
- 上下文对齐验收形态后预填充占比上升，decode-only 的收益被稀释，prefill 侧算子权重上升。

以上是方向判据，不产生任何服务收益标签。

## 5. 现有数据与复测清单

方向级的改进路线**不需要等新采集**。依据已经在仓内：`hotspot-profile-method.md` §7 的长前缀 ACLGraph 对照给出图模式长上下文下的方向变化（预填充侧算子权重上升、下发主导的小算子排名下降），各算子 evidence 的 shape_buckets 给出 batch 维度的摊薄方向。据此选下一批优化对象是合法的；不能做的是给它们排精确名次和写服务结论。

复测清单按用途排序：

1. **排序精化**：用对齐后的脚本起服冒烟（换宿主构建时重新验证参数；失败记录在 `status.json`），重采整网热点，与旧 4096/eager 热点表对比时只记录方向变化。
2. **结论刷新**：已测算子在冻结形态下复测服务 A/B，更新各自的 `serving_end_to_end` 与适用范围，再由验收流程判定。

两项都是有卡时段动作，都不是路线选择的前置条件。
