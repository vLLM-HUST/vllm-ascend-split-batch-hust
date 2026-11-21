# 热点 profile 的采集口径与判读

这份文档解决一个问题：拿到一份整网 profile 之后，怎样判断某个算子值不值得优化，以及哪些结论不能从这份数据里得出。

## 1. 三种采集口径回答不同问题

| 口径 | 负载 | 图模式 | 能回答 | 不能回答 |
|---|---|---|---|---|
| 整网 eager | 短固定提示词 | `--enforce-eager` | 每步调用了哪些算子、调用次数与形状 | 算子内核质量、稳态耗时结构 |
| 整网 ACLGraph | 目标相位负载 | `cudagraph_mode` | 接近生产路径的算子分布与批次形态 | 单个算子的上限 |
| 单算子微基准 | 固定形状桶 | NPUGraph 回放 | 内核自身的下界与优化空间 | 整网收益 |

判读前先确认这份 profile 属于哪一行。把整网 eager 的结果当作内核质量证据，是常见误用。

## 2. 产物字段地图

`ASCEND_PROFILER_OUTPUT/` 下四份文件的分工：

| 文件 | 内容 | 用途 |
|---|---|---|
| `op_statistic.csv` | 按 `OP Type` 汇总的次数与设备时间 | 排热点次序 |
| `kernel_details.csv` | 逐次调用的形状、block 数、各 pipe 时间 | 判瓶颈形态 |
| `step_trace_time.csv` | 计算、通信、空闲的窗口分解 | 判断占比的分母 |
| `operator_details.csv` | 带 `Call Stack` 的调用链 | 定位 callsite |

`op_statistic.csv` 只收计算算子，通信算子不出现在其中。任何基于它的占比都是"计算时间内部"的占比，不是窗口时间占比。分母必须回到 `step_trace_time.csv` 核对。

## 3. 判读规则

`tools/operator_tools/profile_pipes.py` 把 `kernel_details.csv` 折算成每个算子在自身核内的 pipe 占用，并给出形态标签：

| 标签 | 判据 | 优化方向 |
|---|---|---|
| `aic_mte2_bound` | Cube 核搬入占比最高 | 减少权重或激活的搬入字节；融合相邻算子没有帮助 |
| `aic_mac_bound` | Cube 乘加占比最高 | 算力受限于数学工作量，只能改算法 |
| `aiv_vec_bound` | Vector 核计算占比最高 | 减少逐元素工作量或改用更宽的数据通路 |
| `aiv_mte2_bound` | Vector 核搬入占比最高 | 纯搬运，融合与原地改写优先 |
| `launch_or_wait_bound` | 所有 pipe 占比都低于阈值 | 单次工作量太小，收窄 grid 或与其他算子合并 |

形态标签是启发式，用于决定下一步采集方向，不能直接当作结论。判定后仍需在单算子微基准上复核。

## 4. 已知陷阱

1. **占比的分母不是墙钟**。计算占比与窗口时间的差距可能有数量级，先看 `step_trace_time.csv` 再引用百分比。
2. **`--enforce-eager` 会放大调用成本**。逐算子下发时，小算子的耗时主要由下发与同步决定，直接拍平了内核差异。
3. **block 数与有效工作量脱节**。按 vector 核总数下发 grid、但每核只有一行数据时，绝大多数核在空转。`kernel_details.csv` 的 `Block Num` 与形状对照即可看出。
4. **批次形态决定一切**。单 token 解码下 GEMM 是权重搬入受限，批次放大后同一个算子会转为算力受限。跨批次比较只能比方向，不能比数字。
5. **ACLGraph 的 capture 会污染窗口**。首次请求触发图捕获，必须在 `/start_profile` 之前完成预热，否则 trace 记录的是捕获阶段。
6. **通信时长要单独核对**。TP 场景下集合通信的时长包含对端等待，与载荷量级不匹配时需要先确认采集口径，再决定是否引用。

## 5. 工具入口

| 工具 | 用途 |
|---|---|
| `tools/operator_tools/profile_workload.py` | 按指定负载与图模式起服并采集一份整网 profile |
| `tools/operator_tools/profile_pipes.py` | 把 `ASCEND_PROFILER_OUTPUT` 折算成 pipe 构成与形态标签 |
| `.openbitfun/skills/operator-sol-optimizer/scripts/sol.py` | 在给定必要工作量与资源能力后计算 SOL% 与乐观上界 |

从仓根运行 `python -B -m tools.operator_tools.artifact_view <归档目录> --view hotspot --top 5`。长测按 `round_NN_off` 分别排名，每轮列出参与的 rank 和来源；次数和设备时间是该轮各卡之和，不是墙钟。新归档的 `hotspots.json` 与本轮 `profile/` 同目录；旧归档从各卡原始 CSV 只读重建，不依赖 `.openbitfun/`。显式请求热点却没有有效数据时返回非零，不解释为空热点。

`profile_pipes.py` 输出的是 pipe 占用，不是 SOL%。SOL% 还需要匹配的峰值能力与必要工作量两项输入，缺一不得出结论。

## 6. 交付约束

- 整网 profile 的结论只能用于筛选候选算子，不能写成服务收益；
- 候选筛选结果必须落到具体算子、调用位置与形状事实，否则不进入优化队列；
- 优化收益在单算子微基准与组合验收中确认，本流程不产出收益标签。

## 7. 口径切换带来的形态差异

同一模型、同一硬件，只改负载与图模式，热点次序就会变化。拿到一份 profile 先看三个信号，判断它是否可用：

| 信号 | 含义 |
|---|---|
| Cube 核 busy 占比偏低 | 算子在等下发或等搬运，profile 反映的是调用方式而非内核效率 |
| 集合通信时长远大于计算时长 | 对端等待被计入时长，该窗口不代表稳态 |
| 出现预填充专属算子 | 窗口覆盖了预填充，与解码口径不可混算 |

单卡计算时间内部的占比，短提示 eager 与长前缀 ACLGraph 的对照：

| 算子 | 短提示 eager | 长前缀 ACLGraph |
|---|---|---|
| MatMulV2 | 28.49 | 14.72 |
| GroupedMatmul | 23.45 | 27.85 |
| ChunkGatedDeltaRule | 1.59 | 10.68 |
| MatMulV3 | 未出现 | 3.53 |
| AddRmsNormBias | 1.70 | 3.50 |
| MoeInitRoutingV3 | 3.63 | 4.55 |
| CausalConv1d | 1.95 | 2.88 |
| Slice | 0.61 | 2.06 |
| Transpose | 4.95 | 2.27 |
| Cast | 4.50 | 2.10 |

三条可复用的判据：

1. 用短提示 eager 结果选候选，会漏掉预填充侧的重头算子。`ChunkGatedDeltaRule` 与 `MatMulV3` 只在长前缀口径下进入前列。
2. 同一算子在两口径下排名接近，优化价值可以差出一倍以上。`GroupedMatmul` 的 Cube busy 占比在两口径间相差一倍以上，批次形态决定它是否值得动。
3. 集合通信时长在两口径间相差一个数量级，说明短提示 eager 窗口里的通信时长不能当作稳态依据。引用前必须回到 `step_trace_time.csv` 核对。

## 8. 批次形态由调度器决定，不由负载决定

长前缀、固定合成、真实对话三种负载下，同一算子的记录形状可以完全相同：

| 算子 | 三种负载下的记录形状 |
|---|---|
| `GroupedMatmul` | `64,2048;256,2048,512` |
| `MatMulV2` | `8,2048;2048,2048` |

变化的是调用次数与单次时长，不是形状。分块预填充与小批次解码把每步 token 数钉在固定桶，换负载只改变预填充与解码的步数配比。

由此有三条直接可用的推论：

1. 想靠换负载改变批次形态是无效的，能改的是 `max_num_batched_tokens` 与分块参数。
2. 同一记录形状下耗时仍会随负载变化（专家分布不同），说明形状不足以解释耗时，必须把分布一起记录。
3. 按形状算 SOL 时若不绑定分布，同一个形状会给出互相矛盾的结论。

`GroupedMatmul` 是这条规律最直接的例子：形状恒定的情况下，单次时长的差异全部来自激活专家数，而激活专家数只出现在 `group_list` 里，不在形状字符串里。

