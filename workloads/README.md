# 端到端服务评测负载特征与选择规范 (Workloads Specification)

本目录定义 vLLM-Ascend 服务化端到端评测（外环 `serving_loop`）的标准负载（Workload）库、张量形状分布特征及算子绑定规则。所有大体量原始数据集均在仓外容器持久路径隔离存放，由脚本与环境变量自动探测，严禁提交进 Git 仓库。

---

## 1. 标准负载特征画像（数据分布 -> 内部张量形状特征）

| 负载标识 | 数据来源与长度形态 | 步间 Token 数 $T$ 与调度特征 | 系统瓶颈焦点 | 算子适配场景 |
| :--- | :--- | :--- | :--- | :--- |
| **`random`** | 内置定长合成数据（默认输入 `512`、输出 `256`、`--ignore-eos`） | 无长度方差；预热后解码步 $T$ 稳定锁死在并发桶（如 `16` 或 `32`），同组重复测量波动最小 | 稳态解码下的片外内存带宽、向量后处理与算子启动开销 | **解码算子主测基准**：用于干净测量解码与尾声融合算子（如 `gdn_output_norm_fusion`、`moe_epilogue`、`gate_fp32`）的纯净端到端差异 |
| **`sharegpt`** | 真实多轮对话分布（离线 ShareGPT JSON，输入输出呈偏态长尾分布） | 请求异步到达与结束，分块预填充与解码高频交织，步间 $T$ 在动态范围内剧烈跳变 | 动态尺寸下的 Triton 即时编译卡顿风险、非对齐尾块掩码正确性、行数门控算子的实际命中比例 | **动态形状压力与鲁棒性验证**：所有算子进入整网验收时的必配压力负载，检查动态形状下的退化；重编译判断另需编译日志或缓存未命中记录 |
| **`long-prefix`** | 长上下文合成数据（默认输入 `2048`、输出 `128`、`--ignore-eos`） | 大尺寸分块预填充步占主导（$T$ 处于较大区间），矩阵乘与宽张量切片搬运占比高 | 预填充阶段的投影合并、宽张量解包/重排拷贝；共享前缀复用需另设专用输入证明 | **预填充与宽张量搬运优化**：适用于 `gdn_split_fusion`、`gdn_merge_inproj`、长序列注意力掩码与位置编码融合 |

---

## 2. 算子特征驱动的三负载绑定规则

单算子端到端 A/B 按需选测，完整配对优先用于实际交付组合；以下规则只约束已决定执行的 A/B，不要求每个算子各跑一套。组合仍逐项核验数值和调用，并预先冻结主测负载，不能事后挑选。

在内外环分离架构（见 [算子基准规范](../docs/operator-benchmark-standard.md)）中，外环评测不再随意挑选单一负载，而是按算子生效条件绑定 **"主测受控负载 + 动态分布压力负载 + 相位交叉检查负载"**。三个角色各答一个问题，缺一个就可能得出片面结论。

1. **主测受控负载（`primary_workload`，用于判定端到端收益）**：
   - **通用解码算子**（全行数生效）：绑定 `random`（默认并发 `32`），消除请求长度方差带来的调度噪声；
   - **带行数上限门控的小批量解码算子**（如仅在小行数区间启用、大行数回退）：绑定低并发档位的 `random`，确保稳态步落在算子激活区间内；
   - **预填充主导算子**：绑定 `long-prefix`，放大预填充张量计算与重排占比。
2. **动态分布压力负载（`stress_workload`，用于检验真实流量稳定性）**：
   - 统一配对 `sharegpt`，重点核查：① 动态 $T$ 下是否出现首次编译长尾毛刺；② L4 `op_statistic.csv` 中目标算子的实际命中率是否符合预期。
3. **相位交叉检查负载（`cross_check_workload`，用于发现反向效应）**：
   - 取与主测**相反的相位**：主测 `random` 的算子交叉检查 `long-prefix`，主测 `long-prefix` 的算子交叉检查 `random`；
   - 判据：算子在主测相位赢、在另一相位输时，净收益可能被抵消甚至翻负。只测单一相位无法发现这类退化，因此交叉相位必须同样采集并记录，主测决定收益方向，交叉负载的显著退化阻止推荐当前配置；
   - 典型风险：权重外提或预打包类改动在 decode 小 batch 省时间，但大 $T$ 预填充阶段额外搬运 fp32 副本反而变慢。

---

## 3. 执行方式与数据集路径约定

通过 `workloads/run_workload.sh` 或 `python3 -m tools.operator_tools.serving_loop plan` 生成并执行评测命令：

```bash
# 1. 查看指定算子的推荐负载配对与命令
python3 -m tools.operator_tools.serving_loop plan --op gdn_output_norm_fusion --gate-env VLLM_HUST_GDN_OUTPUT_NORM_FUSED=1 --phase decode

# 2. 执行受控定长基准负载
bash workloads/run_workload.sh random 8350 qwen35-35b /tmp/workload_results

# 3. 执行真实长尾对话分布压力负载（自动探测容器内离线 ShareGPT JSON）
bash workloads/run_workload.sh sharegpt 8350 qwen35-35b /tmp/workload_results

# 4. 使用显式配置启动真实采集，输出目录必须尚不存在
SERVING_CONFIG=/path/to/collection_config.json \
  bash workloads/evaluate_outer_loop.sh gate_fp32 \
  "VLLM_HUST_GATE_FP32=1" 8350 qwen35-gate-fp32 /path/to/new_run
```

注意仓内同一条开关在三处出处有两种写法，不要串用：`DELIVERY_INDEX.json` 的 `gate_env`、`serving_loop plan --gate-env` 与 `evaluate_outer_loop.sh` 的入参都写 `变量名=1` 的赋值形式；采集配置 JSON 的 `gate` 写裸变量名（不带 `=1`）。

真实采集由 `tools/operator_tools/serving_collect.py` 执行。配置 JSON 必须包含：

- `python`：已安装匹配宿主 Fork 的隔离 Python 绝对路径。
- `host_commits`：以 `vllm`、`vllm-ascend` 为键的完整提交；采集前在服务解释器内核验安装来源，不匹配即停止。
- `server_command`：完整起服参数数组，必须显式绑定本地地址、端口、模型别名、并行规模和随机种子。
- `gate`、`port`、`model_alias`：与 shell 参数一致的目标开关名、端口和模型别名。`gate` 只填**裸变量名**（形如 `VLLM_HUST_GATE_FP32`），不要带 `=1`；采集器按 OFF/ON 两臂分别注入 `0`/`1`，写成带赋值的整串会被当成一个没人读的变量名，两臂实际同配置，最终只会得到"噪声内"的假结论。入口对该字段做格式校验，不合格直接拒绝采集。
- `model_path`、`sharegpt_path`：本地 tokenizer 和真实对话数据集路径。
- `world_size`、`attention_layers`、`kernel`：用于逐卡验证调用次数的并行规模、目标层数和内核名称子串。
- `environment`：显式运行环境；CANN 的 Python 依赖路径需写入此项，不能借此注入宿主源码替换。
- `startup_timeout`：可选的启动期限，单位秒。
- `phase`、`max_active_rows`：可选的执行相位与激活行数上限，决定主测负载及其并发；默认解码相位。

采集器先独立起服完成固定提示词逐字对拍，再检查逐卡 Profiler 结果，最后按 OFF/ON/ON/OFF 顺序独立起服，每轮运行 `random`、`sharegpt` 与 `long-prefix`。当前归因协议为单请求、固定输出长度、每层每步一次目标调用；不适用此协议的算子不得直接套用。原始响应、命令、服务日志、调用统计和失败原因均保存，任何输出缺失、请求失败或归因不通过都会阻止收益汇总。正式算子证据需在核对模型配置及形状后单独回填，采集器不自动覆盖。

`DRY_RUN=1` 只打印命令或计划，不生成成绩文件，也不属于设备验收。显式指定的数据集路径失效时直接报错，不回退到其他数据。

可通过环境变量显式指定仓外离线数据集路径：
- `VLLM_HUST_DATASET_SHAREGPT`：覆盖本地 ShareGPT JSON 路径。

## MOD 验收边界

这里的负载用于算子外环诊断。MOD 最终验收须按 [专门规范](../docs/mod-acceptance-standard.md) 使用冻结数据集及连续窗口。禁止把本目录的强制长度选项带入完整回答测试，也禁止直接用算子汇总器给出 MOD 合格结论。

产物审核、证据更新、索引更新和指纹生成统一按 [基准规范](../docs/operator-benchmark-standard.md#6-产物交接与离线检查) 执行。
