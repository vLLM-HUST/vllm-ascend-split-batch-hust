# 910B2 双卡算子优化实验计划（带卡执行）

本计划以 **2 × Ascend910B2、TP=2** 为前提，把《Ascend 910B 高级算子优化工程指南》里的六个优化模式，转成
可执行、可判定、可回填的实验条目。指南原文见同目录
`ASCEND_910B_ADVANCED_OPERATOR_OPTIMIZATION_GUIDE.md`。

## 0. 计划定位与不可声称边界

- 本目录**只放计划与架构映射**，不放任何运行产物、原始 Trace、日志或度量数据。
- 每个方向的结论只有回填到 `ops/<operator>/` 与 `DELIVERY_INDEX.json` 后才可对外引用；
  未回填一律标记「未证明」或「仅方向」。
- 离线工具（`tools/analysis/ascend910b_static_audit.py`、`tools/analysis/geglu_ftz_guard.py`）
  里的带宽、算力、片上容量都是**输入参数**。带卡后必须用 D0 测得的常数替换，否则预期值没有意义。
- 算子级结论与服务级结论分开记录；不得把单算子改善外推为服务端到端收益。
- 门控一律 default-off；L3 数值不通过即一票否决，不进入 L4 与服务阶段。

## 1. 执行前置（每次开窗前必须过）

| 项 | 要求 | 校验动作 |
|---|---|---|
| 硬件 | 2 张 Ascend910B2，TP=2 | `python3 -c "from tools.op_agent.guard import check_npu_environment as c; print(c(2))"` |
| 宿主基线 | `based/split-batch-baseline`；vllm-ascend `ad82d77cb`(镜像 `74f0c0a27`)、vllm-hust `f18cf803c` | `git -C /vllm-workspace/vllm-ascend rev-parse --short HEAD` |
| 结果目录 | 必须在仓库外，`bench/evidence_paths.py` 会强制校验 | `export SPLIT_BATCH_RESULTS_DIR=/root/prof/<stamp>` |
| 门控 | 全部 `VLLM_HUST_*=0` 起服后再逐项打开 | `tools/bench-ladder.sh carriers` |
| 单变量 | 贪心采样、固定 seed、`--no-enable-prefix-caching`、固定并发与输入输出长度 | 见 `docs/npu-benchmark-execution-guide.md` |

本机探针当前返回 `HardwareStatus(has_npu=True, device_count=2, is_tp2_ready=True, device_name='Ascend910B2')`，
单卡可用显存约 61 GB。正式开窗前仍需重跑探针，探针不通过不得往下走。

## 2. 方向总览

| 编号 | 方向 | 指南模式 | 落点算子目录 | 一句话判据 |
|---|---|---|---|---|
| D0 | 硬件常数与对齐粒度标定 | 全局前置 | 本文件附录 A | 拿到可复用的 4 个常数 |
| D1 | Cube→UB 片上闭环（Down GEMM + Residual + Norm） | 模式 1 | `ops/moe_epilogue/`（先做 shared-expert down 链路） | 中间激活往返消失，且 L3 通过 |
| D2 | 多级乒乓深度（3~4 级） | 模式 2 | 同 D1 的 kernel | 3 级优于 2 级，4 级递减则如实记录 |
| D3 | 小 M 派发（Token 打包 / Vector 回退） | 模式 3 | `ops/moe_epilogue/`、`ops/gdn_qkv_norm/` | 找到 M 的派发分界点并复现 |
| D4 | GeGLU/SwiGLU 融合与次正规数门禁 | 模式 4 | `ops/gate_fp32/` 旁新增 `ops/ffn_activation/` | 余弦过、误差落在地板内 |
| D5 | 跨核事件提前放行（对标 PDL） | 模式 5 | 机制探针，暂不落算子 | 先证机制可用，再谈收益 |
| D6 | 32 字节对齐与 UB 别名避坑 | 模式 6 | `docs/pitfalls.md` 条目 + 各 kernel | 构造出退化再证明修好 |

## 3. 各方向执行计划

### D0 硬件常数与对齐粒度标定

**运行入口**：同目录 `run_baseline_window.py`，一次运行把本方向四件事全做完，
产物落在 `$SPLIT_BATCH_RESULTS_DIR/<stamp>/`（仓库外），并生成人读的 `SUMMARY.md`。

```bash
export SPLIT_BATCH_RESULTS_DIR=/root/prof
python3 experiments/ascend910b-2card/run_baseline_window.py --stamp 1109-1400
# 只打印计划、不碰设备：--dry-run
```

**目标**：把离线工具里所有假设参数换成这台机器上的真实值，后续所有预期才有分母。

**步骤**
1. 带宽：用一块能塞满 L2 之外的连续搬运 kernel（或 `torch.npu` 大张量拷贝）扫不同尺寸，取稳定段的字节/耗时。
2. Cube 峰值：固定 K、N 扫 M（16 的整倍数），取算力饱和段的 FLOP/耗时。
3. 片上容量：逐步放大 tile 直到 UB/L1 申请失败，记录第一个失败的 tile 字节数。
4. 对齐粒度：构造 1~64 字节步长的 strided 搬运，找出耗时出现台阶的步长。

**预期**
- 得到最小搬运粒度是否为 32 字节的**结论性证据**（指南假设是 32 字节，需证实或推翻）。
- 带宽与算力两个常数代入 `ascend910b_static_audit.py --peak-gb-s/--peak-tflops` 后，机器平衡点应在
  100~400 FLOP/Byte 量级；若偏离，说明容器或驱动版本与指南前提不同，需在附录 A 注明。

**判废**：三次独立复测的相对离散度超过 5%，则该常数不可用，只能标注区间。

**回填位置**：本文件 **附录 A**（下表预留空行），数据源为窗口产物的
`window.json`（`bandwidth.steady_gb_per_s`、`cube.peak.tflops`、
`hbm_ceiling.allocated_gib`、`granularity.verdict.step.row_bytes`）；
同时把定稿常数写进 `docs/pitfalls.md` 的「硬件事实」小节，供其它方向引用。
片上容量一项由窗口显式登记为未测，需 Ascend C 探针补齐后再填。

---

### D1 Cube→UB 片上闭环（Down GEMM + Residual + Norm）

**目标假设**：把 Down GEMM 的输出留在片上直接做残差加法与 RMSNorm，消除中间张量的 4 趟 HBM 往返，
是这条链路唯一有意义的杠杆。

**离线已具备的结论**（`tools/analysis/ascend910b_static_audit.py`，Qwen3.5-35B-A3B TP=2、bf16、K=256、N=2048）
- 分算子写法 6 趟、片上闭环 2 趟，节省恒为输出 tile 的 4 倍，占激活部分 66.7%，与 M 无关；
- 计入每步必读的权重后，节省占总流量：M=1 时 1.5%、M=10 时 12.7%、M=32 时 28.6%；
- 算术强度 M=1 为 0.99、M=32 为 25.6 FLOP/Byte，对 250 FLOP/Byte 的机器平衡点**全段搬运受限**。

**步骤**
1. L4 归因先确认调用点与次数：`tools/bench-ladder.sh profile --tier light`，在 `op_statistic.csv` 里核对
   `MatMulV2` 与后续 `Add`/`RmsNorm` 的配对次数（每层 1 次 × 40 层）。
2. L0 先做**纯算子**对照：原生「matmul → add → rmsnorm」三段 vs 单 kernel 融合，同形状同 dtype。
3. L3 数值：与三段原生链路逐元素比对，门禁按 D4 的 dtype 相对判据（本链路输出为 bf16，禁止用绝对阈值）。
4. L4 归因：融合后目标小算子计数应归零，且 all-kernel 分母不变。

**预期**
- 融合后中间张量不再落 HBM，L4 里对应的 `Add`/`Norm` 调用数降到 0；
- L0 改善幅度**不应超过**该 M 下的节省占比上界（M=1 约 1.5%、M=32 约 28.6%）。若远超上界，
  说明测量里混进了别的变量（多半是 launch 开销差异），必须拆开重测；
- decode（小 M）场景大概率落在噪声带内 —— 这种结果必须如实记录为「方向成立、当前形状无收益」，
  不得改口径包装成加速。

**判废**：L3 出现任何一个元素超出门禁，或无法证明中间张量确实未落 HBM。

**回填位置**
- `ops/moe_epilogue/evidence.json` 的 `l0_microbenchmark`、`l3_numerical_guard`、`l4_attribution`；
- `ops/moe_epilogue/REPORT.md` 按 `docs/operator-benchmark-standard.md` §3 模板重写对应小节；
- `DELIVERY_INDEX.json` 里 `moe_epilogue` 条目的 `status` / `l3_numerical_guard` / `serving_status`；
- 若判定「无收益」，`serving_status` 记 `within_noise` 或 `measured_no_gain`，**不允许**记 `verified_faster`。

---

### D2 多级乒乓深度（3~4 级）

**目标假设**：把关键队列从 2 级加深到 3 级能压掉 MTE2 等待，4 级收益递减。

**离线结论**：tile 16×256、双操作数 bf16 时每级 16 KiB；2/3/4 级分别为 32/48/64 KiB。
128 KiB 上限可开 8 级、192 KiB 开 12 级、256 KiB 开 16 级 —— **本形状下 3~4 级余量充足**，
不需要靠缩 tile 换深度。

**步骤**
1. 在同一 kernel 里用编译期常量做 2/3/4 级三档，其余代码逐字一致；
2. L0 各自跑固定重复次数，取中位数与离散度；
3. L4 light 窗口看 `aic_mte2_time`、`wait time` 与 `cube_utilization` 的变化。

**预期**
- 2→3 级应看到等待时间下降、Cube 占用上升；3→4 级改善明显收窄或持平；
- 若 4 级反而更差，按 3 级交付，并把递减事实写进报告，不删数据。

**判废**：三档之间除队列深度外有任何其它差异（否则不是单变量对照）。

**回填位置**：`ops/moe_epilogue/evidence.json` 的 `l0_microbenchmark.raw_measurements` 增加
`stages` 维度；`REPORT.md` §2 表格补三行；`DELIVERY_INDEX.json` 不变（同一算子）。

---

### D3 小 M 派发（Token 打包 / Vector 回退）

**目标假设**：M 小于 8 时放弃 Cube 改走 Vector，M 大于等于 16 坚决走 Cube，中间区间用跨步打包补齐。

**离线结论**（`small_m_report`）：Cube 原子为 16 行，M=1 时 93.75% 的乘加是零填充、M=10 时 37.5%、
M=17 时 46.9%（补到 32）。

**步骤**
1. L0 扫 M = 1,2,4,8,10,12,16,32 三条实现：Cube 填充、打包后 Cube、Vector GEMV；
2. 打包路径必须把**打包与还原的搬运成本**计入同一次测量，不许只测核心段；
3. L3：三条实现输出与 fp32 参考比对，打包路径要验证行序还原无错位。

**预期**
- 存在一个明确分界点落在 M ∈ [8,16]，分界点两侧最优实现发生切换；
- 打包路线只在 M 不是 16 整倍数且相邻请求可合并时有效；单请求 decode（M=1）下打包无源可合，
  预期退化为 Vector 回退最快；
- 若三条实现某段交叉不清（差异在离散度内），记录为「无明确分界」，不硬指定阈值。

**判废**：打包路径还原后行序与输入不同，或还原成本未计入。

**回填位置**：`ops/moe_epilogue/evidence.json` 的 `l0_microbenchmark.raw_measurements` 增加 `dispatch` 维度；
派发阈值最终写进 `docs/pitfalls.md` 与对应 kernel 的注释（只写阈值与判据，不写叙事）。

---

### D4 GeGLU/SwiGLU 融合与次正规数门禁

**目标假设**：把 Gate/Up 两路矩阵乘的输出在片上直接做激活与逐元素乘，同时用**可判定**的精度门禁，
替代「绝对误差小于某个极小值」这种任何实现都过不了的写法。

**离线结论**（`tools/analysis/geglu_ftz_guard.py`，fp16、gate ∈ [-30,3]、3305 个样本、输出量级 11.99）
- 量化地板（半个台阶）3.906e-3；FTZ 贡献 5.24e-5，为地板的 1.3%；
- FTZ 贡献始终不超过理论界 `最小正规数 × |最大值|` = 2.44e-4；
- bf16 的 FTZ 贡献恒为 0（次正规区间在 1e-38 以下，GeGLU 碰不到）；
- 绝对值 1e-5 的门禁对 fp16/bf16 输出**任何实现都不可达**，工具用 `unreachable_by_any_impl` 标出。

**步骤**
1. 先在卡上验证硬件是否真的开 FTZ：构造一组 GELU 输出恰落次正规区间的输入，看是否被归零；
2. L3 对拍：融合实现 vs 原生 `gelu + mul`，用余弦与 dtype 相对阈值双判据；
3. 若硬件确实开 FTZ，把「激活悬崖」区间与上界记入 pitfall；
4. 只有在 bf16 路线才可考虑跳过 FTZ 相关工作（离线已证贡献为 0），fp16 路线必须带门禁。

**预期**
- 余弦判据应稳定过 0.99999；
- 误差应落在 2 倍半台阶内（fp16 约 7.8e-3、bf16 约 6.3e-2）；
- 若卡上误差显著超离线上界，说明输入量化或近似函数实现与假设不同，需重做误差分解，不得直接放宽容差。

**判废**：出现 NaN/Inf、出现超出门禁的单个元素，或门禁被改成「任何实现都能过」的宽阈值。

**回填位置**：新建 `ops/ffn_activation/`（`README.md`、`REPORT.md`、`evidence.json`、`checksum.sha256`），
并在 `DELIVERY_INDEX.json` 增加一条（`layer` 按实际归属填 `host_fork` 或 `plugin_carrier`，
`l3_numerical_guard` 填 `passed`/`failed` 实测结论）。

---

### D5 跨核事件提前放行（对标 PDL）

**目标假设**：前驱算子只写完 Q 就通过任务事件放行下游 Attention，让下游提前做页表寻址与旋转角计算。

**步骤**
1. 机制探针优先：先在一对最小 kernel 上验证「前驱发事件、后继等事件」这条通路在 Ascend C 与
   当前 CANN 版本上是否有稳定公开接口；
2. 探针通过后再做 QKV→Attention 的分部放行；
3. L4 对比放行前后的空泡（`wait time` 占比），L3 保证数值逐位一致。

**预期**
- 机制探针是本方向的**唯一门票**：找不到稳定接口就记为「机制缺口」，直接收尾，不投入 kernel 工作；
- 若机制可用，预期收益来自 Host 调度空泡下降，量级通常在微秒级；在 decode 短步场景才可能显形，
  prefill 长步场景多半看不见。

**判废**：依赖未公开或随版本变化的接口；或数值不再逐位一致。

**回填位置**：机制结论写 `docs/pitfalls.md`；若最终落了算子，按 D1 同样的四处回填。

---

### D6 32 字节对齐与 UB 别名避坑

**目标假设**：非 32 字节对齐的跨步搬运会退化为低效路径，必须用 `DataCopyPad` 或改排布，
而这条风险在当前模型的默认切片上**不存在**，只在自定义切片上出现。

**离线结论**：Qwen3.5-35B-A3B TP=2 的 5 个切片（hidden 4096 B、kv head 256 B、k 全头 256 B、
shared expert 512 B、gate_up 1024 B）在 bf16 与 fp32 下全部落在 32 字节网格上；
起始偏移只要不是 32 字节整倍数（如偏移 1 个元素 = 2 字节）即判失败。

**步骤**
1. 用离线工具生成一组对照切片：对齐/非对齐各一组，字节量相等；
2. L0 同 kernel 各跑一组，比较搬运耗时是否出现台阶；
3. 对齐组改 `DataCopyPad` 复测，确认台阶消失；
4. 顺带验证 UB 内临时张量是否与目的张量部分重叠（Aliasing），重叠时立即隔离。

**预期**
- 非对齐组的搬运耗时应出现明显台阶（量级上指南给的是数倍退化，需自测确认）；
- `DataCopyPad` 后回到对齐组水平；
- 若两者无差异，说明该版本 MTE 已能处理非对齐，应更新 pitfall 而不是继续沿用旧结论。

**判废**：对照组的字节量与元素数不一致（则不是单变量）。

**回填位置**：`docs/pitfalls.md` 增加或修正条目；涉及具体算子的部分写进该算子 `REPORT.md` 的「风险与边界」小节。

## 4. 统一回填契约

| 层次 | 落点 | 约束 |
|---|---|---|
| 原始产物 | 仓外 `$SPLIT_BATCH_RESULTS_DIR`（`bench/evidence_paths.py` 强制） | 仓库内禁止出现 Trace、日志、DB、csv 原始件 |
| 算子证据 | `ops/<operator>/evidence.json`（`operator-benchmark-evidence/v1`） | 度量数字只落这里，行文里不复述 |
| 算子报告 | `ops/<operator>/REPORT.md` | 按 `docs/operator-benchmark-standard.md` §3 模板 |
| 交付索引 | `DELIVERY_INDEX.json` 的 `operators[]` | `status`、`l3_numerical_guard`、`serving_status` 必须与证据一致 |
| 指纹 | `ops/<operator>/checksum.sha256` | 改过目录内文件后必须重算（见下命令） |
| 坑位 | `docs/pitfalls.md` | 只写判据与约束，不写过程 |
| 禁止落点 | `docs/evidence/**`、仓库根目录、本目录 | 新增文件严禁进入这三处 |

回填与校验命令：

```bash
# 1) 服务侧 A/B 汇总并自动回填算子目录
python3 -m tools.op_agent.serving_loop summarize-ab \
    --workload random \
    --off "$SPLIT_BATCH_RESULTS_DIR/off_1.json" "$SPLIT_BATCH_RESULTS_DIR/off_2.json" \
    --on  "$SPLIT_BATCH_RESULTS_DIR/on_1.json"  "$SPLIT_BATCH_RESULTS_DIR/on_2.json" \
    --update-op-dir ops/<operator>

# 2) 贪心输出逐字对拍（数值门禁）
python3 -m tools.op_agent.serving_loop greedy \
    --baseline "$SPLIT_BATCH_RESULTS_DIR/l3_off.json" \
    --candidate "$SPLIT_BATCH_RESULTS_DIR/l3_on.json" \
    --update-op-dir ops/<operator>

# 3) L4 归因核对
python3 -m tools.op_agent.serving_loop parse-l4 \
    --csv /root/prof/<stamp>/op_statistic.csv --kernel <kernel_name> --expected-calls <n>

# 4) 刷新指纹
python3 -c "from tools.op_agent.custody import write_dir_checksum as w; print(w('ops/<operator>'))"

# 5) 合规门禁（全绿才可提交）
ruff check . && ruff format --check .
python3 -m pytest -q tests/test_benchmark_standard.py tests/test_json_hygiene.py \
    tests/test_document_custody.py tests/test_ops_delivery_policy.py
python3 -m pytest -q -m "not host_tree"
```

## 5. 阶段与依赖

| 阶段 | 内容 | 依赖 | 结束标志 |
|---|---|---|---|
| P0 | D0 常数标定 | 双卡可用 | 附录 A 填满，4 个常数定稿 |
| P1 | D6 对齐基线 + D2 队列深度 | P0 | 对齐与吞吐前提明确，单变量框架就绪 |
| P2 | D1 片上闭环（单算子） | P1 | L0 有方向性结果，L3 判据定稿 |
| P3 | D3 小 M 派发 | P2 | 分界点确定或明确记录「无分界」 |
| P4 | D4 激活融合 | P2 | 门禁可判定并通过 |
| P5 | D5 提前放行（探针优先） | P0 | 机制结论（可用/缺口）已记录 |
| P6 | 服务侧 A/B 与回填 | P2~P4 | `DELIVERY_INDEX.json` 更新，指纹刷新，门禁全绿 |

## 6. 退出与止损条件

- 任一方向 L3 不通过：停在该方向，保持 default-off，结论写「数值未过，方向不采纳」。
- 连续两轮改进都落在离散度内：按「方向成立、当前形状无收益」收尾，不追加调参。
- 需要未公开或随版本漂移的接口才能推进：记为机制缺口，不写入交付面。
- 复现失败：保留失败记录与原始产物索引，禁止只留成功样本。

## 附录 A 硬件常数标定表（D0 回填）

| 常数 | 用途 | 取值 | 离散度 | 测法/日期 | 备注 |
|---|---|---|---|---|---|
| 片外带宽 | `--peak-gb-s` | 待填 | 待填 | 待填 | 决定机器平衡点 |
| Cube 峰值算力 | `--peak-tflops` | 待填 | 待填 | 待填 | 决定机器平衡点 |
| 可用片上容量 | `--ub-kib` | 待填 | 待填 | 待填 | 决定最大乒乓级数 |
| 最小搬运粒度 | 对齐判据 | 待填（预期 32 字节） | 待填 | 待填 | 决定 `DataCopyPad` 使用面 |

## 附录 B 各方向预期速查

| 编号 | 预期方向 | 预期量级（由离线口径推出，带卡后必须替换分母） | 最可能的落空原因 |
|---|---|---|---|
| D1 | 中间激活往返消失 | 上限 = 该 M 下节省占比：M=1 约 1.5%、M=10 约 12.7%、M=32 约 28.6% | 权重流量占比高，小 M 被稀释 |
| D2 | 等待下降 | 2→3 级有改善，3→4 级递减 | 片上容量或编译器自动已加深 |
| D3 | 分界点 | 落在 M ∈ [8,16] | 打包成本吃掉全部收益 |
| D4 | 精度可判定 | 余弦 ≥ 0.99999，误差 ≤ 2 倍半台阶（fp16 约 7.8e-3、bf16 约 6.3e-2） | 硬件近似函数与假设不同 |
| D5 | 调度空泡下降 | 微秒级，短步场景才显形 | 无稳定公开接口 |
| D6 | 对齐退化台阶 | 非对齐组出现明显台阶，`DataCopyPad` 修复 | 该版本 MTE 已容忍非对齐 |
