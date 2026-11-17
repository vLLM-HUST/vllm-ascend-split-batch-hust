# 测试执行工作流

本文件是 Agent 和工程师执行各阶段测试的操作手册。每个阶段写明：
进入条件、执行动作、产出物、退出标准，以及哪些事情**不做**。

权威标准仍在 [operator-benchmark-standard.md](operator-benchmark-standard.md)、
[outer-loop-test-plan.md](outer-loop-test-plan.md) 和
[mod-acceptance-standard.md](mod-acceptance-standard.md)；
本文件只决定"怎么推进"，不重复规范正文。

---

## 核心原则

**已有结论不触发重测。** 只要 `evidence.json` 中某阶段的结论字段存在且
`passed`（或 `measurement_accepted`、`tested`）已经有值，该阶段不重跑，
直接读结论推进下一阶段。

**读证据 vs 执行测量严格分开。** 任何阶段开始前，先查 `evidence.json`
是否已有结论。有结论就读结论。没有结论才执行测量，执行完立即回填，
然后才继续。不允许"先测量后找地方记"——这会导致哈希过期和誊抄错误。

**哈希/指纹只在提交前生成一次。** 执行测量和回填证据的过程中，
不运行 `custody.py`。所有文件回填完毕，确认无误后，最后一步统一生成
目录指纹，然后提交。期间任何重新检查都只读文件内容，不重生成哈希。

---

## 阶段 0：确认当前状态

**进入时机**：每次开始新一轮工作前，或不确定某算子处于什么阶段时。

### 执行

```bash
python3 -c "
import json
idx = json.load(open('DELIVERY_INDEX.json'))
for op in idx['operators']:
    print(op['id'], '|', op['status'], '|', op.get('serving_status',''), '|', op.get('gate_env',''))
"

python3 -c "
import json
ev = json.load(open('docs/mod-acceptance/evidence.json'))
plan = ev['assessment']['combination_plan']
print('tested:', plan['tested'])
print('blocking:')
for b in plan.get('blocking', []):
    print(' -', b)
"
```

### 产出

一张三列表格：算子名 / 最高已完成阶段 / 下一步。
最高已完成阶段的判断规则：

| 检查字段 | 对应阶段 |
|---|---|
| `l3_numerical_guard.passed == true` | L3 完成 |
| `l4_attribution.verified == true` 或宿主算子已有调用计数记录 | L4 完成 |
| `serving_end_to_end.tested == true` | 服务测试完成 |

### 不做

- 不重新跑任何测试
- 不检查哈希或指纹
- 不更新任何文件

---

## 阶段 1：L3 数值检查

**进入条件**：某算子 `l3_numerical_guard` 字段不存在，或 `passed` 为 null。

如果 `l3_numerical_guard.passed` 已经是 true 或 false，直接读结论，不重跑。

### 执行

```bash
env -u PYTHONPATH VLLM_HUST_<GATE>=1 \
  .openbitfun/cpu-tensor-venv/bin/python \
  .openbitfun/skills/triton-op-verifier/scripts/verify.py \
  --op_name <算子名> \
  --verify_dir <仓外新目录> \
  --output <仓外新目录>/verify_result.json
```

命令执行完毕，立即查看 `verify_result.json` 的 `passed` 字段。

### 立即回填

在 `evidence.json` 中写入 `l3_numerical_guard` 字段，包含：
- `passed: true/false`
- `metric`：使用的误差指标
- 失败时保留 `max_abs_diff` 和失败的 case

失败时停止，记录失败，保持算子 `default_enabled: false`，不继续后续阶段。

### 不做

- 不调整容差来让失败变通过
- 不删除失败的 case 记录
- 如果 `evidence.json` 里已经有 `passed: true`，不重跑

---

## 阶段 2：L0 微基准

**进入条件**：L3 已通过，且 `evidence.json` 中没有已接受的 L0 计时结论
（`measurement_accepted` 不为 true，或 `l0_microbenchmark` 字段不存在）。

如果 `measurement_accepted: true` 已经存在，直接读 speedup，不重跑。

### 执行

```bash
env -u PYTHONPATH .openbitfun/cpu-tensor-venv/bin/python \
  ops/<算子名>/bench_l0.py
```

如果 `bench_l0.py` 退出码为 3，该算子无孤立测量入口，
L0 阶段标记为"无测量"，直接进入 L4。

### 立即回填

在 `evidence.json` 写入测量结果，包括：
- `measurement_accepted: true/false`
- 中位数、P90、speedup
- 测量时间戳和环境信息

写完后不再修改这个字段（除非重新跑了新一轮测量并明确标注 run_id）。

### 不做

- 不因为倍率"看起来不对"就重跑
- 不把孤立倍率外推为服务收益
- 宿主 fork 算子（`gdn_conv_pack`、`gdn_merge_inproj`）的 L0
  反映的是消除重复工作，不是 kernel 加速，记录时须注明，不用和服务收益对比

---

## 阶段 3：L4 调用归因

**进入条件**：L3 已通过，且该算子纳入组合候选或需要证明实际生效。
宿主 fork 算子使用 `evidence_fingerprint.host_commit` 核对宿主源码版本，
不重验已有版本。

如果 `l4_attribution.verified: true` 已存在，读结论，不重跑。

### 执行

起服时设置目标开关为 1，发送单批固定请求，收集 `op_statistic.csv`：

```bash
python3 -m tools.operator_tools.serving_attribution \
  --config <采集配置.json> \
  --output <仓外新目录>/l4_attribution.json
```

核对：
- 目标 kernel 命中次数 == 层数 × 步数
- 关闭组目标 kernel 命中次数 == 0
- rank0 和 rank1 均有记录

### 立即回填

`evidence.json` 的 `l4_attribution` 字段写入：
- `verified: true/false`
- `call_count_expected / call_count_actual`
- 逐卡记录（rank0/rank1 分开）
- `host_source_commit`（宿主算子）

### 不做

- 不用 eager 模式的归因替代图回放归因
- 不在 TP2 环境中只记 rank0 的结果就声称全卡验证通过
- 计数不对不当作可忽略偏差——计数不对说明算子没有真正生效，查清原因再继续

---

## 阶段 4：单算子服务测试（按需）

**进入条件**：
1. L3 通过，L4 通过（或宿主算子有调用记录）
2. 该算子有独立公开的 gate 变量（`gate_env` 不是 `unexposed`）
3. 决策：关键路径占比足够高、存在交互冲突风险、或需要独立收益归因

如果 `serving_end_to_end.tested: true` 已存在，读结论，不重跑。

### 执行前确认宿主版本

```bash
python3 -c "
from tools.operator_tools.host_runtime import check_host_runtime
import json
ev = json.load(open('docs/mod-acceptance/evidence.json'))
commits = ev['assessment']['combination_plan']['host_commits']
result = check_host_runtime(commits)
print('host verified:', result)
"
```

确认通过后执行四轮采集：

```bash
python3 -m tools.operator_tools.serving_collect \
  --config <采集配置.json> \
  --output <仓外新目录>
```

四轮全部完成后，才进入回填步骤。任何一轮失败或请求数不完整，
保留失败记录，不汇总结论，重新起服重跑失败的那轮。

### 立即回填（按顺序）

1. 把原始日志、响应文件存入仓外归档目录，记录路径
2. 计算 `speedup = gen_rate_on / gen_rate_off`（两轮 ON 和两轮 OFF 各取中位数）
3. 计算组内波动：`intra_group_spread_pct`
4. 判断结论：`speedup - 1` 超过组内波动最大值才写 `verified_faster`，否则写 `within_noise`
5. 在 `evidence.json` 写入 `serving_end_to_end` 完整字段
6. 在 `DELIVERY_INDEX.json` 对应算子更新 `serving_status`

步骤 5 和步骤 6 必须在同一轮操作里完成，不允许只更新其中一个。

### 不做

- 不用 sharegpt 的正向结果覆盖 random 主测的 `within_noise` 结论
- 不用历史测量结果（`applicability: historical`）填当前结论字段
- `gate_env: unexposed` 的算子不进行服务测试，保持 `pending_outer_loop`

---

## 阶段 5：组合测试

**进入条件**：以下所有条件同时满足：
- 所有纳入 `combination_plan.candidates` 的算子 L3 均已通过
- 所有候选均有 L4 归因记录，或其优化本质是消除加载期冗余且有调用计数
- `combination_plan.blocking` 列表为空

**当前 `blocking` 还有三条未清。在清掉之前不执行本阶段。**

### 清 blocking 的操作

| blocking 条目 | 清除方式 |
|---|---|
| 重新验证组合数值及各候选归因（图回放模式） | 对组合开关全开状态，跑固定提示词逐字对拍，通过后在 `combination_plan` 写 `numerical_guard: passed` |
| 审查行数限制和 prefill/decode fallback | 在目标宿主源码中找到行数 guard 的实际条件，记录在 `combination_plan.row_limit_audit` |
| 采集器只支持单门控 | 确认 `serving_collect.py` 支持多个环境变量同时设置，记录确认结论 |

### 执行

```bash
python3 -m tools.operator_tools.serving_collect \
  --config combination_config.json \
  --output <仓外新目录>/combination-ab
```

`combination_config.json` 中的 gate 字段包含所有候选开关。
顺序：OFF ON ON OFF，每轮三个负载（random / sharegpt / long-prefix）。

### 立即回填

和阶段 4 一样，先存原始文件，再算结论，再写证据，再更新索引。
结论写入 `docs/mod-acceptance/evidence.json` 的 `assessment.combination_plan` 下，
不回填进各算子的 `serving_end_to_end`。

组合测试和算子单项测试的结论是独立的，不互相填充。

---

## 阶段 6：MOD 验收

**进入条件**：组合测试有完整的 OFF/ON 配对结果。

MOD 验收使用冻结的五个数据集，不能用算子测试的负载替代。
当前 `assessment.execution_complete: false`，五个数据集均未执行。

**本阶段在算子组合配对结果出来之前不启动。**

流程见 [mod-acceptance-standard.md](mod-acceptance-standard.md) 和
[mod-acceptance skill](../.openbitfun/skills/mod-acceptance/SKILL.md)。

---

## 证据回填的完整顺序

每次完成一次测量后，回填步骤固定如下，不跳步：

```
1. 把原始产物存入仓外目录，记录路径
2. 写 ops/<算子>/evidence.json
3. 写 DELIVERY_INDEX.json 的对应算子字段（evidence 写完后再改索引）
4. 确认两处结论方向一致（serving_status 和 evidence 里的方向相同）
5. 全部算子都处理完后，统一运行一次指纹生成：
   python3 tools/operator_tools/custody.py --write ops/<算子>
   python3 tools/operator_tools/custody.py --verify ops/<算子>
6. 运行 pytest @tests/offline-checks.args 确认通过
7. 提交
```

步骤 2 和步骤 3 必须在同一轮操作里完成，不允许只更新其中一个。
步骤 5（生成哈希）只在步骤 2-4 全部完成后做一次，不在中间过程中运行。

---

## Agent 执行的决策树

```
开始
  |
  v
读 DELIVERY_INDEX.json 和各算子 evidence.json
  |
  +-- 某算子 l3_numerical_guard.passed 不存在或为 null
  |     => 执行阶段 1（L3）
  |
  +-- 某算子 L3 通过 且 measurement_accepted 不存在
  |     => 执行阶段 2（L0），如退出码=3 则跳过
  |
  +-- 某候选算子 l4_attribution.verified 不存在
  |     => 执行阶段 3（L4）
  |
  +-- 组合 blocking 列表不为空
  |     => 逐条清 blocking，不执行组合测试
  |
  +-- 组合 blocking 为空 且 combination_plan.tested = false
  |     => 执行阶段 5（组合测试）
  |
  +-- 单算子需要独立服务结论（见阶段 4 进入条件）
  |   且 serving_end_to_end.tested = false
  |     => 执行阶段 4
  |
  +-- 所有候选均有完整证据 且 组合已测
        => 执行阶段 6（MOD 验收）
```

如果当前状态不满足任何分支的进入条件，说明有阻塞项未清或等待外部条件，
不擅自启动任何测试。报告当前状态和第一个阻塞项，停下来等指令。

---

## 常见错误与正确处理

| 错误做法 | 正确做法 |
|---|---|
| 看到 `checksum.sha256` 和文件内容不匹配就立即重跑测试 | 哈希不匹配说明文件在上次生成哈希后被修改了；检查哪里改了，修正文件内容，统一重生成哈希，不重跑测试 |
| `measurement_accepted: false` 就重跑 L0 | 读清楚为什么 false；如果原因是"形状合成和真实调用不符"，记录限制，不用这个数字，不重跑 |
| 服务测试数据写入 evidence 后发现 speedup 计算有误，要重测 | 先确认原始日志在仓外目录里是否完整；如果完整，重新计算并覆盖 evidence 字段，不重跑服务测试 |
| 一次对话里测了 L3 但忘了改 DELIVERY_INDEX，下次重测 | 本次对话里只要完成了测量，立即走完"证据回填的完整顺序"全部 7 步，不跨会话留悬空 |
| 把历史服务结果（`applicability: historical`）的数字填入当前 `speedup` 字段 | 历史结果只留在 `historical_only` 字段，不能直接用作当前结论 |
| 某算子 L0 倍率很大，直接写"该算子实现了高速服务收益" | L0 倍率只代表孤立算子；服务收益只能来自服务测试的 speedup 字段 |
