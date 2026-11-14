---
name: operator-sol-optimizer
description: Use when 用户需要优化 Ascend 算子、计算 SOL% 达成率或剩余优化空间、用 msprof 定位瓶颈，或将算子测试结果交接为仓库交付证据。
---

# 算子 SOL 与优化交付

直接由 Agent 读取任务、采集、修改、验证和交付。不启动自动生成循环，不要求实现框架接口。

## 先确定事实

读取项目的 `AGENTS.md`、`GLOSSARY.md`、`docs/operator-benchmark-standard.md`、目标源码、`ops/<operator>/evidence.json` 和 `DELIVERY_INDEX.json`。查明真实调用点、模型和权重格式、dtype、形状、stride、专家分布、相位、输出别名及中间舍入。不要把连续向量链猜成同一种计算，也不要根据模型名称猜调用次数。

本仓历史输入与公式保存在 [legacy-case-facts.json](references/legacy-case-facts.json)。它们仅帮助复现旧实验；重新提取当前宿主事实后再测试，不把 CPU 分支当作设备数值参考。现有算子实现、历史性能、失败记录与证据必须保留。临时项目放仓内隔离且忽略的目录，原始采集放仓外。

没有 NPU 时完成源码核查、算术和 CPU 检查；设备正确性、设备计时、SOL 和 MOD 测试标为未执行。不要把 CPU 时间写成设备成绩。

## 计算 SOL 与优化空间

为每种资源建立“必要工作量 / 匹配的每秒能力”账目，单位统一到秒，再计算：

```text
t_resource = necessary_work / matching_capacity_per_second
t_lower = max(t_resource)
SOL% = 100 * t_lower / t_measured
max_speedup = t_measured / t_lower
reducible_fraction = 1 - t_lower / t_measured
```

这是允许资源充分重叠的乐观下界。存在依赖的阶段按真实依赖图求界；只有证明阶段不可重叠时才能相加。脚本只算资源下界，不自动推导依赖图。

- Cube、Vector、特殊函数和通信分别记账；GEMM 的必要 FLOPs 按数学工作量计，乘加按两次运算。不要把填充或重复计算算成有效工作。
- 峰值必须匹配芯片型号、实际可用核心、dtype、稀疏或稠密模式及频率，附来源。HBM、L2、片上搬运分别用相应容量；缓存命中输入不能按每次从 HBM 读取计。
- 语义最小流量与 Profiler 的实际流量分开。不要把可消除的中间写回放进必要流量，抬高有效工作 SOL。
- MFU 只对应计算项，带宽受限算子的低 MFU 不说明优化空间大。经验标定参考效率与硬件 SOL 分开命名；二者均不能替代 MOD 的独立参考判据。
- 上述倍率和可省比例只是上界，不是承诺。SOL 超过满值时保留原值，停止下结论，检查单位、缓存、计时范围、能力和流量假设，不截断。
- 多 case 按实际调用权重计算 `sum(count * t_lower) / sum(count * t_measured)`，不要平均百分比。逐 case 结果仍需保留。
- 纯视图、别名或没有实际 kernel 的基线不做除零倍率。微小算子同时检查启动和同步成本，不能直接声称触及硬件极限。

用 [sol.py](scripts/sol.py) 复算 JSON 中的 `measured_us` 与 `resources`。每个资源值为 `[必要工作量, 每秒能力]`，所有正数必须有限；零工作资源应省略。输入旁保存设备、数据类型、缓存范围、能力来源与计时方法。脚本不检测设备，算术输出不等于设备验收。

## 采集后再改

先记录基线的重复分布、图模式、同步点和实际 kernel 清单。用 [profile.py](scripts/profile.py) 的 `timeline`、`metrics`、`simulator` 预设减少命令拼写；先加 `--dry-run` 查看参数。正式执行前核对当前安装版本的帮助、支持的指标与芯片参数，记录版本、完整 argv、退出码和产物哈希。预设不保证所有 CANN 版本兼容，不生成假采集结果，也不改变原始退出码。

```bash
python <skill>/scripts/profile.py timeline --application 'python case.py' --output <仓外新目录> --dry-run
python <skill>/scripts/profile.py metrics --application 'python case.py' --output <仓外新目录> --metric PipeUtilization --dry-run
python <skill>/scripts/profile.py simulator --application 'python small_case.py' --output <仓外新目录> --kernel <名称> --soc <实际芯片参数> --dry-run
```

`--application` 只接受已审阅的本地命令，不能拼接外部文本。采集服务端时必须包住实际服务进程；仅包住 HTTP 客户端不能得到服务 kernel。多卡逐卡核对。硬件指标采集与正式计时分开，避免插桩改变结果。非零退出即使留下部分 CSV，也记录失败及不完整范围。

按需调用已有技能：设备确认用 `ascendc-env-check` 与 `npu-arch`；MFU 用 `triton-mfu-analyzer`；Triton 修改用 `triton-latency-optimizer`，一次只改一个优化点；精度失败用 `triton-precision-debug`；功能与基准用 `triton-op-verifier`。瓶颈不明或准备声明硬件极限时，必须先用 `triton-simulator-optimizer` 采集诊断，不另建优化技术目录。技能不可用时报告缺失，不能声称已完成其步骤。

Simulator 会执行整个应用，先缩成保持关键布局和计算语义的单次调用，枚举所有相关 kernel 及 Cube/Vector 流水。小 shape 的流水比例只用于诊断；优化后回到真实尺寸计时，不能用缩小用例的结果替代生产工况。

## 测试和停止条件

先做数值检查再接受计时结论；失败立即停止该候选的性能升级。覆盖固定尺寸、未见尺寸、非连续输入、边界值、输出结构与 dtype、别名和原地写入；随机种子、容差及 NaN/Inf 规则必须来自实际任务。对会修改输入的实现每轮恢复新输入。

分别记录预热、编译、图捕获、正式 replay 与 eager；两组采用相同方法、缓存和同步范围，交错重复并保留分布。动态尺寸是否重编译要查编译缓存键、miss 或编译日志，单次调用变慢不能作为证明。

达到用户冻结的逐 case 目标且无回退后交付。未达标时根据采集结果选下一个单点修改；若预算耗尽或证据不足，保留最优有效实现和真实缺项，不编造达标。SOL 上界不能外推服务收益；只有测得关键路径份额时才用 Amdahl 估算服务上界，全核耗时总和不是关键路径墙钟。

## 证据交接

按基准规范交接实现与任务哈希、shape/dtype/stride、验证器版本、完整命令、退出码、重复统计、计时范围、能力来源及 SOL 账目。缺采集或原始定位时标记待补。

先写 `ops/<operator>/evidence.json`，再同步索引，最后用 `tools/operator_tools/custody.py` 生成并校验目录指纹。实现或依赖改变后，旧服务记录保留为历史；重新核对适用性，不能自动继承。没有独立测量的 `bench_l0.py` 必须退出码为 3。

`layer` 按实际交付归属选现有值，不按调优方式新建层。没有公开可达入口的候选使用 `gate_env=unexposed` 和待验证状态，不捏造门控。算子通过后按外环规范核对真实接入、服务数值和逐卡归因；进入 MOD 交接时使用 `mod-acceptance`，不把局部 SOL 升级为最终合格结论。

修改后跑 `pytest @tests/offline-checks.args` 和隔离 CPU 门禁。行为评估场景见 [cases.json](evals/cases.json)；没有独立 Agent 或 NPU 时明确记录未执行，不把脚本测试当成行为验证。
