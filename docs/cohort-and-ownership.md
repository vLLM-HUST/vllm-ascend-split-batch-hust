# cohort 边界与归属（长期核心约束）

> **本文件是"我们能声称什么、谁负责什么"的权威源。**
> 约束已写进 [AGENTS.md](../AGENTS.md) 硬约束 6；此处给依据、边界与判定口径。
> 来源：上游 @ShuhaoZhangTony 的两条裁定（本仓 issue，逐字原文见
> `knowledge/handoffs/receipts/20260929-upstream-cohort/facts.md`，工作区）：
> - issue #2 comment `5867233862`（2026-09-28，**Frontier 合同审计结论**）
> - issue #4 comment `5891258209` / `5891262039`（2026-09-29，**owner handoff**）

## 1. 上游裁定说了什么（两条，逐字可核）

**① 审计结论（#2，审计对象 = 本仓 `f77dc2214727ab1448874f1aee787a67328f1433`）**

> "当前 … 明确把任何 `speculative_config` 判为 `speculative_decode_conflict`，README 说明这只是 guard、
> 不是 MTP 支持，启用时 cascade 不产生加速；HOST_CONTRACT 也要求拒绝 speculative decoding。
> 统一配置必须保留 native MTP2，不能关闭或替换，所以本轮未绕过 precheck，**也没有生成形式上成功
> 但实际退化为 Native 的点**。… 要进入统一 cohort，需要先设计支持 k+1 verification rows 的
> dual-pad/graph bucket 合同，并补 native MTP2 + chunked prefill、APC、async、FULL_AND_PIECEWISE、
> Qwen3.5 hybrid TP2 的 fail-closed 与真实 replay correctness。**现有 Qwen2.5 数据继续作为独立
> 专项证据，不应外推到该配置。**"

⇒ 两句可直接引用的结论：**未发现作弊式结果**；**不得外推**。

**② 归属 handoff（#4）**

> "owner handoff: 0.1.3 is executable and should remain visible, but its Cascade path deliberately
> does not exercise MTP steps and targets the 0.25 host line. Independent cohort definition,
> kernel-wheel licensing/distribution, NPU evidence, and performance claims remain with the project
> owners; the central team will only preserve the classification and publication boundary."

⇒ 中央团队**只保留"分类与发布边界"**；下列四项**回到本仓 owner**：

| # | 归属项 | 本仓现状（可核） | 缺口 |
|---|---|---|---|
| ① | independent cohort definition | 宿主域 [release.md](release.md) §0.1/§0.3；能力域 [support-matrix.md](support-matrix.md) §5.1 | 缺一页正式 cohort 声明（本文件 §2 即其初版） |
| ② | kernel-wheel licensing / distribution | 许可齐全（wheel 内含 CANN OSL 2.0 协议文本、METADATA 已订正、根 `LICENSE`/`NOTICE`）；已按 GitHub Release 附件分发（算子仓 `v2026.9.27`，附件 = 本地构建字节） | **算子仓仍在个人账号 `Raing5Days/vllm-hust-cascade-kernel`**，org 归属未定 |
| ③ | NPU evidence | [evidence/cascade/](evidence/cascade/README.md)（147 件）+ 目标栈验证程序（C 口径 6/64、参考帧 7/64、6 格 TPOT） | 只覆盖 0.25 线 + Qwen2.5；Qwen3.5/MTP2 域为**未验证** |
| ④ | performance claims | [evidence/cascade/section3-performance.md](evidence/cascade/section3-performance.md) §13.5 | 对外引用必须带 §3 的 cohort 边界 |

## 2. 本仓 cohort（我们能声称的适用域）

**覆盖（声明兼容且已实测）**：

| 维度 | 取值 | 依据 |
|---|---|---|
| 宿主 `vllm-ascend` | **0.25.x 线**：`>=0.25.1rc2.dev125,<0.25.2` | manifest `host.version_range`；两端点均真机验证 |
| vllm core | 配对已验证：`0aee727ff6`（窗口内配对） | 见 release.md §0.3 的配对警告 |
| 模型 | Qwen2.5-14B-Instruct（**行为类证据**）/ Qwen2.5-Coder-14B（**性能替身**，两者不可跨模型比） | evidence/cascade §12、README |
| 请求形态 | **每请求恰好 1 个 query row** | `cascade_plugin._use_cascade_attention` 条件 7 |
| 调度/图 | APC 开 + `FULL_AND_PIECEWISE` + capture sizes 取 `[32,64,128]` | release.md §0.1、pitfalls §1.3/§3.5 |
| 精度档 | Tier-0 bf16（图模式）/ Tier-1 fp32（eager only） | evidence/cascade §3、§5 |

**不覆盖（不得据此声称）**：

- Qwen3.5-35B-A3B hybrid **TP2**、**native MTP2**（k+1 verification rows）、chunked prefill；
- 未跑过的 `async` 组合；
- 0.25 线以外的宿主 build；
- 任何形式的投机解码（含 MTP）——**一律 fail-closed**。

## 3. 引用纪律（写结论时逐条对照）

1. **任何性能/正确性数字必须带 cohort 边界**（至少写：宿主线、模型、请求形态、APC/图配置）。
   缺一项即视为不可引用的孤立数字。
2. **禁止把 Qwen2.5 的数字外推到 Qwen3.5/MTP2 配置**；该配置上跑出来的"cascade ON"
   是**原生路径**（准入守卫已拒），**不构成 cascade 效果** —— 这句话本身就是结论。
3. **MTP 只能写成"边界/拒绝"**，不得写成"支持"。可引用 README
   [「Speculative decoding boundary」](../README.md) 与 `HOST_CONTRACT.md:11`。
4. **要改 cohort**（例如进统一 cohort）先满足两个前置：k+1 verification rows 的
   dual-pad/graph bucket 合同 + 该配置上的真实 replay correctness。二者都缺时，
   不得在文档、Release note 或回帖里暗示"即将支持"。
5. 该前置**当前未立项**（与工作区台账 OPEN-05 的既有裁定一致：维持 fail-closed 边界）。
   立项需项目侧明确裁定；**不得通过"绕过 precheck"来造出形式上的成功点**。

## 4. 变更本文件的条件

下列任一发生即须同步本文件与其引用处（`AGENTS.md` 硬约束 6、`support-matrix.md` §5.1、README）：

- `host.version_range` 变更（放宽/收窄）；
- 上游对 cohort 或归属有新裁定；
- 四项归属中任一的状态变化（如算子仓转 org、补出 Qwen3.5 域证据）；
- MTP / k+1 形态的合同或实现发生变化。
