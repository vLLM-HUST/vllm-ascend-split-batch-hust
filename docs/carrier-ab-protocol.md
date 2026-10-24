# 载体 A/B 流程（2026-10-24 定版）

适用：把某个休眠载体（`VLLM_HUST_*` 开关 / 运行树补丁）从 OFF 切到 ON，判断是否进基线。
每条载体单独一轮，不叠加，不混变量。

**前置**：先算天花板（`docs/carrier-ceiling-closure.md`）。目标算子收益上界低于
墙钟门槛（38.9 ms / 3.5%）且低于 `computing` 门槛（1.1 ms / 0.3%）的载体不安排起服，
直接按天花板结案。剩余六条的结论已在该文档中给出。

## 一、三条度量与各自的噪声界

| 度量 | 命令 | 噪声界（本机实测） | 判据 |
|---|---|---|---|
| L1 时延 | `bench-ladder.sh l1` | 会话间漂移为个位数百分比（同配置 OFF 的逐 token 时延样例存工作台回执） | 漂移范围内的 Δ 不算收益或回归 |
| light L4 | `serve --profile light` + `profile --tier light` | `hcom_allReduce` 跨窗抖动可达 4.8×（346 / 120 / 72 ms，调用次数相同） | **只比 `computing` 与算子级 total**（>1% 且可复现）；stage/free 仅参考 |
| L3 对拍 | `l3` + `compare A B` | 确定性（跨天 8/8 一致） | 用于漂移标注：`identical=k/8` + 首差样例 |

**不使用 MMLU-Pro smoke 做逐条判定**：n=20 时同配置 OFF 出现 0.55 / 0.60 / 0.40（±3 题），无判别力。
smoke 只在候选进基线前跑**一次 ≥200 题**的终局基准，与同栈 OFF 段配对。

## 二、流程

```
# OFF 段
VLLM_HUST_<CARRIER>=0 bash tools/bench-ladder.sh serve
bash tools/bench-ladder.sh l1        # 记录 ttft 等时延中位（逐 token、端到端）
bash tools/bench-ladder.sh l3        # 记录产物的 l3.json 路径
bash tools/bench-ladder.sh stop

# ON 段
VLLM_HUST_<CARRIER>=<正确取值> bash tools/bench-ladder.sh serve
bash tools/bench-ladder.sh l1
bash tools/bench-ladder.sh l3
bash tools/bench-ladder.sh stop

# ON 段 L4
VLLM_HUST_<CARRIER>=<正确取值> bash tools/bench-ladder.sh serve --profile light
bash tools/bench-ladder.sh profile --tier light
bash tools/bench-ladder.sh stop
```

注意：`serve` 在已有服务运行时只打印状态、不重启；起新段前必须先 `stop`。
取值不是布尔的载体必须先查代码（例：`VLLM_HUST_SHARED_GATE_PACK` 合法值为
`513direct|513whole|640direct|640whole`，传 `1` 会被静默解析为不启用）。

## 三、判定

| 结论 | 条件 |
|---|---|
| `accepted` | 收益超噪声（L1 >5% 或 computing/算子级 >1% 可复现）**且** L3 漂移可归因、已在候选 `reason` 中标注 |
| `rejected` | 出现超出噪声的回归（如 TPOT 明显恶化），或收益不成立而漂移成立 |
| `inconclusive` | 收益与漂移都落在噪声内；保持 default-off，写明复测前置条件 |

任何结论都必须附带：OFF/ON 两段的 l1/l3 产物路径、L4 opstat 路径、engage 证据（日志 marker）。

## 四、安全约束

- 载体默认**全部 off**；进基线的载体必须在候选记录里写"数值偏离：<证据路径>"。
- 每轮只动一条载体；同一轮内不得叠加。
- 宿主树 `/vllm-workspace/{vllm-hust,vllm-ascend-hust}` 只读，改动只能落在 `/root/host-trees/*` 的基线分支或本仓的锚点变换表。
- 结论提交只落本地，push 需单独授权。
