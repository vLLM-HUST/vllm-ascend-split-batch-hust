# 遍测遍验模式（NPU 持卡保活循环）

本文件定义一个持续运转的测试-验证队列模式：目标是让持有的 NPU 卡始终在跑**有交付价值的下一件事**，而不是闲等到被回收。规则稳定放本文件；队列的实时状态放 `.openbitfun/sweep-verify-queue.json`（不进 git，每轮就地更新）。

研发优先级只读[当前研发任务](optimization-next.json)，接续命令见 [workflow.md](../workflow.md)。本机队列只记执行进度、重试和 checkpoint；历史 pending 或已结案的服务项不是启动授权。当前项没有已评审的可执行计划就停在设计，不为占卡另取一项。

## 1. 触发与节奏

| 规则 | 内容 |
|---|---|
| 触发 | 卡空闲且当前任务有已评审计划时才启动；只取该计划内的合格项 |
| 单轮预算 | 默认一轮不超过一个自然日；被打断时依赖 `serving_e2e_auto.py` 的 checkpoint 续跑 |
| 串行 | 同一时刻只跑一个队列项，NPU 独占；不并发起两个服务 |
| 停机 | 队列取空、卡被收回或连续两项失败时停下，失败原因写回队列 |

## 2. 队列分层与准入

| 层 | 内容 | 准入判据 |
|---|---|---|
| P0 资格 | 离线门禁 + `repo_audit.py`；验收形态起服冒烟 | 每轮开工必跑；冒烟失败时本轮只做排障，不动其他项 |
| P1 复测 | 冻结形态热点重采；六算子服务 A/B；pending 长测队列 | 冒烟通过；A/B 只跑 `sync_mode: none`，结论不自动回填 |
| P2 探索 | 专家分布、长前缀新热点确认、内存安全检查 | P1 无合格项时才跑；产物只做方向记录，不产服务标签 |

准入统一判据：没有服务调用点或门控读者时必须 skip；只有已接入但缺归因规则时才可只做数值护栏；数值失败一票否决。

## 3. 循环步骤（每轮）

1. 取 `.openbitfun/sweep-verify-queue.json` 中属于已评审计划、`status: pending` 且满足准入的第一项。不得把人工说明当作命令。
2. 核对该项 preconditions（模板 plan 存在、环境变量、上轮冒烟结果）。
3. 按该项 `command` 原样执行；产物落该项 `archive` 指定的仓外目录，manifest 由采集脚本自动写。
4. 跑完把 `status` 改成 `done` / `failed`，`last_result` 写一句事实（不写结论），`updated_at` 打时间戳。
5. `failed` 的项写明原因并回炉到队尾；同因二次失败降级或转人工。
6. 有结构化结论需要落仓时（A/B 结果、指纹刷新），走正常 PR 与一致性门禁；值守会话不得直接改证据标签。

## 4. 无人值守边界

- 不 push main、不合 PR、不 approve；落仓走既有 PR 流程。
- 不动宿主 fork；`host_commits` 缺失或不匹配时停止（`host_runtime.py` 会拦）。
- 原始采集只在仓外归档目录；仓内只允许结构化登记。
- 服务收益标签只能由回填 PR 写入，值守会话只能产出待回填的归档与摘要。
- 队列文件本身不进 git；它的更新不需要任何 git 操作。

## 5. 队列文件 schema

`.openbitfun/sweep-verify-queue.json` 按下面的结构维护；缺字段的项视为非法，本轮跳过并记录：

```json
{
  "schema": "sweep-verify-queue/v1",
  "updated_at": "<UTC 时间戳>",
  "jobs": [
    {
      "id": "P0-SMOKE",
      "tier": "P0",
      "title": "<一句话>",
      "command": "<可直接执行的完整命令>",
      "preconditions": ["<事实条件>"],
      "postconditions": ["<完成后必须为真的事实>"],
      "archive": "<仓外归档目录>",
      "status": "pending | running | done | failed | blocked",
      "last_result": "<一句话事实，不写结论>"
    }
  ]
}
```

## 6. 初始队列内容

队列不进 Git。以下是可用的任务类型，不是默认排期；只有已评审计划明确要求时才加入：

- 验收形态起服冒烟（关闭 `docs/acceptance-shape.json` 的 `unverified_risks`）；
- 冻结形态热点重采（排序精化，与旧 4096/eager 表只比方向）；
- 六个已测算子在冻结形态下的 A/B 复测（前置：用 `profile_hotspot_longrun.sh` 的同套 guard 参数构造冻结形态 base plan）；
- [pending_longrun.py](../tools/operator_tools/pending_longrun.py) 生成的计划和跳过原因。用 `SERVING_BASE_PLAN` 指定已核对宿主提交的模板；不得改写历史计划来通过身份核验；
- 富余时间：GroupedMatmul 专家分布验证、新热点表确认、内存安全检查。

队列内容会消耗也会新增；新增项必须有 `id`、可执行 `command` 和仓外 `archive`，否则不许入队。
