# MoE Shared-Expert Epilogue Multiply Fusion (`moe_epilogue`)

## 1. 背景与优化点

- **宿主源码位置**：`vllm_ascend/ops/fused_moe/shared_experts.py` → `AscendSharedExperts.part2`
- **配套 kernel**：`vllm_ascend/ops/triton/moe_epilogue.py`
- **算子注册**：`torch.ops.vllm.moe_gate_mul`（`vllm_ascend/ops/register_custom_ops.py`）
- **开关**：`VLLM_HUST_MOE_EPILOGUE=1`（默认 `0`）

宿主原始路径在每层每个 decode step 执行：

```python
gate_out, _ = self.layer.expert_gate(hidden_states)
shared_out = F.sigmoid(gate_out) * shared_out
```

`gate_out` 形状 `[T, 1]`，`shared_out` 形状 `[T, 2048]`。这个广播乘在 910B2 上单次约 7.99 µs，是普通小算子（≈2.5 µs）的 3.2 倍——开销来自广播类算子的固定启动成本，与搬数据量无关。

本优化把这个广播乘换成单个 triton kernel，**保留宿主 `F.sigmoid` 不动**。自算 sigmoid 与 Ascend 的 `Sigmoid` 走不同 fp32 近似路径，实测差 1–2 ulp，会改变输出 token；因此只融合乘。

## 2. 为什么必须下沉宿主

`AscendSharedExperts` 是 vllm-ascend 内部类，没有 `vllm.general_plugins` entry point、没有 `CustomOp` 注册面、没有 `--additional-config` 通道。插件层唯一能碰到它的办法是改写宿主源码文本，即运行时打补丁，已被 `docs/container-and-plugin-boundary-spec.md` §3.5 禁止。故本项以宿主 Fork 源码形式交付。

## 3. 形状判定为什么放在算子内部

`part2` 运行在 dynamo 追踪范围内。Python 里按张量形状分支会在追踪期按首个形状常量折叠，而首个被追踪的形状通常是 prefill，导致 decode 永远走不到融合路径。因此形状判定下沉到 opaque op `torch.ops.vllm.moe_gate_mul`，每次调用按真实形状决定走 kernel 还是原生表达式。

## 4. 宿主 Fork 归属

- 仓库：`vLLM-HUST/vllm-ascend-hust`
- Base 分支：`based/split-batch-baseline`（`ad82d77cb`，镜像源码 + 3 个构建文件修复）
- 开发分支：`work/moe-shared-expert-epilogue`
- 提交：`ace3d38423123c101ca2cdca8aeaaf21a7a40c5c`

改动范围（4 个文件，+238 / −1）：

```
vllm_ascend/ops/fused_moe/shared_experts.py   +10 -1   开关与调用点
vllm_ascend/ops/triton/moe_epilogue.py        +97       kernel 与形状判定
vllm_ascend/ops/register_custom_ops.py        +19       moe_gate_mul 注册
tests/ut/ops/test_moe_epilogue.py             +112      单测
```

## 5. 适用范围与回退

fail-closed：任一条件不满足即走原生 `gate * shared_out`。

- 开关未设置或不等于 `1`
- dtype 不是 `torch.bfloat16`
- `shared_out` 非二维或非连续
- `gate` 第二维不为 1，或行数与 `shared_out` 不一致
- 行数超过 `VLLM_HUST_MOE_EPILOGUE_MAXROWS`（默认 `32`），即 prefill 桶

回退方式：`VLLM_HUST_MOE_EPILOGUE=0` + 重启服务 + 重新捕获计算图（已捕获的图不能热撤销）。

## 6. 收益口径

cohort：Qwen3.5-35B-A3B、2×910B2、TP2、bf16、`cudagraph_mode=FULL`、capture `[1,2,4,8,16,32]`、chunked prefill on、APC off、无投机、无量化。

| 层 | 结果 |
|---|---|
| L0 纯算子 | 广播乘 7.99 µs → ≈3 µs（M=8，图内打包 50×） |
| L3 数值护栏 | 8 条固定 prompt 贪心生成**逐字一致** |
| L4 图内归因 | `moe_gate_mul` kernel 2470 次 / 6.35 ms；`Mul` 3187 → 617 次 |
| computing 口径 | `Mul` 22.36 ms → 2.67 ms，净 −13.10 ms；decode −2.54%，computing −1.18% |
| 服务端到端 | 三种口径均落在噪声内，**不主张端到端收益** |

口径说明：computing 指非通信设备时间。L0 加速比不得外推为服务加速；端到端结论以 `within noise` 记录。

## 7. 交付状态

- 承载层级：`host_fork`
- 默认状态：关闭（`default_enabled: false`）
- 当前：宿主分支已提交，待 PR 合入基线后由镜像构建生效；合入后需回填 `evidence.json` 的 `host_commit`
