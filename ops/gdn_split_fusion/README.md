# GDN Split / Rearrange Copy Fusion (`gdn_split_fusion`)

把 GDN 里两处"多段小拷贝"合成单 kernel。两个组件分别落在两个宿主仓库，开关独立，都可单独回退。

| 组件 | 宿主仓库 | 位置 | 开关 |
|---|---|---|---|
| ① 非交错投影的 split + contiguous 融合 | `vllm-ascend-hust` | `vllm_ascend/ops/gdn.py` → `AscendGatedDeltaNetAttention.forward` 非交错分支 | `VLLM_HUST_GDN_SPLIT_FUSION=1` |
| ② 混合 qkv 重排融合 | `vllm-hust` | `vllm/model_executor/layers/mamba/gdn/qwen_gdn_linear_attn.py` → `QwenGatedDeltaNetAttention.rearrange_mixed_qkv` | `VLLM_HUST_GDN_REARRANGE_FUSION=1` |

## 1. 背景

Qwen3.5 的 GDN 层在 decode 阶段有两次"把一条混合 buffer 拆成多块连续张量"的操作，每次在 910B2 上要 5–10 µs，而实际搬运的数据只有几十 KB。开销来自 task 数量，不是带宽。

**组件 ①**（非交错布局、非交错投影）：

```python
mixed_qkv, z = mixed_qkvz.split([qkv_size, z_size], dim=-1)
z = z.reshape(z.size(0), -1, self.head_v_dim)
ba, _ = self.in_proj_ba(hidden_states)
b, a = self._split_ba_for_tp(ba)
b = b.contiguous()
a = a.contiguous()
```

`split` 出的是视图，但 `b`、`a` 各要一次 `contiguous()` 真正拷贝，两次 task。

**组件 ②**（`rearrange_mixed_qkv`）：

```python
query, key, value = torch.split(mixed_qkv, [q_dim, k_dim, v_dim], dim=-1)
fused = torch.cat([query.reshape(-1), key.reshape(-1), value.reshape(-1)], dim=0)
q_contig = fused[0:q_size]
k_contig = fused[q_size : q_size + k_size]
v_contig = fused[q_size + k_size :]
```

一段 `split`、三个 `reshape`、一个 `cat`，共四次 launch，只为产出三个连续 buffer。

## 2. 做法

组件 ①：单 kernel 一次写入 `mixed_qkv`、`b`、`a` 三个 buffer。`z` 不进 kernel——对连续切片做 `reshape` 是纯视图，留在外面还能避免 custom op 返回别名。

组件 ②：单 kernel 直接写 `q`、`k`、`v` 三个连续 buffer，调用方再 `.view(1, seq, -1, head_dim)`。视图后的形状、连续性、数值与原实现逐位一致。

## 3. 为什么组件 ① 必须走 opaque op

`forward` 在 dynamo 追踪范围内。Python 里按张量形状分支会在追踪期按首个形状常量折叠，而首个被追踪的形状是 prefill 分块，decode 就永远走不到融合路径。所以形状判定下沉到 `torch.ops.vllm.gdn_split_pack`，每次调用按真实形状决定。

组件 ② 不需要 op：`rearrange_mixed_qkv` 只从 `qwen_gdn_attention_core` 内部被调用，而后者是 `direct_register_custom_op` 注册的自定义 op，函数体不被追踪，普通 Python 分支安全。

## 4. 宿主归属

**组件 ①**

- 仓库 `vLLM-HUST/vllm-ascend-hust`
- Base `based/split-batch-baseline`（`ad82d77cb`）
- 分支 `work/gdn-split-fusion`
- 提交 `16e66804c`
- 改动 4 个文件，+328 / −6

**组件 ②**

- 仓库 `vLLM-HUST/vllm-hust`
- Base `based/split-batch-baseline`（`f18cf803c`）
- 分支 `work/gdn-rearrange-fusion`
- 提交 `8c9f165259`
- 改动 2 个文件，+133 / −9

## 5. 适用范围与回退

组件 ① fail-closed 条件（任一不满足走原生）：

- 开关未设或不等于 `1`
- 层上有 `split_ba`（KP 布局，切分方式不同）
- dtype 非 `bfloat16`、输入非二维或非连续
- `ba` 宽度不等于 `2 × half`，或行数与 `mixed_qkvz` 不一致
- 行数小于 `VLLM_HUST_GDN_SPLIT_FUSION_MINROWS`（默认 2）或大于 `VLLM_HUST_GDN_SPLIT_FUSION_MAXROWS`（默认 32）

组件 ② fail-closed 条件：

- 开关未设或不等于 `1`
- dtype 非 `bfloat16`、输入非二维或非连续
- 最后一维不等于 `2 × q_dim + v_dim`
- 行数超过 `VLLM_HUST_GDN_REARRANGE_MAXROWS`（默认 32）

两个开关互不依赖，可以分别开关。回退：把对应 env 置 `0` + 重启 + 重新捕获计算图。

## 6. 收益

cohort：Qwen3.5-35B-A3B、2×910B2、TP2、bf16、`cudagraph_mode=FULL`、capture `[1,2,4,8,16,32]`。L0 为图内打包 50× 的单层单步耗时。

| M | 组件 ① 原生 → 融合 | 组件 ② 原生 → 融合 | 合计省 |
|---|---|---|---|
| 1 | 0.24 → 0.23（未融合） | 4.31 → 0.99 | 3.32 µs |
| 8 | 3.32 → 1.46 | 9.31 → 1.42 | **9.75 µs** |
| 16 | 3.34 → 2.07 | 9.35 → 1.96 | 8.66 µs |
| 32 | 4.05 → 3.34 | 10.26 → 3.02 | 7.96 µs |

按 M=8、30 层、64 step 投影：`9.75 × 30 × 64 = 18.7 ms`，相当于 decode 非通信时间（515.2 ms）的 **3.6%**、computing 分母（1108.9 ms）的 **1.7%**。

上表为 L0 微基准，尚未做 L4 图内归因与服务端到端；投影值不可当作实测收益引用。

## 7. 交付状态

- 承载层级：`host_fork`（两个宿主仓库各一处）
- 默认状态：全部关闭
- 当前：两侧代码已提交到特性分支，未合入基线，未重建镜像
- 待办：L3 对拍、L4 归因、端到端（见 REPORT.md 第 6 节）
