# GDN Projection Unpack / Mixed-QKV Rearrange Fusion (`gdn_split_fusion`)

把 GDN 非交错路径上两处"多段小拷贝"各合成单 kernel。**一个宿主仓库、一个开关**。

| 组件 | 位置 | 替代的调用链 |
|---|---|---|
| ① 投影解包 | `vllm_ascend/ops/gdn.py` → `forward` 非交错分支（两条：普通与 in_proj 合并） | `split` + `reshape` + `chunk` + 2×`contiguous` |
| ② 混合 qkv 重排 | `vllm_ascend/ops/gdn.py` → `_forward_core` 三处 `rearrange_mixed_qkv` 调用 | `split` + 3×`reshape` + `cat` |

- 开关：`VLLM_HUST_GDN_UNPACK_FUSED=1`（默认 `0`）
- 行数门限：`VLLM_HUST_GDN_UNPACK_MAXROWS`（默认 4096），下限固定 2 行
- 实现：`vllm_ascend/ops/gdn_epilogue.py`

## 1. 背景

decode 阶段这两处每次要 5–10 µs，而实际搬运的数据只有几十 KB，开销全在 task 数量上。

原实现里 ② 更贵，因为它先对三个非连续视图各做一次 `contiguous`，再用 `cat` 拼成一块，四次搬运只为产出三个连续张量。

## 2. 做法

两个 kernel，都按"核数分块 + UB 感知内层循环"启动：

```python
grid = max(1, min(_vector_core_num(), rows))   # 最多 48 个 program
per_vec = triton.cdiv(rows, grid)              # 每个 program 分多少行
per_iter = _rows_per_iter(2 * width, elem)     # 每批行数，读写双份流量
```

内层 `tl.range(start, end, ROWS_PER_ITER)` 按 UB 容量分批。这样 prefill 的 1867 行不会摊成 1867 个单行 program。

早先的实现用 `grid = (rows,)`（一行一个 program），在 prefill 上碎得厉害，只能靠"超过 32 行就回退"规避——本版去掉了这个上限。

## 3. 组件 ② 为什么留在本仓库

`AscendGatedDeltaNetAttention` 自己重写了 `_forward_core`，三处 `rearrange_mixed_qkv` 调用都在 `vllm_ascend/ops/gdn.py` 里。因此新增 `_rearrange_mixed_qkv()` 覆盖并把三处调用指过去即可，**不需要改 vllm core**。

早先的版本改的是 `vllm/model_executor/layers/mamba/gdn/qwen_gdn_linear_attn.py::rearrange_mixed_qkv` 本体，那意味着要维护第二个宿主仓库（`vllm-hust`）和第二条基线。

## 4. 为什么组件 ① 必须走 opaque op

`forward` 在 dynamo 追踪范围内。Python 里按张量形状分支会在追踪期按首个形状常量折叠，decode 就永远走不到融合路径。所以形状判定下沉到 `torch.ops.vllm_ascend.gdn_unpack`。

组件 ② 也用了 op（`gdn_rearrange`），但它不在追踪区内，用 op 主要是为了有明确的 fake impl 与别名契约。

## 5. fail-closed 契约

op 内部判断，任一不满足就走原生链，**不抛异常**：

- 开关未设置或不等于 `1`
- dtype 非 `bfloat16`
- 输入非二维或非连续
- `ba` 宽度不等于 `2 × half`，或行数与 `qkvz` 不一致
- 宽度与 `q + k + v` 不符
- 行数小于 2 或大于 `VLLM_HUST_GDN_UNPACK_MAXROWS`

注意回退路径用 `clone` 而不是 `contiguous`：单行切片的 leading dim 为 1，PyTorch 会认为它已经连续，`contiguous` 会返回输入视图，违反 custom op 的别名契约。

## 6. 宿主归属

- 仓库 `vLLM-HUST/vllm-ascend-hust`
- Base `based/split-batch-baseline`（`8039119f`）
- 分支 `work/gdn-unpack-fused`
- 提交 `53701fb47`
- 改动 3 个文件，+551 / −26

## 7. 收益

cohort：Qwen3.5-35B-A3B、2×910B2、TP2、bf16、`cudagraph_mode=FULL`。图内打包（decode 50×，prefill 5×），单位 µs/层·step：

| M | ① 原生 → 融合 | ② 原生 → 融合 | 合计省 | 逐位一致 |
|---|---|---|---|---|
| 1 | 0.37 → 1.85（走原生，见 §8） | 0.34 → 1.77 | −2.90 | ✅ |
| 8 | 5.06 → **1.56** | 5.13 → **1.59** | **7.03** | ✅ |
| 32 | 6.19 → **3.43** | 6.18 → **3.17** | 5.78 | ✅ |
| 266 | 9.19 → **6.74** | 10.10 → **7.55** | 5.00 | ✅ |
| 1867 | 14.53 → **9.36** | 22.45 → **11.11** | **16.51** | ✅ |

按 M=8、30 层、64 step 投影：`7.03 × 1920 = 13.5 ms`，约 decode 非通信时间的 **2.6%**。

prefill 也正收益，这是本版相对上一版（prefill 直接回退）的主要改进。

## 8. 已知问题：M=1 的 op 开销

M=1 时 op 内部回退到原生链，但**调用 custom op 本身有约 1.4 µs 固定开销**，因此 ① 从 0.37 µs 变成 1.85 µs、② 从 0.34 µs 变成 1.77 µs。

30 层合计约 84 µs/step，相对 66 ms 的每 token 耗时约 0.13%。目前接受这个代价，因为形状判断必须留在 op 内（dynamo 折叠问题）。如果后续确认 vLLM 对 GDN 的 forward 按形状重新追踪，可以把判断移到 Python 层，彻底消掉这部分开销。

## 9. 交付状态

- 承载层级：`host_fork`（单仓库）
- 默认状态：关闭
- 已做：算子级逐位对拍（5 档 M，含 2 档 prefill）、8 条固定 prompt 的端到端对拍（8/8 一致）
- 未做：图内归因、服务端实测（见 REPORT.md 结论一节）
