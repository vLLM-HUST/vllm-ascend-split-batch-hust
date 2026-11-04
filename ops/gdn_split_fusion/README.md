# GDN 投影解包 / 混合 qkv 重排融合

把 GDN 非交错路径上两处多段拷贝各合成单个 kernel。单仓库、单开关。

| 组件 | 位置 | 替代的调用链 |
|---|---|---|
| ① 投影解包 | `vllm_ascend/ops/gdn.py` → `forward` 非交错分支 | `split` + `reshape` + `chunk` + 2×`contiguous` |
| ② 混合 qkv 重排 | `vllm_ascend/ops/gdn.py` → `_rearrange_mixed_qkv` | `split` + 3×`reshape` + `cat` |

实现：`vllm_ascend/ops/gdn_epilogue.py`（两个 kernel + `torch.ops.vllm_ascend.gdn_unpack` / `gdn_rearrange`）。

## 使用

```
VLLM_HUST_GDN_UNPACK_FUSED=1        开启，默认 0
VLLM_HUST_GDN_UNPACK_MAXROWS        行数上限，默认 4096
```

行数下限固定为 2。以下任一情况回退原生链，不抛异常：开关未开、dtype 非 bf16、输入非二维或非连续、`ba` 宽度与行数不符、行数越界。

## 宿主归属

| 项 | 值 |
|---|---|
| 仓库 | `vLLM-HUST/vllm-ascend-hust` |
| Base | `based/split-batch-baseline`（`8039119f`） |
| 分支 | `work/gdn-unpack-fused` |
| 提交 | `53701fb47`（实现）、`82e140a9a`（修复） |
| 改动 | 3 文件，+551 / −26 |

## 收益

单次调用（NPU 图内打包，M=8，单层单步，真实形状）：

| 组件 | 原生 | 融合 | 省 |
|---|---|---|---|
| ① 投影解包 | 4.71 µs | 1.60 µs | 3.11 |
| ② 混合 qkv 重排 | 4.73 µs | 1.48 µs | 3.25 |
| 合计 | 9.44 µs | 3.08 µs | **6.36** |

全链路归因（`op_statistic`，`Slice` + `ConcatD` 被两个 kernel 替代）：

| 负载 | 原生 | 融合后 | 净省 | 占比 |
|---|---|---|---|---|
| 纯 decode（输入合计 366 token） | 633.5 ms | 283.3 ms | −350.2 ms | 2.41% |
| 短上下文（66k KV token） | 4478.9 ms | 2080.6 ms | −2398.3 ms | 5.00% |
| 长上下文（263k KV token） | 8560 ms | 5163 ms | −3397 ms | 5.05% |

## 结论

**保持默认关闭。**

服务端两个负载各测 8 / 12 腿，差异都落在各自腿间极差内（短上下文 +0.46%、长上下文 −1.27%），不主张收益。两档的算子级收益几乎相同（5.00% 与 5.05%），服务层却一正一负，说明正负号只是噪声——减少 kernel 时间没有传导到 wall time。详见 [REPORT.md](REPORT.md) 第 5 节。

## 已知限制

行数为 1 时会多付约 1.4 µs 的 op 调用开销，此时原生链退化为零 kernel（`contiguous()` 返回视图），因此下限设为 2。
