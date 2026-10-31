# GDN QKV Single-pass Unpack Kernel (`gdn_qkv_unpack`)

## 1. 背景与优化点
- **源码位置**：`src/vllm_ascend_split_batch/gdn_qkv_unpack/`
- **入口函数**：`rearrange_mixed_qkv`
- **优化原理**：使用基于 Triton-Ascend 实现的单 pass 内核，一次拷贝完成 `[T, q]`, `[T, k]`, `[T, v]` 三个行切片向连续扁平缓冲的重排，意图替代原生 PyTorch 链上的 `split + 3×reshape(-1) + cat`（共 4 次算子 Launch）。
- **注册与门控状态**：未挂载为公开 entry point 或 bundle manifest，保持惰性延迟导入。

## 2. 设备核验现状与负性能结论
- **NPU 硬件核验**：在 Atlas 800T A2 (Ascend910B2) 上执行配对 A/B 微基准测试（`tools/bench_gdn_qkv_unpack.py`）。
- **核验结论**：在所测真实与合成 token 形状下，Triton 内核比原生 PyTorch 链更慢（T=1 耗时增加 97.3%，T=16 耗时增加 5.5%，T=64 耗时增加 3.1%）。
- **瓶颈归因**：单次 Triton 内核在 910B2 上的 Launch 启动开销（~110-140 µs）显著超过了纯数据重排搬运节省的时间，而原生实现中 split/view 均为零拷贝元数据变换。
- **交付决策**：严格保持 **default-off**（未接管模型前向），记录负性能结论，不主张任何端到端加速收益。实际运行中 GDN 的核心瓶颈由宿主层卷积权重加载期预打包优化（`ops/gdn_conv_pack`）承载。
