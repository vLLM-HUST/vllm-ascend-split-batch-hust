# GDN 输出归一化：宿主实现与测量范围

`AscendGatedDeltaNetAttention.forward` 在启用且输入满足约束时调用宿主 `gdn_output_norm_rearrange`；否则调用原生 `self.norm`。历史版本曾因 `z` 的非连续列切片回退；注册和连续输入的单测都不能证明模型命中优化。

当前受控服务配对的主测判定为 `within_noise`，只适用于 [evidence.json](evidence.json) 登记的工况。旧工况的正向记录不作为当前结论。

旧 `bench_l0.py` 调用的是本仓 `gdn_norm_gate_2d_kernel`，基线是手写 PyTorch 链，不是宿主的 `AscendRMSNormGated`。它不能重放宿主 L0，现已停止输出计时。重开须使用目标宿主的原生与优化函数、模型的实际 stride，并独立验证数值与调用归因。
