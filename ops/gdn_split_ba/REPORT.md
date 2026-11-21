# GDN TP split_ba 验证范围

在宿主基线 `320eced8c` 上，`AscendGatedDeltaNetAttention._split_ba_for_tp` 读取门控并调用 `vllm_ascend.ops.gdn_input_norm_host.gdn_split_ba`。该服务路径不导入插件库的同名 kernel。

在相同的 eager、关闭前缀缓存工况下，开关两组的固定请求输出逐字一致。两张卡关闭时均未命中目标核，开启时均达到预期调用数。主测 `random`、压力负载 `sharegpt` 和交叉检查 `long-prefix` 的输出吞吐均略降，但均未超过各自的有效噪声线，结论为 `within_noise`。保持默认关闭；此结果不是 MOD 组合验收。

模型、宿主提交、逐负载测量和归档指纹见 [evidence.json](evidence.json)。其中独立实现的历史 L0 记录不是对宿主 kernel 的重测。
