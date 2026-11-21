# GDN TP split_ba 验证范围

在宿主基线 `320eced8c` 上，`AscendGatedDeltaNetAttention._split_ba_for_tp` 读取门控并调用 `vllm_ascend.ops.gdn_input_norm_host.gdn_split_ba`。该服务路径不导入插件库的同名 kernel。

[evidence.json](evidence.json) 保留独立实现的历史 L0 与数值记录，它们不是该宿主实现的重测。宿主接入不等于已有经评审的服务配对，不主张服务收益。
