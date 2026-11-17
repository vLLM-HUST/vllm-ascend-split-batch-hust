# tools/analysis —— 一次性取证脚本，不是通用分析工具

本目录三个脚本都是**为某一次具体取证写的**：窗口、路径、参数按当次采集写死，用完即弃，
不做通用化，也不当产品级工具维护。要复用请复制出去改成自己的窗口与路径，不要在流水线里循环调用。

| 脚本 | 适用窗口与前提 | 输入从哪来 | 产物落哪 |
| :--- | :--- | :--- | :--- |
| `ascend910b_static_audit.py` | 不需要采集窗口：只要维度、dtype 与片上容量参数。用于写 kernel 之前先把搬运字节账和分块可行性算清 | 全部来自命令行参数（`--profile`、`--m`、`--dtype`、`--ub-kib`、`--stages` 等）及文件内的维度常量 | 只把 JSON 打到标准输出，不落文件 |
| `geglu_ftz_guard.py` | 不需要采集窗口：用于低精度激活链路的误差来源分解与 L3 门禁判据 | 命令行参数（`--dtype`、`--up-max`） | 只把 JSON 打到标准输出，不落文件 |
| `matmulv2_shape_attribution.py` | 只适用于已解析出算子统计与 kernel 明细的**单个 L4 窗口**；文件顶部把窗口目录、算子统计文件、模型 `config.json` 三处路径写死，换窗口必须先改常量 | 仓外一次性记录目录里的 `*-opstat.json`、该窗口 `ASCEND_PROFILER_OUTPUT/kernel_details.csv`、模型 `config.json` | 仓外一次性记录目录下 `<stamp>-matmulv2-shape-attribution.json`；`stamp` 取自第一个命令行参数，缺省值只作占位 |

共同纪律：

- 三个脚本都只用标准库，不 import `torch` / `torch_npu`，不读设备，不需要 NPU，可在没有设备的机器上直接跑。
- 都不写仓内文件；需要留证据时由人把输出抄进对应 `ops/<operator>/` 与仓外归档，脚本不代劳。
- 静态审计里的带宽、算力、片上容量都是**输入参数**，输出一律带 `assumption: true`，不能当成设备结论引用。
- 这些脚本不做采集、不起服务、不碰冻结数据集；任何服务侧结论仍按 `docs/operator-benchmark-standard.md` 与 `docs/mod-acceptance-standard.md` 走。
