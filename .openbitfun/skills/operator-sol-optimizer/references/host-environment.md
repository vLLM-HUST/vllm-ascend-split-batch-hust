# 宿主环境选择与启动检查

先发现已有运行环境，再决定是否安装或构建。系统 `python` 的包版本不能代表容器里的全部环境。不要因系统 Python 缺模块就重新克隆宿主、建 venv 或禁用原生编译。

## 当前容器的已核验位置

以下是本机定位线索，每次执行仍需核验；其他机器不得照搬路径。

| 项目 | 位置或值 |
|---|---|
| 服务解释器 | `/root/venvs/split-batch-runtime/bin/python` |
| CPU 检查环境 | `/root/vllm-workspace/vllm-ascend-split-batch-hust/.openbitfun/mod-acceptance-venv/bin/python`，无宿主，含 pytest/ruff/pre-commit |
| vLLM HUST 源码 | `/root/host-trees/vllm-hust` |
| Ascend HUST 源码 | `/root/host-trees/vllm-ascend` |
| vLLM 提交 | `f18cf803c5f63625e2c71253ddaf8b0bad0bad1a` |
| Ascend 提交 | 以 `/root/host-trees/vllm-ascend` 当前 `based/split-batch-baseline` HEAD 为准（每次执行由 `host_runtime.py` 核验，不沿用旧快照哈希） |
| PyTorch / torch-npu | `2.13.0+cpu` / `2.13.0rc1` |
| CANN 环境入口 | `/usr/local/Ascend/ascend-toolkit/set_env.sh` |
| Ascend 扩展 | 宿主包内 `vllm_ascend_C.cpython-312-aarch64-linux-gnu.so` |
| 自定义算子库 | 宿主包内 `_cann_ops_custom/vendors/custom_transformer/op_api/lib/libcust_opapi.so` |

系统安装来源 `/vllm-workspace/vllm` 和 `/vllm-workspace/vllm-ascend` 是另一套环境。不要与 `/root/host-trees/` 混用。包版本字符串可能保留安装时的旧提交；editable 安装以当前 Git HEAD、干净工作树和实际模块来源共同核验。

## 执行顺序

1. 查用户指定的解释器、已有启动配置，以及用户目录下 `venvs/` 中的 `pyvenv.cfg`。限定目录查找，不遍历容器镜像缓存或读取凭据文件。
2. 在候选解释器内检查 `sys.executable`、两个宿主的 `direct_url.json` 和 `find_spec(...).origin`、torch 与 torch-npu 版本。服务配置填写绝对解释器路径及两个完整 `host_commits`，调用仓库 `tools/operator_tools/host_runtime.py` 核验。CPU pre-commit 验证此拒绝逻辑，不加载设备依赖。
3. 在独立子 shell 中先清除继承的 `PYTHONPATH`，再 source CANN 官方 `set_env.sh`。检查 `acl` 是否能导入，记录官方脚本恢复的依赖路径；不要再次清空这些路径，也不要把宿主源码目录注入 `PYTHONPATH`。服务与预检必须采用相同环境。
4. 检查宿主扩展及自定义算子库是否存在；用 `nm -D` 核对所需接口，再做真实启动和预热。存在源码、存在符号、成功加载、运行成功是不同证据，不能互相替代。
5. 缺接口时先区分解释器选错、CANN 配置未加载、动态库搜索路径错误、构建产物缺失和版本不兼容。`COMPILE_CUSTOM_KERNELS=0` 只装 Python，不能作为完整服务环境。确认没有可用已有环境后才考虑安装或构建。

服务预热通过后，按模型数值检查、逐卡调用归因、平衡 A/B 顺序采集。局部源码参考结果不能升级为宿主集成证据。

## 清理边界

误建环境的启动失败不是算子数值失败。清理前停止其进程，核对精确路径和证据引用；只删除误建 venv、重复 checkout、构建缓存与错误环境日志。保留有效算子测量、源码快照、原始命令及历史证据；被引用的参考源码须先验证归档副本哈希并补充归档定位，不改写历史命令。
