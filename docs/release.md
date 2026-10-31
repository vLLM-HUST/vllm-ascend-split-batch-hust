# 发布流程与检查表

## 发布边界

当前交付的是 `pyproject.toml` 与 `src/` 定义的插件包，不是完整 Split-Batch、双流或 full-graph 服务优化。root planner 为 `import_only`，不可激活。已撤除实现不保留操作步骤。各组件的发现、激活、真实调用、算子收益和服务收益以[交付清单](delivery-inventory.md)为准。

升级、卸载或回滚均须重启 worker 并重新捕获图；删除注册不能撤销已有实例或图。宿主补丁与插件 wheel 是独立交付物，卸载插件不等于撤销宿主补丁。

## 算子 bundle 最短交付路径

1. 核对目标版本的公开注册/dispatch 接口和真实 callsite。打包 kernel 或发现 API 不等于模型命中实现。
2. 提供 default-off 开关、fallback、冲突不覆盖及关闭态无重型导入的测试。新 bundle 保持 `import_only`。
3. 对同一实现、callsite、形状和权重完成 L0、L3+compare、light L4；数值失败单独阻断验收，不以吞吐改善抵消。
4. 发布最终 wheel 后验证发现、关闭、启用、双 rank 真实调用、disable/restart 回原生路径。default-off 冒烟、正确性、性能三项证据齐备才能改 `active`。
5. 服务收益须另有配对端到端结果；不得从局部算子耗时推算。字段与缺项处理见[交付清单](delivery-inventory.md#性能证据最小字段)。

### GDN 注册接入验证门

`tools/probe_gdn_registration.py --output <新JSON路径>` 使用实际 serving Python，先导入 `torch_npu`，核对 npugraph_ex API 并对实际注册的 `qwen_gdn_attention_core` 做 CPU export。不加载模型、不执行 NPU kernel，也不证明生产图中 pattern 已命中。

- 外层图仅含 opaque core 时，必须另证内部子图可达或已有细粒度 dispatch 可用。
- `PluggableLayer.register_oot` 是层级接口，不能当作 unpack/norm 的窄接口；不得覆盖整个 forward 或分解含 KV/通信副作用的 core 绕行。
- 新接入必须覆盖匹配/不匹配、跨流、scope、形状与相位守卫，并重新验证 L3+compare 和 graph replay。

### GDN QKV norm tier-2 交付边界

kernel 和可逆 host proposal 保留，公开注册已撤回。结论与重开条件见 `host-proposals/GDN-qkv-norm/registration-closure.json`。没有已证实可达的窄 dispatch，不得把源码哈希检查当作注册成功。

人工审查的 tier-2 路线要求宿主补丁及 `VLLM_HUST_GDN_QKV_NORM=1`、`USE_DEFAULT_FLA_NORM=0`；默认关闭，不自动修改宿主。关闭后重启并重捕获，补丁单独回滚。历史 kernel 数值验收不转写为当前注册接入通过。

## 0. 状态纪律

- manifest `active` 表示允许显式激活，不代表目标模型/形状一定调用，不改变 default-off。
- `activation.environment` 必须是真实可注入值。`configuration` 不是本仓 env 门控的替代入口。
- `host.version_range` 必须有界，禁止 `>=0`；有序比较符中不得包含 `+local`。
- root planner 的协议范围为 `null`，宿主没有独立版本化的执行协议；这不是可激活许可。

### 0.1 宿主与证据域

| 字段 | 约束 |
|---|---|
| manifest 兼容窗口 | `>=0.25.1rc2.dev125,<0.25.2`，指 vllm-ascend distribution，不是 core 版本 |
| 当前算子基准 | 0.28 镜像：vllm `0.28.1.post1.dev143+gf18cf803c`、vllm-ascend `0.25.1rc2.dev125+hust`、CANN 9.1.0、2×910B2 TP2 |
| 权重与负载边界 | Qwen3.5-35B-A3B bf16、chunked prefill on、APC off、`cudagraph_mode=NONE`；个别 FULL graph 证据必须单列配置，不可混算 |
| 历史域 | Qwen2.5/0.25 只在原报告域成立，不继承到当前组件或宿主 |

逐项主张还须绑定实际形状、权重格式和 callsite；权威约束见 [cohort-and-ownership.md](cohort-and-ownership.md)。

### 0.3 兼容窗口不是逐版本验收

版本串可能不随 fork 代码代际同步。历史端点 dev125 与 dev605 的记录不能替代目标树核对；dev605 `fbe4911bb` 的历史配对要求 core `0aee727ff6`。换宿主必须验证实际源码、依赖和调用点，不能仅依据版本在窗口内宣称正确性或收益。

## 1. 版本与构建

在本仓根目录执行，不以 `/vllm-workspace` 为 CWD：

```bash
git status --short
git rev-parse HEAD
python -m build --outdir <新的仓外产物目录>
python tools/check_distribution.py <产物目录>
python -m twine check <产物目录>/*
```

发行版本与所有 manifest 的 `extension_version` 必须一致。不得覆盖已有 PyPI 版本。wheel/sdist 只含运行时、元数据、README、许可和溯源；不含工作台、搜索、数据库、结果、profile、权重或 host proposal。

## 2. 隔离安装冒烟

```bash
python -m venv <新的仓外venv>
<venv>/bin/pip install --no-deps <wheel路径>
<venv>/bin/python -c 'from importlib.metadata import version; print(version("vllm-ascend-split-batch"))'
```

核对本包贡献的 9 个 general plugins、6 个 bundles 及全部 manifest。安装不需要 opt、知识库或个人脚本。管理器同环境可用时再核对 `extension list`、`inspect`；root planner 必须报告不可激活。

## 3. 启用验证

仅对 manifest 为 `active` 的 bundle 执行 check、status、`run --dry-run`、真实起服和生效验证。`/health` 只证明服务就绪，不证明插件 dispatch。核对启动 PID 归属、环境快照、双 rank 调用标记和 fallback。无设备不得把 CPU 测试写成真实调用通过。

## 4. 宿主升级核对清单

- [ ] 按 [HOST_CONTRACT.md](HOST_CONTRACT.md) 核对当前组件实际消费的 seam。
- [ ] `vllm.general_plugins` 加载顺序、冲突处理及关闭态零修改。
- [ ] fi_gelu：`CustomOp.register_oot` 的 `GeluAndMul` 建键、实例化消费、all/none 与 `+/-gelu_and_mul`、图回放和 fallback。
- [ ] fi_sampling：sampler 构造点和 forward 签名、generator/logprobs/batch-invariant/reduce/async fallback。
- [ ] rope-fix 与 AOT guard：运行 `host_tree` 漂移测试，缺源码树即失败；宿主已修复时不得继续覆盖。
- [ ] tier-2 patch：基线/补丁/产物 SHA256、依赖顺序、完整回滚；漂移拒绝，不模糊套补丁。
- [ ] 实际设备数值、dispatch、graph 与性能，缺一项即保留缺口。

`tools/seam_diff.py --selftest` 及两棵源码树对拍仅证明静态面，不证明生产图与设备行为。输入投影权重合并与卷积权重预打包的暂存树重放见[交付清单](delivery-inventory.md#运行树补丁)，不得直接修改只读宿主树。

## 5. 发布检查表

- [ ] 受影响测试、全量 CPU CI、lint 与 format 通过。
- [ ] 具备宿主源码时运行 `host_tree`；真实 NPU 测试和服务生命周期单列结果。
- [ ] 版本、源码范围、起点/最终提交、dirty 文件与被测字节可追溯。
- [ ] wheel/sdist 清单审计、干净环境安装及发现/关闭态检查通过。
- [ ] 三项证据齐备才允许 `active`；未完成、负结果及失效边界保留。
- [ ] 扩展 CI 三条腿全绿后经 PR 合并，不绕过 main 保护。
- [ ] 发布经授权；tag/workflow 结果、PyPI JSON digest、Release 资产字节逐项核对。

## 7. CPU CI 与设备验证

CPU CI 使用 Python 3.10/3.12/3.14：`ruff check .`、`ruff format --check .`、`pytest -q -m "not host_tree"`、build 和 extension inspect。没有 torch/NPU/宿主树的环境应明确 skip 设备测试，显式 deselect `host_tree`；这不是 NPU 验收。

干净环境须能安装 `.[test]` 并运行核心回归，不得依赖仓外工作台、个人脚本或台账数据库。工作台自身的集成回归不属于插件包验收。

## 8. 未完成的设备与发布验证

清扫的 CPU patch replay、metadata 和包构建不能关闭真实模型 dispatch、L0/L3/L4、服务 OFF/ON 和 disable/uninstall 后重启的缺口。逐组件状态见交付清单；没有新硬件证据时不得扩展旧结论。

## 9. 停止、卸载与回滚

关闭目标组件的 env 或管理器 enable 状态，停止明确属于本次服务的进程，确认退出后重新启动并重捕获图。不承诺热撤销，不以全局 kill 作为交付步骤。

```bash
python -m pip uninstall vllm-ascend-split-batch
```

核对本包 entry points 已消失；管理器配置状态独立核对。不要顺带卸载其他独立包。回退须选定旧 wheel 并审查旧支持范围，安装旧版本不会获得新保护。宿主补丁只能按对应清单和精确哈希回滚。

## 11.10 已发布安装资产

本轮源码改动不改变已发布字节，也不自动成为新发行版本。`0.1.3` 早于实现删除，`0.2.0` 是当前发布基线。旧资产与原始记录只用于安装消费者追溯，不是当前能力表。

| 版本 | 安装索引 | Release |
|---|---|---|
| 0.1.0 | [PyPI](https://pypi.org/project/vllm-ascend-split-batch/0.1.0/) | [v0.1.0](https://github.com/vLLM-HUST/vllm-ascend-split-batch-hust/releases/tag/v0.1.0) |
| 0.1.1 | [PyPI](https://pypi.org/project/vllm-ascend-split-batch/0.1.1/) | [v0.1.1](https://github.com/vLLM-HUST/vllm-ascend-split-batch-hust/releases/tag/v0.1.1) |
| 0.1.2 | [PyPI](https://pypi.org/project/vllm-ascend-split-batch/0.1.2/) | [v0.1.2](https://github.com/vLLM-HUST/vllm-ascend-split-batch-hust/releases/tag/v0.1.2) |
| 0.1.3 | [PyPI](https://pypi.org/project/vllm-ascend-split-batch/0.1.3/) | [v0.1.3](https://github.com/vLLM-HUST/vllm-ascend-split-batch-hust/releases/tag/v0.1.3) |
| 0.2.0 | [PyPI](https://pypi.org/project/vllm-ascend-split-batch/0.2.0/) | [v0.2.0](https://github.com/vLLM-HUST/vllm-ascend-split-batch-hust/releases/tag/v0.2.0) |

[固定版本发布原始记录](https://github.com/vLLM-HUST/vllm-ascend-split-batch-hust/blob/185f48f3ac845833beb41bf2b8d34f7f82388128/docs/release.md)。保留其中的失败和字节差异，不以本机重建代替已发布资产。原始结果保管规则见 [EVIDENCE.md](EVIDENCE.md)。
