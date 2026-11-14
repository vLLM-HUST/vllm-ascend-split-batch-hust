# 运行与交付边界

## 工作区与宿主

- 不要以 `/vllm-workspace` 为 CWD 运行 Python 或 vLLM 命令；该目录可能遮蔽已安装的 `vllm` 包。
- 宿主源码目录只读。运行树补丁仅能从本仓 manifest 和锁定的 patch 原件进行检查或在独立暂存树 replay；不得从个人目录补齐缺失文件或直接修改宿主树。
- `host.version_range` 只是兼容声明，不等于窗口内每个 build 已验证。升级宿主前核对实际 seam、签名、调用点和依赖；版本或哈希漂移时 fail closed。
- `import_only` bundle 不可被管理器激活。root planner 仅生成规划描述，不代表宿主执行、双流调度或 graph replay 已接通。

## 插件与证据

- `active` / 可发现不表示每个请求真实命中。需分别保留安装发现、激活配置、目标 callsite 调用和数值证据；无 marker 或观测条件不足应记为 unknown，不推断成功或失败。
- vLLM 对未知 `VLLM_HUST_*` 环境变量的警告不是插件缺席判据。按 serving interpreter 枚举 entry point，并从该服务自身的日志/环境快照核对；调用 shell 的环境不能代替服务环境。
- `gdn-merge-inproj` 插件当前无安全消费 seam，保持 fail-closed、不合并权重。输入投影权重合并与卷积权重预打包属于宿主 Fork 的源码修改，不得把其调用或收益归因给插件；补丁 CPU replay 不代表已部署或经过设备验证。
- 配置被 fail-closed 守卫拒绝后回退原生路径，不构成优化启用、正确性通过或收益证据。
- 数值对照固定输入、请求顺序、形状与权重。HTTP 并发适合评估服务统计，不替代固定批数值判据；关闭态差异、失败和噪声都保留。
- 算子 L0/L3/L4 与端到端服务结果分开。局部算子耗时不能推算服务收益；不同模型、权重、形状或 cohort 只比较方向，不混算数字。

## 打包与变更

- manifest `activation.environment` 填实际激活时注入的值，不写说明文本；新能力默认关闭，测试关闭态不导入重型实现。
- entry point、bundle manifest、`extension_version` 和包版本必须一致。源码模块、测试、宿主补丁和工具不等同于公开插件入口。
- 编译型 `.so` kernel 由仓外独立 wheel 交付；本仓不构建 CCE/Ascend C/CATL kernel。Triton/纯 Python 算子实现留仓且须 lazy import。
- 本仓 CPU 测试、patch replay、包构建不能证明 NPU 数值、真实 dispatch、图回放或 serving 收益。对应证据尚缺时明确标记未证明，不从历史结果外推。
- 历史探针原件（如 [multibuffer-graph.json](https://github.com/vLLM-HUST/vllm-hust-benchmark/blob/4bf3c6b7af6ddb4cdcf193e348302758386f8a2f/archive/mod-git-20261022/split-current-git/snapshot/host-proposals/GDN-qkv-norm/validation/multibuffer-determinism-20261018/multibuffer-graph.json)）已通过 SHA256 固化审计；两份 separate-graph 崩溃日志仍未找回，不得把其调用或收益归因给插件。

历史运行记录仅作原始证据和溯源，不作为当前操作手册；证据位置与散列规则见 [EVIDENCE.md](EVIDENCE.md)。

## 机械化守卫（改这些之前先看判据）

下列约束已落成本地 `pre-commit` 钩子与 CI 判据，命中即拦。判据本体与被守卫的文档句一一对应，
写在对应测试模块的注释里，改判据前先改文档。

| 约束 | 守卫 |
|---|---|
| manifest 的宿主范围两端有界、不带本地后缀，`activation.environment` 是真实注入值 | `tests/test_pitfall_guards.py` |
| 根 bundle 保持 `import_only`，不注入环境变量 | 同上 |
| manifest `extension_version` 与包版本一致 | 同上 |
| sdist 只含运行时、元数据、README、许可与溯源 | 同上 |
| 仓内不出现原始数据后缀文件、超大文件、缓存垃圾 | 同上 |
| 对外行文不写死宿主工作区绝对路径 | 同上 |
| `plugin_carrier` 已公开的门控变量在 `src/` 真实存在；未接线的写 `unexposed`，不标记已集成 | 同上 |
| 索引标签必须被证据支撑（服务结论、L3、L4、取代指针、方向） | `tests/test_delivery_status_consistency.py` |
| 算子目录必备文件、证据指纹未被改写、索引覆盖全部算子目录 | `tests/test_ops_delivery_policy.py`、`tests/test_json_hygiene.py` |
| 文档数量陈述、术语锚点真实性、行文关键词与禁语 | `tests/test_doc_consistency.py` |

## 仍靠人工

以下几条无法从文件内容机械判定，保留为人工核对项：

- 不以 `/vllm-workspace` 为 CWD 运行命令——命令怎么跑不留在仓库里；
- 未知 `VLLM_HUST_*` 警告的含义——必须看服务自身日志与环境快照；
- 宿主树只读、历史证据不得重采、失败记录不得删除——属操作纪律，机械判定会误伤正常操作；
- 从局部算子耗时推算服务收益——层级关系由索引状态门禁覆盖一部分，其余要靠评审；
- 关闭态零差异、关闭态不导入重型实现——由 `tests/test_extension_config_boundary.py` 与
  `tests/test_runtime_patch_manifest_guard.py` 部分覆盖，新增能力仍需逐项核对。

