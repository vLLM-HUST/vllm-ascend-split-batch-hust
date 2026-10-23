# kv-cache planning 模块搬迁兼容层（未通过，阻断在下一处 API 漂移）

- 状态：**unproven / blocked**。不是可用补丁，是"把失败点从导入期推到配置期"的过程证据。
- 适用树：vllm-ascend `35c3e67b1`（`031/plugin-host-compat`，2026-10-19 同步上游 `ee29cc6fc`）
- 适用宿主：vllm-hust `2454e5289`（含 `df6c7ad03` 的 KV-cache planning 重构）
- 领地：`vllm_ascend/` 下 17 个文件的导入/写回目标，加新增模块 `_kv_cache_compat.py`
- 载波：无 env 门。补丁一旦施加即常驻；回滚靠反向 patch

## 解决什么

宿主在 `df6c7ad03`（2026-10-13，`[Core][Refactor] Move KV cache planning logic to kv_cache_planning.py (#61183)`）
把 8 个 KV-cache 规划原语从 `vllm.v1.core.kv_cache_utils` 搬到 `vllm.v1.core.kv_cache_planning`：

| 符号 | 现属模块 |
|---|---|
| `resolve_kv_cache_block_sizes` | `kv_cache_utils`（未搬） |
| `_get_kv_cache_groups_uniform_page_size`、`get_kv_cache_groups`、`_get_packed_kv_cache_groups`、`get_kv_cache_config_from_groups`、`_max_memory_usage_bytes_from_groups`、`_pool_bytes_per_block`、`_warn_if_unannotated_eagle_mamba` | `kv_cache_planning` |

插件按旧布局 `from vllm.v1.core.kv_cache_utils import ...`，导入即 `ImportError`。
朴素改导入还不够：`patch_kv_cache_utils.py` 会把 7 个函数写回宿主模块做 monkey-patch，
写错模块会**静默失效**（宿主不再从那里取），比报错更糟。

本补丁的做法：`_kv_cache_compat.kc` 是一个读写代理，按名字解析"当前拥有它的模块"，
读和写都落到该模块；导入侧同样经代理解析，因此新旧布局都能用。

## 为什么它是 blocked

施加后失败点前进，但没到可用：

| 阶段 | 失败点 | 证据 |
|---|---|---|
| 原始 | `ImportError: cannot import name '_approximate_gcd'` | `p4/blocker-pairing.md` |
| 施加导入层 shim（仅导入） | `AttributeError: ... _get_kv_cache_groups_uniform_page_size`（补丁模块直接读属性） | 同上 |
| 本补丁（导入 + 写回代理） | `AttributeError: 'PassConfig' object has no attribute 'enable_sp'`（`vllm_ascend/platform.py:354`） | `ladder-serve-1023-133336.log` |

第三处与 kv-cache 无关，是另一处独立的宿主 API 漂移。这说明该插件与 `2454e5289`
的失配是**系统性**的，逐点补不可取。

## 复现与回滚

```bash
# 施加（在 vllm-ascend 树上，需干净工作区）
cp _kv_cache_compat.py "$TREE/vllm_ascend/_kv_cache_compat.py"
git -C "$TREE" apply patches/kv-cache-planning-compat.diff
# 回滚
git -C "$TREE" apply -R patches/kv-cache-planning-compat.diff
rm "$TREE/vllm_ascend/_kv_cache_compat.py"
```

sha256：`patches/kv-cache-planning-compat.diff` = `bcc0ca5c3371e73c…`，
`_kv_cache_compat.py` = `2fc8435720f30d30…`。

## 未做

- 无单测：补丁未通过，写测试等于给未验证实现背书。
- 未打包成 wheel：`vllm_ascend` 的 .so 是既有产物，补丁只改 Python 源码；
  真要发布需重打 wheel 并复核 `.so` 哈希不变。
- 未修 `PassConfig.enable_sp` 那一处（独立漂移，另行处理）。

## 重开条件

宿主侧出现"插件与 core 的配对声明"（例如 vllm-ascend 记录目标 vllm 提交），
或插件分支同步到适配该重构的上游版本；届时优先取配对组合，而不是继续加兼容层。
