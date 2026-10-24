# ConcatD 零拷贝解包 — decode T=1 视图路径（runtime-tree 快照）
- 路由 028/D4-const-hoist | env VLLM_HUST_GDN_UNPACK_VIEWS=1 default-off
- 回滚: 恢复链 = 先回 D4 site2 patch 前的 orig（patches/*.orig），或整体覆盖 patches/*.orig + 逐段反向
- 守卫: `tests/test_concatd_views_contiguous.py`、`tests/test_runtime_source_patches.py`。
- 切片输出须连续，满足下游 `l2norm_fwd` 的 `view(-1, head_dim)`；多行切片允许拷贝，不承诺零拷贝。
- 连续化增量: `patches/20261024_contiguous.gdn.py.patch`；在已含 views 的树上 `patch -p1 < <patch>`，用 `patch -R -p1 < <patch>` 回滚。部署前保留该树的 `.orig`，不覆盖其它层。
- 持久化入口: `src/vllm_ascend_split_batch/runtime_source_patches.py`；树内已有载体时由树内实现执行，stock 树由锚点变换生成。两条路径的 helper 必须一致。持久化调用点仍限定连续 T=1，已有树内调用点允许连续多行；本补丁不扩大调用范围。
- L0 适配器: `tools/bench_concatd_views_adapter.py:unpack`，从持久化变换提取同一 helper；调用现有 `opt bench l0 GdnQkvUnpack --candidate tools/bench_concatd_views_adapter.py:unpack --candidate-arg-style packed --tokens 1 --row-strides 4096`，并显式设置 `PYTHONPATH=$PWD/src`。该微基准仅覆盖单行，不证明多行性能或端到端收益。
- 相邻已拒: gdn-qkv-unpack-single-pass（单拷贝）——本候选为零拷贝不同机制, 用户 2026-10-21 授权重开
