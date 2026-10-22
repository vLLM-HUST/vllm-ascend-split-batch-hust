# 方向分支证据批次（custody 登记）

分支 `0.28/comm-overlap-free-attribution` 的窗口证据与交付语料（365 + 66 + 18 件（三批）：
算子 A/B 与 L0 地板 JSON、L3 捕获、opstat 副本、窗口回执、hazard 源码 vintage、
host-proposals 载体包（patches/orig/OWNERSHIP）、撤回套组含配对取证
(`host-proposals/GDN-qkv-norm/validation/l4-multibuffer-20261019`)、
comm-mechanism/trace 验证运行、OPT-truth 探针）已按 custody 政策冻结于：

- 批次分支: `direction-evidence-20261022` @ commit `02add2a27be84b106dd5e8c589b0a5bc3f928bff`
- 快照: `archive/dev-hub-20261022/split-batch-direction/snapshot/`（镜像原始相对路径）
- 内层 manifest: `archive/dev-hub-20261022/split-batch-direction/manifest.json`
  （schema mod-git-custody/v1，449 项，逐文件 bytes/sha256/git_blob，
  source_commit `03b0fa51627b6f0d580abc991421efdc5070439b`）
- 原始取件 URL 形态:
  `https://raw.githubusercontent.com/vLLM-HUST/vllm-hust-benchmark/02add2a27be84b106dd5e8c589b0a5bc3f928bff/archive/dev-hub-20261022/split-batch-direction/snapshot/<原始路径>`

登记约定（C 政策）：活跃窗口证据留方向分支；**窗口收尾批量追加到归档分支并在本文件登记**；
合入 main 前清零。GDN-qkv-norm 撤回套组的恒量（配对 forensics sha256
`13212eccec831dbb7ba8c0cc2dafb43967118a890f5b02f01e35b6e534f06ce3`；rank_pair
all_comm 2624 / allreduce 2592）由 tests/test_attribute_comm_free.py 内嵌守护。
