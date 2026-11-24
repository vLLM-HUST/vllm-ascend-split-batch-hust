# 实验索引

**下一步做什么：** [可勾选计划](实验计划.md) · [当前候选与停止条件](docs/optimization-next.json)

**哪些结论能用：** [交付索引](DELIVERY_INDEX.json) · [验收标准](docs/operator-benchmark-standard.md) · [资料可用性](EVIDENCE_AVAILABILITY.json) · [规范导航](docs/README.md)

按主题名在本页查找即可。`history` 是研发记录，`op` 是交付证据，`proposal` 是宿主提案；`patch`、`legacy` 和 `archive` 不代表当前生效。

按关键词查找：`python -m tools.operator_tools.experiment_index --find gdn`。

以下为自动生成的高度平衡搜索树。键小的进左子树，大的进右子树；缩进表示子节点，不表示技术依赖。
只改源记录，随后运行 `python -m tools.operator_tools.experiment_index --write`。不带参数只检查，不写文件。

- [`op/ffn_activation`](ops/ffn_activation/evidence.json)
  - [`history/gdn-trace-recovery`](docs/optimization-history/gdn-trace-recovery.json)
    - [`archive/rope-fix-smoke-20260912`](docs/evidence/rope-fix-smoke-20260912)
      - [`archive/fi-sampling-s3`](docs/evidence/fi-sampling-s3)
        - [`archive/fi-sampling-frozen-20260909`](docs/evidence/fi-sampling-frozen-20260909)
          - [`archive/cascade`](docs/evidence/cascade)
        - [`archive/rope-fix`](docs/evidence/rope-fix)
      - [`history/gdn-chunk-prep`](docs/optimization-history/gdn-chunk-prep.json)
        - [`archive/zerocost-activation-20260913`](docs/evidence/zerocost-activation-20260913)
          - [`archive/w2b-fi-sampling`](docs/evidence/w2b-fi-sampling)
        - [`history/gdn-recurrent-vrows`](docs/optimization-history/gdn-recurrent-vrows.json)
    - [`legacy/ascend-pr-281`](provenance/legacy-patches/ascend-pr-281)
      - [`history/scheduler-padding-screen`](docs/optimization-history/scheduler-padding-screen.json)
        - [`history/gmm1-swiglu-screen`](docs/optimization-history/gmm1-swiglu-screen.json)
          - [`history/gmm-route-reconciliation`](docs/optimization-history/gmm-route-reconciliation.json)
        - [`legacy/ascend-pr-280`](provenance/legacy-patches/ascend-pr-280)
      - [`legacy/core-pr-263`](provenance/legacy-patches/core-pr-263)
        - [`legacy/ascend-pr-283`](provenance/legacy-patches/ascend-pr-283)
          - [`legacy/ascend-pr-282`](provenance/legacy-patches/ascend-pr-282)
        - [`legacy/core-pr-273`](provenance/legacy-patches/core-pr-273)
  - [`op/moe_epilogue`](ops/moe_epilogue/evidence.json)
    - [`op/gdn_output_norm_fusion`](ops/gdn_output_norm_fusion/evidence.json)
      - [`op/gdn_merge_inproj`](ops/gdn_merge_inproj/evidence.json)
        - [`op/gdn_conv_pack`](ops/gdn_conv_pack/evidence.json)
          - [`op/gate_fp32`](ops/gate_fp32/evidence.json)
        - [`op/gdn_norm_gate`](ops/gdn_norm_gate/evidence.json)
      - [`op/gdn_split_fusion`](ops/gdn_split_fusion/evidence.json)
        - [`op/gdn_split_ba`](ops/gdn_split_ba/evidence.json)
          - [`op/gdn_qkv_norm`](ops/gdn_qkv_norm/evidence.json)
        - [`op/moe_down_residual_norm`](ops/moe_down_residual_norm/evidence.json)
    - [`plan/native-route-probe`](experiments/native-route-probe/EXPERIMENT_PLAN.md)
      - [`patch/gdn-chunk-prep/candidate`](experiments/gdn-chunk-prep/candidate.patch)
        - [`op/moe_gating_routing`](ops/moe_gating_routing/evidence.json)
          - [`op/moe_gate_merge`](ops/moe_gate_merge/evidence.json)
        - [`plan/ascend910b-2card`](experiments/ascend910b-2card/EXPERIMENT_PLAN.md)
      - [`proposal/W1R-gmm-tuning-config`](host-proposals/W1R-gmm-tuning-config/proposal_meta.json)
        - [`proposal/GDN-qkv-norm`](host-proposals/GDN-qkv-norm/proposal_meta.json)
        - [`proposal/comm-trace-stop-gil`](host-proposals/comm-trace-stop-gil/proposal_meta.json)
