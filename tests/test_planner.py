from vllm_ascend_split_batch import SplitBatchConfig, plan_dual_pad, precheck_reason


def test_precheck_fails_closed_for_conflicts() -> None:
    config = SplitBatchConfig(enabled=True)
    reason = precheck_reason(
        config,
        num_requests=8,
        graph_mode="full",
        uniform_decode=True,
        has_lora=True,
        is_mla=False,
        is_mrope=False,
        speculative_decode=False,
    )
    assert reason == "lora_conflict"


def test_dual_pad_plan_saves_padding() -> None:
    plan, reason = plan_dual_pad(
        num_requests=10,
        total_tokens=10,
        main_capture_sizes=(8, 16),
        parallel_capture_sizes=(4, 8),
    )
    assert reason == "split"
    assert plan is not None
    assert plan.slices[0].token_stop == 8
    assert plan.slices[1].padded_tokens == 4
    assert plan.padding_saved == 4


def test_exact_graph_hit_does_not_split_by_default() -> None:
    plan, reason = plan_dual_pad(
        num_requests=16,
        total_tokens=16,
        main_capture_sizes=(8, 16),
    )
    assert plan is None
    assert reason == "exact_graph_hit"
