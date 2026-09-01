"""Split-batch planning contracts and inert runtime metadata."""

from .planner import (
    DualPadPlan,
    SplitBatchConfig,
    SplitSlice,
    plan_dual_pad,
    precheck_reason,
)


class VllmAscendSplitBatchContractProposal:
    """Metadata-only proposal; this class performs no runtime activation."""


__all__ = [
    "DualPadPlan",
    "SplitBatchConfig",
    "SplitSlice",
    "VllmAscendSplitBatchContractProposal",
    "plan_dual_pad",
    "precheck_reason",
]
