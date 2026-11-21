"""The fixed GMM1 screen cannot join eager, mixed-rank or wrong-shape operators."""

import sys
from decimal import Decimal
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from tools.analysis import gmm1_swiglu_screen as screen  # noqa: E402


def chain():
    defs = [
        ("MoeInitRoutingV3", "16,2048;16,8", "128,2048;128;128;128"),
        (screen.GMM, screen.GATE_UP, screen.MID),
        ("SwiGlu", screen.MID, screen.ACT),
        (screen.GMM, screen.DOWN, "128,2048"),
    ]
    return [
        {
            "Type": op,
            "Input Shapes": x,
            "Output Shapes": y,
            "Input Data Types": "DT_BF16;DT_BF16;INT64"
            if op == screen.GMM
            else "DT_BF16",
            "Output Data Types": "DT_BF16",
            "Start Time(us)": str(i * 5),
            "Duration(us)": "2",
            "Stream ID": "12",
            "Task ID": str(i),
            "Model ID": "49",
            "Device_id": "0",
        }
        for i, (op, x, y) in enumerate(defs)
    ]


def test_fixed_chain_budget():
    facts = screen.pairs(chain(), "0", Decimal(100))
    assert facts["matched"] == 1 and facts["gmm1_stage_pct"] == 2
    assert facts["swiglu_stage_pct"] == 2 and facts["gap_median_us_not_attributed"] == 3
    assert facts["nominal_intermediate_write_plus_read_bytes_per_call"] == 524288
    assert facts["gmm1_share_all_groupedmatmul_us_pct"] == 50


@pytest.mark.parametrize(
    "key,value",
    [
        ("Type", "Mul"),
        ("Input Shapes", "16,512"),
        ("Model ID", "4294967295"),
        ("Stream ID", "13"),
        ("Device_id", "1"),
        ("Task ID", "15"),
        ("Input Data Types", "INT8"),
    ],
)
def test_false_adjacent_match_fails(key, value):
    rows = chain()
    rows[2][key] = value
    with pytest.raises(ValueError):
        screen.pairs(rows, "0", Decimal(100))


def test_screen_preserves_unmeasured_status():
    record = screen.load(screen.RECORD)
    assert record["expert_distribution"]["values"] is None
    assert record["performance_verdict"] == "not_measured_no_candidate"
    for ranks in record["observations"]["per_round_rank"].values():
        assert set(ranks) == {"0", "1"}
        for obs in ranks.values():
            assert obs["matched"] == 600
