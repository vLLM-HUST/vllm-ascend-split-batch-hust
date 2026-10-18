# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 vLLM-HUST contributors.

import importlib.util
from pathlib import Path

import pytest

torch = pytest.importorskip('torch')

PATH = Path(__file__).resolve().parents[1] / 'tools/bench_gdn_qkv_norm.py'
spec = importlib.util.spec_from_file_location('bench_gdn_norm', PATH)
bench = importlib.util.module_from_spec(spec)
spec.loader.exec_module(bench)


def outputs():
    flat = torch.zeros(65536, dtype=torch.bfloat16)
    return bench.candidate.output_views(flat)


def test_output_gate_accepts_exact():
    bench.check_outputs(outputs(), outputs())


@pytest.mark.parametrize('error', ['q', 'v_signed_zero', 'nan', 'alias', 'length'])
def test_output_gate_rejects_regressions(error):
    actual, expected = outputs(), outputs()
    if error == 'q':
        actual[0][0, 0, 0, 0] = 1
    elif error == 'v_signed_zero':
        actual[2][0, 0, 0, 0] = -0.0
    elif error == 'nan':
        actual[0][0, 0, 0, 0] = float('nan')
    elif error == 'alias':
        actual = tuple(x.clone() for x in actual)
    elif error == 'length':
        actual = actual[:2]
    with pytest.raises((AssertionError, ValueError)):
        bench.check_outputs(actual, expected)
