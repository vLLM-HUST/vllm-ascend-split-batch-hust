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


def test_real_input_replay_keeps_failure_visible(tmp_path, monkeypatch):
    host = torch.zeros(16, 4096, dtype=torch.bfloat16)
    expected = bench.baseline(host, lambda t: t)
    actual = tuple(t.clone() for t in expected)
    actual[1][0, 0, 0, 0] = 0.25
    path = tmp_path / 'capture.pt'
    torch.save({'input': host, 'expected': expected, 'actual': actual}, path)
    monkeypatch.setattr(bench.candidate, 'unpack_norm', lambda x: actual)
    report = bench.replay_capture(path, lambda t: t, device='cpu')
    assert report['saved_baseline_matches']
    assert report['captured']['k']['different'] == 1
    assert report['replayed']['k']['different'] == 1
    monkeypatch.setattr(bench.candidate, 'unpack_norm', lambda x: expected)
    repaired = bench.replay_capture(path, lambda t: t, device='cpu')
    assert repaired['captured']['k']['different'] == 1
    assert repaired['replayed']['k']['different'] == 0


def test_replay_rejects_reference_drift(tmp_path, monkeypatch):
    host = torch.zeros(16, 4096, dtype=torch.bfloat16)
    expected = bench.baseline(host, lambda t: t)
    path = tmp_path / 'capture.pt'
    torch.save({'input': host, 'expected': expected, 'actual': expected}, path)
    monkeypatch.setattr(bench.candidate, 'unpack_norm', lambda x: expected)
    with pytest.raises(AssertionError):
        bench.replay_capture(path, lambda t: t + 1, device='cpu')
