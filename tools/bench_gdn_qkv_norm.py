# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 vLLM-HUST contributors.

from __future__ import annotations

import argparse
import hashlib
import json
import statistics
import time
from pathlib import Path

import torch

from vllm_ascend_split_batch import gdn_qkv_norm as candidate
from vllm_ascend_split_batch.gdn_qkv_unpack.reference import (
    rearrange_mixed_qkv_reference,
)


def unpack(x):
    return rearrange_mixed_qkv_reference(
        x, q_dim=1024, v_dim=2048, head_k_dim=128, head_v_dim=128,
    )


def baseline(x, norm):
    q, k, v = unpack(x)
    return norm(q), norm(k), v


def check_outputs(actual, expected):
    for i, (a, e) in enumerate(zip(actual, expected, strict=True)):
        assert a.shape == e.shape and a.stride() == e.stride()
        if i == 2:
            assert torch.equal(a.cpu().view(torch.int16), e.cpu().view(torch.int16))
        else:
            torch.testing.assert_close(a, e, rtol=0, atol=0, equal_nan=True)
            assert torch.equal(torch.isnan(a), torch.isnan(e))
    ptrs = [t.untyped_storage().data_ptr() for t in actual]
    assert len(set(ptrs)) == 1
    assert [t.storage_offset() for t in actual] == [0, 16384, 32768]


def timed(fn, iters):
    torch.npu.synchronize()
    begin, end = [torch.npu.Event(enable_timing=True) for _ in range(2)]
    start = time.perf_counter()
    begin.record()
    for _ in range(iters):
        fn()
    end.record()
    enqueue = time.perf_counter()
    torch.npu.synchronize()
    done = time.perf_counter()
    return {
        'host_enqueue_us': (enqueue - start) * 1e6 / iters,
        'event_eager_chain_us': begin.elapsed_time(end) * 1000 / iters,
        'host_total_us': (done - start) * 1e6 / iters,
    }


def graph_check(norm):
    x = torch.randn(16, 4096, dtype=torch.bfloat16, device='npu')
    for _ in range(3):
        candidate.unpack_norm(x)
    torch.npu.synchronize()
    graph = torch.npu.NPUGraph()
    with torch.npu.graph(graph):
        actual = candidate.unpack_norm(x)
    pointers = [t.data_ptr() for t in actual]
    for step in range(8):
        value = torch.randn_like(x) * (0.0 if step == 0 else 10.0 ** (step - 4))
        x.copy_(value)
        graph.replay()
        torch.npu.synchronize()
        check_outputs(actual, baseline(value, norm))
        assert [t.data_ptr() for t in actual] == pointers
    return {'status': 'PASS', 'replays': 8, 'changing_input': True,
            'stable_output_addresses': True, 'scope': 'isolated kernel graph only'}


def replay_capture(path, norm, device='npu'):
    from vllm_ascend_split_batch.gdn_qkv_norm_diagnostic import tensor_difference

    saved = torch.load(path, map_location='cpu', weights_only=True)
    host = saved['input']
    if host.shape != (16, 4096) or host.dtype != torch.bfloat16:
        raise ValueError('replay requires the captured T16 bf16 packed input')
    x = host.to(device)
    actual = candidate.unpack_norm(x)
    expected = baseline(x, norm)
    for a, b in zip(expected, saved['expected'], strict=True):
        torch.testing.assert_close(a.cpu(), b, rtol=0, atol=0, equal_nan=True)
    return {
        'source': str(path),
        'sha256': hashlib.sha256(path.read_bytes()).hexdigest(),
        'saved_baseline_matches': True,
        'captured': {key: tensor_difference(a, b) for key, a, b in zip(
            ('q', 'k', 'v'), saved['actual'], saved['expected'], strict=True)},
        'replayed': {key: tensor_difference(a, b) for key, a, b in zip(
            ('q', 'k', 'v'), actual, expected, strict=True)},
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--out', type=Path, required=True)
    parser.add_argument('--iters', type=int, default=40)
    parser.add_argument('--profile', action='store_true')
    parser.add_argument('--graph-only', action='store_true')
    parser.add_argument('--replay', type=Path)
    args = parser.parse_args()
    if args.replay and (args.profile or args.graph_only):
        parser.error('--replay is correctness-only; no profile/graph-only')
    if args.iters < 1:
        parser.error('--iters must be positive')
    if args.out.exists():
        parser.error('refusing to overwrite evidence')
    report = {
        'schema': 'gdn-qkv-norm-l0.v1', 'status': 'running',
        'candidate': 'gdn-unpack-qk-l2norm-fusion',
        'shape': {'T': 16, 'row_stride': 4096, 'q': 1024, 'k': 1024,
                  'v': 2048, 'head_dim': 128, 'phase': 'pure decode'},
        'weights': {'model': 'Qwen3.5-35B-A3B', 'precision': 'bf16',
                    'format': 'ND', 'tp': 2, 'input': 'seeded synthetic'},
        'shape_source': '/root/包体恢复/records/probes/'
                        '20261018T112829Z_gdn_qkv_unpack_hotspot_recheck.json',
        'callsite': '/vllm-workspace/vllm-ascend/vllm_ascend/ops/gdn.py:'
                    '480,619-620 pure non-spec decode',
        'scope': 'isolated matched consumer-chain; NOT full-model L3/L4',
        'timing_contract': 'host enqueue and event overlap; event includes '
                           'host supply gaps, never subtract them for kernel time',
        'iters': args.iters, 'repeats': 3, 'warmup': 3,
        'correctness': [], 'timings': {}, 'traces': {},
    }
    start = time.monotonic()
    try:
        import torch_npu
        import triton
        from vllm.third_party.flash_linear_attention.ops import l2norm

        assert l2norm.USE_DEFAULT_FLA_NORM == 0
        report['reference'] = {
            'path': l2norm.__file__,
            'sha256': hashlib.sha256(Path(l2norm.__file__).read_bytes()).hexdigest(),
        }
        torch.npu.set_device(0)
        triton.runtime.driver.active.utils.get_device_properties(0)
        torch.manual_seed(0)
        if args.replay:
            report['scope'] = 'real-input differential replay only; no timing'
            report['weights']['input'] = 'captured real model input'
            report['replay'] = replay_capture(args.replay, l2norm.l2norm_fwd)
            different = any(d['different']
                            for d in report['replay']['replayed'].values())
            report['status'] = 'divergence' if different else 'ok'
            if different:
                raise SystemExit(3)
            return
        if args.graph_only:
            report['scope'] = 'graph replay correctness only; no timing'
            report['graph_check'] = graph_check(l2norm.l2norm_fwd)
            report['status'] = 'ok'
            return
        x = torch.randn(16, 4096, dtype=torch.bfloat16).npu()
        for case in ['random', 'zero', 'small', 'special', 'offset']:
            host = torch.randn(16, 4096, dtype=torch.bfloat16)
            if case == 'zero':
                host.zero_()
            if case == 'small':
                host.mul_(1e-8)
            if case == 'special':
                host[0, :128] = 0
                host[1, 0] = float('nan')
                host[2, 1024] = float('inf')
                host[3, 2048:2053] = torch.tensor(
                    [-0.0, 0.0, float('nan'), float('inf'), -float('inf')])
            value = host.npu()
            if case == 'offset':
                storage = torch.empty(16 * 4096 + 32, dtype=host.dtype, device='npu')
                value = storage[32:].view(16, 4096)
                value.copy_(host)
            before = value.cpu().view(torch.int16).clone()
            actual = candidate.unpack_norm(value)
            expected = baseline(value, l2norm.l2norm_fwd)
            check_outputs(actual, expected)
            assert torch.equal(before, value.cpu().view(torch.int16))
            report['correctness'].append({'case': case, 'status': 'PASS'})
        flat = torch.empty(65536, dtype=x.dtype, device=x.device)
        funcs = {
            'baseline': lambda: baseline(x, l2norm.l2norm_fwd),
            'candidate': lambda: candidate.unpack_norm(x),
            'candidate_launch_only': lambda: candidate.launch_into(x, flat),
        }
        for fn in funcs.values():
            for _ in range(3):
                fn()
        torch.npu.synchronize()
        report['timings'] = {name: [] for name in funcs}
        for repeat in range(3):
            order = list(funcs) if repeat % 2 == 0 else list(reversed(funcs))
            for name in order:
                report['timings'][name].append(timed(funcs[name], args.iters))
        report['medians'] = {
            name: {key: statistics.median(r[key] for r in runs) for key in runs[0]}
            for name, runs in report['timings'].items()
        }
        if args.profile:
            for name in ['baseline', 'candidate']:
                path = args.out.with_suffix('') / name
                handler = torch_npu.profiler.tensorboard_trace_handler(
                    str(path), analyse_flag=False)
                with torch_npu.profiler.profile(
                    activities=[torch_npu.profiler.ProfilerActivity.CPU,
                                torch_npu.profiler.ProfilerActivity.NPU],
                    on_trace_ready=handler, record_shapes=True,
                ):
                    for _ in range(5):
                        with torch.profiler.record_function('gdn_chain_' + name):
                            funcs[name]()
                    torch.npu.synchronize()
                report['traces'][name] = {'path': str(path), 'calls': 5,
                                         'analysis': 'separate offline process'}
        report['status'] = 'ok'
    except Exception as exc:
        report['status'] = 'failed'
        report['error'] = repr(exc)
        raise
    finally:
        report['wall_s'] = time.monotonic() - start
        with args.out.open('x') as f:
            json.dump(report, f, indent=2, ensure_ascii=False)
            f.write('\n')
        print(json.dumps(report, indent=2, ensure_ascii=False), flush=True)


if __name__ == '__main__':
    main()
