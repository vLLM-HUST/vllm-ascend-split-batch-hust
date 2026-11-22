"""Single-device diagnostic, not a serving route. Run from a fresh external archive."""

import argparse
import hashlib
import importlib.metadata
import json
import os
import platform
import runpy
import sys
from pathlib import Path
from statistics import median, quantiles

EXPECTED_GDN = "a32bf3d9ac16d4d89c87af1182475ca9142e755d78a38ef715bfaebb6d790780"


def save(name, data):
    Path(name).write_text(json.dumps(data, indent=2) + "\n")


def fingerprint(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repository", required=True, type=Path)
    parser.add_argument(
        "--trace-only", action="store_true", help="profile graph replays; never run L0"
    )
    args = parser.parse_args()
    if not __debug__ or Path.cwd().is_relative_to(args.repository.resolve()):
        raise SystemExit("run with assertions enabled from a fresh external archive")
    if any(
        Path(name).exists() for name in ("identity.json", "l3-cases.json", "exit-code")
    ):
        raise SystemExit("use a new archive for each attempt")
    record = json.loads((args.repository / "docs/optimization-next.json").read_bytes())
    runtime = runpy.run_path(
        str(args.repository / "tools/operator_tools/host_runtime.py")
    )
    hosts = runtime["check_host_runtime"](record["host_commits"])
    import torch
    import torch_npu
    from vllm_ascend import utils
    from vllm_ascend.ops import gdn

    assert utils.enable_custom_op(), "host recurrent custom op not enabled"
    src = Path(gdn.__file__)
    assert src.resolve().is_relative_to(
        Path(hosts["vllm-ascend"]["module_origin"]).resolve().parent
    )
    assert fingerprint(src) == EXPECTED_GDN
    op = torch.ops._C_ascend.npu_recurrent_gated_delta_rule
    ext = sys.modules["vllm_ascend.vllm_ascend_C"].__file__
    torch.npu.set_device(0)
    torch.manual_seed(42)
    save(
        "identity.json",
        {
            "hosts": hosts,
            "adapter": str(src),
            "adapter_sha256": fingerprint(src),
            "extension": ext,
            "extension_sha256": fingerprint(ext),
            "op": str(op.default._schema),
            "python": sys.executable,
            "platform": platform.platform(),
            "packages": {
                k: importlib.metadata.version(k) for k in ["torch", "torch-npu"]
            },
            "cann": os.getenv("ASCEND_HOME_PATH"),
            "device": torch.npu.get_device_name(0),
            "torch_npu": torch_npu.__file__,
        },
    )

    def data(n):
        def rand(*shape):
            return torch.randn(shape).bfloat16().npu()

        return (
            rand(1, n, 8, 128),
            rand(1, n, 8, 128),
            rand(n, 16, 128),
            -torch.rand(n, 16).npu(),
            torch.rand(n, 16).bfloat16().npu(),
            torch.tensor([0] + [1] * n, dtype=torch.int32, device="npu"),
            torch.tensor([5, 2][:n], dtype=torch.int32, device="npu"),
        )

    def call(split, inputs, state):
        q, k, v, g, beta, lengths, slots = inputs
        n = v.shape[0]
        q, k = gdn.l2norm_fwd(q).squeeze(0), gdn.l2norm_fwd(k).squeeze(0)
        if split:
            assert n in (1, 2) and state.is_contiguous() and v.is_contiguous()
            v = v.view(n, 32, 64)
            view = state.view(-1, 32, 64, 128)
            assert view.data_ptr() == state.data_ptr()
            state = view
            g, beta = g.repeat_interleave(2, dim=1), beta.repeat_interleave(2, dim=1)
        return op(
            query=q,
            key=k,
            value=v,
            state=state,
            g=g,
            beta=beta,
            scale=128**-0.5,
            actual_seq_lengths=lengths,
            ssm_state_indices=slots,
        ).view(n, 16, 128)

    def equal(a, b, label):
        a, b = a.cpu().contiguous(), b.cpu().contiguous()
        assert (
            a.dtype == b.dtype
            and a.shape == b.shape
            and torch.isfinite(a).all()
            and torch.isfinite(b).all()
        ), label
        if not torch.equal(a.view(torch.uint8), b.view(torch.uint8)):
            raise ValueError(
                f"{label}: max diff={(a.float() - b.float()).abs().max().item()}"
            )

    def capture(split, inputs, pool):
        graph = torch.npu.NPUGraph()
        with torch.npu.graph(graph):
            out = call(split, inputs, pool)
        return graph, out

    def l3(n):
        inputs = data(n)
        initial = (torch.randn(8, 16, 128, 128) * 0.0625).npu()
        a, b = initial.clone(), initial.clone()
        assert len({a.data_ptr(), b.data_ptr(), initial.data_ptr()}) == 3
        outs = [call(False, inputs, a), call(True, inputs, b)]
        equal(*outs, "eager output")
        equal(a, b, "eager state")
        ga, oa = capture(False, inputs, a)
        gb, ob = capture(True, inputs, b)
        a.copy_(initial)
        b.copy_(initial)
        ref = initial.clone()
        active = set()
        for step, ids in enumerate(([5, 2][:n], [5, 2][:n], [3, 6][:n])):
            inputs[-1].copy_(torch.tensor(ids, dtype=torch.int32))
            active.update(ids)
            old = a.cpu()
            ga.replay()
            gb.replay()
            torch.npu.synchronize()
            gold = call(False, inputs, ref)
            equal(oa, ob, "graph output")
            equal(a, b, "graph state")
            equal(ob, gold, "graph vs eager output")
            equal(b, ref, "graph vs eager state")
            for slot in ids:
                assert not torch.equal(old[slot], a[slot].cpu()), (step, slot)
            other = [i for i in range(8) if i not in active]
            equal(a.cpu()[other], initial.cpu()[other], "untouched state")
        return {
            "rows": n,
            "slots": 8,
            "eager_and_graph_bitwise": True,
            "replays": 3,
            "inputs": [
                {
                    "shape": list(x.shape),
                    "dtype": str(x.dtype),
                    "stride": list(x.stride()),
                    "offset": x.storage_offset(),
                }
                for x in inputs
            ],
        }

    def bench(n):
        inputs = data(n)
        pool = torch.zeros((846, 16, 128, 128), device="npu")
        seed = (torch.randn(n, 16, 128, 128) * 0.0625).npu()
        graphs = []
        for split in (False, True):
            call(split, inputs, pool)
            graphs.append(capture(split, inputs, pool))
        ids = [5, 2][:n]

        def sample(arm):
            for i, slot in enumerate(ids):
                pool[slot].copy_(seed[i])
            start, end = (torch.npu.Event(enable_timing=True) for _ in range(2))
            start.record()
            graphs[arm][0].replay()
            end.record()
            end.synchronize()
            return start.elapsed_time(end) * 1000

        timings = {"off": [], "on": []}
        for i in range(110):
            for arm in (0, 1) if i % 2 == 0 else (1, 0):
                us = sample(arm)
                if i >= 10:
                    timings[("off", "on")[arm]].append(us)
        save(f"samples-{n}.json", timings)
        return {
            key: {
                "median_us": median(xs),
                "p90_us": quantiles(xs, n=100, method="inclusive")[89],
                "p99_us": quantiles(xs, n=100, method="inclusive")[98],
            }
            for key, xs in timings.items()
        }

    def trace(n):
        prof = torch_npu.profiler
        inputs = data(n)
        pool = torch.zeros((846, 16, 128, 128), device="npu")
        seed = (torch.randn(n, 16, 128, 128) * 0.0625).npu()
        ids = [5, 2][:n]
        finals = []
        for split in (False, True) if n == 1 else (True, False):
            arm = "on" if split else "off"
            tag = f"rows-{n}-{arm}"
            call(split, inputs, pool)
            graph, out = capture(split, inputs, pool)
            for _ in range(10):
                graph.replay()
            for i, slot in enumerate(ids):
                pool[slot].copy_(seed[i])
            torch.npu.synchronize()
            print(f"trace {tag}: 16 replays", flush=True)
            with prof.profile(
                activities=[prof.ProfilerActivity.CPU, prof.ProfilerActivity.NPU],
                record_shapes=True,
                on_trace_ready=prof.tensorboard_trace_handler(str(Path(tag).resolve())),
            ):
                for _ in range(16):
                    graph.replay()
                    torch.npu.synchronize()
            finals.append((out.cpu(), pool[ids].cpu()))
        for x, y in zip(*finals, strict=True):
            equal(x, y, "after traced replays")

    with torch.inference_mode():
        if args.trace_only:
            for n in (1, 2):
                trace(n)
            return
        cases = []
        for n in (1, 2):
            print(f"L3 rows={n}", flush=True)
            cases.append(l3(n))
            save("l3-cases.json", cases)
        for n in (1, 2):
            print(f"L0 rows={n}", flush=True)
            save(f"l0-{n}.json", bench(n))


if __name__ == "__main__":
    main()
