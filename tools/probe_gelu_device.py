from __future__ import annotations

import argparse
import json
import os
import statistics
from importlib.metadata import version
from pathlib import Path

PRECISION = {
    "rtol": 2.0**-6,
    "atol": 2.0**-6,
    "required_matched_ratio": 0.99,
    "max_abs_error_limit": max(1.0, 32 * 2.0**-7),
}
SHAPES = ((64, 2048), (8, 27648))
WARMUP = 20
TIMED = 100
REPEATS = 3


def mixed_tolerance(actual, golden) -> dict:
    import numpy as np

    a = actual.detach().float().cpu().numpy().astype(np.float64)
    g = golden.detach().float().cpu().numpy().astype(np.float64)
    assert a.shape == g.shape
    abs_err = np.abs(a - g)
    passed = abs_err <= PRECISION["atol"] + PRECISION["rtol"] * np.abs(g)
    total = a.size
    matched = 1.0 if total == 0 else float(passed.sum()) / total
    worst = 0.0 if total == 0 else float(abs_err.max())
    return {
        "matched_ratio": matched,
        "max_abs_error": worst,
        "is_pass": (
            matched >= PRECISION["required_matched_ratio"]
            and worst <= PRECISION["max_abs_error_limit"]
        ),
    }


def summarize(seconds: list[float]) -> dict:
    per_iter_ms = [s / TIMED * 1e3 for s in seconds]
    return {
        "median_ms_per_iter": statistics.median(per_iter_ms),
        "min_ms_per_iter": min(per_iter_ms),
        "repeats": len(per_iter_ms),
    }


def _versions() -> dict:
    names = (
        "torch",
        "torch-npu",
        "vllm",
        "vllm-ascend",
        "triton",
        "vllm-ascend-split-batch",
    )
    return {name: version(name) for name in names}


def _timed(fn, x) -> list[float]:
    import time

    import torch

    for _ in range(WARMUP):
        fn(x)
    torch.npu.synchronize()
    samples = []
    for _ in range(REPEATS):
        start = time.perf_counter()
        for _ in range(TIMED):
            fn(x)
        torch.npu.synchronize()
        samples.append(time.perf_counter() - start)
    return samples


def probe() -> dict:
    import torch
    import torch.nn.functional as F
    import torch_npu  # noqa: F401
    import vllm.model_executor.layers.activation as act_mod
    from vllm.config import VllmConfig, set_current_vllm_config
    from vllm.model_executor.custom_op import op_registry_oot

    from vllm_ascend_split_batch import fi_gelu_plugin as plugin
    from vllm_ascend_split_batch.fi_gelu import api, reference

    result: dict = {
        "schema": "gelu-device-gate.v1",
        "shape_facts": [
            {"M": m, "K": 2 * n, "inter": n, "phase": "standalone-op"}
            for m, n in SHAPES
        ],
        "weight_labels": (
            "activation microbench: no model weights, bf16, no weight format"
        ),
        "precision_gate": PRECISION,
        "iterations": {"warmup": WARMUP, "timed": TIMED, "repeats": REPEATS},
    }
    previous_env = os.environ.get(plugin.ENV_ENABLE)
    os.environ.pop(plugin.ENV_ENABLE, None)
    try:
        torch.npu.set_device(0)
        device_name = torch.npu.get_device_name(0)
        result["device"] = {"count": torch.npu.device_count(), "name": device_name}
        result["versions"] = _versions()
        cls = act_mod.GeluAndMul
        x = torch.randn(64, 2048, dtype=torch.float32).bfloat16().npu()
        try:
            api.gelu_and_mul(x)
        except Exception as exc:
            result["status"] = "KERNEL_UNAVAILABLE"
            result["error"] = repr(exc)
            result["release_accepted"] = False
            return result

        plugin._reset_for_tests()
        result["default_off"] = {
            "load_returns": plugin.load(),
            "key_absent": "GeluAndMul" not in op_registry_oot,
        }
        os.environ[plugin.ENV_ENABLE] = "1"
        host_attrs = dict(cls.__dict__)
        with set_current_vllm_config(VllmConfig()):
            off_instance = cls(approximate="none")
            assert plugin.load() is True
            provider = plugin._registered_cls
            op = cls(approximate="none")
            result["registration"] = {
                "provider_is_host_subclass": issubclass(provider, cls),
                "host_class_unchanged": cls.__dict__ == host_attrs,
                "provider_name_restored": provider.name == cls.name == "gelu_and_mul",
                "new_instances_select_provider": type(op) is provider,
                "pre_registration_instances_stay_host": type(off_instance) is cls,
                "dispatch_bound_to_adapter": (
                    op._forward_method.__func__ is provider.forward_oot
                ),
            }
            numeric = []
            for m, n in SHAPES:
                shape_x = torch.randn(m, 2 * n, dtype=torch.float32).bfloat16().npu()
                out = op(shape_x)
                direct = api.gelu_and_mul(shape_x)
                golden = reference.gelu_and_mul_reference(
                    shape_x.cpu(), dtype=torch.float32
                )
                check = mixed_tolerance(out, golden)
                check.update(
                    {
                        "M": m,
                        "inter": n,
                        "bit_exact_vs_direct_kernel": bool(torch.equal(out, direct)),
                    }
                )
                numeric.append(check)
            result["op_numeric"] = numeric

            x_fp32 = torch.randn(4, 128, dtype=torch.float32).npu()
            d = x_fp32.shape[-1] // 2
            fp32_out = op(x_fp32)
            fp32_expected = F.gelu(x_fp32[..., :d]) * x_fp32[..., d:]
            tanh_x = torch.randn(8, 512, dtype=torch.float32).bfloat16().npu()
            d_tanh = tanh_x.shape[-1] // 2
            tanh_out = cls(approximate="tanh")(tanh_x)
            tanh_expected = (
                F.gelu(tanh_x[..., :d_tanh], approximate="tanh") * tanh_x[..., d_tanh:]
            )
            result["fallback"] = {
                "fp32_delegates": bool(torch.equal(fp32_out, fp32_expected)),
                "tanh_delegates": bool(torch.equal(tanh_out, tanh_expected)),
            }
            for _ in range(WARMUP):
                op(x)
            torch.npu.synchronize()
            handle = torch.npu.NPUGraph()
            with torch.npu.graph(handle):
                graph_out = op(x)
            for _ in range(5):
                handle.replay()
            torch.npu.synchronize()
            replay = graph_out.clone()
            eager = op(x).clone()
            torch.npu.synchronize()
            result["graph_replay"] = {
                "bit_exact": bool(torch.equal(replay, eager)),
            }
            timings = {}
            for m, n in SHAPES:
                shape_x = torch.randn(m, 2 * n, dtype=torch.float32).bfloat16().npu()
                on = summarize(_timed(op, shape_x))
                host_op = off_instance
                off = summarize(_timed(host_op, shape_x))
                timings[f"M{m}_inter{n}"] = {
                    "provider": on,
                    "host": off,
                    "delta_provider_minus_host_ms": (
                        on["median_ms_per_iter"] - off["median_ms_per_iter"]
                    ),
                }
            result["l0_microbench"] = timings
            result["l0_interpretation"] = (
                "op-level directional microbench only; NOT an L4 operator claim"
            )
            assert plugin.load() is True
            plugin.uninstall()
            result["uninstall"] = {"future_instances_host": type(cls()) is cls}
        result["status"] = (
            "PASS"
            if all(c["is_pass"] for c in numeric)
            and result["graph_replay"]["bit_exact"]
            and all(result["registration"].values())
            and result["default_off"]
            == {
                "load_returns": False,
                "key_absent": True,
            }
            else "FAIL"
        )
        result["l3_serving"] = "NOT_RUN"
        result["l3_compare"] = "NOT_RUN"
        result["l4"] = "NOT_RUN"
        result["performance_claim"] = False
        result["release_accepted"] = False
        return result
    finally:
        plugin.uninstall()
        plugin._reset_for_tests()
        if previous_env is None:
            os.environ.pop(plugin.ENV_ENABLE, None)
        else:
            os.environ[plugin.ENV_ENABLE] = previous_env


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    result = probe()
    with args.output.open("x") as stream:
        json.dump(result, stream, indent=2)
        stream.write("\n")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
