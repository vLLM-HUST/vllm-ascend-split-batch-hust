# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 vLLM-HUST contributors.

from __future__ import annotations

import argparse
import ast
import importlib
import importlib.util
import inspect
import json
from importlib.metadata import version
from pathlib import Path


def graph_boundary(graph) -> dict:
    targets = [str(n.target) for n in graph.nodes if n.op == "call_function"]
    opaque = "vllm.qwen_gdn_attention_core.default" in targets
    return {
        "call_function_targets": targets,
        "opaque_gdn_core": opaque,
        "decision": "outer_fx_cannot_match_core_internals" if opaque else "inspect",
        "limit": "CPU export of the actual registered op, not a serving graph dump",
    }


def core_calls(source: str) -> dict:
    tree = ast.parse(source)
    cls = next(
        n
        for n in tree.body
        if isinstance(n, ast.ClassDef) and n.name == "AscendGatedDeltaNetAttention"
    )
    core = next(
        n
        for n in cls.body
        if isinstance(n, ast.FunctionDef) and n.name == "_forward_core"
    )
    calls = sorted(
        {ast.unparse(n.func) for n in ast.walk(core) if isinstance(n, ast.Call)}
    )
    return {
        "core_line": core.lineno,
        "calls": calls,
        "direct_l2norm": "l2norm_fwd" in calls,
        "instance_unpack": "self.rearrange_mixed_qkv" in calls,
        "limit": "static call inventory, not proof of all possible extension points",
    }


def probe() -> dict:
    import torch
    import torch_npu  # noqa: F401
    from vllm.ir import register_op
    from vllm.ir.op import IrOp
    from vllm.model_executor.custom_op import PluggableLayer
    from vllm.model_executor.layers.mamba.gdn import qwen_gdn_linear_attn

    npugraph_ex = importlib.import_module("npugraph_ex")

    class CoreCall(torch.nn.Module):
        def forward(self, x, b, a, out):
            torch.ops.vllm.qwen_gdn_attention_core(x, b, a, out, "probe", False)
            return out

    args = (
        torch.empty(16, 4096, dtype=torch.bfloat16),
        torch.empty(16, 16, dtype=torch.bfloat16),
        torch.empty(16, 16, dtype=torch.bfloat16),
        torch.empty(16, 16, 128, dtype=torch.bfloat16),
    )
    exported = torch.export.export(CoreCall(), args)
    ascend = importlib.util.find_spec("vllm_ascend")
    source = Path(ascend.origin).parent / "ops/gdn.py"
    return {
        "schema": "gdn-registration-probe.v1",
        "versions": {
            name: version(name)
            for name in ("torch", "torch-npu", "vllm", "vllm-ascend")
        },
        "apis": {
            "register_op": str(inspect.signature(register_op)),
            "register_impl": str(inspect.signature(IrOp.register_impl)),
            "register_replacement": str(
                inspect.signature(npugraph_ex.register_replacement)
            ),
            "register_oot": str(inspect.signature(PluggableLayer.register_oot)),
        },
        "ir_ops": sorted(IrOp.registry),
        "core_source": inspect.getsourcefile(qwen_gdn_linear_attn),
        "core_schema": str(torch.ops.vllm.qwen_gdn_attention_core.default._schema),
        "export_code": exported.graph_module.code,
        "boundary": graph_boundary(exported.graph),
        "ascend_source": str(source),
        "inner_core": core_calls(source.read_text()),
        "device_execution": False,
        "release_accepted": False,
    }


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
