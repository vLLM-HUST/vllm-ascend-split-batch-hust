from __future__ import annotations

import argparse
import ast
import functools
import hashlib
import inspect
import json
import logging
import sys
from pathlib import Path
from types import ModuleType, SimpleNamespace


def definition(path: Path, name: str):
    tree = ast.parse(path.read_text())
    node = next(n for n in tree.body if isinstance(n, ast.ClassDef) and n.name == name)
    return compile(
        ast.fix_missing_locations(
            ast.Module(
                body=[
                    ast.ImportFrom(
                        module="__future__",
                        names=[ast.alias(name="annotations")],
                        level=0,
                    ),
                    node,
                ],
                type_ignores=[],
            )
        ),
        str(path),
        "exec",
    )


def probe(core: Path, ascend: Path) -> dict:
    import torch
    import torch.nn.functional as F

    from vllm_ascend_split_batch import fi_gelu_plugin as plugin

    custom_path = core / "vllm/model_executor/custom_op.py"
    activation_path = core / "vllm/model_executor/layers/activation.py"
    utils_path = ascend / "vllm_ascend/utils.py"
    utils_tree = ast.parse(utils_path.read_text())
    register = next(
        n
        for n in utils_tree.body
        if isinstance(n, ast.FunctionDef) and n.name == "register_ascend_customop"
    )
    assert "GeluAndMul" not in {
        n.value
        for n in ast.walk(register)
        if isinstance(n, ast.Constant) and isinstance(n.value, str)
    }
    config = SimpleNamespace(
        custom_ops=["all"], enabled_custom_ops=set(), disabled_custom_ops=set()
    )
    platform = SimpleNamespace(
        **{
            name: (lambda: False)
            for name in ("is_rocm", "is_cpu", "is_tpu", "is_xpu", "is_cuda_alike")
        }
    )
    platform.is_out_of_tree = lambda: True
    custom = ModuleType("vllm.model_executor.custom_op")
    custom.__dict__.update(
        nn=torch.nn,
        torch=torch,
        functools=functools,
        inspect=inspect,
        logger=logging.getLogger("probe"),
        current_platform=platform,
        op_registry={},
        op_registry_oot={},
        get_cached_compilation_config=lambda: config,
    )
    exec(definition(custom_path, "CustomOp"), custom.__dict__)
    activation = ModuleType(plugin.HOST_MODULE)
    activation.__dict__.update(
        CustomOp=custom.CustomOp, torch=torch, F=F, current_platform=platform
    )
    exec(definition(activation_path, "GeluAndMul"), activation.__dict__)
    for name in (
        "vllm",
        "vllm.config",
        "vllm.config.compilation",
        "vllm.model_executor",
        "vllm.model_executor.layers",
    ):
        module = ModuleType(name)
        module.__path__ = []
        sys.modules[name] = module
        if "." in name:
            parent, child = name.rsplit(".", 1)
            setattr(sys.modules[parent], child, module)
    sys.modules["vllm.config.compilation"].CompilationMode = SimpleNamespace(NONE=0)
    for module in (custom, activation):
        sys.modules[module.__name__] = module
        parent, child = module.__name__.rsplit(".", 1)
        setattr(sys.modules[parent], child, module)

    host_cls = activation.GeluAndMul
    host_attrs = dict(host_cls.__dict__)
    old = host_cls()
    calls = []

    def kernel(x):
        calls.append(x)
        return "kernel-reached"

    plugin._fi_api = SimpleNamespace(gelu_and_mul=kernel)
    assert plugin.load(), "Run with VLLM_HUST_FI_GELU=1"
    try:
        op = host_cls()
        assert type(op) is plugin._registered_cls
        assert host_cls.__dict__ == host_attrs
        assert op._forward_method.__func__ is type(op).forward_oot
        assert op.name == host_cls.name == "gelu_and_mul"
        x = torch.randn(4, 8, dtype=torch.bfloat16)
        assert torch.equal(op(x), old(x))
        fake_npu = SimpleNamespace(
            dtype=torch.bfloat16,
            shape=(4, 8),
            dim=lambda: 2,
            device=SimpleNamespace(type="npu"),
        )
        assert op(fake_npu) == "kernel-reached"
        for options, eligible in (
            (["none"], False),
            (["none", "+gelu_and_mul"], True),
            (["all", "-gelu_and_mul"], False),
        ):
            config.custom_ops = options
            instance = host_cls()
            assert instance.enabled() is eligible
            assert torch.equal(instance(x), old(x))
            if eligible:
                assert instance(fake_npu) == "kernel-reached"
            else:
                assert instance._forward_method.__name__ == "forward_native"
        assert "GeluAndMul" not in config.enabled_custom_ops
        assert "gelu_and_mul" in config.enabled_custom_ops
        assert "gelu_and_mul" in config.disabled_custom_ops
        assert plugin.load()
    finally:
        plugin.uninstall()
    assert type(host_cls()) is host_cls
    return {
        "schema": "gelu-oot-source-execution.v1",
        "status": "PASS",
        "sources": {
            str(p): hashlib.sha256(p.read_bytes()).hexdigest()
            for p in (custom_path, activation_path, utils_path)
        },
        "checks": [
            "original class unchanged",
            "host constructor selects provider",
            "host forward trampoline reaches adapter",
            "CPU fallback bit-exact",
            "custom_ops all/none and +/-gelu_and_mul preserved",
            "Ascend registration source has no GeluAndMul collision",
            "uninstall restores future construction",
        ],
        "kernel_stub_calls": len(calls),
        "scope": "Unmodified host class definitions executed with real CPU PyTorch; "
        "controlled config/OOT platform, stub kernel and NPU metadata. "
        "Not a normal host import, worker startup, model run or NPU execution.",
        "l3": "NOT_RUN",
        "l4": "NOT_RUN",
        "release_accepted": False,
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--core", type=Path, required=True)
    parser.add_argument("--ascend", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    result = probe(args.core, args.ascend)
    with args.output.open("x") as stream:
        json.dump(result, stream, indent=2)
        stream.write("\n")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
