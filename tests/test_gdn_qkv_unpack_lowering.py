"""Optional Ascend910B2 offline compilation; no device launch."""

import os
import subprocess
import sys
from pathlib import Path

import pytest


@pytest.mark.skipif(
    os.environ.get("GDN_TEST_OFFLINE_COMPILE") != "1",
    reason="requires the Ascend Triton compiler toolchain",
)
@pytest.mark.parametrize("dtype", ["bf16", "fp16", "fp32"])
@pytest.mark.parametrize("single_tile", [False, True])
def test_boundary_mask_lowers_to_bounded_copy(dtype, single_tile, tmp_path):
    script = r'''
import sys
from pathlib import Path
import triton
from triton.compiler import ASTSource
from triton.backends.compiler import GPUTarget
from vllm_ascend_split_batch.gdn_qkv_unpack.kernels import (
    _unpack_mixed_qkv_kernel as kernel,
)

dtype, single_tile, output = sys.argv[1:]
constants = {"BLOCK": 4096}
if single_tile == "True":
    constants["n_col_tiles"] = 1
signature = {
    name: ("*" + dtype if i < 2 else "i32")
    for i, name in enumerate(kernel.arg_names)
    if name not in constants
}
compiled = triton.compile(
    ASTSource(kernel, signature, constants),
    target=GPUTarget("npu", "Ascend910B2", 0),
)
ir = compiled.asm["ttadapter"]
Path(output).write_text(ir)
assert "hivm.hir.sync_block_lock" not in ir
assert "hivm.hir.sync_block_unlock" not in ir
assert "arith.cmpf" not in ir
assert "memref.subview" in ir
assert "tensor.extract_slice" in ir
assert "arith.minsi" in ir
assert "arith.maxsi" in ir
'''
    result = subprocess.run(
        [sys.executable, "-c", script, dtype, str(single_tile),
         str(tmp_path / "kernel.ttadapter")],
        capture_output=True,
        text=True,
        timeout=120,
    )
    assert result.returncode == 0, result.stdout + result.stderr


def test_boundary_mask_remains_integer():
    import ast

    path = (Path(__file__).resolve().parents[1] / "src" /
            "vllm_ascend_split_batch/gdn_qkv_unpack/kernels.py")
    tree = ast.parse(path.read_text())
    masks = [
        node.value for node in ast.walk(tree)
        if isinstance(node, ast.Assign)
        and any(isinstance(t, ast.Name) and t.id == "mask" for t in node.targets)
    ]
    assert len(masks) == 1
    assert ast.dump(masks[0]) == ast.dump(ast.parse("col < width", mode="eval").body)
