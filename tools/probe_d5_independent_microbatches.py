# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 vLLM-HUST contributors.

from __future__ import annotations

import argparse
import ast
import hashlib
import json
from pathlib import Path
from types import SimpleNamespace as NS

import numpy as np


def load_method(source, name, namespace):
    tree = ast.parse(source)
    matches = [
        node
        for node in ast.walk(tree)
        if isinstance(node, ast.FunctionDef) and node.name == name
    ]
    if len(matches) != 1:
        raise ValueError(f"expected exactly one {name}, got {len(matches)}")
    node = matches[0]
    node.decorator_list = []
    module = ast.Module(
        body=[
            ast.ImportFrom(
                module="__future__", names=[ast.alias("annotations")], level=0
            ),
            node,
        ],
        type_ignores=[],
    )
    exec(compile(ast.fix_missing_locations(module), "<host-method>", "exec"), namespace)
    return namespace[name], {
        "method": name,
        "line": node.lineno,
        "end_line": node.end_lineno,
        "source": ast.get_source_segment(source, node),
    }


def expect_assert(call, expected):
    try:
        call()
    except AssertionError as error:
        if str(error) != expected:
            raise ValueError(f"unexpected assertion: {error}") from error
        return {"exception": "AssertionError", "message": str(error)}
    raise ValueError(f"host no longer rejects this input: {expected}")


def probe(core, ascend):
    files = {
        "v1": ascend / "vllm_ascend/worker/model_runner_v1.py",
        "v2": ascend / "vllm_ascend/worker/v2/model_states/mamba_hybrid.py",
        "worker": ascend / "vllm_ascend/worker/worker.py",
        "workspace": core / "vllm/v1/worker/workspace.py",
        "context": core / "vllm/forward_context.py",
        "slices": core / "vllm/v1/worker/ubatch_utils.py",
        "gdn": ascend / "vllm_ascend/ops/gdn.py",
    }
    texts = {key: path.read_text() for key, path in files.items()}
    witnesses = {}
    slice_namespace = {
        "np": np,
        "UBatchSlice": lambda req, tok: NS(
            request_slice=req, token_slice=tok, num_tokens=tok.stop - tok.start
        ),
    }
    load_method(texts["slices"], "_pad_out_ubatch_slices", slice_namespace)
    split, witnesses["splitter"] = load_method(
        texts["slices"], "maybe_create_ubatch_slices", slice_namespace
    )
    slice_cases = []
    for counts in ([1, 1], [3, 1]):
        slices, _ = split(True, np.array(counts), sum(counts), 2, 2)
        request_sets = [
            set(range(s.request_slice.start, s.request_slice.stop)) for s in slices
        ]
        slice_cases.append(
            {
                "stub_request_tokens_not_device_shapes": counts,
                "request_slices": [
                    [s.request_slice.start, s.request_slice.stop] for s in slices
                ],
                "token_slices": [
                    [int(s.token_slice.start), int(s.token_slice.stop)] for s in slices
                ],
                "shared_request_indices": sorted(request_sets[0] & request_sets[1]),
            }
        )
    if slice_cases[0]["shared_request_indices"] != [] or (
        slice_cases[1]["shared_request_indices"] != [0]
    ):
        raise ValueError("request boundary splitting changed; review state ownership")
    mode = NS(NONE="NONE", FULL="FULL")
    determine, witnesses["v1_selection"] = load_method(
        texts["v1"],
        "_determine_batch_execution_and_padding",
        {
            "np": np,
            "CUDAGraphMode": mode,
            "BatchDescriptor": lambda n: NS(num_tokens=n),
            "enable_sp": lambda _: False,
        },
    )
    parallel = NS(
        data_parallel_size=1, enable_dbo=True, use_ubatching=True, num_ubatches=2
    )
    runner = NS(
        _pad_for_sequence_parallelism=lambda n: n,
        input_batch=NS(
            num_computed_tokens_cpu=np.array([8, 8]),
            num_prompt_tokens=np.array([8, 8]),
            lora_id_to_lora_request={},
        ),
        use_dcp=False,
        model_config=NS(is_hybrid=True, is_encoder_decoder=False),
        uniform_decode_query_len=1,
        vllm_config=NS(
            parallel_config=parallel, observability_config=NS(cudagraph_metrics=False)
        ),
        parallel_config=parallel,
    )
    selections = []
    for counts, max_tokens in [(np.array([1, 1]), 1), (np.array([16, 16]), 16)]:
        result = determine(
            runner,
            int(counts.sum()),
            2,
            counts,
            max_tokens,
            False,
            allow_microbatching=True,
            force_eager=True,
        )
        if result[2] is not False:
            raise ValueError("v1 microbatch selection changed; review device path")
        selections.append(
            {
                "request_tokens": counts.tolist(),
                "allow_microbatching": True,
                "enable_dbo": True,
                "should_ubatch": result[2],
            }
        )

    class GDNBuilder:
        pass

    group = NS(get_metadata_builder=lambda _: GDNBuilder())
    build, witnesses["v1_metadata"] = load_method(
        texts["v1"],
        "_build_attn_group_metadata",
        {
            "self": NS(attn_groups=[[group]], device_metadata_providers=None),
            "skip_gdn_state_update": False,
            "GDNAttentionMetadataBuilder": GDNBuilder,
            "cascade_attn_prefix_lens": None,
        },
    )
    v1_forced = [
        expect_assert(
            lambda ubid=ubid: build(0, 0, None, {}, {}, {}, ubid=ubid),
            "UBatching not supported with GDN yet",
        )
        for ubid in (0, 1)
    ]
    prepare, witnesses["v2_metadata"] = load_method(texts["v2"], "prepare_attn", {})
    v2_forced = expect_assert(
        lambda: prepare(None, None, None, None, None, None, None, ubatch_idx=1),
        "DBO is not supported on Ascend",
    )
    workspace_id, witnesses["workspace_id"] = load_method(
        texts["workspace"],
        "_get_workspace_id",
        {"_workspace_lane": NS(get=lambda: 1), "dbo_current_ubatch_id": lambda: 0},
    )
    try:
        workspace_id(NS(_num_lanes=1))
    except RuntimeError as error:
        lane_error = str(error)
    else:
        raise ValueError("second workspace lane unexpectedly configured")
    if workspace_id(NS(_num_lanes=2)) != 1:
        raise ValueError("workspace lane capability changed")
    _, witnesses["worker_init"] = load_method(texts["worker"], "init_device", {})
    init_calls = [
        ast.unparse(n)
        for n in ast.walk(ast.parse(witnesses["worker_init"]["source"].lstrip()))
        if isinstance(n, ast.Call)
        and isinstance(n.func, ast.Name)
        and n.func.id == "init_workspace_manager"
    ]
    if init_calls != ["init_workspace_manager(self.device, num_ubatches)"]:
        raise ValueError("worker workspace initialization changed")
    if "num_ubatches = 1" not in witnesses["worker_init"]["source"]:
        raise ValueError("worker ubatch initialization changed")
    return {
        "schema": "d5-independent-microbatch-interface-probe.v1",
        "kind": "CPU execution of source-extracted methods with explicit stubs",
        "not_proven": [
            "normal host import",
            "NPU execution",
            "numerical equivalence",
            "physical overlap",
            "performance",
            "all possible worker subclasses",
        ],
        "verdict": "BLOCKED_CURRENT_HOST_PATH",
        "sources": {
            key: {
                "path": str(path),
                "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
            }
            for key, path in files.items()
        },
        "splitter_interface_cases": slice_cases,
        "v1_selection": selections,
        "v1_forced_gdn_metadata": v1_forced,
        "v2_forced_second_microbatch": v2_forced,
        "workspace": {
            "default_second_lane_error": lane_error,
            "two_lane_manager_can_select_lane_1": True,
            "worker_init_call": init_calls[0],
            "worker_num_ubatches": 1,
        },
        "witnesses": witnesses,
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--core", type=Path, required=True)
    parser.add_argument("--ascend", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    if args.out.exists():
        parser.error("refusing to overwrite evidence")
    result = probe(args.core, args.ascend)
    with args.out.open("x") as output:
        json.dump(result, output, indent=2)
        output.write("\n")
    print(f"{result['verdict']}: {args.out}; NOT a device result")


if __name__ == "__main__":
    main()
