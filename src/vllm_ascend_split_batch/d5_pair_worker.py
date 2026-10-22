# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 vLLM-HUST contributors.

import json
import os
from copy import copy
from pathlib import Path
from statistics import median

import torch
import torch_npu
from vllm.forward_context import get_forward_context, override_forward_context
from vllm.v1.worker.workspace import init_workspace_manager, use_workspace_lane
from vllm_ascend.worker.worker import NPUWorker

from .d5_decode_pair import (
    eligible,
    min_step,
    min_tokens,
    require_disjoint,
    split_decode,
)


class D5PairWorker(NPUWorker):
    def init_device(self):
        super().init_device()
        if os.environ.get("VLLM_HUST_D5_PAIR") == "1":
            init_workspace_manager(self.device, 1, num_lanes=2)

    def load_model(self):
        super().load_model()
        if os.environ.get("VLLM_HUST_D5_PAIR") != "1":
            return
        config = self.vllm_config
        if not config.model_config.enforce_eager or config.speculative_config:
            raise RuntimeError("D5 diagnostic requires eager and no speculation")
        if config.parallel_config.data_parallel_size != 1:
            raise RuntimeError("D5 diagnostic requires DP1")
        self.d5_done = False
        self.d5_step = 0
        for name, layer in self.get_model().named_modules():
            if getattr(layer, "layer_type", None) == "linear_attention":
                if layer.layer_scale or layer.use_attn_reduce_scatter_for_moe:
                    raise RuntimeError("D5 does not support layer scale or SP")
                self.d5_layer_name = name
                layer.register_forward_pre_hook(self._probe, with_kwargs=True)
                break
        else:
            raise RuntimeError("no GDN decoder layer")

    def execute_model(self, scheduler_output):
        if os.environ.get("VLLM_HUST_D5_PAIR") == "1":
            self.d5_step += 1
        return super().execute_model(scheduler_output)

    def _probe(self, layer, args, kwargs):
        if self.d5_done or self.d5_step < min_step():
            return
        ctx = get_forward_context()
        prefix = layer.linear_attn.prefix
        meta = (
            ctx.attn_metadata.get(prefix)
            if isinstance(ctx.attn_metadata, dict)
            else None
        )
        hidden = kwargs.get("hidden_states", args[0] if args else None)
        residual = kwargs.get("residual", args[1] if len(args) > 1 else None)
        if hidden is None or not eligible(meta, hidden.shape[0]):
            return
        if hidden.shape[0] < min_tokens():
            return
        self.d5_done = True
        directory = Path(os.environ["D5_PAIR_DIR"])
        directory.mkdir(parents=True, exist_ok=True)
        try:
            report = self._run_pair(layer, hidden, residual, meta, ctx)
        except Exception as error:
            report = {"status": "FAIL", "error": repr(error)}
            raise
        finally:
            path = directory / f"rank{self.rank}-step{self.d5_step}.json"
            with path.open("x") as output:
                json.dump(report, output, indent=2)
            print(f"D5_PAIR_RESULT {path}", flush=True)

    def _run_pair(self, layer, hidden, residual, meta, ctx):
        n = hidden.shape[0]
        cut = n // 2
        metas = [split_decode(meta, 0, cut), split_decode(meta, cut, n)]
        require_disjoint(*metas)
        root = torch.npu.current_stream()
        streams = [root, torch.npu.Stream()]
        states = layer.linear_attn.kv_cache
        saved = [state.clone() for state in states]
        inputs = [hidden[:cut].clone(), hidden[cut:].clone()]
        residuals = (
            [None, None]
            if residual is None
            else [residual[:cut].clone(), residual[cut:].clone()]
        )
        contexts = []
        for m in metas:
            c = copy(ctx)
            c.attn_metadata = dict(ctx.attn_metadata)
            c.attn_metadata[layer.linear_attn.prefix] = m
            c.num_tokens = c.max_tokens_across_dp = m.num_actual_tokens
            c.padded_length = c.padded_num_tokens = m.num_actual_tokens
            c.all_moe_layers = None
            c.moe_layer_index = 0
            contexts.append(c)
        torch.npu.synchronize()

        def reset():
            for state, original in zip(states, saved, strict=True):
                state.copy_(original)
            torch.npu.synchronize()

        def attn(i):
            x = inputs[i].clone()
            r = residuals[i]
            if r is None:
                r = x
                x = layer.input_layernorm(x)
            else:
                x, r = layer.input_layernorm(x, r.clone())
            x = layer.linear_attn(hidden_states=x)
            return layer.post_attention_layernorm(x, r)

        def run(overlap):
            reset()
            begin = torch.npu.Event(enable_timing=True)
            end = torch.npu.Event(enable_timing=True)
            begin.record(root)
            out = [None, None]
            pending = [None, None]
            with override_forward_context(contexts[0]), use_workspace_lane(0):
                pending[0] = attn(0)
            ready = root.record_event()
            with override_forward_context(contexts[0]), use_workspace_lane(0):
                out[0] = (layer.mlp(pending[0][0]), pending[0][1])
            stream = streams[1] if overlap else root
            with torch.npu.stream(stream):
                stream.wait_event(ready)
                with override_forward_context(contexts[1]), use_workspace_lane(1):
                    pending[1] = attn(1)
                    out[1] = (layer.mlp(pending[1][0]), pending[1][1])
                done = stream.record_event()
            root.wait_event(done)
            end.record(root)
            end.synchronize()
            return out, [s.clone() for s in states], begin.elapsed_time(end)

        report = {
            "status": "PENDING",
            "kind": "real-input split decoder diagnostic",
            "layer": self.d5_layer_name,
            "prefix": layer.linear_attn.prefix,
            "model": self.model_config.model,
            "precision": str(hidden.dtype),
            "tp": self.parallel_config.tensor_parallel_size,
            "tokens": n,
            "split": [cut, n - cut],
            "phase": "decode",
            "state_slots": [m.non_spec_state_indices_tensor.tolist() for m in metas],
            "input_shape": list(hidden.shape),
            "repeats": [],
            "weight_formats": {
                name: {"shape": list(p.shape), "format": torch_npu.get_npu_format(p)}
                for name, p in layer.named_parameters()
                if "w13" in name or "w2" in name
            },
        }
        try:
            a, sa, _ = run(False)
            b, sb, _ = run(True)
            for arm_a, arm_b in zip(a, b, strict=True):
                for x, y in zip(arm_a, arm_b, strict=True):
                    torch.testing.assert_close(x, y, rtol=0.01, atol=0.01)
            for x, y in zip(sa, sb, strict=True):
                torch.testing.assert_close(x, y, rtol=0.0001, atol=0.0001)
            report["numerical"] = "PASS"
            for _ in range(10):
                run(False)
                run(True)
            for repeat in range(3):
                times = {False: [], True: []}
                for iteration in range(30):
                    for arm in [False, True] if iteration % 2 == 0 else [True, False]:
                        _, _, ms = run(arm)
                        times[arm].append(ms)
                report["repeats"].append(
                    {
                        "repeat": repeat,
                        "serial_ms": median(times[False]),
                        "interleaved_ms": median(times[True]),
                        "serial_samples": times[False],
                        "interleaved_samples": times[True],
                    }
                )
            report["status"] = "COMPLETE"
        finally:
            torch.npu.synchronize()
            reset()
        return report
