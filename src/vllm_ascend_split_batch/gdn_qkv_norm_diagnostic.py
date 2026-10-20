# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 vLLM-HUST contributors.

from __future__ import annotations

import json
from pathlib import Path


def attention_class_for_model(model, core):
    classes = {
        type(layer)
        for layer in model.modules()
        if getattr(type(layer), "_forward_core", None) is core
    }
    if len(classes) != 1:
        raise RuntimeError(f"expected one bound GDN class, found {len(classes)}")
    return classes.pop()


def equality_mask(a, b, *, bitwise=False):
    import torch

    if bitwise:
        return a.view(torch.int16) == b.view(torch.int16)
    return (a == b) | (torch.isnan(a) & torch.isnan(b))


def tensor_difference(actual, expected, *, bitwise=False):
    import torch

    if actual.shape != expected.shape or actual.dtype != expected.dtype:
        raise ValueError("diagnostic shape/dtype mismatch")
    a, b = actual.detach().cpu(), expected.detach().cpu()
    same = equality_mask(a, b, bitwise=bitwise)
    indices = (~same).nonzero()
    first = indices[0].tolist() if len(indices) else None
    finite = torch.isfinite(a) & torch.isfinite(b)
    return {
        "different": len(indices),
        "first_index": first,
        "actual": float(a[tuple(first)]) if first is not None else None,
        "expected": float(b[tuple(first)]) if first is not None else None,
        "max_abs_finite": float((a.float() - b.float()).abs()[finite].max())
        if finite.any()
        else None,
        "bitwise_equal": torch.equal(a.view(torch.int16), b.view(torch.int16)),
    }


class NormDiagnostic:
    def __init__(self, directory, rank, max_steps=128, metadata=None):
        if max_steps < 1:
            raise ValueError("max_steps must be positive")
        self.directory = Path(directory)
        self.directory.mkdir(parents=True, exist_ok=True)
        self.rank = rank
        self.max_steps = max_steps
        self.steps = 0
        self.covered_steps = 0
        self.seen_layers = set()
        self.metadata = dict(metadata or {})
        self.finished = False
        self.current_layer = None
        self.slots = {}

    def observe(self, x, actual, expected):
        import torch

        if self.current_layer is None:
            raise RuntimeError("missing GDN layer context")
        name = self.current_layer.prefix
        if name not in self.slots:
            self.slots[name] = {
                "input": torch.empty_like(x),
                "actual": tuple(torch.empty_like(t) for t in actual),
                "expected": tuple(torch.empty_like(t) for t in expected),
                "counts": torch.zeros(3, dtype=torch.int64, device=x.device),
                "active": torch.zeros(1, dtype=torch.int64, device=x.device),
            }
        slot = self.slots[name]
        slot["input"].copy_(x)
        for i, (a, b) in enumerate(zip(actual, expected, strict=True)):
            slot["actual"][i].copy_(a)
            slot["expected"][i].copy_(b)
            same = equality_mask(a, b, bitwise=i == 2)
            slot["counts"][i].copy_((~same).sum())
        slot["active"].fill_(1)

    def begin_step(self):
        import torch

        if not self.finished:
            with torch.inference_mode():
                for slot in self.slots.values():
                    slot["active"].zero_()

    def end_step(self, scheduled_tokens):
        import torch

        if self.finished:
            return
        self.steps += 1
        active = []
        first = None
        for name, slot in self.slots.items():
            if not slot["active"].item():
                continue
            active.append(name)
            counts = slot["counts"].cpu().tolist()
            if any(counts) and first is None:
                first = (name, slot, counts)
        self.covered_steps += bool(active)
        self.seen_layers.update(active)
        report = {
            "schema": "gdn-norm-first-divergence.v1",
            "model_scope": self.metadata,
            "covered_steps": self.covered_steps,
            "seen_layers": sorted(self.seen_layers),
            "rank": self.rank,
            "step": self.steps,
            "scheduled_tokens": scheduled_tokens,
            "active_layers": active,
            "scope": "same-input native/fused QKV shadow; diagnostic, not L3 PASS",
        }
        if first is not None:
            name, slot, counts = first
            report.update(layer=name, counts=counts, status="DIVERGENCE")
            report["differences"] = {
                key: tensor_difference(a, b, bitwise=key == "v")
                for key, a, b in zip(
                    ("q", "k", "v"), slot["actual"], slot["expected"], strict=True
                )
            }
            payload = {
                "metadata": report,
                "input": slot["input"].cpu(),
                "actual": tuple(t.cpu() for t in slot["actual"]),
                "expected": tuple(t.cpu() for t in slot["expected"]),
            }
            with (self.directory / f"rank{self.rank}-first.pt").open("xb") as f:
                torch.save(payload, f)
            self.finished = True
        elif self.steps >= self.max_steps:
            report["status"] = "NO_DIVERGENCE" if self.seen_layers else "NO_COVERAGE"
            self.finished = True
        else:
            return
        with (self.directory / f"rank{self.rank}-summary.json").open("x") as f:
            json.dump(report, f, indent=2)
            f.write("\n")

    def install(self, api, attention_class, norm):
        original_unpack = api.unpack_norm
        original_core = attention_class._forward_core
        probe = self

        def unpack(x):
            actual = original_unpack(x)
            if probe.current_layer is None:
                raise RuntimeError("missing GDN layer context")
            q, k, v = probe.current_layer.rearrange_mixed_qkv(x)
            probe.observe(x, actual, (norm(q), norm(k), v))
            return actual

        def core(layer, *args, **kwargs):
            previous = probe.current_layer
            probe.current_layer = layer
            try:
                return original_core(layer, *args, **kwargs)
            finally:
                probe.current_layer = previous

        api.unpack_norm = unpack
        attention_class._forward_core = core
