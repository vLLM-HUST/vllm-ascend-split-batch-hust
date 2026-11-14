#!/usr/bin/env python3
"""Hardware and environment guard for NPU evaluation."""

from __future__ import annotations

import os
from typing import NamedTuple


class HardwareStatus(NamedTuple):
    has_npu: bool
    device_count: int
    is_tp2_ready: bool
    device_name: str
    message: str


def check_npu_environment(required_cards: int = 2) -> HardwareStatus:
    """Check if the execution environment meets NPU hardware requirements.

    Strict fail-closed checks per AGENTS.md.
    """
    # Prevent device autoload hang on CPU-only containers
    if "TORCH_DEVICE_BACKEND_AUTOLOAD" not in os.environ:
        os.environ["TORCH_DEVICE_BACKEND_AUTOLOAD"] = "0"

    try:
        import contextlib

        import torch

        with contextlib.suppress(ImportError):
            import torch_npu  # noqa: F401

        # Check torch_npu availability
        if not hasattr(torch, "npu"):
            return HardwareStatus(
                has_npu=False,
                device_count=0,
                is_tp2_ready=False,
                device_name="None",
                message="torch.npu module is not available in environment.",
            )

        if not torch.npu.is_available():
            return HardwareStatus(
                has_npu=False,
                device_count=0,
                is_tp2_ready=False,
                device_name="None",
                message=(
                    "No Ascend NPU devices found (/dev/davinci* missing). "
                    "Please switch workspace spec to npu-2x in console."
                ),
            )

        count = torch.npu.device_count()
        name = torch.npu.get_device_name(0) if count > 0 else "Unknown"
        is_ready = count >= required_cards

        msg = (
            f"NPU ready: {count}x {name}"
            if is_ready
            else f"Need {required_cards} NPUs for TP={required_cards}, found {count}."
        )

        return HardwareStatus(
            has_npu=True,
            device_count=count,
            is_tp2_ready=is_ready,
            device_name=name,
            message=msg,
        )

    except Exception as e:
        return HardwareStatus(
            has_npu=False,
            device_count=0,
            is_tp2_ready=False,
            device_name="Error",
            message=f"Error inspecting NPU environment: {str(e)}",
        )
