# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 vLLM-HUST contributors.

from __future__ import annotations

import hashlib
import importlib.util
import json
import logging
import os
import sys
from importlib.machinery import PathFinder
from pathlib import Path

ENV_ENABLE = "VLLM_HUST_GDN_QKV_NORM"
HOST_MODULE = "vllm_ascend.ops.gdn"
HOST_SHA256 = "840fe9707fd27dba2ece121db730abb289add590aa05540a0c3a96f5a26841c3"


def host_source() -> Path:
    loaded = sys.modules.get(HOST_MODULE)
    if loaded is not None:
        origin = getattr(loaded, "__file__", None)
        if origin:
            return Path(origin)
        spec = getattr(loaded, "__spec__", None)
        if spec is not None and spec.origin:
            return Path(spec.origin)
    root = importlib.util.find_spec("vllm_ascend")
    if root is None or root.submodule_search_locations is None:
        raise RuntimeError("GDN host package is unavailable")
    ops = PathFinder.find_spec("vllm_ascend.ops", root.submodule_search_locations)
    if ops is None or ops.submodule_search_locations is None:
        raise RuntimeError("GDN host ops package is unavailable")
    spec = PathFinder.find_spec(HOST_MODULE, ops.submodule_search_locations)
    if spec is None or not spec.origin:
        raise RuntimeError("GDN host source is unavailable")
    return Path(spec.origin)


def preflight() -> dict:
    if os.environ.get(ENV_ENABLE) != "1":
        return {"status": "disabled", "enabled": False}
    try:
        if os.environ.get("USE_DEFAULT_FLA_NORM", "0") != "0":
            raise RuntimeError("GDN fusion requires USE_DEFAULT_FLA_NORM=0")
        allowed = os.environ.get("VLLM_PLUGINS")
        if allowed is not None and "gdn-qkv-norm" not in allowed.split(","):
            raise RuntimeError("VLLM_PLUGINS excludes gdn-qkv-norm")
        path = host_source()
        digest = hashlib.sha256(path.read_bytes()).hexdigest()
        if digest != HOST_SHA256:
            raise RuntimeError(
                f"GDN host seam mismatch: {path} sha256={digest}; "
                "requires the reviewed GDN-qkv-norm host proposal"
            )
        loaded = sys.modules.get(HOST_MODULE)
        if (loaded is not None
                and getattr(loaded, "_FUSE_GDN_QKV_NORM", None) is not True):
            raise RuntimeError("GDN was loaded without fusion; restart with env set")
    except (OSError, ImportError, ValueError, AttributeError, RuntimeError) as exc:
        return {"status": "blocked", "enabled": True, "reason": str(exc)}
    return {
        "status": "ready",
        "enabled": True,
        "host_source": str(path),
        "host_sha256": digest,
        "scope": "T16 bf16/ND pure non-spec decode; host guards authoritative",
        "activation": "host seam verified, not proof of kernel execution",
    }


def load() -> bool:
    result = preflight()
    if result["status"] == "blocked":
        raise RuntimeError(result["reason"])
    if result["status"] == "disabled":
        return False
    logging.getLogger(__name__).info("GDN QKV norm preflight: %s", result)
    return True


if __name__ == "__main__":
    result = preflight()
    print(json.dumps(result, sort_keys=True))
    raise SystemExit(3 if result["status"] == "blocked" else 0)
