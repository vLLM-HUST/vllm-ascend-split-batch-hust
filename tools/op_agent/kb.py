#!/usr/bin/env python3
"""Ascend and Triton Knowledge Base query interface for op_agent."""

from __future__ import annotations

import json
import os
import subprocess
from typing import Any


def query_ascend_kb(
    query: str,
    limit: int = 3,
    kb_root: str | None = None,
) -> list[dict[str, Any]]:
    """Query the local CANN / Ascend knowledge base via askb engine.

    Returns structured card items with title, local path, and concise summary.
    """
    if kb_root is None:
        kb_root = os.environ.get("CANNBOT_KNOWLEDGE_ROOT", "/root/cannbot-knowledge")
    script_path = os.path.join(
        kb_root,
        ".agents",
        "skills",
        "knowledge-query",
        "scripts",
        "knowledge_query.py",
    )

    if not os.path.exists(script_path):
        return []

    cmd = [
        "python3",
        script_path,
        "--knowledge-root",
        kb_root,
        "search",
        "--query",
        query,
        "-k",
        str(limit),
    ]

    try:
        res = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            check=True,
            timeout=15,
        )
        data = json.loads(res.stdout)
        results = []
        for item in data.get("results", []):
            results.append(
                {
                    "title": item.get("title", ""),
                    "path": item.get("local_path", ""),
                    "summary": item.get("description", "")[:250],
                }
            )
        return results
    except Exception:
        return []


def read_kb_card(file_path: str, max_chars: int = 2000) -> str:
    """Read the content of a specific knowledge base card."""
    if not os.path.exists(file_path):
        return f"File not found: {file_path}"
    try:
        with open(file_path, encoding="utf-8") as f:
            content = f.read(max_chars)
        return content
    except Exception as e:
        return f"Error reading card: {str(e)}"
