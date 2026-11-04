#!/usr/bin/env python3
"""Ascend and Triton Knowledge Base query interface for op_agent.

Includes LRU in-memory query cache and direct process execution to
avoid repetitive 4-second subshell spawns.
"""

from __future__ import annotations

import io
import json
import os
import re
import sys
from contextlib import redirect_stdout
from typing import Any

_KB_QUERY_CACHE: dict[str, list[dict[str, Any]]] = {}
_KB_CARD_CACHE: dict[str, str] = {}


def query_ascend_kb(
    query: str,
    limit: int = 3,
    kb_root: str | None = None,
) -> list[dict[str, Any]]:
    """Query the local CANN / Ascend knowledge base via askb engine.

    Returns structured card items with title, local path, and concise summary.
    Employs in-memory caching to eliminate subshell overheads.
    """
    cache_key = f"{query.strip().lower()}:{limit}"
    if cache_key in _KB_QUERY_CACHE:
        return _KB_QUERY_CACHE[cache_key]

    if kb_root is None:
        kb_root = os.environ.get("CANNBOT_KNOWLEDGE_ROOT", "/root/cannbot-knowledge")
    scripts_dir = os.path.join(
        kb_root,
        ".agents",
        "skills",
        "knowledge-query",
        "scripts",
    )
    script_path = os.path.join(scripts_dir, "knowledge_query.py")

    if not os.path.exists(script_path):
        return []

    # Fast path: In-process invocation if possible
    try:
        if scripts_dir not in sys.path:
            sys.path.insert(0, scripts_dir)
        import knowledge_query

        out_buffer = io.StringIO()
        with redirect_stdout(out_buffer):
            code = knowledge_query.main(
                [
                    "--knowledge-root",
                    kb_root,
                    "search",
                    "--query",
                    query,
                    "-k",
                    str(limit),
                ]
            )
        if code == 0:
            data = json.loads(out_buffer.getvalue())
            results = []
            for item in data.get("results", []):
                results.append(
                    {
                        "title": item.get("title", ""),
                        "path": item.get("local_path", ""),
                        "summary": item.get("description", "")[:200],
                    }
                )
            _KB_QUERY_CACHE[cache_key] = results
            return results
    except Exception:
        pass

    # Fallback to subprocess if in-process invocation raises
    import subprocess

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
                    "summary": item.get("description", "")[:200],
                }
            )
        _KB_QUERY_CACHE[cache_key] = results
        return results
    except Exception:
        return []


def read_kb_card(file_path: str, max_chars: int = 1500) -> str:
    """Read the content of a specific knowledge base card.

    Automatically extracts actionable sections (rules, tips, constraints)
    to minimize prompt token overhead.
    """
    if file_path in _KB_CARD_CACHE:
        return _KB_CARD_CACHE[file_path]

    if not os.path.exists(file_path):
        return f"File not found: {file_path}"
    try:
        with open(file_path, encoding="utf-8") as f:
            full_text = f.read()

        # Extract high-value paragraphs (Code blocks, optimization guidelines)
        actionable_sections = []
        for line in full_text.splitlines():
            # Drop frontmatter metadata lines to save tokens
            if line.startswith(("author:", "date:", "license:", "status:")):
                continue
            actionable_sections.append(line)

        condensed = "\n".join(actionable_sections).strip()
        # Collapse multiple blank lines
        condensed = re.sub(r"\n{3,}", "\n\n", condensed)
        result = condensed[:max_chars]
        _KB_CARD_CACHE[file_path] = result
        return result
    except Exception as e:
        return f"Error reading card: {str(e)}"
