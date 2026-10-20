# SPDX-License-Identifier: Apache-2.0

"""Deployment probe for the opt-ops workbench, safe on an unprivileged runner.

The workbench lives under ``/root`` (mode 0750), which the GitHub runner user
cannot traverse.  Up to Python 3.13 ``Path.exists()`` re-raises
``PermissionError`` for any errno outside ``ENOENT``/``ENOTDIR``/``EBADF``/
``ELOOP``; 3.14 defers to ``os.path.exists`` and returns ``False``.  A skip guard
written as ``if not path.exists(): pytest.skip(...)`` therefore *errors* per test
on the 3.10/3.12 legs while the 3.14 leg skips cleanly (run 39437993180: 37 setup
errors, all in the three workbench test modules).

``deployed_file`` answers the same question without the errno split: the file is
either readable here, or the deployment is absent from this host's point of view.
"""

from __future__ import annotations

from pathlib import Path


def deployed_file(path: str | Path) -> bool:
    """Whether ``path`` is a regular file this process may read."""
    try:
        return Path(path).is_file()
    except OSError:
        return False
