#!/usr/bin/env python3
"""Placeholder L0 entry for gdn_split_fusion.

This operator is delivered on the host side (``layer: host_fork``); no
plugin-side microbenchmark script exists yet. The acceptance records stay in
``evidence.json`` and ``REPORT.md``. Replace this file with a real ``bench_l0.py``
before recording a new L0 measurement here.

Exit code 3 means "no measurement provided", never "benchmark passed".
"""

if __name__ == "__main__":
    print(__doc__)
    raise SystemExit(3)
