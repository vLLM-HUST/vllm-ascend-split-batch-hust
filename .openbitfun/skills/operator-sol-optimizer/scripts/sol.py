"""Optimistic resource roofline arithmetic; no hardware defaults or measurements."""

import argparse
import json
import math
from pathlib import Path


def positive(value):
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError("Expected a finite positive number")
    if not math.isfinite(value) or value <= 0:
        raise ValueError("Expected a finite positive number")
    return value


def calculate(measured_us, resources):
    positive(measured_us)
    if not resources:
        raise ValueError("At least one resource budget is required")
    terms = {}
    for name, (work, capacity_per_second) in resources.items():
        terms[name] = positive(work) / positive(capacity_per_second) * 1e6
        positive(terms[name])
    lower = max(terms.values())
    consistent = lower <= measured_us
    return {
        "model": "optimistic_resource_roofline",
        "resource_lower_bound_us": terms,
        "lower_bound_us": lower,
        "measured_us": measured_us,
        "sol_pct": 100 * lower / measured_us,
        "consistent": consistent,
        "max_speedup": measured_us / lower if consistent else None,
        "reducible_fraction": 1 - lower / measured_us if consistent else None,
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "input",
        type=Path,
        help="JSON: measured_us, resources {name: [work, capacity_per_second]}",
    )
    args = parser.parse_args()
    data = json.loads(args.input.read_text())
    print(
        json.dumps(
            calculate(data["measured_us"], data["resources"]), indent=2, allow_nan=False
        )
    )


if __name__ == "__main__":
    main()
