"""Small msprof command presets. Check installed msprof help before collecting."""

import argparse
import json
import subprocess
from pathlib import Path

ROOT = Path(__file__).resolve().parents[4]


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("mode", choices=("timeline", "metrics", "simulator"))
    parser.add_argument(
        "--application", required=True, help="Trusted command passed to msprof"
    )
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--kernel")
    parser.add_argument("--soc")
    parser.add_argument("--metric", default="PipeUtilization")
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args(argv)
    output = args.output.resolve()
    if not args.output.is_absolute() or output == ROOT or ROOT in output.parents:
        raise ValueError("Profile output must be absolute and outside the repository")
    if output.exists():
        raise ValueError("Use a new output directory to avoid mixing runs")
    cmd = ["msprof"]
    if args.mode == "simulator":
        if not args.kernel or not args.soc:
            parser.error("simulator requires --kernel and --soc")
        cmd += [
            "op",
            "simulator",
            f"--kernel-name={args.kernel}",
            f"--soc-version={args.soc}",
            "--launch-count=1",
            "--timeout=60",
        ]
    elif args.mode == "metrics":
        cmd += [f"--aic-metrics={args.metric}"]
    cmd += [f"--application={args.application}", f"--output={output}"]
    print(json.dumps({"command": cmd, "dry_run": args.dry_run}), flush=True)
    if args.dry_run:
        return 0
    return subprocess.run(cmd, check=False).returncode


if __name__ == "__main__":
    raise SystemExit(main())
