"""Read one collection archive and report the views that drive decisions.

An archive carries several artifacts written by different steps, and they name
the same operator three ways: a short name (``GroupedMatmul``), an aclnn call
name (``aclnnGroupedMatmulV5_GroupedMatmul_GroupedMatmul``), and a kernel name.
Joining them by hand is where most of the re-derivation happens, so the join
lives here instead.

Four views:

``wall``
    Wall-clock split of the collection window. Device time is only part of it;
    the rest is idle and unoverlapped communication, which no kernel change can
    reach.

``hotspot``
    Per-operator device time, split into the part explained by the typical call
    (``median_us × calls``) and the part contributed by the long calls. For a
    run that mixes prefill and decode, the long-call column is the prefill one.

``sol``
    Roofline verdict per operator, taken from the archive's own checklist.

``e2e``
    Upper bound on end-to-end gain per operator, evaluated with the repository's
    own SNR pre-screen (``serving_loop.evaluate_e2e_snr_gate``) after scaling
    the saving by the device share of wall-clock time::

        device_share = Computing / Stage
        saving_pct   = share_pct × (100 − sol_pct) / 100

    This is the number that decides whether an optimisation can be observed at
    all. A candidate whose upper bound sits under the noise floor cannot show a
    result no matter how well it is implemented.

Nothing here writes to an archive or to the repository.
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
from dataclasses import dataclass, field
from pathlib import Path

from tools.operator_tools import hotspot_round
from tools.operator_tools.serving_loop import evaluate_e2e_snr_gate

NOISE_FLOOR_PCT = 1.5


def normalize(name: str) -> str:
    """Map an aclnn call name onto the short operator name the summaries use.

    ``aclnnMatmul_MatMulCommon_MatMulV2`` and ``MatMulV2`` are the same
    operator. Names that already are short pass through unchanged.
    """
    if name.startswith("aclnn") and "_" in name:
        return name.rsplit("_", 1)[-1]
    return name


@dataclass
class Archive:
    root: Path
    sources: dict = field(default_factory=dict)

    def _first(self, pattern: str) -> Path | None:
        hits = sorted(self.root.rglob(pattern))
        if not hits:
            return None
        self.sources[pattern] = hits[0].name
        if len(hits) > 1:
            self.sources[pattern] += (
                f" (1 of {len(hits)}: {', '.join(h.name for h in hits[1:])})"
            )
        return hits[0]

    def wall_clock(self) -> dict | None:
        path = self._first("step_trace_*.csv")
        if path is None:
            return None
        with path.open(newline="") as handle:
            rows = list(csv.DictReader(handle))
        if not rows:
            return None
        row = rows[0]

        def number(key: str) -> float:
            try:
                return float(row.get(key) or 0.0)
            except (TypeError, ValueError):
                return 0.0

        stage = number("Stage")
        if stage <= 0:
            return None
        computing = number("Computing")
        free = number("Free")
        comm = number("Communication(Not Overlapped)")
        return {
            "source": str(path),
            "stage_us": stage,
            "computing_us": computing,
            "free_us": free,
            "comm_us": comm,
            "overlapped_us": number("Overlapped"),
            "computing_ratio": computing / stage,
            "free_ratio": free / stage,
            "comm_ratio": comm / stage,
        }

    def pipes(self) -> list[dict] | None:
        path = self._first("pipes_*.json")
        if path is None:
            return None
        body = json.loads(path.read_text())
        return body.get("ops") or None

    def hotspots(self) -> list[dict] | None:
        path = self._first("hotspots.json")
        if path is not None:
            summary = json.loads(path.read_bytes())
            if isinstance(summary, list):
                return summary
            if summary["schema"] != hotspot_round.SCHEMA:
                raise ValueError(f"unsupported hotspot summary: {path}")
            hotspot_round.check_ranks(
                summary["rank_sources"], hotspot_round.invocation_size(self.root)
            )
        elif self.is_round():
            summary = hotspot_round.summarize(
                self.root / "profile", hotspot_round.invocation_size(self.root)
            )
        else:
            return None
        for rank, source in summary["rank_sources"].items():
            self.sources[f"rank{rank}"] = source
        return summary["ops"]

    def is_round(self) -> bool:
        return self.root.match("round_*_off")

    def checklist(self) -> list[dict] | None:
        path = self._first("checklist.json")
        if path is None:
            return None
        return json.loads(path.read_text())

    def api(self) -> list[dict] | None:
        """Host-side API time, largest first.

        This is where a blocking wait shows up. A synchronize call that takes
        tens of milliseconds per invocation is the host waiting for the device,
        not the device doing work.
        """
        path = self._first("api_statistic*.csv")
        if path is None:
            return None
        rows = []
        with path.open(newline="") as handle:
            for row in csv.DictReader(handle):
                try:
                    total = float(row["Time(us)"])
                    count = int(row["Count"])
                except (KeyError, TypeError, ValueError):
                    continue
                rows.append(
                    {
                        "level": row.get("Level", ""),
                        "api": row.get("API Name", ""),
                        "total_us": total,
                        "count": count,
                        "avg_us": total / count if count else 0.0,
                    }
                )
        rows.sort(key=lambda item: item["total_us"], reverse=True)
        return rows

    def kernel_wait(self) -> list[dict] | None:
        """Per-kernel wait time against its own duration.

        A kernel whose wait dwarfs its duration is a synchronisation point, not
        compute: the interesting ones are the ones you can remove.
        """
        path = self._first("kernel_details.csv")
        if path is None:
            return None
        agg: dict[str, dict] = {}
        with path.open(newline="") as handle:
            for row in csv.DictReader(handle):
                try:
                    duration = float(row["Duration(us)"])
                    wait = float(row.get("Wait Time(us)") or 0.0)
                except (KeyError, TypeError, ValueError):
                    continue
                name = row.get("Name", "")
                slot = agg.setdefault(
                    name,
                    {"kernel": name, "count": 0, "duration_us": 0.0, "wait_us": 0.0},
                )
                slot["count"] += 1
                slot["duration_us"] += duration
                slot["wait_us"] += wait
        rows = []
        for slot in agg.values():
            ratio = (
                slot["wait_us"] / slot["duration_us"] if slot["duration_us"] else 0.0
            )
            rows.append({**slot, "wait_over_duration": ratio})
        rows.sort(key=lambda item: item["wait_us"], reverse=True)
        return rows


def hotspot_view(archive: Archive) -> list[dict]:
    """Per-operator time split between typical and long calls.

    ``median_us × calls`` estimates what the whole operator would have cost if
    every call ran at the typical duration. The remainder is what the long calls
    added — in a mixed prefill/decode run, that is the prefill share.
    """
    ops = None if archive.is_round() else archive.pipes()
    if ops:
        rows = []
        for item in ops:
            calls = item.get("calls") or 0
            median = item.get("median_us") or 0.0
            total = item.get("total_us") or 0.0
            rows.append(
                {
                    "op": normalize(item["op"]),
                    "raw": item["op"],
                    "calls": calls,
                    "total_us": total,
                    "share_pct": item.get("share_pct") or 0.0,
                    "median_us": median,
                    "typical_us": median * calls,
                    "long_call_us": max(total - median * calls, 0.0),
                    "bottleneck_form": item.get("bottleneck_form") or "",
                    "busy_pipe_pct": item.get("busy_pipe_pct") or 0.0,
                }
            )
        rows.sort(key=lambda row: row["total_us"], reverse=True)
        return rows

    summary = archive.hotspots()
    if not summary:
        return []
    return [
        {
            "op": normalize(item["op"]),
            "raw": item["op"],
            "calls": item.get("count") or 0,
            "total_us": item.get("total_us") or 0.0,
            "share_pct": item.get("share_pct") or 0.0,
            "median_us": None,
            "typical_us": None,
            "long_call_us": None,
            "bottleneck_form": "",
            "busy_pipe_pct": 0.0,
        }
        for item in summary
    ]


def e2e_view(archive: Archive) -> list[dict]:
    """Upper bound on end-to-end gain per operator, with the decidability flag.

    The saving is expressed against the archive's own device total, then handed
    to the repository SNR pre-screen with the device share of wall-clock time so
    the idle part of the window is accounted for.
    """
    wall = archive.wall_clock()
    if wall is None:
        return []
    rows = hotspot_view(archive)
    verdicts = {item["op"]: item for item in (archive.checklist() or [])}
    out = []
    for row in rows:
        verdict = verdicts.get(row["op"], {})
        sol_pct = verdict.get("sol_pct")
        if sol_pct is None:
            continue
        gap_device = row["share_pct"] * (100.0 - float(sol_pct)) / 100.0
        gate = evaluate_e2e_snr_gate(
            net_saved_ms=gap_device,
            all_kernel_total_ms=100.0,
            serving_noise_floor_pct=NOISE_FLOOR_PCT,
            device_share=wall["computing_ratio"],
        )
        gap_e2e = gate["expected_all_kernel_delta_pct"]
        out.append(
            {
                "op": row["op"],
                "share_pct": row["share_pct"],
                "sol_pct": float(sol_pct),
                "device_gap_pct": gap_device,
                "e2e_upper_pct": gap_e2e,
                "decidable": gap_e2e >= NOISE_FLOOR_PCT,
                "snr_ratio": gate["snr_ratio"],
                "recommend_full_ab": gate["recommend_full_ab"],
                "bottleneck_form": row["bottleneck_form"],
            }
        )
    out.sort(key=lambda row: row["e2e_upper_pct"], reverse=True)
    return out


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("archive", type=Path)
    parser.add_argument(
        "--view",
        choices=("wall", "hotspot", "sol", "e2e", "api", "wait", "all"),
        default="all",
    )
    parser.add_argument("--top", type=int, default=20)
    args = parser.parse_args(argv)

    archive = Archive(args.archive)
    if not archive.root.is_dir():
        print(f"archive not found: {archive.root}")
        return 1

    try:
        rounds = sorted(p for p in archive.root.glob("round_*_off") if p.is_dir())
        manifest = archive.root / "collection_manifest.json"
        n = (
            json.loads(manifest.read_bytes()).get("rounds")
            if manifest.is_file()
            else None
        )
        if n is not None and (
            n < 1
            or {p.name for p in rounds} != {f"round_{i:02d}_off" for i in range(n)}
        ):
            raise ValueError(f"expected {n} rounds, found {[p.name for p in rounds]}")
        for root in rounds or [archive.root]:
            if rounds:
                print(f"== {root.name} ==")
            rc = show(Archive(root), args.view, args.top)
            if rc:
                return rc
        return 0
    except (ValueError, OSError, TypeError, KeyError, IndexError) as exc:
        print(f"invalid archive {archive.root}: {exc}", file=sys.stderr)
        return 1


def show(archive, view, top):
    # Report the sources actually used by the requested views.
    rows_hotspot = hotspot_view(archive) if view in ("hotspot", "all") else []
    if view in ("wall", "e2e", "all"):
        archive.wall_clock()
    if view in ("sol", "e2e", "all"):
        archive.checklist()
    if view in ("api", "all"):
        archive.api()
    if view in ("wait", "all"):
        archive.kernel_wait()
    if archive.sources:
        print("== artifacts read ==")
        for pattern, name in sorted(archive.sources.items()):
            print(f"  {pattern:<22} {name}")
        print()

    if view in ("wall", "all"):
        wall = archive.wall_clock()
        print("== wall clock ==")
        if wall is None:
            print("  no step_trace artifact in this archive")
        else:
            stage = wall["stage_us"]
            print(f"  {'stage':<14}{stage:>12.0f} us   100.0%")
            for label, key in (
                ("computing", "computing_us"),
                ("free", "free_us"),
                ("communication", "comm_us"),
            ):
                value = wall[key]
                print(f"  {label:<14}{value:>12.0f} us   {value / stage * 100:>5.1f}%")
            print(f"  {'overlapped':<14}{wall['overlapped_us']:>12.0f} us")
        print()

    if view in ("hotspot", "all"):
        rows = rows_hotspot
        print("== hotspot ==")
        if not rows:
            print("  no pipes or hotspots artifact in this archive")
            if view == "hotspot" or archive.is_round():
                return 1
        else:
            print(
                f"  {'op':<28}{'calls':>7}{'share%':>8}"
                f"{'total_us':>11}{'long_us':>9}  form"
            )
            for row in rows[:top]:
                long_call = (
                    "-" if row["long_call_us"] is None else f"{row['long_call_us']:.0f}"
                )
                print(
                    f"  {row['op'][:26]:<28}{row['calls']:>7}{row['share_pct']:>8.2f}"
                    f"{row['total_us']:>11.0f}{long_call:>9}  {row['bottleneck_form']}"
                )
        print()

    if view in ("sol", "all"):
        verdicts = archive.checklist() or []
        print("== sol ==")
        if not verdicts:
            print("  no checklist artifact in this archive")
        else:
            for item in verdicts[:top]:
                sol = item.get("sol_pct")
                sol_text = "n/a" if sol is None else f"{float(sol):.1f}"
                print(
                    f"  {item['op'][:28]:<30} share={item.get('share_pct', 0):>6.2f}%"
                    f"  sol={sol_text:>6}  {item.get('section', '')}"
                )
        print()

    if view in ("e2e", "all"):
        rows = e2e_view(archive)
        print(f"== e2e potential (noise floor {NOISE_FLOOR_PCT}%) ==")
        if not rows:
            print("  needs both a step_trace artifact and a checklist artifact")
        else:
            print(f"  {'op':<28}{'share%':>8}{'sol%':>7}{'gap%':>8}{'e2e%':>8}  dec")
            for row in rows[:top]:
                mark = "yes" if row["decidable"] else "no"
                gap = row["device_gap_pct"]
                e2e = row["e2e_upper_pct"]
                print(
                    f"  {row['op'][:26]:<28}{row['share_pct']:>8.2f}"
                    f"{row['sol_pct']:>7.1f}{gap:>8.2f}{e2e:>8.2f}  {mark}"
                )
            reachable = [row for row in rows if row["decidable"]]
            print()
            print(
                f"  candidates above the noise floor: {len(reachable)} of {len(rows)}"
            )
        print()

    if view in ("api", "all"):
        rows = archive.api()
        print("== host api ==")
        if not rows:
            print("  no api_statistic artifact in this archive")
        else:
            grand = sum(row["total_us"] for row in rows) or 1.0
            print(
                f"  {'api':<40}{'time_ms':>10}{'count':>8}{'avg_us':>10}{'share%':>8}"
            )
            for row in rows[:top]:
                print(
                    f"  {row['api'][:38]:<40}{row['total_us'] / 1000:>10.1f}"
                    f"{row['count']:>8}{row['avg_us']:>10.1f}"
                    f"{row['total_us'] / grand * 100:>8.1f}"
                )
        print()

    if view in ("wait", "all"):
        rows = archive.kernel_wait()
        print("== kernel wait ==")
        if not rows:
            print("  no kernel_details artifact in this archive")
        else:
            total_wait = sum(row["wait_us"] for row in rows) or 1.0
            print(
                f"  {'kernel':<40}{'count':>8}{'dur_us':>11}"
                f"{'wait_us':>10}{'wait/dur':>10}"
            )
            for row in rows[:top]:
                print(
                    f"  {row['kernel'][:38]:<40}{row['count']:>8}"
                    f"{row['duration_us']:>11.0f}{row['wait_us']:>10.0f}"
                    f"{row['wait_over_duration']:>10.2f}"
                )
            print(f"  total wait {total_wait:.0f} us")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
