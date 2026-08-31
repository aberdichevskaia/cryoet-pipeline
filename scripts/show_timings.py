#!/usr/bin/env python3
"""Per-step time and disk for a project, from what the runner measured.

    python3 scripts/show_timings.py --project outputs/ribo

Reads `step_timings.jsonl`, which `run_series.py --execute` appends to as it
goes. The numbers it prints are the ones the workplan needs before a quota
request or a sandbox can be sized: wall-clock per step, disk per step, and what
one series costs end to end.

Disposable, like the rest of `scripts/`. Its replacement is the `run` entity,
which will record all of this properly and with a fingerprint; this exists
because W5 cannot be planned without the numbers and W2 has not happened yet.
"""

from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--project", required=True, type=Path)
    args = parser.parse_args()

    table = args.project / "step_timings.jsonl"
    if not table.is_file():
        print(f"no timings yet: {table}")
        print("they are written by scripts/run_series.py --execute")
        return 1

    rows = [json.loads(line) for line in table.read_text().splitlines() if line.strip()]
    if not rows:
        print("timing table is empty")
        return 1

    by_step: dict[str, list[dict]] = defaultdict(list)
    by_series: dict[str, list[dict]] = defaultdict(list)
    for row in rows:
        by_step[row["step"]].append(row)
        by_series[row["series"]].append(row)

    # Steps in the order they were first run, which is the pipeline's own order.
    order: list[str] = []
    for row in rows:
        if row["step"] not in order:
            order.append(row["step"])

    print(f"{'step':26} {'runs':>4} {'median min':>11} {'max min':>8} {'median GB':>10}")
    print("-" * 63)
    for step in order:
        entries = by_step[step]
        minutes = sorted(entry["seconds"] / 60.0 for entry in entries)
        gigabytes = sorted(entry["disk_delta_bytes"] / 1e9 for entry in entries)
        median = minutes[len(minutes) // 2]
        median_gb = gigabytes[len(gigabytes) // 2]
        print(
            f"{step:26} {len(entries):>4} {median:>11.1f} {minutes[-1]:>8.1f} "
            f"{median_gb:>+10.2f}"
        )

    print()
    print(f"{'series':12} {'steps':>6} {'total min':>10} {'total GB':>9} {'failed':>7}")
    print("-" * 48)
    for series in sorted(by_series):
        entries = by_series[series]
        total_minutes = sum(entry["seconds"] for entry in entries) / 60.0
        total_gb = sum(entry["disk_delta_bytes"] for entry in entries) / 1e9
        failed = sum(1 for entry in entries if entry["exit_code"] != 0)
        print(
            f"{series:12} {len(entries):>6} {total_minutes:>10.1f} "
            f"{total_gb:>+9.2f} {failed:>7}"
        )

    complete = [
        series
        for series, entries in by_series.items()
        if all(entry["exit_code"] == 0 for entry in entries)
    ]
    if complete:
        per_series = [
            sum(entry["seconds"] for entry in by_series[series]) / 3600.0
            for series in complete
        ]
        per_series_gb = [
            sum(entry["disk_delta_bytes"] for entry in by_series[series]) / 1e9
            for series in complete
        ]
        median_hours = sorted(per_series)[len(per_series) // 2]
        median_gb = sorted(per_series_gb)[len(per_series_gb) // 2]
        print()
        print(f"one series, median of {len(complete)} without failures:")
        print(f"  {median_hours:.2f} hours, {median_gb:+.2f} GB")
        for count in (5, 8, 50):
            print(
                f"  x{count:<3} {median_hours * count:>7.1f} hours "
                f"{median_gb * count:>+8.1f} GB"
            )
        print()
        print("Wall-clock as measured, which is not GPU-hours: a step that waited")
        print("on the filesystem or shared a node counts here and would not there.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
