#!/usr/bin/env python3
"""Blind visual QC for a Dynamo particle export with PyTom scores in its manifest.

Example:
    python scripts/qc_dynamo_picks.py \
        --export /path/to/mra-export --out /path/to/pick-qc

After reviewing the plots, fill /path/to/pick-qc/thresholds.json and rerun with
--thresholds /path/to/pick-qc/thresholds.json. The resulting selected.tbl is a
subtable of initial.tbl and uses the original particle data folder and tags.

The suggested bend in a rank curve only centers a review sample. It is not a
score cutoff or an estimate of the false-positive rate. GT labels are never read.
The original table and particle files are not changed.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import random
from collections import defaultdict
from dataclasses import dataclass, replace
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


@dataclass(frozen=True)
class Pick:
    tag: int
    run: str
    tomogram_number: int
    score: float
    x: float
    y: float
    z: float
    file: str
    table_line: str
    table_xyz: tuple[float, float, float]
    rank: int = 0


def load_picks(export: Path) -> dict[str, list[Pick]]:
    manifest = json.loads((export / "export_manifest.json").read_text())
    table_lines = (export / "initial.tbl").read_text().splitlines()
    by_tag: dict[int, tuple[str, list[float]]] = {}
    for line in table_lines:
        if not line.strip():
            continue
        fields = [float(value) for value in line.split()]
        if len(fields) != 42:
            raise ValueError(f"Dynamo table row has {len(fields)} columns, expected 42")
        tag = int(fields[0])
        if fields[0] != tag or tag in by_tag:
            raise ValueError(f"Duplicate or invalid Dynamo tag: {fields[0]}")
        by_tag[tag] = (line, fields)

    grouped: dict[str, list[Pick]] = defaultdict(list)
    seen: set[int] = set()
    for record in manifest["particles"]:
        tag = int(record["tag"])
        if tag in seen or tag not in by_tag:
            raise ValueError(f"Duplicate manifest tag or no matching table row: {tag}")
        seen.add(tag)
        line, fields = by_tag[tag]
        native = record["native_record"]
        score = float(native["score"])
        if not math.isfinite(score):
            raise ValueError(f"Nonfinite score for tag {tag}")
        xyz = (float(native["x_px"]), float(native["y_px"]), float(native["z_px"]))
        table_xyz = tuple(fields[23:26])
        if int(fields[19]) != int(record["tomogram_number"]):
            raise ValueError(f"Tomogram number mismatch for tag {tag}")
        if any(abs(table_xyz[i] - xyz[i] - 0.5) > 1e-3 for i in range(3)):
            raise ValueError(f"Coordinate mismatch for tag {tag}")
        particle_file = export / record["file"]
        if not particle_file.is_file():
            raise FileNotFoundError(particle_file)
        grouped[str(record["run"])].append(
            Pick(
                tag=tag,
                run=str(record["run"]),
                tomogram_number=int(record["tomogram_number"]),
                score=score,
                x=xyz[0],
                y=xyz[1],
                z=xyz[2],
                file=str(record["file"]),
                table_line=line,
                table_xyz=table_xyz,
            )
        )

    if seen != set(by_tag):
        raise ValueError(f"Manifest/table tag mismatch: {len(seen)} versus {len(by_tag)}")
    if not grouped:
        raise ValueError("No particles in export manifest")
    result: dict[str, list[Pick]] = {}
    for run, picks in sorted(grouped.items()):
        ordered = sorted(picks, key=lambda pick: (-pick.score, pick.tag))
        result[run] = [replace(pick, rank=rank) for rank, pick in enumerate(ordered, 1)]
    return result


def review_windows(picks: list[Pick]) -> tuple[int, dict[str, list[Pick]]]:
    """Find a curve bend for sampling, then take disjoint high/bend/low pools."""
    n = len(picks)
    scores = np.array([pick.score for pick in picks], dtype=float)
    if n < 15:
        raise ValueError("At least 15 picks per tomogram are needed for three review bands")
    if scores[0] == scores[-1]:
        bend = n // 2
    else:
        x = np.linspace(0.0, 1.0, n)
        y = (scores - scores[-1]) / (scores[0] - scores[-1])
        gap = (1.0 - x) - y
        lower, upper = max(1, round(0.05 * n)), min(n - 1, round(0.75 * n))
        bend = int(np.argmax(gap[lower:upper]) + lower)

    half_width = max(10, round(0.05 * n))
    start, end = max(0, bend - half_width), min(n, bend + half_width + 1)
    low_start = max(end, math.floor(0.75 * n))
    return bend, {
        "high": picks[:start],
        "bend": picks[start:end],
        "low": picks[low_start:],
    }


def stable_sample(picks: list[Pick], count: int, seed: int, run: str, band: str) -> list[Pick]:
    digest = hashlib.sha256(f"{seed}:{run}:{band}".encode()).digest()
    generator = random.Random(int.from_bytes(digest[:8], "big"))
    return sorted(generator.sample(picks, min(count, len(picks))), key=lambda pick: pick.rank)


def write_table(path: Path, picks: list[Pick]) -> None:
    path.write_text("".join(pick.table_line + "\n" for pick in picks))


def write_csv(path: Path, picks: list[Pick], band_by_tag: dict[int, str] | None = None) -> None:
    names = [
        "run", "tomogram_number", "band", "tag", "rank", "score",
        "x_native_px", "y_native_px", "z_native_px",
        "x_dynamo_px", "y_dynamo_px", "z_dynamo_px", "particle_file",
    ]
    with path.open("w", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(names)
        for pick in picks:
            writer.writerow(
                [
                    pick.run, pick.tomogram_number,
                    "" if band_by_tag is None else band_by_tag[pick.tag],
                    pick.tag, pick.rank, f"{pick.score:.9g}",
                    pick.x, pick.y, pick.z, *pick.table_xyz, pick.file,
                ]
            )


def plot_scores(
    grouped: dict[str, list[Pick]],
    bends: dict[str, int],
    sampled: dict[str, dict[str, list[Pick]]],
    out: Path,
) -> None:
    ncols = 2
    nrows = math.ceil(len(grouped) / ncols)
    colors = {"high": "#177245", "bend": "#e38420", "low": "#2468a0"}
    fig, axes = plt.subplots(nrows, ncols, figsize=(12, 3.5 * nrows), squeeze=False)
    for ax, (run, picks) in zip(axes.flat, grouped.items(), strict=False):
        ax.plot([p.rank for p in picks], [p.score for p in picks], color="#30343b", lw=1.5)
        bend = bends[run]
        ax.axvline(bend + 1, color="#888888", lw=1, ls="--", label="review bend")
        for band, points in sampled[run].items():
            ax.scatter(
                [p.rank for p in points], [p.score for p in points], s=20,
                label=f"{band} sample", color=colors[band], zorder=3,
            )
        ax.set_title(f"{run}  |  {len(picks)} peaks")
        ax.set_xlabel("Rank (highest score first)")
        ax.set_ylabel("PyTom score")
        ax.grid(alpha=0.2)
    for ax in list(axes.flat)[len(grouped):]:
        ax.set_visible(False)
    axes.flat[0].legend(fontsize=8)
    fig.suptitle("Blind peak review — bend is a sampling guide, not a cutoff", fontsize=13)
    fig.tight_layout()
    fig.savefig(out / "score_rank_by_tomogram.png", dpi=180)
    fig.savefig(out / "score_rank_by_tomogram.svg")
    plt.close(fig)

    fig, axes = plt.subplots(nrows, ncols, figsize=(12, 3.3 * nrows), squeeze=False)
    values = [p.score for picks in grouped.values() for p in picks]
    bins = np.linspace(min(values), max(values), 35)
    for ax, (run, picks) in zip(axes.flat, grouped.items(), strict=False):
        ax.hist([p.score for p in picks], bins=bins, color="#496e91")
        ax.set_title(run)
        ax.set_xlabel("PyTom score")
        ax.set_ylabel("Number of peaks")
        ax.grid(axis="y", alpha=0.2)
    for ax in list(axes.flat)[len(grouped):]:
        ax.set_visible(False)
    fig.suptitle("Peak-score distributions (not CC-volume distributions)", fontsize=13)
    fig.tight_layout()
    fig.savefig(out / "score_histograms.png", dpi=180)
    plt.close(fig)


def write_galleries(
    export: Path, sampled: dict[str, dict[str, list[Pick]]], out: Path,
) -> None:
    """Make quick contact sheets of unaligned subtomograms, with three central slabs."""
    import mrcfile

    for run, bands in sampled.items():
        for band, picks in bands.items():
            if not picks:
                continue
            ncols = 4
            nrows = math.ceil(len(picks) / ncols)
            fig, axes = plt.subplots(nrows, ncols, figsize=(16, 2.4 * nrows), squeeze=False)
            for ax, pick in zip(axes.flat, picks, strict=False):
                with mrcfile.open(export / pick.file, permissive=False) as mrc:
                    volume = np.asarray(mrc.data, dtype=np.float32)
                if volume.ndim != 3 or not np.isfinite(volume).all():
                    raise ValueError(f"Invalid particle volume: {pick.file}")
                views = []
                for axis in range(3):
                    middle = volume.shape[axis] // 2
                    indices = range(max(0, middle - 2), min(volume.shape[axis], middle + 3))
                    slab = -np.take(volume, list(indices), axis=axis).mean(axis=axis)
                    low, high = np.percentile(slab, [2, 98])
                    image = np.clip((slab - low) / max(high - low, 1e-6), 0, 1)
                    views.append(image)
                separator = np.ones((views[0].shape[0], 3), dtype=np.float32)
                tile = np.concatenate(
                    [views[0], separator, views[1], separator, views[2]], axis=1
                )
                ax.imshow(tile, cmap="gray", vmin=0, vmax=1, interpolation="nearest")
                ax.set_title(f"tag {pick.tag} · rank {pick.rank} · {pick.score:.4f}", fontsize=9)
                ax.axis("off")
            for ax in list(axes.flat)[len(picks):]:
                ax.set_visible(False)
            fig.suptitle(
                f"{run} / {band}: raw unaligned central slabs Z | Y | X (protein white)",
                fontsize=12,
            )
            fig.tight_layout()
            fig.savefig(out / f"gallery_{run}_{band}.png", dpi=150)
            plt.close(fig)


def write_threshold_template(out: Path, grouped: dict[str, list[Pick]]) -> Path:
    path = out / "thresholds.json"
    if not path.exists():
        path.write_text(json.dumps({run: None for run in grouped}, indent=2) + "\n")
    return path


def apply_thresholds(
    export: Path, out: Path, threshold_file: Path, grouped: dict[str, list[Pick]],
) -> Path:
    try:
        raw = json.loads(threshold_file.read_text())
    except (OSError, json.JSONDecodeError) as exc:
        raise ValueError(f"Cannot read threshold JSON: {threshold_file}") from exc
    if not isinstance(raw, dict) or set(raw) != set(grouped):
        raise ValueError(
            "Threshold JSON must contain exactly these tomograms: " + ", ".join(grouped)
        )
    thresholds: dict[str, float] = {}
    for run, value in raw.items():
        if isinstance(value, bool) or not isinstance(value, (int, float)):
            raise ValueError(f"Set a numeric threshold for {run}; null is only a placeholder")
        threshold = float(value)
        if not math.isfinite(threshold) or not 0 <= threshold <= 1:
            raise ValueError(f"Threshold for {run} must be finite and between 0 and 1")
        thresholds[run] = threshold

    manifest_bytes = (export / "export_manifest.json").read_bytes()
    table_bytes = (export / "initial.tbl").read_bytes()
    rule_bytes = json.dumps(thresholds, sort_keys=True, separators=(",", ":")).encode()
    input_sha = hashlib.sha256(manifest_bytes + table_bytes).hexdigest()
    selection_id = hashlib.sha256(input_sha.encode() + rule_bytes).hexdigest()[:12]
    selected_all: list[Pick] = []
    rejected_all: list[Pick] = []
    summary: list[dict[str, object]] = []
    selected_by_run: dict[str, list[Pick]] = {}
    for run, picks in grouped.items():
        threshold = thresholds[run]
        selected = sorted((p for p in picks if p.score >= threshold), key=lambda p: p.tag)
        rejected = sorted((p for p in picks if p.score < threshold), key=lambda p: p.tag)
        selected_by_run[run] = selected
        selected_all.extend(selected)
        rejected_all.extend(rejected)
        summary.append({
            "run": run, "min_score": threshold,
            "selected": len(selected), "rejected": len(rejected), "total": len(picks),
        })
    selected_all.sort(key=lambda p: p.tag)
    rejected_all.sort(key=lambda p: p.tag)
    if not selected_all:
        raise ValueError("Thresholds select no particles")
    selection_dir = out / "selections" / f"score_{selection_id}"
    tables = selection_dir / "tables"
    tables.mkdir(parents=True, exist_ok=True)
    for run, selected in selected_by_run.items():
        write_table(tables / f"{run}_selected.tbl", selected)
    write_table(selection_dir / "selected.tbl", selected_all)
    write_table(selection_dir / "rejected.tbl", rejected_all)
    write_csv(selection_dir / "selected.csv", selected_all)
    write_csv(selection_dir / "rejected.csv", rejected_all)
    with (selection_dir / "selection_summary.csv").open("w", newline="") as handle:
        writer = csv.DictWriter(
            handle, fieldnames=["run", "min_score", "selected", "rejected", "total"]
        )
        writer.writeheader()
        writer.writerows(summary)
    (selection_dir / "selection_manifest.json").write_text(
        json.dumps(
            {
                "rule": "native_record.score >= per-tomogram min_score",
                "thresholds": thresholds,
                "source_manifest_and_table_sha256": input_sha,
                "selected_count": len(selected_all),
                "rejected_count": len(rejected_all),
                "tomograms": summary,
                "particle_files_copied": False,
            },
            indent=2,
        ) + "\n"
    )
    return selection_dir


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--export", required=True, type=Path,
        help="Directory with initial.tbl and export_manifest.json",
    )
    parser.add_argument("--out", required=True, type=Path, help="QC output directory")
    parser.add_argument(
        "--per-band", type=int, default=20,
        help="Maximum sampled picks per run and band",
    )
    parser.add_argument("--seed", type=int, default=20261003, help="Reproducible sample seed")
    parser.add_argument(
        "--gallery", action="store_true",
        help="Also render quick contact sheets of the sampled MRC particles (requires mrcfile)",
    )
    parser.add_argument(
        "--thresholds", type=Path,
        help="JSON mapping each tomogram to a minimum PyTom score; writes a Dynamo subtable",
    )
    args = parser.parse_args()
    if args.per_band < 1:
        parser.error("--per-band must be positive")

    grouped = load_picks(args.export)
    args.out.mkdir(parents=True, exist_ok=True)
    template = write_threshold_template(args.out, grouped)
    tables = args.out / "tables"
    tables.mkdir(exist_ok=True)
    bends: dict[str, int] = {}
    samples: dict[str, dict[str, list[Pick]]] = {}
    all_picks: list[Pick] = []
    all_samples: list[Pick] = []
    band_by_tag: dict[int, str] = {}
    summary_rows: list[list[object]] = []
    for run, picks in grouped.items():
        bend, pools = review_windows(picks)
        bends[run] = bend
        samples[run] = {}
        write_table(tables / f"{run}_all.tbl", picks)
        all_picks.extend(picks)
        for band, pool in pools.items():
            selected = stable_sample(pool, args.per_band, args.seed, run, band)
            samples[run][band] = selected
            all_samples.extend(selected)
            for pick in selected:
                band_by_tag[pick.tag] = band
            write_table(tables / f"{run}_{band}.tbl", selected)
            summary_rows.append(
                [run, len(picks), band, len(pool), len(selected), bend + 1, picks[bend].score]
            )
    write_table(tables / "review_samples_all.tbl", all_samples)
    write_csv(args.out / "all_picks.csv", all_picks)
    write_csv(args.out / "review_samples.csv", all_samples, band_by_tag)
    with (args.out / "review_summary.csv").open("w", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow([
            "run", "total_picks", "band", "pool_size", "sample_size", "bend_rank", "bend_score",
        ])
        writer.writerows(summary_rows)
    plot_scores(grouped, bends, samples, args.out)
    if args.gallery:
        write_galleries(args.export, samples, args.out)
    print(f"Wrote {len(all_picks)} picks and {len(all_samples)} review samples to {args.out}")
    print("Bend markers guide sampling only; plots and samples used no cutoff or GT labels.")
    if args.thresholds is None:
        print(f"Fill per-tomogram thresholds in {template} after visual review.")
    if args.thresholds is not None:
        selection_dir = apply_thresholds(args.export, args.out, args.thresholds, grouped)
        print(f"Dynamo selection: {selection_dir / 'selected.tbl'}")


if __name__ == "__main__":
    main()
