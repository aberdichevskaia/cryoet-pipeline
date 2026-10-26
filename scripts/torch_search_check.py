#!/usr/bin/env python3
"""The torch search kernel on real particles, against Dynamo's own search and the numpy port.

    python scripts/torch_search_check.py outputs/ryr1/dynamo/bin8-2x-pad/mra_pad_ssn_g2atan \\
        --iteration 5 --particles 16 --device cuda --global-step 9.0 --global-low 13

For one finished iteration of a Dynamo project searched at the full box (dim equal
to the box: a cone round, not a CPU global round), and a seeded sample of its
particles, every channel's search is run three ways from the iteration's own
inputs: its starting table, Dynamo's prepared reference
(starting_reference_transformed) and search mask (mask_transformed), under the
iteration's card. Dynamo's answer is the refined table; the numpy kernel
(dynamo_search.refine_particle, float64 on the CPU) is the port the replays
checked; the torch kernel (dynamo_search_torch, float32 on --device) is the one
to check. It prints, for each, the angle (deg) and shift (px) from Dynamo's pose
and the CC difference, per channel the median and the largest, and each kernel's
time per candidate orientation.

With --global-step it also times the torch kernel alone on a 360 deg search at
that step (cone and in-plane), refined twice by 2 as the setup's global rounds
are, low-pass --global-low Fourier px, at the full box: what a global round
would cost on this device without Dynamo's CPU downsampling.

Nothing is written. Disposable, like the rest of scripts/.
"""

from __future__ import annotations

import argparse
import math
import time
from pathlib import Path

import numpy as np
import torch

from cryoet_pipeline.backends import dynamo_search as core
from cryoet_pipeline.backends import dynamo_search_torch as kernel
from cryoet_pipeline.reports.dynamo_project import read_card, read_em, read_map, read_table

STARTING = "results/ite_{ite:04d}/starting_values"


def _settings(card: dict[str, str], box: int) -> core.SearchSettings:
    """The kernel's settings from an iteration card, refusing what the kernel does not model."""

    dim = int(float(card.get("dim", box)))
    if dim != box:
        raise SystemExit(f"this iteration searched at dim {dim}, not the box {box}: Dynamo's CPU "
                         "downsampling is not ported; pick a cone round")
    if card.get("sym", "c1") != "c1":
        raise SystemExit(f"symmetry {card.get('sym')} is not modelled")
    limits = [float(value) for value in card["area_search"].replace(",", " ").split()]
    if len(set(limits)) != 1:
        raise SystemExit(f"area_search {limits}: only a sphere is modelled")
    return core.SearchSettings(
        cone_aperture=float(card["cone_range"]), cone_sampling=float(card["cone_sampling"]),
        inplane_range=float(card["inplane_range"]),
        inplane_sampling=float(card["inplane_sampling"]),
        refine=int(float(card["refine"])), refine_factor=float(card["refine_factor"]),
        high=float(card["high"]), low=float(card["low"]), shift_limit=limits[0],
        area_search_modus=int(float(card["area_search_modus"])),
    )


def _angle(first: np.ndarray, second: np.ndarray) -> float:
    relative = core.euler_matrix(first) @ core.euler_matrix(second).T
    return math.degrees(math.acos(max(-1.0, min(1.0, (float(np.trace(relative)) - 1) / 2))))


def _particle(folder: Path, tag: int) -> np.ndarray:
    for extension in ("mrc", "em"):
        path = folder / f"particle_{tag:05d}.{extension}"
        if path.exists():
            return read_map(path).astype(float)
    raise SystemExit(f"no particle {tag} in {folder}")


def _candidates(result: core.SearchResult) -> int:
    return sum(int(level["candidate_count"]) for level in result.levels)


def main() -> int:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("project", type=Path, help="a finished Dynamo project folder")
    parser.add_argument("--iteration", type=int, required=True,
                        help="a finished iteration searched at the full box")
    parser.add_argument("--particles", type=int, default=16, help="sample size")
    parser.add_argument("--seed", type=int, default=20261026)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--chunk", type=int, default=256, help="candidates per batch")
    parser.add_argument("--global-step", type=float, default=None,
                        help="also time a 360 deg search at this step (deg)")
    parser.add_argument("--global-low", type=float, default=None,
                        help="the low-pass (Fourier px) of that search; default the card's")
    args = parser.parse_args()

    project = args.project.resolve()
    ite = args.iteration
    starting = project / STARTING.format(ite=ite)
    searched = project / f"results/ite_{ite:04d}/currently_surviving_references_ite_{ite:04d}.txt"
    channels = [int(value) for value in searched.read_text().split()]
    mask = read_em(starting / f"mask_transformed_ite_{ite:04d}.em").astype(float)
    box = mask.shape[0]
    first_card = read_card(project / f"cards/ite_{ite:04d}/card_iteref_ref_{channels[0]:03d}"
                           f"_ite_{ite:04d}.card")
    data = project.parent / first_card.get("folder_data", "particles")
    device = torch.device(args.device)
    print(f"{project.name}, iteration {ite}, channels {channels}, box {box}, particles from "
          f"{data}, torch on {device}"
          + (f" ({torch.cuda.get_device_name(device)})" if device.type == "cuda" else ""))

    sample = None
    rows_by_channel = {}
    for channel in channels:
        card = read_card(project / f"cards/ite_{ite:04d}/card_iteref_ref_{channel:03d}"
                         f"_ite_{ite:04d}.card")
        settings = _settings(card, box)
        reference = read_em(starting / f"starting_reference_transformed_ref_{channel:03d}"
                            f"_ite_{ite:04d}.em").astype(float)
        start = read_table(starting / f"starting_table_ref_{channel:03d}_ite_{ite:04d}.tbl")
        refined = read_table(project / f"results/ite_{ite:04d}/averages/refined_table_ref_"
                             f"{channel:03d}_ite_{ite:04d}.tbl")
        if sample is None:
            tags = np.sort(start[:, 0].astype(int))
            rng = np.random.default_rng(args.seed)
            sample = np.sort(rng.choice(tags, size=min(args.particles, len(tags)),
                                        replace=False))
            volumes = {int(tag): _particle(data, int(tag)) for tag in sample}
        starts = {int(row[0]): row for row in start}
        dynamo = {int(row[0]): row for row in refined}
        rows = []
        times = {"numpy": [0.0, 0], "torch": [0.0, 0]}
        for tag in sample:
            row = starts[int(tag)]
            inputs = (volumes[int(tag)], reference, mask, row[6:9], row[3:6],
                      (float(row[13]), float(row[14])), settings)
            began = time.perf_counter()
            ours = core.refine_particle(*inputs)
            times["numpy"][0] += time.perf_counter() - began
            times["numpy"][1] += _candidates(ours)
            began = time.perf_counter()
            fast = kernel.refine_particle_torch(*inputs, device=device, chunk=args.chunk)
            if device.type == "cuda":
                torch.cuda.synchronize(device)
            times["torch"][0] += time.perf_counter() - began
            times["torch"][1] += _candidates(fast)
            truth = dynamo[int(tag)]
            rows.append((
                _angle(ours.angles, truth[6:9]), float(np.linalg.norm(ours.shift - truth[3:6])),
                ours.correlation - truth[9],
                _angle(fast.angles, ours.angles), float(np.linalg.norm(fast.shift - ours.shift)),
                fast.correlation - ours.correlation,
            ))
        rows_by_channel[channel] = rows
        table = np.array(rows)
        print(f"channel {channel}: {settings.cone_aperture:g} deg cone in "
              f"{settings.cone_sampling:g} deg steps, refine {settings.refine}, low "
              f"{settings.low:g} px, {len(sample)} particles")
        for name, columns in (("numpy vs Dynamo", (0, 1, 2)), ("torch vs numpy", (3, 4, 5))):
            angle, shift, cc = (table[:, column] for column in columns)
            print(f"  {name:16s} angle median {np.median(angle):.3f}, max {angle.max():.3f} "
                  f"deg; shift median {np.median(shift):.3f}, max {shift.max():.3f} px; "
                  f"CC difference max {np.abs(cc).max():.2e}")
        for name, (seconds, count) in times.items():
            print(f"  {name:5s} {1000 * seconds / max(count, 1):8.2f} ms per candidate "
                  f"({count:,} candidates)")

    if args.global_step is not None:
        step = args.global_step
        card = read_card(project / f"cards/ite_{ite:04d}/card_iteref_ref_{channels[0]:03d}"
                         f"_ite_{ite:04d}.card")
        base = _settings(card, box)
        settings = core.SearchSettings(
            cone_aperture=360, cone_sampling=step, inplane_range=360, inplane_sampling=step,
            refine=2, refine_factor=2, high=base.high,
            low=args.global_low if args.global_low is not None else base.low,
            shift_limit=base.shift_limit, area_search_modus=base.area_search_modus,
        )
        reference = read_em(starting / f"starting_reference_transformed_ref_{channels[0]:03d}"
                            f"_ite_{ite:04d}.em").astype(float)
        assert sample is not None
        tag = int(sample[0])
        row = {int(r[0]): r for r in read_table(
            starting / f"starting_table_ref_{channels[0]:03d}_ite_{ite:04d}.tbl")}[tag]
        began = time.perf_counter()
        result = kernel.refine_particle_torch(volumes[tag], reference, mask, row[6:9], row[3:6],
                                              (float(row[13]), float(row[14])), settings,
                                              device=device, chunk=args.chunk)
        if device.type == "cuda":
            torch.cuda.synchronize(device)
        seconds = time.perf_counter() - began
        count = _candidates(result)
        per = seconds / count
        total = len(read_table(starting / f"starting_table_ref_{channels[0]:03d}_ite_"
                               f"{ite:04d}.tbl")) * len(channels)
        print(f"360 deg at {step:g} deg, refined twice, low {settings.low:g} px, box {box}: "
              f"{count:,} candidates for one particle in {seconds:.1f} s, "
              f"{1000 * per:.2f} ms each; for {total:,} particle-channel searches about "
              f"{per * count * total / 3600:.1f} h on this one device")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
