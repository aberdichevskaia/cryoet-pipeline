#!/usr/bin/env python3
"""The torch search kernel on real particles, against Dynamo's own search and the numpy port.

    python scripts/torch_search_check.py outputs/ryr1/dynamo/bin8-2x-pad/mra_pad_ssn_g2atan \\
        --iteration 5 --particles 16 --device cuda --global-step 9.0 --global-low 13

For one finished iteration of a Dynamo project searched at the full box (dim equal
to the box: a cone round, or a global round set up at the full box, not one that
Dynamo's CPU downsampled), and a seeded sample of its particles, every channel's
search is run three ways from the iteration's own
inputs: its starting table, Dynamo's prepared reference
(starting_reference_transformed) and search mask (mask_transformed), under the
iteration's card. Dynamo's answer is the refined table; the numpy kernel
(dynamo_search.refine_particle, float64 on the CPU) is the port the replays
checked; the torch kernel (dynamo_search_torch, float32 on --device) is the one
to check. It prints, for each, the angle (deg) and shift (px) from Dynamo's pose
and the CC difference, per channel the median and the largest, and each kernel's
time per candidate orientation.

A global round set up at the full box (dynamo_mra_setup.m's global_full_box, on a
small export of scripts/sample_export.py) is checked the same way, every particle
with --particles 0. There the numpy kernel takes minutes per search, so
--skip-numpy compares the torch kernel with Dynamo directly: on cone rounds the two
kernels gave the same poses (2026-10-26, A40).

Every search that ends more than 0.1 deg from Dynamo's pose is listed with three
CCs: ours, Dynamo's, and ours at Dynamo's orientation (its shift searched again).
All three alike is a near-tie, two poses that score the same (a C1 search of a
near-symmetric particle such as RyR1 has one per symmetry copy). Ours above the
other two: Dynamo's search stopped short of the higher peak. Dynamo's above ours
at its own orientation: a CC our scoring does not reproduce, as Dynamo's spline
gave 1.00000 on the Mac toy (2026-10-27). Ours below ours at Dynamo's orientation:
our search missed a peak it scores, which would be the port's fault.

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
from dataclasses import replace
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
    parser.add_argument("--particles", type=int, default=16,
                        help="sample size; 0 for every particle")
    parser.add_argument("--skip-numpy", action="store_true",
                        help="compare the torch kernel with Dynamo directly, without numpy")
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
            size = len(tags) if args.particles == 0 else min(args.particles, len(tags))
            sample = np.sort(rng.choice(tags, size=size, replace=False))
            volumes = {int(tag): _particle(data, int(tag)) for tag in sample}
        starts = {int(row[0]): row for row in start}
        dynamo = {int(row[0]): row for row in refined}
        rows = []
        apart = []
        at_their_orientation = replace(settings, cone_aperture=0, inplane_range=0, refine=0)
        times = {"numpy": [0.0, 0], "torch": [0.0, 0]}
        for tag in sample:
            row = starts[int(tag)]
            inputs = (volumes[int(tag)], reference, mask, row[6:9], row[3:6],
                      (float(row[13]), float(row[14])), settings)
            ours = None
            if not args.skip_numpy:
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
            first = fast if ours is None else ours
            between = (np.nan,) * 3 if ours is None else (
                _angle(fast.angles, ours.angles), float(np.linalg.norm(fast.shift - ours.shift)),
                fast.correlation - ours.correlation)
            rows.append((
                _angle(first.angles, truth[6:9]), float(np.linalg.norm(first.shift - truth[3:6])),
                first.correlation - truth[9], *between,
            ))
            if rows[-1][0] > 0.1:
                theirs = kernel.refine_particle_torch(
                    inputs[0], reference, mask, truth[6:9], truth[3:6], inputs[5],
                    at_their_orientation, device=device, chunk=args.chunk)
                apart.append((int(tag), rows[-1][0], rows[-1][1], first.correlation, truth[9],
                              theirs.correlation))
        rows_by_channel[channel] = rows
        table = np.array(rows)
        print(f"channel {channel}: {settings.cone_aperture:g} deg cone in "
              f"{settings.cone_sampling:g} deg steps, refine {settings.refine}, low "
              f"{settings.low:g} px, {len(sample)} particles")
        comparisons = ((("torch vs Dynamo", (0, 1, 2)),) if args.skip_numpy
                       else (("numpy vs Dynamo", (0, 1, 2)), ("torch vs numpy", (3, 4, 5))))
        for name, columns in comparisons:
            angle, shift, cc = (table[:, column] for column in columns)
            print(f"  {name:16s} angle median {np.median(angle):.3f}, max {angle.max():.3f} "
                  f"deg, {int(np.sum(angle > 0.1))} over 0.1; shift median {np.median(shift):.3f}, "
                  f"max {shift.max():.3f} px; CC difference max {np.abs(cc).max():.2e}")
        for tag, angle, shift, ours, dynamo_cc, at_theirs in apart:
            print(f"  tag {tag}: {angle:.2f} deg and {shift:.2f} px from Dynamo's pose; CC ours "
                  f"{ours:.5f}, Dynamo's {dynamo_cc:.5f}, ours at its orientation {at_theirs:.5f}")
        for name, (seconds, count) in times.items():
            if count:
                print(f"  {name:5s} {1000 * seconds / count:8.2f} ms per candidate "
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
