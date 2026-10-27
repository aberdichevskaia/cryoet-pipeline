#!/usr/bin/env python3
"""Where the torch search kernel's time goes on one device, on a Dynamo project's own searches.

    python scripts/torch_search_profile.py outputs/ryr1/dynamo/bin8-2x-pad/mra_pad_ssn_cone5 \\
        --iterations 1 3 --particles 32 --device cuda --processes 1 2 4

The port's cone rounds took as long with 221 candidates per search as with 300 (cone5
on two A40s, 2026-10-27: 26-27 min an iteration, about 0.45 s a search), so the time
is not in the candidates. For each iteration (a finished one searched at the full box,
as scripts/torch_search_check.py takes them) a seeded sample of particles is searched
in every channel from the iteration's own inputs, four ways:

1. the kernel as the port runs it, in this process: seconds per search and per
   candidate;
2. the same searches with each step of the kernel timed between device
   synchronizations: the kernel's code is unchanged, its functions are wrapped in
   this process only. Steps: the grid on the host (angle_list, rotation matrices),
   the particle's filters on the host and on the device, the mask's support, the
   rotation, the candidates' filter, the correlation, the peak; the rest of the
   search is "other". The synchronizations take away a little overlap, so the steps
   add up to a little more than 1;
3. on CUDA, torch.profiler over one search per channel: how long the GPU itself
   computes per search, against 1's wall clock, and which kernels take it;
4. the kernel in each of --processes processes sharing the one device, as
   MultiDeviceSearch deals particles (each count a fresh pool, timed on its second
   pass over the searches): whether several processes per GPU fill it.

2 and 4 must give 1's poses and CCs bit for bit; each says whether it did.
Nothing is written. Disposable, like the rest of scripts/.
"""

from __future__ import annotations

import argparse
import os
import time
import warnings
from collections import defaultdict
from pathlib import Path

import numpy as np
import torch
import torch_search_check as check

from cryoet_pipeline.backends import dynamo_search_torch as kernel
from cryoet_pipeline.reports.dynamo_project import read_card, read_em, read_table

# The kernel's functions timed in 2, each under its step's name.
STEPS = {
    "angle_list": "grid (host)",
    "_matrices": "grid (host)",
    "bandpass": "particle: host filters",
    "wedge": "particle: host filters",
    "half_filter": "particle: host filters",
    "cell_support": "mask support",
    "support_reach": "mask support",
    "resample_batch": "rotate",
    "_apply_half_filter": None,  # the particle's (one volume) or the candidates' (a batch)
    "local_correlation_batch": "correlate",
    "restricted_peaks": "peak",
}
ORDER = ("grid (host)", "particle: host filters", "particle: device filters", "mask support",
         "rotate", "filter candidates", "correlate", "peak", "other")


def _synchronize(device: torch.device) -> None:
    if device.type == "cuda":
        torch.cuda.synchronize(device)
    elif device.type == "mps":
        torch.mps.synchronize()


class _Steps:
    """Wall time per step, each call bracketed by device synchronizations.

    A wrapped call inside another counts only toward its own step.
    """

    def __init__(self, device: torch.device) -> None:
        self.device = device
        self.seconds: dict[str, float] = defaultdict(float)
        self._open: list[float] = []

    def wrap(self, function, step):
        def timed(*args, **kwargs):
            name = step or ("particle: device filters" if args[0].dim() == 3
                            else "filter candidates")
            _synchronize(self.device)
            self._open.append(0.0)
            began = time.perf_counter()
            try:
                return function(*args, **kwargs)
            finally:
                _synchronize(self.device)
                elapsed = time.perf_counter() - began
                self.seconds[name] += elapsed - self._open.pop()
                if self._open:
                    self._open[-1] += elapsed

        return timed


def _inputs(project: Path, ite: int, particles: int, seed: int):
    """The iteration's channels, mask, and per channel its settings, prepared reference
    and starting rows; a seeded sample of tags and their volumes."""

    starting = project / check.STARTING.format(ite=ite)
    searched = (project / f"results/ite_{ite:04d}/currently_surviving_references_ite_"
                f"{ite:04d}.txt")
    channels = [int(value) for value in searched.read_text().split()]
    mask = read_em(starting / f"mask_transformed_ite_{ite:04d}.em").astype(float)
    box = mask.shape[0]
    cards = {channel: read_card(project / f"cards/ite_{ite:04d}/card_iteref_ref_{channel:03d}"
                                f"_ite_{ite:04d}.card") for channel in channels}
    data = project.parent / cards[channels[0]].get("folder_data", "particles")
    settings, references, rows = {}, {}, {}
    for channel in channels:
        settings[channel] = check._settings(cards[channel], box)
        references[channel] = read_em(starting / f"starting_reference_transformed_ref_"
                                      f"{channel:03d}_ite_{ite:04d}.em").astype(float)
        table = read_table(starting / f"starting_table_ref_{channel:03d}_ite_{ite:04d}.tbl")
        rows[channel] = {int(row[0]): row for row in table}
    tags = np.sort(np.array(list(rows[channels[0]]), dtype=int))
    size = len(tags) if particles == 0 else min(particles, len(tags))
    sample = [int(tag) for tag in np.sort(np.random.default_rng(seed).choice(tags, size=size,
                                                                            replace=False))]
    volumes = {tag: check._particle(data, tag) for tag in sample}
    return channels, mask, settings, references, rows, sample, volumes


def _kernel_time(event) -> float:
    return float(getattr(event, "self_device_time_total", 0) or 0)


def _gpu_time(run, searches: int) -> str:
    """3: what the GPU computes per search, and its busiest kernels."""

    from torch.autograd import DeviceType
    from torch.profiler import ProfilerActivity, profile

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        with profile(activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA]) as prof:
            run()
        events = prof.key_averages()
    kernels = [event for event in events if event.device_type == DeviceType.CUDA]
    if not kernels:  # no kernel rows: their time is on the operators that launched them
        kernels = [event for event in events if event.device_type == DeviceType.CPU]
    kernels.sort(key=_kernel_time, reverse=True)
    total = sum(_kernel_time(event) for event in kernels) / 1e3 / searches
    lines = [f"{total:.1f} ms of GPU computing per search"]
    for event in kernels[:8]:
        lines.append(f"       {_kernel_time(event) / 1e3 / searches:7.2f} ms  "
                     f"{event.count // searches:4d}x  {event.key[:90]}")
    return "\n".join(lines)


def _same(first, second) -> bool:
    return all(np.array_equal(a.angles, b[0]) and np.array_equal(a.shift, b[1])
               and a.correlation == b[2] for a, b in zip(first, second, strict=True))


def _profile_iteration(project: Path, ite: int, args: argparse.Namespace,
                       device: torch.device) -> None:
    channels, mask, settings, references, rows, sample, volumes = _inputs(
        project, ite, args.particles, args.seed)
    searches = [(channel, tag) for channel in channels for tag in sample]

    def search(channel: int, tag: int):
        row = rows[channel][tag]
        return kernel.refine_particle_torch(
            volumes[tag], references[channel], mask, row[6:9], row[3:6],
            (float(row[13]), float(row[14])), settings[channel], device=device,
            chunk=args.chunk)

    def run(pairs=searches):
        results = [search(channel, tag) for channel, tag in pairs]
        _synchronize(device)
        return results

    run([(channel, sample[0]) for channel in channels])  # kernels loaded, FFT plans made
    began = time.perf_counter()
    plain = run()
    seconds = (time.perf_counter() - began) / len(searches)
    counts = [check._candidates(result) for result in plain]
    levels = ", ".join(str(level["candidate_count"]) for level in plain[0].levels)
    first = settings[channels[0]]
    print(f"{project.name}, iteration {ite}: channels {channels}, {len(sample)} particles "
          f"each ({len(searches)} searches), cone {first.cone_aperture:g} deg in "
          f"{first.cone_sampling:g} deg steps, refine {first.refine}, {np.mean(counts):.0f} "
          f"candidates per search (levels {levels} for the first), box {mask.shape[0]}")
    print(f"  1. kernel, one process: {seconds:.3f} s per search, "
          f"{1000 * seconds * len(searches) / sum(counts):.2f} ms per candidate")

    steps = _Steps(device)
    originals = {name: getattr(kernel, name) for name in STEPS}
    for name, step in STEPS.items():
        setattr(kernel, name, steps.wrap(originals[name], step))
    try:
        began = time.perf_counter()
        staged = run()
        total = time.perf_counter() - began
    finally:
        for name, function in originals.items():
            setattr(kernel, name, function)
    steps.seconds["other"] = total - sum(steps.seconds.values())
    print(f"  2. steps between synchronizations ({total / len(searches):.3f} s per search):")
    for name in ORDER:
        share = steps.seconds.get(name, 0.0)
        print(f"       {name:26s} {1000 * share / len(searches):8.1f} ms  "
              f"{100 * share / total:5.1f}%")
    print("     poses and CCs equal to 1 bit for bit: "
          + ("yes" if _same(plain, [(r.angles, r.shift, r.correlation) for r in staged])
             else "NO"))

    if device.type == "cuda":
        pairs = [(channel, sample[0]) for channel in channels]
        try:
            report = _gpu_time(lambda: run(pairs), len(pairs))
        except Exception as error:  # the profiler's API moves between torch versions
            report = f"profiler unavailable: {type(error).__name__}: {error}"
        print(f"  3. {report}")
        cache = torch.backends.cuda.cufft_plan_cache[device.index or 0]
        print(f"     cuFFT plans cached: {cache.size} of {cache.max_size}")

    name = f"{device.type}:{device.index or 0}" if device.type == "cuda" else device.type
    particles = [volumes[tag] for tag in sample]

    def dealt_by(pool) -> list:
        results = []
        for channel in channels:
            results += pool(particles, references[channel], mask,
                            [(rows[channel][tag][6:9], rows[channel][tag][3:6]) for tag in sample],
                            [(float(rows[channel][tag][13]), float(rows[channel][tag][14]))
                             for tag in sample], settings[channel])
        return results

    lines = []
    for count in args.processes:
        pool = kernel.MultiDeviceSearch([name] * count, chunk=args.chunk)
        try:
            dealt_by(pool)  # starts the processes, loads their kernels, makes their FFT plans
            began = time.perf_counter()
            dealt = dealt_by(pool)
            per = (time.perf_counter() - began) / len(searches)
        finally:
            pool.close()
        lines.append(f"{count}: {per:.3f} s per search ({seconds / per:.2f}x 1), "
                     f"{'same' if _same(plain, dealt) else 'NOT the same'} bit for bit")
    if lines:
        print(f"  4. processes sharing {name}: " + "; ".join(lines))


def main() -> int:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("project", type=Path, help="a Dynamo project with finished iterations")
    parser.add_argument("--iterations", type=int, nargs="+", required=True,
                        help="finished iterations searched at the full box")
    parser.add_argument("--particles", type=int, default=32,
                        help="sample size per iteration; 0 for every particle")
    parser.add_argument("--seed", type=int, default=20261027)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--chunk", type=int, default=kernel.DEFAULT_CHUNK,
                        help="candidates per batch")
    parser.add_argument("--processes", type=int, nargs="*", default=[1, 2, 4],
                        help="process counts to share the device among in 4; none to skip it")
    args = parser.parse_args()

    device = torch.device(args.device)
    if device.type == "cuda" and device.index is None:
        device = torch.device("cuda", torch.cuda.current_device())
    cpus = len(os.sched_getaffinity(0)) if hasattr(os, "sched_getaffinity") else os.cpu_count()
    what = (torch.cuda.get_device_name(device) if device.type == "cuda" else device.type)
    print(f"torch {torch.__version__}, CUDA {torch.version.cuda}, {what}; this process may use "
          f"{cpus} CPUs, torch {torch.get_num_threads()} threads; SLURM_CPUS_PER_TASK "
          f"{os.environ.get('SLURM_CPUS_PER_TASK', 'unset')}, OMP_NUM_THREADS "
          f"{os.environ.get('OMP_NUM_THREADS', 'unset')}")
    for ite in args.iterations:
        _profile_iteration(args.project.resolve(), ite, args, device)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
