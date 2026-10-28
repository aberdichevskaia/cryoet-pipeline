"""Dynamo 1.1.511's multireference alignment (MRA), ported for replay before integration.

Standalone, as the search kernel was before it became `backends/dynamo_search`:
nothing in the pipeline calls this yet. It is to be checked against a real
Dynamo MRA project first (benchmark `DYNAMO_MRA.md`), then wired in.

What one iteration does, read from Dynamo's CPU project path
(`+dpkmulticore/run.m`, `+dpkproject/preprocessReferences.m`,
`dynamo_plugin_post_multireference_tutorial.m`, `dynamo_average.m`):

1. Every live channel's starting reference is prepared as the iteration setup
   prepares it: the project bandpass, then normalization over the alignment mask
   thresholded at 0.5 (the kernel's `prepare_reference`).
2. Every live channel searches every particle, each channel from its own poses
   of the previous iteration, with the ordinary search (`search_particles`).
3. Every particle goes to the channel with the highest search CC, the lowest
   channel number on a tie (`assign`).
4. Every channel that received particles averages them at its own poses the
   way the MRA plugin calls `dynamo_average` (`average_class`); that average is
   the channel's next reference. A channel that received none dies.

A project without MRA (`mra = 0`: one channel, as a map seed makes it) has no
plugin to average: the project averages its one channel itself, as every
single-reference project does (`average_particles`, the `fsc = true` assembly of
odd and even halves), and that average, not normalized again, is the next
reference (`assembly="project"`; on a Dynamo 1.1.511 toy of this kind, 4.7e-6
of the average's std from Dynamo's own, where the plugin's way is 2 std off).

Scope, refused rather than guessed: the classification mask equal to the
alignment mask (so the CC compared is the search's own, table column 10), the
search kernel's own scope (`use_CC = 1`, `localnc = 1`, C1; a smaller `dim` by
interpolation, as `mra_iteration` makes it, not by binning),
no thresholding, `fmin` counting particles (no `fminFraction`), no compensation
smoothing, references updated from the weighted average. Dynamo can also stop
early when every reference saturates; that check is not ported.

Numerical adaptation of the Dynamo routines named above. The BSD option of the
following original joint notice is selected for these adaptations; this does
not relicense any other Dynamo component.

Author: Daniel Castano-Diez, April 2012 (daniel.castano@unibas.ch)
Copyright (c) 2012 Daniel Castano-Diez and Henning Stahlberg
Center for Cellular Imaging and Nano Analytics
Biocenter, University of Basel

This software is issued under a joint BSD/GNU license. You may use the
source code in this file under either license. However, note that the
complete Dynamo software packages have some GPL dependencies,
so you are responsible for compliance with the licenses of these packages
if you opt to use BSD licensing. The warranty disclaimer below holds
in either instance.

This complete copyright notice must be included in any revised version of the
source code. Additional authorship citations may be added, but existing
author citations must be preserved.

This program is free software; you can redistribute it and/or modify
it under the terms of the GNU General Public License as published by
the Free Software Foundation; either version 2 of the License, or
(at your option) any later version.

This program is distributed in the hope that it will be useful,
but WITHOUT ANY WARRANTY; without even the implied warranty of
MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the
GNU General Public License for more details.

You should have received a copy of the GNU General Public License
along with this program; if not, write to the Free Software
Foundation, Inc., 59 Temple Place, Suite 330, Boston, MA  2111-1307 USA
"""

from __future__ import annotations

from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass

import numpy as np

from cryoet_pipeline.backends.dynamo_search import (
    DEFAULT_SETTINGS,
    FOURIER_WEIGHT_THRESHOLD,
    ClassAverage,
    SearchSettings,
    average_class,
    average_particles,
    downsized,
    expanded_shift,
    fweight_divide,
    prepare_reference,
    resize_box,
    resize_mask,
    search_particles,
)

__all__ = [
    "ChannelSearch",
    "ClassAverage",
    "MraIteration",
    "assign",
    "average_class",
    "fweight_divide",
    "mra_iteration",
    "run_mra",
]


@dataclass(frozen=True)
class ChannelSearch:
    angles: np.ndarray
    shifts: np.ndarray
    correlations: np.ndarray


@dataclass(frozen=True)
class MraIteration:
    computed: list[int]
    searches: dict[int, ChannelSearch]
    assignment: np.ndarray
    surviving: list[int]
    averages: dict[int, ClassAverage]


def assign(correlations: Mapping[int, np.ndarray]) -> np.ndarray:
    """Each particle's channel: the highest CC, the lowest channel number on a tie.

    The plugin fills a score row over the channels in order and takes MATLAB's
    `max`, whose first maximum is the lowest channel; `argmax` over the sorted
    channels is that.
    """
    channels = sorted(correlations)
    if not channels:
        raise ValueError("no live channel to assign particles to")
    scores = np.stack([np.asarray(correlations[k], dtype=float) for k in channels])
    if not np.isfinite(scores).all():
        raise ValueError("every live channel needs a finite CC for every particle")
    return np.asarray(channels)[np.argmax(scores, axis=0)]


def mra_iteration(
    particles: Sequence[np.ndarray],
    references: Mapping[int, np.ndarray],
    starts: Mapping[int, tuple[np.ndarray, np.ndarray]],
    tilt_ranges: Sequence[tuple[float, float]],
    mask,
    settings: SearchSettings = DEFAULT_SETTINGS,
    *,
    fmin: int = 1,
    workers: int = 1,
    search: Callable[..., list] = search_particles,
    assembly: str = "mra",
) -> MraIteration:
    """One iteration: every live channel searches every particle, then they swap.

    `mask` is the project's soft mask, used as Dynamo uses it: thresholded at
    0.5 for the search and the reference normalization, its nonzero voxels for
    the class averages' normalization. `search` is the kernel, called as
    `search_particles` is: the numpy one, or the torch one bound to a device
    (`dynamo_search_torch.search_particles_torch`). `assembly` is "mra", the
    plugin's class averages, or "project": the one channel of a project without
    MRA, averaged as the project averages it.

    With `settings.dim` below the box the search runs at that box, as the iteration
    setup and `pipeline_align_one_particle` make it: every particle and reference
    resized (`resize_box`), the mask resized and thresholded (`resize_mask`), the
    band and shift limit scaled (`downsized`), the starting shifts scaled down and
    the found ones back (`expanded_shift`). The classes are averaged from the
    particles at their own box, as the plugin averages them.
    """
    if assembly not in ("mra", "project"):
        raise ValueError(f"assembly is 'mra' or 'project', got {assembly!r}")
    if set(references) != set(starts):
        raise ValueError("every live channel needs a reference and starting poses")
    if assembly == "project" and len(references) != 1:
        raise ValueError(f"a project without MRA has one channel, not {len(references)}")
    box = np.asarray(mask).shape[0]
    small = settings.dim if settings.dim not in (0, box) else 0
    if small:
        inner = downsized(settings, box)
        searched = [resize_box(particle, small) for particle in particles]
        binary = resize_mask(mask, small)
    else:
        inner, searched = settings, particles
        binary = (np.asarray(mask) > 0.5).astype(float)
    computed = sorted(references)
    searches = {}
    for k in computed:
        reference = resize_box(references[k], small) if small else references[k]
        prepared = prepare_reference(reference, binary, inner)
        angles, shifts = (np.asarray(v, dtype=float) for v in starts[k])
        if small:
            shifts = (small / box) * shifts
        results = search(
            searched,
            prepared,
            binary,
            list(zip(angles, shifts, strict=True)),
            tilt_ranges,
            inner,
            workers=workers,
        )
        found = np.array([r[0] for r in results])
        moved = np.array([r[1] for r in results])
        if small:
            moved = np.array([expanded_shift(a, s, box, small)
                              for a, s in zip(found, moved, strict=True)])
        searches[k] = ChannelSearch(found, moved, np.array([r[2] for r in results]))
    assignment = assign({k: searches[k].correlations for k in computed})
    surviving = sorted({int(k) for k in assignment})
    averages = {}
    for k in surviving:
        members = np.flatnonzero(assignment == k)
        group = ([particles[i] for i in members], searches[k].angles[members],
                 searches[k].shifts[members], [tilt_ranges[i] for i in members])
        averages[k] = (_project_average(*group, mask) if assembly == "project"
                       else average_class(*group, mask, fmin=fmin))
    return MraIteration(computed, searches, assignment, surviving, averages)


def _project_average(particles, angles, shifts, tilt_ranges, mask) -> ClassAverage:
    """`average_particles`, the project's own averaging, as a `ClassAverage`."""

    average, raw, weights = average_particles(particles, angles, shifts, tilt_ranges, mask)
    return ClassAverage(average, raw, weights,
                        (weights >= FOURIER_WEIGHT_THRESHOLD).astype(float),
                        FOURIER_WEIGHT_THRESHOLD, len(particles))


def run_mra(
    particles: Sequence[np.ndarray],
    references: Mapping[int, np.ndarray],
    angles: np.ndarray,
    shifts: np.ndarray,
    tilt_ranges: Sequence[tuple[float, float]],
    mask,
    settings: SearchSettings | Sequence[SearchSettings] = DEFAULT_SETTINGS,
    *,
    iterations: int,
    fmin: int = 1,
    workers: int = 1,
    search: Callable[..., list] = search_particles,
) -> list[MraIteration]:
    """`iterations` MRA iterations from one starting table shared by every channel.

    Each channel carries its own poses for every particle into the next
    iteration, members or not, as each channel's refined table becomes its next
    starting table in Dynamo; its class average becomes its next reference.
    `settings` is one for every iteration, or one per iteration: a schedule of
    rounds of one iteration each, as the coarse-to-fine setup makes.
    """
    if iterations < 1:
        raise ValueError("at least one iteration")
    schedule = ([settings] * iterations if isinstance(settings, SearchSettings)
                else list(settings))
    if len(schedule) != iterations:
        raise ValueError(f"{len(schedule)} settings for {iterations} iterations")
    starts = {k: (np.asarray(angles, float), np.asarray(shifts, float)) for k in references}
    references = dict(references)
    history = []
    for current in schedule:
        step = mra_iteration(
            particles, references, starts, tilt_ranges, mask, current, fmin=fmin,
            workers=workers, search=search,
        )
        history.append(step)
        references = {k: step.averages[k].average for k in step.surviving}
        starts = {k: (step.searches[k].angles, step.searches[k].shifts) for k in step.surviving}
    return history
