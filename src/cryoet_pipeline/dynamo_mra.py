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

Scope, refused rather than guessed: the classification mask equal to the
alignment mask (so the CC compared is the search's own, table column 10), the
search kernel's own scope (`use_CC = 1`, `localnc = 1`, C1, no resizing),
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

from collections.abc import Mapping, Sequence
from dataclasses import dataclass

import numpy as np

from cryoet_pipeline.backends.dynamo_search import (
    DEFAULT_SETTINGS,
    ClassAverage,
    SearchSettings,
    average_class,
    fweight_divide,
    prepare_reference,
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
) -> MraIteration:
    """One iteration: every live channel searches every particle, then they swap.

    `mask` is the project's soft mask, used as Dynamo uses it: thresholded at
    0.5 for the search and the reference normalization, its nonzero voxels for
    the class averages' normalization.
    """
    if set(references) != set(starts):
        raise ValueError("every live channel needs a reference and starting poses")
    binary = (np.asarray(mask) > 0.5).astype(float)
    computed = sorted(references)
    searches = {}
    for k in computed:
        prepared = prepare_reference(references[k], mask, settings)
        angles, shifts = (np.asarray(v, dtype=float) for v in starts[k])
        results = search_particles(
            particles,
            prepared,
            binary,
            list(zip(angles, shifts, strict=True)),
            tilt_ranges,
            settings,
            workers=workers,
        )
        searches[k] = ChannelSearch(
            np.array([r[0] for r in results]),
            np.array([r[1] for r in results]),
            np.array([r[2] for r in results]),
        )
    assignment = assign({k: searches[k].correlations for k in computed})
    surviving = sorted({int(k) for k in assignment})
    averages = {}
    for k in surviving:
        members = np.flatnonzero(assignment == k)
        averages[k] = average_class(
            [particles[i] for i in members],
            searches[k].angles[members],
            searches[k].shifts[members],
            [tilt_ranges[i] for i in members],
            mask,
            fmin=fmin,
        )
    return MraIteration(computed, searches, assignment, surviving, averages)


def run_mra(
    particles: Sequence[np.ndarray],
    references: Mapping[int, np.ndarray],
    angles: np.ndarray,
    shifts: np.ndarray,
    tilt_ranges: Sequence[tuple[float, float]],
    mask,
    settings: SearchSettings = DEFAULT_SETTINGS,
    *,
    iterations: int,
    fmin: int = 1,
    workers: int = 1,
) -> list[MraIteration]:
    """`iterations` MRA iterations from one starting table shared by every channel.

    Each channel carries its own poses for every particle into the next
    iteration, members or not, as each channel's refined table becomes its next
    starting table in Dynamo; its class average becomes its next reference.
    """
    if iterations < 1:
        raise ValueError("at least one iteration")
    starts = {k: (np.asarray(angles, float), np.asarray(shifts, float)) for k in references}
    references = dict(references)
    history = []
    for _ in range(iterations):
        step = mra_iteration(
            particles, references, starts, tilt_ranges, mask, settings, fmin=fmin, workers=workers
        )
        history.append(step)
        references = {k: step.averages[k].average for k in step.surviving}
        starts = {k: (step.searches[k].angles, step.searches[k].shifts) for k in step.surviving}
    return history
