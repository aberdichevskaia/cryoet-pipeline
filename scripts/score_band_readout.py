"""Does a pytom score map see the specimen? One line per map.

    python scripts/score_band_readout.py TOMOGRAM.(zarr|mrc) SCORES.mrc [SCORES2.mrc ...]

The specimen band is read off the tomogram itself: the per-slice standard
deviation of a lightly smoothed copy, thresholded halfway between its floor
(vacuum) and its plateau (specimen). A map that sees the specimen puts its best
peaks inside that band. A map that does not spreads them through the volume like
noise, in proportion to the band's share of the searchable depth.

Each map's job JSON is looked for beside it (`..._scores.mrc` -> `..._job.json`);
when found, the best peak is also given in units of the search's own sigma, the
scale the pipeline report uses, with the number of separated peaks above the
one-false-positive height for that search.

Measured on local copies (2026-10-15): the old pipeline map of TS_0005 put 44%
of its top 100 in the band where noise gives 48%; a plain normalised
cross-correlation with the same template on the same volume put 100% there.
"""

import json
import math
import sys
from pathlib import Path

import mrcfile
import numpy as np
from scipy import ndimage
from scipy.special import erfcinv

EDGE_PX = 20
"""Peaks this close to a face are skipped: a template box cannot be centred there."""

PEAK_SEPARATION_PX = 15
"""About a ribosome radius at bin8, as pytom's own extraction separates them."""


def read_volume(path: str) -> np.ndarray:
    if path.rstrip("/").endswith(".zarr"):
        import zarr

        return np.asarray(zarr.open(path, mode="r")[:], np.float32)
    with mrcfile.open(path, permissive=True) as mrc:
        return np.asarray(mrc.data, np.float32)


def specimen_band(volume: np.ndarray) -> np.ndarray:
    smoothed = ndimage.gaussian_filter(volume, 2.0)
    profile = smoothed.reshape(volume.shape[0], -1).std(axis=1)
    floor, plateau = np.percentile(profile, 10), np.percentile(profile, 95)
    return profile > (floor + plateau) / 2


def search_statistics(scores_path: Path) -> tuple[float | None, float | None]:
    job = scores_path.with_name(scores_path.name.replace("_scores.mrc", "_job.json"))
    if not job.is_file():
        return None, None
    stats = json.loads(job.read_text()).get("job_stats") or {}
    sigma, search_space = stats.get("std"), stats.get("search_space")
    one_fp = math.sqrt(2) * erfcinv(2.0 / search_space) if search_space else None
    return sigma, one_fp


def main() -> None:
    tomogram_path, *score_paths = sys.argv[1:]
    volume = read_volume(tomogram_path)
    band = specimen_band(volume)
    rows = np.flatnonzero(band)
    print(f"specimen band from the tomogram: z {rows.min()}..{rows.max()} "
          f"({band.sum()} of {band.size} slices)")
    searchable = band[EDGE_PX:-EDGE_PX].mean()
    for name in score_paths:
        path = Path(name)
        scores = read_volume(name)
        if scores.shape != volume.shape:
            print(f"{path.name}: shape {scores.shape} is not the tomogram's {volume.shape}")
            continue
        valid = np.zeros(scores.shape, bool)
        valid[EDGE_PX:-EDGE_PX, EDGE_PX:-EDGE_PX, EDGE_PX:-EDGE_PX] = True
        peaks = (ndimage.maximum_filter(scores, size=PEAK_SEPARATION_PX) == scores) & valid
        coords = np.argwhere(peaks)
        values = scores[tuple(coords.T)]
        order = np.argsort(-values)
        inside = band[coords[order[:100], 0]].mean()
        line = (f"{path.name}: top-100 peaks inside the specimen band {inside:.0%} "
                f"(noise would give {searchable:.0%})")
        sigma, one_fp = search_statistics(path)
        if sigma:
            line += f"; best {values[order[0]] / sigma:.2f} sigma"
            if one_fp is not None:
                above = int((values / sigma >= one_fp).sum())
                line += f"; {above} peaks above the one-false-positive height {one_fp:.2f}"
        print(line)


if __name__ == "__main__":
    main()
