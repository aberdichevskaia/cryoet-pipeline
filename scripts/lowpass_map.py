#!/usr/bin/env python3
"""Low-pass an average to the resolution its data actually supports.

    python3 scripts/lowpass_map.py ~/Downloads/*top800-cls4-s1_class0*.mrc --angstrom 35

Averages are written unfiltered, out to the Nyquist limit of the particle pixel
spacing — 19.6 A at bin8. The tilt-series alignment on this data supports 31 to
41 A (residuals 0.69 to 0.92 nm through `resolution.py`), so roughly the finer
half of the frequency range in a written map carries no signal. Displayed at a
surface threshold that noise reads as a crust of specks over whatever body is
there, which is a display artefact and not a property of the average.

So this exists to separate two questions that look identical in ChimeraX:

  does the refinement produce a body?        -> look at the map filtered to 35 A
  does it produce detail finer than that?    -> it cannot, and neither can any
                                                refinement, until the tilt-series
                                                alignment improves

The filter is the pipeline's own `low_pass_filter_volume`, a raised-cosine edge
at a stated cutoff, rather than ChimeraX's `volume gaussian` — a Gaussian has no
cutoff, so "sDev 12" is not a resolution and two people will not mean the same
thing by it. Here the number on the command line is the number in the map.

Writes `<name>_lp<angstrom>.mrc` beside each input, so the unfiltered original
stays exactly as the pipeline wrote it.

Disposable, like the rest of `scripts/`. Its replacement is `average-particles`
writing a filtered map alongside the raw one, at the limit its own QC already
derives, so that the map a person opens first is the one that can be believed.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import torch

from cryoet_pipeline.filters import low_pass_filter_volume


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("maps", nargs="+", type=Path, help="Average MRC files.")
    parser.add_argument(
        "--angstrom",
        type=float,
        required=True,
        help="Cutoff. 35 is what the alignment residuals on EMPIAR-11538 support; "
        "see resolution.py for the derivation from a per-series residual.",
    )
    parser.add_argument(
        "--pixel-spacing-angstrom",
        type=float,
        default=None,
        help="Overrides the MRC header, which is normally what should be trusted.",
    )
    args = parser.parse_args()

    import mrcfile

    for path in args.maps:
        with mrcfile.open(path, permissive=True) as handle:
            volume = np.asarray(handle.data, dtype=np.float32)
            header_apix = float(handle.voxel_size.x) or 0.0
        apix = args.pixel_spacing_angstrom or header_apix
        if not apix:
            print(f"{path.name}: no pixel spacing in the header; pass one")
            continue
        if volume.ndim != 3:
            print(f"{path.name}: not a 3D map ({volume.shape})")
            continue
        nyquist = 2.0 * apix
        if args.angstrom < nyquist:
            print(
                f"{path.name}: {args.angstrom:.1f} A is finer than the Nyquist limit "
                f"of {nyquist:.1f} A at {apix:.3f} A/voxel, so the filter would pass "
                "the whole map unchanged"
            )
            continue

        filtered = low_pass_filter_volume(
            torch.as_tensor(volume),
            cutoff_angstrom=args.angstrom,
            pixel_spacing_angstrom=apix,
        ).numpy()

        out = path.with_name(f"{path.stem}_lp{args.angstrom:.0f}.mrc")
        with mrcfile.new(out, overwrite=True) as handle:
            handle.set_data(filtered.astype(np.float32))
            handle.voxel_size = apix
        # The standard deviation is what a surface threshold is quoted against,
        # so the drop is the useful number: it says how much of what was on
        # screen was above the supported limit.
        print(
            f"{out.name}: sd {volume.std():.3f} -> {filtered.std():.3f} "
            f"({100 * (1 - filtered.std() / volume.std()):.0f}% of the amplitude was "
            f"finer than {args.angstrom:.0f} A)"
        )

    print()
    print("Open these next to the unfiltered ones at the same sdLevel. sdLevel is")
    print("relative to each map's own spread, so a filtered map is thresholded on")
    print("what survived rather than on the noise that was removed.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
