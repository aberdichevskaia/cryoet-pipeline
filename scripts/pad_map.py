#!/usr/bin/env python3
"""Centre-pad a map to a target box, so one reference file serves every command.

    python3 scripts/pad_map.py \\
        --map outputs/ribo/templates/TS_0005/TS_0005_pytom_bin8_template.mrc \\
        --box 48

A prepared template is built at the size its own density needs — 39 voxels for
this ribosome — while particles are extracted at the box size picked for
extraction, 48. Two commands then disagree about whose job it is to reconcile
that:

  compute-fsc      pads the smaller map centrally and says so (cli.py:3805)
  average-particles refuses: "initial reference shape (39, 39, 39) does not
                    match particle box size (48, 48, 48)" (subtomogram.py:1616)

Both behaviours are defensible on their own and the pair is not: the same two
files are comparable for one command and not for the other. Fixing that belongs
in the pipeline, by giving `_load_reference_mrc` the same central padding with
the same justification. Until then this writes the padded file once.

Doing it as a file rather than inside each command is better anyway, for a
reason that matters here: `--initial-reference` and `--model-map` then point at
the SAME object. Otherwise the search would start from a 39-box reference while
the FSC compared against a padded 48-box one, and no reconciliation happening
anywhere is easier to trust than two independent reconciliations agreeing.

The padding matches `_pad_to_common_box` exactly — `pad // 2` before, the
remainder after, zeros — so a map padded here is what compute-fsc would have
produced internally, and a curve computed against it is comparable with curves
computed before this script existed.

Zeros because outside a template's own box there is no density. Note what that
means for a correlation search: the reference is compact density inside a shell
of zeros, which is the ordinary situation for an unmasked correlation against a
compact reference, and it is what template matching does too.

Disposable, like the rest of `scripts/`. Its replacement is the backend padding
its own reference.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--map", required=True, type=Path, help="MRC to pad.")
    parser.add_argument(
        "--box",
        required=True,
        type=int,
        help="Target cubic box, in voxels. The particle box size, normally.",
    )
    parser.add_argument(
        "--out",
        type=Path,
        default=None,
        help="Defaults to <name>_pad<box>.mrc beside the input.",
    )
    args = parser.parse_args()

    import mrcfile

    with mrcfile.open(args.map, permissive=True) as handle:
        volume = np.asarray(handle.data, dtype=np.float32)
        apix = float(handle.voxel_size.x)

    if volume.ndim != 3:
        print(f"{args.map.name}: not a 3D map ({volume.shape})")
        return 1
    if len(set(volume.shape)) != 1:
        print(f"{args.map.name}: only cubic boxes are handled, got {volume.shape}")
        return 1
    size = volume.shape[0]
    if size > args.box:
        print(
            f"{args.map.name} is {size} voxels, larger than the requested {args.box}. "
            "Cropping would discard density; pick the larger box, or crop "
            "deliberately with a tool that says what it dropped."
        )
        return 1
    if size == args.box:
        print(f"{args.map.name} is already {args.box}^3; nothing to do")
        return 0

    pad = args.box - size
    before = pad // 2
    padded = np.pad(
        volume, ((before, pad - before),) * 3, mode="constant", constant_values=0.0
    )

    out = args.out or args.map.with_name(f"{args.map.stem}_pad{args.box}.mrc")
    with mrcfile.new(out, overwrite=True) as handle:
        handle.set_data(padded.astype(np.float32))
        handle.voxel_size = apix
    print(
        f"wrote {out}\n  {size}^3 -> {args.box}^3, {before} voxels before and "
        f"{pad - before} after on each axis, {apix:.4f} A/voxel unchanged"
    )
    print(
        "  Pass this as BOTH --initial-reference and --model-map, so the search "
        "and the FSC refer to one reference."
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
