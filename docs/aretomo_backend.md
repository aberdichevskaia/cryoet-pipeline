# AreTomo2 fine-alignment backend

## Why it exists

Lamellae carry no gold, so the pipeline aligned them by IMOD patch tracking. On
the first EER lamella series it met, EMPIAR-11830
`01082023_BrnoKrios_Arctis_WebUI_Position_1`, patch tracking did not hold:
fine alignment ended with a mean residual of 18.5 nm (job 3901834, 7 November
2026), and 17.7 nm when the same route was repeated off the cluster, against a
1.5 nm QC limit. Leaving out the views the authors left out did not change that
(16.4 nm). The patches wander one by one, worst on stacked membranes. The mdoc
tilt angles fit as well as any tiltalign solves.

The authors aligned the same series with AreTomo and reconstructed it with IMOD.
Their deposited `AreTomo/` folder holds one global `.xf`, the `.tlt` with the
+8 deg tilt offset AreTomo measured, and the `newst.com` and `tilt.com` that
used them (bin 8, thickness 2048 unbinned pixels, no X-axis tilt, no shift).
This backend is that route.

## What it does

`fine-align-tilt-series --backend aretomo` replaces the two fiducial steps and
tiltalign. `align-tilt-series` still runs first: it decides which views are
left out, and gives the starting tilt axis. Then the backend:

1. writes the included views, at full resolution and sorted by tilt, into one
   MRC, with their nominal angles in an angle file;
2. runs AreTomo2 with `-VolZ 0` (align only), `-TiltAxis <axis> 1` (refine
   within 6 deg), `-AlignZ`, `-TiltCor`, `-DarkTol` and `-Gpu`;
3. reads its `.aln`: per view the tilt axis, the shift and the tilt angle, and
   the views it dropped as dark, which are then left out of every later step
   too;
4. turns each view into the IMOD transform AreTomo2 itself writes for RELION 4,
   A = R(-ROT) with the shift A (-TX, -TY), at the `.aln`'s precision;
5. registers a fine alignment with the same id, folder and canonical files as
   tiltalign's, so that `build-final-aligned-stack`, the CTF steps and
   `reconstruct-tomogram` run unchanged.

AreTomo2 solves no X-axis tilt and no thickness. Reconstruction takes X-axis
tilt 0 and no Z shift, as the authors' `tilt.com` did, and needs the thickness
from the lane: `[reconstruct] thickness` is required, and the runner refuses a
lane without it.

## QC

There are no fiducials, so there is no residual in tiltalign's sense. AreTomo2's
projection matching measures each view against a reprojection of the others
and prints the shift it gives each view; the reference view, the one nearest
0 deg after the tilt offset, is never printed. It works in rounds
(`CProcessThread::mAlign`): with a tilt axis given and refined, three rounds,
and a common-line search of the axis before the second and third (within 6 and
then 3 deg). Within a round (`mProjAlign`) it makes passes until the largest
shift is under 2 px, which ends the round with that pass kept, or until a pass
moves a view further than the last kept one, which is undone, though its table
is printed; at most 10 passes. On Position_1 (10 November 2026) that was 6
passes in 3 rounds of 2, the axis 86.73, then 83.70, then 83.21 deg, and the
first pass of round 2 large again after the axis moved. The QC
(`AretomoAlignmentQc`, qc_type `fine_alignment`) reports the last pass:

- `last_pass_correction_mean_nm` against the fine-alignment thresholds
  (warning 0.8 nm, failure 1.5 nm);
- `converged`: whether AreTomo2's own 2 px rule was met; a warning if not;
- the tilt axis it solved, the tilt offset it applied, the views it found dark,
  and its rotation scores.

The last pass is what AreTomo2 still wanted to move, not a residual after
moving. It is the closest number its output gives. Its relation to tiltalign's
residual has not been measured.

The tilt offset is the best of 31 whole degrees from -15 to +15
(`CTiltOffsetMain::mSearch`), with no finer step: Position_1's +12.00 is good
to half a degree, and an offset beyond 15 deg cannot be found.

`cryoet report` draws the same numbers on the tilt-alignment page, read back
from the AreTomo2 log kept beside the alignment: the last pass's shift on each
tilt (paired with the aligned views by tilt, either as printed or moved by the
measured offset, else not drawn), the mean and largest shift of every pass on
a log scale against the 2 px stop, with the rounds shaded and named by the
axis each started from and an undone pass drawn hollow, and the QC's numbers.
The aligned-tilts page names the tilt with the largest last-pass shift, and
the summary table puts the last pass's mean in the residual column, marked
with an asterisk. Before 2026-11-10 the page said only that tiltalign's files
could not be read.

## Lane config

```toml
[align]
binning = 8
min_std_ratio = 0.2
tilt_axis_angle = 84.1
# Optional: views to leave out of every step, by nominal angle (within 0.5 deg).
exclude_tilt_angles = [-75, -72, -69, -66, -12, -9, 6]

# No [fiducials] section.

[fine_align]
backend = "aretomo"
# aretomo_align_z_nm = 200    # default: AreTomo2's own 600 px of the tilt stack
# aretomo_tilt_cor = true     # default: measure the tilt offset and apply it
# aretomo_dark_tol = 0.7      # AreTomo2's default
# aretomo_gpu = 0

[reconstruct]
thickness = 250               # required with AreTomo2

[executables]
aretomo = "$ARETOMO"          # AreTomo2; else AreTomo2 on PATH
```

The stage that runs `fine-align-tilt-series` needs a GPU (`--partition=gpu`,
`--gres=gpu:...`); `submit_lanes.py` refuses the lane otherwise.

## What was read, not run

AreTomo2's behaviour here comes from its source (czimaginginstitute/AreTomo2,
main, 1.1.3), and has not run in the pipeline yet:

- the `.aln` format and its section numbering (`MrcUtil/CSaveAlnFile.cpp`,
  `CLoadMain.cpp`): views by their section in the stack it read, sorted by tilt;
- the `.aln` goes next to the output, named after the input with `.mrc`
  replaced by `.aln` (`CSaveAlnFile::DoIt`);
- `-VolZ` is 1200 in the code although the help says 0;
- the RELION `.xf` (`ImodUtil/CSaveXF.cpp`, `mSaveForRelion`). The rotation
  part is checked against the authors' deposited `.xf` (axis -84.6 deg on
  their Y-mirrored stack). The shift part is unchecked;
- the projection-matching table and the 2 px rule (`ProjAlign/
  CProjAlignMain.cpp`, `CProcessThread.cpp`);
- `--version` prints `AreTomo2 version 1.1.3`. AreTomo3 is refused: it reads
  movies and mdocs by prefix, not one stack.

Expected on Position_1, from the authors' files: a tilt axis near +84.6 deg (our
stack is not Y-mirrored), a tilt offset near 8 deg in magnitude, and a
lamella that lies flat in the reconstruction.
