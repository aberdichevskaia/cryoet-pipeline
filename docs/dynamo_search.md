# `dynamo-search`: Dynamo's local search and averaging

`average-particles --backend dynamo-search` is the default refinement backend.
It searches each particle with a CPU numerical adaptation of Dynamo 1.1.511's
local alignment and rebuilds the reference as Dynamo's project averages it
(`average_particles`). It does not call Dynamo or MATLAB. `cone-search` and
`fft-refinement` stay available.

## Why it is the default

Measured on 223 CZII ribosomes, identical particles, pytom poses and extraction
(benchmark: `DYNAMO_NATIVE_HYBRID.md`, `DYNAMO_HYBRID_ABLATION.md`):

- Dynamo's angle changes were neutral; `cone-search`'s cost about 3 A.
- Its shift search removed a half voxel every particle had been extracted off
  by, which no internal control had seen (`particles.CROP_POSE_CONVENTION`).

The module itself, not only its kernel, reproduces Dynamo: run through the
CLI on refine223's stacks, reference and mask, it returned its kernel's poses
for all 223 particles to the STAR's six decimals, and Dynamo's for 211. The
other twelve are the kernel's two replay failures and ten particles whose
answer Dynamo's five-digit starting table changes (benchmark:
`DYNAMO_MODULE_PARITY.md`).

Once that extraction offset is fixed, the search itself adds little on those
particles (0.3 A); the case for it is that it does not degrade them and that it
can repair a systematic pose error. Template FSC there is reference-biased.

## Why it averages as Dynamo does

Until 2026-10-06 the backend averaged with this pipeline's compensated reducer,
because at the same poses its map had a template FSC about 7 A better than
Dynamo's averaging gave. That FSC was unmasked. Three checks on CZII, each at
identical poses or from an identical start, took the case apart:

- Accuracy. 227 ground-truth-matched particles at their PyTom poses, FSC with
  the 390 A mask against EMD-16721, a human 80S from an unrelated sample, not
  the EMD-3883 the particles were picked with: the two maps agree within 0.02
  on every shell from 60 to 25 A. Unmasked, this pipeline's map stays ahead,
  as the old template FSC said; the difference lies outside the particle.
- Independent halves. 288 particles selected by score, Dynamo's own
  half-set project against this backend run on each half the same way: the
  same masked FSC down to 24 A.
- Artefacts near Nyquist, one each, both traced by swapping one averaging step
  at a time. The pipeline's reducer correlates the halves outside the mask,
  because it divides by the particle count instead of each voxel's count of
  valid contributions. Dynamo's averaging makes masked half-set FSC rise
  towards Nyquist (0.27 at 22.9 A back up to 0.69 at 20 A on those particles);
  its interpolated wedge weights carry that.

With no loss in accuracy, averaging as Dynamo does makes the whole backend
reproduce Dynamo, search and map, each replayed against it: 8e-5 for the
average on refine223, 6.5e-5 on the score-selected project, 4e-7 for the
`dynamo_average` start. Dynamo's rise of half-set FSC near Nyquist comes with
it, so read half-set FSC there with care, and only masked; `compute-fsc` warns
when a curve rises like that (`fsc.rise_toward_nyquist`).

## What one iteration does

1. The current reference is prepared as Dynamo's iteration setup prepares
   every starting reference (`+dpkproject/preprocessReferences.m`): the search
   bandpass, then normalization over the search mask (`prepare_reference`).
2. Every particle is searched against it, held fixed until all particles are
   done: a multilevel cone and in-plane grid around its pose, a moving-mask
   local NCC, and a shift search inside a sphere re-centred on each level's
   winner.
3. The reference is rebuilt from the particles at the new poses as Dynamo's
   project assembles it (`average_particles`, its `fsc = true` path): each
   particle at its inverse pose, normalised over the alignment mask's nonzero
   voxels, with every voxel divided by the particles that really reached it;
   odd and even particles summed apart, each normalised, merged; the merged
   wedge weight divided where it reaches 0.1 of its maximum and every other
   component zeroed; the map not normalised again.

The first reference is `dynamo_average` of the initial poses with `fcompensate`
and `fmin 1` (`average_class`), the console call that makes a Dynamo project's
starting template, or `--initial-reference`: a raw map, as Dynamo's
`file_template_initial` is, never
a reference Dynamo has already transformed, which would be prepared twice.
Without step 1 the search differed from Dynamo's on every one of 25 refine223
particles (median 1.25 degrees, one finest-level step); the module parity runs
before 2026-09-28 fed Dynamo's transformed reference and are unaffected, and
parity v3, from Dynamo's raw template with this preparation, wrote the same
STAR byte for byte. `--num-iterations` defaults to 1. Two iterations reproduce
Dynamo's on the CZII score-selected set: every pose that differs is either one
Dynamo's spline defect gave (`Source behavior` item 7) or inherited from Dynamo
writing its tables at five digits and its maps in float32 between iterations.
Without `--gold-standard` the step is K = 1 and not gold-standard: every
particle is scored against an average that contains it.

## Independent halves (`--gold-standard`)

With `--gold-standard` the step follows Dynamo's half-set projects
(`dynamo_vpr_eo`):

1. One start for both halves: the initial reference, or `dynamo_average` of all
   particles at their initial poses, band-passed as Dynamo's
   `dynamo_bandpass(v, [1 low 2])`. `low` comes from
   `--gold-standard-start-lowpass-angstrom`, 60 A by default, which is the CZII
   session's `[1 8 2]` at a 48-voxel box of 10.012 A. Below it the halves share
   the start; above it each builds its own map from its own data.
2. Each half, the particles of `rlnRandomSubset` 1 and 2, then runs every
   iteration alone: its own reference, search and rebuild. Nothing of one half
   reaches the other: a test replaces every particle of one half by noise, and
   the other half's poses and map do not change by a bit.
3. The map written as the average is every particle at the pose its own half gave
   it. The two half maps are written beside it (`half_map_paths`), and QC records
   their FSC with the alignment mask and without: the curve, its 0.5 and 0.143
   crossings, and any rise toward Nyquist, which is also a warning.
   `resolution_angstrom` is the masked 0.143 crossing. Below ten particles in a
   half the FSC is reported but flagged as uninterpretable.

The halves are `rlnRandomSubset`, assigned once at extraction and never
recomputed, so a particle stays in its half through every step. The column is
not random, though. Extraction writes `1 + index % 2` in each tomogram's pick
order, which is pytom's score order, so the halves alternate by score rank. They
are better balanced in score than a random split, and every tomogram is split
evenly. Dynamo's `dynamo_vpr_eo` uses the same parity over the table's rows, but
recomputes it whenever a project is made. Neither keeps neighbours with
overlapping boxes in the same half.

Checked against Dynamo's own half-set project on the CZII score-selected set
(288 particles, three iterations, the halves as Dynamo set them):

- The prepared start of both halves matched Dynamo's to 3e-8.
- After three iterations 126 and 106 of 144 poses matched. Each of the other 56,
  searched again from Dynamo's own inputs, was either Dynamo's spline defect
  (one) or gave Dynamo's pose, so it was inherited from Dynamo's five-digit
  tables and float32 maps.
- The QC's half-map FSC agreed with the FSC of Dynamo's own half maps within
  0.006 masked and 0.010 unmasked, and flagged the same masked rise toward
  Nyquist, 0.25 at 22.9 A back up to 0.68 at 20.0 A. Before the dip it fell
  through 0.5 at 25.3 A, which QC reports as `crossing_before_dip_0_5_angstrom`.

## Parameters

| option | default | meaning |
|---|---|---|
| `--cone-range-deg` | 15 | full cone aperture, first level |
| `--cone-sampling-deg` | 5 | cone step, first level |
| `--inplane-range-deg`, `--inplane-sampling-deg` | 15, 5 | in-plane range and step |
| `--refine-levels` | 2 | levels after the first; each shrinks by `--refine-factor` (2) |
| `--search-highpass-px`, `--search-lowpass-px` | 2, 16 | search band in Fourier pixels |
| `--shift-range-px` | 3 | shift sphere radius |
| `--gold-standard` | off | refine the two halves of `rlnRandomSubset` independently; see above |
| `--gold-standard-start-lowpass-angstrom` | 60 | band-pass of the halves' common start, as `dynamo_bandpass [1 low 2]` |
| `--shift-search-mode` | 2 | Dynamo's `limm`: 2 re-centres the shift sphere on each level's winner, 1 keeps it on the box centre (item 6) |
| `--alignment-mask-diameter-angstrom` | particle diameter | binary search mask, a sphere about `(N - 1) / 2`; it also normalises the particles |
| `--alignment-mask` | — | a mask file instead, used as Dynamo uses one: voxels above 0.5 are searched, nonzero voxels normalise the particles |
| `--workers` | the job's CPUs | processes searching particles in parallel; the poses do not depend on it |

The grid, band and shift defaults are Dynamo's `refine223` project on CZII: a
48-voxel box at 10.012 A, where 16 Fourier pixels is 30 A. The band is in
Fourier pixels, so a different box needs its own `--search-lowpass-px`. That
project's mask was a soft sphere whose part above 0.5 is a 390.468 A (39 px)
sphere: `--alignment-mask-diameter-angstrom 390.468` reproduces its search mask
voxel for voxel, the particle-diameter default does not, and only
`--alignment-mask` with the file itself reproduces its normalisation, which
reaches out to the soft edge. The wedge is always modelled and the
average always compensated, so `--missing-wedge` and `--fourier-compensation`
are refused; `--wedge-min-tilt-deg` and `--wedge-max-tilt-deg` override the
manifests. The tilt axis must be y.

## Conventions

The kernel works in Dynamo's frame: ZYX arrays, XYZ shifts, `[tdrot, tilt,
narot]` degrees, `M = Rz(narot) Rx(tilt) Rz(tdrot)`, rotation centre `N / 2`.
Native poses use `x_in = cN + A (x_out - cN) + s` with `cN = (N - 1) / 2`. The
backend converts with `native_to_dynamo` / `dynamo_to_native`: `M = A.T`,
`d = s + (A - I) (cD - cN)`, exact Euler angles (not Dynamo's 1e-4 pole
tolerance). The kernel never sees a native pose. Even cubes only.

The search mask is centred on `cN`, where native poses put the particle, not
on the kernel's rotation centre; the kernel rotates it with the reference. A
mask about `N / 2` sits (0.5, 0.5, 0.5) voxels off the particle. On 25 of
refine223's particles, with Dynamo's own reference and mask diameter, that
alone moved all 25 poses off Dynamo's, by a median of one finest-level step
(1.25 degrees).

## QC

`qc/<scope>/subtomogram_averaging/*.json` (`DynamoSearchQc`) records the search
settings, per-iteration mean and minimum CC and pose movement, the whole run's
`pose_change`, the alignment mask (file or diameter, and how many voxels it
searches and normalises), the last average's Fourier weight inside the Nyquist
sphere (minimum, median, and the fraction below 0.1, which the averaging
zeroes), with `--gold-standard` the half counts and the half-map FSC, the wedge
boundary policy, and
`beyond_grid_count`: particles that moved further than the grid can reach
(`grid_reach_deg`), which is Dynamo's legacy in-plane seed at work. Those moves
are kept, as Dynamo keeps them, and reported as a warning.

## Speed

The kernel is NumPy on the CPU: 4 to 7 s of one core per particle and
iteration at box 48 with the defaults (measured on the Mac: 3.9 to 4.5 s on a
free core, 7.4 s while it was busy), about Dynamo's own 4.4 core-seconds
(refine223: 246 s of search on four processes). Per particle that is 221
candidate orientations, each a rotation of the reference and mask and about
seven 48^3 FFTs. A cluster CPU core is slower: 223 particles took 2195 s on
`cpusrv19`, 9.8 s each, one after another, and 373 s in 16 processes.

So the particles are searched in parallel, as Dynamo's CPU project does it:
the reference stays fixed until all of them are done, so each is independent,
and `--workers` processes share them. The default is the CPUs the job was
given, `SLURM_CPUS_PER_TASK` or else the process's affinity mask. Every
particle is still searched by the same code on the same float64 inputs, and
nothing in the search sums across threads, so the poses do not depend on the
worker count; a test holds the STAR to the same bytes. Workers are spawned, so
a script that calls the backend directly needs the usual
`if __name__ == "__main__":` guard; the CLI has one. The averaging is NumPy in
the main process, 12 ms a particle at box 48.

## Source behavior retained deliberately

Items 8 and 9 describe `average_particles`, which the backend averages with.

1. The CPU project applies the bandpass twice to a particle: database processing,
   then its wedge/band filter. The rotated template gets that filter once.
   Gaussian shell shoulders use smoothing 2, truncated below weight 0.13.
2. Project preparation thresholds the alignment mask at **>0.5**, even without
   resizing. The core receives this binary mask. Rotation requires all eight
   interpolation corners to have nonzero support. The original soft mask is
   used separately for averaging normalization: its nonzero voxels define an
   ROI, not weights.
3. Local NCC normalizes particle values in a moving ROI at each shift; it is not
   a dot product of globally masked volumes.
4. Rotation follows the CPU MEX float32 interpolation, Fourier centre, and strict
   eight-corner boundary rule. FFT/correlation use NumPy double precision;
   bitwise FFT equivalence to MATLAB is not claimed.
5. Candidate order, last-winner ties, matrix-to-Euler pole tolerance, and the
   legacy in-plane seed are retained. That seed can generate large SO(3) jumps;
   compatibility does not endorse the behavior.
6. Mode 2 (`area_search_modus=2`, Dynamo's `limm 2`) centres a search sphere
   on the rounded preceding winner at each refinement level. A particle whose
   peak lies outside it therefore follows the sphere outward from level to level
   and from iteration to iteration. Mode 1 keeps the sphere on the box centre.
   Subpixel fitting uses unrestricted neighbours in both modes, so the final
   displacement may exceed the sphere by up to a voxel.
7. Peak fitting follows `dynamo_peak_subpixel_direction`, not a parabola. A
   degenerate fit without a finite valid extremum keeps the integer peak. In
   that case Dynamo returns `pv(1) - 1` unchecked, and in single precision this
   can be complex. Under mode 2 the next level rounds it into the sphere's
   centre and `circshift` aborts the project (CZII MRA, iteration 3, tag 868:
   coarse z = 6.00 + 0.60i). Mode 1 recentres nothing, so the search cannot
   abort there. A complex value from the last level still makes the
   iteration's whole table complex, though. Dynamo's binary table writer then
   stores the interleaved buffer as if it were real, and the assembly aborts on
   the scrambled table (CZII MRA arm B under mode 1, iteration 5, tag 696:
   y = -1 + 2.10i).
   Near the fit's own singularity (a true offset of 0.293 voxel for refine223
   tags 53 and 170), Dynamo's single-precision arithmetic also returns finite
   nonsense that can win the search: integer offsets, and scores that are
   simple fractions (0.5, 0.75, 19/24, 1; also CZII MRA tag 868 in iteration 2).
   The port fits in double and reproduces neither. Scores in (1,1.05) are
   clamped to 1 as in Dynamo; larger overshoots remain visible. Ill-conditioned
   extrema may differ between runtimes. No tolerance is adjusted to conceal
   mismatches.
8. Averaging divides by actual valid resampling contributions per voxel
   (`explicitBalanceNan=1`). Particles are normalized after alignment using ROI
   sample standard deviation.
9. The project's `fsc=true` assembly equally merges separately normalized
   odd/even raw averages and normalized wedge weights. The merge hard-codes a
   Fourier-weight threshold **0.1** and does not normalize the final map again.
   These halves were **not independently refined**.

## Source and attribution

Distribution inspected: `dynamo-v-1.1.511_MCR-9.8.0_MACI64_withMCR`.
The complete original joint BSD/GNU notice is retained in `backends/dynamo_search.py`;
the BSD option is selected for the adapted routines. The repository's MIT
license does not replace that notice. No Dynamo binaries/runtime are bundled.

Relevant sources within that distribution:

- `matlab/src/+dpkproject/pipeline_align_one_particle.m`
- `matlab/src/dynamo__align_motor.m`
- `matlab/src/dynamo_angleincrement2list.m`, `dynamo_compose_rotations.m`,
  `dynamo_matrix2euler.m`
- `matlab/src/+dpkgeom/+angles/closeFormulaBestTdrotForIncrementList.m`
- `matlab/src/dynamo_peak_subpixel.m`
- `matlab/src/dynamo_normalize_roi.m`, `dynamo_fweight_divide.m`, `dynamo_average.m`
- `matlab/src/+dpkproject/+pipeline/genericInput2Average.m`
- `matlab/src/dynamo_resize_mask.m`, `dynamo_database_volume_process.m`
- `mex/src/classes/volume_float.cpp`, `volume_noncubic.cpp`, `orientation.cpp`

See the benchmark's `DYNAMO_CORE_REPLAY.md` for the fixed 223-particle comparison.
Numerical equivalence is not independent evidence of improved resolution.
