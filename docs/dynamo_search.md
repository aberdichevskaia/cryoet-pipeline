# `dynamo-search`: Dynamo's local search, this pipeline's averaging

`average-particles --backend dynamo-search` is the default refinement backend.
It searches each particle with a CPU numerical adaptation of Dynamo 1.1.511's
local alignment and rebuilds the reference with the pipeline's Fourier-
compensated reducer, the one `cone-search --fourier-compensation` uses. It does
not call Dynamo or MATLAB. `cone-search` and `fft-refinement` stay available.

## Why it is the default

Measured on 223 CZII ribosomes, identical particles, pytom poses and extraction
(benchmark: `DYNAMO_NATIVE_HYBRID.md`, `DYNAMO_HYBRID_ABLATION.md`):

- Dynamo's angle changes were neutral; `cone-search`'s cost about 3 A.
- Its shift search removed a half voxel every particle had been extracted off
  by, which no internal control had seen (`particles.CROP_POSE_CONVENTION`).
- With the same poses, this pipeline's averaging beat Dynamo's by about 7 A.

The module itself, not only its kernel, reproduces Dynamo: run through the
CLI on refine223's stacks, reference and mask, it returned its kernel's poses
for all 223 particles to the STAR's six decimals, and Dynamo's for 211. The
other twelve are the kernel's two replay failures and ten particles whose
answer Dynamo's five-digit starting table changes (benchmark:
`DYNAMO_MODULE_PARITY.md`).

Once that extraction offset is fixed, the search itself adds little on those
particles (0.3 A); the case for it is that it does not degrade them and that it
can repair a systematic pose error. Template FSC there is reference-biased.

## What one iteration does

1. Every particle is searched against the current reference, held fixed until
   all particles are done: a multilevel cone and in-plane grid around its pose,
   a moving-mask local NCC, and a shift search inside a sphere re-centred on
   each level's winner.
2. The reference is rebuilt from the raw particles at the new poses with
   `subtomogram._compensated_reference`: each particle's wedge at its own tilt
   range, frequencies divided by how many particles measured them.

The first reference is that same average of the initial poses, or
`--initial-reference`. `--num-iterations` defaults to 1, the protocol that was
validated. K = 1, and not gold-standard: every particle is scored against an
average that contains it.

## Parameters

| option | default | meaning |
|---|---|---|
| `--cone-range-deg` | 15 | full cone aperture, first level |
| `--cone-sampling-deg` | 5 | cone step, first level |
| `--inplane-range-deg`, `--inplane-sampling-deg` | 15, 5 | in-plane range and step |
| `--refine-levels` | 2 | levels after the first; each shrinks by `--refine-factor` (2) |
| `--search-highpass-px`, `--search-lowpass-px` | 2, 16 | search band in Fourier pixels |
| `--shift-range-px` | 3 | shift sphere radius, re-centred each level |
| `--alignment-mask-diameter-angstrom` | particle diameter | binary search mask, a sphere about `(N - 1) / 2` |
| `--workers` | the job's CPUs | processes searching particles in parallel; the poses do not depend on it |

The grid, band and shift defaults are Dynamo's `refine223` project on CZII: a
48-voxel box at 10.012 A, where 16 Fourier pixels is 30 A. The band is in
Fourier pixels, so a different box needs its own `--search-lowpass-px`. That
project's mask was a 390.468 A (39 px) sphere, which
`--alignment-mask-diameter-angstrom 390.468` reproduces voxel for voxel and the
particle-diameter default does not. The wedge is always modelled and the
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
`pose_change`, Fourier coverage, the wedge boundary policy, and
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
`if __name__ == "__main__":` guard; the CLI has one. The averaging runs on
`--device`.

## Source behavior retained deliberately

Items 8 and 9 describe the kernel's own `average_particles`, kept so the
benchmark's replay can still check averaging against Dynamo; the backend
averages with the pipeline's reducer instead.

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
6. Mode 2 centres a search sphere on the rounded preceding winner at each
   refinement level. Subpixel fitting uses unrestricted neighbours, so the
   final displacement may exceed 3 voxels.
7. Peak fitting follows `dynamo_peak_subpixel_direction`, not a parabola. A
   degenerate fit without a finite valid extremum keeps the integer peak rather
   than propagating NaN/-Inf. Scores in (1,1.05) are clamped to 1 as in Dynamo;
   larger overshoots remain visible. Ill-conditioned extrema may differ between
   runtimes. No tolerance is adjusted to conceal mismatches.
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
