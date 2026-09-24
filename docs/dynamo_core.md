# Experimental Dynamo-compatible K=1 core

`cryoet_pipeline.dynamo_core` is an explicit CPU numerical kernel, **not a
registered production backend**. It does not change cone-search, FFT refinement,
particle-frame scoring, CLI defaults, or existing artifacts.

## Scope and API

One iteration, one frozen external reference, C1, even cubes, single Y-axis
wedge, `use_CC=1`, `localnc=1`, isotropic mode-2 shift limits, multilevel angular
search, then raw-particle averaging. No classification, training, GT-dependent
selection, rejection, adaptive filtering, GPU, CTF, or project orchestration.

- `refine_particle(particle, reference, mask, angles, shift, tilt_range, settings)`
  returns final pose, CC, and each search level's winner.
- `average_particles(particles, angles, shifts, tilt_ranges, mask)` returns the
  compensated average, unweighted average, and Fourier weights. It consumes raw
  data, never search-filtered or repeatedly interpolated buffers.
- `SearchSettings()` contains the actual refine223 protocol: cone15/5 degrees,
  in-plane 15/5 degrees, refine 2/factor 2, Fourier radii 2..16, shift 3 voxels.

Arrays are **ZYX**, shifts **XYZ voxels**, angles Dynamo `[tdrot, tilt, narot]`.
`M = Rz(narot) Rx(tilt) Rz(tdrot)`; rotation centre is `N/2`, zero-based.
These are NOT native RELION poses. Use the existing benchmark exporter's matrix
and half-voxel centre conversion. Do not feed native poses directly.

The benchmark consumes reference/masks already prepared by the real Dynamo
project. Project preparation is not ported. The reference is updated once, only
after all particle poses have been searched. Both fixed and refined poses must
use the same averaging helper, to avoid confounding alignment and averaging.

## Source behavior retained deliberately

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
The complete original joint BSD/GNU notice is retained in `dynamo_core.py`;
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
