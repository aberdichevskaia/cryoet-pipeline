# Export to RELION 5

## Purpose

After MRA or a Dynamo-style refinement a person says which class looks right.
`export-relion` is to write one folder from that class that RELION 5 can run as
it is, for the high-resolution refinement this pipeline does not do: Refine3D on
pseudo-subtomograms, then CtfRefine and Bayesian polishing. RELION extracts the
pseudo-subtomograms from the tilt images itself, so the export carries the tilt
series and their geometry, not the pipeline's subtomograms. A RELION 3.1-style
export of our own subtomograms with 3D CTF volumes would need no geometry, but
it would cap the resolution at our tomogram's and allow neither CtfRefine nor
polishing.

Agreed on 9 November 2026: RELION 5, the geometry first, checked by RELION.

## The folder (planned)

```text
<project>/relion/<label>/
  README.md              what is inside, the numbers, the checks it passed, the commands
  run.json               the step's record (run_records.md)
  tomograms.star         per series: optics, size, hand, its tilt_series/<TS>.star
  tilt_series/<TS>.star  per tilt image: geometry, CTF, pre-exposure
  tilt_series/<TS>.mrcs  the raw tilt images the alignment kept, unaligned, not phase-flipped
  particles.star         the class: positions, poses, rlnRandomSubset
  optimisation_set.star
  reference/             the class average in RELION's frame, rescaled and low-passed; the mask
  relion_refine.sbatch   Extract + Refine3D, C1, local searches around the MRA poses
```

The folder is the RELION project: paths in it are relative, RELION runs from it.
The step submits nothing.

## Built: the tilt geometry

`cryoet_pipeline.relion_export` writes RELION's five numbers per tilt image
(rlnTomoXTilt, YTilt, ZRot, X/YShiftAngst) from the fine alignment's transforms
and what IMOD's tilt was given (thickness, -XAXISTILT, -SHIFT 0,z). The module's
docstring has the formulas, from RELION 5.0's `Tomogram::setProjectionMatrix`
(7f45a71) and IMOD's newstack and tilt. It also writes tomograms.star and the
per-series tables, and computes pre-exposures and the astigmatism angle in the
raw image's frame.

RELION is given IMOD's standard orientation (`clip rotx`), as RELION's own IMOD
import gives it (ytilt is the .tlt angle there too). The pipeline's canonical
tomogram is that orientation reversed in z, a mirror image (`transpose(1, 0, 2)`
in the reconstruction backend, measured 15 October). Particle positions, poses
and maps from the pipeline therefore go through a z mirror on their way out.

### Measured (9 November 2026)

`scripts/relion_geometry_check.py` has RELION's own
`relion_tomo_reconstruct_tomogram` rebuild a series from the export and compares
it with the pipeline's tomogram. On EMPIAR-10164 TS_01 (outputs/dev-ts01-registry,
bin8, 40 views, XAXISTILT -1.62) with RELION 5.0 built on a Mac:

| ytilt | xtilt | best match to the pipeline's tomogram | NCC | next |
|---|---|---|---|---|
| +tilt | -XAXISTILT | reversed in z (the standard orientation) | 0.681 | 0.027 |
| -tilt | +XAXISTILT | as it is (the pipeline's mirror frame) | 0.681 | 0.027 |
| +tilt | +XAXISTILT | reversed in z | 0.108 | 0.023 |
| -tilt | -XAXISTILT | as it is | 0.108 | 0.023 |

The first row is what the module writes. Shift against the pipeline's tomogram in
128-voxel boxes: 0.00 voxel at the centre, at most 0.07 voxel (0.4 A) at the
four quarter positions. A reconstruction with `-SHIFT 0,20` (local IMOD) lands as
well, which fixed the z shift's sign: in the standard orientation IMOD's centre
of rotation is at (Nz - 1)/2 - shift.

The continuous parts (centres, the transform's inverse, its shift, the z shift)
were derived and then predicted the 0.00 voxel; only the two discrete signs were
chosen by the data, and the wrong choices fail by a factor of six or more.

`series_sources` finds what a registry tomogram was made from (fine alignment,
raw stack, final stack, CTF model, manifest) and `export_series` writes the raw
tilt images the tomogram used, binned as the final stack bins them, with each
view's row: CTF from the CTF model (CTFFIND's U, V and azimuth, paired by z
value), pre-exposure in acquisition order (the mdoc's DateTime, else z order),
the mdoc's dose or a given one. Every fallback is listed in its `notes`. The
geometry check runs through this code.

## Built: particles

A particle the pipeline aligned is a box cut from the canonical tomogram at
`start` (extraction: round(pick) - N//2) with Dynamo's angles and shift about
N/2, as MRA records them: reference point r lies at `start + N/2 + A (r - N/2) + d`,
A = M^T of the Dynamo matrix. RELION 5 puts it at `pos + E (r - int(N/2))`
(ParticleSet::getMatrix4x4), E the classic RELION Euler matrix (the same rows as
`rotations.relion_euler_to_matrix`), pos = centred / pixel + size / 2. Into
IMOD's standard orientation:

- position: z reversed, `(Nz - 1) - z`; a voxel of a tomogram b times coarser
  than the stack RELION reads sits at `b u + (b - 1)/2`; centred on size / 2
  (`TomogramPlacement.centred_angstrom`);
- pose: `E = F A F`, F = diag(1, 1, -1) (`standard_frame_matrix`), as
  rlnAngleRot/Tilt/Psi;
- reference: mirrored in z about int(N/2), index k to N - k for an even box
  (`standard_frame_reference`; np.flip alone would move the centre one voxel).

Boxes must be even, as MRA's are (dynamo_search refuses odd ones); an odd box
puts half a voxel between Dynamo's centre and RELION's that nothing has checked,
so `relion_pose` refuses it. `write_particles` writes `data_particles` as RELION
reads it straight after picking (RELION makes the optics table from
tomograms.star), with rlnRandomSubset for all or none.

### Measured (10 November 2026)

`scripts/relion_particle_check.py` places 8 particles on structured parts of
TS_01 (no beads), gives each a random orientation and a shift of up to 2 voxels,
and compares RELION's `relion_tomo_reconstruct_particle` at the exported pose with
the pipeline's own `average_class` (what makes the MRA class averages) of the
same box at the same pose, mirrored as above:

| stack RELION reads | per particle: NCC, next reversal | all 8 | offsets |
|---|---|---|---|
| bin8 (as the tomogram) | 0.51-0.77, at most 0.14 | 0.68 (next 0.10) | at most 0.17 voxel |
| bin4 (RELION at --bin 2) | 0.51-0.73, at most 0.14 | 0.66 (next 0.09) | at most 0.14 voxel |

The first run, before the comparison was fixed, failed everywhere (NCC 0.02);
the cause was RELION's map, not the export (below).

### RELION facts found on the way

- `reconstruct_particle` writes `data_merged.mrc` (the backprojected data) and
  `merged.mrc` (the data divided by its Fourier weights). A few particles' weights
  are a few planes, so `merged` is mostly noise: on a gold bead `data_merged`
  held the bead at the centre and `merged` only streaks. It also inverts the
  contrast (`flip_value` is true in 5.0). The check compares `-data_merged`.
- `--no_ctf` fills the weights with 1; nothing else changes.
- `reconstruct_tomogram` with `--x0/--y0` put a region at twice its origin
  (found by correlating it against the whole tomogram, NCC 0.88). Not used.
- `reconstruct_tomogram --bin 2` from a bin4 stack of odd-sized images is not on
  IMOD's grid: 0.2-0.7 voxel, with a scale along the images' odd axis. Its
  particles at the same binning are exact (bin4 row above), so the geometry
  check runs at the tomogram's binning and binning is checked on particles.

### What RELION's model drops

A rotation and a shift per image: tiltalign's per-view magnification (MagOption 1
here), stretch and skew are not representable, and RELION's own IMOD import
drops them too. The export reports how far that moves the aligned image's
corners (`RelionTilt.dropped_px`): on TS_01, magnifications 0.9996-1.0013, up to
0.88 bin8 pixel, 4.8 A. Polishing absorbs some of it. To keep it, the exported
images would have to be resampled by each view's magnification, or the
alignment solved without one; not decided.

## Open

- **Hand (rlnTomoHand).** RELION's documentation: it "has to be determined
  experimentally. In our experiments, it has always been -1." The check writes
  -1; nothing here measures it. It does not matter for geometry or at ~20 A, it
  matters for the defocus gradient at high tilt and high resolution.
- **Astigmatism angle.** CTFFIND fits on the aligned stack
  (`astigmatism_frame: aligned_stack`); `defocus_angle_in_raw_frame` turns the
  angle back by the alignment's rotation. Not checked against a fit on raw images.
- **Dose.** rlnMicrographPreExposure counts every acquired view in acquisition
  order, the left-out ones too. Some mdocs give no dose (10164: ExposureDose 0),
  so the lane has to give one.
- **The step itself** (next): `export-relion` reading an MRA run's class (the
  run's selection `class <c>`, poses of the chosen iteration, positions from the
  setup table), decoys out, duplicates after the MRA shifts, beads, halves from
  the consensus split, the class average mirrored and rescaled to the extraction
  pixel, a mask by the processing rules, README, sbatch and run.json.
- **CTF from a CTF model** is unit-tested only: the local TS_01 project has no
  CTF model, so no RELION run has read a fitted defocus from this export yet.
- **RELION at MDC.** Unknown whether RELION 5 is installed on max. The check
  runs anywhere RELION 5 does, on a series copied at its tomogram's binning.

## RELION 5 on a Mac, for checks

CPU only, no GUI, built 9 November 2026 in `~/Developer/relion5` (source at
ver5.0 7f45a71, conda env `~/Developer/relion5/env` from conda-forge: cmake 3.31
(4.x rejects RELION's CMakeLists), openmpi, llvm-openmp, c-compiler, cxx-compiler,
fftw, libtiff, libpng, libjpeg-turbo, zlib, python 3.11):

```bash
cmake -S src -B build -DCMAKE_BUILD_TYPE=Release -DGUI=OFF -DCUDA=OFF -DFETCH_WEIGHTS=OFF \
  -DPYTHON_EXE_PATH=$PWD/env/bin/python3 -DCMAKE_PREFIX_PATH=$PWD/env
cmake --build build -j 8 --target reconstruct_tomogram reconstruct_particle image_handler
```

`--o` of relion_tomo_reconstruct_tomogram is a folder: the volume is written to
`<o>tomograms/rec_<name>.mrc`.
