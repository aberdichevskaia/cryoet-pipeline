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
- **Particles** (next): positions through the z mirror into centred Angstroms;
  poses `F A F` with F = diag(1, 1, -1); the reference mirrored about the box
  centre N/2 (np.flip moves the centre one voxel for even boxes); decoys out,
  duplicates after the MRA shifts, beads, halves from the consensus split. Check:
  RELION's `relion_tomo_reconstruct_particle` against the class average.
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
