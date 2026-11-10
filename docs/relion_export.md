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

## The step

```bash
python scripts/export_relion.py --run <project>/mra/<NAME> --class 1 --dry-run
python scripts/export_relion.py --run <project>/mra/<NAME> --class 1
```

It reads a run of the registry (`<project>/mra/<NAME>`, the run-record layout): the
class's members after the last finished iteration (`--iteration` for another),
decoys left out, with `--agree RUN:CLASS` only those also in another run's class
(as the consensus split takes them), their poses of that iteration, and their picks
from the setup's table. Each particle's tomogram is its particle stack's parent in
the registry. A particle whose centre (box corner + N/2 + the MRA shift) came within
`--min-distance` (a quarter of the box) of a better-scoring one of its series is a
duplicate and left out. Halves: particles of a series within a box of each other are
linked and a linked group goes to one half, so no half holds the other's density;
groups are dealt with `--seed` to the smaller half of their series.

The dry run prints those numbers and every series' stack (views, binning, size)
and writes nothing; the person confirms the numbers. Then:

```text
<project>/relion/<NAME>-class<C>/
  README.md              the numbers, the frame, what the rows fall back on, the commands
  run.json               the step's record (run_records.md), written last
  tomograms.star         per series: optics, size, hand, its tilt_series/<TS>.star
  tilt_series/<TS>.star  per tilt image: geometry, CTF, pre-exposure
  tilt_series/<TS>.mrcs  the raw tilt images the tomogram used, unaligned, not phase-flipped
  particles.star         the class: positions, poses, rlnRandomSubset
  optimisation_set.star
  reference/             the class average and the MRA mask, mirrored in z, at the MRA's pixel
  relion_commands.sh     Extract at the MRA's pixel and box, Refine3D local from the poses, C1
  selection.json         each particle's tag, RELION name and half; duplicates and what they duplicate
```

The folder is the RELION project: paths in it are relative, RELION runs from it.
The step submits nothing. The tilt images are binned to 1 A per pixel or coarser by
default (`--stack-binning`): super-resolution sums are written at the camera's pixel.
RELION's first round extracts at the MRA's pixel and box, so the reference fits as it
is (relion_refine refuses a reference of another size); finer pixels are later rounds.
`--relion-check` has RELION reconstruct the exported class and compares it with the
exported reference.

### Tried (10 November 2026)

On a registry run made locally from TS_01: 640 picks on structured parts of the
tomogram with random orientations, extracted at 48 voxels (`cryoet extract-particles
--picks-star`), one MRA iteration of two channels (`scripts/mra_port_run.py --source`,
28 minutes on the Mac's CPU). Class 1 held 319 particles. The export took 19 s: no
duplicates, halves 160 and 159 with 34 groups of neighbours kept together, the tilt
images binned 2 (1.35 A, 2.3 GB), RELION at --bin 4 for the MRA's 5.4 A. With
`--relion-check`, RELION's reconstruction of the class from the tilt images was the
exported reference: NCC 0.749 against 0.088 for the next reversal, offset 0.01 voxel
(0.80 against its weighted map). RELION read `optimisation_set.star` there, and its
Extract (`relion_tomo_subtomo`, the first line of relion_commands.sh) ran on the
folder as written. relion_refine without MPI refuses `--split_random_halves`, as it
should; the script runs `relion_refine_mpi` under mpirun.

The gold-standard Refine3D itself was not run on the Mac: there an Open MPI program
hangs at its start (`mpirun -n 2 relion_refine_mpi` sits at 0% CPU and prints
nothing, while `mpirun -n 2 hostname` runs), and without MPI RELION 5.0 refuses half
sets ("Mixing gold-standard separation"; the `--debug_split_random_half` its message
suggests is no option of 5.0). Up to that refusal Refine3D read the Extract's
optimisation set, the pseudo-subtomograms, the reference and the mask, set its local
searches (3.75 deg, 100 orientations) and began its first expectation. Without half
sets (`--K 1 --iter 2`, the script's arguments otherwise) two whole iterations ran on
the exported particles (318 of 319; Extract left one out). They moved the poses by a
median 11 deg, which says nothing either way: these picks hold no particle for poses
to converge on. The gold-standard command is for max (or any machine with a working
MPI), with a real class.

### The assistant (10 November 2026)

The terminal assistant exports on request with the skill `relion-export`
(assistant/plugin/skills/relion-export): the run and class the person names, the
dry run shown with each series' notes, an explicit yes, then the export with
`--reason`. As for submissions, the gate is in the script: every dry run and export
goes into the project's journal.jsonl (scripts/submit_lanes.py's journal), and an
export from an agentic client must be the same person's last dry run of the same run
and class, unchanged, at most 30 minutes old, or it is refused. The dry run prints
what each series' rows fall back on (no dose, no CTF model, order from z values), so
a missing dose is given with `--dose-per-tilt` before anything is written. The
client allows the dry run alone; the export asks the person. The guard blocks RELION
itself (relion_* programs, an export's relion_commands.sh, mpirun, mpiexec): those
run in a job on the cluster, by the person. Not yet tried in a live session.

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
- **Beads** are not looked for: a class with gold beads in it exports them. MRA
  usually puts them in a class of their own; the fiducial model would say where
  they are on bead-aligned series.
- **Runs in Dynamo's layout** (export folders) are refused: the step reads runs of
  the registry, whose setup table names each particle's stack.
- **CTF from a CTF model** is unit-tested only: the local TS_01 project has no
  CTF model, so no RELION run has read a fitted defocus from this export yet.
- **A real class.** The step has run on a class of picks with random orientations
  (below), not on a particle: RyR1 or Gag on max is the first real one.
- **RELION at MDC.** Found 10 November 2026: Environment Modules (no `module
  spider`) list `relion/v5/gpu/sm86-cu11.7` (an A40 build; the group's
  `/fast/AG_Kudryashev/software/relion-5.0.1`), `relion/v4/gpu/sm86-cu11.7` and
  someone's `relion/xchu/v4`, `/v5`; nothing on PATH. `relion_commands.sh` loads the
  module `RELION5_MODULE` names, stops if RELION is not found, and gives Refine3D
  the job's GPUs. `module show`: RELION 5.0.1 (`build-gpu_sm86_cu11.7`), with
  `openmpi/v5.0.10` loaded as a requirement. On the login node max023 its
  `relion_refine --version` fails: `libucp.so.0` (UCX, Open MPI 5's network
  library) is not found. Not yet tried on a GPU node; relion_commands.sh now runs
  `relion_refine --version` first and stops, naming what is missing, if it fails.

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
