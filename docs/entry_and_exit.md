# Entry and exit at any step

Drafted 2026-11-15, revised and agreed 2026-11-16: the same questions for every
case, two of them answered from lists that hold only measured answers,
measurements as checks of the answers, the assistant for the long tail, and the
six decisions under Decided. Order of work item 2, `import-tomogram`, is built
(Status, at the end).

## Purpose

[database_design.md](database_design.md) lists entry and exit at any step among
its requirements. [roadmap.md](roadmap.md) section 5 names the general fix, an
import per artifact kind that converts external files and registers them, and
section 7 the one field such an import must write on the day it exists: whether
an artifact was made here, imported, or attested by a person. That field cannot be
added afterwards. This draft decides what an import is in the registry and in the
run records, which entry points there are and what each needs, which checks an
import makes, and what the exits lack.

The first case to arrive is a colleague who wants the MRA classification, on
GPUs, of tomograms reconstructed elsewhere. It is one case of many. The next
person may bring other files from other programs, so nothing here is shaped by
one case's answers.

## What exists

**Frames and stacks.** `cryoet init` and `prepare-tilt-series` take movies.
`import-tilt-stack` takes an assembled stack with its mdoc. It registers
`TILT_STACK` and `TILT_ANGLES` (backend `external_stack_import`, storage `source`,
`can_recompute` false) and writes the manifest from the mdoc. That the stack's
slices follow the mdoc's ZValues is assumed, not checked (its docstring). A stack
without an mdoc, or one image file per tilt, has no entry.

**The bypass.** The `bypass-external-inputs` branch has been in main since
September (5ed72cf, 99a59d0). `extract-particles --tomogram-mrc M --picks-star S
[--pick-template-box-px T]` goes through
[bypass_external_inputs.py](../src/cryoet_pipeline/bypass_external_inputs.py) and
registers both inputs (backend `external-bypass`, ids
`<TS>:tomogram:external-bypass` and `<TS>:picks:external-bypass`). Read from the
code, it does not do the following:

- **The volume's orientation.** The MRC's sections are taken as z, and nothing
  else about its orientation is asked. A volume straight from IMOD's `tilt`, Y as sections,
  enters with y and z exchanged. One in IMOD's standard orientation (`clip rotx`,
  `trimvol -rx`, RELION 5's own reconstruction) enters mirrored against the
  project's own tomograms. Those are that orientation reversed in z
  ([relion_export.py](../src/cryoet_pipeline/relion_export.py), measured with
  IMOD on 15 October and with RELION on 9 November).
- **The angles.** The tomogram records no tilt angles, so the wedge downstream
  (pytom, the MRA) comes from the manifest, in whatever sign the manifest has.
- **The picks' format.** The STAR is not converted. Extraction reads
  `rlnCoordinateX/Y/Z` as voxels of this tomogram and `rlnAngle*` as pytom's.
  - A STAR at another scale is read without complaint, and its boxes land at the
    wrong places or off the volume. That covers RELION 3/4, in unbinned pixels,
    and Warp, at its own pixel.
  - RELION 5's centred Ångström columns are refused, for want of
    `rlnCoordinateX`.
  - Dynamo tables and IMOD models are not read at all.
- **Picking.** `pick-particles` has no such flag.

One combination needs no change: an external tomogram with picks found in another
volume. The CLI resolves it, and
`test_an_external_tomogram_alone_overrides_what_the_picks_name` pins that, with the
extraction mocked. The backend's `_validate_picks` then refuses it, because the
picks name another tomogram.

**The benchmark's registration.** The script
`cryoet-pipeline-benchmarking/czii_native_pipeline_benchmark/tools/register_external.py`
registers the CZII tomograms through the bypass's function. It writes their
manifests itself, filling the frame fields with stand-ins: `subframe_path` is the
tomogram, `num_subframes` 1, `binning` 1, `source_mdoc` the angle list.
`prepare-template` and `pick-particles` then run on them. That is how picking on
external tomograms runs today. `fetch_czii.py` beside it states its coordinate
contract in its manifest (voxel coordinates on the annotation grid, origin zero).
Every import should have that shape.

**The MRA from a folder.** `mra_setup.read_export` needs only `initial.tbl` and
`particles/particle_*` (MRC or EM), a pixel size in column 36 and single-axis
wedges (ftype 1). `export_manifest.json` is optional. Without it the run knows its
input by the table alone (`mra_runs.export_fingerprint`). So a Dynamo data folder
from elsewhere goes through the port's MRA today, outside the registry. Its
particles are known by their table, not by their voxels.

**Exits.**

- RELION 5, for a class of an MRA run, checked by RELION itself
  (`scripts/export_relion.py`, [relion_export.md](relion_export.md)).
- Dynamo's layout, for a lane's particle stacks (`scripts/export_dynamo.py`).
- Picks over the tomogram for ChimeraX and 3dmod (`visualize-picks`).
- IMOD's own files at every IMOD step: `.xf`, `.tlt`, `.defocus`, and the `.rec`
  as `tilt` wrote it.
- Maps as MRC, and the PDF reports.

## The entry points

What a person can arrive with, from the start of the chain to its end. Each entry
lists what the steps after it read that nothing else supplies. The second line says
what exists today and what can go wrong without an error.

- **Movies with mdocs or names.** `init`.
- **An assembled stack with its mdoc.** `import-tilt-stack`. Silent risk: the slice
  order against the ZValues.
- **A stack with an angle file, or one image per tilt.** Needs the angles, and the
  dose for weighting. No entry.
- **An alignment** (IMOD `.xf` + `.tlt`, AreTomo `.aln`). Needs the transforms in
  IMOD's convention. No entry. Silent risk: the centre the transforms are about.
  pytom's derived low-pass needs a fine-alignment residual that an import does not
  have; `pick-particles` refuses then, which is right.
- **A CTF fit** (IMOD `.defocus`, CTFFIND, Warp). Needs defocus per view, in view
  order. No entry. Silent risk: the view order, and which image, raw or
  aligned, the astigmatism angle refers to.
- **A tomogram.** Needs its orientation (axes, mirror), its pixel size, the
  reconstruction's tilt angles in that orientation, and what was done to it (denoised,
  wedge filled, CTF corrected). Today: the bypass, and picking through the
  benchmark's script. Silent risks: IMOD's y and z exchanged, a mirror (the hand,
  and the side of an asymmetric wedge), a filled wedge.
- **A tomogram with picks** (STAR as each program writes it, Dynamo `.tbl`, IMOD `.mod`,
  copick or CZ portal points, CSV). Needs the tomogram the coordinates refer to,
  their units, origin and scale, the angle convention, whether there are angles,
  and where the centre is. Today: the bypass, pytom's STAR only. Silent risks: the
  scale (a binning), a 0- or 1-based origin, centred Ångström, angles relative to a
  prior, zeros read as poses.
- **Particles** (a Dynamo data folder with its table; RELION subtomograms with a
  STAR). Needs the voxels, the poses, the wedge per tomogram and the pixel size.
  Today: the MRA reads Dynamo's layout, outside the registry. Silent risks: column
  36 is often 0 in Dynamo's tables; RELION 3/4's CTF volumes, which the MRA does
  not model.
- **A segmentation or a mask** (MemBrain run elsewhere, one drawn by hand). Needs
  the tomogram it masks, and has the same orientation questions as a tomogram. No entry.
  The planned lamella mask enters the same way.
- **Even and odd tomograms** (cryoCARE, IsoNet2). Needs the pair. No entry.
- **Template, reference and mask maps.** Need the pixel size, the axis order and
  the hand. Taken as files today (`--source-map`, `[mra] reference`,
  `[mra] mask_file`). Silent risk: the header's axis order, as with EMD-4805's
  3,2,1 (handled in `prepare-template` since 12 November).
- **An edit of an export** (picks cleaned in ArtiaX or 3dmod). Needs the artifact
  it edits. No entry. Silent risk: the edit loses its lineage.

## Same questions, listed answers

Whoever brings the data usually knows how it was made: the program, the
orientation it wrote, what was done to the volumes, the format of the picks. From
one person to the next the answers change, not the questions. So:

- **The questions are the same for every case**, one fixed set per entry point.
  - For a volume: its orientation, its pixel size, where its angles come from,
    what was done to it, and its hand.
  - For picks: the tomogram they refer to, their format, what the format leaves
    open (the pixel size of the coordinates, say), and where the centre is.

  The person answers them. The assistant proposes the answers from the files
  first, as `show_folder.py` does for frames, and the person confirms.
- **Two questions are answered from a list**: the volume's orientation, and the
  picks' format. Each answer on a list says what files it covers, what the import
  does to them, and the measurement that showed it right, with its date. For
  volumes the import is one piece of code that applies the chosen answer's
  transform, written and tested once, whatever the program. For picks there is one
  reader per file type (STAR, Dynamo's table, IMOD's model, CSV), and the answer
  says what the columns mean. The lists as they stand:

  | orientation | the volume | the import | how we know |
  |---|---|---|---|
  | `pipeline` | our own export | nothing | ours |
  | `imod-tilt` | `tilt`'s output, Y as sections (the `.rec`) | `transpose(1, 0, 2)` | what `reconstruct-tomogram` does |
  | `imod-parallel` | `tilt -PARALLEL`, or `clip flipyz` of the `.rec` | nothing | measured 15 October (IMOD 5.1.12) |
  | `imod-standard` | after `clip rotx`, `trimvol -rx` or `tilt -RotateBy90` | z reversed | measured 15 October (IMOD 5.1.12); an etomo reconstruction of 11538 TS_0002 matches it |
  | `relion5` | RELION 5's own reconstruction | z reversed | measured 9 November on TS_01: NCC 0.68, the other seven reversals 0.03 at most |
  | `aretomo`, `warp`, `cz-portal` | | refused | not measured |

  The angles that come with the volume are the ones it was reconstructed with:
  the `.tlt` that `tilt` was given, or RELION 5's `rlnTomoYTilt`. For every
  orientation above they enter unchanged, and only the voxels are transformed
  (5 below says why).

  | format | coordinates | angles | how we know |
  |---|---|---|---|
  | pytom's STAR | `rlnCoordinateX/Y/Z`, voxels of the tomogram picked | `rlnAngleRot/Tilt/Psi` | our own picking and extraction; an even template box's half voxel is accounted for (`pick_template_box_px`) |
  | RELION 5's STAR | `rlnCenteredCoordinate*Angst`: Å from the centre, in `imod-standard` | `rlnAngleRot/Tilt/Psi`, in `imod-standard` | our export writes it, and RELION placed the particles where we did, 10 November (`relion_particle_check.py`) |
  | Dynamo's table | position and shift, in voxels | tdrot, tilt, narot | as a particle folder, the port's MRA matches Dynamo; as positions in a tomogram, to be measured |
  | IMOD model | `model2point`: pixels of the volume it was made on, in that volume's orientation | none | to be measured |
  | CSV | the columns the person names | if named | nothing to measure; the checks below test it |
  | RELION 3/4, Warp, copick | | | not measured, refused |

- **A new program is a new answer, not a new import.** Someone with AreTomo
  tomograms gets no `import-aretomo`. AreTomo reconstructs a tilt series we have,
  its volume is compared with ours over the eight axis reversals, as RELION 5's
  was, and `aretomo` joins the list with that measurement. From then on anyone
  with AreTomo volumes answers `aretomo`. An option that changes the output makes
  its own answer, as IMOD's three show.
- **An answer with no measurement is refused.** Until AreTomo is measured,
  `aretomo` is not accepted, and the import says what measuring it needs.
  Otherwise a belief ("AreTomo writes as IMOD does") would sit on the list, and
  every volume entered with it would come in mirrored without a word if it were
  wrong. Someone who knows their volume is in an orientation already on the list
  answers with that one. It is recorded as their declaration, and the checks test
  it.
- **Measurements check the answers.** A person who knows can still be wrong: the
  other of two `trimvol` options, picks from the bin4 volume declared as bin8. The
  checks below catch the gross cases. Where nobody knows, as with a deposition
  that does not say, the measurement is the only answer there is. What it cannot
  settle, the hand above all, is recorded as unknown.
- **The long tail goes through the assistant.** For a format not on the list, the
  assistant writes a script that converts it into a neutral, declared form: a CSV
  with stated columns and conventions, or an MRC in a listed orientation. The
  tested import reads that form. The script's hash goes into the import's record,
  and the result passes the same checks, with its angles dropped by default. A
  format that comes back gets its own answer on the list, measured like the
  others.

## What an import is

1. **An import is a run.**
   - Steps `import-tomogram`, `import-picks`, `import-particles`, and so on. Each
     writes `run.json` as [run_records.md](run_records.md) has it.
   - Its inputs are the external files, each with its absolute path and the sha256
     of its bytes. `RunInput` already allows an absolute path. The hash costs one
     pass over the file, which the copy into the project makes anyway.
   - Its content parameters are what the person declared about the files (below).
   - Its outputs are the canonical artifacts. They are registered in
     `artifacts.json` as now, so every later step finds them the usual way.

   The same file with the same declarations gives the same fingerprint, so
   importing it twice does nothing. The same file with another declaration, the
   other hand say, is another node. Both can be carried on and compared. The
   imports would be the first registry steps to write run records; the MRA is the
   only writer now.

2. **The artifact says it was imported.** A field `provenance` on `Artifact`, with
   three values:
   - `derived`: made by a run of this pipeline from registered inputs. This is the
     default, so every registry on disk stays readable.
   - `imported`: entered from outside, with no upstream here.
   - `attested`: a person's edit, made outside, of an artifact that is here.

   It is not called `origin`, because roadmap section 7 gives that name to the
   run's human, agent or system. What descends from an imported or attested
   artifact is not marked one artifact at a time. It is a question about the
   lineage (is any ancestor not derived?), answered by `ArtifactRegistry` now and
   by the closure table in the database, and every report asks it. An average made
   from an imported tomogram says so on its first page, with what was declared at
   the import. In old registries, `external-bypass` and `external_stack_import`
   artifacts read as imported.

3. **Declared and measured, checked against each other.** An import records two
   things.
   - What it was told: the source program and version, the orientation, the
     processing, the hand.
   - What it measured: the shape, the header's pixel size and axis map, which axis
     is thin, the wedge seen in the volume's Fourier transform, the extent of the
     coordinates.

   A declaration that a measurement can test is tested, and a disagreement refuses
   the import. A declaration no measurement can test, the hand above all, stays a
   declaration. It is named as one wherever it matters: template matching with a
   chiral template, a comparison with an EMDB map.

4. **One canonical orientation, entered from the list.** The canonical tomogram
   is ZYX: `tilt`'s output transposed, which is IMOD's standard orientation
   reversed in z, a mirror image (measured 15 October; relion_export.py). An
   import names the orientation its volume is in, from the list above, and
   applies that answer's transform to the voxels and its sign to the angles. A
   volume of unknown orientation can be imported only when the wedge measured in
   it settles the axes, and its hand is then recorded as unknown.

5. **The tilt geometry travels with the volume.**
   - **The sign.** A mirror of z, or of x, turns a view at +θ into one at −θ, when
     the angles are read in one convention fixed to the volume's axes. No program
     hands angles over that way. IMOD's `.tlt` and RELION 5's `rlnTomoYTilt` are
     the angles a volume was reconstructed with, whatever orientation it was
     turned into afterwards. For each orientation on the list, the transform gives
     the volume the pipeline itself reconstructs with those same angles:
     - for IMOD's three, by construction: the same `tilt` run, its outputs
       related as measured on 15 October;
     - for RELION 5, as measured on 9 November: given the `.tlt` angles, RELION
       rebuilds the pipeline's tomogram reversed in z, and given them negated,
       the pipeline's own.

     So the angles enter unchanged. A volume mirrored by someone after its
     reconstruction is a different case: its angles would change sign, and an
     asymmetric range (TS_0306's tomogram holds −40..+58) would move to the other
     side. Such a volume is not on the list. A symmetric range is unaffected either
     way. The draft agreed earlier on 16 November had `imod-standard` and `relion5`
     negate the angles. That was wrong, and it was corrected the same day.
   - **Where the angles go.** The import writes them, in the canonical orientation, as
     the tomogram's own `tilt_angles_path`. Reconstruction writes that field, and
     pytom and the MRA already read it first.
   - **Their source.** The reconstruction's own angle file (`.tlt`, AreTomo's
     `.aln`); else the mdoc's nominal angles, said to be nominal; else a stated
     range.
   - **What was done to the volume** is a declaration with consequences. A
     wedge-filled volume (IsoNet, DeepDeWedge) must not be wedge-compensated, so
     the MRA refuses it, or runs it with a full range by an explicit choice. A
     denoised volume is fine to pick on. The averaging then wants the raw volume
     of the same series, imported beside it as its pair (`DENOISED_TOMOGRAM`
     exists).

6. **A series without frames.** Every step takes `--manifest`, and `TiltImage`
   requires a frame path, a frame count, a pixel size and a binning. That is why
   the benchmark writes stand-ins. Proposed:
   - those four fields become optional;
   - the manifest gains `entry` (`frames`, `stack`, `tomogram`, `picks`,
     `particles`), and keeps the angles, and the dose and order when an mdoc gives
     them;
   - a step that needs frames refuses a series without them, and says so.

   This is the audit roadmap section 5 asks for: what each step quietly needs that
   only the frames path produces. Here it is done once, for every entry, rather
   than found one step at a time.

   An imported series may also be one the project already has, its frames entered
   through `init`. The import then attaches to it, after checking that the angles
   agree and that the pixel sizes differ by a binning. Two tomograms of one series
   can then be compared under one set of picks: an external reconstruction as a
   control of ours.

7. **Picks are coordinates in a named tomogram.** An import of picks names the
   registered tomogram its coordinates refer to, and states:
   - their units: voxels of which pixel size, or Ångström;
   - their origin: 0- or 1-based, corner or centre, or centred on the volume as in
     RELION 5;
   - the angle convention, and whether there are angles at all;
   - where the particle's centre is. pytom's even template box puts it half a voxel
     off. For other pickers the centre is to be stated, not assumed.

   It writes the pipeline's own picks STAR, in that tomogram's voxels, the only
   form extraction reads. An external STAR is never handed on. Picks without angles
   are marked so, and the MRA's starting poses then come from its global rounds,
   not from zeros read as poses.

   The formats are the list above: first pytom's STAR, RELION 5's STAR, Dynamo's
   table, IMOD models and a CSV with stated columns; then, each once measured,
   Warp, RELION 3/4, copick and CZ portal points. Picks made in another tomogram of the same series (bin8 picks for a
   bin4 extraction) are not an import. They are a transfer between two canonical
   tomograms, a step of its own.

8. **Particles.** A Dynamo data folder and its table become a `PARTICLE_STACK`:
   Zarr, one chunk per particle, with the pipeline's particle STAR. Each series'
   wedge comes from the table's columns. The MRA then reads them from the registry
   with its usual reports, and the particles are known by the hash of their voxels,
   not by the table alone. RELION 3/4 subtomograms come with CTF volumes the MRA
   does not model, so they are refused until someone needs them.

9. **Re-entry of an edit.** `--derived-from <artifact>` imports a file as an edit
   of a project artifact that was exported, such as picks after cleaning in ArtiaX
   or 3dmod. Its orientation and format are the export's, so nothing is declared. Its provenance is
   `attested`, its parent is the exported artifact, and its QC says what changed:
   kept, moved, added. The database design's manual steps start here.

10. **The bypass retires into the imports.** `--tomogram-mrc` and `--picks-star` go
    once `import-tomogram` and `import-picks` exist. Registries holding their
    artifacts stay readable, as imported, orientation unknown. The benchmark's
    `register_external.py` moves to `import-tomogram --orientation cz-portal` once
    that orientation is measured.

## Checks an import makes

Each is independent of what was declared, because a guard built from the numbers
it checks is an identity.

- **The wedge in the volume.** The amplitude of the tomogram's 3D Fourier
  transform is low inside the missing wedge, in the plane across the tilt axis.
  The wedge's edges give the tilt axis's direction and the range, its side
  included when the range is asymmetric. That is compared with the declared
  angles, brought into the canonical orientation. A filled wedge shows as one, which
  tests the processing declaration too. Built as `wedge_check.py` and tried on
  16 November where the answer is known (Status below). On the pipeline's own
  TS_01 tomogram it finds the range's centre to within 0.6°, and on the views up
  to +16° to within 0.3°. It reads a z- or x-reversed copy as mirrored and a
  y-reversed one as agreeing, as the geometry predicts. Not tried yet on denoised
  volumes, on lamellae, or at other binnings.
- **Picks land on particles.** The average of boxes at the picks is compared with
  averages at random positions in the same tomogram, the null of
  `scripts/decoy_picks.py`. A wrong scale or exchanged axes give an average as flat
  as the decoys'. With angles, the average at the stated poses is compared with one
  at shuffled poses. That control has a known limit: it is blind when every
  particle shares a tilt.
- **Round trip.** Export to the format, import back, and get the same artifact:
  voxels equal, positions within 0.01 voxel, poses within 0.01°. Necessary, but
  circular: an export and an import that both mirror pass it together.
- **External program.** For each orientation on the list, the same tilt series is reconstructed
  by the external program, imported, and compared with ours over the eight axis
  reversals. The predicted one must be best, by twice the next, as
  `scripts/relion_geometry_check.py` does for RELION. This check can fail when the
  round trip passes.

## Exits

The exits cover what has been asked so far. An export is data leaving, so by the
database design's rule an exporter can even be generated on demand. An importer
cannot be, because what enters informs decisions. Gaps worth closing:

- **Picks or particle stacks to RELION 5 without an MRA run.** The export now
  starts from a class. This serves someone who leaves after picking.
- **A tomogram as MRC in a listed orientation** (IMOD's standard), with the
  orientation in a sidecar, for napari, ChimeraX or another picker. The `.rec` is `tilt`'s raw
  output, made for 3dmod. This is also the round-trip twin of `import-tomogram`.
- **The lineage of a result as a methods record.** Every run from the raw data, or
  from the imported roots, to the result, with programs, versions and parameters.
  It is what a paper's methods section needs, and the registry and the run records
  hold it.
- **copick or CZ portal points**, when something is to be deposited or shared.

Every importer gets its exporter twin, for the round trip.

## A worked case: tomograms from elsewhere to the MRA

This is the first case to arrive, and an example of the questions and lists above
rather than their driver: classification of tomograms made elsewhere. Today it runs with the
benchmark's script as the pattern:

1. register the tomograms and write their manifests;
2. `prepare-template`;
3. `pick-particles`, with `--pytom-low-pass` given (without a fine alignment it
   cannot be derived, and the command refuses rather than guess);
4. `extract-particles --variant L`;
5. the MRA from the registry: `SOURCE`, `LANE=L`, `SERIES`.

That is right if three things hold:

- the MRC's sections are z, so not `tilt`'s raw output;
- the volume is in the pipeline's orientation (`tilt -PARALLEL`, `clip flipyz`),
  or else its range is symmetric. A volume in IMOD's standard orientation enters
  mirrored, and with the `.tlt` angles its asymmetric wedge lands on the wrong
  side;
- the wedge is not filled.

The hand matters for pytom with a chiral template, not for the MRA's own classes.
With picks of one's own, the bypass reads only pytom's STAR, in the tomogram's
voxels. With Dynamo particles and a table, the port's MRA reads the folder (column
36 filled), outside the registry.

With the imports, this case is the questions above, answered once:

- which program made the tomograms, and in what orientation;
- the angle files;
- whether the volumes are denoised or wedge-filled;
- the picks' format and pixel size;
- the particle and its template.

The measurements check those answers, and the chain from `pick-particles` on is
unchanged.

## Order of work

1. This document, agreed 16 November.
2. `import-tomogram`, testable on the Mac with IMOD and dev-ts01-registry:
   - the orientations `pipeline`, `imod-tilt`, `imod-parallel`, `imod-standard` and
     `relion5`;
   - the series without frames;
   - `provenance` and `run.json`;
   - the wedge check.

   These five are measured already, and cover IMOD's and RELION 5's users. Built
   16 November, uncommitted (Status below).
3. `import-picks`: pytom's STAR, RELION 5's STAR, Dynamo's table, IMOD models and
   CSV, with the decoy check and `--derived-from`. Dynamo's table is measured
   against Dynamo's own cropping, and IMOD models against `findbeads3d`'s model
   of the gold beads in one of our tomograms, before either is offered. The
   bypass flags go.
4. `import-particles`: a Dynamo folder and its table.
5. Exits: picks to RELION 5, the tomogram in a listed orientation, the methods
   record.
6. Lane inputs `tomograms`, `picks` and `particles`, and the assistant's readers
   that propose the declarations from the files, as `show_folder.py` does for
   frames.
7. Then, as asked: the orientations `aretomo`, `warp` and `cz-portal`, and the
   picks of RELION 3/4, Warp and copick, each once measured; alignment, CTF,
   segmentation and even/odd imports; stacks without an mdoc.

## Decided

All six agreed on 16 November.

1. **Run records.** Imports write `run.json` from the start, the first registry
   steps that do. Section 7 says attribution cannot be backfilled, and an import's
   inputs are the one thing nobody can recompute.
2. **`provenance` on `Artifact` now.** Its downward class is computed from the
   lineage, not stored.
3. **The manifest's frame fields optional**, with `entry`, rather than stand-ins.
4. **A volume of unknown orientation** is accepted when the measured wedge settles
   its axes, with its hand recorded unknown, once the wedge check is shown to work.
   Until then it is refused.
5. **Copies, not references.** Volumes are copied into the project, which the Zarr
   needs anyway. Picks and particles are converted. Nothing is referenced in place.
6. **The first slice is the mechanism with the most common answers.** That
   means the orientations of IMOD and RELION 5 on the list, and, for picks,
   pytom's and RELION 5's STAR, Dynamo's table, IMOD models and CSV. AreTomo,
   Warp, the CZ portal and RELION 3/4 follow, each once measured. No single case
   decides the order.

## Open

- **The CZII ground truth sits about one voxel off in z:** +2.6, −2.1, −10 Å,
  constant in the tomogram's axes. A mirror made about N/2 by one program and about
  (N − 1)/2 by another moves everything by exactly one voxel along the mirrored
  axis and by nothing along the others. That is a hypothesis for the z part, not
  checked. It is worth testing when the `cz-portal` orientation is measured.
- **`import-tilt-stack`'s slice order.** It assumes the stack's slices follow the
  mdoc's ZValues.
- **The canonical orientation's own hand.** It is IMOD's standard orientation, mirrored.
  Whether that is the specimen's true hand is a separate question (RELION's
  `rlnTomoHand` is determined by experiment). An import cannot be more right about
  the hand than the orientation it maps into.

## Status

16 November 2026. Order of work item 2 is built and tested on the Mac. It is
uncommitted and has not run on max.

**Built.**

- `cryoet import-tomogram` ([tomogram_import.py](../src/cryoet_pipeline/tomogram_import.py)).
  - It asks for the volume, its angles (IMOD's `.tlt`, or RELION 5's tilt-series
    STAR), its orientation from the list, what was done to it (`--processing raw`,
    or a set of `ctf-corrected`, `denoised`, `wedge-filled`) and the series id.
  - The pixel size is read from the header. A `--pixel-size` given as well is
    checked against it.
  - Before anything is copied, it refuses a header whose axis order is not 1, 2, 3,
    a volume whose z does not come out its thinnest axis, and a volume the wedge
    check contradicts. `--overrule-wedge-check` lets the last through, and the
    record says so.
  - The run writes `imports/<series>/tomogram[_<label>]/`:
    - `run.json`, with the volume and the angle file as inputs (absolute path and
      sha256 of the bytes) and the answers as content parameters;
    - the canonical Zarr, written as reconstruction writes its own;
    - the angles, unchanged;
    - `import_qc.json` and three central slices.
  - The same files with the same answers are imported once. Other answers about
    the same files need `--variant`.
- **The registry.** The tomogram is registered with `provenance: imported`,
  `tilt_angles_path` and `included_z_values`, so picking and the MRA find it as
  they find their own. `ArtifactRegistry.not_derived_lineage` answers whether
  anything upstream of an artifact was imported or attested. The artifacts of
  `external-bypass` and `external_stack_import` in old registries read as
  imported.
- **The manifest.** Its frame fields are optional, and `entry` says what the
  series entered as. `import-tilt-stack` now writes `entry: stack`. A series
  entered as frames still needs all four fields. The twelve steps before a
  tomogram refuse a series whose entry lacks what they read.

**Measured.** On 16 November with `scripts/import_tomogram_check.py`, IMOD 5.1.12
on the Mac, EMPIAR-10164 TS_01 at bin8 (a slab of 528 slices), twice: on all 40
views (−53.5..+63.8), and on the 24 views up to +16°.

- **The orientations.** Six ways IMOD writes the volume give the same canonical
  volume, with a largest difference of 0:
  - `tilt` as it is, as `imod-tilt`;
  - `-PARALLEL` and `clip flipyz`, as `imod-parallel`;
  - `-RotateBy90`, `clip rotx` and `trimvol -rx`, as `imod-standard`.
- **The wedge.** It agrees with the `.tlt` unchanged in every case. Fitted centre
  against the declared centre and its mirror's:

  | views | fitted | declared | mirror |
  |---|---|---|---|
  | all 40 | +4.5 | +5.1 | −5.1 |
  | up to +16° | −18.5 | −18.8 | +18.8 |

- **A mirror answered as its pair.** The `clip rotx` volume answered `imod-parallel`
  is refused as mirrored.
- **Plain noise** shows a contrast of 0.03 at most, against the 0.15 below which
  the check says the wedge is filled.

**Not built, or not checked.**

- Attaching an import to a series the project has from its frames.
- An mdoc as the angle source. So an imported series picks without dose weighting.
- A stated range as the angle source.
- Volumes of unknown orientation (decision 4): the wedge check has been tried only
  on raw bin8 reconstructions.
- `relion5`'s answer was not measured again; that needs RELION, which ran here on
  9 November.
- The exit twin, and lanes that take tomograms (items 5 and 6).
- Nothing has run on max or on a colleague's volume yet.
