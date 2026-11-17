# Entry and exit at any step

Drafted 2026-11-15, revised and agreed 2026-11-16: the same questions for every
case, two of them answered from lists that hold only measured answers,
measurements as checks of the answers, the assistant for the long tail, and the
six decisions under Decided. Order of work items 2 to 4, the imports of
tomograms, picks and particles, are built and committed; item 5, the exits, is
built (Status, at the end).

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
  | Dynamo's table | our voxel floor(p) + shift of the tomogram Dynamo read (columns 24–26 and 4–6) | tdrot, tilt, narot, as the port's MRA reads them | measured 16 November with Dynamo 1.1.511 (`dynamo_table_check.py`): dtcrop centres its box on the voxel holding p, and Dynamo's average puts the particle at that centre plus the shift; as a particle folder, the port's MRA matches Dynamo |
  | IMOD model | `model2point`'s (X, Y, Z) is voxel (X − 0.5, Y − 0.5, Z) of the volume it was made on, in that volume's orientation | none | measured 16 November: findbeads3d on beads of known centre, in IMOD's three layouts, to 0.13 voxel (`imod_model_check.py`) |
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
   - `imported`: entered from outside, with no upstream here. Imported picks
     name the tomogram they are coordinates in as their parent.
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
   does not model, so they are refused until someone needs them. Built 16 November
   (Status below).

9. **Re-entry of an edit.** `--derived-from <artifact>` imports a file as an edit
   of a project artifact that was exported, such as picks after cleaning in ArtiaX
   or 3dmod. Its orientation and format are the export's, so nothing is declared. Its provenance is
   `attested`, its parent is the exported artifact, and its QC says what changed:
   kept, moved, added. The database design's manual steps start here. Built 16
   November for 3dmod edits of `visualize-picks`' model (Status below). ArtiaX's
   saved STAR is not measured yet, and is refused.

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
  particle shares a tilt. Built for positions as `picks_check.py` (Status below):
  - The random positions lie in the picks' own region, with the picks' own
    depths.
  - The picks are also set against their own mirrors along each axis, so picks
    taken in the other orientation of a mirror pair show as such.
  - It cannot see a shift smaller than the box.
  - It cannot tell a z mirror of picks that sit near the middle of the slab, or
    on a lattice.
  - The control with shuffled poses is not built.
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

Built 16 November (Status, Exits): picks or particle stacks to RELION 5
(`scripts/export_relion.py --particles`), a tomogram in a listed orientation
(`cryoet export-tomogram`), and the methods record (`cryoet methods-record`). copick
and CZ portal points are not built.

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
   16 November, committed (dc75610; Status below).
3. `import-picks`: pytom's STAR, RELION 5's STAR, Dynamo's table, IMOD models and
   CSV, with the decoy check and `--derived-from`. Dynamo's table is measured
   against Dynamo's own cropping, and IMOD models against `findbeads3d`'s model
   of the gold beads in one of our tomograms, before either is offered. The
   bypass flags go. Built 16 November, committed (519d3ac, 231a730, 5bc5491,
   97a64c2; Status below):
   - pytom's STAR, RELION 5's STAR, CSV, IMOD models and Dynamo's table, with the
     check;
   - `--derived-from` for 3dmod edits;
   - not yet: edits saved by ArtiaX.

   The bypass flags stay for now. Three ground-truth tools of the benchmark
   (`run_centered_gt_pair.py`, `run_voxel_gt_control.py`,
   `run_gt_positive_control.py`) pass `--picks-star`. Moving them to
   `import-picks` changes their half-voxel path, so it is a decision about those
   controls.
4. `import-particles`: a Dynamo folder and its table. Built 16 November,
   committed (62430de; Status below).
5. Exits: picks to RELION 5, the tomogram in a listed orientation, the methods
   record. Built 16 November, uncommitted (Status below).
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

16 November 2026. Order of work items 2 to 4 are built, tested on the Mac and
committed (dc75610; 519d3ac, 231a730, 5bc5491, 97a64c2; 62430de). Item 5 is built
and uncommitted (Exits, at the end). None has run on max.

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
- Lanes that take tomograms (item 6). The exit twin is built (Exits, below).
- Nothing has run on max or on a colleague's volume yet.

**import-picks, built 16 November, committed (519d3ac).**

- **The command.** `cryoet import-picks` ([picks_import.py](../src/cryoet_pipeline/picks_import.py))
  asks for the picks file, its format from the list, the series and the tomogram
  the coordinates refer to, and the particle's diameter (for the check). Each
  format adds its own questions:
  - `pytom-star`: the template box. pytom's pick lies half a voxel above the
    particle's centre for an even box.
  - `relion5-star`: the rows' `rlnTomoName` and the tilt series' pixel. The pixel
    comes from an optics table, RELION's `tomograms.star`, or
    `--relion-tilt-series-pixel-size`; given `tomograms.star`, RELION's
    tomogram size is checked against the named tomogram's.
  - `csv`: the columns, their units (voxels or angstrom), and where zero is:
    `voxel-0`, `voxel-1`, `corner` or `centre`. There are no defaults: these
    are stated, not assumed.

  Picks taken in a volume of another orientation (`--picked-in`; by default the
  tomogram's own import orientation) are mapped by that orientation's transform,
  and their poses by `M A M`. A mirror in `M` is recorded, because the poses are
  then relative to the picker's reference mirrored.
- **What it writes.** The pipeline's picks STAR, particle centres in the
  tomogram's voxels, in a run folder `imports/<series>/picks[_<label>]/` with
  `run.json` (the picks file and the tomogram as inputs) and `picks_qc.json`. The
  picks are registered with `provenance: imported`, their tomogram as parent, and
  `pick_center_offset_px` 0. Extraction reads that offset in place of a template
  box. Picks without angles carry zeros and `angles: none`; the MRA does not read
  that mark yet.
- **Refused.**
  - Formats not measured: `relion3-star`, `warp-star`, `copick`, each with what
    measuring it needs.
  - RELION 5 particles with nonzero `rlnOrigin*` or `rlnTomoSubtomogram*`: how
    RELION combines them was not measured here.
  - A binning that is not a whole number of the tilt series' pixels.
- **The check** ([picks_check.py](../src/cryoet_pipeline/picks_check.py)). The
  spread of the picks' low-passed average inside its sphere is compared with 19
  sets at random positions in the picks' own region, and with the picks mirrored
  along each axis. Below 3 standard deviations of the random sets is `flat`, and
  refused. A mirror landing 1.2 times better, and itself past 5, is `mirrored`,
  and refused. From 5 the picks `land`. Between 3 and 5 is `weak`: imported, with a
  warning. `--overrule-picks-check` lets a refusal through and the record says so.
- **Tests.** On a synthetic tomogram with particles at known centres
  (`tests/test_picks_import.py`, 17):
  - pytom's picks taken in the pipeline's, IMOD's standard and tilt's raw
    orientation give the centres back to 1e-5 voxel, and the same rotations;
  - RELION 5 picks written with `relion_export`'s own functions import as its
    inverse;
  - CSVs with each origin give the centres back;
  - exchanged axes and a doubled scale are refused, and so are picks taken in
    the mirror orientation;
  - extraction of imported picks runs, and the particle stack's lineage names
    the imported picks and tomogram.
- **On real picks.** 407 Gag picks of TS_01 from the October averaging
  (`outputs/progress-report/gag/fsc`), in the local bin8 TS_01 tomogram. Those
  picks come from the cluster's reconstruction, which may not match this one.
  - As given: 4.3 standard deviations, `weak`. `cryoet import-picks` on them
    (pytom-star, box 24, into a copy of dev-ts01-registry) gives 4.4 and imports
    them with a warning.
  - With x and y exchanged: 0.1, `flat`.
  - Mirrored in z: 5.0. The check cannot tell this mirror for picks on a lattice
    near the middle of the slab.

  The first threshold (the picks' average at 1.5 times the random sets' median)
  would have refused these picks, and so did a null drawn from the whole volume.
  Both were replaced.
- **Not built.**
  - `--derived-from` for edits saved by ArtiaX.
  - The control with shuffled poses.
  - The MRA reading `angles: none`.
  - Moving the benchmark off the bypass flags.

**IMOD models, measured 16 November** (`scripts/imod_model_check.py`, IMOD 5.1.12
on the Mac).

- **Known centres.** Dark beads at exact, partly fractional centres, in one
  canonical volume written in IMOD's three layouts. findbeads3d modelled them,
  `model2point -float` read the models, and the points went through each
  layout's map to the canonical volume.
  - `(X − 0.5, Y − 0.5, Z)` lands on the centres in every layout, at bead sizes 10
    and 18.5 px. The mean is within 0.013 voxel in X, 0.13 in Y and 0.08 in Z;
    findbeads3d's Y is the noisiest, with an sd of 0.2 to 0.3.
  - Without the half pixel, X and Y miss by 0.5, and with it on Z as well, Z
    misses by 0.5.
  - `clip rotx` of tilt's layout is the canonical volume reversed in z, voxel
    for voxel.
- **Round trip.** `point2model` and then `model2point` give the points back.
  `model2point -float` prints two decimals, so its points are good to 0.005
  voxel.
- **TS_01's own gold.** The test the plan named, findbeads3d on our tomogram's
  beads, cannot reach a half voxel. Set against the beads' smoothed density
  minima, findbeads3d's centres differ by −0.6 in x and by 0.4 to 0.9 on other
  axes, depending on the layout. The beads in a reconstruction are not
  symmetric: the missing wedge and the range is asymmetric. So the format was
  measured on beads whose centre is known, and the real beads are reported
  beside it as information.
- **Reading a model.** `import-picks --format imod-model` reads it with IMOD's
  `model2point` and asks `--imod-object` when the model has several objects.
  The points are the volume's voxels in that volume's orientation (`--picked-in`).

**Edits of exported picks, built 16 November** (`import-picks --derived-from`).

- **What it reads.** `visualize-picks` exports a series' picks for 3dmod: the
  canonical tomogram as MRC and every pick as a point of one object, at
  `(x + 0.5, y + 0.5, z)`. It now writes `export.json` beside them, naming the
  picks and the tomogram. A person deletes, moves and adds points in 3dmod and
  saves the model. `import-picks --derived-from <picks> --picks <model>` reads it
  back. The format, orientation and tomogram are the export's, so none of them is
  asked, and naming one is refused. When the export record lies beside the model,
  it has to name the same picks.
- **How a pick is counted.** Points are paired one to one with the exported picks,
  the nearest pairs first:
  - kept: within 0.02 voxel;
  - moved: within the particle's radius;
  - removed: an exported pick with no point near;
  - added: a point with no pick near.
- **What is written.** Kept picks go back to their particles' centres as the
  exported ones would: a pytom pick sits half a voxel above its centre for an
  even template box. A moved or added point is taken as the centre the person
  put it at. Kept and moved picks keep their poses and scores; added ones have
  none, and the set says `angles: partial`. The picks are registered as
  `attested`, with the exported picks as parent and the counts as parameters.
  The QC also lists the removed rows and how far each moved pick went. The check
  against random positions runs as for any import.
- **Tested** with IMOD's own tools on the synthetic tomogram
  (`tests/test_picks_import.py`):
  - `visualize-picks` exports pytom-like picks;
  - the model is edited as in 3dmod (five deleted, five moved by (2, −1, 1),
    three added) and saved with `point2model`;
  - the import counts them right, puts each where it belongs to 0.006 voxel, and
    the lineage reaches the imported tomogram.
- **Refused:** an edit saved by ArtiaX (a STAR file), which is not measured.

**Dynamo's table, measured 16 November** (`scripts/dynamo_table_check.py`,
Dynamo 1.1.511 on the Mac).

- **Read first.** Dynamo's source, `dynamo_table_crop.m` of 1.1.591, says that
  dtcrop crops at the position alone (the shifts play no part). Its box of side
  L runs over the 1-based indices c − L/2 to c + L/2 − 1, with c = ceil(p), and
  "positions are supposed to be expressed in semiintegers".
- **Positions.** dtcrop cropped a noise volume, every voxel distinct, at eight
  positions: whole, semi-integer and other fractions. Each crop was found in the
  volume, equal up to dtcrop's normalisation of each particle to mean 0 and sd 1
  (correlation 1 − 2·10⁻⁷). The axes were neither swapped nor flipped. The first
  voxel was floor(p) − L/2 for all eight. The source's ceil rule fits six: it
  differs on whole positions, which 1.1.511 crops as it crops p + 0.5. So
  Dynamo's box centre is our voxel floor(p), the voxel that holds p. For the
  semi-integers Dynamo documents, that is p − 0.5, the + 0.5 the pipeline's own
  Dynamo export adds.
- **Shifts.** Six crops of one blob, at positions moved off it by d, were
  averaged by `daverage`:
  - with shifts +d, the blob sits at the box centre (8.02, 8.08, 8.07 against
    8), peak 14.8;
  - with −d it is smeared and off centre, peak 5.8.

  So a particle is at the box centre plus the shift.
- **What the import does.** A row is the particle at our voxel floor(p) + shift
  of the tomogram Dynamo read. That volume is the one picked in (`--picked-in`),
  read as an MRC's columns, rows and sections. The pose is `dynamo_to_native` of
  tdrot, tilt, narot, the conversion the port's MRA uses and matches Dynamo
  with. `--dynamo-tomogram` picks the rows of column 20 when the table holds
  several tomograms. Entries Dynamo wrote as complex are read by their real part.
- **Tested** on the synthetic tomogram with tables written as Dynamo keeps them:
  whole or semi-integer positions, the fraction in the shift, and poses from
  `native_to_dynamo`. The particles come back to 1e-5 voxel, with the same
  rotations.

**import-particles, built 16 November, committed (62430de)**
([particles_import.py](../src/cryoet_pipeline/particles_import.py)).

- **What it takes.** A Dynamo data folder (`particle_<tag>.em` or `.mrc`, five or
  six digits) and its table. `--series 1=TS_01` names the project's series of each
  tomogram number in column 20, and an unnamed number is refused. The columns are
  read as Dynamo's `dynamo_columninfo.m` numbers them.
- **What it writes.** Per series, a `PARTICLE_STACK`: the voxels as they are, in a
  Zarr with one chunk per particle, and the pipeline's particle STAR. It lands in
  a run folder `imports/<series>/particles[_<label>]/` with `run.json`, whose
  inputs are the table and the particle files' bytes. The stack is
  registered as `imported`, with no parent.
- **The conversions.**
  - Poses and shifts go through `dynamo_to_native`, the conversion the port's
    MRA matches Dynamo with.
  - The wedge is the table's single-axis ytilt, negated into the pipeline's
    sign as `dynamo_ytilt` negates it on the way to Dynamo (measured 19 October),
    and the stack carries it as `tilt_range_deg`. `reconstructed_tilt_range` now
    takes a stack's own range first, there being no tomogram to read it from.
  - The pixel size is column 36 (apix), checked against `--pixel-size` when both
    are given.
  - The positions are kept as p − 0.5, for the record.
  - The series gets a manifest with `entry: particles` and no views: what its
    particles came from is not here. Such a manifest is the only one allowed
    without images.
- **Its checks.**
  - Every row's file is there, all boxes are cubic and even, one pixel size, one
    wedge per series, and the wedge type is 1. Anything else is refused.
  - **The wedge in the particles.** Boxes cut from one tomogram and not turned all
    hold its wedge in the same place. So the mean of their power spectra is set
    against the table's range, with the same fit and verdicts as for a tomogram
    (`wedge_check.measure_particles_wedge`). A table whose wedge has the other
    sign is refused; `--overrule-wedge-check` lets it through and is recorded.
- **Tested** (`tests/test_particles_import.py`, 8): noise particles with a wedge
  over −50..+60, in the pipeline's convention, and a table of two series giving
  Dynamo's ytilt.
  - After the import, the MRA's own reader of the registry
    (`registry_particles`) gives back the shifts, rotations, ytilt, positions,
    tomogram numbers, pixel size and box the table gave, to the six decimals the
    STAR holds, and the voxels exactly.
  - The wedge check agrees on both series, and refuses the table with ytilt
    negated.
- **Not built.** RELION subtomograms, and particles attached to a tomogram the
  project has. Imported particles are in no tomogram here, so they cannot be
  re-extracted.


**Exits, built 16 November, uncommitted** (Order of work item 5).

- **A tomogram in a listed orientation.** `cryoet export-tomogram`
  ([tomogram_export.py](../src/cryoet_pipeline/tomogram_export.py)).
  - It writes a tomogram of the project as MRC in an orientation from the same
    list the import reads. It undoes that answer's transform, so the list has one
    entry for both directions, and an unmeasured orientation is refused here too.
  - Beside the MRC go the angles the volume was reconstructed with, unchanged, and
    `export.json`. That records:
    - the orientation and how we know it;
    - what was done to the volume;
    - the tomogram and the imported or attested artifacts in its lineage;
    - the `import-tomogram` command that enters the files again.
  - The export is a run with `run.json`, and is never written over.
  - The angles are the tomogram's own `tilt_angles_path`, else the tilt file of
    the aligned stack it was made from, the lookup picking uses. A tomogram with
    neither is refused.
  - **Tested**:
    - the round trip, for every listed orientation: exported, then imported with
      the sidecar's own command, the volume comes back voxel for voxel, with the
      same angles, and the wedge agrees;
    - against IMOD's own `clip`: tilt's output imported as `imod-tilt`, then
      exported, equals `clip rotx` of it as `imod-standard`, `clip flipyz` of it as
      `imod-parallel`, and the file itself as `imod-tilt`.

    `tests/test_tomogram_export.py`, 15.
- **Picks or particle stacks to RELION 5 without an MRA run.**
  `scripts/export_relion.py --project P --particles <ids> [--box B]`
  (`relion_export.read_particle_set`).
  - Each particle enters the class export's conversion at the pose an MRA run of
    its stack would start from:
    - for a stack, the row `registry_particles` gives a run;
    - for picks, what extraction would record, the pick less its centre offset,
      with its angles.

    The dry run, the journal, the gate for the assistant, the tilt series,
    `tomograms.star`, the duplicates and the halves are the class export's.
  - There is no `reference/`. `relion_commands.sh` extracts and, when the picks
    have poses, reconstructs the average at them. It writes no refinement: which
    comes first is a choice about the data.
  - Picks without poses, or with only some, go as positions only, with no
    `rlnAngle` columns, as RELION's own picking writes them.
  - Refused:
    - particles imported with their own table, which are in no tomogram here;
    - picks in an imported tomogram, which has no tilt images here;
    - picks in a denoised tomogram, for now;
    - a series given twice;
    - picks without `--box`.
  - **Tested** (`tests/test_relion_particles_export.py`, 14):
    - a stack's members are the rows an MRA run of it starts from;
    - picks' members are those of their extracted stack, to the six decimals of
      the stack's STAR;
    - where RELION puts a particle does not depend on the box.
  - **Checked by RELION** on 16 November, on the Mac (RELION 5.0 at 7f45a71). The
    407 Gag picks of TS_01 (imported earlier, pytom-star, box 24) went out as 404
    particles, 3 left out as duplicates, box 32, tilt images at the tomogram's
    pixel. `--relion-check` had RELION reconstruct them at the exported poses, and
    set that against the pipeline's own average of the same boxes at the same
    poses (the MRA's `average_class`):
    - the map as written correlates best, NCC 0.514, the next of the eight axis
      reversals 0.100;
    - the offset is 0.04 voxel;
    - 3 boxes leaving the tomogram were left out of the average.

    The class export's commands are unchanged byte for byte.
- **The methods record.** `cryoet methods-record --project P --artifact <id>` or
  `--run <project>/mra/<NAME>` ([methods_record.py](../src/cryoet_pipeline/methods_record.py)).
  - It writes `methods.md` and `methods.json` in a run folder. These hold every
    step the result descends from, parents first. Per step:
    - the backend, the programs' versions as recorded, and the parameters;
    - artifacts that differ only by a trailing number are one step, the varying
      parameters as a range;
    - paths stay in the JSON only.
  - What entered from outside is listed with its declarations, the import's
    content parameters, and the fingerprints of its files. An MRA run adds its
    plan and its nodes' records.
  - It holds nothing the records do not. A version recorded as `external`, and
    the pipeline's own code version where no run record keeps it, are written as
    not recorded. On dev-ts01-registry that is IMOD's version at every step: its
    records predate the version probes.
  - **Tested**: `tests/test_methods_record.py`, 5.
- **Not built.**
  - copick and CZ portal points.
  - Picks in denoised tomograms to RELION.
  - A methods record of an MRA run read from an export folder: its lineage is in
    the registry the export came from.
