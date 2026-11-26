# Results in a browser: from PDF reports to pages

Drafted 2026-11-26 from a discussion with Anna on 2026-11-25 and 2026-11-26. A
prototype of step 1, with Mol\* for the template, the classification runs and
their FSC, was built the same day (Status, at the end).
What was agreed and what is only proposed are listed apart, at
the end, with the open questions.

## Purpose

The reports are PDFs: `cryoet report` for a series, the classification report for
an MRA, `scripts/show_tilts.py` for raw tilts. They show fixed pictures chosen in
advance: a tomogram at three depths, picks in score bands, an MRA's classes as
runs of z sections. What a PDF cannot do is let the reader page through a
tomogram, move a score threshold and watch which picks stay, or turn a map.

The aim is pages that show what the reports show, interactively. In the long run
they grow into the GUI for lanes, which is an existing goal; the way there is a
series of steps, each useful on its own.

Work happens on the cluster, the only place with NVIDIA GPUs, and the results
live there. So the pages are read from the cluster, not from copies downloaded
after each run.

## What exists

**What the PDF draws, and with what.** `reports/collect.py` reads the registry
into one `SeriesReport` per series (artifacts, QC records, picks, CTF, alignment,
template). `reports/imaging.py` makes the pictures: `slab_mean` over slabs
`SLAB_ANGSTROM` (100 Å) thick, `display_range`, `block_mean` down to
`MAX_DRAWN_PX` (512), `sphere_sections`, `central_views`, `crop_centered`. The
sections use them: `sections/tomogram.py` shows the tomogram at three depths and
from two sides, `sections/picking.py` the picks by score band,
`sections/gallery.py` picks turned into the template's frame by their angles,
each the mean of a slab about 10 nm thick. `reports/classification.py` draws an
MRA from `ClassificationRun`, which `reports/classification_reader.py` fills from
a Dynamo project or from run records.

**Slab stacks already exist for one volume.** `score_preview.py` keeps pytom's
score map as maxima over slabs one particle diameter thick: megabytes in place of
a gigabyte per series at bin8, read one slab at a time through a memory map. The
tomogram pages here would do the same with means.

**Viewers outside the pipeline.** `picks_visualization.py` writes one folder for
ChimeraX with ArtiaX (the whole set, each pick drawn as the template at its
orientation) and for 3dmod (one slice, to see whether a pick sits on density). Its
docstring records two silent coordinate disagreements: ArtiaX's two `.star`
readers, and IMOD models adding half a voxel in X and Y but not in Z.

**The state of a lane, as data.** `scripts/show_project.py --json` gives, for
every series of a lane, each step's last run and QC verdict, the warnings and the
lane's Slurm jobs, from `step_timings.jsonl`, the registry and the Slurm logs, with
sacct and squeue where they can run. `scripts/show_failure.py --json` gives the
lines that matter from the logs of failed steps. Both are read-only, and both are
what the terminal assistant reads. `show_project.py` calls itself a stand-in until
the run entity of [roadmap.md](roadmap.md) section 7 records all of this.

**How a project is changed.** `submit_lanes.py` and `submit_recipe.py` show a dry
run, submit on a yes, take a `--reason`, and append to the project's
`journal.jsonl` ([agent_architecture.md](agent_architecture.md), "The journal").
The registry is written under `fcntl.lockf` (`artifacts.py`).

## The ladder

1. **Results of one series.** Tomogram slabs, picks over them with a score
   threshold, the template and the averages in 3D. Read-only.
2. **Pages that follow a running lane.** The result pages update as artifacts
   appear, and a lane page shows `show_project` and `show_failure`. Read-only.
3. **Actions.** Rerun a step, change a threshold, submit a lane. This is where the
   GUI begins.

Steps 1 and 2 only read, so the large jump is step 3, not monitoring. Nothing in a
later step requires redoing an earlier one, provided one decision is taken now:
**the page draws itself from data** (JSON and images) produced by Python, and is
never HTML with values written into it. Then step 2 is the same data, fetched
again, plus one more page over JSON that already exists.

## Where the pages live

**A server on the cluster is the main way.** It runs in the pipeline's
environment, in a slogin job or a small sbatch job of its own when it should
outlive a session (a running lane takes hours), and prints the line to open it
from a laptop (`ssh -L` with the node and the port). With a server, the page
fetches its data, nothing is copied, and following a lane costs nothing extra.

- **Access.** A port open on a compute node can be reached by every user of the
  cluster, and the data are unpublished. Either a token in the URL, as Jupyter
  does, or listening on 127.0.0.1 only and tunnelling to the node itself through
  the login node (`ssh -J`), where the site lets a user ssh to a node running
  their job.
- **Perhaps no tunnel.** Where a site runs JupyterHub or Open OnDemand, the server
  could be reached through their proxy, behind their sign-in. Whether max or
  cryoem.se has either is not known (open question 1). VS Code Remote-SSH forwards
  ports by itself, for whoever uses it.
- **In steps 1 and 2 the server writes nothing in the project** but its own cache
  of images, and never takes the registry lock.

**A static export is the copy.** The same pages and data in one folder that opens
from disk: for the PI, for the archive, for a site where no tunnel can be had. A
page opened from disk (`file://`) may load `<img>` and `<script src>`, but the
browser refuses `fetch()`; so the export writes its data as `.js` files that set a
variable, and the images as files. The PDF stays as it is for now (open question
7).

## Libraries, and a node without internet

Every node used so far has had internet, but a fallback is needed. The proposal
is to have no CDN path at all: the installation fetches each library once, at a
pinned version with its sha256 (as the reference library checks its sources
against `configs/library.toml`, [library.md](library.md)), and
the server serves its own copies. The laptop then needs no internet, there is one
code path rather than a primary and a fallback, and the version in use is the one
tested. Where the node has no internet at installation, the install checklist
gets one item: copy the folder of libraries. The static export copies them into
itself.

Candidates; Mol\* was chosen and is in the prototype (Status):

- **Mol\*** for maps. It reads MRC/CCP4/MAP directly and shows isosurfaces, slices
  and volume rendering; EMDB's entry pages use it. One JS bundle of a few MB, no
  build step. Fit for templates, masks, averages and classes (a 60³ box in
  float32 is under 1 MB); not for whole tomograms, which at bin8 in float32 are
  hundreds of MB. Its API loads from a URL; loading from bytes, which the static
  export needs, is to be checked in the prototype.
- **NGL**, lighter, also reads MRC.
- **A chart library** for residuals, defocus, class sizes, FSC, score histograms
  (open question 2).
- **Neuroglancer**, later and only if bin8 slabs fall short: full-resolution
  browsing of bin4 with picks as annotations; the CZII data portal is built on it.
  It reads OME-Zarr over HTTP, and zarr is already a dependency here.

## Step 1: what the pages show

**One component does most of it: a stack of images, a slider, points on top.** It
serves the tomogram slabs, the score map's slabs, the raw tilts (as
`show_tilts.py` reads them), the CTF power spectra per tilt, and an MRA's classes
over the iterations.

- **Tomogram.** A stack of slab means through the whole depth, drawn by
  `reports/imaging.py` as the PDF draws its three, so a slab on the page and the
  same slab in the PDF are one picture. Slab thickness about a particle radius
  when the lane has a particle, `SLAB_ANGSTROM` otherwise, with the step equal to
  the thickness so that every particle lies whole in at least one slab (open
  question 4). Side views (XZ, YZ) as well: they show the lamella and the depth of
  the picks, an interactive form of the z-spread check. Brightness and contrast
  can be moved in the browser on top of the stored range; a caption says what was
  done (filter, slab, range) so that a slab is not taken for a raw slice.
- **Picks.** Circles where the particle's sphere cuts the slab
  (`sphere_sections`), coloured by score. A histogram of scores with a threshold
  slider, tied to the circles and to a count of picks above it. A click on a pick
  shows its central views turned into the template's frame, as the gallery does,
  with score and coordinates. The page writes nothing: a chosen threshold is
  shown as the line for the lane's TOML, which a person pastes, or hands to the
  assistant with its dry run, so that the journal stays the one record of changes.
- **Score map.** The slabs `score_preview.py` already keeps, on the same slider.
- **Template and mask.** Central sections and Mol\*, the mask translucent.
- **Alignment and CTF.** Residuals per view, defocus per tilt as charts; spectra
  per tilt on the slider.
- **MRA.** Classes per iteration on the slider, class sizes over iterations, Mol\*
  per class, FSC with its 0.143 crossing.
- **Run times.** From `step_timings.jsonl`, read by `reports/timing.py` as for
  the PDF.

**Where the images come from.** Step 1 needs no change to the pipeline: the
server cuts a series' images on first request from the artifacts the registry
names, keeps them in a cache, and cuts again when an artifact is newer than its
images. Reading a bin8 tomogram slab by slab through a memory map takes seconds.
Where the cache lives, and under which storage class, is open question 3.

**Sizes.** Estimated first: a 5760×4092 tilt at bin8 gives about 720×512 in the
plane; for a tomogram 300 sections deep, slabs of 10 sections are 30 images,
about 1.5 MB at ~50 KB each; every section is about 15 MB; the volume itself in
uint8 about 110 MB, which no page should carry. Measured on 2026-11-26 on the
Mac, with the prototype's PNGs on dev-ts01 (TS_01 fine, 296×928×960 at 5.4 Å,
about 1 GB in float32): 16 XY slabs of 19 voxels and 49 XZ slabs, 65 pictures,
23 MB, about 350 KB each, cut in 3.9 s and read from the cache in 0.01 s. PNG is
seven times the JPEG guess; JPEG or WebP is the obvious change if a tunnel finds
23 MB slow.

## Checks before believing a page

- **The coordinate frame of the circles.** The project has had mirrored
  tomograms (transpose equal to `tilt -PARALLEL`) and a half-voxel offset, and a
  wrong overlay fails silently. The circles use the same transform as the PDF's
  picking pages, and are checked against a truth that does not come from them: a
  phantom with a bead at a known voxel, and CZII ground truth with its known
  offset of about one voxel in z. Circles agreeing with the PDF's circles is not
  that check.
- **Mol\* and our headers.** That it honours the axis order (`mapc/mapr/maps`;
  EMD-4805 has 3,2,1), the origin, the modes we write, and the hand: an
  asymmetric phantom map, opened in Mol\* and in ChimeraX, must look the same.
- **Contrast.** The caption on every stack says what was done to it.

## Step 2: following a lane

The server notices new or newer artifacts and the open pages fetch a small status
document and refresh what changed. A lane page draws `show_project.py --json`; a
failed step opens `show_failure.py --json`. Both call sacct and squeue, so the
server caches their answer and asks at most once a minute, whatever the number of
open pages. When the run entity replaces `show_project.py`, only the source behind
the lane page changes.

## Step 3: actions, direction only

Every action is an existing command: the page shows its dry run, submits on a
confirmation with a reason, and the journal gets the entry the terminal would
have written. The page never edits a TOML or the registry itself, so the project
keeps one way of being changed, whoever changes it: a person in a terminal, the
assistant, a button. The server then writes, so access matters more than in steps
1 and 2: one user per server. Not designed further here.

## Agreed

On 2026-11-25 and 2026-11-26:

1. **Iterative.** Results first, then following a lane, then actions; each step
   useful on its own.
2. **Read from the cluster.** Results are not downloaded to be looked at.
3. **Internet is usually there**, on every node so far, **but a fallback is
   needed** for where it is not.

## Proposed, not yet agreed

1. A server on the cluster as the main way; a static export as the copy.
2. Pages drawn from data, never HTML with values written in.
3. No CDN: libraries pinned with sha256, fetched at installation, served by the
   server.
4. In steps 1 and 2 the server writes nothing in the project but its image cache.
5. Images made by `reports/imaging.py`, the PDF's functions, not by new ones.
6. A threshold or any other choice made on a page leaves it as a line to paste,
   until step 3; in step 3 every action goes through the existing commands, dry
   run first.

## Open questions

1. Does max, or cryoem.se, run JupyterHub or Open OnDemand?
2. Which chart library: Plotly (large, complete), uPlot (small, line charts),
   Vega-Lite (declarative, moderate)?
3. Where the image cache lives and its storage class
   ([storage_policy.md](storage_policy.md)).
4. Slab thickness: about a particle radius, or `SLAB_ANGSTROM` everywhere?
5. One server per user for all their projects, or one per project?
6. A placed-back view in the browser (the average at every pick's pose, as ArtiaX
   shows it), for lattices such as Gag's: after step 1, with its own check of the
   angle convention, a turned average compared with the particle's crop.
7. The PDF: kept for now; once pages exist, see whether anyone still opens it.

## Status

**2026-11-26: a prototype of step 1, without Mol\*.** `scripts/serve_results.py`
starts the server (`src/cryoet_pipeline/dashboard/`: `data.py` cuts and places,
`server.py` serves with the token, `static/` is the page). For each series: XY
slabs through z and XZ slabs through y on sliders, each view marking where the
other's slab cuts it; the picks as sphere sections coloured by score; a score
histogram (in σ where the search recorded its noise) with the one-false-peak
height, a threshold slider, and the `[pick] cut_off` line with its count; picks
stepped through by score or clicked, each shown as the PDF gallery's tile, beside
the template's. No library: plain JS and canvas.

The tomogram, picks and template are resolved by the report command's own
`_report_artifacts`, a private function of `cli.py` imported as it is; an
unlabelled registry with two tomograms (dev-ts01: coarse and fine) is refused as
the PDF refuses it.

Checked: on test_report's synthetic project, whose first 24 picks are dark blobs
written into the volume, every circle cut through its centre sits within a
pixel of its blob's darkness centroid in the PNG, with no common shift (mean
under 0.02 px), at full size and reduced three times, in XY and XZ
(tests/test_dashboard.py, 13 tests; 7 of 7 mutants caught: either half voxel,
either flip, rows counted from the volume after reduction, a slab band's edge,
any token accepted). In the browser, the canvas under those circles' centres
read 0 of 255 at all 18 on view, the decoys' about 200 against a picture mean of
about 180. Not
checked on real picks: there are none on the Mac; the first run on the cluster
is the test.

**Mol\*, the same day.** Mol\* 5.13.1 (MIT, on npm since 2026-10-08), its viewer
build only: `molstar.js` (5.3 MB) and `molstar.css`, pinned in
`configs/viewer_assets.toml` by the sha256 jsdelivr's package listing gave, and
fetched by `scripts/fetch_viewer_assets.py` into `vendor/` (git-ignored), whose
downloads matched those hashes. `--from <folder>` copies them on a node without
internet, under the same check. The server serves them and the page loads them
only when asked (a button), so slicing never waits for 5 MB. It shows the
template's own file at 1σ on the protein's side and the mask at 0.5,
translucent; Mol\*'s own panel changes the levels. The protein's side is
measured, the template's mean inside its mask against outside it, and falls back
to prepare-template's record of inverting it only without a mask of the same
shape. The server hands out only the template and mask files the series
resolves to, by those two names.

Checked against the MRC format's own definition rather than another viewer: a
tripod phantom, box X 16, Y 14, Z 12 at 10 Å, arms along +x (long), +y and +z
(short) ending in voxels of 10, 20 and 30, written once in the standard axis
order and once with X slowest (mapc, mapr, maps = 3, 2, 1, as EMD-4805). Read
back from Mol\*'s parsed volume in the browser, both files gave a grid of
16 × 14 × 12 along X, Y, Z, a cell of 160 × 140 × 120 Å from origin 0, and the
three markers at (13, 4, 3), (4, 10, 3) and (4, 4, 7), where they were written.
Drawn, the long arm points right, the medium one up and the short one at the
viewer, as Mol\*'s axis indicator has x, y and z: a right-handed reading. Not
checked: a map with a non-zero origin or start, and the same file in ChimeraX.
Tests: 17 in test_dashboard; 10 of 11 mutants caught, the eleventh equivalent
(a map name outside the two is refused a second time further on).

**Classification and FSC, 2026-11-26.** The page's choice of what to show lists
the tilt-series, then the classification runs in `<project>/mra` (folders with a
plan.json, the run-record layout; `--mra` names another folder), then the FSC
curves beside them (`<out>_fsc.json` from `scripts/consensus_halves.py maps`).
`dashboard/mra.py` reads a run with the classification report's own reader
(`read_classification`, without the initial average) and gives, per run: the
class sizes over the iterations as a chart whose click picks an iteration, the
particles that changed class (`class_changes`), and for each class of the chosen
iteration, the starting references as iteration 0, its members, decoys and
median CC, every z section on a shared slider and its central XY, XZ and YZ
sections, one grey scale per map, values as they are, y and z upwards, in the
report's channel colours. A class goes to Mol\* in its colour at 1σ on the
protein's side, with the alignment mask translucent if asked; the side is
measured once per run, on the last iteration's largest class inside the mask
against outside it. The map handed to Mol\* is the reader's array written back
as a standard MRC at the run's pixel size, so Mol\* draws what the pictures show
in either layout. The FSC view draws the curve against spatial frequency, with
0.5 and 0.143 and where it falls through them as consensus_halves reads it (a/b
for more than one fall); the curve without the mask only behind a check box, as
the slides show the masked one alone; facts on the halves, their start, the
pixel size and any rise toward Nyquist; and the mean of the halves in Mol\* with
their mask. Old Dynamo-layout projects are not listed: the folder of runs lists
only plan.json runs.

Checked: the pictures against marker voxels placed by hand in a 9-voxel box,
read back from the PNGs (every section, and XY, XZ, YZ each in its place); the
maps for Mol\* against the reader's arrays (class, reference and mask, header
1, 2, 3, the pixel size); the run against `read_classification` on a real short
MRA computed by `run_plan` (test_mra_setup's twelve particles, three
iterations), sizes and changes included. In the browser, the drawn z section 5
of that run's last class correlates 1.000 with the run's own
`ite_0003/average_1.mrc` read with y upwards, 0.686 with y downwards, 0.489
transposed. The FSC view's 0.5 at 29.2 Å and "never crossed" at 0.143 match the
file (its last point 0.1487 at Nyquist). Tests: 11 in test_dashboard_mra; 14 of
14 mutants caught on the new and changed code (flips, order of sections,
transposed views, a transposed or unscaled map, file names, the reference of
iteration 0, frequency, the protein's side).

Unverified so far: the tunnel at either site, whether the sites allow ssh to a
node running one's job, the page with 600 picks, Mol\* on a real template, and
every classification view on a real run: none is on the Mac.
