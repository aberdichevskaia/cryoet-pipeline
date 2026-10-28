# cryoet-pipeline

Extensible Python MVP for cryo-electron tomography preprocessing.

The first target is EMPIAR-10164, processing two tilt-series (`TS_01` and `TS_43`)
from multiframe MRC files plus SerialEM `.mdoc` metadata to visual/QC-ready tomograms.

## MVP principles

- Python API first; CLI is a thin wrapper.
- Internal storage will use chunked Zarr artifacts plus JSON manifests.
- External exports must remain compatible with IMOD-style workflows (`.st`, `.rec`, `.tlt`, `.xf`).
- Third-party tools live behind backend adapters so they can be replaced or customized later.
- The MVP is not agentic, but the code should expose clean state and artifact boundaries for future agentic orchestration.

## Current status

The pipeline runs deterministically from multiframe MRC movies through to a
subtomogram average with a resolution estimate:

```text
ingest -> motion correction -> tilt stack -> coarse alignment -> QC preview
       -> fiducial seed -> tracking -> fine alignment -> final aligned stack
       -> CTF estimation -> CTF correction -> reconstruction
       -> [restoration] -> [segmentation]
       -> template preparation -> picking -> extraction
       -> refinement/classification -> FSC
```

Every stage is a replaceable backend chosen with `--backend`; see
[docs/backends.md](docs/backends.md) for what is available at each stage and how
to choose. Stages in brackets are optional.

The canonical tomogram is stored as Zarr in `ZYX` order. IMOD `.xf`, `.st`, and
`.rec` files are retained as compatibility and visual-QC outputs. The coarse
aligned stack is diagnostic only and cannot be selected by the reconstruction
command. Every stage writes a machine-readable QC report with a `pass`,
`warning`, or `fail` status.

Planned work, and the design decisions already settled for it, are in
[docs/roadmap.md](docs/roadmap.md).

## Processing your own data on the cluster

This section is for running a dataset end to end, from tilt-series to a
subtomogram average, without writing code. You write one config file and run
two submissions. In between, you look at the picks and choose score thresholds.

```text
download data ─▶ lane config ─▶ submission 1 (Slurm) ─▶ review picks, set thresholds ─▶ submission 2 (Slurm)
                                 tilt-series → particles                                  average + resolution
```

### Will it work for my data?

The order of the steps is fixed. The config chooses three things:

| Choice | Options | In the config |
|---|---|---|
| How the data comes in | MRC **movies** per tilt + mdocs, motion-corrected here; or one assembled MRC **stack** per tilt-series + its mdoc | `[lane] input = "frames"` or `"stack"` |
| Alignment | **gold beads**, or **patch tracking** when there is no gold (e.g. FIB lamellae) | block A or block B of the config |
| Membranes | segmented with MemBrain-seg, optionally used to restrict picking; or skipped | `[segment]` present or deleted |

Two combinations have been run on real data: stacks + patches
([`ribo-bin8.toml`](configs/lanes/ribo-bin8.toml), EMPIAR-11538) and movies +
beads + membranes ([`gag-bin8.toml`](configs/lanes/gag-bin8.toml), EMPIAR-10164).
The others produce valid commands but have not been run yet. If yours is one of
them, expect to be the first, and tell the maintainers what happened.

The template for picking comes from EMDB (`emdb_id`) or from a local map
(`source_map`). Movies that are not gain-corrected take a gain reference in
`[motion]`.

What is not supported yet:

- **Movies as TIFF or EER.** Motion correction reads MRC movies only.
- **Metadata without a SerialEM `.mdoc`.** Tilt angles, dose and frame names all
  come from the mdoc.
- **Clusters other than ours.** The Slurm settings in the configs are for the
  Helmholtz Munich cluster: its partitions, the shared IMOD install, and the
  `cryoet-pipeline` conda environment.

### 0. One-time setup

1. Clone the repository on the cluster and install it into the `cryoet-pipeline`
   conda environment (`python -m pip install -e ".[io]"`). IMOD, CTFFIND and
   pytom-match-pick must also be installed, plus MotionCor3 for movies and
   MemBrain-seg for membranes; ask the maintainers where the shared copies are.
2. Set the environment variables the configs refer to, e.g. in your
   `~/.bashrc`: `PYTOMO_MP_DIR` (the folder with the `pytom_*.py` scripts; note
   the O), `MOTIONCOR3`, `MEMBRAIN_EXE`, `MEMBRAIN_CKPT`. To use another IMOD
   than the shared one, set `IMOD_DIR`.
3. Run everything below from the root of your clone, on the login node, with the
   environment active. The jobs run in the clone they were submitted from.

   ```bash
   conda activate cryoet-pipeline
   ```

### 1. Get the data

**From EMPIAR.** Every deposition arranges its folders differently, so first
describe the layout in a short YAML file. Browse
`https://ftp.ebi.ac.uk/empiar/world_availability/<id>/data/`, copy
[`configs/empiar-layouts/example.yaml`](configs/empiar-layouts/example.yaml) and
set the folders and file name patterns (details in
[Downloading a different EMPIAR deposition](#downloading-a-different-empiar-deposition)).
Then download with `--dry-run` first, to see which files it would fetch:

```bash
cryoet download-empiar --out data/empiar-11538 --empiar-id 11538 --tilt-series TS_0201 --tilt-series TS_0388 --layout configs/empiar-layouts/11538-untreated.yaml --dry-run
```

**Your own data.** Copy it under `data/<name>/`: the `.mdoc` files plus either
the tilt stacks or the movies. For movies, the file names must match the
`SubFramePath` entries in the mdocs. The folder they came from does not matter,
only the file name.

### 2. Write the lane config

Copy [`configs/lanes/example.toml`](configs/lanes/example.toml) to a new file,
e.g. `configs/lanes/<name>-bin8.toml`, and work through it top to bottom. Every
line marked `EDIT` depends on your data:

| Section | What to set |
|---|---|
| `[lane]` | `name`; `input` (movies or stacks); `variant`, the label every output gets (e.g. `bin8`) |
| `[run]` | `project` (a new output folder), `series` (tilt-series ids), and the Slurm stages: partition, GPU, memory and time per job |
| `[data]` | where the movies and mdocs, or the stacks and mdocs, are; `{series}` is replaced by each id |
| `[motion]` (movies) | the gain reference, if the movies are not gain-corrected |
| `[binning]` | the binning for the CTF fit and for the tomogram. Choose by pixel size in Å, as the comment explains |
| alignment | keep block A (gold beads) or block B (patch tracking) and delete the other. Each block sets `[fiducials]`, `[fine_align]` and `[reconstruct]` together; switching only `[fiducials]` leaves the other two wrong for the method |
| `[ctf]` | voltage, Cs, amplitude contrast, and a defocus search range that brackets the nominal defocus |
| `[segment]` | keep it for membranes, or delete it |
| `[template]` | `emdb_id` or `source_map`, the low-pass, the mask radius |
| `[pick]` | particle diameter in Å, how many candidates to keep per tomogram, GPU options. The membrane mask is off: turn it on only for particles on or next to membranes |
| `[extract]` | box size in pixels, about 1.5× the particle |

For how the values in a real config were chosen, read the comments in
`ribo-bin8.toml` and `gag-bin8.toml`. Their values were measured on *their*
datasets and will be wrong for yours.

The `[run]` stages split the fixed chain of steps into Slurm jobs. A job runs
every tilt-series of the project in turn, and the next job starts only when the
previous one finished without errors. Stages have to follow one another without
skipping or repeating a step; the submitter checks this. To list the steps your
config gives:

```bash
python3 scripts/run_series.py --project outputs/<project> --config configs/lanes/<name>-bin8.toml --series X --list-steps
```

### 3. Submission 1: tilt-series to extracted particles

First see what would be submitted and whether anything is missing:

```bash
python3 scripts/submit_lanes.py configs/lanes/<name>-bin8.toml --dry-run
```

Everything in the config is checked before anything is submitted: unknown
steps, gaps between stages, mismatched bead/patch settings, missing sections,
input files or folders that are not there, and environment variables the
config uses but your shell does not set. Fix what it lists, then submit for real:

```bash
python3 scripts/submit_lanes.py configs/lanes/<name>-bin8.toml
```

For movies, a project that does not exist yet is created first (`cryoet
init`, on the login node). An existing project is never
re-created, because that would erase it; to add tilt-series later, use a new
project. The script then only queues the jobs and exits. Follow them with `squeue -u $USER`;
each job writes `slurm-<job name>-<id>.log` in the repository root. If a job
fails, the log ends with the failed step and the tilt-series. Fix the cause, then
resubmit from that stage. Steps that already finished in that stage run
again, which is why `--overwrite` is needed:

```bash
python3 scripts/submit_lanes.py configs/lanes/<name>-bin8.toml --from-stage gpu --overwrite
```

`--overwrite` holds for that submission only; with `--dry-run` it also lists the
recorded runs it may replace. `OVERWRITE=1` set in the shell is refused. Every
dry run and every submission is written to the project's `journal.jsonl`.
Series that one of your queued or running jobs of the same project works on are
refused, so wait until that job has ended. The batch scripts that run a lane's
steps another way, such as the averaging in step 5 or the CTF window test, go
through `scripts/submit_recipe.py`, with the same checks and the same journal;
`python3 scripts/submit_recipe.py --list` names them.

Each step writes a QC report (`pass`, `warning` or `fail`) under
`outputs/<project>/qc/`. A `warning` does not stop the chain, so read the reports
anyway, especially for alignment and CTF. `visualize-picks` writes each tomogram
with its picks overlaid, for ChimeraX (`.cxc`) and 3dmod (`.mrc` + `.mod`). Open
one or two before you go on.

For an overview of the whole project, write the report: one PDF with a table of
all tilt-series and each step's verdict, the score curves of all tomograms side
by side, then, per tilt-series, the tilt alignment (residual and beads or
patches per tilt, a few aligned tilts), the CTF fit, the tomogram, its
membranes, the picks on the tomogram and their scores against noise, the
correlation map and galleries of picks by score, each particle low-passed as for
picking and turned to the template's orientation by pytom's angles. It reads the project and runs nothing. It needs
matplotlib: `python -m pip install -e ".[io,viz]"`.

```bash
cryoet report --project outputs/<project> --variant bin8 --config configs/lanes/<name>-bin8.toml
```

The PDF goes to `outputs/<project>/reports/report-bin8.pdf`. The correlation
map page exists only for picks made since `pick-particles` started saving a
picture of the map; older picks get every other page.

### 4. Review the picks and set thresholds

Template matching keeps every candidate it was asked for, including the false
ones. Choosing the score cut-off is your decision, made by looking at the
particles:

```bash
python3 scripts/export_dynamo.py --project outputs/<project> --variant bin8 --series TS_01 --series TS_43 --out outputs/<project>/dynamo/export
```

```bash
python3 scripts/qc_dynamo_picks.py --export outputs/<project>/dynamo/export --out outputs/<project>/dynamo/pick-qc --gallery
```

This writes score distributions and galleries of particles sampled across the
score range, per tomogram, plus `pick-qc/thresholds.json` with one empty
(`null`) entry per tilt-series. Fill in a minimum score for each tilt-series.
The plots suggest where to look, not where to cut.

### 5. Submission 2: average and resolution

```bash
python3 scripts/submit_recipe.py average configs/lanes/<name>-bin8.toml --overwrite --set THRESHOLDS=outputs/<project>/dynamo/pick-qc/thresholds.json --set MASK=380 --dry-run
```

This re-extracts each tilt-series with its threshold, then refines all of them
together as two independent half-sets. The new extraction replaces the one
Submission 1 made under the lane's label, which is why `--overwrite` is needed;
the dry run lists what it replaces. `MASK` is the alignment mask: either a
diameter in Å (a bit larger than the particle) or an MRC mask in the particle
box. The project comes from the config's `[run]` section, and so do the series
unless `--series` names others. The job runs on the partition the lane
reconstructs on, with 32 CPUs for up to 24 hours (`--time` for longer). Check
what the dry run lists, then run the same command without `--dry-run`. Optional
settings, such as `--set ITERATIONS=5`, are listed by `--list` and at the top of
`scripts/average_lane.sbatch`.

The average, the two half-maps and the particle STAR file are written to
`outputs/<project>/averages/`. The FSC and the resolution estimate are in the QC
report under `outputs/<project>/qc/`.

## The terminal assistant

A terminal assistant for the steps above is being built: Claude Code, started by
`scripts/assistant` with its own instructions, permission rules and two skills so
far, the status of a project with the reasons a step failed, and writing a lane
config. It submits only through the scripts above, after a dry run and an
explicit yes. It has not yet been run with a model or on the cluster. How to
start it, and what it may and may not do: [assistant/README.md](assistant/README.md).

## Development

```bash
python -m pip install -e ".[dev,io]"
pytest
```

## Download the MVP dataset subset

Avoid downloading EMPIAR-10164 as a browser ZIP: the selected subset is large and
browser-generated ZIP files can fail without a valid central directory. Download
the two MVP tilt-series as individual files instead:

```bash
cryoet download-empiar --out data/empiar-10164 --dry-run
cryoet download-empiar --out data/empiar-10164
```

This writes:

```text
data/empiar-10164/data/frames/TS_01_*.mrc
data/empiar-10164/data/frames/TS_43_*.mrc
data/empiar-10164/data/mdoc-files/TS_01.mrc.mdoc
data/empiar-10164/data/mdoc-files/TS_43.mrc.mdoc
```

### Downloading a different EMPIAR deposition

`download-empiar` only ships a built-in directory layout for EMPIAR-10164
(`frames/` + `mdoc-files/`, `TS_<id>_*.mrc` frame names). Other depositions are
organized however their depositor chose to organize them, so there is no
generic auto-discovery — instead, describe the layout once in a small YAML
file and pass it with `--layout`:

```yaml
# configs/empiar-layouts/11538-untreated.yaml
groups:
  - remote_dir: mdoc_files
    file_pattern: "{tilt_series}.mrc.mdoc"
  - remote_dir: Untreated_dataset
    file_pattern: "{tilt_series}.mrc"
```

```bash
cryoet download-empiar \
  --out data/empiar-11538 \
  --empiar-id 11538 \
  --tilt-series TS_0002 \
  --layout configs/empiar-layouts/11538-untreated.yaml \
  --dry-run
```

Each `groups` entry names one remote subdirectory (relative to the
deposition's `data/` root, as browsable at
`https://ftp.ebi.ac.uk/empiar/world_availability/<id>/data/`) and a glob
pattern identifying files that belong to one requested tilt-series id within
it; `{tilt_series}` is substituted before matching, and `*`/`?` wildcards are
supported for depositions with multiple files per tilt-series (like
EMPIAR-10164's per-frame files). Copy
[`configs/empiar-layouts/example.yaml`](configs/empiar-layouts/example.yaml)
to a new file under the same directory and edit it for your deposition; only
that example is tracked in git (see `.gitignore`) since most layouts are
one-off/exploratory rather than a fixed part of the pipeline.

Initialize a local project manifest after the files are present:

```bash
cryoet init \
  --frames data/empiar-10164/data/frames \
  --mdocs data/empiar-10164/data/mdoc-files \
  --out outputs/dev \
  --device auto
```

Movies may be MRC or TIFF (`tifffile` must be installed for TIFF). A local
dataset without mdocs is read from its file names instead: `--frames-pattern`
is a regular expression matched against each whole file name, whose named
groups `series`, `tilt` and `angle` give the series id, the tilt number (the
images are ordered by it) and the tilt angle. The pixel size is read from the
movie headers, for TIFF from the resolution tags SerialEM writes (IMOD's `header`
shows it); `--pixel-size` overrides it. Without `--tilt-series`, every series
found is included. Without mdocs, `align-tilt-series` also needs the tilt axis
(`--tilt-axis-angle`, `[align] tilt_axis_angle` in a lane config).

```bash
cryoet init \
  --frames data/my-dataset/frames \
  --frames-pattern 'tomogram_(?P<series>\d+)_(?P<tilt>\d+)_(?P<angle>[-+]?\d+(?:\.\d+)?)\.tif' \
  --out outputs/my-dataset
```

`init` prints the series it found and, for each, the tilt count, angle range,
frames per tilt and pixel size; check them against what was collected before
running anything.

### Depositions without raw frames

Some depositions ship one already-assembled stack per tilt-series rather than
individual movies. There is nothing to motion-correct, so `init` and
`prepare-tilt-series` do not apply; import the stack directly and continue from
`align-tilt-series`:

```bash
cryoet import-tilt-stack \
  --mdoc data/empiar-11538/data/mdoc_files/TS_0002.mrc.mdoc \
  --stack data/empiar-11538/data/Untreated_dataset/TS_0002.mrc \
  --tilt-series-id TS_0002 \
  --out outputs/dev
```

Per-tilt metadata comes from the `.mdoc`; pixel spacing is read from the stack's
own MRC header.

Run the current baseline stages:

```bash
cryoet prepare-tilt-series \
  --manifest outputs/dev/manifests/TS_01.json \
  --registry outputs/dev/artifacts.json \
  --out outputs/dev \
  --storage-policy working \
  --device cpu

cryoet align-tilt-series \
  --manifest outputs/dev/manifests/TS_01.json \
  --registry outputs/dev/artifacts.json \
  --out outputs/dev \
  --binning 8 \
  --imod-dir /Applications/IMOD \
  --device cpu

cryoet qc-coarse-alignment \
  --manifest outputs/dev/manifests/TS_01.json \
  --registry outputs/dev/artifacts.json \
  --out outputs/dev \
  --preview-binning 16 \
  --imod-dir /Applications/IMOD \
  --device cpu

cryoet generate-fiducial-seed \
  --manifest outputs/dev/manifests/TS_01.json \
  --registry outputs/dev/artifacts.json \
  --out outputs/dev \
  --tracking-binning 4 \
  --fiducial-diameter-nm 10 \
  --imod-dir /Applications/IMOD \
  --device cpu

cryoet track-fiducials \
  --manifest outputs/dev/manifests/TS_01.json \
  --registry outputs/dev/artifacts.json \
  --out outputs/dev \
  --imod-dir /Applications/IMOD \
  --device cpu

cryoet fine-align-tilt-series \
  --manifest outputs/dev/manifests/TS_01.json \
  --registry outputs/dev/artifacts.json \
  --out outputs/dev \
  --imod-dir /Applications/IMOD \
  --device cpu

cryoet build-final-aligned-stack \
  --manifest outputs/dev/manifests/TS_01.json \
  --registry outputs/dev/artifacts.json \
  --out outputs/dev \
  --output-binning 8 \
  --imod-dir /Applications/IMOD \
  --device cpu

cryoet estimate-ctf \
  --manifest outputs/dev/manifests/TS_01.json \
  --registry outputs/dev/artifacts.json \
  --out outputs/dev \
  --ctf-voltage-kv 300 \
  --ctf-spherical-aberration-mm 2.7 \
  --device cpu

cryoet correct-ctf \
  --manifest outputs/dev/manifests/TS_01.json \
  --registry outputs/dev/artifacts.json \
  --out outputs/dev \
  --imod-dir /Applications/IMOD \
  --device cpu

cryoet reconstruct-tomogram \
  --manifest outputs/dev/manifests/TS_01.json \
  --registry outputs/dev/artifacts.json \
  --out outputs/dev \
  --imod-dir /Applications/IMOD \
  --device cpu

cryoet segment-tomogram \
  --manifest outputs/dev/manifests/TS_01.json \
  --registry outputs/dev/artifacts.json \
  --out outputs/dev \
  --backend membrain-seg \
  --membrain-executable /path/to/membrain \
  --membrain-model /path/to/membrain_model.ckpt \
  --device cuda
```

`estimate-ctf` defaults to CTFFIND5 and writes an IMOD-format `.defocus` model
plus the diagnostic power-spectrum image. `correct-ctf` phase-flips with that
model, taking voltage and spherical aberration from it rather than from its own
flags so the two cannot disagree. `reconstruct-tomogram` then prefers the
CTF-corrected stack automatically. Both stages are optional; skipping them
reconstructs from the plain aligned stack.

`segment-tomogram` consumes the reconstructed canonical Zarr tomogram directly;
IsoNet2 restoration is not required. The MemBrain-seg adapter exports a
temporary MRC with voxel size metadata, runs the external `membrain segment`
command, and canonicalizes the selected segmentation output back to Zarr. Use
`--membrain-output-name` if MemBrain-seg writes multiple `.mrc`/`.rec` outputs
and the adapter cannot infer the intended segmentation volume.

### Optional and standalone stages

```bash
# Motion-correct without assembling a stack. prepare-tilt-series does both;
# this is for inspecting corrected projections on their own.
cryoet correct-motion --manifest ... --registry ... --out outputs/dev

# Fit one near-zero-tilt image first. estimate-ctf picks the result up
# automatically and skips its own first pass.
cryoet estimate-defocus-hint --manifest ... --registry ... --out outputs/dev \
  --ctf-voltage-kv 300 --ctf-spherical-aberration-mm 2.7

# Denoise/restore a reconstructed tomogram. Needs a trained IsoNet2 model.
cryoet restore-tomogram --manifest ... --registry ... --out outputs/dev \
  --isonet2-model /path/to/model
```

## Particle picking and subtomogram averaging

Fetch a reference map from EMDB and build a matching template and mask, pick
against it, then look at the picks before committing to averaging:

```bash
cryoet prepare-template \
  --manifest outputs/dev/manifests/TS_01.json \
  --registry outputs/dev/artifacts.json \
  --out outputs/dev \
  --emdb-id EMD-2638 \
  --template-particle-diameter-angstrom 250

cryoet pick-particles \
  --manifest outputs/dev/manifests/TS_01.json \
  --registry outputs/dev/artifacts.json \
  --out outputs/dev \
  --pytom-particle-diameter 250 \
  --device cuda

cryoet visualize-picks \
  --manifest outputs/dev/manifests/TS_01.json \
  --registry outputs/dev/artifacts.json \
  --out outputs/dev
# then: chimerax <the printed .cxc file>
#   or: 3dmod <the printed .mrc file> <the printed .mod file>
```

`prepare-template` caches the downloaded map under `<out>/emdb-cache`, resamples
it to the tomogram's voxel size, and registers the template and mask so
`pick-particles` finds them without explicit paths. Pass `--source-map` instead
of `--emdb-id` to use a local map. Contrast inversion is on by default, since
EMDB maps are bright and tomogram particles are dark; the QC report warns if the
produced template does not look inverted as requested.

`visualize-picks` writes one directory that both viewers read, since they answer
different questions and neither answers the other's. ChimeraX with ArtiaX shows
the whole set in 3D, each pick drawn as the template at the orientation found
for it, which is where a systematically wrong orientation or a lattice picked as
a cloud becomes visible. 3dmod shows one slice at a time with the picks drawn on
it, which is how you answer whether a given pick sits on real density. Use
`--sphere-radius-px` to change the size the picks are drawn at in 3dmod; it
defaults to the radius the template mask was built with.

To keep only the picks above a score, read a threshold off the report's score
curves and register the kept ones under a label of their own; extraction, the
Dynamo export, averaging and the report then select them with `--variant`:

```bash
cryoet select-picks \
  --manifest outputs/dev/manifests/TS_01.json --manifest outputs/dev/manifests/TS_43.json \
  --registry outputs/dev/artifacts.json --out outputs/dev \
  --variant bin8 --output-variant bin8-sel
```

With no threshold, as above, the thresholds come from a file in a fixed place,
`<out>/picks/<variant>_thresholds_sigma.json`. The first run writes it with every
series null and stops; fill in a number per series and run the same command again:

```json
{
  "TS_01": 7.5,
  "TS_43": 8.0,
  "_how": "Minimum score per tomogram, in σ of its own search ...",
  "_hints": {
    "TS_01": {"picks": 600, "best_sigma": 12.3, "lowest_sigma": 6.4, "one_false_positive_sigma": 7.36}
  }
}
```

Keys starting with `_` are notes and are not read; the hints say where each
series' scores lie. A series left null is refused, by name. Instead of the file,
`--minimum-sigma 7.5` gives every series one threshold, `--thresholds-sigma FILE`
reads another file of the same layout, and `--minimum-score` / `--thresholds-score`
take pytom's raw LCCmax, as `scripts/qc_dynamo_picks.py` writes it.

σ is each tomogram's own search noise, so one σ means the same in every series
while a raw score does not. `--dry-run` prints what each threshold keeps and
registers nothing. A value that looks like the other unit (a σ under 1, a score
over 1) is refused rather than read as given.

Then extract subvolumes, refine and classify them, and estimate resolution:

```bash
cryoet extract-particles \
  --manifest outputs/dev/manifests/TS_01.json \
  --registry outputs/dev/artifacts.json \
  --out outputs/dev \
  --box-size-px 96

cryoet average-particles \
  --manifest outputs/dev/manifests/TS_01.json \
  --registry outputs/dev/artifacts.json \
  --out outputs/dev \
  --particle-diameter-angstrom 250 \
  --device cuda

cryoet compute-fsc \
  --mode half-set \
  --particle-stack outputs/dev/particles/TS_01/TS_01_cubic-crop_particles.zarr \
  --metadata outputs/dev/averages/TS_01/TS_01_dynamo-search_particles.star \
  --pixel-spacing-angstrom 9.504 \
  --out outputs/dev/qc/TS_01/fsc_half_set.json \
  --plot outputs/dev/qc/TS_01/fsc_half_set.png \
  --device cuda
```

`average-particles` defaults to the `dynamo-search` backend: Dynamo's local
search and averaging, adapted to NumPy and validated against Dynamo itself,
one class, one iteration unless
`--num-iterations` says otherwise. `--gold-standard` refines the two halves of
`rlnRandomSubset` independently from a common band-passed start and reports
their FSC; see [docs/dynamo_search.md](docs/dynamo_search.md).
`--backend cone-search` searches orientations on a cone and refines several
references at once when `--num-classes` is above one; see
[docs/backends.md](docs/backends.md) for it and for the `fft-refinement` baseline.

`compute-fsc` also runs in `--mode map-to-model` to compare the average against
the template. Both modes accept `--mask` so background noise outside the
particle is not counted. A curve that rises again toward Nyquist after a dip is
reported as a warning, with where it fell through the threshold before the dip:
the final descent the resolution is read from then lands beyond it. Note that the half-set FSC is still optimistic: both
halves are refined against a shared reference, rebuilt from all particles every
iteration, so it is not yet a true gold-standard estimate — see
[docs/roadmap.md](docs/roadmap.md). A separate and larger defect has been fixed:
particles were averaged without removing their own mean, and zero padding under
rotation turned that offset into an envelope which correlated between any two
half-sets. It reported ~65 A from pure noise with perfectly independent halves,
and got worse with more particles.

On a Linux/NVIDIA processing host where MotionCor3 is already managed by the
user or cluster, replace only the motion backend:

```bash
cryoet prepare-tilt-series \
  --manifest outputs/dev/manifests/TS_01.json \
  --registry outputs/dev/artifacts.json \
  --out outputs/dev \
  --motion-backend motioncor3 \
  --motioncor3-executable /path/to/MotionCor3 \
  --motioncor3-gpu 0 \
  --device cuda \
  --storage-policy working
```

The adapter defaults to a `5 x 5` patch grid, keeps MotionCor3 logs and
alignment metadata, and does not save the corrected frame stack. Dose weighting
is intentionally disabled until pixel size and dose metadata are validated.
See [docs/motioncor3_backend.md](docs/motioncor3_backend.md).

Fine alignment iterates `AngleOffset` and `AxisZShift` until the two-surface
positioning correction converges. Reconstruction then uses the positioned
transforms, solved tilt angles, recommended thickness, and X-axis tilt without
applying the Z shift a second time. Explicit CLI options can override the
reconstruction values for controlled experiments. Pixel spacing can likewise
be overridden during fiducial seeding when acquisition metadata needs
calibration.

For specimens without gold beads, switch both fiducial steps to patch tracking:

```bash
cryoet generate-fiducial-seed ... --backend imod-patch-seed
cryoet track-fiducials       ... --backend imod-patch-track
```

Patch size is derived from the image size, the requested patch count, and the
margin rather than entered directly.

The default device is `auto`: CUDA is preferred on Linux GPU machines, Apple
Silicon MPS is used when available, and CPU is the fallback. On M4 Macs this is
intended for ingest, QC, small tests, and backends that support PyTorch MPS;
CUDA-oriented third-party reconstruction tools may still require Linux + NVIDIA.

For GPU work on Linux, install the CUDA-compatible PyTorch build separately if needed, then:

```bash
python -m pip install -e ".[gpu,io,dev]"
```

FSC plots need matplotlib, which lives in its own extra so it does not pull in
the GPU stack:

```bash
python -m pip install -e ".[viz]"
```

## Documentation

- [docs/backends.md](docs/backends.md) — every stage's backends and how to choose
- [docs/roadmap.md](docs/roadmap.md) — planned work and settled design decisions
- [docs/database_design.md](docs/database_design.md) — project state, provenance
  graph, and the database decision
- [docs/pipeline_engine.md](docs/pipeline_engine.md) — how a plan becomes cluster
  jobs: controller, run states, execution backends
- [docs/agent_architecture.md](docs/agent_architecture.md) — forward design for a
  copilot: tool surface, gates, failure triage
- [docs/motioncor3_backend.md](docs/motioncor3_backend.md) — MotionCor3 adapter
- [docs/architecture_decisions.md](docs/architecture_decisions.md),
  [docs/storage_policy.md](docs/storage_policy.md),
  [docs/qc_strategy.md](docs/qc_strategy.md) — design background
