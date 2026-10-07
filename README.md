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
