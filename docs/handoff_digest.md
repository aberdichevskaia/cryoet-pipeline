# CryoET Pipeline Handoff Digest

Last updated: 2026-06-24

This file summarizes the context, decisions, implemented code, downloaded data,
and immediate next steps for continuing the cryo-ET MVP on another machine or in
another Codex thread.

## Project Goal

Build an extensible end-to-end cryo-electron tomography pipeline from raw tilt
movies to tomograms, and later to segmentation, particle picking, and
subtomogram averaging. The MVP is intentionally not agentic yet, but the code
should be modular enough that agentic orchestration, QC review, or parameter
suggestion can be attached later.

Near-term MVP target:

```text
raw multiframe MRC + SerialEM .mdoc
  -> ingest manifests
  -> motion correction baseline
  -> corrected projections
  -> tilt stack
  -> alignment
  -> reconstruction
  -> visual/QC-ready tomograms
```

The first real dataset target is EMPIAR-10164, specifically `TS_01` and `TS_43`.
These two tilt-series were chosen because they cover both frame-count cases in
the dataset:

- `TS_01`: 41 tilts, 8 frames per tilt
- `TS_43`: 41 tilts, 10 frames per tilt

## Architectural Decisions

Use this as the guiding storage rule:

```text
IMOD layout = compatibility/export layer
Zarr + manifests/registry = internal truth
copick = downstream adapter for tomograms/segmentations/picks
```

Meaning:

- Do not treat IMOD `.st/.rec/.xf/.tlt` files as the internal source of truth.
- Keep structured manifests/registry as the pipeline truth.
- Store large internal arrays in Zarr when that layer is implemented.
- Generate IMOD-compatible folders only as export/exchange outputs so the user
  can continue manually in IMOD-family tools at any point.
- Use copick later for downstream tomograms, segmentations, picks, meshes, and
  object classes.

Other fixed decisions:

- Python API first; CLI is a thin wrapper.
- External tools are used through adapters first. Vendor/fork only when real
  customization requires it.
- Runtime device is `auto`, resolving as CUDA -> Apple Silicon MPS -> CPU.
- M4/MPS should support ingest, QC, small tests, and backends that support
  PyTorch MPS, but CUDA-oriented third-party reconstruction tools may still need
  Linux + NVIDIA.
- First motion correction backend should be a simple in-house global
  phase-correlation baseline, not MotionCor3 parity.
- MotionCor3 should be considered later as a Python backend/binding/adapter, not
  a full Python rewrite.

## Repository And Current Layout

Local repo path used during development:

```text
/Users/anna_berdichevskaia/Documents/cryoet-pipeline
```

Important files:

```text
pyproject.toml
README.md
docs/mvp_plan.md
docs/handoff_digest.md
src/cryoet_pipeline/cli.py
src/cryoet_pipeline/empiar.py
src/cryoet_pipeline/project.py
src/cryoet_pipeline/models.py
src/cryoet_pipeline/runtime.py
src/cryoet_pipeline/ingest/mdoc.py
src/cryoet_pipeline/backends/protocols.py
tests/test_empiar.py
tests/test_mdoc.py
tests/test_models.py
tests/test_project.py
tests/test_runtime.py
```

Current data/output layout:

```text
data/empiar-10164/data/frames/TS_01_*.mrc
data/empiar-10164/data/frames/TS_43_*.mrc
data/empiar-10164/data/mdoc-files/TS_01.mrc.mdoc
data/empiar-10164/data/mdoc-files/TS_43.mrc.mdoc

outputs/dev/project.json
outputs/dev/manifests/TS_01.json
outputs/dev/manifests/TS_43.json
```

`data/` and `outputs/` are ignored by git.

## What Has Been Implemented

### Package scaffold

- `pyproject.toml` defines the package, CLI entrypoint, dependencies, optional
  groups, pytest config, ruff config, and mypy config.
- CLI command is exposed as `cryoet`.

Install locally:

```bash
cd /Users/anna_berdichevskaia/Documents/cryoet-pipeline
python -m pip install -e ".[dev,io]"
```

Optional GPU extra:

```bash
python -m pip install -e ".[dev,io,gpu]"
```

### Runtime device selection

Implemented in `src/cryoet_pipeline/runtime.py`:

- `DevicePreference`: `auto`, `cuda`, `mps`, `cpu`
- `resolve_device("auto")`: picks CUDA if available, then MPS, then CPU
- Injectable torch module for tests

### Core typed models

Implemented in `src/cryoet_pipeline/models.py`:

- `ProjectConfig`
- `TiltImage`
- `TiltSeriesManifest`
- `Artifact`
- `ArtifactKind`
- `AxisOrder`

### SerialEM `.mdoc` parsing

Implemented in `src/cryoet_pipeline/ingest/mdoc.py`.

Parses:

- `TiltAngle`
- `SubFramePath`
- `NumSubFrames`
- `PixelSpacing`
- `Binning`
- `ExposureTime`
- `ExposureDose`
- `Defocus`
- `RotationAngle`

It maps Windows-style `SubFramePath` values from `.mdoc` to local frame files in
the provided `frames_dir`.

Important EMPIAR-10164 caveat captured in manifests:

- raw frame pixel spacing is 1.35 A
- `.mdoc` pixel spacing is 5.4 A because `.mdoc` refers to binned output

### EMPIAR downloader helper

Implemented in `src/cryoet_pipeline/empiar.py` and exposed through CLI:

```bash
cryoet download-empiar-10164 --dry-run
cryoet download-empiar-10164
```

It downloads only the selected tilt-series files as individual files, avoiding
browser-generated ZIP archives and Aspera token issues.

Default selected tilt-series:

```text
TS_01, TS_43
```

The helper writes:

```text
data/empiar-10164/data/frames/TS_01_*.mrc
data/empiar-10164/data/frames/TS_43_*.mrc
data/empiar-10164/data/mdoc-files/TS_01.mrc.mdoc
data/empiar-10164/data/mdoc-files/TS_43.mrc.mdoc
```

It uses `.part` files and skips already completed files.

### Project initialization and validation

Implemented in `src/cryoet_pipeline/project.py` and exposed through CLI:

```bash
cryoet init \
  --frames data/empiar-10164/data/frames \
  --mdocs data/empiar-10164/data/mdoc-files \
  --out outputs/dev \
  --device auto
```

This now does real work:

- reads `.mdoc` files
- validates expected tilt count
- validates expected frames per tilt for known MVP series
- validates local frame files exist
- writes `outputs/dev/project.json`
- writes per-series manifests under `outputs/dev/manifests/`

Observed real-data validation:

```text
TS_01 frames: 41, mdoc: present, frames per tilt: 8
TS_43 frames: 41, mdoc: present, frames per tilt: 10
part files: 0
data size: ~39G
```

### Backend protocols

Implemented in `src/cryoet_pipeline/backends/protocols.py`:

- `MotionCorrectionBackend`
- `TiltAlignmentBackend`
- `ReconstructionBackend`

These are intentionally interfaces only. Compute backends are not implemented
yet.

## Things Discussed But Not Yet Implemented

### Internal storage layout

Planned future output layout:

```text
outputs/dev/
  project.json
  manifests/
    TS_01.json
    TS_43.json
  registry/
    artifacts.jsonl
    runs.jsonl
  store/
    TS_01/
      corrected_projections.zarr/
      aligned_stack.zarr/
      tomogram.zarr/
    TS_43/
      ...
  work/
    TS_01/
      motion/
      qc/
    TS_43/
      motion/
      qc/
  exchange/
    imod/
      TS_01/
        TS_01.st
        TS_01.tlt
        TS_01.xf
        TS_01.rec
      TS_43/
        ...
```

Current implemented output is only:

```text
outputs/dev/project.json
outputs/dev/manifests/TS_01.json
outputs/dev/manifests/TS_43.json
```

### Motion correction baseline

Next intended implementation step:

```text
one multiframe MRC movie
  -> read frames
  -> estimate global shifts with phase correlation
  -> apply integer-pixel shifts
  -> average aligned frames
  -> write corrected projection MRC
  -> write motion QC JSON
```

Start with one file:

```text
data/empiar-10164/data/frames/TS_01_000_0.0.mrc
```

Proposed output:

```text
outputs/dev/work/TS_01/motion/TS_01_000_0.0/
  corrected.mrc
  motion.json
```

The first backend should be simple and explicit:

- global translational correction only
- phase correlation
- integer-pixel shifts first
- average-only fallback for debugging
- no patch correction yet
- no dose weighting yet
- no subpixel refinement yet
- no MotionCor3 parity yet

### Later stages

Not implemented yet:

- motion correction for whole tilt-series
- corrected projection stack
- `.st` and `.tlt` writing
- tilt-series alignment
- `.xf` writing
- reconstruction
- `.rec` writing
- Zarr store
- artifact registry
- QC plots/previews
- copick export
- segmentation/picking/STA stubs beyond concept

## Issues Encountered And Decisions

### Browser ZIP failed

The EMPIAR browser ZIP download produced a file around 33G that started like a
ZIP but lacked the ZIP end-of-central-directory record. `unzip -t` reported:

```text
End-of-central-directory signature not found
```

Conclusion: do not use browser-generated ZIP for this subset.

### Aspera failed

Aspera Connect opened, but logs showed:

```text
No token specified for authentication=token request.
http.status 406 Not Acceptable
```

Conclusion: Aspera handoff from the EMPIAR page failed due to a missing token.
Use the project downloader over HTTPS instead.

### MotionCor3 Python rewrite

Discussed and decided:

- full MotionCor3 rewrite in Python is too large for MVP
- pure Python loops would be far too slow
- NumPy/PyTorch/CuPy can be fast if operations remain vectorized/GPU-side
- best future production route is a MotionCor3 adapter/binding, not a full
  rewrite
- current MVP should implement a simple in-house phase-correlation baseline

## How To Resume On Another Machine

1. Clone or copy the repository.
2. Create/install an environment:

```bash
cd cryoet-pipeline
python -m pip install -e ".[dev,io]"
```

3. Run tests:

```bash
pytest
```

4. Download the MVP subset if data is not copied already:

```bash
cryoet download-empiar-10164 --dry-run
cryoet download-empiar-10164
```

5. Initialize manifests:

```bash
cryoet init \
  --frames data/empiar-10164/data/frames \
  --mdocs data/empiar-10164/data/mdoc-files \
  --out outputs/dev \
  --device auto
```

6. Expected files:

```text
outputs/dev/project.json
outputs/dev/manifests/TS_01.json
outputs/dev/manifests/TS_43.json
```

7. Continue with the next step: implement one-movie motion correction baseline.

## Suggested Next Commit / Next Work Item

If current changes are not committed, a reasonable commit message is:

```text
Add EMPIAR ingest manifest setup
```

Next code task:

```text
Add one-movie phase-correlation motion correction baseline
```

Minimum acceptance for that next task:

- can read `TS_01_000_0.0.mrc`
- detects number of frames
- estimates per-frame global shifts
- writes a 2D corrected projection MRC
- writes `motion.json`
- has synthetic tests for known shifts
- does not process the whole tilt-series yet

