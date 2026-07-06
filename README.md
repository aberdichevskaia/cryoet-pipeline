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

The current implementation covers a deterministic fiducial-based baseline from
multiframe MRC movies through a positioned, visual-QC-ready tomogram:

```text
ingest -> frame averaging -> tilt stack -> coarse IMOD alignment
       -> aligned preview/QC -> automatic fiducial seed -> bead tracking
       -> fine alignment -> final aligned stack -> positioned reconstruction
```

The canonical tomogram is stored as Zarr in `ZYX` order. IMOD `.xf`, `.st`, and
`.rec` files are retained as compatibility and visual-QC outputs. The coarse
aligned stack is diagnostic only and cannot be selected by the reconstruction
command. CTF estimation and correction remain later stages.

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
cryoet download-empiar-10164 --dry-run
cryoet download-empiar-10164
```

This writes:

```text
data/empiar-10164/data/frames/TS_01_*.mrc
data/empiar-10164/data/frames/TS_43_*.mrc
data/empiar-10164/data/mdoc-files/TS_01.mrc.mdoc
data/empiar-10164/data/mdoc-files/TS_43.mrc.mdoc
```

Initialize a local project manifest after the files are present:

```bash
cryoet init \
  --frames data/empiar-10164/data/frames \
  --mdocs data/empiar-10164/data/mdoc-files \
  --out outputs/dev \
  --device auto
```

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

cryoet reconstruct-tomogram \
  --manifest outputs/dev/manifests/TS_01.json \
  --registry outputs/dev/artifacts.json \
  --out outputs/dev \
  --imod-dir /Applications/IMOD \
  --device cpu
```

Fine alignment iterates `AngleOffset` and `AxisZShift` until the two-surface
positioning correction converges. Reconstruction then uses the positioned
transforms, solved tilt angles, recommended thickness, and X-axis tilt without
applying the Z shift a second time. Explicit CLI options can override the
reconstruction values for controlled experiments. Pixel spacing can likewise
be overridden during fiducial seeding when acquisition metadata needs
calibration.

The default device is `auto`: CUDA is preferred on Linux GPU machines, Apple
Silicon MPS is used when available, and CPU is the fallback. On M4 Macs this is
intended for ingest, QC, small tests, and backends that support PyTorch MPS;
CUDA-oriented third-party reconstruction tools may still require Linux + NVIDIA.

For GPU work on Linux, install the CUDA-compatible PyTorch build separately if needed, then:

```bash
python -m pip install -e ".[gpu,io,dev]"
```
