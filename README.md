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

This repository starts with the project skeleton, typed data models, `.mdoc` parsing, backend protocols, and CLI placeholders.
The compute-heavy motion/alignment/reconstruction implementations are intentionally added behind interfaces in later steps.

## Development

```bash
python -m pip install -e ".[dev,io]"
pytest
```

The default device is `auto`: CUDA is preferred on Linux GPU machines, Apple
Silicon MPS is used when available, and CPU is the fallback. On M4 Macs this is
intended for ingest, QC, small tests, and backends that support PyTorch MPS;
CUDA-oriented third-party reconstruction tools may still require Linux + NVIDIA.

For GPU work on Linux, install the CUDA-compatible PyTorch build separately if needed, then:

```bash
python -m pip install -e ".[gpu,io,dev]"
```
