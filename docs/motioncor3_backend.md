# MotionCor3 backend

## Role

`MotionCor3MotionCorrectionBackend` is an external-tool adapter behind the
pipeline-owned `MotionCorrectionBackend` protocol. It can replace `average` or
`phase-corr` without changing tilt-stack preparation or any later stage.

The adapter does not install, compile, or manage MotionCor3. Production use
requires a Linux host with an NVIDIA CUDA GPU and a MotionCor3 executable
provided by the user or cluster.

## Current behavior

For each tilt movie, the adapter:

1. validates the complete multiframe MRC before starting any process;
2. runs MotionCor3 on one movie with global and patch correction;
3. defaults to a `5 x 5` patch grid and one configured GPU;
4. saves MotionCor3 alignment metadata and both process and tool logs;
5. validates that the result is one finite 2D projection with the expected size;
6. converts the result to the storage-policy format and registers a standard
   `CORRECTED_PROJECTION` artifact.

The adapter uses Fourier binning `1.0`, does not save an aligned frame stack, and
explicitly disables dose weighting and MotionCor3's built-in CTF estimation.
These choices avoid hidden resolution loss, large duplicate intermediates, and
mixing future CTF work into the motion-correction stage. Dose weighting will be
enabled only after raw pixel size and per-frame/accumulated dose are validated.

## CLI

```bash
cryoet prepare-tilt-series \
  --manifest outputs/dev/manifests/TS_01.json \
  --registry outputs/dev/artifacts.json \
  --out outputs/dev \
  --motion-backend motioncor3 \
  --motioncor3-executable /path/to/MotionCor3 \
  --motioncor3-gpu 0 \
  --motioncor3-patch-x 5 \
  --motioncor3-patch-y 5 \
  --device cuda \
  --storage-policy working
```

Optional settings include a calibrated `--motioncor3-pixel-size`, a gain MRC
with `--motioncor3-gain`, and gain rotation/flip values. Without an explicit
pixel-size override, the adapter uses `raw_pixel_spacing_angstrom` from the
tilt-series manifest.

Outputs are written under:

```text
corrected/<tilt-series>/*_mc3.mrc|zarr
motion/motioncor3/<tilt-series>/alignment/
logs/motioncor3/<tilt-series>/*.log
```

## Tests

The default test suite never needs MotionCor3 or CUDA. It injects a fake process
runner and covers command construction, MRC and Zarr canonicalization, artifact
provenance, CUDA enforcement, missing executables, process failures, malformed
outputs, and output preflight behavior.

Official project: <https://github.com/czimaginginstitute/MotionCor3>
