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
motion/motioncor3/<tilt-series>/*.fmint      (EER movies only)
logs/motioncor3/<tilt-series>/*.log
```

When a gain is given, MotionCor3's output is read for whether it used it. It
only warns when it cannot (any mode but 32-bit float) and sums without one, so
"Gain reference not found" stops the step; "Gain reference has been loaded"
is recorded as `gain_loaded: true` in the artifact. A build that words it
otherwise leaves `gain_loaded` null.

## EER movies

EER movies (Falcon cameras, `.eer`) are passed with `-InEer`. Their frame
count, sensor size and pixel come from the file's tags and metadata, read
without decoding; MotionCor3 decodes and sums the frames itself, and the other
motion backends refuse them. Two settings apply:

- `--motioncor3-eer-sampling` (`[motion] eer_sampling`, default 1):
  `-EerSampling`. 1 renders the electron events on the sensor's pixels; 2 on a
  grid twice as fine, which `--motioncor3-ftbin 2` must then bin back, since a
  sum finer than the movie pixel is refused. 3 (16K) is not offered: MotionCor3
  caps it at 2 for compression 65002 without changing the pixel size it reports.
- `--motioncor3-eer-fractions` (`[motion] eer_fractions`, default 10): the
  groups the EER frames are summed in before alignment, written per movie as
  the `-FmIntFile` line `N G 0`. G is N // fractions: MotionCor3 adds the
  N % G frames left over to the last groups, one each, and writes past the end
  of its array when they outnumber the groups (`DataUtil/CFmIntParam.cpp`).
  The dose column is 0, as `-FmDose` is: no dose weighting.

An mdoc's `NumSubFrames` does not count EER frames (Tomography writes 1 for a
movie of 728), so `cryoet init` takes the count from each file and keeps the
mdoc's in the image's metadata as `mdoc_num_subframes`.

The gain of a Falcon EER dataset is usually a TIFF named `*.gain`, 32-bit float
at the sensor size. By the sources, MotionCor3 reads a TIFF bottom row first
(`TiffUtil/CLoadTiffImage.cpp`) and decodes EER frames top row first
(`EerUtil/CDecodeEerFrame.cpp`), so the gain should need `--motioncor3-gain-flip
1`. That is not yet measured: `scripts/eer_gain_flip.sbatch` decides it with the
MotionCor3 that runs the lanes, on EMPIAR-11830.

The sums keep the EER files' row order. IMOD reads an EER file as any TIFF,
bottom row first, and its alignframes can apply the files' TIFF Orientation tag
(2 on EMPIAR-11830), so its sums of the same movies differ from MotionCor3's by
a flip; the tilt axis and the hand belong to the orientation the sums have.

## Tests

The default test suite never needs MotionCor3 or CUDA. It injects a fake process
runner and covers command construction, MRC and Zarr canonicalization, artifact
provenance, CUDA enforcement, missing executables, process failures, malformed
outputs, and output preflight behavior.

Official project: <https://github.com/czimaginginstitute/MotionCor3>
