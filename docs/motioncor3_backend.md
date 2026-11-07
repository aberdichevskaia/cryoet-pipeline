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
5. judges its patch alignment from MotionCor3's patch log (see "Patch
   alignment QC");
6. validates that the result is one finite 2D projection with the expected size;
7. converts the result to the storage-policy format and registers a standard
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
- `--motioncor3-eer-fractions` (`[motion] eer_fractions`): the groups the EER
  frames are summed in before alignment, written per movie as the `-FmIntFile`
  line `N G 0`. Unset, the count comes from each movie's dose: as many groups
  as keep each at 0.2 e/Å² or more, and at least 2. The dose is the file's
  `totalDose` in e/pixel over the movie pixel squared; a movie that states no
  dose per pixel gets 10 groups. The rule of thumb for tomography is at least
  0.1–0.15 e/Å² per aligned frame, below which the frames no longer align,
  and tilt movies of physical frames carry 0.2–0.3; more groups follow the
  drift more finely, fewer align more surely. An EMPIAR-11830 tilt, 6.53
  e/pixel at 1.9037 Å or 1.80 e/Å², gets 9. The artifact records the tilt's
  dose, the groups made and the dose per group, also when the count is given.
  G is N // fractions: MotionCor3 adds the N % G frames left over to the last
  groups, one each, and writes past the end of its array when they outnumber
  the groups (`DataUtil/CFmIntParam.cpp`). It so makes N // G groups, a few
  more than asked at times (50 of 728 frames gives 52). The dose column is 0,
  as `-FmDose` is: no dose weighting.

An mdoc's `NumSubFrames` does not count EER frames (Tomography writes 1 for a
movie of 728), so `cryoet init` takes the count from each file and keeps the
mdoc's in the image's metadata as `mdoc_num_subframes`.

The gain of a Falcon EER dataset is usually a TIFF named `*.gain`, 32-bit float
at the sensor size. MotionCor3 reads a TIFF bottom row first
(`TiffUtil/CLoadTiffImage.cpp`) and decodes EER frames top row first
(`EerUtil/CDecodeEerFrame.cpp`), so such a gain has to be turned upside down.
With EER movies and a `.gain`, the backend passes `-FlipGain 1` unless
`--motioncor3-gain-flip` (`[motion] gain_flip`) gives a flip, and the artifact
records `gain_flip_given: false`. An MRC gain keeps MotionCor3's default, 0: it
lies the way whatever wrote it laid it. TIFF movies go through the same reader
as the gain and need no flip by the sources (not measured).

Measured on 5 November 2026 by `scripts/eer_gain_flip.sbatch` (job 3890116, the
MotionCor3 the MDC lanes run, which printed "Gain reference has been loaded"),
from `-Align 0` sums of three EMPIAR-11830 movies with no gain and with each
flip:

- over the three movies, only the flipped gain anticorrelates with the counts
  (r −0.016 against −0.0005, where 1/√n is 0.00025);
- each movie's sum is flatter with `-FlipGain 1`, and so are the three summed;
  the gap (in squared high-passed spread over the mean, 1.4×10⁻⁴) is the same
  at 6.5 and at 19.5 electrons per pixel, as a fixed pattern of the sensor
  gives;
- the applied gain read off the sums (corrected over uncorrected) turns exactly
  upside down with the flag (correlation 0.997), and it does not look the same
  upside down (0.004), so the flag matters.

Read from one movie alone, the applied gain does not come out as the gain
(0.15–0.19 in the same upside-down test), so that script's per-movie verdicts
are not used; why one movie differs is not known.

The sums keep the EER files' row order. IMOD reads an EER file as any TIFF,
bottom row first, and its alignframes can apply the files' TIFF Orientation tag
(2 on EMPIAR-11830), so its sums of the same movies differ from MotionCor3's by
a flip; the tilt axis and the hand belong to the orientation the sums have.

## Patch alignment QC

With more than one patch, MotionCor3 writes `<movie>-Patch-Patch.log` into
`-LogDir`: for each patch, its shift in every frame against the whole-frame
alignment, in the pixels of the frames it aligned (the log's `# Movie size`).
After each movie the backend reads it. A patch counts as moved beyond the
tolerance when its largest shift exceeds 1.5 nm, the fine alignment's failure
threshold, since a tile of the sum put further off is a defect alignment cannot
take out; a NaN or overflowing shift counts as beyond. A tilt's patch alignment
failed when at least half its patches moved beyond. The artifact records this
as `patch_alignment` (frames, patches, `tolerance_px`, `patches_beyond`,
`median_patch_shift_px`, `failed`, the log), or as the reason the tilt was not
judged: no log, a log older than the run, one that does not read, or a patch
count other than the grid's. With `-Patch 1 1` there is nothing to judge and it
is null.

`prepare-tilt-series` (and `correct-motion`) then writes
`qc/<TS>/motion/<TS>_motion_qc.json`, registered as QC type
`motion_correction`: `fail` naming the tilts whose patch alignment failed,
`warning` when a tilt was not judged, `pass` otherwise. The command prints the
verdict and its warnings, so the lane's log has them. The verdict stops
nothing: the stack is built either way. A run without patch alignment removes
the report of an earlier one, so the report in place describes the latest
motion correction.

Measured on EMPIAR-11830 Position_1 (7 November 2026, 5 × 5 patches, 41 tilts
from −75 to +45°): five tilts, −75/−72/−69 and +42/+45, had all 25 patches
beyond 7.9 px, each by 125 px or more (medians 470–700 px). Of the other 36,
none had more than one patch beyond, that one by at most 11.9 px, and their
medians were at most 1.25 px. Those five
are the tilts whose sums showed straight seams on the patch grid and broke
coarse alignment. `scripts/mc3_patch_check.sbatch` (job 3901454) found the
tiles of the −75 and +45 sums 150–480 px from where a `-Patch 1 1` sum has
them, and those of −45, −15 and +15 within 1.4 px.

Not known: why those five. They are the darkest tilts and also the only ones
with 3 or 4 frames (the others have 5–9), since an EER movie's groups follow its
dose, and MotionCor3 aligns patches on groups of 4 frames (`-Group 1 4`, its
default, as its output prints). Too few frames and too little signal per patch
both fit. Rerunning a 4-frame tilt with more groups, or a 5-frame one with 4,
would tell them apart: `scripts/mc3_patch_check.sbatch` with `FRACTIONS` does
that and says which (its header has the command for Position_1).

Not checked: the log's name for MRC and TIFF movies. `<movie stem>-Patch-Patch.log`
is what MotionCor3 wrote for EER. A tilt whose log is not found is reported as
not judged, not as passed.

## Tests

The default test suite never needs MotionCor3 or CUDA. It injects a fake process
runner and covers command construction, MRC and Zarr canonicalization, artifact
provenance, CUDA enforcement, missing executables, process failures, malformed
outputs, and output preflight behavior. `tests/test_motion_patch_qc.py` covers
the patch log as MotionCor3 writes it (the start of a real one), the rule, the
record of each tilt, and the series report.

Official project: <https://github.com/czimaginginstitute/MotionCor3>
