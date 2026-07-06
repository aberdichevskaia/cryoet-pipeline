# QC strategy

## Principle

Every processing stage should produce lightweight, machine-readable QC signals
and small human-inspectable previews. The MVP should not depend on full
interactive 3D web visualization.

## MVP visualization stance

For the first release:

- use IMOD locally for expert inspection of `.mrc`, `.st`, and `.rec` exports;
- generate small binned previews for quick qualitative checks;
- prefer PNG/AVI/GIF-style summaries over large interactive 3D web rendering;
- keep full web cockpit visualization for a later release.

## Stage-level QC expectations

### Ingest

- mdoc parses successfully;
- expected tilt count is present;
- expected frames per tilt are present for known benchmark series;
- every mdoc frame path resolves to a local file;
- raw and mdoc pixel spacing differences are recorded.

### Motion correction

- corrected projection count matches tilt count;
- every projection has expected 2D shape and dtype;
- no NaN or infinite values;
- variance is nonzero;
- small binned projection previews can be generated for selected tilts;
- registry records backend, parameters, source frame file, size, and storage role.

### Tilt stack

- stack count matches corrected projection count;
- tilt-angle order matches manifest order;
- stack axes and pixel spacing are recorded;
- `.tlt` export matches manifest angles when requested.

### Alignment

- dose-symmetric acquisition order is normalized to ascending tilt angle for
  correlation, then transforms are mapped back to manifest order;
- `tiltxcorr` relative transforms are converted with `xftoxg` before they enter
  canonical state or are applied to an image stack;
- canonical and exported `.xf` transforms always have global semantics;
- low-variance input tilts are recorded and excluded from alignment rather than
  silently producing extreme transforms;
- transform file or alignment artifact exists;
- transform count matches tilt count when applicable;
- transform values are finite and within broad sanity bounds;
- a reduced prealigned MRC preview is retained for inspection in `3dmod`;
- residual shifts are measured on the preview and recorded with explicit
  warning and failure thresholds;
- automatic fiducial seeds and tracked models record contour counts, point
  counts, per-view coverage, and explicit QC status;
- fine alignment records mean, RMS, percentile, and maximum point residuals;
- high-residual points may be pruned only within a configured fraction, and the
  original model, cleaned model, threshold, and rerun count remain traceable;
- two-surface analysis records recommended tomogram thickness, Z shift, tilt
  angle offset, and X-axis tilt in unbinned coordinates;
- fine alignment reruns with the total `AngleOffset` and `AxisZShift` until the
  remaining surface correction is within explicit tolerances or the bounded
  iteration limit is reached;
- alignment summary plots or logs are recorded.

### Reconstruction

- tomogram artifact exists;
- reconstruction accepts only the final fine-aligned stack and its matching
  fine-alignment artifact;
- solved tilt angles are used instead of reverting to acquisition angles;
- `AngleOffset` and `AxisZShift` are applied in fine alignment, while
  reconstruction applies only the remaining X-axis tilt and thickness; the
  same Z shift must not be applied twice;
- volume shape, dtype, voxel size, and axes are recorded;
- IMOD `YZX` reconstruction layout is explicitly converted to canonical `ZYX`;
- no NaN or infinite values;
- small binned slices or preview movie are generated;
- reconstruction parameters and parent artifacts are recorded.

## Failure handling

A failed stage should avoid producing ambiguous downstream state. Prefer:

- explicit exceptions for missing files, invalid shapes, unsupported formats, and
  malformed metadata;
- partial output avoidance when possible;
- registry updates only after successful artifact writes;
- small error summaries that a future UI or copilot can display.

## Future direction

Once deterministic stage QC is stable, an external controller can use the same QC
records to suggest reruns, parameter forks, or backend substitutions. That
controller should be layered on top of the pipeline, not embedded inside the
processing functions.
