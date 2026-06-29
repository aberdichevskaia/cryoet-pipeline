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
- low-variance input tilts are recorded and excluded from alignment rather than
  silently producing extreme transforms;
- transform file or alignment artifact exists;
- transform count matches tilt count when applicable;
- transform values are finite and within broad sanity bounds;
- a reduced prealigned MRC preview is retained for inspection in `3dmod`;
- residual shifts are measured on the preview and recorded with explicit
  warning and failure thresholds;
- alignment summary plots or logs are recorded.

### Reconstruction

- tomogram artifact exists;
- volume shape, dtype, voxel size, and axes are recorded;
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
