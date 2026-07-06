# MVP plan

## Summary

Build a non-agentic, extensible Python MVP that processes EMPIAR-10164 `TS_01`
and `TS_43` from multiframe MRC plus `.mdoc` metadata to visual/QC-ready
tomograms.

Internal storage is Zarr plus JSON manifests. External exports must remain
compatible with IMOD-style workflows: `.st`, `.rec`, `.tlt`, `.xf`.

The MVP should be agent-ready but not agentic: deterministic stages, clean
backend contracts, structured artifacts, logs, and QC signals should make it easy
to attach a future copilot or web cockpit without rewriting processing code.

## Decisions

- Runtime target: Linux + CUDA for full-size production processing; Apple Silicon
  MPS and CPU are supported as development/runtime preferences where backends allow it.
- Package shape: typed Python API first, CLI as a thin wrapper.
- First dataset target: EMPIAR-10164 `TS_01` and `TS_43`.
- Storage: Zarr artifact store plus policy-controlled MRC/IMOD exports.
- Third-party policy: adapters first; vendor or fork only when real
  customization requires it.
- Motion correction: first backend is average-only for baseline/debugging;
  global phase-correlation or third-party adapters can follow.
- Alignment: first backend is IMOD `tiltxcorr` plus `xftoxg` for coarse global
  alignment, followed by a separate fiducial-tracking/fine-alignment backend.
  `tttsa`, AreTomo, and other tools remain replaceable adapter options.
- Reconstruction: first backend is IMOD `tilt`; `torch-tomogram` and other
  reconstruction tools remain replaceable adapter options.
- Quality target: visual/QC-ready, not publication-grade.
- Visualization: no full 3D web viewer in the MVP; prefer IMOD exports plus
  lightweight previews.
- Benchmarks: public datasets and known failure cases should drive acceptance
  before expanding scope.

## First implementation slice

1. Project scaffold, package config, CLI entrypoint.
2. Core models: project config, tilt-series manifest, artifact records.
3. SerialEM `.mdoc` parser and EMPIAR file mapping.
4. Backend protocols for motion correction, alignment, and reconstruction.
5. Artifact registry with storage roles, retention policies, and size tracking.
6. Average-only motion-correction backend and tilt-stack preparation backend.
7. `prepare-tilt-series` command that corrects movies and prepares
   alignment-ready stack/angle artifacts as one user-facing step.
8. Canonical alignment models plus an IMOD coarse-alignment adapter that
   converts `tiltxcorr` relative transforms to global transforms with `xftoxg`,
   then emits normalized JSON and an IMOD-compatible `.xf`.
9. Coarse-alignment QC with a retained bin16 prealigned preview, residual-shift
   metrics, and machine-readable `pass`, `warning`, or `fail` status. This stack
   is diagnostic and is not a reconstruction input.
10. Automatic IMOD fiducial seeding and bead tracking with model coverage QC.
11. Fiducial-based `tiltalign` fine alignment with robust fitting, bounded
    outlier pruning, residual QC, and bounded two-surface positioning iterations
    that apply `AngleOffset` and `AxisZShift`.
12. Final aligned-stack generation with solved angles and explicit calibration
    provenance.
13. Positioned IMOD reconstruction from the fine-aligned stack, with `.rec`
    compatibility output, canonical `ZYX` Zarr, and central-slice QC artifacts.
14. Tests for metadata parsing, artifact serialization, registry behavior, CLI
    commands, backend contracts, and baseline processing.

## Supporting docs

- [architecture_decisions.md](architecture_decisions.md)
- [project_scope.md](project_scope.md)
- [agent_readiness.md](agent_readiness.md)
- [benchmark_strategy.md](benchmark_strategy.md)
- [qc_strategy.md](qc_strategy.md)
- [storage_policy.md](storage_policy.md)

## Acceptance for MVP

- Local EMPIAR data is provided by the user.
- `TS_01` validates as 41 tilts with 8 frames per tilt.
- `TS_43` validates as 41 tilts with 10 frames per tilt.
- Both tilt-series produce corrected projections, `.st`, `.tlt`, `.xf`, `.rec`,
  Zarr artifacts, and QC reports.
