# Project scope

## Product goal

Build a robust, deterministic cryo-ET preprocessing pipeline that can process
large public or lab datasets from raw frames through tomogram reconstruction.
The design should later support a web cockpit or agentic copilot, but the MVP
itself is a conventional tested pipeline.

The motivating user is a biologist who wants to process many tomograms without
manually editing fragile scripts or diagnosing silent failures across many
intermediate tools.

## First release scope

The first release focuses on:

```text
raw multiframe movies + acquisition metadata
  -> motion correction
  -> corrected projections
  -> tilt stack
  -> CTF metadata or estimation hook
  -> tilt-series alignment
  -> tomogram reconstruction
  -> lightweight QC outputs
```

Particle picking, subtomogram averaging, full web visualization, and autonomous
parameter search are intentionally out of scope for the first release.

## Non-goals for MVP

- No full end-to-end particle-picking-to-structure pipeline.
- No agent embedded inside the numerical processing loop.
- No full interactive 3D web viewer.
- No single hard dependency on one external tomography package.
- No assumption that intermediate IMOD/MRC files are the internal source of
  truth.

## Target operating assumptions

- Raw microscope output can be terabyte-scale.
- Most large intermediate artifacts should be cacheable or recomputable.
- Processing may eventually run on HPC or GPU workstations, while orchestration
  should remain reproducible from structured manifests and registry state.
- Public benchmark datasets are the first validation target.

## UX direction

A later user-facing system may expose a web cockpit with chat, status,
configuration, logs, QC previews, and visualization. The code being built now
should provide the stable API and artifact state that such a UI can drive.
