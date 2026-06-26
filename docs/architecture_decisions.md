# Architecture decisions

## Purpose

This document records the boundary between the pipeline's internal source of
truth and external ecosystem integrations. The goal is to reuse strong existing
tools without coupling the core pipeline to any one backend, file layout, model
registry, or downstream workflow.

## Core rule

```text
Pipeline manifests + artifact registry + Zarr arrays are the internal truth.
Everything else is an adapter, export format, or backend implementation.
```

The pipeline should keep its own typed project state, normalized manifests,
artifact records, lineage, parameters, and large array storage. External tools
may read from or write to these records through adapters, but they should not
become the canonical state of the project.

## Internal truth

The internal representation consists of:

- project configuration and tilt-series manifests;
- artifact records with paths, kinds, lineage, parameters, versions, and array
  metadata;
- Zarr stores for large intermediate and derived arrays;
- JSON manifests and registries for portable, inspectable state.

Internal records should use pipeline-owned models. Backend-specific objects from
TeamTomo, IMOD, AreTomo, BioImage.IO, Croissant, RELION, Dynamo, or copick should
be converted at adapter boundaries.

## External layers

### BioImage.IO

BioImage.IO is an adapter layer for machine-learning models. It is useful for
denoising, restoration, segmentation, heatmap prediction, and future picking
models when those models can be packaged with BioImage.IO metadata and weights.

BioImage.IO should not be used as the workflow engine, artifact registry, or
storage model for the cryo-ET preprocessing pipeline.

### Croissant

Croissant is a dataset metadata and export layer. It can describe raw frames,
metadata files, manifests, tomograms, segmentations, picks, labels, and
train/validation/test splits in a standardized ML dataset format.

Croissant descriptions should be generated from the pipeline's internal
manifests and artifact registry. Croissant should not replace the internal
project state.

### copick

copick is the downstream cryo-ET exchange layer for tomograms, segmentations,
picks, meshes, object classes, and related analysis artifacts. It is the preferred
bridge to downstream annotation, segmentation, picking, and subtomogram-analysis
tools that understand copick conventions.

copick should be treated as an adapter/export target, not as the canonical record
of preprocessing state.

### IMOD, RELION, and Dynamo

IMOD, RELION, and Dynamo compatibility is an export and interchange concern.
The pipeline should write IMOD-style `.st`, `.rec`, `.tlt`, `.xf`, RELION STAR
files, and Dynamo tables when needed so users can continue in established tools.

These formats should remain compatible outputs, not the internal source of truth.

## Storage principle

Large artifacts should be classified by storage role and retention policy as
soon as they are written. Raw data is immutable external input; Zarr plus
manifests and the artifact registry are the internal working store; IMOD/MRC
files are compatibility or debug outputs unless explicitly promoted.

The detailed storage policy lives in [storage_policy.md](storage_policy.md).

## Backend design principle

Backends should depend on pipeline-owned protocols and models. A backend may call
TeamTomo, IMOD, AreTomo, BioImage.IO, MotionCor-style tools, or custom Python
implementations internally, but its public boundary should accept pipeline
manifests/artifacts and return pipeline artifacts.

This keeps the project modular: replacing a backend should not require rewriting
the surrounding pipeline state, artifact lineage, or downstream exports.

## Testing rule

Every pipeline stage should be added with tests at the same time as the code.
The minimum expectation is:

- unit tests for pure parsing, serialization, validation, and path-planning code;
- contract tests for backend protocols using fake or lightweight backends;
- focused integration tests for each pipeline stage once a baseline
  implementation exists;
- fixture-based tests with tiny synthetic data before running full EMPIAR-sized
  datasets;
- regression tests for artifact lineage, parameters, and exported metadata.

External tools and heavyweight GPU backends should stay behind adapters so they
can be mocked, skipped, or tested with small fixtures without making the default
test suite slow or machine-specific.

## MVP implication

The near-term MVP should prioritize:

1. a small artifact registry;
2. stable backend protocols for each processing stage;
3. a simple internal motion-correction baseline;
4. IMOD-compatible exports for manual inspection;
5. later adapters for BioImage.IO, Croissant, copick, TeamTomo, and other
   ecosystem tools.
