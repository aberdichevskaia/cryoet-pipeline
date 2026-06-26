# Benchmark strategy

## Principle

Development should be benchmark-first. Pipeline stages, future copilot behavior,
and backend substitutions should be validated against public datasets with known
input structure and expected output properties before being trusted on large lab
data.

## First benchmark target

The MVP starts with EMPIAR-10164:

- `TS_01`: 41 tilts, 8 frames per tilt;
- `TS_43`: 41 tilts, 10 frames per tilt.

These tilt-series exercise the two frame-count cases in the initial dataset and
are small enough to use as the first real-data integration target.

## What to measure

Each stage should define both structural and qualitative checks.

Structural checks include:

- expected file counts;
- expected shapes and dtypes;
- expected tilt angle count and ordering;
- expected pixel spacing and binning metadata;
- absence of NaN or infinite values;
- non-empty outputs with plausible variance;
- registry lineage and parameters matching the run.

Qualitative checks include:

- small binned image previews;
- tilt-series or tomogram preview movies;
- summary plots for alignment or CTF diagnostics once those stages exist.

## Failure scenario corpus

The team should collect common failure cases from existing pipeline support
channels and mailing lists. These should become regression fixtures or scenario
tests when possible.

Examples of useful failure scenarios:

- missing or partial frame downloads;
- mdoc paths that do not map to local files;
- mismatched tilt counts;
- wrong frame count per tilt;
- malformed or missing alignment outputs;
- silent tool success with empty or corrupted downstream files;
- parameter choices that produce visibly poor reconstructions.

## Agentic comparison requirement

If a future copilot proposes parameter changes or backend substitutions, its
behavior should be compared with simpler deterministic baselines such as fixed
defaults, grid search, or Optuna-style parameter search. Agentic behavior should
earn its complexity against benchmark scenarios instead of being assumed better.

## Acceptance direction

Before expanding scope beyond reconstruction, the pipeline should be able to run
the chosen benchmark subset through reconstruction and produce:

- complete registry lineage;
- stage-level QC records;
- small visual previews;
- reproducible commands/configuration;
- clear failure messages for intentionally broken benchmark variants.
