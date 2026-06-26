# Storage policy

## Core rule

```text
Raw data is immutable external input.
Zarr plus manifests and artifact registry are the internal working store.
IMOD/MRC-style files are compatibility/debug exports unless explicitly promoted.
QC previews should be small and cheap to keep.
```

The pipeline must avoid silently duplicating terabyte-scale data. Every artifact
record should say what role the file plays, whether it can be recomputed, and how
large it is when the size is known.

## Artifact storage metadata

Artifacts carry these storage fields:

- `storage_role`: `source`, `canonical`, `cache`, `export`, `temporary`, or `qc`;
- `retention_policy`: `keep`, `recompute`, or `delete_after_export`;
- `can_recompute`: whether the artifact can be regenerated from tracked inputs
  and parameters;
- `size_bytes`: recorded size for file or directory artifacts when known.

The artifact registry can report total known size and size grouped by storage
role. Unknown sizes count as zero in totals; they should be filled by backends
whenever possible.

## Named policies

### debug

Use this while developing and visually checking outputs.

```text
format: mrc
storage_role: cache
retention_policy: keep
can_recompute: true
```

This creates IMOD-friendly files and keeps them for inspection. It is convenient
but disk-heavy.

### working

Use this for normal internal processing once the stage is trusted.

```text
format: zarr
storage_role: cache
retention_policy: recompute
can_recompute: true
```

This keeps recomputable intermediate arrays in a Python-native store and avoids
treating compatibility exports as internal truth.

### minimal

Use this for storage-constrained runs where intermediates should be treated as
short-lived.

```text
format: zarr
storage_role: temporary
retention_policy: delete_after_export
can_recompute: true
```

The pipeline can still record enough lineage to regenerate the artifact, but
cleanup tooling may remove it after downstream artifacts or exports exist.

## Current MVP behavior

`cryoet correct-motion` defaults to `--storage-policy debug`, because the current
development loop uses IMOD/`3dmod` for visual QC.

For lower disk usage, run:

```bash
cryoet correct-motion \
  --manifest outputs/dev-ts01-registry/manifests/TS_01.json \
  --registry outputs/dev-ts01-registry/artifacts.json \
  --out outputs/dev-ts01-registry \
  --storage-policy working
```

The first full `TS_01` debug run will create 41 float32 MRC corrected
projections. This is useful for visual inspection but should be treated as a
cache, not as canonical project state.
