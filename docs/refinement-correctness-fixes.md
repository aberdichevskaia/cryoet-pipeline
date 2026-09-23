# Refinement correctness fixes — 2026-09-23

These are contract/geometry fixes, not a change of alignment scorer or a claim
that refinement now improves the CZII maps. No training, dataset rerun, search
parameter sweep, or automatic migration of old results is part of this change.

## Corrected behavior

- **Zero cone-search shift range:** `shift_range_px=0` preserves the supplied
  shifts at every iteration, including nonzero starting shifts. Previously the
  schedule opened a one-voxel search even when zero was requested. Positive
  search ranges retain their existing one-voxel floor.
- **Extraction poses:** the raw crop remains exactly the same array, beginning
  at `round(pick) - box_size // 2`. Initial STAR sampling shifts now equal
  `pick - crop_start - (box_size - 1) / 2` in tomogram XYZ axes. This retains
  fractional pick coordinates and reconciles even boxes with the resampler's
  geometric centre. An integer pick in an even box therefore starts at
  `(0.5, 0.5, 0.5)`, not `(0, 0, 0)`. Coordinates, Euler angles, selection and
  crop-edge rules do not change. No interpolation is added during extraction.
- **Map-to-template FSC padding:** maps of different parity now share the
  same geometric centre. Integer zero-padding is followed by a fixed +0.5 voxel
  Fourier translation on each axis when necessary. This is not fitted map
  registration. Fourier amplitudes are preserved except for the unpaired
  Nyquist planes of an even target, whose real half-sample cosine is zero.
  Fractional translation is band-limited/periodic on the padded grid; its
  interpolated boundary is not an exactly compact zero extension. Same-parity
  padding remains bit-for-bit integer padding. The CLI logs the correction.

## Existing projects

New cubic-crop artifacts and Zarr attributes carry
`extraction_pose_convention = "geometric-center-v1"`. Missing markers mean
legacy/unknown extraction, not an invitation to change the artifact label.
Old STARs are still read as written: no half-voxel shift is secretly added at
load time, and already-refined shifts must not be corrected a second time.

Keep historical projects unchanged. A corrected extraction establishes a new
baseline: compare fixed-pose averaging and refinement using the **same newly
extracted stacks and initial STARs**. Do not compare a new refined average to
an old zero-shift baseline as if only refinement changed. PyTom matching and
GT selection need not be repeated to reuse the same picks.

Old FSC JSON remains historical. When comparing different-parity maps, recompute
both the initial and refined map-to-template FSC with the same updated code;
that operation does not require rerunning alignment. Map-to-template FSC still
is not an independent resolution estimate.

## Regression coverage

`tests/test_subtomogram_backend.py` checks the actual backend's zero/positive
shift windows with and without wedge scoring, unchanged crop voxels, and the
sampling frame for integer/fractional centres in odd/even boxes.
`tests/test_fsc.py` checks centre registration in both argument orders, unchanged
same-parity padding, and Fourier amplitudes outside Nyquist planes.

The companion benchmark fixes the diagnostic runner's three-value reference
update API, adds transfer-scale corrections to existing extraction shifts
instead of replacing them, and refuses GT-control recrops from legacy source
stacks before starting work. Tiny API and fixture tests verify those contracts;
they are not new scientific experiments.
