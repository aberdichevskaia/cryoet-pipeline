# Refinement correctness fixes — 2026-09-23

These are contract/geometry fixes, not a change of alignment scorer or a claim
that refinement now improves the CZII maps. No training, dataset rerun, search
parameter sweep, or automatic migration of old results is part of this change.

## 2026-09-25 — the extraction premise was wrong for pytom picks

The extraction fix below took a pytom pick as the particle's centre. It is not.
pytom-match-pick reports where its template's voxel `T // 2` lands, but voltools
rotates the template about `(T - 1) / 2`; for an even template box the pick sits
half a voxel above the rotation centre along every lab axis, whatever the pose.
Initial shifts are now `pick - offset - crop_start - (box_size - 1) / 2`, with
`offset = particles.pick_center_offset_px(T)`: 0.5 for an even template, 0 for
an odd one. `T` comes from the pytom job the picking step records, or from
`--pick-template-box-px` for external picks; extraction refuses to guess. The
crop is unchanged. The marker is `rotation-center-v2`, and stacks also record
`pick_template_box_px` and `pick_center_offset_px`.

Measured on CZII, template 48 and crop 48: `geometric-center-v1` left every
particle half a voxel off in the lab frame. A Dynamo search moved them by
(-0.47, -0.45, -0.48) voxels, and zero shifts alone reproduced its map (26.70 A
against 30.04 A). The unmarked zero-shift extraction was right by cancellation,
the two boxes having the same parity, so its fixed-pose map is a correctly
centred baseline: the advice below not to compare against it is withdrawn.
`tests/test_pick_center_convention.py` pins pytom's convention with a numpy
replica of its paste, correlation, flip and roll, end to end through extraction.

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

Cubic-crop artifacts from this fix carry
`extraction_pose_convention = "geometric-center-v1"`, superseded above by
`rotation-center-v2`. Missing markers mean
legacy/unknown extraction, not an invitation to change the artifact label.
Old STARs are still read as written: no half-voxel shift is secretly added at
load time, and already-refined shifts must not be corrected a second time.

Keep historical projects unchanged. A corrected extraction establishes a new
baseline: compare fixed-pose averaging and refinement using the **same newly
extracted stacks and initial STARs**. Do not compare a new refined average to
an old zero-shift baseline as if only refinement changed. (Withdrawn
2026-09-25: that sentence kept the zero-shift baseline out of every comparison,
and it was the correctly centred one.) PyTom matching and
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
