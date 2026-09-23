# Opt-in particle-frame angular scoring

The existing `cone-search` scorer remains the default. Select the new path with:

```text
--backend cone-search --alignment-frame particle --missing-wedge --shift-range-px 0
```

This is an **angular-only** implementation. It retains arbitrary supplied
initial shifts, including geometric-centre and fractional-pick corrections,
but refuses shift search. An alignment mask is also refused. These restrictions
are validated both by the CLI and the backend, not silently ignored. The FFT
backend does not accept this mode. Fourier-compensated reference building is
still an independent option; selecting the scoring frame does not enable it.

## Observation model

Native poses satisfy `aligned(x) = raw(A x + s)`, with physical XYZ coordinates
relative to `(box_size - 1) / 2`. To predict the unchanged particle:

```text
prediction(u) = reference(A.T (u - s))
```

The native resampler therefore receives `A.T` and `-A.T @ s`. Using `-s` would
be incorrect for a general rotation. It samples the **unfiltered** reference
once with trilinear interpolation and zero exterior values. The data are not
rotated, shifted or re-interpolated while scoring candidate angles.

Mean-subtracted observation and prediction each receive exactly one copy of
the same raised-cosine low-pass response on the particle grid. Its cutoff is
`initial_low_pass_angstrom`, held constant as in the existing backend. Both are
restricted to the particle's own fixed laboratory-frame missing wedge, and NCC
is normalised with the rFFT Hermitian multiplicities. The wedge is not rotated
or re-discretised for every candidate. No spatial mask or fitted map alignment
is inserted.

After choosing a pose, the original mean-centred raw particle is sampled once
at that pose for the native averaging/reference-update code. Filtered scoring
data and predicted references never replace the source particles in the output.
Angle/shift conventions, candidate schedule, class assignment and reference
updates otherwise remain unchanged. Starting and final reported correlations
use the selected scorer, but references can change between iterations, so a
larger final NCC is not proof of improved structure.

## Provenance and limits

QC schema 6 records `alignment_frame`, `alignment_score`, and
`alignment_low_pass_angstrom`. The new score is `particle-frame-wedge-ncc`.
Older reports without the frame field decode as `reference`; existing score
names and default behaviour are preserved.

This is a better-defined discrete observation model, not an exact inverse of
resampling. It does not model CTF, coloured noise or amplitude transfer beyond
the binary wedge. Nor does it resolve uncertain pose maxima, implement a new
search schedule, or create independent half-set refinement.

The fixed synthetic gate lives in the benchmark repository as
`czii_native_pipeline_benchmark/tools/run_particle_frame_gate.py`. It calls these
native search kernels, with the original seed, 24 orientations and SNR 0.1.
Passing this gate is required before a bounded real-data comparison; it is not
a CZII result or an independent resolution claim.
