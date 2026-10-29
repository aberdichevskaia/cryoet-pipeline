# Pipeline map

Every step of a lane, in the order the code runs them. For each step: the lane
keys that set it, the `cryoet` flags they become, where the outputs, QC records,
and logs land, and what the QC checks.

The sources of truth are [`scripts/run_series.py`](../../scripts/run_series.py)
(the step order and the mapping from keys to flags) and
[`configs/lanes/example.toml`](../../configs/lanes/example.toml) (the keys, with
comments on how to choose them). If this file disagrees with them, they are
right; fix this file.

## How a lane runs

```text
lane TOML
  → scripts/submit_lanes.py   checks every config, runs `cryoet init` for a new
                               frames project, submits one Slurm job per
                               [[run.stage]], chained with afterok
  → scripts/lane.sbatch        activates the environment and IMOD, runs the
                               preflight checks, then run_series.py --execute
  → scripts/run_series.py      for each series in turn, one `cryoet <step>` per
                               step from..until; stops at the first failure
```

- **The order is code.** The config chooses three things only: `[lane] input`
  (`frames` or `stack`), the alignment method (`[fiducials]`: beads or patches),
  and whether membranes are segmented (`[segment]` present or not). List a
  lane's steps with
  `python3 scripts/run_series.py --project P --config C --series X --list-steps`.
- **Stages** are consecutive slices of that chain, each one Slurm job over every
  series, with its own resources. A gap or an overlap between stages is refused
  before anything is submitted. A stage may start mid-chain:
  `ryr1-bin8.toml` starts its CPU stage at `build-ctf-stack`.
- **When a step fails**, the job stops and the Slurm log says
  `FAILED at <step> (exit N) for <series>` with the `--from` to resume from.
  Steps after the failure, and series after the failed one, do not run.
- **Unknown keys are refused.** A key that no step reads stops `run_series.py`
  and `submit_lanes.py` before anything runs: a misspelling, a key in the wrong
  section, an unknown section. The refusal names every such key at once, with the
  key it was probably meant to be. The table of keys is `LANE_KEYS` in
  `run_series.py`. Before 27 October 2026 such keys were ignored, and the step ran
  with its default.
- **Submitting leaves a record.** Since 28 October 2026 every dry run and every
  submission of `submit_lanes.py` is appended to `<project>/journal.jsonl`. An
  entry holds the plan, the user and the client. A submission also stores the
  config text, its diff against the lane file it was copied from, the commit,
  `--reason` and the job IDs.
- **Recipe scripts go through `scripts/submit_recipe.py`,** with the same checks
  and the same journal: `ctf-window`, `pick-ablation`, `null-decoys`, `repick` and
  `average` (`--list` says what each runs). One recipe is one job, named
  `<project>-<recipe>`, on the partition of the lane's stage that runs the step it
  repeats, with the recipe's own CPUs, memory and time. Its parameters come from
  `--set NAME=VALUE` only; one left in the shell is refused. Outputs already under
  its labels are a problem without `--overwrite`.
- **Outputs are replaced only with `--overwrite`,** for that one submission.
  `OVERWRITE` set in the shell is refused.
- **A series that is busy is refused.** If a queued or running job of the same
  project works on a series the lane would run, the dry run and the submission
  both refuse it. The job's series come from the journal or its log. A job named
  `<project>-<stage>` that neither accounts for counts as being on every series.
  Only the user's own jobs are asked about.
- **A submission from the assistant has to be the dry run it showed.** It needs
  `--reason`, and it is refused unless the config's latest journal entry is the
  same person's dry run of exactly this plan: no problems, at most 30 minutes old.
  For a recipe, the latest entry of the same recipe on the same series counts,
  so dry runs of the lane and of recipes on one config do not stand in for each
  other.
- **`[lane] variant`** labels the outputs from `build-final-aligned-stack` on.
  Everything before it is shared by every lane in the project and unlabelled.
  So is the CTF fit stack and the CTF model, so that any lane can use them.

### Which combinations have run on real data

Of the eight combinations of input, alignment and membranes, two had run on
real data as of 27 October 2026:

- **stack + patches, no membranes:** `ribo-bin8.toml`, EMPIAR-11538.
- **frames + beads + membranes:**
  - `gag-bin8.toml`, EMPIAR-10164, MRC movies;
  - the RyR1 lanes at MDC, TIFF movies.

The other six produce valid commands but have not been run. A lane using one of
them gets a warning that it is the first. Keep this list in step with the README.

### Where things are

Paths are relative to the project directory (`[run] project`), with `<TS>` for
the series id.

```text
artifacts.json              the registry: every artifact, its path and parameters
manifests/<TS>.json         per series: tilt images, angles, raw pixel size, notes
step_timings.jsonl          one line per step run: seconds, disk delta, exit code,
                            and (from 27 October 2026) the lane that ran it
emdb-cache/                 maps fetched from EMDB
reports/                    PDF reports (`cryoet report`)
qc/<TS>/<step>/             QC records (JSON) for most steps, see below
```

The Slurm log of a lane job is `slurm-<project name>-<stage>-<job id>.log` in
the directory the job was submitted from: the repository root, for the jobs
`submit_lanes.py` submits. A job submitted by hand from another folder logs
there, under the job name it was given; `sacct -j <job id> -o WorkDir` says
where, and `show_project.py` and `show_failure.py` look there too. The log holds
the job header (node, GPU, environment, the config, the steps), the preflight
result, a `# ---- <TS> (<lane>) ----` line for each series with every command as
run under it, the time and disk used per step, and the failure line.

External programs write their own logs next to the step's outputs, named after
the program: `<TS>_..._beadtrack.log`, `<TS>_fine_tiltalign.log`,
`<TS>_..._newstack.log`, and so on. The real error is often in these, not in the
Slurm log.

### Reading results

- `python3 scripts/show_project.py <lane config>` shows every series of the lane
  at once.
  - Per step: the last run and its exit code (from `step_timings.jsonl`), and
    the QC verdict.
  - Below the table: the crashes with the Slurm log that holds them, the QC
    records that did not pass with their warnings, and the lane's jobs. Where
    `sacct` and `squeue` can be run, it adds their state and the jobs still
    waiting.
  - Shared steps count whichever lane ran them. Labelled steps count only
    under the lane's variant, so experiments under other labels are left out.
    A job's failure of a labelled step counts only for the lane the job ran,
    which its log names. Lanes of one project share job names, so the jobs list
    gives each job's lane.
  - `--json` gives the same for a program; `--series` narrows it.
- `python3 scripts/show_failure.py <lane config>` shows, for every step that
  failed, the lines that matter.
  - From the step's part of the Slurm log: the step's command, the lines that
    look like errors, and its last lines.
  - From the logs its programs wrote while it ran: each command that exited
    non-zero, with its error output and exit code.
  - A job that stopped in its preflight, or stopped in the middle of a step,
    counts as a failure too. A failure fixed by a later run does not. Another
    lane's preflight stop counts only if every step that job was to run is
    shared.
  - After the failures, without excerpts: each step that failed in a job and
    passed when run again since, with those jobs; and each step that ran but
    whose result failed QC, with its QC file and first warning. "0 failed steps"
    means nothing is broken now, not that nothing ever failed.
  - The output is bounded and carries line numbers. `--series`, `--step` and
    `--json` work as above.
- `python3 scripts/show_series.py --project P --series TS` — the series'
  artifacts, its alignments, every QC record with status and warnings, and what
  to open in 3dmod or ChimeraX. `--kind` narrows the artifact table.
- `python3 scripts/show_timings.py` over `step_timings.jsonl` — time and disk
  per step.
- `cryoet report` — the PDF report for biologists.

Every QC record has a `status` (`pass`, `warning`, `fail`) and a list of
`warnings` in free text. A QC failure does not stop the chain by itself; the
step exits 0 and the next step runs.

## The steps

Required keys are marked with `*`. The arrow shows the flag a key becomes. A key
that is absent adds no flag, and the command's own default applies.

### 0. `cryoet init` — frames lanes only, project-level

Run by `submit_lanes.py` on the submit node for a frames lane whose project does
not exist yet. It is never run on an existing project, because it starts the
registry empty: adding a series to an existing frames project is refused, and a
new project is the way.

- Keys: `[data] frames`* → `--frames`; exactly one of `mdocs` → `--mdocs` and
  `frames_pattern` → `--frames-pattern`; `pixel_size_angstrom` →
  `--pixel-size`; `[run] series` → `--tilt-series`, once per series.
- Writes: `manifests/<TS>.json`, a new `artifacts.json`.
- Prints the tilts and angle range found per series. Check them before the jobs
  run, especially with `frames_pattern`.

### 1. `prepare-tilt-series` — frames lanes

Motion correction, then the tilt stack from the corrected sums.

- Keys: `[motion] backend`* → `--motion-backend`; `storage_policy` →
  `--storage-policy`; `gpu` → `--motioncor3-gpu`; `patch_x`, `patch_y` →
  `--motioncor3-patch-x`, `--motioncor3-patch-y`; `pixel_size_angstrom` →
  `--motioncor3-pixel-size`; `gain_reference`, `gain_rotation`, `gain_flip` →
  `--motioncor3-gain`, `--motioncor3-gain-rotation`, `--motioncor3-gain-flip`;
  `ftbin` → `--motioncor3-ftbin`; `[executables] motioncor3` →
  `--motioncor3-executable`. With MotionCor3, `--device cuda` is always added.
- Writes: `corrected/<TS>/` (one sum per tilt), `stacks/<TS>/` (the tilt
  stack). MotionCor3's working files are in `motion/motioncor3/<TS>/`, its logs in
  `logs/motioncor3/<TS>/`.
- Preflight in `lane.sbatch`: `$MOTIONCOR3` is executable, its libraries load
  (`ldd`), and it has code the job's GPU can run (`cuobjdump`).
- `storage_policy`: `working` keeps Zarr; `debug` (the command's default) writes
  one float32 MRC per tilt and fills disks.

### 1′. `import-tilt-stack` — stack lanes

- Keys: `[data] mdoc`*, `stack`*, with `{series}` replaced by the id →
  `--mdoc`, `--stack`; the id → `--tilt-series-id`.
- Writes: the manifest, the registry entries, and `stacks/<TS>/`. Refuses if the
  stack or the manifest exists already.

### 2. `align-tilt-series` — coarse alignment

- Keys: `[align] binning`* → `--binning`; `min_std_ratio`* →
  `--min-std-ratio`; `tilt_axis_angle` → `--tilt-axis-angle`. That last key is
  required with `frames_pattern`, since nothing else gives the axis.
- Writes: `alignments/<TS>/` (transforms; `tiltxcorr` and `xftoxg` logs).
- Views with too little variance (`min_std_ratio`) are excluded and listed.

### 3. `qc-coarse-alignment`

- Keys: `[qc_coarse] preview_binning`* → `--preview-binning`;
  `residual_warning_px`* → `--residual-warning-px`.
- Writes: `qc/<TS>/coarse_alignment/` (QC record, a coarse-aligned preview stack
  at the preview binning, logs).
- QC: residual shift, 95th percentile above `residual_warning_px` (2 preview
  px by default) → warning; maximum above 5 preview px → failure. The failure
  threshold is not a lane key. One preview pixel is `preview_binning` stack
  pixels.
- A failure here does not stop the chain, and a later step can recover it: on
  RyR1 097 the coarse QC failed and fine alignment on the beads was fine.

### 4. `generate-fiducial-seed`

- Keys: `[fiducials] seed_backend`* → `--backend` (`imod-autofidseed` for
  beads, `imod-patch-seed` for patches).
  - Beads: `tracking_binning` → `--tracking-binning`; `fiducial_diameter_nm` →
    `--fiducial-diameter-nm`; `raw_pixel_spacing_angstrom` →
    `--raw-pixel-spacing`; `target_beads` → `--target-beads`.
  - Patches: `pre_binning` → `--pre-binning`; `num_patches_x`,
    `num_patches_y` → `--num-patches-x`, `--num-patches-y`; `patch_overlap` →
    `--patch-overlap`; `patch_margin_px` → `--patch-margin-px`. Unset, the
    margin is 5% of the binned size on each axis, which is etomo's rule. That is
    the default since 27 October 2026; before, it was 100 px.
- Writes: `fiducials/<TS>/` (seed model, `<TS>_seed_qc.json`, `autofidseed`
  log).
- Both seeds track on a binned, prealigned stack. Until 27 October 2026 the
  patch seed did not prealign it, so the coarse shift was applied twice (playbook
  A-3).
- The seed backend and the tracking backend must be a pair (beads with beads,
  patches with patches). `run_series.py` refuses a mixed pair.

### 5. `track-fiducials`

- Keys: `[fiducials] track_backend`* → `--backend`; `min_tracked_fiducials` →
  `--min-tracked-fiducials`; `coverage_warning`, `coverage_failure` →
  `--coverage-warning`, `--coverage-failure`.
  - Beads only: `rounds` → `--rounds`.
  - Patches only: `patch_filter_sigma1`, `patch_filter_radius2`,
    `patch_filter_sigma2` → the `--patch-filter-*` flags.
  - Each backend refuses the other's flags.
- Writes: `fiducials/<TS>/` (tracked model, `<TS>_tracking_qc.json`,
  `beadtrack` log).
- QC: fewer fiducials than the minimum (10 by default) → failure. Coverage (the
  share of views in which they are tracked) below 0.8 → warning, below 0.5 →
  failure.
- Refuses a patch seed made before 27 October 2026, saying "rerun
  generate-fiducial-seed".

### 6. `fine-align-tilt-series`

- Keys: `[fine_align] robust_fitting` → `--robust-fitting`;
  `auto_prune_outliers` → `--auto-prune-outliers`; `max_pruned_fraction` →
  `--max-pruned-fraction` (0.02 by default, sized for beads; 0.08 for a 5 × 5
  patch grid); `surfaces_to_analyze` → `--surfaces-to-analyze` (2 for beads, 1
  for patches).
- Writes: `alignments/<TS>/fine/` (`<TS>_fine_alignment_qc.json`, the
  `tiltalign` logs, the outlier-pruning log).
- QC: mean residual above 0.8 nm → warning, above 1.5 nm → failure. Largest
  point residual above 5 tracking px → warning, above 20 → failure. Also warned:
  large post-positioning tilt-angle or Z corrections, and a two-surface analysis
  that put too few points on one surface.

### 7. `build-ctf-stack` — `build-final-aligned-stack`, unlabelled

- Keys: `[binning] ctf_fit`* → `--output-binning`. No variant: every lane of
  the project shares this stack.
- Writes: `aligned/<TS>/` (the aligned stack, `newstack` log).

### 8. `estimate-ctf` — unlabelled

- Keys: `[ctf] voltage_kv`* → `--ctf-voltage-kv`;
  `spherical_aberration_mm`* → `--ctf-spherical-aberration-mm`;
  `amplitude_contrast` → `--ctf-amplitude-contrast`; `min_defocus_angstrom`,
  `max_defocus_angstrom` → `--ctffind-min-defocus-angstrom`,
  `--ctffind-max-defocus-angstrom`; `defocus_search_window_angstrom` →
  `--ctffind-defocus-search-window-angstrom` (per view, either side of the
  0-degree fit; ±1000 Å by default); `defocus_step_angstrom` →
  `--ctffind-defocus-step-angstrom`; `max_excluded_views` →
  `--ctf-max-excluded-views`; `determine_sample_tilt` →
  `--ctffind-determine-sample-tilt` or its `--no-` form; `[executables]
  ctffind` → `--ctffind-executable`.
- Without `[executables] ctffind`, the step takes `ctffind5`, then `ctffind`,
  from `PATH`. At MDC `ctffind` can be CTFFIND 4.1.14, which the step refuses;
  name CTFFIND 5 through `$CTFFIND5` there. Since 28 October 2026 the dry run
  of a lane or recipe that runs this step without naming a CTFFIND asks the one
  on `PATH` for its version, and refuses CTFFIND 4. One it cannot find or ask is
  noted, not refused.
- Writes: `ctf/<TS>/` (the IMOD `.defocus` model and a diagnostic stack of
  measured against fitted Thon rings); QC in `qc/<TS>/ctf/`.
- QC: every view whose fit stops at the low-resolution end of the fit range
  counts as unfitted. Any unfitted view → warning, listed with its angle. Up to
  `max_excluded_views` (5 by default) of them are left out at correction.
- The command also takes `--variant`. That is how a CTF experiment writes
  alongside the lane's model without touching it (see
  [rerun_recipes.md](rerun_recipes.md)).

### 9. `build-final-aligned-stack` — labelled

- Keys: `[binning] tomogram`* → `--output-binning`; `[lane] variant`* →
  `--variant`.
- Writes: `aligned/<TS>/`.

### 10. `correct-ctf`

- Keys: `[lane] variant` → `--variant`.
- Writes: `ctf/<TS>/`; QC in `qc/<TS>/ctf/`.
- QC: the views left out as unfitted (by z); a warning when the model was fitted
  at a pixel size other than the stack's.

### 11. `reconstruct-tomogram`

- Keys: `[reconstruct] thickness` → `--thickness` (in binned pixels);
  `[lane] variant` → `--variant`.
- Thickness, if unset, is derived from the beads, which sit on both faces.
  With patches it is required: `run_series.py` refuses a patch lane without it,
  because patches cannot measure the sample depth.
- Writes: `tomograms/<TS>/` (canonical Zarr in ZYX order, and an IMOD `.rec` in
  YZX section order for 3dmod; the `tilt` log); QC with central slices in
  `qc/<TS>/reconstruction/`.

### 12. `segment-tomogram` — only with `[segment]`

- Keys: `[segment] model`* → `--membrain-model`; `executable` →
  `--membrain-executable`; `sliding_window_size` →
  `--membrain-sliding-window-size`; `test_time_augmentation` →
  `--membrain-test-time-augmentation` or its `--no-` form; `[lane] variant` →
  `--variant`.
- Writes: `segmentations/<TS>/`; QC in `qc/<TS>/segmentation/`.
- Preflight: `$MEMBRAIN_EXE` and `$MEMBRAIN_CKPT` exist, and the GPU has
  compute capability 7.5 or newer.

### 13. `prepare-template`

- Keys: `[template]` exactly one of `emdb_id` → `--emdb-id` and `source_map` →
  `--source-map`; `low_pass_angstrom`* → `--template-low-pass-angstrom`;
  `particle_diameter_angstrom` (else `[pick] particle_diameter_angstrom`) →
  `--template-particle-diameter-angstrom`; `mask_radius_px`, `mask_sigma_px`,
  `box_size_px` → `--template-mask-radius-px`, `--template-mask-sigma-px`,
  `--template-box-size-px`; `[executables] pytom_create_template`*,
  `pytom_create_mask`* → the `--pytom-create-*-executable` flags; `[lane]
  variant` → `--variant`.
- Mask and box follow from the particle diameter at its widest: the mask radius
  is half of it plus 3 px, with a 1 px soft edge, and the box is twice the
  diameter.
- Writes: `templates/<TS>/`; QC in `qc/<TS>/template/`. The fetched map is
  cached in `emdb-cache/emd_<id>.map`. Without it in the cache, the job node needs
  outbound access to download it, and the preflight warns.
- QC warns when the template's contrast does not look as expected after the
  inversion, when the mask is much wider or narrower than the density, when the
  mask had to be derived from the map, and when the density is anisotropic but
  the mask is a sphere.

### 14. `pick-particles` — template matching with pytom

- Keys, all in `[pick]` unless stated:
  - `particle_diameter_angstrom`* → `--pytom-particle-diameter`;
  - `extract_particle_diameter_angstrom` → `--pytom-extract-particle-diameter`;
  - `z_axis_symmetry` → `--pytom-z-axis-symmetry`;
  - `number_of_particles` → `--pytom-number-of-particles`. Left out, it is
    twice what pytom recommends;
  - `number_of_false_positives` → `--pytom-number-of-false-positives`;
  - `cut_off` → `--pytom-cut-off`. 0 keeps the best `number_of_particles` peaks
    with no score floor. Setting both `cut_off` and `number_of_false_positives`
    is refused;
  - `low_pass_angstrom` → `--pytom-low-pass`;
  - `tophat_filter`, `random_phase_correction`, `per_tilt_weighting`,
    `spectral_whitening` → the matching `--pytom-*` flags;
  - `ctf_weighting` → `--pytom-ctf-weighting` or its `--no-` form;
  - `gpu_ids`* → `--pytom-gpu-ids`;
  - `volume_split` → `--pytom-volume-split` (e.g. `"2,2,1"`);
  - `use_segmentation_mask` → `--use-segmentation-mask` or its `--no-` form.
    It needs `[segment]`;
  - `segmentation_mask_dilation_px` → `--segmentation-mask-dilation-px`;
  - `[executables] pytom_match_template`*, `pytom_extract_candidates`* → the
    executable flags;
  - `[lane] variant` → `--variant`.
- `--device cuda` is always added.
- Writes: `picks/<TS>/`; QC in `qc/<TS>/picking/`.
- QC records how many picks there are, their scores against the noise of the
  search (where one false positive is expected, how many picks sit above that),
  and whether the picks are confined in z or spread through the volume. A
  doubtful significance or a doubtful z spread gives a warning. The z spread is
  read against a uniform fill of the depth pytom searched, less one particle
  radius at each face, where its extraction drops peaks, and counts as
  confined three of that fill's own standard errors below it. A spread wider
  than that is named as such: more picks near the top and bottom of the volume.

### 15. `visualize-picks`

- Keys: `[lane] variant` → `--variant`.
- Writes `<TS>_open_in_chimerax.cxc`, `<TS>.mrc`, and `<TS>_picks.mod`: picks
  over the tomogram for ChimeraX and 3dmod. It registers nothing, so it can be
  rerun at any time without changing what later steps see. It still refuses to
  replace its own files without `--overwrite`.

### 16. `extract-particles`

- Keys: `[extract] box_size_px`* → `--box-size-px` (about twice the particle
  diameter in tomogram pixels, even); `pad_edges` → `--pad-edges`;
  `[lane] variant` → `--variant`.
- Writes: `particles/<TS>/`; QC in `qc/<TS>/particle_extraction/`.
- QC: more than half the particles dropped at the tomogram edge → warning; none
  survived → failure.

### After the lane: averaging

[`scripts/average_lane.sbatch`](../../scripts/average_lane.sbatch) runs after
the pick review, submitted as the recipe `average` (`submit_recipe.py average
<config> --overwrite --set THRESHOLDS=... --set MASK=...`). It re-extracts every
series at its threshold, then refines them together with independent halves. The
new extraction replaces the lane's, under the lane's label unless `--set VARIANT`
names another, so it needs `--overwrite`; since 28 October 2026 it no longer
re-extracts with `--overwrite` by default.

- Writes: `averages/<scope>/`; QC in `qc/<scope>/subtomogram_averaging/`.
- QC warns when:
  - the smaller half-set has fewer than 10 particles;
  - the halves never cross FSC 0.143, so no resolution is measured;
  - refinement did not improve the mean correlation;
  - some particles end with a negative correlation;
  - fewer than 10% of particles correlate above 0.3 with the starting
    reference.

## Rules every step follows

- **A step refuses to overwrite** its own outputs unless given `--overwrite`.
  `submit_lanes.py --overwrite` gives it to every step of the submitted stages,
  and its dry run lists the recorded runs it may replace. `submit_recipe.py
  --overwrite` gives it to the recipe's job, and its dry run lists what is
  registered under the recipe's labels. Replacing outputs is a decision, not a
  default.
- **Every input is resolved from the registry**, so a step can be run alone once
  the steps before it have run. Typing fewer steps is how work is skipped. Nothing
  inspects state to skip work by itself.
- **Two jobs must not work on the same series at the same time.** They write the
  same files, and the later registry commit is refused. Jobs on different series
  of one project may overlap: each commits its own entries under a lock.
