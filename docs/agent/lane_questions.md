# Questions for a new lane

A person describes data and a goal: "the frames are here, I want to get to
particle picking". Most of the lane config is known before anyone is asked:
the data gives part of it, the site part, and the lab's rules the rest. This
file says, for every key the pipeline reads, where its value comes from. It
lists the few questions only the person can answer. The `lane-config` skill
follows it.

## The order

1. **The data.** Run `scripts/show_folder.py` on the folder, as
   [data_folders.md](data_folders.md) says. It gives `[lane] input`, `[data]`
   and `[run] series`, and the values of the rows marked *data* below: the
   pixel size, the frames per tilt, the tilts and angles. From the mdocs it
   also reads the voltage, the tilt axis and the target defocus.
2. **The site.** Copy the rows marked *site* from the closest lane that ran on
   the same cluster: `ryr1-bin8.toml` or `ribo-mdc.toml` at MDC, `gag-bin8.toml`
   or `ribo-bin8.toml` in Munich.
3. **The cards.** An instrument card and a sample card, in the form of
   [templates/](templates/), answer the rows marked *card*. Ask the person only
   for what the cards leave out.
4. **The rules.** Fill in the rows marked *rule* without asking: the lab's
   rules, the code's defaults, or the values of `configs/lanes/example.toml`.
   Write the rule next to the value as its source.
5. **The questions.** Show the person what was read, then ask what is left in
   one message, each question with its default.
6. **The combination.** Say which combination of input, alignment and
   membranes the lane is; EER movies are an input of their own, "EER frames".
   Say whether it has run on real data, from
   [pipeline_map.md](pipeline_map.md), "Which combinations have run on real
   data". The dry run warns about it too.
7. **The check.** Run `run_series.py --list-steps`, then the dry run, as the
   `lane-config` skill says.

## What the goal sets

Each `[[run.stage]]` is one Slurm job, running the steps from its `from` to its
`until`. The last stage's `until` is where the lane stops:

| The person wants | The last stage ends at |
|---|---|
| tomograms | `reconstruct-tomogram` |
| membranes | `segment-tomogram`, with `[segment]` |
| particles picked | `visualize-picks`: the picks and their preview |
| particles to average | `extract-particles`, as in `example.toml` |

Whatever the goal, `[template]`, `[pick]` and `[extract]` must be filled in:
`run_series.py` requires their keys in every lane, even one that stops at the
tomogram.

## The questions

Ask only those whose answer was not read. Each answer goes into the config with
the person and the date as its source.

1. **What is to be picked, and from which map?** An EMDB entry (`EMD-…`) or a
   map file. This fills `[template] emdb_id` or `source_map`. If the person
   knows none, propose one to three with the `emdb-template` skill
   (`scripts/show_emdb.py`), and let the person choose.
2. **How wide is it at its widest, in Å or nm?** This fills
   `[pick] particle_diameter_angstrom`. The template's mask and box, the
   angular step and `[extract] box_size_px` all follow from it. That is why it
   must be the widest diameter, not a typical one. If the person does not know
   it, measure the template map: `python3 scripts/show_emdb.py measure
   EMD-<id>`, or a map file with `--level`. It proposes the diameter a mask
   around the box centre needs, since prepare-template draws the mask there.
   Give the person that number with the widest extent beside it, and let the
   person confirm. A map's box ("N A across" in `show_emdb.py entry`) is not
   the particle, and a map is measured with this tool, not with code of your
   own.
3. **Are there gold beads, and of what size?** With beads, keep alignment
   block A and set `[fiducials] fiducial_diameter_nm`. Without them (a FIB
   lamella, usually), keep block B, patch tracking, and ask 3a.
   - 3a. **How thick is the sample, roughly, in nm?** Patches cannot measure
     it, so `[reconstruct] thickness` is required. Convert it to binned pixels
     and be generous, so the tomogram holds vacuum above and below. If the
     person does not know, propose a value and say how it was chosen.
4. **Does it sit in or on a membrane? If so, how far from the membrane does it
   reach, in nm?** If yes, keep `[segment]`, set
   `[pick] use_segmentation_mask = true`, and convert that distance to
   `segmentation_mask_dilation_px` in tomogram pixels. If no, leave the mask
   off, since with it a cytosolic particle is searched for only next to
   membranes. `[segment]` can then be deleted, which skips the segmentation.
5. **Only if no card says it: are the movies gain-corrected? If not, which file
   is the gain reference, and how is it rotated and flipped?** This fills
   `[motion] gain_reference`, `gain_rotation` and `gain_flip`.
   `show_folder.py` names the files that look like a gain, by their name only.
6. **Only if no card says it: the spherical aberration, Cs, in mm.** This fills
   `[ctf] spherical_aberration_mm`.
7. **Only if `show_folder.py` says nothing gives it: the tilt-axis angle.** That
   happens without mdocs, or with mdocs whose RotationAngle the parser does not
   take as the axis. This fills `[align] tilt_axis_angle`.
8. **One series first, or all of them?** The default is one, so that its report
   can be read before the rest run. This fills `[run] series`.
9. **Where do the outputs go?** The default is a new folder,
   `outputs/<lane name>`. `init` runs only for a project that does not exist
   yet, and `submit_lanes.py` refuses series that an existing project has no
   manifests for: adding them would erase the project. This fills
   `[run] project`.

Not asked unless the person raises them: the number of particles (by the lab's
rule, twice what pytom recommends), the symmetry (`[pick] z_axis_symmetry`,
off), and the binning (by the lab's rule, a new dataset starts at bin8).

## Every key

*From* says where the value comes from: *data* (`show_folder.py`, the mdocs),
*card* (an instrument or sample card), *person* (the questions above), *site*
(the closest lane at the same site), *rule* (the lab's rules, the code's
defaults, `example.toml`). "Unset" means the key is left out of the config.
[pipeline_map.md](pipeline_map.md) says which flag each key becomes.

| Keys | From | How |
|---|---|---|
| `[lane] name` | rule | `<dataset>-<variant>`, such as `ryr1-bin8`; also the file name, `configs/lanes/<name>.toml`, unless the person names another place |
| `[lane] input` | data | `frames` or `stack`, as `show_folder.py` reads the folder |
| `[lane] variant` | rule | `bin8`: a new dataset starts at bin8. It labels every output from the tomogram on |
| `[run] project` | person | question 9 |
| `[run] series` | data, person | the ids `show_folder.py` found; question 8 picks one or all |
| `[run] stage` | rule, site | one `[[run.stage]]` table per job, as below |
| `[run.stage] name, from, until` | rule | the stages of `example.toml`. With `input = "stack"`, a CPU stage from `import-tilt-stack` replaces `motion`. Without `[segment]`, the GPU stage starts at `prepare-template`. The last `until` is the goal's step. Stages follow one another without skipping a step |
| `[run.stage] time, sbatch` | site | from the closest lane at the same site: partitions, QOS and GPUs differ between sites |
| `[data] frames, mdocs, frames_pattern` | data | the fragment `show_folder.py` prints, for a frames lane |
| `[data] mdoc, stack` | data | the fragment `show_folder.py` prints, for a stack lane |
| `[data] pixel_size_angstrom` | card | unset: the movie headers give it. Set it only to override a header known to be wrong, and name what says so |
| `[motion] backend` | rule | `motioncor3`, the only backend that reads EER |
| `[motion] storage_policy, gpu, patch_x, patch_y` | rule | the values of `example.toml` |
| `[motion] pixel_size_angstrom` | rule | unset: `example.toml` gives no reason to set it |
| `[motion] gain_reference, gain_rotation, gain_flip` | card, person | question 5; unset when the movies are gain-corrected |
| `[motion] ftbin` | data | 2 when `show_folder.py` says the frames have the K3's super-resolution size, and with `eer_sampling = 2`; otherwise unset |
| `[motion] eer_sampling, eer_fractions` | rule | EER only. Unset, eer_sampling is 1 and eer_fractions comes from each movie's dose, about 0.2 e/Å² per group ([motioncor3_backend.md](../motioncor3_backend.md), "EER movies") |
| `[binning] tomogram` | rule | 8, as a new dataset starts at bin8. The particle should still be a few tens of pixels across: its diameter / (stack pixel × 8). The worked examples have 30 and 46 px |
| `[binning] ctf_fit` | rule | small enough that the fit stack's Nyquist, 2 × stack pixel × `ctf_fit`, is finer than about 6 Å. The stack pixel is the movie pixel times `ftbin` |
| `[align] binning` | rule | the same as `[binning] tomogram` |
| `[align] min_std_ratio` | rule | the value of `example.toml` |
| `[align] tilt_axis_angle` | data, person | unset when the mdocs give the axis; question 7 when `show_folder.py` says nothing does |
| `[align] exclude_tilt_angles` | person | unset; only views the person has looked at and rejected, by nominal angle, each naming one view within 0.5 deg. Never set to make a QC pass |
| `[qc_coarse] preview_binning, residual_warning_px` | rule | the values of `example.toml` |
| `[fiducials] seed_backend, track_backend` | person | block A or block B of `example.toml`, by question 3 |
| `[fiducials] fiducial_diameter_nm` | card, person | the bead size, question 3; block A only |
| `[fiducials] tracking_binning` | rule | block A's value |
| `[fiducials] raw_pixel_spacing_angstrom` | rule | unset: read from the data and checked against the stack |
| `[fiducials] pre_binning, num_patches_x, num_patches_y, patch_overlap` | rule | block B's values |
| `[fiducials] patch_margin_px` | rule | unset: etomo's margin, 5% of the binned image per axis |
| `[fiducials] target_beads, rounds, min_tracked_fiducials, coverage_warning, coverage_failure` | rule | unset: the code's defaults |
| `[fiducials] patch_filter_sigma1, patch_filter_radius2, patch_filter_sigma2` | rule | unset: the code's defaults |
| `[fine_align] robust_fitting, auto_prune_outliers` | rule | the values of the block kept |
| `[fine_align] surfaces_to_analyze` | rule | 2 with beads, which sit on both faces; 1 with patches, which are coplanar. The config does not enforce it, so keep the block's value |
| `[fine_align] max_pruned_fraction` | rule | unset with beads; 0.08 with patches, as block B explains |
| `[fine_align] backend` | rule, person | unset: tiltalign after the `[fiducials]` steps. `aretomo` only when the person asks for AreTomo2; that lane has no `[fiducials]`, needs `[reconstruct] thickness` and a GPU stage, and has not run on real data yet (docs/aretomo_backend.md) |
| `[fine_align] aretomo_align_z_nm, aretomo_tilt_cor, aretomo_dark_tol, aretomo_gpu` | rule | unset: AreTomo2's own defaults, with the tilt offset applied (docs/aretomo_backend.md) |
| `[reconstruct] thickness` | person, rule | with beads, unset: the beads measure it. With patches, required: question 3a |
| `[ctf] voltage_kv` | data, card | the mdocs' `Voltage`, else the instrument card |
| `[ctf] spherical_aberration_mm` | card, person | question 6 |
| `[ctf] amplitude_contrast` | rule | 0.07, the value of `example.toml` |
| `[ctf] min_defocus_angstrom, max_defocus_angstrom` | data, rule | the mdocs' `TargetDefocus`, with 0.5 to 1 µm to spare on each side, in Å of underfocus. A TargetDefocus of −2.5 to −4 µm gives about 15000 to 50000 |
| `[ctf] defocus_search_window_angstrom, defocus_step_angstrom, max_excluded_views, determine_sample_tilt` | rule | unset: the code's defaults |
| `[segment] executable, model` | site | the environment variables of the closest lane at the same site |
| `[segment] sliding_window_size, test_time_augmentation` | rule | the values of `example.toml` |
| `[template] emdb_id, source_map` | person | question 1; exactly one of the two |
| `[template] particle_diameter_angstrom` | rule | unset: `run_series.py` passes `[pick] particle_diameter_angstrom` |
| `[template] low_pass_angstrom` | rule | about 2 × the tomogram pixel, as `example.toml` says |
| `[template] mask_radius_px, mask_sigma_px, box_size_px` | rule | unset: derived from the diameter, as the lab's rule says (radius half the diameter plus 3 px, a soft edge, a box twice the diameter). Set them only for a map with no single particle in it, such as a lattice |
| `[pick] particle_diameter_angstrom` | person, card | question 2; `show_emdb.py measure` proposes it from the template map, for the person to confirm |
| `[pick] gpu_ids` | site | the value of the closest lane at the same site |
| `[pick] low_pass_angstrom` | rule | about 1.3 × the tomogram's Nyquist (2 × the tomogram pixel), by the lab's rule: a Nyquist of 19.6 Å gives 25 Å |
| `[pick] number_of_particles, number_of_false_positives, cut_off` | rule | unset: twice the peaks above pytom's false-positive cut-off, by the lab's rule. A number only when the person asks for one |
| `[pick] z_axis_symmetry` | person | unset unless the person asks, and only for a template whose symmetry axis is its z axis, as `example.toml` says |
| `[pick] random_phase_correction, per_tilt_weighting, spectral_whitening, tophat_filter` | rule | the values of `example.toml` |
| `[pick] ctf_weighting, extract_particle_diameter_angstrom` | rule | unset: the code's defaults |
| `[pick] volume_split` | rule | unset, unless picking runs out of GPU memory, as `example.toml` says |
| `[pick] use_segmentation_mask, segmentation_mask_dilation_px` | person | question 4 |
| `[extract] box_size_px` | rule | required in every lane: an even number near 2 × diameter / tomogram pixel, by the lab's rule |
| `[extract] pad_edges` | rule | unset: the code's default |
| `[executables] motioncor3, pytom_create_template, pytom_create_mask, pytom_match_template, pytom_extract_candidates` | site | the environment variables of the closest lane at the same site |
| `[executables] ctffind` | site | as the closest lane at the same site names it |
| `[executables] aretomo` | site | only with `[fine_align] backend = "aretomo"`: as the closest lane at the same site names it, else AreTomo2 on PATH |

`tests/test_lane_questions.py` checks that this table names every key the
pipeline reads (`LANE_KEYS` and `STAGE_KEYS` in `scripts/run_series.py`), and
nothing else. A key added to the pipeline needs its row here in the same change.
