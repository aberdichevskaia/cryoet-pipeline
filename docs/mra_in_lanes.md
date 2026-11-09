# MRA in lanes

## Purpose

Until now an MRA run was one `sbatch` of `scripts/mra_port_run.sbatch` with about
ten variables in the environment, typed by hand: no dry run, no journal, no check
of the queue, and nothing the assistant could submit. This document makes the MRA a
stage of a lane, submitted by `scripts/submit_lanes.py` like every other stage,
with defaults that follow the lab's rules, so that a lane can classify its
particles right after picking or after the picks have been looked at. Agreed with
Anna on 2026-11-09 (the decisions at the end).

## What a lane says

**The stage turns it on.** A lane runs an MRA when one of its `[[run.stage]]`
tables runs the step `mra`:

```toml
[[run.stage]]
name = "mra"
from = "mra"
until = "mra"
time = "1-00:00:00"
sbatch = ["--partition=gpu", "--gres=gpu:a40:4", "--cpus-per-task=8", "--mem=64G"]
```

`mra` is the last step of every lane's chain, after `extract-particles`. It is not
a step of `scripts/run_series.py`, which renders one series at a time: the MRA
classifies the particles of all the lane's series together, in one job of
`scripts/mra_port_run.sbatch`. Its stage therefore runs nothing else, and like any
stage it must follow the one before it without a gap: the stage that ends at
`extract-particles`. A lane without such a stage runs no MRA, as every lane written
before this did.

**`[mra]` holds only what differs from the defaults.** Left out, the MRA follows
the rules below. A key set here replaces one default, and the dry run says which
values are the lane's and which the rules'. `[mra]` without a stage that runs `mra`
is refused, like any key nothing reads.

| Key | Default | What it is |
|---|---|---|
| `name` | `<[lane] variant>-<one letter per seed>`, e.g. `bin8-ssn`; for a class run `<from_run>-c<classes>-<letters>` | the run, `<project>/mra/<name>` |
| `seeds` | `["subset", "subset", "noise"]` | one kind per channel; or `["map"]` with `reference` |
| `subset_particles` | 300 | the particles a `subset` or `blob` seed averages |
| `reference` | none | the map a `map` seed refines from, in the particles' box |
| `schedule` | `"coarse_to_fine"` | or `"fixed"` |
| `first_lowpass_nyquist`, `last_lowpass_nyquist` | 2.0 and 1.1 | the first and last round's low-pass, in multiples of the particles' Nyquist (2 × their pixel) |
| `first_lowpass_angstrom`, `last_lowpass_angstrom` | none | the same in Å, instead of the multiples |
| `lowpass_steps` | `"angstrom"` | rounds in equal steps of resolution, or `"fourier"` |
| `diameter_angstrom` | `[pick] particle_diameter_angstrom` | the particle where it is widest: each round's angular step and the mask |
| `global_rounds` | 2 (none for `fixed`) | the first rounds, which search every orientation |
| `global_step_deg` | none | a coarser grid for the global rounds, refined down to their step |
| `global_full_box` | false | search the global rounds at the full box |
| `cone_steps` | 3 | each cone's width in angular steps; one number, or one per round |
| `antialias` | false | low-pass every map to a global round's smaller box before shrinking it (not Dynamo's) |
| `shift_limit_px` | 3 | how far a particle may move from its box centre, in px of its box |
| `mask_radius_px`, `mask_edge_px` | the particle's radius + 3 px, 2 px | the sphere that aligns, classifies and averages |
| `mask_file` | none | a mask of any shape in the particles' box, instead of the sphere |
| `decoy_label` | none | the label of decoy stacks of the same series, classified after the particles |
| `from_run` | none | an earlier run of the project, whose classes are run again (below) |
| `from_classes` | none | its classes, by number, as its report shows them |
| `from_iteration` | its last | the iteration whose classes they are; it must have finished |
| `start_poses` | `"run"` | the poses that iteration gave the particles, or `"picked"`: their picks' |

The defaults are Anna's (2026-11-09): seeds subset, subset, noise; five rounds of
one iteration from 2 × to 1.1 × Nyquist, the first two over every orientation and
the last three in cones three steps wide; every round's angular step
atan(resolution / diameter), the rule since 2026-10-24. The low-pass is relative to
the particles' Nyquist, so it is known only once the particles are: the job reads
their pixel size from the stacks, and the dry run shows the values in Å when the
stacks exist already. At RyR1 bin8 (52 px at 13.44 Å) the defaults give the rounds
of `mra_pad_ssn_g2atan`, the run that agreed with Dynamo: low-pass 13, 15, 17, 20
and 24 Fourier px, the global rounds at 30 and 34 px.

**The particles are the picks.** The MRA takes every particle the lane extracted,
so how many is decided at picking: twice what pytom recommends (the rule since
2026-10-19), or, new here, twice the number the person expects in a tomogram,
`[pick] expected_particles_per_tomogram`. Either way the doubtful half is there for
the noise channel to take. Twice the expected number is taken without a score floor
unless `cut_off` or `number_of_false_positives` sets one, as twice the
recommendation is. `number_of_particles` stays what it was, an exact count, and is
refused beside `expected_particles_per_tomogram`.

## When it runs

| The person wants | Submission |
|---|---|
| the MRA right after picking | `submit_lanes.py <lane>`: the MRA job waits for the extraction stage (afterok) |
| to look at the picks first | `submit_lanes.py <lane> --until-stage <the extraction stage>`, then later `submit_lanes.py <lane> --from-stage mra` |
| another MRA of the same particles | a copy of the lane with another `[mra]` (and `name`), `--from-stage mra` |

The lane says what MRA to run; when to run it is the submission's choice, and the
journal records it with the rest of the plan. `--until-stage` is new, the mirror of
`--from-stage`.

The dry run of a lane with an MRA stage shows the run's folder, its particles and
every setting with its source, and refuses:

- a stage that runs `mra` together with other steps;
- an MRA submitted without the extraction before it, when a series has no particle
  stack under the lane's label (or the decoys' label);
- an existing run folder of the same name, when the extraction runs again in the
  same submission: the run could not carry on with new particles;
- an existing run of the same name with other settings, when the MRA runs alone; an
  existing run with the same settings is carried on, as `mra_port_run.sbatch` always
  carries a run on.

**An MRA job reads, it does not write, the series.** The check of the queue counts
it as busy on its series for anything that would run steps on them, since a new
extraction would change the particles under it. It does not count it against
another MRA of the same series under another name, so that two designs can run side
by side; one of the same name is refused, as the run's lock would refuse it.

## Again on a class

After a run, a class may look worth more: "class 2 looks good; run only its
particles again, perhaps with other settings". That is a config copy whose `[mra]`
names the run and its classes, and anything else to change, submitted with
`--from-stage mra`:

```toml
[mra]
from_run = "bin8-ssn"
from_classes = [2]
global_rounds = 0          # for instance: only cones, from the poses it found
```

The particles are those the run's iteration (its last unless `from_iteration`
names another) put in the classes, read from the registry as the run read them,
decoys among them as it read them; each starts at the pose that iteration gave it
in its class, or at its pick with `start_poses = "picked"`. Classes are already
selections in a run's records ([run_records.md](run_records.md)), so the new run's
setup names the iteration's record as its input, as a consensus half names its
split ([halves_from_runs.md](halves_from_runs.md)); the run, its iteration, the
classes and where the poses come from are among its variables
(`mra_setup.plan_from_classes`). Every other setting is the copy's `[mra]` over the
defaults, as for any run: the label, the series and the decoys are the earlier
run's, and refused if the lane's differ.

The dry run names the run, the iteration and the particles in each of its classes,
and refuses an iteration that has not finished, a class with no particle, a lane of
other particles, and classes whose particles were extracted again since the run
read them. A class smaller than two subsets of 300 particles needs
`subset_particles` or other seeds; the setup says so.

## The job

`submit_lanes.py` submits the stage as `scripts/mra_port_run.sbatch` with `CONFIG`,
`SOURCE` (the lane's project), `SERIES`, `NAME` and `CONFIG_SHA256`, and passes none
of the MRA variables a shell may hold (`SEEDS`, `EXPORT`, `PROJECT`, ...): the
dry run lists any it leaves out. The job reads `[mra]` through
`cryoet_pipeline.mra_lane`, the function the dry run uses, and refuses a lane file
that changed since the submission, so that what runs is what was shown. The run is
`<project>/mra/<name>`, made by `mra_setup.plan_from_registry` as before, with the
lane's prepared template drawn in its report.

**Provenance.** The setup's variables, and so the run's fingerprint, are the numbers
in Å: a run requested by a lane and one requested by hand with the same numbers are
the same plan. How the numbers were asked for goes beside them in `plan.json`
(`setup.request`): the lane file and its sha256, `[mra]` as written, and every value
with its source (the lane, or the rule it was derived by). The journal holds the
lane's text, the submission's options and who submitted it, as for every stage.

**For the database and a GUI.** `cryoet_pipeline.mra_lane.MraSettings` is the step's
parameter schema in pydantic, as [database_design.md](database_design.md) asks of
every step: typed, with defaults and limits, content parameters apart from where
the job runs (the stage). A form can be drawn from it, and `create_run` can take it
when the controller exists; the TOML is its file form.

## The assistant

When a lane is filled in up to picking, the `lane-config` skill asks: "MRA right
after picking, or look at the picks first?" Either way it writes the MRA stage, from
the site's GPU stage, and `[mra]` only for what the person changes. "Right after"
submits the whole lane; "first look" submits with `--until-stage`, and the MRA later
with `--from-stage mra`, each after its own dry run and yes. Among the questions it
also offers `expected_particles_per_tomogram`, with twice pytom's recommendation as
the default. When the person says a class looks good and wants its particles run
again, it writes a config copy with `from_run`, `from_classes` and the changes they
ask for, and submits it with `--from-stage mra` in the same way.

## Not in this step

Consensus halves from a lane's run, a score cut of a run as the particles of the
next (classes are in), decoy picking as a lane step, and the database itself.

## Decided on 2026-11-09

1. The MRA is set in the lane's TOML, not submitted as a recipe: a typed schema the
   database and a GUI can take, one file form the person and the assistant already
   write, and a recipe's placement, queue check and outputs fit the MRA badly.
2. A stage that runs `mra` turns it on; `[mra]` holds only changes to the defaults.
   New lanes get the stage from `example.toml` and from the assistant; the lane files
   there are unchanged.
3. The defaults above (Anna).
4. The particles: twice the expected number per tomogram, or twice pytom's
   recommendation (Anna).
5. The assistant asks whether to run the MRA right after picking or after the picks
   are looked at (Anna).
6. `antialias` is off by default. With the low-pass relative to Nyquist the global
   rounds shrink the box about 1.5-1.8 times at any binning, as `mra_pad_ssn_g2atan`
   did at bin8 and agreed with Dynamo. The bin4 halves whose global round failed
   (2026-11-07) started at bin8's 53.76 Å, shrinking 102 px to 30.
7. When the MRA runs is an option of the submission (`--until-stage`), not a key of
   the lane.
8. A run on some classes of an earlier run, with any settings, is an MRA like any
   other: `[mra] from_run` and `from_classes` in a config copy (Anna, 2026-11-10).

## Status

2026-11-10: built and tested locally, not yet run on a cluster.
