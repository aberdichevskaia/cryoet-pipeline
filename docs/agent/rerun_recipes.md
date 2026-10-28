# Safe rerun recipes

These are the ways of rerunning part of a lane that are known to be safe. The
assistant picks one of them and fills in its parameters. It does not invent new
ones, and it writes no new batch scripts. Every submission is shown as a dry run
first and needs an explicit yes. A lane is submitted with `scripts/submit_lanes.py`,
and a recipe script (section C) with `scripts/submit_recipe.py`, which makes the
same checks and writes the same journal.

Step names, keys, and output folders are in [pipeline_map.md](pipeline_map.md).

## Before any rerun

1. **Check that nothing else is working on the same series.** Two jobs on one
   series write the same files, and the later registry commit is refused. Since
   28 October 2026 the dry run checks this. It asks `squeue -u $USER` and
   refuses series that a queued or running job of the project works on. A job's
   series come from the journal, or else from its log, once it runs. A job named
   `<project>-<stage>` that neither accounts for counts as being on every series.
   A recipe submitted through `submit_recipe.py` is journaled as a lane is, so
   its job is seen while it waits too. A batch script given to `sbatch` by hand
   is seen only once its log names the project.
2. **Know what will be replaced.**
   - `python3 scripts/show_project.py <config>` shows where every series of the
     lane stands.
   - `python3 scripts/show_failure.py <config>` shows why each failed step
     failed, then the steps that failed before and passed when run again, and
     the results that failed QC.
   - `python3 scripts/show_series.py --project P --series TS` shows what exists.
   - `python3 scripts/run_series.py --project P --config C --series TS --from A
     --until B` prints the exact commands those steps would run.
   - A step refuses to replace its own outputs without `--overwrite`.
     `submit_lanes.py --overwrite` gives that flag to every step of every
     submitted stage, for that submission only, and `submit_recipe.py
     --overwrite` gives it to the recipe's job. `OVERWRITE=1` set in the shell
     is refused by both.
3. **Dry run.** `python3 scripts/submit_lanes.py <config> [--from-stage S]
   [--overwrite] --dry-run` prints the sbatch commands and any problems, and
   submits nothing. With `--overwrite` it also lists the recorded runs whose
   outputs may be replaced. The dry run goes into the project's `journal.jsonl`.
   For a recipe: `python3 scripts/submit_recipe.py <recipe> <config> [--series
   TS] [--set NAME=VALUE] [--overwrite] --dry-run`. With `--overwrite` it lists
   what is registered under the recipe's labels and what it replaces outside
   them; without it, labels that already hold outputs are a problem.
4. **Say why.** The reason, the date, and who asked go at the top of the config
   copy, and into the journal as `--reason`.
5. **Submit what was shown.** After a yes, run the same command without
   `--dry-run` and with `--reason`. From the assistant, `submit_lanes.py` refuses
   anything else. The config's latest journal entry has to be the same person's
   dry run of exactly this plan: no problems, at most 30 minutes old. One dry run
   stands for one submission. After a refusal, run the dry run again, show it,
   and ask again. `submit_recipe.py` does the same for a recipe on its series;
   its plan adds the recipe script, its `--set` parameters, the files they name,
   and the sbatch options.

## Where config copies go

A one-off rerun uses a copy of the lane config, never an edit of
`configs/lanes/<lane>.toml`. The lane file is the record of how the lane's
results were made.

Put the copy where both the login node and the compute nodes can see it: the
project space or the site's scratch. Never use `/tmp`, which compute nodes do not
share (and at MDC holds only 100 MB). Paths inside the copy stay relative to the
repository root, because `lane.sbatch` runs from there. Keep `[run] project`
unchanged unless the recipe says otherwise.

A change that should become permanent goes into the lane file itself, with a
comment saying what changed, when, and why.

## A. Resume after a failure whose cause was outside the config

**When:** a step failed for a reason the config does not set — memory, time
limit, a node, the environment — and that cause has been fixed. The fix might be
resources in `[[run.stage]]`, an environment variable, or a rebuilt program.

When a stage job fails at series k:

- the series before k finished the stage;
- k has partial outputs from the failed step;
- the series after k did not run;
- the later stages were cancelled.

**Steps:**

1. Copy the config. Set `[run] series` to the series that did not finish: k and
   the ones after it.
2. Dry run: `python3 scripts/submit_lanes.py <copy> --from-stage <stage>
   --overwrite --dry-run`. `--overwrite` is needed because k has partial outputs.
   The only outputs it replaces are k's own from this stage, and they are
   recomputed the same way. The dry run lists the ones it has a record of.
3. Show the dry run, get a yes, then submit the same command without
   `--dry-run` and with `--reason`.

If the stage failed on its first series, before any series finished it, resubmit
the lane file itself with `--from-stage <stage>`. Add `--overwrite` only if the
failed step wrote part of its outputs; a rerun without it then stops with
"already exists".

**Not for a crash that reproduces.** Rerunning unchanged is right only when the
cause was transient or has been fixed. Otherwise use recipe B.

## B. One series, one changed parameter, same project

**When:** one series needs a different value from the rest of the lane. Example:
RyR1 series 161 needed `tracking_binning = 3` where the lane has 2, because
`beadtrack` segfaults at 2.

1. Copy the config. Set `[run] series = ["<TS>"]`. Change the one key, with a
   comment.
2. Make the first stage start at the first step that reads the key. The map says
   which step reads which key. The stages after it stay consecutive.
3. Run `... --overwrite --dry-run`, show it, get a yes, submit.
4. Record the exception in the lane file as a comment. Split the lane's stages so
   that a later `--from-stage` does not redo that series' work with the lane's
   value. `ryr1-bin8.toml` does this for 161: its CPU stage starts at
   `build-ctf-stack`, after alignment.

Steps before `build-final-aligned-stack` are shared by every lane of the project.
Changing them for a series changes what every lane sees for that series, so say
so when proposing it.

## C. An experiment as a variant, with nothing overwritten

**When:** testing whether a parameter is the cause, before changing the lane.

The steps that take `--variant` write labelled outputs next to the lane's, which
stay untouched. These are `estimate-ctf` and every step from
`build-final-aligned-stack` on.

The model is [`scripts/ctf_window_test.sbatch`](../../scripts/ctf_window_test.sbatch):
one series, one step, four variants under their own labels, then
[`scripts/ctf_fit_table.py`](../../scripts/ctf_fit_table.py) to compare them.
It is submitted as the recipe `ctf-window`:

```bash
python3 scripts/submit_recipe.py ctf-window configs/lanes/ryr1-bin8.toml --series 039 --dry-run
```

A recipe runs where the lane runs the step it repeats: the partition, QOS and GPU
of that `[[run.stage]]`. The CPUs, memory and time are the recipe's own, from its
`#SBATCH` lines; `--time` asks for longer. Its job is named
`<project>-<recipe>`.

| variant | search window per view | CTFFIND5 fits the sample tilt |
|---|---|---|
| `ctfw1` | ±1000 Å, as in the lane | no |
| `ctfw5` | ±5000 Å | no |
| `ctfw10` | ±10000 Å | no |
| `ctfw5-tilt` | ±5000 Å | yes |

Two things in it need attention:

- **With `--overwrite` it replaces a shared output.** It first rebuilds the
  series' unlabelled CTF fit stack, which every lane of the project uses, and
  then fits again under labels that already hold fits. Stacks built before the
  `newstack -size` fix (16 October 2026) are cropped, so they need the rebuild.
  `--set KEEP_STACK=1` keeps the stack while still replacing the fits. Without
  `--overwrite` it fits the stack the series has. Before 28 October 2026 it
  rebuilt the stack by default.
- **It needs two environment variables.** `IMOD_DIR` must name the site's IMOD.
  `CTFFIND5` must name the CTFFIND 5 executable wherever the lane refers to it as
  `$CTFFIND5`. The dry run checks both.

`python3 scripts/submit_recipe.py --list` names every recipe script, what it
runs, and the `--set` parameters it takes:

| recipe | script | series | what it writes |
|---|---|---|---|
| `ctf-window` | `ctf_window_test.sbatch` | one | CTF fits under `ctfw1`, `ctfw5`, `ctfw10`, `ctfw5-tilt` |
| `pick-ablation` | `pick_ablation.sbatch` | one | picks under `bin8-abl-*` |
| `null-decoys` | `null_decoys.sbatch` | the lane's | a phase-randomised copy, picks and particles under `<lane>-null` |
| `repick` | `repick.sbatch` | the lane's | picks and particles under `--set LABEL=...` |
| `average` | `average_lane.sbatch` | the lane's | particles and an average under the lane's label, so it needs `--overwrite` |

The template-matching ablation,
[`scripts/pick_ablation.sbatch`](../../scripts/pick_ablation.sbatch) with
[`scripts/score_band_readout.py`](../../scripts/score_band_readout.py), follows
the same pattern. It was written for the ribo lanes as they were before
15 October 2026, and it edits flags that no lane renders today, so its dry run
refuses every lane. It is kept as the record of how those settings were chosen.

Once a recipe is confirmed on two or three datasets, it becomes a default in the
code or in the lane template and stops being an experiment.

## D. A new lane on the same project

**When:** a different binning, template, or picking set-up for the whole
dataset, while keeping the existing lane for comparison.

1. Copy the lane file within `configs/lanes/`, with a new `[lane] name` and a new
   `variant`. Keep the same `[run] project`.
2. Start its first stage at `build-final-aligned-stack`. The shared steps before
   it have already run. Rerunning them would be refused without `--overwrite`,
   and with it would replace what the first lane uses.
3. Dry run, get a yes, submit. No `--overwrite` is needed: the labels differ, so
   nothing is replaced.

## E. Picks only

**When:** a picking parameter changes — the low-pass, the corrections,
`volume_split`.

- **To replace the lane's picks:** use a config copy (recipe A or B) whose stage
  starts at `pick-particles`, with `--overwrite`. Extraction downstream must
  rerun too.
- **To keep the old picks for comparison:** this is still open. Nobody has
  checked whether `pick-particles` under a new variant finds the tomogram and
  template under the lane's label. Read how it resolves its inputs before relying
  on it, or use recipe D. The recipe `repick` comes close: it names the old
  picks' template, mask and count instead of looking them up, takes the other
  picking settings from the config, writes under `--set LABEL=...`, and compares
  the two lists. It was written to repeat a search unchanged (`scripts/repick.py`
  says why), so read its header before using it for a changed parameter.

## Never

- Run two jobs on the same series at once.
- Edit the lane file to run a one-off.
- Give `--overwrite` to a whole multi-series stage to fix one series.
- Get round a refused submission by unsetting what marks the assistant, by
  calling `sbatch` or `lane.sbatch` directly, or by editing the journal.
- Give a recipe script to `sbatch` directly. It goes through
  `submit_recipe.py`, and a script that `submit_recipe.py --list` does not name
  is not the assistant's to run.
- Rerun a crash that reproduces without changing anything.
- Run `cryoet init` on an existing project. It starts the registry empty, so new
  series for a frames lane go into a new project.
- Copy the config into `/tmp`.
- Write a new batch script for an experiment these recipes do not cover. Stop and
  ask instead.
