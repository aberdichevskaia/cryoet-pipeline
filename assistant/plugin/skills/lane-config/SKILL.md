---
name: lane-config
description: Write or change a lane config (a TOML file in the format of configs/lanes/example.toml), check it, and submit it after a dry run and an explicit yes. Use when the person describes data to process, wants a new lane, a changed parameter, a rerun with other settings, an MRA classification (also of one class of an earlier MRA again), or one of the recipes.
---

# Lane config: write, check, submit

Paths are from the repository root. The keys, and how to choose them, are in
`configs/lanes/example.toml` and `docs/agent/pipeline_map.md`. The safe ways to
rerun are in `docs/agent/rerun_recipes.md`.

## 1. A rerun: first, is a job on the series?

A rerun is anything submitted to a project that already exists: a changed
parameter, a resume, a recipe, a new lane on the same project. Follow "Before
any rerun" in `docs/agent/rerun_recipes.md`. Do its first step before you ask a
question or propose a plan: check that no job is working on the series.

```bash
squeue --me                                # your queued and running jobs
python3 scripts/show_project.py <config>   # the project's jobs and their logs
```

The `series` line at the top of a job's log names the series it works on.
`show_project.py` prints the log's path from the repository root. A job still
queued has no log yet: the `submitted` entry of `<project>/journal.jsonl` that
holds its ID names its series. Count a job of the project whose series nothing
names as working on all of them.

If a job is working on a series the request touches:

- Do not submit now. Name the job: its ID, name, state and series.
- Say why as what would happen, not only as a rule. Both jobs would write the
  same files, and the registry refuses the later of their two commits: that
  step fails with nothing registered, and has to run again once the other job
  has ended.
- Offer to prepare the config copy now. The dry run, the yes and the
  submission wait until the job has ended: the dry run refuses a busy series,
  and a submission must follow its dry run within 30 minutes.
- Cancelling the job needs its own explicit yes. Say first what it stops:
  everything the job has left, on every series, not only this one.

## 2. Start from the closest config

| The person wants | Start from | Written to |
|---|---|---|
| a new dataset | `configs/lanes/example.toml`, or a lane in `configs/lanes/` with the same kind of data | `configs/lanes/<name>.toml`; the client asks the person to approve the new file |
| a one-off rerun or a test | a copy of the lane's own file | the project space or the site's scratch, never `/tmp` |
| a permanent change to a lane | nothing written: propose the edit | the person makes it, with a comment saying what changed, when and why |

Never edit an existing file in `configs/lanes/`. A lane file is the record of
how the lane's results were made. Paths inside a config copy stay relative to
the repository root.

## 3. Fill it in

- **A new dataset.** Follow `docs/agent/lane_questions.md`. It says, for every
  key, whether the value comes from the data, a card, the site, a rule or the
  person, and which questions to ask.
  - First read what can be read: the folder (below), the closest lane at the
    same site, and any instrument or sample card.
  - Then show the person what was read, and ask what is left in one message,
    each question with its default. Do not ask for a value that was read.
  - The goal ("up to particle picking") sets the last stage's `until`.
  - Say whether the lane's combination of input, alignment and membranes has
    run on real data.
- **The data.** For `[lane] input`, `[data]` and `[run] series`, read the folder
  with `scripts/show_folder.py`, never with Python of your own. The tool's verdict
  comes from the pipeline's own functions, which `init` and `import-tilt-stack`
  run:

  ```bash
  python3 scripts/show_folder.py <folder>                        # tree, names, mdocs
  python3 scripts/show_folder.py <frames folder> --pattern '<re>' # names, no mdocs
  ```

  Without mdocs, write the pattern from the fields the tool prints, as
  `docs/agent/data_folders.md` says (the main naming schemes first). Then run the
  tool again with `--pattern`, until every series is accepted. Show the person
  the table of series, tilts, angles, frames and pixel size, and its warnings.
  Ask them to confirm that this is what was collected, and compare it with the
  tilt scheme they describe or the instrument card. Only then copy the tool's
  config fragment into the config. If the tool says there is no entry, or
  refuses, say so and stop. Do not bend the data into another input.
- **The source of every value.** Next to every value you set or change, write a
  comment saying where it came from:
  - the person (who, and when);
  - a header (the movie, the mdoc, the EMDB entry);
  - a rule (the documented default, or the comment in `example.toml`);
  - a guess, labelled as one.

  An instrument card or a sample card, in the form of
  `docs/agent/templates/instrument_card.md` and `sample_card.md`, counts as a
  source.
- **Defaults stay unless there is a reason.** The code's defaults for the mask,
  the box, the low-pass and the angular step already follow the lab's rules.
  Change one only with a reason, and write the reason.
- **The template.** Ask the person for the EMDB entry or the map. If they know
  none, the `emdb-template` skill proposes one to three for them to choose.
- **The MRA.** Once the lane is filled in up to picking, ask question 10 of
  `docs/agent/lane_questions.md`: the MRA right after picking, or the picks
  looked at first? Either way write the `mra` stage of `example.toml`, with the
  partition, GPUs and time of the site's GPU stage. Write `[mra]` only for what
  the person changes, each with its source: the defaults are the lab's rules,
  and the dry run shows them. The answer decides the submission (section 5).
  With no MRA at all, leave the stage out.
- **Values you cannot fill in.** Ask. Do not leave a placeholder from
  `example.toml` in a config you submit.

## 4. Check it

```bash
python3 scripts/run_series.py --project <project> --config <config> --series <TS> --list-steps
```

This lists the steps the config's choices give. It also refuses any key the
pipeline does not read, names it, and suggests what it was probably meant to
be. Fix every key it names.

## 5. Dry run, yes, submit

A lane:

```bash
python3 scripts/submit_lanes.py <config> [--from-stage <stage>] [--overwrite] --dry-run
```

A lane with an `mra` stage:

| The person said | Submit |
|---|---|
| the MRA right after picking | the whole lane: the MRA job waits for the extraction |
| look at the picks first | `--until-stage <the stage that ends at extract-particles>`; later, when the person asks for the MRA, `--from-stage mra` |
| another MRA of the same particles | a config copy with another `[mra] name` and the changed keys, `--from-stage mra` |
| "class x looks good: run only its particles again", with other settings or not | a config copy with `[mra] from_run` (the earlier run's name), `from_classes` (its class numbers, from its report), a new `name` if the default does not suit, and the changed keys; `--from-stage mra` |

For a class run again, name the earlier run and its iteration as the report
shows them. `from_iteration` is its last unless the person names another, and the
particles start at the poses that iteration gave them unless `start_poses =
"picked"`. The dry run says how many particles the classes hold. A class smaller
than two subsets of 300 needs `subset_particles`, or other seeds: say so when
the dry run refuses it.

A recipe, which is one of the experiments written in advance:

```bash
python3 scripts/submit_recipe.py --list
python3 scripts/submit_recipe.py <recipe> <config> [--series <TS>] [--set NAME=VALUE] [--overwrite] --dry-run
```

Then:

1. Show the person what the dry run printed:
   - the jobs and their stages;
   - for an MRA, its run folder, its particles, and the settings that are not
     the lab's rules;
   - every "would refuse" line, and what it means;
   - with `--overwrite`, the outputs it may replace.
2. If it refused anything, fix the cause and run the dry run again. If it
   refused a busy series, wait for the job, as section 1 says.
3. Ask whether to submit, naming `--overwrite` if it is there. That needs its
   own yes.
4. After an explicit yes, run the same command without `--dry-run`, within 30
   minutes. Add `--reason "<what was asked, by whom>"`.
5. Report the job IDs it printed, and the command that shows how the jobs are
   doing: `python3 scripts/show_project.py <config>`.

If anything changes after the yes, the yes no longer covers it: run the dry run
again and ask again. The config, the series, the stages, `--overwrite` and the
parameters all count as changes. The scripts refuse a submission that differs
from the last dry run. When they refuse, show the refusal and follow it.
