---
name: lane-config
description: Write or change a lane config (a TOML file in the format of configs/lanes/example.toml), check it, and submit it after a dry run and an explicit yes. Use when the person describes data to process, wants a new lane, a changed parameter, a rerun with other settings, or one of the recipes.
---

# Lane config: write, check, submit

Paths are from the repository root. The keys, and how to choose them, are in
`configs/lanes/example.toml` and `docs/agent/pipeline_map.md`. The safe ways to
rerun are in `docs/agent/rerun_recipes.md`.

## 1. Start from the closest config

| The person wants | Start from | Written to |
|---|---|---|
| a new dataset | `configs/lanes/example.toml`, or a lane in `configs/lanes/` with the same kind of data | `configs/lanes/<name>.toml`; the client asks the person to approve the new file |
| a one-off rerun or a test | a copy of the lane's own file | the project space or the site's scratch, never `/tmp` |
| a permanent change to a lane | nothing written: propose the edit | the person makes it, with a comment saying what changed, when and why |

Never edit an existing file in `configs/lanes/`. A lane file is the record of
how the lane's results were made. Paths inside a config copy stay relative to
the repository root.

## 2. Fill it in

- **The data.** For `[lane] input` and `[data]`, read the folder as
  `docs/agent/data_folders.md` says: the main naming schemes first, and the
  person confirms the reading before anything runs.
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
- **The template.** Ask the person for the EMDB entry or the map. Proposing
  one from EMDB is a later skill.
- **Values you cannot fill in.** Ask. Do not leave a placeholder from
  `example.toml` in a config you submit.

## 3. Check it

```bash
python3 scripts/run_series.py --project <project> --config <config> --series <TS> --list-steps
```

This lists the steps the config's choices give. It also refuses any key the
pipeline does not read, names it, and suggests what it was probably meant to
be. Fix every key it names.

## 4. Dry run, yes, submit

A lane:

```bash
python3 scripts/submit_lanes.py <config> [--from-stage <stage>] [--overwrite] --dry-run
```

A recipe, which is one of the experiments written in advance:

```bash
python3 scripts/submit_recipe.py --list
python3 scripts/submit_recipe.py <recipe> <config> [--series <TS>] [--set NAME=VALUE] [--overwrite] --dry-run
```

Then:

1. Show the person what the dry run printed:
   - the jobs and their stages;
   - every "would refuse" line, and what it means;
   - with `--overwrite`, the outputs it may replace.
2. If it refused anything, fix the cause and run the dry run again.
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
