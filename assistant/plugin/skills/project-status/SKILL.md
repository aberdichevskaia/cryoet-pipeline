---
name: project-status
description: Say where a lane's project stands and why something in it failed. Use when the person asks how a project, a lane, a series or a job is doing, whether it finished, what a QC warning means, or why a step or a job failed.
---

# Project status and diagnosis

Paths are from the repository root. The knowledge behind this skill is in
`docs/agent/`: `pipeline_map.md` for the steps, their outputs, QC and logs,
`playbook.md` for problems met before, and `rerun_recipes.md` for the fixes.

## 1. Find the lane config

Use the config the person names. Otherwise use the file in `configs/lanes/`
whose `[run] project` is the project they mean. If two lanes share the project,
ask which one, or show both. A config copy made for a rerun is the one that
rerun was submitted with: the project's `journal.jsonl` names it.

## 2. Where it stands

```bash
python3 scripts/show_project.py <config>
```

Add `--series <TS>` (repeatable) to look at a few series, or `--json` when you
need the fields. It shows, per series and step:

- the last run and its QC verdict;
- the warnings;
- the lane's jobs: finished, failed, running, or still waiting.

Report what it shows, in a few lines: which series are done, which failed and
at which step, which are running or waiting. Do not paste the whole table unless
asked.

## 3. Why a step failed

```bash
python3 scripts/show_failure.py <config> [--series <TS>] [--step <step>]
```

For each failed step it gives the step's lines from its Slurm log: the command,
the error lines and the tail. It also gives the logs the step's programs wrote
while it ran. Read these before anything else. Open a whole log only around the
lines an excerpt points at.

After the failures it lists what is not failing now but did fail: steps that
failed in a job and passed when run again, with those jobs, and steps that ran
but whose result failed QC, with the first warning. "0 failed steps" means
nothing is broken now, not that nothing ever failed: say which of these it
lists. A QC failure is a verdict on a result, not a crash; its details are in
`show_project.py`.

Two more sources, where needed:

- `python3 scripts/show_series.py --project <project> --series <TS>`: what one
  series has, and its QC documents;
- `sacct -j <job id> -o JobID,State,ExitCode,Elapsed,MaxRSS,ReqMem,Timelimit`,
  and `squeue -u $USER`: what Slurm knows, such as a time limit, memory, or a
  node failure.

## 4. Name the cause

1. Look the symptom up in `docs/agent/playbook.md`, under its step.
2. Do what the entry's "How to confirm" says before naming its cause, and say
   which source the entry has.
3. If no entry fits, describe what the evidence shows. Say what is missing to
   go further, and stop there. That is a complete answer.

Every conclusion cites a file and line, a QC field, or `sacct`. Two readings
that both fit are both given, with what would tell them apart.

## 5. Propose what to do

Propose only a recipe from `docs/agent/rerun_recipes.md`, with its parameters
filled in, and say why that recipe fits. If none fits, say so and ask; do not
invent one. To submit, follow the `lane-config` skill: dry run, show, explicit
yes, then the same command with `--reason`.

Hand over to a person, with the evidence gathered, in these cases:

- a playbook entry says to stop;
- the failure reproduces with nothing changed;
- the cause looks like a bug in the code.
