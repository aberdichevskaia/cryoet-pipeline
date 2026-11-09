# Cluster, environment and submission

> **Draft for review, 8 November 2026.**
>
> - Mark each entry by its ID: `E-2 ✓` if it is right, `E-3 ✗` if it is wrong
>   (say why), `E-4 ✎` if it is right but needs a change (say which).
> - Answer the questions at the end, a line each, and add what is missing.
> - Any language and any form: edit this file, comment on it, or reply by email.
> - Entries marked *for the maintainer* are about our code; skip them if you like.
>
> This box and the questions go once the review is in.

Kinds and evidence levels are explained at the top of [ctf.md](ctf.md). How a
lane runs, where its logs are and how to read results is in
[pipeline_map.md](../pipeline_map.md), "How a lane runs". Safe reruns are in
[rerun_recipes.md](../rerun_recipes.md).

- **How a lane runs.** `scripts/submit_lanes.py` submits it as Slurm jobs, one
  per stage, chained. Each job runs `scripts/lane.sbatch`: a header, a preflight,
  then the steps.
- **The job's environment.** A job takes the environment of the shell that
  submits it (`--export=ALL`) and reads no `.bashrc` of its own.
- **Submitting.** A dry run comes first, always. From the assistant, a
  submission also needs a fresh dry run of the same plan and `--reason`.
- **Site facts at MDC (max)**, as of early November 2026:
  - Partitions `normal` and `gpu`. GPUs: A40 (139), H100 (24), H200 (16) and
    B300 (16).
  - The `gpu` QOS allows 100 GPUs per user and 7 days per job, 1 day by
    default. Jobs on H100s must stay under 6 hours.
  - `/fast` is GPFS. `/tmp` holds only 100 MB.
  - Environments are micromamba's. The group's tool paths are in a file sourced
    from `.bashrc`, as the README's one-time setup describes.
- **Site facts in Munich:** GPUs can be MIG slices such as `3g.20gb` (P-1 in
  [picking.md](picking.md)).

## How a healthy run looks

- **The dry run** lists the jobs it would submit and no problems.
- **The job header** names the node, the GPU, the environment and the tools. It
  ends with `preflight   ok`.
- **The end of the log** is `finished <time>`, and `sacct -j <job>` says
  `COMPLETED`.
- **`python3 scripts/show_project.py <lane config>`** shows each step's last
  run with exit code 0 and its QC verdict.

## Where to look first

1. **`python3 scripts/show_failure.py <lane config>`**: each failed step with
   its command and error lines, from the Slurm log and from the programs' own
   logs.
2. **The Slurm log**, `slurm-<project>-<stage>-<job>.log`. A job submitted from
   another folder logs there; `sacct -j <job> -o WorkDir` says where (E-12).
3. **`sacct -j <job> -o JobID,State,Elapsed,AllocTRES,NodeList`**: how the job
   ended, how long it ran, and what it was given.
4. **The project's `journal.jsonl`**: every dry run and submission, with the
   config, the reason and the job ids.

## Already checked by the code

- **Before the queue.** The dry run checks the lane's keys, folders, tool
  variables and named programs, and the busy series.
- **Before the first step.** The job's preflight checks the tools each stage
  needs, the GPU, and that the registry can be locked from the node.
- **Each step** refuses to replace its own outputs without `--overwrite`.

## Problems

### E-1. An install dies with "No space left on device"

**Kind:** crash · **Evidence:** measured

**Error text:** `OSError: [Errno 28] No space left on device`

**Where it shows:** `pip` or `micromamba` fails while unpacking a large package
such as torch.

**How to confirm:** `df -h /tmp`. At MDC it is 100 MB.

**Causes, common to rare:**
1. The temporary directory is too small.

**What to try:** set `TMPDIR` to a directory on the large filesystem, then
install again.

**What not to do:** read it as a network failure. It was first misread as one.

**When to stop and ask someone:** when no writable large filesystem is known.

**Applies to:** every version.

**Source:** MDC install, 13 October 2026.

### E-2. The dry run refuses: the series is busy

**Kind:** refused · **Evidence:** code

**Error text:**
- `<job> works on <series>, which <lane> would run too; two jobs on one series
  write the same files, so submit after it has ended`
- `<job> is this project's, and nothing on record says which series it works
  on, so it may be any of them; submit after it has ended`

**Causes, common to rare:**
1. A job of the same project, queued or running, works on the same series. Two
   jobs on one series write the same files, and the later registry commit is
   refused.

**What to try:** wait for that job, or cancel it if it should not run. Then run
the dry run again. `squeue --me` lists your jobs.

**What not to do:** submit around the check, for example with `sbatch` by hand.

**Applies to:** since 28 October 2026.

**Source:** read in `scripts/submit_lanes.py`.

### E-3. The config has keys nothing reads

**Kind:** refused · **Evidence:** measured

**Error text:** `the config has keys nothing reads, so they would be ignored:`
followed by one line per key, `[<section>] <key>: did you mean <key>?` or
`[<section>] <key>: it belongs in [<section>]`.

**Causes, common to rare:**
1. A misspelt key, or a key in the wrong section.
2. A key from an older version of the pipeline, which the message then names
   as renamed.

**What to try:** fix the key as the message suggests, and run the dry run again.

**Applies to:** since 27 October 2026. Before that, an unknown key was silently
ignored.

**Source:** read in `scripts/run_series.py`. Until 27 October 2026 a misspelt
key reached no step: the step ran with its default and said nothing.

### E-4. The assistant's submission is refused

**Kind:** refused · **Evidence:** code

**Error text:**
- `its dry run at <time> is <n> minutes old, and a dry run stands for <m>`
- `it was submitted at <time>, after its last dry run`
- `a submission from the assistant needs --reason: what was asked, and by whom`

**What it means:** from the assistant, a submission must follow the same
person's dry run of exactly this plan, made at most 30 minutes before. One dry
run stands for one submission.

**What to try:** run the dry run again, show it, get a yes, and submit the same
command without `--dry-run` and with `--reason`.

**Applies to:** since 28 October 2026.

**Source:** read in `scripts/submit_lanes.py`.

### E-5. `OVERWRITE` set in the shell is refused

**Kind:** refused · **Evidence:** code

**Error text:** `OVERWRITE is set in the environment, and submit_lanes.py does
not pass it on: give --overwrite to replace outputs in this submission only, or
unset OVERWRITE`

**What to try:** `unset OVERWRITE`, and give `--overwrite` to the one
submission that should replace outputs.

**Applies to:** since 28 October 2026. `submit_recipe.py` refuses it the same
way.

**Source:** read in `scripts/submit_lanes.py`.

### E-6. A step refuses to replace its outputs

**Kind:** crash · **Evidence:** code

**Error text:** `<step> outputs already exist: <path>`, for example `alignment
outputs already exist: …` or `reconstruction outputs already exist: …`.

**Causes, common to rare:**
1. A job failed partway through a step and left part of its outputs (recipe A).
2. A rerun of a step that already finished, without `--overwrite`.

**What to try:** for a resume after a failure, recipe A, with `--overwrite`.
For a deliberate rerun, the dry run with `--overwrite`, which lists the recorded
runs it may replace.

**What not to do:** delete outputs by hand to get past the refusal.

**Source:** read in the backends.

### E-7. A recipe's labels already hold results

**Kind:** refused · **Evidence:** code

**Error text:** `<series> already has <kind> under <labels>, and the job would
stop at it: give --overwrite to replace it`

**What to try:** decide whether the old results may go. If they may, run the
recipe's dry run with `--overwrite` and read what it lists as replaced. If not,
run the recipe under other labels where it allows that.

**Applies to:** since 28 October 2026.

**Source:** read in `scripts/submit_recipe.py`.

### E-8. The registry cannot be locked, or a commit was refused

**Kind:** crash · **Evidence:** measured

**Error text:**
- preflight: `preflight: cannot lock the registry from this node (above)`
- `<registry> was written by another command while this one held the registry
  lock, so on this filesystem the lock does not keep out commands on other
  nodes. Nothing was registered: …`
- `a POSIX lock on <file> failed with <error> (…). … On Lustre, ENOSYS means the
  client is mounted without -o flock.`
- `<file> has been held by another command for over <n> s; a commit holds it for
  well under a second, so look for a hung cryoet process. …`
- `this command's changes no longer apply to the registry as it is now (<why>);
  nothing was registered`

**Causes, common to rare:**
1. Two jobs changed the same entries, for example two jobs on one series (E-2).
2. A filesystem whose locks do not reach across nodes.
3. A hung `cryoet` process holding the lock.

**What to try:**
- For cause 1, rerun the step after the other job has ended.
- For causes 2 and 3, stop and tell the maintainer, with the node and the
  filesystem (`findmnt -T <project>`).

**When to stop and ask someone:** for every message here except a plain
conflict with another job.

**Applies to:** since 29 October 2026 at MDC.

**Source:** MDC, 29 October 2026. Five jobs of one project started together on
five nodes, and three lost their registry records. GPFS's flock does not
exclude across nodes, while POSIX locks do, so the registry lock now uses them.
This was checked across two nodes the same day.

### E-9. A program is not found

**Kind:** crash · **Evidence:** seen

**Error text:**
- `IMOD <program> executable not found; tried <places>. Set IMOD_DIR, or pass
  --imod-dir, or --<program>-executable`
- `<program> not found; pass --<flag> or add it to PATH` (pytom-match-pick)
- `MemBrain-seg executable not found; …`

**Causes, common to rare:**
1. The tool variables (`IMOD_DIR`, `PYTOMO_MP_DIR`, `MEMBRAIN_EXE`, …) are not
   set in the shell the job was submitted from (C-4 in [ctf.md](ctf.md)).
2. The program sits in a folder the submitting user cannot read (E-10).

**What to try:** set the variables, check them with `ls -l`, and resubmit
(recipe A). `python3 scripts/build_library.py --site <site> --dry-run --config
<lane config>` shows which tool it finds where, and writes nothing.

**Source:** MDC, 5 November 2026: in a login shell every tool variable was
unset, and IMOD was not found.

### E-10. Another person's environment cannot be read

**Kind:** crash · **Evidence:** measured

**Where it shows:** a program in someone else's environment is not found, or
cannot be run, although the path is right.

**How to confirm:** `namei -l <path>` shows every folder on the way and its
permissions.

**Causes, common to rare:**
1. A home folder closed to others (`drwx------`), with the environment inside
   it. At MDC one person's pytom-match-pick environment sat in such a folder.

**What to try:** your own environments, as the README's one-time setup
describes. Each person keeps their own clone, pipeline environment and
pytom-match-pick environment (0.14.0). The group's other tools come from the
group's folder.

**What not to do:** open another person's home folder to the group without
asking them.

**Source:** MDC, 5–6 November 2026; decided on 6 November 2026.

### E-11. A job ran out of time

**Kind:** crash · **Evidence:** general

**Where it shows:** `sacct` says `TIMEOUT`, and the log ends in the middle of a
step.

**What to try:**
1. Resume from the stage that stopped (recipe A), with `--overwrite`, since the
   step left part of its outputs.
2. Give the stage more time in its `sbatch` list. At MDC the default is 1 day
   and the limit 7 days; H100 jobs stay under 6 hours.

**What not to do:** rerun a lane stage unchanged with the same time. A step
starts again from its beginning.

**Source:** general for lane jobs. A Dynamo MRA project is the exception: it
goes on from where it stopped, so its job is submitted again unchanged
([dynamo_classification.md](../../dynamo_classification.md)).

## Looks wrong, is fine

### E-12. The job's log is not in the repository's root

**Kind:** false alarm · **Evidence:** measured

**What it means:** a job logs into the folder it was submitted from.
`submit_lanes.py` submits from the repository's root, but a job given to
`sbatch` by hand from elsewhere logs there. `sacct -j <job> -o WorkDir` says
where. `show_project.py` and `show_failure.py` look there too since 29 October
2026.

**Source:** Munich, October 2026: the jobs had been submitted from a folder
outside the repository.

### E-13. `show_failure.py` says "0 failed steps", yet jobs failed

**Kind:** false alarm · **Evidence:** code

**What it means:** "0 failed steps" means nothing is broken now. Steps that
failed and passed when run again are listed after the failures, under "rerun
since", and so are results that failed QC.

**Source:** [pipeline_map.md](../pipeline_map.md), "Reading results".

### E-14. Every MRC read warns of a deprecation

**Kind:** false alarm · **Evidence:** measured

**Error text:** `DeprecationWarning: Setting the dtype on a NumPy array …`

**What it means:** with NumPy 2.5, `mrcfile` itself sets a dtype in a way NumPy
deprecates. It comes from `mrcfile`, not from our code, and changes nothing.

**Source:** MDC environment (Python 3.13, NumPy 2.5), 5 November 2026.

### E-15. The dry run warns that the lane is the first of its kind

**Kind:** reading results · **Evidence:** code

**Error text:** `warning: <combination> has not run on real data yet, so this
lane is the first: expect it to need fixing, and tell the maintainers what
happened …`

**What it means:** of the combinations of input, alignment and membranes, only
some have run on real data. EER movies count as an input of their own. The
others give valid commands but have not run.

**What to try:** run one series first, and tell the maintainer what happened.

**Source:** read in `scripts/submit_lanes.py`; the list is in
[pipeline_map.md](../pipeline_map.md), "Which combinations have run on real
data".

## Questions for reviewers

Please answer in a line each; "don't know" is a useful answer.

- **Q-E1.** Which partitions and GPUs do you use for which steps, and what
  limits have you hit?
- **Q-E2.** Where do you keep projects, and how much space does one take?
- **Q-E3.** Have you met a cluster or environment problem that is not here? In
  a line each: what you saw, what it turned out to be, what helped.
