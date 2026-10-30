# Case evidence

One folder per case of [eval_cases.md](../eval_cases.md), named by its id. Each
holds what a run of that case needs, so that the case runs anywhere, the same way
each time, with no cluster and no Slurm:

| File | What it is |
|---|---|
| `case.toml` | the request (the run's prompt), a note on what differs from eval_cases.md, the `pass_if` and `must_not` lines for grading, the playbook entries a held-out run removes (`playbook`), the jobs `squeue` lists (`squeue`, as `id\|name\|state\|reason`), and `[env]`, the variables the run gets, where `{bin}` names the run's stand-ins (`CTFFIND5 = "{bin}/ctffind5"`) |
| `tree/` | the case's evidence: files laid over a copy of the repository at the same paths, such as `outputs/<project>/…`, Slurm logs in the root, and config copies |
| `setup.sh` | when the evidence needs a command run first, in the copy, before the assistant starts: E12 dates its running job to the run, E14 writes the frames its mdocs name, E17 writes the earlier dry run into the journal |

The `.gitignore` here keeps every file of a `tree/` in git. Without it, the
repository's own rules leave out `outputs/`, `data/` and Slurm logs at any
depth, and until 30 October 2026 they left out the cases' evidence too.

The constructed cases E11–E14 and E16–E18 have folders since 29 October 2026.
Their files were made up to give the evidence each case describes, and the
values they need, such as the stand-in paths in `[env]`, only have to be set.
The cases from history E01–E05 and E08 have folders since 30 October 2026. They
are trimmed copies from the clusters, made by `scripts/collect_case.py`, with
user names and home paths as placeholders that their `setup.sh` fills in.

## Running them

```bash
python3 scripts/run_cases.py --list
python3 scripts/run_cases.py E11 E12 --runs 3
python3 scripts/run_cases.py --all --runs 3
python3 scripts/run_cases.py --all --runs 1 --dry-run
```

Each run starts the assistant through `scripts/assistant`, headless, with the
case's request as the prompt. It works in its own copy of this repository's
tracked files, with the case's `tree/` on top. Stand-ins for `sbatch`, `scancel`,
`squeue` and `sacct` come first on its `PATH`, so nothing is ever submitted.

The copy differs from a clone in two ways, both found on the first real runs on
30 October 2026:

- it leaves out the cases, this runner and their tests (`EVALUATION` in
  `run_cases.py`), because a run's grep for the misspelled key of E11 brought
  back that case's Pass-if line. `scripts/collect_case.py` stays, because a
  collected case's `setup.sh` runs it in the copy;
- it is a git repository of its own, with one commit, because the client keeps
  auto memory per repository: inside this checkout, a run read the memory of the
  sessions that develop this code.

The run is started with `--permission-mode acceptEdits`, so it may write files,
such as a config copy, as a person who approves edits would let it. In the copy,
that mode also lets `mkdir`, `touch`, `cp`, `mv`, `rm`, `rmdir` and `sed` through
without asking, which a session would ask about. Every deny and ask rule of
`assistant/settings.json` still holds, and so does the guard on shell commands.
Anything that would ask is refused, because nobody is there to answer it.
Claude Code's documentation says so for each of these (checked on 29 October
2026).

The runs need Claude Code signed in on the machine that runs them. Its version
goes into each run's record, and the same version on the Mac and on the cluster
keeps runs comparable. `--dry-run` builds the copies and shows the commands
without starting the assistant.

The results go into `outputs/eval-runs/<UTC time>/`. Each run gets:

- its transcript (`transcript.jsonl`);
- its answer (`answer.md`);
- what it did (`actions.md`), with any Slurm call flagged;
- a record (`record.json`): case, model, client version, commit and the files
  changed since, the plugins and the memory the client used, cost, turns.

`grading.md` holds every case's lines as boxes per run, for the person who
grades. It flags a run that read a memory kept before it, or a `CLAUDE.md` from
a folder above its own. `--held-out` also runs each case without its own
playbook entries (eval_cases.md, "Held out").

Each run costs model use on the account the client is signed in to. Three runs of
every built case are 21 runs.
