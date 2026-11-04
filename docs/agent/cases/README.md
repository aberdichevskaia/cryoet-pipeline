# Case evidence

One folder per case of [eval_cases.md](../eval_cases.md), named by its id. Each
holds what a run of that case needs, so that the case runs anywhere, the same way
each time, with no cluster and no Slurm:

| File | What it is |
|---|---|
| `case.toml` | the request (the run's prompt), the person's next messages (`follow_up`), a note on what differs from eval_cases.md, the `pass_if` and `must_not` lines for grading, the playbook entries a held-out run removes (`playbook`), the jobs `squeue` lists (`squeue`, as `id\|name\|state\|where`, with the time it has run after a fifth `\|`), what `sacct` knows of a job (`[sacct.<id>]`, field = value, as in a collected case's `manifest.json`), and `[env]`, the variables the run gets, where `{bin}` names the run's stand-ins (`CTFFIND5 = "{bin}/ctffind5"`) |
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
`sacct` answers for the jobs the case records: the `"slurm"` map of a collected
case's `manifest.json` and the case's `[sacct]`. It gives each job the copy's
root as its WorkDir, where the job ran from in the clone.

The copy differs from a clone in two ways, both found on the first real runs on
30 October 2026:

- it leaves out the cases and this runner (`EVALUATION` in `run_cases.py`),
  because a run's grep for the misspelled key of E11 brought back that case's
  Pass-if line. Since 4 November it also leaves out `docs/agent_architecture.md`,
  whose starting case set states the answers, and `tests/`, whose fixtures hold
  past incidents. `scripts/collect_case.py` stays, because a collected case's
  `setup.sh` runs it in the copy;
- it is a git repository of its own, with one commit, because the client keeps
  auto memory per repository: inside this checkout, a run read the memory of the
  sessions that develop this code. The commit comes after `setup.sh`. Made before
  it, every file whose placeholders `setup.sh` filled in showed as changed in the
  client's git status, and the runs of E05 and E08 hesitated over it.

A run on a cluster node also sees the cluster: its software, such as a CTFFIND 5
under `/fast`, and its module messages in every command. The record names the
host, and runs are compared only with runs from the same host.

A headless run is one turn, and a run that ends on a question to the person
cannot reach a Pass-if line such as "it shows the dry run". A case's `follow_up`
lines are the person's next messages. Once a turn has ended with exit 0, the next
one goes to the same session (`--resume`), and so on in order. `answer.md` and
`actions.md` give each message's part under its own heading. They are written for
the answers a run can be expected to ask for: E14's names the gain, the template,
the names and where to write the config, and says to take the run's proposals for
the rest.

The run is started with `--permission-mode acceptEdits`, so it may write files,
such as a config copy, as a person who approves edits would let it. In the copy,
that mode also lets `mkdir`, `touch`, `cp`, `mv`, `rm`, `rmdir` and `sed` through
without asking, which a session would ask about. Every deny and ask rule of
`assistant/settings.json` still holds, and so does the guard on shell commands.
Anything that would ask is refused, because nobody is there to answer it. That
includes an edit under `configs/lanes/`, so a follow-up that needs a new config
names a place in the project for it.
Claude Code's documentation says so for each of these (checked on 29 October
2026).

A run that could not reach Anthropic is flagged in `grading.md`, and the batch
stops after two such runs in a row, since every run after them fails the same
way. On 3 November 2026 a Mac on battery fell asleep, and 21 runs failed over
three hours. Run a long batch on a Mac as `caffeinate -i python3
scripts/run_cases.py …`.

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
playbook entries (eval_cases.md, "Held out"), and `--only-held-out` runs only
that copy. A held-out copy also goes without the other lines that tell the
case's answer: [held_out.toml](held_out.toml) names them, a worked example or a
comment about the incident, with what replaces each, and its `markers` are the
answer in a few patterns that a test looks for in every held-out copy. A run
stops when a line it should replace is no longer in its file. The capabilities
that the cases brought stay, such as the `ctf-window` recipe, `volume_split` and
`CTFFIND5`: a run may still reason its way to them. Until 4 November the answers
of E01, E02, E03 and E05 were left in, so earlier held-out runs of those cases say
nothing about reasoning.

Each run costs model use on the account the client is signed in to. Three runs of
every built case are 21 runs.
