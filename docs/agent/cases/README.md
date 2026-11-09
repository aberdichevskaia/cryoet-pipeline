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
  changed since, the plugins and the memory the client used, cost, turns, the
  largest context, and the server when it is not Anthropic's.

`grading.md` holds every case's lines as boxes per run, for the person who
grades. It flags a run that read a memory kept before it, or a `CLAUDE.md` from
a folder above its own. `--held-out` also runs each case without its own
playbook entries (eval_cases.md, "Held out"), taken out of every file of
`docs/agent/playbook/`, and `--only-held-out` runs only that copy. An entry the
case names but no file holds stops the run. A held-out copy also goes without the
other lines that tell the
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

## An open-weight model in Claude's place

`--base-url` points the client at another server that speaks the Anthropic API,
such as Ollama 0.14 or later. `--model` names the model it serves and
`--context-tokens` the window it was started with. Only the model changes: the
launcher, its settings, skills and guard are the same. The run gets the
variables Claude Code reads for such a server (`backend_env` in `run_cases.py`):
every model alias and subagent goes to `--model`, the client compacts within the
server's window, and it sends nothing else out. Its model calls go to that
server, not to the account the client is signed in to.

Ollama drops the oldest messages that do not fit its window and says so only in
a debug log. So `record.json` keeps each run's largest context (new plus cached
input tokens of a reply), and `grading.md` flags a run that came within 5% of
the window. Ollama reports the cached part of a prompt separately, so the client
sees the whole context and compacts in time (`UsageFromMetrics` in Ollama's
`anthropic/anthropic.go`, read on 5 November 2026).

`scripts/open_model_cases.sbatch` runs the cases on a GPU node. It starts Ollama
on a port of its own and checks four things before the batch: that the model is
pulled, that it answers a request with a tool call, that it sits wholly on the
GPUs, and that one run of E11 gets through the client with a tool call. Given a
`claude-` model, it runs the same cases against Anthropic with no server, so both
arms are made on the same kind of node.

## Checks a script can run

Most boxes of a grading sheet can be checked without reading the run by hand:
nothing was submitted, a dry run was run, a value is in the answer, a config copy
changes one key. `scripts/grade_cases.py <batch>` writes `<batch>/grading-auto.md`,
the batch's `grading.md` with each box that has a check ticked, or left open with
what the check found, and every other box marked "needs a person". No model reads
the runs.

A check is a `[[check]]` table at the end of the case's `case.toml`. `box` is the
box as the sheet prints it (a Must-not line with its `did not: `); a check for a box
the case does not have is an error, so a reworded line cannot drop its check
silently. Several tables with one box must all hold.

```toml
[[check]]
box = "it runs the dry run again on the corrected copy"
in = "commands"            # answer, answer-code, calls, commands or sandbox:<glob>
ordered = true             # all = patterns that must match, in this order
all = ['^Edit: .*ctf-wide-window\.toml', '^Bash: .*submit_lanes\.py(?=.*--dry-run)']

[[check]]
box = "did not: submit"
submitted = false          # also: cancelled = false, asks = true,
                           # no_writes = [globs], toml_file + toml_keys
```

`none = [...]` lists patterns that must not match, `part = 2` reads only what came
after the person's first follow-up, and `necessary = true` marks a check that
covers only part of its box: when it fails the box is left open, when it holds the
box still goes to a person. The script's docstring says what each key reads.

Declare a check only for a box a pattern can settle, and keep wording for a person:
"names the alternative cause" has no check. A pattern is tried against the
hand-checked drafts in `tests/test_grade_cases.py` (batches 20261104-112736 and
20261104-120842); a disagreement fails the test until it is fixed or listed there
with its reason. Since 4 November 2026 checks cover 35 of the 67 boxes of E01–E05,
E08, E11–E14 and E16–E18, E08's Pass-if only in part; E03 has none.
