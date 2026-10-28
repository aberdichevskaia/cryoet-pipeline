# The terminal assistant

The pipeline's assistant for biologists is Claude Code, started by
[`scripts/assistant`](../scripts/assistant) with its own settings,
instructions and skills. The design is in
[docs/agent_architecture.md](../docs/agent_architecture.md), sections *First
increment* and *Portability*. This is its first increment: two skills, and the
rules around them.

## Starting it

On a login node, from your clone of the repository:

```bash
scripts/assistant
```

| Command | What it does |
|---|---|
| `scripts/assistant -c` | continues the last session |
| `scripts/assistant --model claude-sonnet-5-5` | starts another model; give a full model ID, not an alias |
| `scripts/assistant --print-command` | shows what would run, and runs nothing |

It needs Claude Code (`claude`) on PATH, signed in to an account whose terms
exclude training on the data (agent_architecture.md, "What leaves the cluster").
Run it under your own Unix account, so that the jobs, the files and the quotas
are yours.

## What is here

| File | What it is |
|---|---|
| [instructions.md](instructions.md) | the rules in force in every turn: the gates, what never to do, how to cite evidence, and the words to use |
| [plugin/skills/project-status](plugin/skills/project-status/SKILL.md) | where a project stands and why something failed: `show_project.py`, then `show_failure.py`, then the playbook and the recipes |
| [plugin/skills/lane-config](plugin/skills/lane-config/SKILL.md) | writing a lane config with the source of every value, checking it, then dry run, yes, and submit |
| [settings.json](settings.json) | the client's permission rules, and the guard on shell commands |
| [guard.py](guard.py) | that guard: it blocks `sbatch`, `srun` and `salloc`, batch scripts run directly, and writes to a journal, wherever they sit in a command line |

The skills are thin. They point at the knowledge base in
[docs/agent/](../docs/agent/README.md), which is plain markdown that a person
or another client can read as well.

## What the launcher sets up

- **`CRYOET_ASSISTANT` and `CRYOET_ASSISTANT_MODEL`.** `submit_lanes.py` and
  `submit_recipe.py` copy both into the project's journal, and hold the
  assistant to their gate:
  - a dry run first, shown to the person;
  - an explicit yes;
  - then the same command with `--reason`, within 30 minutes.

  Claude Code's own `CLAUDECODE` and `AI_AGENT` trigger the same gate, so a
  session started without the launcher is held to it too. That session's
  journal entries just do not name the model.
- **`--setting-sources user`.** Your own Claude Code settings apply, but not the
  `.claude/settings*.json` of the clone. Those belong to development and allow
  far more.
- **The model is passed as a full ID.** Switching it inside a session (`/model`)
  leaves the journal naming the first one, so start a new session instead.
- **Options that would undo this are refused.** These include
  `--dangerously-skip-permissions`, `--permission-mode`, `--settings`,
  `--allowed-tools`, `--system-prompt`, `--plugin-dir`, `--safe-mode` and
  `--bare`.

## Two layers, neither of them a sandbox

1. **The scripts' own checks** are the first layer, and the one that moves to
   another client or model unchanged. `submit_lanes.py` and `submit_recipe.py`
   check:
   - the dry run, the yes, the 30 minutes and `--reason`;
   - `--overwrite`;
   - busy series;
   - the journal.
2. **The client's rules** are the second layer:
   - [settings.json](settings.json) lets the read-only scripts, dry runs,
     `squeue` and `sacct` run without asking;
   - it asks before `scancel` and before writing anything in `configs/lanes/`;
   - it denies `sbatch`, `git`, `run_series.py --execute`, and edits to the
     code, the docs, the journal and its own files;
   - it turns off the modes in which Claude Code would not ask: bypass, and
     auto, where a classifier approves;
   - [guard.py](guard.py) catches what the permission patterns miss. Claude
     Code's documentation says they miss `env sbatch`, `bash -c '...'` and
     `/usr/bin/sbatch`.

What the rules do not cover:

- **A command that hides what it runs.** A shell under your own account can
  still call anything: a Python one-liner that runs `sbatch` gets past both
  rules. Such a command is not on the allow list, though, so Claude Code asks
  you before running it. Read what it asks.
- **Your own allow rules.** Your own Claude Code settings still load, and their
  allow rules add to these. Deny and ask rules win over any allow rule,
  whichever file it is in.
- **A managed setting.** One machine-wide setting,
  `allowManagedPermissionRulesOnly`, would void every rule here. `/status`
  inside a session shows what applies.

## Not done yet

- **No session has answered yet.** On the Mac, Claude Code 2.1.231 reads the
  settings without a warning, but its command-line client there is not signed
  in. On the cluster, the Claude Code version is unknown, and so is whether
  your seat includes Claude Code.
- **Two of the four planned skills are missing:** reading a data folder
  ([docs/agent/data_folders.md](../docs/agent/data_folders.md) is its
  knowledge), and proposing a template from EMDB (roadmap section 23).
- **No eval run yet.** The cases are in
  [docs/agent/eval_cases.md](../docs/agent/eval_cases.md), and none of them has
  been run against the assistant.
- **Stricter loading needs a newer Claude Code.** Claude Code 2.1.248 and later
  has `--restricted`, which loads only managed settings and `--settings`, so
  even the person's own settings would stay out. It also removes Bash unless
  `--tools` names it, so it needs a test before the launcher uses it.
