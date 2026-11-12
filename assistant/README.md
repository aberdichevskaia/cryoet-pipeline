# The terminal assistant

The pipeline's assistant for biologists is Claude Code, started by
[`scripts/assistant`](../scripts/assistant) with its own settings,
instructions and skills. For a person whose account is ChatGPT's, the same
launcher starts OpenAI's Codex with the same instructions, skills and guard
(`--codex`, below). The design is in
[docs/agent_architecture.md](../docs/agent_architecture.md), sections *First
increment* and *Portability*. This is its first increment: four skills, and the
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
| `scripts/assistant -p "…" --permission-mode acceptEdits` | a headless run, as `scripts/run_cases.py` makes them: edits go through as a person who approves them would let them; only with `-p` |
| `scripts/assistant --codex` | starts the assistant in Codex instead; `-c`, `--model` and `--print-command` work the same |
| `scripts/assistant --codex --login` | signs the assistant's Codex in, once; add `--device-auth` on a login node, where no browser opens |

`export CRYOET_ASSISTANT_BACKEND=codex`, in `~/.bashrc` say, makes Codex the
default, and `--claude` still starts Claude Code.

It needs Claude Code (`claude`) on PATH, or Codex (`codex`) for `--codex`,
signed in to an account whose terms exclude training on the data
(agent_architecture.md, "What leaves the cluster"). Run it under your own Unix
account, so that the jobs, the files and the quotas are yours.

## What is here

| File | What it is |
|---|---|
| [instructions.md](instructions.md) | the rules in force in every turn: the gates, what never to do, how to cite evidence, and the words to use |
| [plugin/skills/project-status](plugin/skills/project-status/SKILL.md) | where a project stands and why something failed: `show_project.py`, then `show_failure.py`, then the playbook and the recipes |
| [plugin/skills/emdb-template](plugin/skills/emdb-template/SKILL.md) | one to three EMDB entries proposed as the picking template with `show_emdb.py`, each checked against the lane, for the person to choose |
| [plugin/skills/empiar-download](plugin/skills/empiar-download/SKILL.md) | an EMPIAR entry read with `show_empiar.py`, its download layout written and checked against each series' mdoc, and the download command handed to the person |
| [plugin/skills/lane-config](plugin/skills/lane-config/SKILL.md) | writing a lane config with the source of every value: the data read with `show_folder.py`, then checking it, dry run, yes, and submit |
| [plugin/skills/relion-export](plugin/skills/relion-export/SKILL.md) | a class of an MRA run exported for RELION 5 with `export_relion.py`: dry run, shown with its notes, yes, export with `--reason`; RELION itself is the person's to run |
| [settings.json](settings.json) | the client's permission rules, and the guard on shell commands |
| [guard.py](guard.py) | that guard: it blocks `sbatch`, `srun` and `salloc`, batch scripts run directly, writes to a journal, RELION's programs and `mpirun`, `run_series.py --execute`, and a shell that would run what comes on its input, wherever they sit in a command line; and Codex's patches of files that settings.json's Edit rules deny |
| [codex/](codex/) | the same for Codex: [config.toml](codex/config.toml) (sandbox, approvals by the person, nothing but the shell and patches), [hooks.json](codex/hooks.json) (the guard), [assistant.rules](codex/assistant.rules) (settings.json's command rules, in Codex's form), and [instructions.md](codex/instructions.md) (three notes after the instructions) |

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
- **No MCP servers and no editor.** `--strict-mcp-config` and
  `ENABLE_CLAUDEAI_MCP_SERVERS=false` keep out the MCP servers of your own
  settings, your plugins and your claude.ai account. `CLAUDE_CODE_AUTO_CONNECT_IDE=false`
  keeps out the editor's, such as VS Code's over SSH. The first session in
  Munich had 21 such servers, none of them for the pipeline, and each person's
  set differs. `/mcp` in a session should list none.
- **Options that would undo this are refused.** These include
  `--dangerously-skip-permissions`, `--permission-mode`, `--settings`,
  `--allowed-tools`, `--system-prompt`, `--plugin-dir`, `--mcp-config`, `--ide`,
  `--safe-mode` and `--bare`.
- **The reference library, read-only** ([docs/library.md](../docs/library.md)).
  - `CRYOET_LIBRARY` names the folder of snapshots, or `none` for a session
    without it.
  - Unset, a session for you takes the site's folder, such as MDC's
    `/fast/AG_Kudryashev/cryoet-library`. A headless run (`-p`) takes one only
    when the variable names it.
  - The session reads the snapshot `current` names, given with `--add-dir`. The
    instructions gain `library.md` and the snapshot's path, and the launcher
    prints the snapshot's name.
  - The assistant searches the library with `scripts/search_library.py`.
  - `guard.py` refuses every write to the library, by any tool, and
    `scripts/build_library.py` in any form.

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
  `allowManagedPermissionRulesOnly`, would void every rule here. Another,
  `allowManagedHooksOnly`, would void the guard. `/status`
  inside a session shows what applies.

## In Codex

`scripts/assistant --codex` starts Codex with the same instructions, skills and
guard. The settings below were read from Codex 0.167.2 and checked on the Mac on
12 November 2026, against a stand-in model server on 127.0.0.1 that asked for
what the guard must stop. No session with a real model has run yet.

- **Its own Codex home.** `CODEX_HOME` is `~/.cryoet-assistant/codex` (or
  `CRYOET_ASSISTANT_CODEX_HOME`). It holds the assistant's sign-in and sessions.
  None of your own `~/.codex` applies: your settings, rules, hooks, MCP servers
  and `AGENTS.md` stay out, and the assistant's settings stay out of your own
  Codex. At every start the launcher copies `codex/config.toml`, `hooks.json`
  and `assistant.rules` into it. It also links `skills/cryoet` to this clone's
  skills, which Codex lists as `cryoet:lane-config` and so on. Sign in once with
  `--login`.
- **The guard runs before every shell command and every patch.** That includes
  the calls GPT-6.1 makes from inside its JavaScript tool, and a patch given to
  the shell as `apply_patch <<'EOF'`, which Codex applies too. A command blocked
  with exit 2 is never run, and the model gets the guard's message.
  - Codex runs a hook only once someone has trusted it in `/hooks`. An untrusted
    hook is skipped without a word: on 12 November a patch the guard blocks went
    through that way. So the launcher passes `--dangerously-bypass-hook-trust`,
    and the home holds no hook but the guard.
  - Codex has no rules for files, so the guard reads the Edit rules of
    `settings.json` for Codex's patches.
- **The sandbox does the asking.** Commands run read-only and without the
  network (`-s read-only -a on-request`), and `approvals_reviewer = "user"`.
  - Every patch asks you, as Claude Code asks before each edit.
  - A command that needs to write or to reach the network asks you too.
  - The rules in `assistant.rules` are `settings.json`'s, in Codex's form. Their
    `allow` runs a command outside the sandbox (checked: a `touch` outside the
    workspace worked under read-only), which is why the Slurm queries, the
    `show_*.py` scripts and the dry runs are there. `prompt` asks before
    `scancel`, and `forbidden` refuses `sbatch`, `git` and the library builder.
  - Codex's rules match the start of a command, so a dry run runs outside the
    sandbox only with `--dry-run` right after the script.
    [codex/instructions.md](codex/instructions.md) tells the model so, and the
    tests check the rules against `settings.json`.
- **Nothing else acts.** Web search, apps (ChatGPT's connectors), browser and
  computer use, and Codex's own skills are off. `--strict-config` stops the
  start if this Codex does not know one of these settings. `--no-daemon` gives
  the session's commands the launcher's variables, rather than those of a
  background server started earlier.
- **The journal.** Commands see `CRYOET_ASSISTANT` ("cryoet-assistant; codex
  0.167.2"), `CRYOET_ASSISTANT_MODEL` (`gpt-6.1-sol` unless `--model` says) and
  Codex's own `CODEX_THREAD_ID`. `submit_lanes.py` counts the last one as an
  assistant too, so a Codex session started without the launcher is held to the
  gate, and journals it as the session's id.

Not checked yet, and only a real session can show it: how the terminal UI asks
for a patch or an escalation, signing in to the separate home with a workspace
account, and whether the project's trust question still appears. On max: whether
Codex's sandbox (bubblewrap) runs on a login node, and whether the Slurm queries
and dry runs run there through the allow rules.

## Sessions so far

- **Munich, 28 October 2026**, on a login node, with Claude Code 2.1.312 and a
  Helmholtz Munich Enterprise seat, which includes Claude Code. The rules, the
  guard and the skill worked, and it read a picking failure right. The MCP and
  editor switches above came after it.
- **Munich, 29 October.** This session checked those switches: `/mcp` listed no
  servers, and `/status` gave the VS Code extension as installed, not
  connected.
- **MDC, 5 November**, on a login node, with Claude Code 2.1.318 and the
  library. Asked what newstack does, it searched the snapshot with
  `scripts/search_library.py` and answered from the manual page, citing
  "IMOD 5.1.12, man/newstack.html". MDC's sessions open with Claude Code's note
  that the organization may apply global configuration. The guard still ran
  there: it refused a `python3` command that named the library
  ([docs/library.md](../docs/library.md), *Read only, enforced in layers*).

## Evaluation

The cases are in [docs/agent/eval_cases.md](../docs/agent/eval_cases.md). Their
evidence and how to run them are in
[docs/agent/cases/](../docs/agent/cases/README.md). `scripts/run_cases.py` runs
each case headless, in its own copy of the repository, with stand-ins for
Slurm. Batches have run on the Mac since 30 October 2026. Their records are in
`outputs/eval-runs/`, which git leaves out.

## Not done yet

- **The grades are drafts.** A model of the same family drafted them, and
  `scripts/grade_cases.py` pre-ticks the boxes a script can check. A tick is a
  pointer; the grade is a person's.
- **The latest held-out runs are not graded.** These are the runs that go
  without every line naming their case's answer, which they have done since
  4 November.
- **Five cases have no evidence yet:** E06, E07, E09, E10 and E15.
- **Codex has no headless runs.** `scripts/run_cases.py` drives Claude Code
  only, so the cases have not run in Codex, and `--codex -p` is refused until
  they can.
- **No run against an open-weight model.** `scripts/run_cases.py --base-url`
  and `scripts/open_model_cases.sbatch` are written, and nothing has run on max.
- **Stricter loading needs a newer Claude Code.** Claude Code 2.1.248 and later
  has `--restricted`, which loads only managed settings and `--settings`, so
  even the person's own settings would stay out. It also removes Bash unless
  `--tools` names it, so it needs a test before the launcher uses it.
