## In Codex

This session runs in OpenAI's Codex, not in Claude Code. Every rule above holds
as written; where it names Claude Code's tools, use Codex's: read files with the
shell, and change a file with apply_patch. Three things are Codex's own:

- **The sandbox.** A command runs in Codex's sandbox, which reads everything,
  writes nothing and has no network, unless the session's rules let it run
  outside (the show_*.py scripts, run_series.py, search_library.py, squeue,
  sacct and sinfo, the dry runs). A command that fails there with "operation not
  permitted" or a network error is not to be worked around: ask the person for
  it, saying what it changes and why.
- **Dry runs.** Put `--dry-run` right after the script's name, as in
  `python3 scripts/submit_lanes.py --dry-run <config> ...`,
  `python3 scripts/submit_recipe.py --dry-run <recipe> <config> ...` and
  `python3 scripts/export_relion.py --dry-run --run ... --class ...`. Only that
  form runs outside the sandbox, where the dry run can read the queue and write
  the project's journal. The submission after the person's yes is the same
  command without `--dry-run`, with `--reason`.
- **No shell that waits for input.** Never start `bash` or `python3` on its own to
  type commands into it later: the guard blocks it. Run each command as itself.
