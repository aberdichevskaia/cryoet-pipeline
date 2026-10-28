# Agent knowledge base

What the terminal assistant reads. It is written so that any model and any client
can use it, and so that a person can check every line. The design is in
[agent_architecture.md](../agent_architecture.md), sections *First increment*
and *Portability*.

| File | What it is |
|---|---|
| [pipeline_map.md](pipeline_map.md) | every lane step: keys, flags, outputs, QC, logs |
| [rerun_recipes.md](rerun_recipes.md) | the safe ways to rerun part of a lane, and the recipe scripts that `scripts/submit_recipe.py` submits |
| [playbook.md](playbook.md) | problems met so far: how to recognise, confirm and fix them |
| [data_folders.md](data_folders.md) | reading a new data folder: the main naming schemes first, then field by field |
| [eval_cases.md](eval_cases.md) | past failures with expected answers, to measure the assistant |
| [templates/](templates/) | what the biologists fill in: instrument and sample cards, playbook entries, failure cases, QC ranges |

Rules for these files:

- **Plain markdown, nothing specific to one client.** Skills are thin wrappers
  that point here.
- **The code wins.** Where these files and the code disagree, the code is right.
  Fix the file, in the same change as the code when you can.
- **Every playbook entry and case names its source.** Something seen on one
  dataset says so.

Status: drafts of 27 October 2026, not yet reviewed.
