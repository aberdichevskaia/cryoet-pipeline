# Playbook

Problems we have met, one file per step, in [playbook/](playbook/). Each entry
follows [templates/playbook_entry.md](templates/playbook_entry.md) and names its
source. An entry seen on one dataset says so; it is a lead, not a rule.

Look a symptom up in the file of the step where it shows. The kinds of problem
and the levels of evidence an entry states are explained at the top of
[ctf.md](playbook/ctf.md).

| File | Steps | Entries |
|---|---|---|
| [data.md](playbook/data.md) | the data folder, `cryoet init`, `import-tilt-stack` | D-1 to D-9 |
| [motion.md](playbook/motion.md) | `prepare-tilt-series` (MotionCor3) | M-1 to M-10 |
| [alignment.md](playbook/alignment.md) | coarse alignment, fiducial seed and tracking, fine alignment | A-1 to A-7 |
| [ctf.md](playbook/ctf.md) | `build-ctf-stack`, `estimate-ctf`, `correct-ctf` | C-1 to C-15 |
| [tomogram.md](playbook/tomogram.md) | `build-final-aligned-stack`, `reconstruct-tomogram` | S-1, R-1 to R-5 |
| [membranes.md](playbook/membranes.md) | `segment-tomogram` | G-1 to G-4 |
| [template.md](playbook/template.md) | `prepare-template` | T-1 to T-7 |
| [picking.md](playbook/picking.md) | `pick-particles`, `visualize-picks` | P-1 to P-9 |
| [averaging.md](playbook/averaging.md) | `extract-particles`, averaging, classification, FSC | X-1 to X-3, V-1 to V-12 |
| [cluster.md](playbook/cluster.md) | Slurm, the environment, submission | E-1 to E-15 |

Each file also says what a healthy run looks like, where to look first, what the
code already checks, and what to check before trusting a result.

- **Entry IDs do not change.** Other documents and the evaluation name them. A
  new entry takes the next number of its file's letter.
- **One incident, one entry.** Other entries point to it by its ID and do not
  tell it again.

The step names, keys, and output folders are in [pipeline_map.md](pipeline_map.md),
and the recipes the entries point to are in [rerun_recipes.md](rerun_recipes.md).

Status: drafts of 7–8 November 2026, under review by the biologists. Each file
has a review box at its top and questions at its end, which go once the review
is in. AreTomo alignment is not covered yet.
