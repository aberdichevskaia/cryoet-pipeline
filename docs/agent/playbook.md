# Playbook

Problems we have met, one file per step, in [playbook/](playbook/). Each entry
follows [templates/playbook_entry.md](templates/playbook_entry.md) and names its
source. An entry seen on one dataset says so; it is a lead, not a rule.

Look a symptom up in the file of the step where it shows. What every entry
states about itself is explained below, in
[What an entry states](#what-an-entry-states).

| File | Steps | Entries |
|---|---|---|
| [data.md](playbook/data.md) | the data folder, `cryoet init`, `import-tilt-stack` | D-1 to D-9 |
| [motion.md](playbook/motion.md) | `prepare-tilt-series` (MotionCor3) | M-1 to M-10 |
| [alignment.md](playbook/alignment.md) | coarse alignment, fiducial seed and tracking, fine alignment (tiltalign or AreTomo2) | A-1 to A-13 |
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

## What an entry states

Every entry states its kind, so that a reader knows how the problem shows, and
its evidence, so that a reader knows how far to trust it. When an entry is
cited, its evidence goes with it.

- **Kinds:**
  - crash: the job stops;
  - refused: the dry run stops the submission;
  - warning: the QC or the report warns;
  - silent: nothing warns;
  - false alarm: it looks wrong and is fine;
  - reading results: how to read an output that is easy to misread;
  - slow.
- **Evidence:**
  - measured: seen on our data and checked;
  - seen: met on our data, not checked further;
  - code: read in our code, not met in a run;
  - manual: in the tool's documentation, at the version named;
  - general: general knowledge or a colleague's practice, not checked here.

An entry also gives the error text word for word when there is one, and what it
applies to: versions, dates and sites. Entries marked *for the maintainer* are
about our code.

The step names, keys, and output folders are in [pipeline_map.md](pipeline_map.md),
and the recipes the entries point to are in [rerun_recipes.md](rerun_recipes.md).

Status: drafts of 7–8 November 2026, under review by the biologists. Each file
has a review box at its top and questions at its end, which go once the review
is in. AreTomo alignment (A-8 to A-13) was added on 11 November 2026, from one
series.
