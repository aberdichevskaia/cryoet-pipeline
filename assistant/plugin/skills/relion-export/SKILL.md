---
name: relion-export
description: Export a class of an MRA run to a folder RELION 5 runs as it is, for high-resolution refinement. Use when the person says a class of an MRA run looks right and wants it in RELION, asks to send particles or a class to RELION, or asks how to go on from MRA to RELION.
---

# A class of an MRA run, for RELION 5

Paths are from the repository root. What the export writes, its frame and how it
was checked: `docs/relion_export.md`.

## 1. The run and the class

- A project's MRA runs are in `<project>/mra/<NAME>/` (`plan.json`, `setup/`,
  `ite_0001/`, ...); the project is the lane config's `[run] project`. Use the run
  and the class the person names. If they name a class but not the run and the
  project has more than one run, list them (`ls <project>/mra`) and ask.
- A class is a channel of the run: `ite_<n>/average_<c>.mrc` are its classes after
  iteration n. The choice of class is the person's. Asked which looks best, say what
  the run shows (class sizes, decoys per class) and leave the choice to them.
- Only a run of the registry can be exported: one made from a lane (its MRA stage,
  or `--source`). The script refuses another and says why; pass that on.
- Unless the person names an iteration, the last finished one is exported. Say
  which.

## 2. Dry run, shown

```bash
python3 scripts/export_relion.py --run <project>/mra/<NAME> --class <C> --dry-run
```

Add only what the person asks for, to the dry run and the export alike:

- `--agree <project>/mra/<OTHER>:<C2>`: only the particles also in that run's class;
- `--iteration <n>`; `--stack-binning <b>`, when the default (tilt images binned to
  1 A per pixel or more) is not what they want;
- `--dose-per-tilt <e/A^2>`, when a note says the mdoc gives no dose (below);
- `--relion-check $(command -v relion_tomo_reconstruct_particle)`: RELION
  reconstructs the exported class and the script compares it with the exported
  reference. It runs RELION on this node: only when the person asks, RELION 5 is
  found, and the class is a few hundred particles.

Show the person what it printed, in a few lines:

- the class's particles, the decoys left out (decoys in a class mean it holds noise
  too), the duplicates left out, the two halves;
- each series: views, the tilt images' binning, the stack's size in GB, and its
  notes. A note "no dose" means RELION would weight no tilt by dose: ask the person
  for the dose per tilt and dry-run again with `--dose-per-tilt`. The other notes
  (no CTF model, optics assumed, order from z values) go to the person as they are;
- where it will write, `<project>/relion/<NAME>-class<C>`.

Then ask whether to export.

## 3. Export

After an explicit yes, within 30 minutes of the dry run, the same command without
`--dry-run` and with `--reason "<what was asked, and by whom, in their words>"`.
The script checks that it is the dry run you showed: a changed option, a newly
finished iteration, another person's dry run or an older one is refused. Show the
refusal, dry-run again and ask again; never get round it. Every dry run and export
goes into the project's journal.

Tell the person:

- the folder, and how many particles went in;
- its notes: what the rows fall back on;
- that maps RELION makes from it are the mirror of the pipeline's MRA maps (RELION
  is given IMOD's standard orientation, the pipeline's tomograms are its mirror);
- that rlnTomoHand is -1, not measured for their data;
- the next step: `relion_commands.sh` in the folder, Extract at the MRA's pixel,
  then a gold-standard Refine3D from the MRA poses. The folder's README.md says why.

## 4. RELION itself

Never run RELION, `relion_commands.sh` or `mpirun` yourself; the guard blocks them.
On a login node they would take the node, and Refine3D needs MPI. Give the person
the commands of `relion_commands.sh`, to run in a job on the cluster with their own
resources: as many tasks as `MPI` ranks (3), `THREADS` CPUs each (8), and GPUs. The
script loads RELION from the module `RELION5_MODULE` names, or finds it on PATH;
`module avail relion` lists the cluster's builds.

## 5. What it does not do

- It does not look for gold beads: a class that holds beads takes them to RELION.
- It writes over nothing: a folder already there is refused; another export goes
  to another `--out`.
- Its geometry and particle conversion were checked with RELION's own programs on
  EMPIAR-10164 (docs/relion_export.md). On a new dataset, where RELION 5 runs,
  `scripts/relion_geometry_check.py` and `scripts/relion_particle_check.py` check
  them again; they are the person's to run.
