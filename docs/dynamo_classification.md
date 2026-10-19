# Dynamo MRA classification and its report

Real Dynamo's multireference alignment (MRA) on an export of
`scripts/export_dynamo.py`, one project per seed design, and a PDF report per
project. The pipeline runs none of it yet; the report's pages are written so
that the pipeline's own classification can feed them later.

## Make and run a project on the MDC cluster

From the repo root on max, one job per project:

```bash
EXPORT=<export folder> NAME=mra_nss SEEDS="subset subset noise" sbatch --export=ALL --job-name=mra_nss scripts/dynamo_mra_mdc.sbatch
```

The job activates the group's Dynamo
(`/fast/AG_Kudryashev/artsemi/software/Dynamo`, shipped MATLAB Runtime 9.9,
destination `standalone`: the installation has no OpenMP wrappers), makes the
project with `scripts/dynamo_mra_setup.m` if `EXPORT` does not hold it yet,
computes it on the job's cores (32 by default; the project is made for exactly
that many), and writes the report beside it. A project that exists is computed
on from where Dynamo stopped, so a job that ran out of time is submitted again
unchanged. After every iteration the job log says how long it took by Dynamo's
own clock (`>>> iteration 3 (round 3) took 25 min 03 s`), and at the end it
lists them all with their sum; they are also kept in
`<NAME>_iteration_times_<job>.txt` beside the project.

`SEEDS` gives one starting reference per channel:

| kind | starting reference |
|---|---|
| `subset` | the average of 300 random particles at their exported (pytom) poses |
| `blob` | the average of 300 random particles at random orientations: a featureless blob |
| `noise` | white noise |

Channel k always averages the k-th block of 300 particles of one seeded random
order, so a project's subsets are disjoint and channel k holds the same
particles in every project. Every channel starts with the whole table, and
particles swap between channels (`mra 1`). By default the search is the card of
refine223 and the CZII MRA arms (cone and in-plane 15/5, two refinement levels,
band 2..16, shifts within 3 px of the box centre with `limm 1`, 5 iterations),
made for ribosomes at bin8: check it before other data.

`SCHEDULE=coarse_to_fine` searches coarse to fine instead, in five Dynamo rounds
of one iteration each. The low-pass moves in equal steps of Fourier pixels from
`FIRST_LOWPASS` (40 Å) to `LAST_LOWPASS` (22 Å), and the angular step follows it
as atan(resolution / diameter), `DIAMETER` being the particle's (290 Å, the
80S). The schedule stops short of Nyquist (19.6 Å here): with no independent
halves, the last frequencies are where a search fits noise most easily. The
cone and the in-plane range stay three steps wide (Dynamo's `cone_range` is the
cone's full opening), as in the default card, so every iteration costs about the
same. At 9.784 Å/px in a 48 px box:

| round | low-pass | step | cone and in-plane |
|---|---|---|---|
| 1 | 12 px, 39 Å | 7.7° | 23.1° |
| 2 | 14 px, 34 Å | 6.6° | 19.8° |
| 3 | 17 px, 28 Å | 5.4° | 16.2° |
| 4 | 19 px, 25 Å | 4.9° | 14.7° |
| 5 | 21 px, 22 Å | 4.4° | 13.2° |

Every round is given every search parameter, `nref` and `mra` included: a
parameter put without a round goes to round 1 only, and later rounds would fall
back to Dynamo's defaults, one reference and no swapping.

```bash
EXPORT=<export folder> NAME=mra_nbb_ctf SEEDS="blob blob noise" SCHEDULE=coarse_to_fine sbatch --export=ALL --job-name=mra_nbb_ctf scripts/dynamo_mra_mdc.sbatch
```

Two things measured on the Mac (Dynamo 1.1.511, 2026-10-16) bear on the design:

- **A channel that no particle chooses is dropped for good.** Against two
  12-particle `subset` seeds, on 49 ribo-mdc particles, a white-noise seed got
  no particle in the first iteration, nor did an average with its phases
  replaced by noise; a `blob` kept one. pytom chose every pick and pose by its
  match to the template, and `subset` seeds are averages at those very poses, so
  any pick, false or not, matches them better than a seed without structure.
- **Two projects whose parallel pools start together can fail to open them,
  and can break every pool after them:** the MATLAB Runtime keeps every pool's
  job files in one folder under the home directory. Two pools started together
  on the Mac wrote one job's type twice (`concurrentconcurrent` in its
  `Job<N>.metadata.txt`), and from then on every pool failed with
  "Unknown type" until that job's files were moved aside. Start each job a
  quarter of an hour after the previous one starts
  (`--dependency=after:<job>+15`). A job whose pool does not open stops with a
  message, computes nothing, and lists any pool job record MATLAB cannot read.

Each particle and channel costs about 4 core-seconds on the Mac, so a
three-channel iteration of 2,703 particles is about 9 core-hours. A Munich
cluster core took about twice as long for the same search; MDC's is unmeasured.

## The decoy control

Does a class collect junk, or only split the picks at random? Only junk whose
identity is known can say. `scripts/decoy_picks.py` draws boxes at random
positions in the same tomograms (depths from the real picks, the real list's
spacing), each carrying the orientation and score of a real pick and the real
list's template box, so extraction crops them exactly as it crops the picks.
`export_dynamo.py --decoy-variant` puts them after every real particle in one
export and marks them in the manifest; Dynamo classifies them blind, and the
report adds a page with each class's share of all decoys against its share of
all real particles.

```bash
python scripts/decoy_picks.py --registry outputs/ribo-mdc/artifacts.json --series TS_0306 --real-variant bin8 --count 120 --variant bin8-decoy --out outputs/ribo-mdc/picks/TS_0306
```

then, per series, `cryoet extract-particles ... --box-size-px 48 --variant
bin8-decoy`, and the export with `--variant bin8 --decoy-variant bin8-decoy`.
120 decoys per tomogram make about 18% of the set: enough to count, few enough
to leave the real set mostly as it was. The reading, written before the run: a
class collects junk if it holds at least twice the share of decoys it holds of
real particles and the 95% interval of its decoy share leaves out its real share.

## The report

```bash
python scripts/dynamo_classification_report.py <export folder>/mra_nss
```

writes `<export folder>/mra_nss_classification_report.pdf`. It reads the project
and the export manifest beside it (pixel size, tilt-series and picking score of
each particle) and changes nothing, so it can run while Dynamo still iterates.
The pages:

1. the parameters: data per tilt-series, how each starting reference was made,
   masks, the search of every round, the time of every iteration;
2. 32 particles at their starting poses, each the mean of its 10 central z
   sections;
3. the average of all particles at their starting poses, computed with the
   pipeline's port of `dynamo_average` (about a minute for 2,700 particles), or
   any map given with `--initial-average`;
4. the starting references;
5. class sizes after every iteration, with the particles that changed class;
6. every iteration's class averages: the central XY, XZ and YZ sections beside
   ten successive z sections, the way Dynamo's `dslices` shows a map;
7. the last classes by tilt-series and by picking score;
8. particles of each last class at its own poses.

What the reader relies on, read from real projects: a particle's class is
column 34 of the refined table of a channel that the next iteration still lists
(a dropped channel's table is never rewritten and still claims every particle),
and an iteration is finished once the project's log.txt says `iteration N ok`.
The next iteration's list of channels is not enough: Dynamo writes it while the
MRA plugin is still rewriting tables and averages, with every channel in it.
The code is `src/cryoet_pipeline/reports/dynamo_project.py` (reading) and
`src/cryoet_pipeline/reports/classification.py` (pages).
