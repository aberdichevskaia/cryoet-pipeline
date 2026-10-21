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

### Decoys chosen by template matching

Random boxes are the easy case: on the ribo-mdc picks the noise-seeded channel
took 86% of them and 0.5% of the real particles. pytom's false picks are not
random boxes: each is noise that matched the template better than everything
around it, and an average of such boxes looks like the template.
`scripts/null_decoys.py` makes decoys the same way. Per series it writes a copy
of the picks' tomogram with every Fourier phase replaced by white noise's (the
same amplitudes, so the same spectrum, wedge, CTF zeros and variance, and
nothing anywhere; `src/cryoet_pipeline/null_volumes.py`), registers it as a full
tomogram labelled `bin8-null`, runs the lane's own pick on it with the template
and mask the real picks were matched with, keeping the 120 best peaks with no
score floor, and extracts them with the lane's box. It refuses a lane config that
no longer matches how the real picks were matched (low-pass, per-tilt weighting,
volume split). On MDC, one GPU job for the five series and the export, then the
classification after it, with the search mra_nbb_ctf_decoy ran with the random
decoys (its low-passes 39-22 A stepped in frequency, from before the default
became steps in A):

```bash
PROJECT=outputs/ribo-mdc CONFIG=configs/lanes/ribo-mdc.toml SERIES="TS_0306 TS_0463 TS_0493 TS_0564 TS_0623" EXPORT=outputs/ribo-mdc/dynamo/export_null sbatch --export=ALL --job-name=null-decoys scripts/null_decoys.sbatch
EXPORT=$PWD/outputs/ribo-mdc/dynamo/export_null NAME=mra_nbb_ctf_null SEEDS="blob blob noise" SCHEDULE=coarse_to_fine LOWPASS_STEPS=fourier sbatch --export=ALL --job-name=mra_nbb_ctf_null --dependency=afterok:<job> scripts/dynamo_mra_mdc.sbatch
```

The job log says, per series, how far the copy is from the tomogram (same
spread, correlation near zero), how the real picks' scores stand against the
null's best (what noise of this spectrum reaches against this template over a
volume of the same size), and every setting the two pytom jobs differ in besides
the tomogram, which should be none. The export marks the decoys as pytom picks in
phase-randomised tomograms, and the report says so.

The reading, written before the run, with the rule above: if a channel collects
these decoys too, Dynamo can clean pytom's list; if every channel holds them in
the share it holds of real particles, it cannot tell template-selected noise
from the picks, and the cleaning belongs to picking. Two differences from the
real false picks are known. The copy spreads the lamella's contrast and the
vacuum's evenly over the volume, so its picks come from any depth and sample the
tomogram's average texture; and boxes closer than half a box to an edge are
dropped at extraction, which takes more of these than of the real picks. A
separation could therefore come from the spectrum alone, and should be checked
for that before it is read as structure.

A second run on the same export changes one thing, the first low-pass: 80 A
instead of 40, so rounds of 6, 10, 14, 17 and 21 px (78, 47, 34, 28 and 22 A) and
steps from 15.1 to 4.4 deg, with the cone 3 steps wide as before and so about the
same cost per iteration. Started a quarter of an hour after the first:

```bash
EXPORT=$PWD/outputs/ribo-mdc/dynamo/export_null NAME=mra_nbb_ctf80_null SEEDS="blob blob noise" SCHEDULE=coarse_to_fine LOWPASS_STEPS=fourier FIRST_LOWPASS=80 sbatch --export=ALL --job-name=mra_nbb_ctf80_null --dependency=afterok:<job>,after:<job of mra_nbb_ctf_null>+15 scripts/dynamo_mra_mdc.sbatch
```

Why the first low-pass: in both nbb decoy runs the noise channel took the random
decoys in the first iteration (mra_nbb_ctf_decoy: 83% after it, 87% after five),
so the first round is where the sorting is decided. And pytom matched after
whitening up to 25 A, where the band coarser than 80 A holds a few per cent of
the Fourier coefficients (shells up to 7 of 24 in its 60 px box): unless those
weigh far more than the rest, the noise it selected looks like a ribosome at
25-50 A and like nothing in particular at 80 A, where a ribosome has its mass.
At 39 A the first round still sees much of the band the selection was made in.
The reading, written before the run: if this run collects the null decoys by the
rule above and the 40 A run does not, the low band is what tells a ribosome from
selected noise. The wider first cone (45 deg against 23) comes with the step
rule, so it is part of the change, not a second one.

More iterations were not added. In both nbb decoy runs the noise channel's share
of the decoys had stopped rising by the third iteration (83, 86, 87, 87, 87% and
76, 84, 85, 85, 86%) and its size by the second (580, 583, 582, 584, 584); judging
by the sizes, what still changed class later (73, then 51 particles) moved
between the two blob channels, which hold random halves of one structure. If the
null decoys are still moving into the noise channel at the fifth iteration, that
changes.

### Picks made with the solved tilt angles

The first null run's job comparison found the real and the null searches
differing in their tilt angles: until 6b8e3be (2026-10-18) pick-particles gave
pytom the tracking stack's stage angles, while the tomograms are reconstructed
from the solved ones, 9-11 deg away on ribo-mdc. `scripts/repick.py` repeats the
lane's search, as its picks record it, under a new label, so the angles are the
one thing that changes, extracts the new picks, and compares them with the old
picks and with the null search. On MDC, with an export of the new picks followed
by the null decoys, which were picked the same way:

```bash
PROJECT=outputs/ribo-mdc CONFIG=configs/lanes/ribo-mdc.toml SERIES="TS_0306 TS_0463 TS_0493 TS_0564 TS_0623" LABEL=bin8-solved EXPORT=outputs/ribo-mdc/dynamo/export_solved_null sbatch --export=ALL --job-name=repick scripts/repick.sbatch
```

The reading, written before the run: the new search's job should differ from
the old one's in the tilt angles alone and from the null search's in nothing.
Summed over the series, picks above the null's best more than a tenth up: the
stage angles cost real picks, and averaging and classification move to the new
lists; within a tenth: they did not matter at bin8; more than a tenth down: the
solved angles search worse, which points at the angles fine alignment solved.

## Comparing two runs particle by particle

A split such as 74/13/13 says nothing alone: classification of noise also makes
classes. What a real split does is reproduce: the same particles land in the same
classes when the run starts from other seeds or has other particles mixed in.

```bash
python scripts/compare_dynamo_classes.py outputs/ribo-mdc/dynamo/export/mra_sbb outputs/ribo-mdc/dynamo/export_decoy/mra_sbb_decoy --group-first 2,3 --group-second 2,3
```

prints how many particles each pair of classes shares, against what two
independent runs with these class sizes would share; the adjusted Rand index (0
at chance, 1 when identical) and the share of particles in the same class under
the best pairing of classes, each against the same statistic over 2,000 shuffles
of the second run's labels; and, given a group of channels in each run (the
blob- or noise-seeded ones), how the two groups overlap against the
hypergeometric chance. Particles are matched by tag after checking, in both
export manifests, that each tag names the same series, stack and index; decoys
are left out. `--out` also writes the numbers as JSON. The statistics are in
`src/cryoet_pipeline/class_comparison.py`, shared with
`scripts/compare_classifications.py`.

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

With `--template <map>` (MRC or EM; on max
`outputs/ribo-mdc/templates/TS_0306/TS_0306_pytom_bin8_template.mrc`, and
`TEMPLATE=` for scripts/dynamo_mra_mdc.sbatch) every page of particles or maps
shows the template in its top right corner, drawn as that page draws its own: as
a gallery tile on the particle pages, as its central XY, XZ and YZ sections on
the map pages, on its own grey scale. It is cut or padded to the particles' box
and resampled to their pixel size when its header gives another. No turning is
needed: for picks from pytom the references' frame is the template's (the average
of 400 ribo-mdc particles at their starting poses correlates +0.90 with the lane's
EMD-16721 template as drawn, +0.72 under the best of the other 47 axis orders and
flips). A
template that cannot be read is listed on the last page, and the corners stay empty.

What the reader relies on, read from real projects: a particle's class is
column 34 of the refined table of a channel that the next iteration still lists
(a dropped channel's table is never rewritten and still claims every particle),
and an iteration is finished once the project's log.txt says `iteration N ok`.
The next iteration's list of channels is not enough: Dynamo writes it while the
MRA plugin is still rewriting tables and averages, with every channel in it.
The code is `src/cryoet_pipeline/reports/dynamo_project.py` (reading) and
`src/cryoet_pipeline/reports/classification.py` (pages).
