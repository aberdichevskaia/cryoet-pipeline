# Particles and averages: `extract-particles`, averaging, classification, FSC

> **Draft for review, 8 November 2026.**
>
> - Mark each entry by its ID: `V-3 ✓` if it is right, `V-4 ✗` if it is wrong
>   (say why), `V-5 ✎` if it is right but needs a change (say which).
> - Answer the questions at the end, a line each, and add what is missing.
> - Any language and any form: edit this file, comment on it, or reply by email.
> - Entries marked *for the maintainer* are about our code; skip them if you like.
>
> This box and the questions go once the review is in.

Kinds and evidence levels are explained at the top of [ctf.md](ctf.md).
Extraction is step 16 of [pipeline_map.md](../pipeline_map.md). Averaging comes
after the lane, as the recipe `average`. Multi-reference classification (MRA)
and its report are in
[dynamo_classification.md](../../dynamo_classification.md); consensus halves
are in [halves_from_runs.md](../../halves_from_runs.md).

- **Extraction** cuts a box around each pick from the lane's tomogram
  (`[extract] box_size_px`, about twice the particle's diameter, even).
- **Averaging** (`submit_recipe.py average`) re-extracts each series at its
  score threshold and refines the particles in two independent halves.
- **Classification** (MRA) runs our port of Dynamo's MRA (`dynamo-search`, the
  default backend) on GPUs, or Dynamo itself on MDC's CPUs. Its runs live in
  `<project>/mra/<NAME>`, and every run writes a PDF report.

## How a healthy run looks

- **Extraction.** 17 s per series (HIV-1 Gag), with no more than a few
  particles dropped at the tomogram's edges.
- **HIV-1 Gag at bin8.** 618 particles from two tomograms. The FSC between the
  two tomograms' averages, with a soft mask, first falls below 0.143 at 16.4 Å
  and last at 14.1 Å; it is read as about 14 Å (V-6).
- **MRA run times.** On RyR1 at bin4 an iteration took about 28 min on GPUs,
  20 of them averaging on the CPU, before averaging ran on threads
  (6 November 2026).
- **The averaging QC** has no warnings (V-2).

## Where to look first

1. **The averaging QC** in `qc/<scope>/subtomogram_averaging/`, and the maps and
   FSC in `averages/<scope>/`.
2. **The MRA report** (PDF) beside the run, with the classes, the particles per
   class, and the decoy page when decoys were run.
3. **The job log of an MRA run.** It gives the time of each iteration, and a
   refusal when the run's folder is in use (V-11).
4. **The extraction QC** in `qc/<TS>/particle_extraction/`.

## Already checked by the code

- **One box size.** Particles of different box sizes are not averaged together
  (X-2).
- **Half-sets.** An empty half-set, or a half-set label other than 1 or 2, is
  refused.
- **One job per run folder.** A second job on an MRA run's folder is refused,
  naming the job that holds it (V-11).
- **Search settings.** Options that need a missing wedge, or that cannot be
  combined, are refused with the reason.

## Checks before trusting the result

1. **What the FSC compares.** Halves split before alignment (gold standard),
   or halves of a joint fit (V-3)?
2. **The mask's share of the FSC.** Measure it with phase randomisation before
   reading a masked FSC (V-5).
3. **The CTF zeros of the tomograms**, before reading dips and recoveries in
   the FSC (V-6).
4. **A class that looks like the template proves nothing by itself.** Decoys
   say whether a class collects junk (V-8).
5. **Classes that differ by view direction** may be split by the missing wedge,
   not by structure (V-7).

## Problems

### X-1. Particles dropped at the tomogram's edge

**Kind:** warning · **Evidence:** code

**Error text** (extraction QC): `<n>/<m> particles dropped for being too close
to the tomogram edge for the requested box size`

**What it means:** a box that would reach past the tomogram is dropped. The
extraction QC warns when more than half are dropped, and fails when none
survive.

**What to try:**
- If many are dropped, check the box size against the particle (about twice its
  diameter) and the tomogram's thickness (R-1 in [tomogram.md](tomogram.md)).
- `[extract] pad_edges` keeps them, padded.

**Source:** read in `backends/subtomogram.py` and `pipeline_map.md`. On HIV-1
Gag, 657 picks were kept after filtering and 618 were averaged; whether the 39
were dropped at the edges is not checked.

### X-2. Particles of two box sizes in one average

**Kind:** crash · **Evidence:** code

**Error text:** `<stack> has a box of <n> px but the first stack has <m> px;
particles can only be averaged together at one box size, so re-extract them with
a single --box-size-px`

**What to try:** re-extract every series with the same `[extract]
box_size_px`.

**Source:** read in `backends/subtomogram.py`.

### X-3. Particles extracted before 25 September 2026 sit half a voxel off

**Kind:** silent, *for the maintainer* · **Evidence:** measured

**Where it shows:** nowhere by itself. Averages of old extractions are slightly
blurred, and a comparison with Dynamo finds a gain where the two should agree.

**Causes, common to rare:**
1. Our extraction put each box's centre half a voxel off. A half-voxel offset
   in pytom's coordinates was found and fixed at the same time.

**What to try:** extract again particles extracted before 25 September 2026.

**Source:** found in September 2026 by comparing our averaging with Dynamo's.

### V-1. Few particles agree with the starting reference

**Kind:** warning · **Evidence:** code

**Error text** (averaging QC): `fewer than 10% of particles correlate with the
initial reference above <c> before refinement - check the rot/tilt/psi convention
and initial reference box size/pixel spacing`

**Causes, common to rare:**
1. The poses use another Euler convention than the averaging expects.
2. The starting reference has another box size or pixel size than the particles.
3. The picks are mostly not the particle (P-2 in [picking.md](picking.md)).

**What to try:** check the reference's box and pixel size against the particle
stack first. Then compare one particle at its pose with the reference in 3dmod
or ChimeraX.

**Source:** read in `backends/subtomogram.py`.

### V-2. Reading the averaging QC's other warnings

**Kind:** reading results · **Evidence:** code

**Error text and meaning:**
- `the smaller half-set holds <n> particles, fewer than <m>; …`: the FSC beside
  the map is drawn from a wide distribution. Do not read it as a resolution.
- `the two half-sets never disagreed enough to cross <t>, so no resolution was
  measured; …`: with few particles, the curve is dominated by what the halves
  share, not by the map's quality.
- `refinement did not improve the mean correlation: …`: the search found nothing
  better than the orientations it was given.
- `minimum final correlation <c> is negative; some particles never converged
  onto the average`.
- `<f> of particles changed class in the last iteration; classification has not
  converged, so run more iterations`.

**What to try:** read each as it says. None of them stops the run.

**Source:** read in `backends/subtomogram.py`.

### V-3. Consensus halves give consistency, not resolution

**Kind:** reading results · **Evidence:** measured

**Where it shows:** halves split after a joint classification or refinement
give an FSC that looks like a resolution.

**What it means:** the particles keep the poses of the joint fit, so the two
halves share its noise. Their FSC says how consistent the map is with itself.
It is not a resolution.

**How to confirm:** how far the particles moved after the split. On RyR1 the
median was 1.7° and 1.6°, about one search step of the last round.

**What to try:** call it consistency in reports. For a resolution, use halves
split before alignment, refined independently (V-4).

**Source:** RyR1 at MDC: consensus halves at bin8 gave 0.143 at 33 Å, and at
bin4 24.7 Å masked (37.3 Å unmasked), October and November 2026.

### V-4. Gold-standard halves lose their alignment in a global first round

**Kind:** silent · **Evidence:** measured; the cause is likely, not shown on the
data

**Where it shows:** after a global first round, the halves' orientations agree
with the earlier poses no better than random rotations, and their averages are
noise. Their FSC still gives a number.

**How to confirm:** compare the poses after the round with the earlier ones.
For random rotations the median turn is about 132°.

**Causes, common to rare:**
1. The global round shrinks the particles to a small box without low-passing
   first, as Dynamo does. What lies above the small box's Nyquist folds into
   the shells the round reads. On white noise this left 26.9 times the noise
   from a 102 px box (bin4) and 2.6 times from a 52 px box (bin8).

**What to try:** the port's `--antialias` option (`ANTIALIAS=1` in
`scripts/mra_port_run.sbatch`, since 7 November 2026) low-passes before
shrinking. On a synthetic test it recovered the orientations at a noise level
where the plain round did not. Whether it rescues the RyR1 halves is not known
yet.

**What not to do:** report the FSC of such halves as a resolution. RyR1's
34.5 Å was not one.

**When to stop and ask someone:** before a global round on a fine box.

**Source:** RyR1 gold halves at bin4 (ryr1-gold-bin4), 7 November 2026: median
turns of 125–127°, against 132° for random rotations.

### V-5. The mask makes the FSC better than the map

**Kind:** reading results · **Evidence:** measured

**Where it shows:** the masked FSC sits above the unmasked one, or its tail
rises near Nyquist.

**Causes, common to rare:**
1. A binary mask: its sharp edge correlates between the halves at every
   resolution.
2. A tight mask on a small particle.

**What to try:**
1. Use a soft mask.
2. Measure what the mask adds with phase randomisation: randomise the phases
   beyond a cut-off, mask, and compute the FSC again. On HIV-1 Gag and RyR1 the
   soft mask added about 0 at the crossing.

**What not to do:** read a resolution off a masked FSC without that control.

**Source:**
- HIV-1 Gag at bin8, October 2026: a binary mask raised the tail, while a soft
  one added −0.002 to 0.000 in the 15–10 Å band.
- RyR1 at bin4, November 2026: the mask added about 0 at 24.7 Å.

### V-6. Dips in the FSC at the tomograms' CTF zeros

**Kind:** reading results · **Evidence:** measured

**Where it shows:** the FSC falls and recovers more than once, so it crosses
0.143 at several resolutions.

**What it means:** at a CTF zero a tomogram carries no signal. With two
tomograms at different defocus, the FSC dips where their zeros fall, and
recovers between them.

**How to confirm:** compute the CTF zeros of each tomogram from its mean defocus
in the estimation QC, and compare them with the dips.

**What to try:** state before reading which crossing counts, the first or the
last. Then read it and say why.

**Source:** HIV-1 Gag at bin8, October 2026. TS_01 (3.93 µm) and TS_43
(2.56 µm) share a zero at about 16 Å, where the FSC first falls; the 18.2 Å
peak and the 15.0–14.4 Å recovery lie between zeros. The last fall, about 14 Å,
was read as the resolution. Phase flipping was checked on the same curve (C-15
in [ctf.md](ctf.md)).

### V-7. Classes split by the missing wedge

**Kind:** reading results · **Evidence:** measured

**Where it shows:** MRA classes differ mainly by the direction they are seen
from: one holds top views, another side views. Their averages differ the way
the missing wedge distorts them.

**What it means:** a class can collect particles because their missing wedge
lies alike, not because their structure differs.

**How to confirm:** the distribution of each class's orientations relative to
the tilt axis.

**What not to do:** read such classes as conformations.

**Source:** RyR1 MRA at MDC, October 2026.

### V-8. A class that looks like the template is no evidence

**Kind:** reading results · **Evidence:** measured

**Where it shows:** classes, even those seeded with featureless blobs, take the
template's shape within one iteration.

**What it means:** pytom chose every pick and pose by its match to the
template, so any pick, false or not, averages towards it at those poses.

**How to confirm:** a decoy control ([dynamo_classification.md](../../dynamo_classification.md),
"The decoy control"). It puts boxes the classification does not know to be junk
into the same run, and reports each class's share of them.
- Decoys at random positions mostly went to the noise class (86%, against 0.5%
  of real picks).
- Decoys chosen by template matching on phase-randomised tomograms were not
  separated from real picks.

**What not to do:** take the template's shape in a class as proof that its
particles are real.

**Source:** EMPIAR-11538 ribosomes at MDC, October 2026.

### V-9. A class with no particles is gone for good

**Kind:** reading results · **Evidence:** measured

**Where it shows:** a run that started with three classes reports fewer.

**What it means:** in MRA a class that no particle chooses in an iteration is
dropped, and its reference is not seeded again. A white-noise seed got no
particle in the first iteration, while a blob kept one.

**Source:** Dynamo 1.1.511 on the Mac, 16 October 2026.

### V-10. Dynamo's parallel pools fail when two jobs start together

**Kind:** crash · **Evidence:** measured

**Error text:** MATLAB's pool reports "Unknown type", and every later pool fails
with it too.

**Causes, common to rare:**
1. The MATLAB Runtime keeps every pool's job files in one folder under the home
   directory. Two pools started together wrote one job's record wrongly.

**What to try:**
- Move the broken job's files aside; the job's message lists them.
- Start each Dynamo job a quarter of an hour after the previous one:
  `--dependency=after:<job>+15`.

**Applies to:** Dynamo itself, not our port.

**Source:** Dynamo 1.1.511 on the Mac, October 2026.

### V-11. An MRA run's folder is held by another job

**Kind:** refused · **Evidence:** measured

**Error text:** `REFUSED: <run folder> is taken by job <id> …; this job ran
nothing`

**Causes, common to rare:**
1. Another job is computing, or resuming, the same run. Two jobs in one run's
   folder would overwrite each other's iterations.

**What to try:** wait for the named job, or cancel it if it should not run. Then
submit again.

**Applies to:** since 30 October 2026, for the port and for Dynamo jobs.

**Source:** RyR1 at MDC, 30 October 2026: two resume jobs had run at once in one
folder, which led to the lock. The lock was checked on max the same day.

### V-12. Dynamo on a GPU behaves differently

**Kind:** reading results, *for the maintainer* · **Evidence:** seen

**What it means:** Dynamo's GPU search, unlike its CPU one:
- needs particles in EM format;
- needs a project with one core;
- resizes only by binning in powers of two;
- uses a symmetric missing wedge.

**What to try:** use the port's GPU search (`dynamo-search`, the default) unless
Dynamo itself is the point.

**Source:** Dynamo GPU runs, October 2026.

## Questions for reviewers

Please answer in a line each; "don't know" is a useful answer.

- **Q-V1.** How do you split particles into halves, and when: before the first
  alignment, or after a consensus?
- **Q-V2.** What mask do you use for FSC: shape, softness, distance from the
  density? Do you correct for it?
- **Q-V3.** How do you decide that a class is a real conformation and not a view
  or junk class?
- **Q-V4.** Which crossing do you read when the FSC crosses 0.143 more than
  once?
- **Q-V5.** Have you met an averaging or classification problem that is not
  here? In a line each: what you saw, what it turned out to be, what helped.
