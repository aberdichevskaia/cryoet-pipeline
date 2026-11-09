# Tomograms: `build-final-aligned-stack`, `reconstruct-tomogram`

> **Draft for review, 8 November 2026.**
>
> - Mark each entry by its ID: `R-2 ✓` if it is right, `R-3 ✗` if it is wrong
>   (say why), `R-4 ✎` if it is right but needs a change (say which).
> - Answer the questions at the end, a line each, and add what is missing.
> - Any language and any form: edit this file, comment on it, or reply by email.
> - Entries marked *for the maintainer* are about our code; skip them if you like.
>
> This box and the questions go once the review is in.

Kinds and evidence levels are explained at the top of [ctf.md](ctf.md). The
steps are in [pipeline_map.md](../pipeline_map.md), steps 9 and 11.

- **What the steps do.**
  - `build-final-aligned-stack` applies the fine alignment at the lane's
    `[binning] tomogram`.
  - `correct-ctf` corrects that stack ([ctf.md](ctf.md)).
  - `reconstruct-tomogram` back-projects it with IMOD's `tilt`, at the
    thickness the lane gives or the beads measured.
- **Per lane.** Both steps are labelled with the lane's variant, so lanes of
  one project keep their own stacks and tomograms.
- **Outputs.** The tomogram is written twice: a Zarr in ZYX order for the
  pipeline, and an IMOD `.rec` in YZX section order for 3dmod.
- **No denoising.** There is no denoising or missing-wedge restoration step yet.
  IsoNet2 is not usable.

## How a healthy run looks

- **Run times per series.** `build-final-aligned-stack` 8–21 s and
  `reconstruct-tomogram` 20–31 s at bin4 and bin8 (RyR1, HIV-1 Gag).
- **The reconstruction QC** is `pass`. That says only that the stack was
  CTF-corrected and that the volume holds finite values. It judges neither the
  thickness nor where the specimen sits (R-1).
- **The central slices** in `qc/<TS>/reconstruction/` show the specimen as a
  band in the XZ slice, with empty space above and below it.
- **The particle size.** The particle is a few tens of pixels across at the
  tomogram's binning: a ribosome at about 30 px (9.8 Å/px), Gag at 46 px
  (5.4 Å/px).

## Where to look first

1. **The central slices** in `qc/<TS>/reconstruction/`: XY, XZ and YZ.
2. **The `.rec`** in `tomograms/<TS>/`, opened in 3dmod, and the `tilt` log
   next to it.
3. **`python3 scripts/find_slab.py --tomogram <the tomogram's .zarr>`.** It
   prints where the specimen lies in Z, from the variance of each slice, and
   the `--pytom-search-z` value for picking.
4. **The tomogram's views.** The corrected stack's record lists the views it
   holds. Views CTF left out are missing (C-7 in [ctf.md](ctf.md)).

## Already checked by the code

- **A thickness on patch lanes.** A patch lane without `[reconstruct]
  thickness` is refused before the job runs, since patches do not measure the
  specimen's depth (R-1).
- **Matching angles.** The stack, its views and the solved tilt angles must
  match, or the step stops.
- **Finite values.** A reconstruction with non-finite values stops the step.

## Checks before trusting the result

1. **The specimen sits inside the volume** with empty space above and below.
   Check the XZ slice, or `find_slab.py` (R-1).
2. **The full field of view.** Every view of the aligned stack covers the full
   field, with no blank band along one edge (S-1).
3. **The hand**, before a tomogram is compared with one from another program or
   a map is compared with a published one (R-3).
4. **The binning.** The particle is a few tens of pixels across at this binning.

## Problems

### S-1. Aligned stacks with blank borders

**Kind:** silent · **Evidence:** measured

**Where it shows:** an aligned stack (`aligned/<TS>/`) opened in 3dmod shows
part of every view blank, about 30% on RyR1 at bin4. The CTF fit stack can be
affected too, and fits then run on less of the image.

**How to confirm:** compare the view size and content with the raw stack, and
check whether the stack was built before 16 October 2026.

**Causes, common to rare:**
1. `newstack -size` cropped the output. This was a bug in our code, fixed on
   16 October 2026.

**What to try:** rebuild the affected stack with `--overwrite`, then rerun what
was computed from it.

**What not to do:** work around wrong geometry with config values.

**When to stop and ask someone:** whenever an output is geometrically wrong while
its inputs look right. Suspect the code, and report it with the command, the
input and output shapes, and a view of the slices.

**Applies to:** stacks built before 16 October 2026.

**Source:** RyR1 bin4 lane; fixed 16 October 2026.

### R-1. The tomogram is mostly vacuum, or too thin to hold the specimen

**Kind:** silent · **Evidence:** measured

**Where it shows:** nowhere in the QC, which passes either way.
- Too thin: the XZ slice is filled edge to edge with specimen. `find_slab.py`
  then finds no edges, since there is no empty space to find them against.
- Too thick: the specimen fills only part of the volume. Template matching then
  searches the vacuum too, and picks land there ([picking.md](picking.md)).

**How to confirm:** the XZ central slice, and `find_slab.py` on the tomogram.

**Causes, common to rare:**
1. A thickness derived from the fiducials where they do not sample the
   specimen's depth. Gold beads on both faces do; patches do not. Patch lanes
   must now give the thickness.
2. A generous thickness, set to be safe on a first pass.

**What to try:**
1. A generous first pass that surely holds the specimen. 320 bin8 px held a
   177 px lamella on EMPIAR-11538.
2. Then set `[reconstruct] thickness` per series from `find_slab.py`, on a
   config copy (recipe B). Or restrict picking in Z with its `--pytom-search-z`
   value.

**What not to do:** trust a derived thickness on a lamella without looking at
the XZ slice.

**Applies to:** every version. Patch lanes have needed an explicit thickness
since 12 October 2026.

**Source:** EMPIAR-11538.
- On TS_0002, a thickness derived from patches gave 25 bin8 px, a 24 nm box for
  a FIB lamella. The reconstruction QC passed, and `find_slab.py` found the
  specimen filling 84% of the box.
- On other series the lamella filled about half of a generous volume, and
  roughly half the picks sat in vacuum.

### R-2. The fine alignment's thickness rests on a stray point

**Kind:** warning · **Evidence:** code

**Error text** (fine-alignment QC): `two-surface analysis sorted <n> points onto
the two surfaces, fewer than <m> on one of them; the recommended thickness is the
offset of a stray point, not a specimen dimension. Pass an explicit thickness …`

**Where it shows:** the fine-alignment QC warns. A bead lane without
`[reconstruct] thickness` would then reconstruct at that thickness.

**Causes, common to rare:**
1. Beads on one face only, or very few beads on one face.

**What to try:** set `[reconstruct] thickness` for that series, from the XZ
slice of a first, generous reconstruction.

**Applies to:** bead lanes without an explicit thickness.

**Source:** read in `backends/fine_alignment.py`.

### R-3. Our tomograms have the opposite hand to etomo's

**Kind:** reading results · **Evidence:** measured

**Where it shows:** a tomogram of ours and one made with etomo from the same
data are mirror images of each other.

**What it means:** our volume is the transpose of `tilt`'s output, which is
what `tilt -PARALLEL` writes, while etomo rotates that output instead. The two
are mirror images. On EMPIAR-11538, flipping one tomogram did not change the
template matcher's peaks.

**When it matters:** when comparing with an etomo tomogram, and when judging the
hand of a final map against a published one. *General:* template matching with
an asymmetric reference of the wrong hand finds weaker peaks.

**What not to do:** flip tomograms or maps to make them match something without
writing down that you did.

**When to stop and ask someone:** before deciding the hand of a map that will be
published.

**Source:** EMPIAR-11538 at MDC, 15 October 2026.

### R-4. The tomogram looks softer than a manual etomo reconstruction

**Kind:** reading results, open · **Evidence:** measured; the cause is not

**Where it shows:** the mid spatial frequencies of our tomogram are weaker than
those of an etomo reconstruction of the same series.

**Causes, common to rare:**
1. Not known. The alignment is suspected.

**What to try:** nothing by config yet. Note the series, and compare the
alignment residuals of both routes.

**When to stop and ask someone:** whenever picking or averaging results are to be
explained by this. It is not shown to be the cause of anything.

**Applies to:** the IMOD patch route on EMPIAR-11538, October 2026.

**Source:** EMPIAR-11538 ribosome lanes at MDC, October 2026, while diagnosing
the picking (P-2 in [picking.md](picking.md)).

### R-5. The reconstruction QC says CTF correction was not applied

**Kind:** warning · **Evidence:** code

**Error text:** `CTF correction was not applied`

**Causes, common to rare:**
1. The tomogram was reconstructed from a stack that `correct-ctf` did not
   write, for example in a reconstruction run outside the lane's chain.

**What to try:** check the lane's steps with `run_series.py --list-steps`, and
rerun from `correct-ctf` if it was skipped.

**Source:** read in `backends/reconstruction.py`.

## Questions for reviewers

Please answer in a line each; "don't know" is a useful answer.

- **Q-R1.** How do you choose the reconstruction thickness for lamellae, and
  for whole cells?
- **Q-R2.** At which binning do you look at tomograms first, and at which do
  you pick?
- **Q-R3.** Do you reconstruct with SIRT-like filtering or denoise before
  picking? Which, and why?
- **Q-R4.** How do you check the hand of your tomograms?
- **Q-R5.** Have you met a reconstruction problem that is not here? In a line
  each: what you saw, what it turned out to be, what helped.
