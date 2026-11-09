# Template: `prepare-template`

> **Draft for review, 8 November 2026.**
>
> - Mark each entry by its ID: `T-1 ✓` if it is right, `T-2 ✗` if it is wrong
>   (say why), `T-3 ✎` if it is right but needs a change (say which).
> - Answer the questions at the end, a line each, and add what is missing.
> - Any language and any form: edit this file, comment on it, or reply by email.
> - Entries marked *for the maintainer* are about our code; skip them if you like.
>
> This box and the questions go once the review is in.

Kinds and evidence levels are explained at the top of [ctf.md](ctf.md). The
step, its keys and its QC are in [pipeline_map.md](../pipeline_map.md), step 13.
Choosing an EMDB map has its own helper, `scripts/show_emdb.py`.

- **What the step does.** It makes the template and its mask for template
  matching with pytom-match-pick (0.14.0 at MDC): a map from EMDB
  (`[template] emdb_id`) or a local one (`source_map`), scaled to the
  tomogram's pixel size.
- **The rules every lane follows:**
  - the template is low-passed to the tomogram's Nyquist
    (`low_pass_angstrom`, 2 × the tomogram's pixel size), set in each lane;
  - the mask's radius is half the particle's widest diameter plus 3 px, with a
    1 px soft edge (the code's default);
  - the box is twice the diameter (the code's default).

  The diameter is `[pick] particle_diameter_angstrom`, the particle at its
  widest. The 3 px margin is a colleague's rule; pytom's tutorial uses about
  1 px plus the soft edge.
- **Per series.** Templates are written per series, in `templates/<TS>/`.

## How a healthy run looks

- **Run time per series.** 9–96 s.
- **The template QC** in `qc/<TS>/template/` has no warnings. The template
  looks like the particle, with the contrast of the tomogram, and the mask
  covers it with a little room.
- **Examples:**
  - RyR1 at 13.44 Å/px: a radius of 340 / 2 / 13.44 + 3 = 15.6 px, and a box of
    52 px;
  - a ribosome at 9.78 Å/px: a radius of 18 px with a 1 px soft edge, and a box
    of 60 px.

## Where to look first

1. **The template QC** and its warnings.
2. **The template panel of the report**: the central sections of template and
   mask.
3. **`templates/<TS>/`** in 3dmod: the template, the mask, and how the two sit
   together.

## Already checked by the code

- **Mask and box.** A mask that does not fit in the box is refused (T-3).
- **The mask against the density.** A mask much wider or narrower than the
  density, or a sphere around an elongated density, is warned (T-2).
- **The contrast.** A contrast that does not look as expected after inversion
  is warned (T-1).
- **The map's file.** A map that is not a cubic 3D volume with finite values is
  refused.

## Checks before trusting the result

1. **The map comes from other data than these tomograms.** If it does not,
   averages of the picks are no independent check of it (T-5).
2. **The map's pixel size** is right, so that the scaling to the tomogram is
   (D-1 in [data.md](data.md)).
3. **The hand** of the map against the tomograms (R-3 in
   [tomogram.md](tomogram.md)).
4. **No membrane or neighbour density inside the mask** (G-4 in
   [membranes.md](membranes.md)).

## Problems

### T-1. The template's contrast may not match the tomogram's

**Kind:** warning · **Evidence:** code

**Error text:** `template density does not look <expected> after applying
invert=<value>; check that the contrast matches the tomogram, since a mismatch
makes template matching quietly find nothing`

**Causes, common to rare:**
1. The map has the opposite contrast convention to the tomogram: density white
   in one, black in the other.
2. The check misreads an unusual map, such as a lattice or a map with much
   solvent noise.

**How to confirm:** open the template and the tomogram side by side in 3dmod.
Protein density should have the same shade in both.

**What to try:** if the contrast is wrong, tell the maintainer: the inversion
is not a lane key.

**What not to do:** run picking on a template whose contrast you have not
checked. Template matching does not fail with the wrong contrast, it just finds
nothing.

**Source:** read in `backends/template.py`.

### T-2. The mask cuts the particle, or is mostly background

**Kind:** warning · **Evidence:** code

**Error text:**
- `mask is <n> px across along <axis> but 95% of the density spans <m> px (<a>
  angstrom), and that is a floor; the mask cuts the particle, which lowers every
  score without failing`
- `mask is <n> px across along <axis> but 95% of the density spans only <m> px
  (<a> angstrom); the mask is mostly background, which dilutes the correlation`
- `density is not isotropic — x/y/z span <x>/<y>/<z> px — but the mask is a
  sphere; …`

**Causes, common to rare:**
1. `[pick] particle_diameter_angstrom` is not the particle's widest diameter.
2. The map holds more than the particle: a membrane, neighbours, a lattice.

**What to try:** measure the particle at its widest in the map, and set the
diameter to that. Set `mask_radius_px` and `box_size_px` by hand only for a map
with no single particle in it (T-4).

**What not to do:** shrink the mask to raise the scores.

**Source:** read in `backends/template.py`.

### T-3. The mask does not fit in the box

**Kind:** crash · **Evidence:** code

**Error text:** `mask diameter <n> px does not fit in the template box of <m> px
(<a> angstrom at <p> angstrom/pixel). Pass --template-box-size-px with at least
<k>, or check that the particle is really that large`

**What to try:** check the particle diameter first. If it is right, set
`[template] box_size_px` to at least the size the message names.

**Source:** read in `backends/template.py`.

### T-4. A lattice or helical map has no particle size

**Kind:** crash · **Evidence:** seen

**Error text:** `the density fills this map — measured extents <x>/<y>/<z> px in
a <n> px box — so there is no particle size to derive. This is what a lattice or
a helical reconstruction looks like; …`

**Causes, common to rare:**
1. The map is a lattice or a helix, as EMD-2638 (the immature HIV-1 Gag
   lattice) is. Nothing in it says where one particle ends.

**What to try:** set `[template] mask_radius_px` and `box_size_px` by hand, and
say in the lane file why.
- The Gag lane uses a 12 px sphere (130 Å) in the map's own 28 px box.
- A larger box would make pytom pad the lattice with zeros, a hard step just
  past the mask.

**When to stop and ask someone:** whether to pick a lattice as discrete
particles at all is still open for Gag.

**Source:** HIV-1 Gag, EMPIAR-10164 with EMD-2638, October 2026.

### T-5. The template comes from the same particles

**Kind:** silent · **Evidence:** measured

**Where it shows:** nowhere. The EMDB entry was made from the same data, or
from the same study, as the tomograms being picked.

**What it means:** this is fine for picking. But an average of the picks is
then no independent check of that map: it can look like the template because
it was found with it.

**How to confirm:** read the EMDB entry's sample and its EMPIAR link. EMPIAR
cross-references are not complete. EMPIAR-11830 lists only EMD-18306, while the
authors' own map of the same data is EMD-51802. `show_emdb.py` reads an entry
and says what it can about this.

**What to try:** if a map from other data exists, use it for the template, or
at least for the comparison.

**When to stop and ask someone:** before calling an average a confirmation of a
published map.

**Source:**
- EMPIAR-11538: EMD-16721 is the published map of the same data.
- RyR1 at MDC: EMD-54612 is probably from the same study; that is not
  confirmed.
- EMPIAR-11830: the cross-references miss the authors' map, and EMD-65574 is
  also made from 11830's data. Found on 7 November 2026.

### T-6. The EMDB map is not cached on the job's node

**Kind:** warning, then crash · **Evidence:** code

**Error text** (preflight): `warning: EMD-<id> not cached in <project>/emdb-cache;
prepare-template will try to download it, which needs outbound access from this
node`

**Causes, common to rare:**
1. The first run of a template, on a compute node without internet access.

**What to try:** put the map at `<project>/emdb-cache/emd_<id>.map` from a
machine that can download it, before submitting.

**Source:** `scripts/lane.sbatch`.

### T-7. The map's file states no pixel size

**Kind:** crash, *for the maintainer* · **Evidence:** code

**Error text:** `reference map has no usable voxel size: <path>; pass
template_input_voxel_size_angstrom explicitly`

**What to try:** the lanes have no key for it. Fix the map's header, or ask the
maintainer. Take the value from the map's EMDB entry.

**Source:** read in `backends/template.py`.

## Questions for reviewers

Please answer in a line each; "don't know" is a useful answer.

- **Q-T1.** How do you choose the map for a template: same species, same
  state, own average? How do you low-pass it?
- **Q-T2.** How much room do you leave between the particle and the mask, and
  how soft an edge?
- **Q-T3.** How do you handle a membrane protein whose map carries a piece of
  membrane (G-4)?
- **Q-T4.** For lattices such as Gag: do you pick them as particles, and with
  what template and mask?
- **Q-T5.** Have you met a template problem that is not here? In a line each:
  what you saw, what it turned out to be, what helped.
