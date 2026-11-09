# Membranes: `segment-tomogram`

> **Draft for review, 8 November 2026.**
>
> - Mark each entry by its ID: `G-1 ✓` if it is right, `G-2 ✗` if it is wrong
>   (say why), `G-3 ✎` if it is right but needs a change (say which).
> - Answer the questions at the end, a line each, and add what is missing.
> - Any language and any form: edit this file, comment on it, or reply by email.
> - Entries marked *for the maintainer* are about our code; skip them if you like.
>
> This box and the questions go once the review is in.

Kinds and evidence levels are explained at the top of [ctf.md](ctf.md). The
step is in [pipeline_map.md](../pipeline_map.md), step 12.

- **What the step does.** MemBrain-seg segments the membranes of each tomogram
  on a GPU. At MDC on 5 November 2026 it was MemBrain-seg 0.0.10, with the
  checkpoint `MemBrain_seg_v10_alpha.ckpt`.
- **Only some lanes run it:** those with a `[segment]` section.
- **What uses it.** Picking can be restricted to the membranes it finds:
  `[pick] use_segmentation_mask` and `segmentation_mask_dilation_px`
  ([picking.md](picking.md)).

## How a healthy run looks

- **Run time per series.** 233–520 s (RyR1 at bin4, HIV-1 Gag at bin8).
- **The segmentation** follows the membranes visible in the tomogram, with
  little else labelled.
- **The QC** in `qc/<TS>/segmentation/` is `pass`. It always is (G-3): it
  records the share of voxels labelled membrane, `foreground_fraction`, and
  judges nothing.

## Where to look first

1. **The job's preflight**, in the Slurm log. It checks `$MEMBRAIN_EXE`,
   `$MEMBRAIN_CKPT` and the GPU (G-1, G-2).
2. **`python3 scripts/show_failure.py <lane config> --step segment-tomogram`.**
3. **The segmentation** in `segmentations/<TS>/`, opened over the tomogram in
   3dmod or napari.

## Already checked by the code

- **Before the job runs anything.** The preflight checks that MemBrain-seg is
  executable, that the checkpoint is a file, and that the GPU's compute
  capability is 7.5 or newer.
- **The output.** It must have the tomogram's shape and hold finite values.

## Checks before trusting the result

1. **Look at the segmentation over the tomogram** before restricting picking
   to it. The QC does not judge it (G-3).
2. **`foreground_fraction`** against the other series of the lane. A series far
   from the rest is worth a look. *General.*

## Problems

### G-1. MemBrain-seg cannot run on this GPU

**Kind:** crash · **Evidence:** seen

**Error text:**
- preflight: `MemBrain-seg needs compute capability >= 7.5, this GPU has <cap>`
- before that check: a PyTorch "no kernel image is available" error

**Causes, common to rare:**
1. A V100 (compute capability 7.0). The installed PyTorch has no code for it.

**What to try:** give the stage that runs `segment-tomogram` a newer GPU, in its
`sbatch` list.

**When to stop and ask someone:** when the partition has no GPU of 7.5 or newer.

**Applies to:** every version.

**Source:** Munich cluster setup, October 2026.

### G-2. The preflight rejects `MEMBRAIN_CKPT` or `MEMBRAIN_EXE`

**Kind:** crash · **Evidence:** seen

**Error text** (preflight):
- `MEMBRAIN_CKPT is not a file: '<path>'`
- `MEMBRAIN_EXE is not an executable: '<path>'`

**Causes, common to rare:**
1. A path typed wrong. In Munich the checkpoint's path had lost its leading
   `/`, so it was read relative to the repository.
2. The variable is not set in the shell the job was submitted from (C-4 in
   [ctf.md](ctf.md)).

**What to try:** `ls -l "$MEMBRAIN_CKPT" "$MEMBRAIN_EXE"` in the submitting
shell, fix the variable, and resubmit (recipe A).

**Applies to:** every version.

**Source:** Munich, October 2026, jobs 41075010 and 41075011.

### G-3. The segmentation QC passes whatever the segmentation looks like

**Kind:** silent · **Evidence:** code

**Where it shows:** nowhere. The QC records `foreground_fraction` and is always
`pass`.

**What it means:** an empty segmentation, or one that labels much more than
membranes, goes on to picking unremarked. With
`[pick] use_segmentation_mask = true`, picking then searches nowhere, or almost
everywhere.

**What to try:** look at the segmentation of one series or more before relying
on it, and compare `foreground_fraction` across the lane's series.

**When to stop and ask someone:** when the segmentation misses membranes that
are plain in the tomogram.

**Applies to:** every version.

**Source:** read in `backends/segmentation.py`.

### G-4. The template carries membrane, and the search is restricted to membranes

**Kind:** silent · **Evidence:** seen

**Where it shows:** the template panel of the report. In the central XZ and YZ
sections, a curved membrane band runs across the mask, under the protein's
transmembrane part.

**What it means:** inside the mask the membrane is matched along with the
protein. A search restricted to membranes then gets a share of its score at
every membrane position, particle or not.

**What to try:** nothing is built for it yet. Keep it in mind when reading the
scores of a membrane protein, and tell the maintainer about any template with a
membrane in it.

**Applies to:** templates of membrane proteins taken from maps of native
membranes.

**Source:** RyR1 (EMD-54612), seen in ryr1-bin4's report, October 2026. Not
handled.

## Questions for reviewers

Please answer in a line each; "don't know" is a useful answer.

- **Q-G1.** At which pixel size does MemBrain-seg work best on your data? Do
  you bin before segmenting?
- **Q-G2.** How do you judge a segmentation good enough to restrict picking to
  it?
- **Q-G3.** How far from the membrane do you let picking search
  (`segmentation_mask_dilation_px`), for which proteins?
- **Q-G4.** Have you met a segmentation problem that is not here? In a line
  each: what you saw, what it turned out to be, what helped.
