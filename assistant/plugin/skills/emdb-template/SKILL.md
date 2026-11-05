---
name: emdb-template
description: Propose one to three EMDB entries as the template for particle picking, each with its reasons, for the person to choose; check a chosen entry against the lane (box, pixel, low-pass, a membrane in the map, symmetry, independence). Use when the person does not know which map to pick with, or names an EMDB entry.
---

# EMDB: propose a template

The weighing follows `docs/roadmap.md`, section 23. The tool only reads: it
downloads no map. `prepare-template` fetches the chosen map when the lane runs.

## 1. What is to be picked

Ask the person for the molecule, the species, the state (ligands,
conformation) and the sample, unless a sample card in the form of
`docs/agent/templates/sample_card.md` gives them. The tomogram's pixel and the
particle's widest diameter come from the lane (`docs/agent/lane_questions.md`).

## 2. Search

```bash
python3 scripts/show_emdb.py search '<query>' [--method subtomogramAveraging] [--empiar <id>]
```

The query is in Solr syntax, such as `ryanodine AND receptor`. The tool counts
the entries by method. It lists subtomogram averages first, whole maps before
maps of a region, and newer entries before older ones. Its flags are read from
the titles only. `--empiar` marks the entries that EMPIAR entry
cross-references, which are maps made from the data being processed.

## 3. Read the likely entries

```bash
python3 scripts/show_emdb.py entry EMD-<id> --pixel <tomogram A/px> --diameter <A> [--empiar <id>]
```

Read two to five entries, not just their titles. Each check says what
`prepare-template` will do with the map: pad it up to the template box or keep
a larger box, resample it, and low-pass it.

## 4. Weigh, in this order

- The same molecule and species first, then the same state when the sample
  has one.
- Whole maps over maps of a region.
- For tomograms, in situ subtomogram averages over isolated single-particle
  maps. A map that holds a membrane matches every membrane unless the mask
  leaves the membrane out. Say so, and suggest a look in ChimeraX.
- The box and pixel against the lane's: the map must hold the whole particle.
- Resolution matters little, since the template is low-passed to about the
  tomogram's Nyquist.
- Independence: a map made from the same data is a fine template, but it is
  no independent check of the result. Say so.

## 5. Propose, and let the person choose

Propose one to three entries. Give each one its reasons and what speaks against
it. The person chooses. Write `[template] emdb_id` with the entry, and the
person's choice as its source.

If no map fits, say so. A map simulated from a PDB model is the fallback, and
this pipeline has no step for it yet.
