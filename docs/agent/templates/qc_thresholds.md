# QC ranges for your data

For each number our QC reports, the table gives the threshold the code uses
today. Fill in what you would call good, doubtful and bad for your data, and leave
a row empty if you have no feel for it. Ranges depend on the sample and the
binning, so say which you have in mind.

Filled in by: · Sample type: · Binning: · Date:

| Step | Number | Unit | Today | Good | Doubtful | Bad | Comment |
|---|---|---|---|---|---|---|---|
| Coarse alignment | 95th percentile of the residual shift | preview px¹ | warning above 2 | | | | |
| Coarse alignment | largest residual shift | preview px¹ | failure above 5 | | | | |
| Bead tracking | share of views in which the beads are tracked | 0–1 | warning below 0.8, failure below 0.5 | | | | |
| Bead tracking | number of beads tracked | count | failure below 10 | | | | |
| Fine alignment | mean residual | nm | warning above 0.8, failure above 1.5 | | | | |
| Fine alignment | largest single-point residual | tracking px² | warning above 5, failure above 20 | | | | |
| Fine alignment | share of points dropped as outliers | 0–1 | limit 0.02 for beads; 0.08 used for a 5 × 5 patch grid | | | | |
| CTF estimation | views whose fit stops at the low-resolution end of the range | count | any gives a warning; up to 5 are left out at correction | | | | |
| CTF estimation | fitted resolution per view | Å | only the bound above is checked | | | | |
| CTF estimation | astigmatism | µm | not checked | | | | |
| Picking | candidates kept per tomogram | count | twice what pytom recommends | | | | |
| Extraction | particles dropped at the tomogram edge | share | warning above one half | | | | |
| Averaging | particles in the smaller half-set | count | warning below 10 | | | | |
| Averaging | particles correlating with the starting reference above 0.3 | share | warning below 10% | | | | |
| Averaging | FSC level that defines the resolution | — | 0.143 | | | | |
| Classification | particles changing class in the last iteration | share | warning above 10% | | | | |

¹ Pixels of a 16× binned preview of the tilt stack: one preview pixel is 16
pixels of the stack.
² Pixels of the binned stack the beads are tracked on (binning 4 in our example
settings).

## What else do you look at?

Numbers or pictures you check that are not in this table, and what tells you that
a step went well or badly:
