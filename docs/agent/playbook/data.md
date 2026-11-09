# Data: the folder, `cryoet init`, `import-tilt-stack`

> **Draft for review, 8 November 2026.**
>
> - Mark each entry by its ID: `D-4 ✓` if it is right, `D-5 ✗` if it is wrong
>   (say why), `D-6 ✎` if it is right but needs a change (say which).
> - Answer the questions at the end, a line each, and add what is missing.
> - Any language and any form: edit this file, comment on it, or reply by email.
> - Entries marked *for the maintainer* are about our code; skip them if you like.
>
> This box and the questions go once the review is in.

Kinds and evidence levels are explained at the top of [ctf.md](ctf.md). The
steps are in [pipeline_map.md](../pipeline_map.md), steps 0 and 1′. Reading a
new folder is in [data_folders.md](../data_folders.md), and where each lane key
comes from is in [lane_questions.md](../lane_questions.md).

- **Two kinds of lane.**
  - A frames lane starts from movies, one per tilt: MRC, TIFF or EER. On
    submission, `cryoet init` reads the folder and writes one manifest per
    series. Motion correction comes next ([motion.md](motion.md)).
  - A stack lane starts from assembled tilt stacks and their mdocs, through
    `import-tilt-stack`.
- **`init` runs once per project.** It writes an empty registry. That is why a
  frames project never takes new series: a new project is the way (D-5).
- **The tilt metadata comes from one of two sources:** the mdocs
  (`[data] mdocs`) or the file names (`[data] frames_pattern`), never both.

## How a healthy run looks

- For each series, `init` prints the number of tilts and the angle range it
  found. The numbers match the acquisition:
  - RyR1 at MDC: 29 tilts, −42° to +42° in 3° steps;
  - EMPIAR-11830 Position_1: 41 tilts, −75° to +45° in 3° steps, 728 EER frames
    each.
- The manifest's raw pixel size matches the detector and the published data
  (D-1). It is on the first line of `show_series.py`. Examples:
  - K3 super-resolution TIFFs (RyR1): 0.84 Å, from the TIFF tags;
  - Falcon 4 EER (EMPIAR-11830): 1.9037 Å from the EER metadata, while the
    mdocs say 1.82–1.97 by microscope.
- The dry run lists no `[data]` problem.

## Where to look first

1. **Before a lane exists:** `python3 scripts/show_folder.py <folder>` reads a
   folder and writes nothing. It shows the tree, the fields of the file names
   and what three files hold. For an EMPIAR entry use
   `python3 scripts/show_empiar.py`, which compares the mdocs with the frames.
2. **`init`'s own printout:** series, tilts and angle range per series.
3. **`manifests/<TS>.json`:** tilts, angles, raw pixel size, and `notes`, where
   `init` says what it could not read.
4. **The dry run** of `submit_lanes.py`, which checks the `[data]` keys and
   folders before anything runs.

## Already checked by the code

- **The `[data]` keys and folders.** The dry run refuses missing `[data]` keys,
  missing folders, and a stack lane's missing stack or mdoc (D-3).
- **The frames pattern.** `init` refuses a pattern that matches nothing, or that
  leaves files unmatched, and two movies that share a tilt (D-4).
- **The movies.** `init` refuses a movie it cannot read, an EER whose frame
  count differs from its own metadata, and pixel sizes that differ between
  movies (D-6, D-7).
- **Adding series.** A frames project that would have to be started again to
  add series is refused (D-5).
- **The tilt axis of Tomography mdocs.** The tilt axis is not derived from a
  Tomography (Tomo5) mdoc, because the usual rule is 90° off there (D-8).

## Checks before trusting the result

1. **Tilts and range** per series, against the acquisition or the instrument
   card. A missing tilt or a shifted range means the names or the mdocs were
   read wrongly (D-2).
2. **The pixel size**, against the detector and any published map of the same
   data (D-1). Everything downstream scales with it, defocus included.
3. **The gain reference** belongs to the session the movies were recorded in
   (D-9).
4. **The tilt axis angle**, when the lane sets it by hand (D-8).

## Problems

### D-1. The pixel size in the headers is not the published one

**Kind:** silent · **Evidence:** measured

**Where it shows:** the manifest's raw pixel size, on the first line of
`show_series.py`, differs from the pixel size of a published map of the same
data. Templates and averages then come out at a slightly wrong scale.

**How to confirm:** compare the manifest with the deposition, the paper and the
EMDB map of the same data. A map deposited at another pixel size is the
strongest evidence.

**Causes, common to rare:**
1. The headers carry the nominal pixel size, not the calibrated one.

**What to try:** ask a person which value is right before changing anything.
For a frames lane the pixel size is read at `cryoet init`
(`[data] pixel_size_angstrom`), so a different value means a new project.

**What not to do:**
- Override the value silently.
- Combine particles from projects processed at different pixel sizes.

**When to stop and ask someone:** always: choosing the value is a decision
about the data.

**Applies to:** every version.

**Source:** EMPIAR-11538. The stack headers say 1.223 Å (nominal); EMD-16721,
the published map of the same data, is at 1.188 Å. The manual etomo processing
used 1.188 and the lanes 1.223. Found on 15 October 2026.

### D-2. Angles read wrongly from file names

**Kind:** silent · **Evidence:** measured

**Where it shows:** for each series, `cryoet init` prints the number of tilts and
the angle range it found. The numbers do not match the acquisition: tilts are
missing, the range is shifted, or two series are merged.

**How to confirm:** compare the printout with the tilt scheme on the instrument
card, and with the file list.

**Causes, common to rare** (the traps of the RyR1 folder, roadmap section 22):
1. Signed angles such as `+03.0`.
2. Zero tilt written as `-00.0`.
3. Leading zeros in the series id.
4. A tilt number that follows the angle rather than the acquisition order.

**What to try:** test the `frames_pattern` regular expression against the real
file names before `init`, with `show_folder.py --pattern`. Then check the count
and the range per series. Start from the main naming schemes and their patterns
in [data_folders.md](../data_folders.md).

**What not to do:** start the lane before the `init` printout has been checked.

**When to stop and ask someone:** when the names are ambiguous, for example when
it is unclear which number is the series and which the tilt. List the readings
rather than pick one.

**Applies to:** every version.

**Source:** the RyR1 dataset at MDC, named like `tomogram_039_001_-42.0.tif`: 29
tilts, −42° to +42° in 3° steps.

### D-3. The dry run refuses the lane's `[data]`

**Kind:** refused · **Evidence:** code

**Error text** (dry run):
- `[data] needs exactly one of mdocs and frames_pattern; init reads the tilt
  metadata from one of them`
- `[data] frames is not set; init needs it for a new project`
- `[data] frames: no directory <path>` (the same for `mdocs`)
- for a stack lane: `[data] mdoc: no file <path>` or `[data] stack: no file
  <path>`

**Causes, common to rare:**
1. A key is missing or misspelt, or a path is relative to the wrong folder.
   Paths in a lane are relative to the repository root, because the job runs
   from there.
2. Both `mdocs` and `frames_pattern` are given, or neither.

**What to try:** fix the key or the path, and run the dry run again. A stack
lane writes `{series}` in its paths where the series id goes.

**Applies to:** every version.

**Source:** read in `scripts/submit_lanes.py`.

### D-4. The frames pattern does not fit the file names

**Kind:** crash · **Evidence:** code; the traps behind it are measured (D-2)

**Error text** (`cryoet init`):
- `no file in <folder> matches the frames pattern <pattern>; first names: <names>`
- `<n> files in <folder> do not match the frames pattern <pattern>, e.g.
  <name>. Fix the pattern, or move what is not a frame out of the folder;
  series are chosen by their ids, not by the pattern`
- `frames pattern needs the named groups <groups>; missing <group>, …`
- `<name>: the tilt group must be an integer and the angle group a number, got
  tilt=<value>, angle=<value>`
- `<series>: two movies share <field> <value>: <name> and <name>`
- `no movies of series <id> in <folder>; the frames pattern finds the series
  <ids>`

**Causes, common to rare:**
1. The pattern was written from one example name and misses a variant: signed
   angles, `-00.0`, angles without a decimal part, a prefix with several parts
   (D-2, [data_folders.md](../data_folders.md)).
2. Other files sit in the folder: gain references, logs, `.mdoc` files next to
   the movies.
3. The series id in `[run] series` is written differently from the folder, for
   example without its leading zeros.

**What to try:** `show_folder.py <folder> --pattern '<regex>'` shows what the
pattern matches and what it leaves out, without running `init`.

**What not to do:** loosen the pattern until everything matches without looking
at what it now matches.

**When to stop and ask someone:** when two readings of the names both fit.

**Applies to:** every version.

**Source:** read in `ingest/frame_names.py`.

### D-5. New series for an existing frames project

**Kind:** refused · **Evidence:** code

**Error text** (dry run): `<project> already exists but has no manifest for
<series>, and init would erase the project to add them; use a new project for
them`

**Causes, common to rare:**
1. `init` writes a new, empty registry. Running it on a project that already
   has one would throw away every series processed so far.

**What to try:** a new lane with a new `[run] project` for the new series. Data
from two projects are combined at averaging, not in one project.

**What not to do:** delete the project's registry or manifests to get past the
refusal.

**Applies to:** every version.

**Source:** read in `scripts/submit_lanes.py`.

### D-6. A movie cannot be read, or is cut short

**Kind:** crash · **Evidence:** code; the EER details are measured

**Error text** (`cryoet init`):
- `<file>: its TIFF structure cannot be read (<reason>); an incomplete copy?`
- `<file>: <n> EER frames on disk, its metadata declares <m>; an incomplete copy?`
- `<file>: expected a multiframe TIFF movie, found <n> page`
- `<file>: TIFF pages differ in shape: <shapes>`
- `<file>: no EER acquisition metadata (TIFF tag 65001); not an EER movie`
- `unknown movie format <suffix> (known: <formats>): <file>`

**Causes, common to rare:**
1. A download or a copy that stopped early. An EER file keeps its directory at
   its end, so a cut file does not open at all.
2. Files that are not movies: single integrated images, or EER files that hold
   an integrated page as well.

**What to try:**
1. Compare the file sizes with the source. For EMPIAR, `show_empiar.py` checks
   the listing against the numbers the entry declares.
2. Fetch the file again. The downloader keeps a `.part` file when the size does
   not match, and retries when EBI refuses for a while (HTTP 403 and the like).

**What not to do:** remove the tilt from the series without saying so in the
lane file.

**Applies to:** every version; EER since 5 November 2026.

**Source:** read in `movies.py` and `empiar.py`. On EMPIAR-11830 the EER
directory sits at the file's end; EBI answered HTTP 403 during the download of
5 November 2026.

### D-7. No pixel size, or two

**Kind:** crash · **Evidence:** code

**Error text** (`cryoet init`):
- `<file> has no pixel size in its header; give the raw pixel size explicitly`
- `<file>: TIFF X and Y resolution give different pixel sizes, <a> and <b>
  angstrom`
- `inconsistent raw pixel spacing in the movie headers; <file> has <a>
  angstrom, <file> has <b> angstrom`

**Causes, common to rare:**
1. TIFFs written without resolution tags.
2. Movies from two sessions or two detector modes in one folder.

**What to try:** set `[data] pixel_size_angstrom` to the calibrated pixel size
of the movies (D-1). Split movies from different sessions into different
series or projects.

**When to stop and ask someone:** when nobody knows the calibrated value.

**Applies to:** every version.

**Source:** read in `movies.py`.

### D-8. The mdocs are not SerialEM's

**Kind:** crash or silent · **Evidence:** measured

**Error text:**
- `init`: `KeyError: 'Binning'`, or the name of another field, when an mdoc
  lacks a field `init` needs. Every tilt section needs `TiltAngle`,
  `SubFramePath`, `NumSubFrames`, `PixelSpacing` and `Binning`.
- coarse alignment: `tilt-axis angle is unavailable: the manifest has no
  SerialEM RotationAngle for every tilt (its notes say why when an mdoc was
  read); provide tilt_axis_angle_deg explicitly ([align] tilt_axis_angle in a
  lane)`

**Where it shows:** `init` stops on a missing field, or alignment asks for the
tilt axis.

**Causes, common to rare:**
1. Tomography (Tomo5) mdocs. They are named `<id>.mdoc`, say `NumSubFrames = 1`
   for EER movies of hundreds of frames, and use `RotationAngle` differently.
   - `init` takes the EER frame count from the movies.
   - It keeps the mdoc's `RotationAngle` in the manifest's metadata only. The
     usual rule, axis = `RotationAngle` − 90°, is 90° off for these mdocs. On
     EMPIAR-11830 the axis lay along `RotationAngle` itself (mod 180°) in four
     sessions of four, with its direction opposite in two.
2. mdocs written by a script that leaves fields out.

**What to try:**
- For Tomo5 data, set `[align] tilt_axis_angle` from the authors' processing or
  from a first coarse alignment, and say where it came from in the lane file.
- For a missing field, add it to the mdocs, or start a stack lane if assembled
  stacks exist.

**What not to do:** take the axis from `RotationAngle − 90` for Tomo5 mdocs.

**When to stop and ask someone:** when the hand of the axis cannot be settled.
On EMPIAR-11830 it was taken from the authors' processing and is not verified.

**Applies to:** since 5 November 2026.

**Source:** EMPIAR-11830 (Falcon 4, Tomo5), November 2026.

### D-9. Which gain reference

**Kind:** silent · **Evidence:** measured

**Where it shows:** nowhere by itself. A wrong or mismatched gain leaves a fixed
pattern in every sum, and the bad-pixel correction in motion correction works on
it.

**How to confirm:** the session the movies come from, against the gain's name
and date. An EER movie's metadata names the gain used at acquisition, which may
not be the one deposited.

**Causes, common to rare:**
1. One gain per session. EMPIAR-11830 deposits 34 of them, and its EER metadata
   names a different (acquisition-time) gain than the one deposited for the same
   session.
2. A gain of the right session in the wrong orientation. That belongs to motion
   correction: see M-3 in [motion.md](motion.md).

**What to try:** one `[motion] gain_reference` per session. Series from sessions
with different gains go into different lanes or projects.

**When to stop and ask someone:** when the session of a series is not known.

**Applies to:** every version.

**Source:** EMPIAR-11830, 5–7 November 2026.

## Questions for reviewers

Please answer in a line each; "don't know" is a useful answer.

- **Q-D1.** How are frames and mdocs named in your sessions (SerialEM,
  Tomo5, PACEtomo, other)? An example name helps most.
- **Q-D2.** Where does your pixel size come from: the header, a calibration of
  the microscope, a published map? How far apart have these been?
- **Q-D3.** Which gain reference goes with which session, and where do you keep
  it? Do you collect a new one per session?
- **Q-D4.** Which tilt scheme do you use, and how would we see in the files that
  a tilt was skipped or repeated?
- **Q-D5.** For Tomo5 data: how do you get the tilt axis angle and its hand?
- **Q-D6.** What goes wrong at the microscope that only shows later in
  processing? A line each: what you saw, what it was.
