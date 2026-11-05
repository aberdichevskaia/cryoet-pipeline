# Reference library for the assistant

## Purpose

The assistant ([agent_architecture.md](agent_architecture.md)) reads our own
notes in `docs/agent/`. It does not read the documentation of the tools the
pipeline runs. This document decides how that documentation is kept as a
library the assistant can search:

- what goes in;
- where it lives;
- how it is built;
- how the assistant is kept from changing it;
- how we will know it helps.

The first user is an open-weight model at MDC, which has no other way to look
anything up. Claude reads the same library.

Status, 2026-11-05:

- **Built:** the installation channel and the site card, by
  `scripts/build_library.py` from the sources in `configs/library.toml`.
  Tested on the Mac. The first cluster snapshot is MDC's `mdc-2026-11-05`.
  It was built on max025, from a shell that had the tool variables; the login
  shell of the first dry run had none of them. The snapshot holds 380
  documents. Every tool was found with its version, and there were no problems
  and no notes: IMOD's manual has a page for every IMOD program the pipeline
  runs, and the node has all the manual pages asked for.
- **Built:** the assistant's read access, read-only, and its search:
  - `scripts/assistant` finds the snapshot;
  - `assistant/guard.py` refuses every write to the library;
  - `scripts/search_library.py` searches it.

  A real headless run on the Mac checked all three. See
  [Read only, enforced in layers](#read-only-enforced-in-layers) and
  [Search](#search).
- **Not built yet:** the web sources and the git channel.

## Core rule

```text
The library is files: each source converted to markdown, its original beside it.
This repository holds the recipe (the source list and the scripts), never the content.
A snapshot is built per site, never changed in place, and named by every record that used it.
The assistant reads the library and never writes to it.
An index, if one is ever needed, is rebuilt from the files.
```

## Decided

- **Manuals before papers.** The traps met so far were about tools, not
  biology. Papers come last, if at all.
- **What goes in first.** The manuals and wikis of every tool the pipeline
  runs, GPU and CUDA versions, Slurm and bash. The list is in
  [What goes in](#what-goes-in).
- **Not in the project database.** [database_design.md](database_design.md)
  answers this itself, in three ways:
  - its core rule makes a database a rebuildable index over files, never the
    only copy;
  - in its terms the library is a `shared_asset`, not project state. It is
    lab-wide and immutable, with a natural key for "already downloaded?" and a
    content hash for "the same thing?";
  - that database does not exist yet.
- **Not in this repository.** Three reasons:
  - the repository is public: github.com/aberdichevskaia/cryoet-pipeline
    answered without sign-in on 2026-11-04;
  - the sources carry their own licences;
  - some site wikis sit behind a sign-in.
- **This repository holds the list of sources and the scripts that download
  the library from each source's own site.** Each site downloads its own copy,
  so nothing is redistributed.
- **Snapshots live in a folder the group can read, one per site, not one per
  clone.** At MDC that folder is under `/fast/AG_Kudryashev/`.
- **The assistant reads the library and never writes to it.** This is a
  fundamental rule, not a setting: no mode, prompt or option changes it. It is
  enforced in layers below the prompt, and every layer is in place before the
  assistant is given the library. See
  [Read only, enforced in layers](#read-only-enforced-in-layers).
- **The installation channel is mandatory and is built first.**
- **The builder finds each tool's version itself** and asks only for what it
  cannot find.
- **Each snapshot carries a `library.json`, and the converters are
  deterministic code.**
- **Installing the pipeline on a host will set the library's folder.** That
  comes later. Until then the launcher finds the folder; see
  [Where snapshots live](#where-snapshots-live).

## Three kinds of tool knowledge

Which kind a fact is decides whether a library can help with it at all.

- **Written in a source**: a manual page, a wiki, a README. The library helps.
- **True only of the installed version**: flags and defaults. Only the
  installation says this exactly, through its usage text, its own manual and
  `man`.
- **Found by measurement.** An example is that `tilt -LOCALFILE` patches are
  indexed by specimen X,Y. The `tiltalign` manual page gives the file's format
  but never says this; it was measured with `tilt -ProjectModel`. No library
  holds such a fact. It belongs in our own notes in `docs/agent/`, which stay
  in this repository.

Before a fact becomes a knowledge case (see
[How we will know it helps](#how-we-will-know-it-helps-proposed)), its answer is
found in the source, at the version the cluster runs, and quoted. If no such
line exists, the fact is of the third kind.

The assistant reads in three tiers, as other agentic systems do (see
[How others do it](#how-others-do-it)):

1. `assistant/instructions.md`, always;
2. `docs/agent/` (the pipeline map, the playbook, the recipes), when a task
   needs it;
3. the library, through search only, never read whole.

## What goes in

The programs listed here are the ones the code and the lane files call, as of
2026-11-05.

| Source | Contents | Channel |
|---|---|---|
| IMOD | The whole manual (5.6 MB of HTML, 4.3 MB of text). The code calls newstack, tilt, tiltalign, tiltxcorr, beadtrack, autofidseed, xftoxg, xfproduct, ctfplotter, ctfphaseflip, imodinfo, point2model and model2point. | installation: `VERSION`, and `html/IMOD.qch` or the HTML files in `html/`, else the formatted pages in `man/`. The Mac install has `IMOD.qch`; the cluster installs show theirs in a dry run. |
| MotionCor3 | README, usage text | web; installation |
| CTFFIND5 | cisTEM documentation, usage text | web; installation |
| pytom-match-pick | Documentation of the installed version; usage text of the four `pytom_*.py` scripts | web; installation |
| MemBrain-seg | Documentation, usage text | web; installation |
| Dynamo | Wiki pages on tables, MRA, alignment parameters and GPU use | web, dated |
| NVIDIA, CuPy, PyTorch | CUDA driver and toolkit compatibility, compute capabilities, GPU sharing modes; the CUDA install matrices of CuPy (which pytom-match-pick runs on) and PyTorch (which the dynamo-search port runs on) | web, dated |
| Slurm | `man` pages of the installed version (sbatch, sacct, squeue, sinfo, scontrol); the site's own wiki pages | installation; site wiki (see [sign-in sources](#how-a-snapshot-is-built)) |
| bash | The installed `man bash`. A Mac has bash 3.2. | installation |
| Site card | Each tool's version, GPU types and drivers, CUDA modules, partitions | collected on the site, dated |

Proposed, not decided:

- environment modules and micromamba;
- data formats: MRC2014, SerialEM mdoc, RELION STAR, Dynamo tables, EER;
- the tools' tutorials, where they exist.

Later, once they run: IsoNet2, miss-alignment, membrain-pick.

Not now:

- **Papers.**
- **EMDB and EMPIAR entries.** They describe one dataset, so they belong with
  that project's data, not in the library.

## How a snapshot is built

Decided:

- the installation channel comes first;
- the builder finds the versions;
- each snapshot carries a `library.json`;
- the converters are deterministic.

The rest is proposed.

```text
installation   the site's own copy: an installed manual, man pages, usage text   exact version
web, pinned    a source kept in git, at the tag or commit that matches            exact version
web, dated     a page without versions, such as the Dynamo wiki or NVIDIA's       retrieval date + sha256
```

- **Runs.** The builder runs on each site, on a login node, in the pipeline's
  environment:

  ```bash
  python3 scripts/build_library.py --site mdc --dry-run
  python3 scripts/build_library.py --site mdc --root /fast/AG_Kudryashev/cryoet-library
  ```

  The dry run prints what was found and what would be read, and writes
  nothing, as `submit_lanes.py` does. `--config` adds lane configs whose
  executables to document. The real run also collects the site card.
- **Versions.** The builder resolves each executable the way the pipeline
  does: from a lane config and the tool variables (`IMOD_DIR`, `CTFFIND5`,
  `MOTIONCOR3`, `PYTOMO_MP_DIR`, `MEMBRAIN_EXE`). Then it asks the tool itself:

  ```text
  IMOD               $IMOD_DIR/VERSION (5.1.12 on the Mac)
  CTFFIND            its banner, which backends/ctf.py already parses
  pytom-match-pick   package metadata in the environment its scripts run in
  MemBrain-seg       package metadata in the environment it runs in
  MotionCor3         not known yet how it reports its version
  Slurm              sinfo -V
  bash               bash --version
  ```

  A version the builder cannot find is listed in the dry run. A command-line
  option supplies it, or it stays `unknown`. The builder never prompts.
- **IMOD's manual.** The builder reads `html/IMOD.qch` (Qt help: SQLite, each
  page zlib-compressed), else the HTML files in `html/`, else the formatted
  pages in `man/`. On the Mac that gives 364 pages, 5.1 MB of markdown, in
  about 4 s. `autodoc/` holds the same options again, so it is not read.
- **Web sources** are fetched by the same run, from the list of sources. A
  pinned source is fetched at the tag its version names, a dated page as it is
  on that day. Nobody downloads them by hand.
- **Sources behind a sign-in.**
  - If a site wiki is a git repository (a GitLab project wiki is one), the
    builder clones it with the user's own credentials and pins the commit.
  - Otherwise its pages are exported by hand into a folder the builder reads.

  Neither path stores credentials. These sources are listed in a local file
  outside git.
- **`library.json`**, the library's `run.json`:
  - for each source, the channel, the URL or path, the version, the sha256 of
    the original and the converter's version. The version is a tool version, a
    tag or commit, or a retrieval date;
  - for the snapshot, its id, host and date.
- **Converters** turn HTML, PDF and Qt help into markdown. They are
  deterministic code, tested on small fixtures. The original is kept, so that
  a better converter can redo the text.
  - database_design.md's rule applies here: generated converters are for data
    that leaves the system, not for data that informs a decision.
  - A summary written by a model is not part of the library. It may go into
    `docs/agent/`, together with the source line it rests on.
- **Failures.** A source that has disappeared or changed shape fails the build
  loudly, and the previous snapshot stays. A diff between two snapshots lists
  what changed in the documentation.
- **Coverage.** Every program the code calls must have a source, so that a new
  backend cannot arrive without its documentation. Two checks keep this:
  - `tests/test_build_library.py` fails when a lane executable key in
    `run_series.LANE_KEYS` has no tool in the list of sources. The fix is one
    line in `configs/library.toml`.
  - Each build lists, as a problem, any IMOD program the code resolves
    (`resolve_imod_executable("…")`) that has no page in the installed manual.
- **Politeness.** Downloads go one request at a time, under a generic client
  name. Each site's terms are checked when its source is added.

## Read only, enforced in layers

The rule is decided. Layers 1–5 are built (2026-11-05); layer 6 is still to be
checked.

1. **File permissions.** The builder is the only writer. A finished snapshot
   loses its write bits, and the group can only read. For every user except
   the snapshot's owner this holds without any software of ours. Whether ACLs
   on `/fast` change this is still to be checked.
2. **Deny rules** in `assistant/settings.json`:
   `Edit(//fast/AG_Kudryashev/cryoet-library/**)` for MDC's folder, and
   `scripts/build_library.py` under `python3` and `python`. An `Edit` rule
   covers every file-editing tool. A deny rule holds in every mode and over any
   allow rule.
3. **The guard hook** (`assistant/guard.py`) runs before every Bash call and
   every Edit, Write, MultiEdit and NotebookEdit call.
   - It refuses an edit of a file in the library, by any of those tools.
   - It refuses scripts/build_library.py in any form.
   - A command line that names the library may run only commands that read:
     `cat`, `head`, `tail`, `grep`, `ls`, `find` without its actions, `wc`,
     `stat` and the like. No output may be redirected into the library either.
   - "Names" covers the folder in the text, its variables
     (`$CRYOET_LIBRARY_SNAPSHOT`), and any word that resolves into it: globs
     expanded, links followed, and a relative path taken from the folder the
     call runs in.
   - It guards the session's library (`CRYOET_LIBRARY_ROOT`) and the sites'
     folders, also when a session names none.
4. **The launcher** refuses options that would drop layer 2 or 3, as it does for
   `--settings` and the bypass modes.
5. **Tests** in `tests/test_assistant.py` check layers 2–4. They were
   mutation-checked: 11 of 12 mutations were caught. The 12th dropped the check
   of the call's folder, which the per-word resolution already covers.
6. **To check:** Claude Code's Bash sandbox (`sandbox.enabled`) would stop
   writes outside the working directory at the OS level, the owner's included,
   if it runs on the login nodes.

A real headless run checked it (Claude Code 2.1.312, claude-haiku-5-5, on the
Mac, 2026-11-05):
- a Write and a `touch` into the snapshot were both stopped by the guard;
- Read worked through `--add-dir`, with no question asked;
- nothing was written.

Limits:

- The owner can restore the write bits by hand. Layers 2–6 keep the assistant
  from doing that, not a person. A snapshot owned by a separate service account
  would close this gap too.
- An administrator's managed settings (`allowManagedPermissionRulesOnly`) can
  void layer 2. Layer 1 does not depend on them.

## Where snapshots live

The folder is decided; the rest is proposed.

- **Location.** At MDC: `/fast/AG_Kudryashev/cryoet-library/`, a group folder
  rather than a personal one. Inside, each snapshot is `<site>-<date>/`; a
  second build on the same day gets a suffix. `current` points at the latest.
- **Permissions.** The group can read. The directory has setgid, so that new
  files take the group. A built snapshot loses its write bits (see
  [Read only, enforced in layers](#read-only-enforced-in-layers)).
- **One snapshot per site.** The installation channel differs by site, so
  snapshots do too. An eval run on a Mac uses a copy of the snapshot of the
  case's cluster, not one built on the Mac. The cluster is the `cluster` field
  in [cases.toml](agent/cases/cases.toml).
- **Finding it, for now** (built in `scripts/assistant`):
  1. `CRYOET_LIBRARY` names the folder of snapshots, or `none` for a session
     without the library.
  2. Unset, a session for a person takes the first site folder in
     `LIBRARY_SITES` that exists.
  3. A headless run (`-p`) takes a library only when `CRYOET_LIBRARY` names
     one. An evaluation therefore reads it only when it says so, and
     scripts/run_cases.py does not pass the variable on.

  The session reads the snapshot that `current` names when it starts, by its
  own name, so a build during a session changes nothing it reads. The launcher
  then:
  - gives that snapshot with `--add-dir`, read-only;
  - adds `assistant/library.md` and the snapshot's path to the instructions;
  - exports `CRYOET_LIBRARY_ROOT` and `CRYOET_LIBRARY_SNAPSHOT` for the guard;
  - prints the snapshot's name.

  With no library it adds nothing, and the instructions are the same as
  before. It does not rely on a user's shell startup files. Once installing the
  pipeline on a host sets the folder, that setting replaces the site list.
- **Retention.** A snapshot that an eval record names is kept. This is
  garbage collection by reachability from roots, as in database_design.md.

## Search

- **No index now.** IMOD's manual is 5.6 MB of HTML and 4.3 MB of text, about
  1.1 million tokens. That is far too much for any context and trivial for
  grep. All the manuals together come to an estimated tens of megabytes.
- **The search script came first.** Claude Code 2.1.312 has no Grep tool of its
  own, and `grep` through Bash is not on the assistant's allow list, so it would
  ask the person. `scripts/search_library.py` is on the list:
  - it searches the session's snapshot line by line, ignoring case, as plain
    text or with `--regex`;
  - `--tool` limits it to one tool's pages, and `-e` takes a flag that starts
    with "-";
  - pages with more matching lines come first, each shown with its tool, the
    tool's version and its source, then its lines with their numbers;
  - its output is bounded.

  In the run checked above, the assistant answered a question about a newstack
  option from the page and cited "IMOD 5.1.12, man/newstack.html, line 479",
  with no question asked of the person.
- **Embeddings** come only if a case fails on a paraphrase. None of the cryo-EM
  agents read on 2026-11-04 describes embeddings. Claude Code is reported to
  have dropped a local vector index for its own search.
- **Papers would need an index.** Open full text comes to about 0.2–0.3
  billion tokens (a Europe PMC count from 2026-11-03).
  - The index would be a derived file, opened read-only and rebuilt from the
    files.
  - SQLite's locking problems on network filesystems only concern writers.
    Those problems are why database_design.md does not use it as the primary
    store.

## Versions the pipeline records today

`software_versions` in the records, as of 2026-11-05:

```text
IMOD, pytom-match-pick, MemBrain-seg, IsoNet2   "external"
MotionCor3                                      the motioncor3_version parameter, which no lane sets: "unknown"
CTFFIND                                         the version read from its banner
numpy, torch, zarr, mrcfile                     the installed package versions
```

So the documentation for the version that ran a step cannot be looked up yet.
Every one of these tools does state its version when asked. At MDC on
2026-11-05 the builder read:

```text
IMOD 5.1.12, CTFFIND 5.0.2, MotionCor3 1.2.4, pytom-match-pick 0.14.0,
MemBrain-seg 0.0.10, Slurm 26.05.4, bash 5.1.8
```

Recording these versions is not needed to build the library, because the
builder finds the versions on its own. It is needed to match a run to the
documentation of the version that ran it. One detection function should serve
both the builder and the backends; changing the backends is a separate step.

The library's version keys should be the same strings the pipeline will record.
In database_design.md, the backend version is part of a run's fingerprint.

## How we will know it helps (proposed)

- **Knowledge cases.** Two or three cases whose answer is written in a source
  and nowhere in this repository.
  - Each is chosen before any baseline run, with its answer quoted from the
    source at the version the cluster runs. Choosing cases by their baseline
    failures would overstate the gain, because some of them pass when run
    again.
  - Candidates go in [eval_cases.md](agent/eval_cases.md), which runs cannot
    read. This file is copied into the eval sandboxes, so it names no case's
    answer.
- **Arms.** Model (Claude, an open-weight model) × library (off, on).
- **Regression.** The existing cases also run with the library on, because a
  model may start quoting manuals instead of reading the evidence.
- **Records.** A run record names the snapshot, as it names the commit.

## How others do it

Read on 2026-11-04, partly through summaries of the methods; check a detail
before citing it.

- [CryoPilot](https://www.biorxiv.org/content/10.64898/2026.10.29.762908v1)
  (bioRxiv, October 2026): one processing guide compiled from CryoSPARC
  tutorials and case studies. The agent searches it.
- [StructAgent](https://www.biorxiv.org/content/10.64898/2026.05.18.725842v1)
  (bioRxiv, May 2026) has three tiers:
  - a short brief, always loaded;
  - topic files with rules traced to their papers, loaded on demand;
  - paper digests, searched by a script.

  It also has skill documents (inputs, commands, checks, recovery), built from
  official documentation, tutorials and task logs. The paper says its knowledge
  base and skills were not tested separately.
- [cryoAgent](https://www.biorxiv.org/content/10.64898/2026.04.16.718662v1)
  (bioRxiv, April 2026): hand-written prompts and thresholds; no knowledge
  base is described.
- [Biomni](https://www.biorxiv.org/content/10.1101/2025.05.30.656746.full.pdf):
  a know-how library of short protocols and troubleshooting guides, retrieved
  by relevance.
- [HPC-LLM](https://arxiv.org/pdf/2605.16347) (arXiv, May 2026): HPC centres'
  documentation with dense retrieval and a fine-tuned 8B model. Site details
  such as partition names and GPU types are where generic advice fails.
- [Context7](https://github.com/upstash/context7): library documentation keyed
  by version.

None of the three cryo-EM papers measures what its knowledge part adds. The
arms above are for measuring that.

## Later

Parked on 2026-11-05, while the work is at MDC.

- **Munich's site wiki.** It is most likely a GitLab Pages site built from a
  repository, behind the institute's GitLab sign-in. Its address is kept
  outside this repository. The builder is to clone that repository with the
  user's own key, pin the commit, and convert its pages. That is the git
  channel, which the builder does not have yet.
- **CTFFIND's questions.** `ctffind --help` lists only CTFFIND's command-line
  flags, and ends with "Unknown long option 'help'". The parameters CTFFIND
  asks for interactively are not in that output. They have to come from the
  web channel, that is, from cisTEM's documentation.

## Open questions

1. The folder in Munich.
2. Which snapshot the constructed cases (cluster `none`) use.
3. Recording the real versions of external tools (see
   [Versions](#versions-the-pipeline-records-today)).
4. Which of MotionCor3's outputs the 1.2.4 was read from: `version_from` in
   MDC's `library.json`.
5. The MDC snapshot documents the installs of the shell it was built from.
   IMOD and pytom-match-pick there come from a personal folder and a personal
   environment, while CTFFIND, MotionCor3 and MemBrain-seg come from the
   group's `software/`. Do the group's lanes run the same installs?
6. Whether MDC's cluster has a wiki of its own (Munich's: see
   [Later](#later)).
7. Whether Claude Code's Bash sandbox runs on the login nodes.
8. Whether the snapshot's permissions hold on GPFS: `/fast` may carry ACLs
   beside the mode bits.
9. A first session at MDC with the library: whether the assistant finds it,
   searches it and cites it there as it did on the Mac.
