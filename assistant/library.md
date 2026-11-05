## The reference library

This session can read the reference library. It holds the documentation of the
tools the pipeline runs, at the versions installed on this site:

- IMOD's manual;
- the manual pages of Slurm and bash;
- what CTFFIND, MotionCor3, pytom-match-pick and MemBrain-seg print for `--help`;
- the site card, `site.md`: tool versions, partitions, GPU types, modules.

`library.json` says what each page was read from. docs/library.md says how the
library is built.

- **Searching.** Search it with `python3 scripts/search_library.py <word>`, by
  the flag, option or message you want explained. Use `-e <flag>` for a word
  that starts with "-", and `--tool <name>` for one tool's pages. It prints
  each matching page with its tool, the tool's version and its source, and the
  matching lines with their numbers. Then read the page with Read, from that
  line.
- **What it is for.** It tells what a tool's flags, options and messages mean.
  What happened in a run still comes from that run's logs and QC records.
- **Citing.** Name what you used: the tool, its version and the page, such as
  "IMOD 5.1.12, man/newstack.html".
- **Versions.** If a run's log shows a version of a tool other than the
  library's, say so before relying on the page.
- **Read-only.** It is read-only for you, as for everyone. Never write, move or
  delete anything in it, and never ask the person to.
