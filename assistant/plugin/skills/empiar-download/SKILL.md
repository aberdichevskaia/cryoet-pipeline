---
name: empiar-download
description: Read what an EMPIAR entry holds, write its download layout (a small YAML file), check it against the entry and each series' mdoc, and give the person the download command. Use when the person names an EMPIAR entry to fetch, or asks what one contains.
---

# EMPIAR: read the entry, write the layout, check it

Paths are from the repository root. A layout is the YAML file that
`cryoet download-empiar --layout` takes. `configs/empiar-layouts/example.yaml`
says how to write one. `configs/empiar-layouts/11830.yaml` is the worked
example of a folder per series and a gain shared by all of them.

## 1. Ask only what the entry cannot say

Ask for three things:
- the entry's number;
- which part of it is wanted: a condition, how many series, which ones;
- where the files are to go.

Never ask how the deposition is organised: read it.

## 2. Read the entry

```bash
python3 scripts/show_empiar.py <id>
python3 scripts/show_empiar.py <id> <folder of data/>
```

The first command prints the image sets the EMPIAR API declares: folder,
category, format, number of series and pixel size. It then lists the folders of
`data/` to two levels:
- a folder of many folders is shown by three of them;
- a folder of mdocs is shown by the series ids its names give.

The second command goes deeper into one folder. EBI's server is slow at times,
so give each call up to 10 minutes.

## 3. Write the layout

Write `configs/empiar-layouts/<id>.yaml`, or `<id>-<variant>.yaml` for one part
of an entry. The client asks the person to approve the new file. Write one
group per kind of file:
- the mdocs; the frames, or one assembled stack per series; and a gain shared
  by all series, with `shared: true`;
- `{tilt_series}` goes in `remote_dir` when each series has a folder, and in
  `file_pattern` when the series lie side by side;
- `local_dir` gathers each kind in one folder (`mdocs`, `frames`, `gain`),
  which is what a lane reads;
- above each group, a comment says where it was seen.

## 4. Check it

```bash
python3 scripts/show_empiar.py <id> --layout <layout> --series <A> --series <B> [--out <folder>]
```

For each series, the tool shows the files the download would take, chosen as
`cryoet download-empiar` chooses them, and their size. It then compares the
frames the series' mdoc names with the frames the layout takes. A pattern that
also takes another series' frames shows up there. Fix the layout until every
series is accepted. A refusal comes from the downloader or the mdoc parser:
show it as it is.

What the tool cannot check:
- A file taken once for all series, such as a gain, has to belong to every
  series asked for. The tool prints the folder it comes from; compare that
  with the series. In EMPIAR-11830 each session has a gain of its own.
- A stack's images are counted only after the download, by
  `scripts/show_folder.py`.
- Sizes are as the listings round them.

## 5. Show, ask, hand over

Show the person:
- the table of series;
- the size in all;
- where the files land;
- every warning.

Ask whether to download. The download can run for hours, so give the person
the command the tool printed, to run in their own shell. Do not run it
yourself. Running it again resumes it.

## 6. After the download

Read the folder with `scripts/show_folder.py`, as the `lane-config` skill says.
