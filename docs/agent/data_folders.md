# Reading a data folder

The skill of [roadmap.md](../roadmap.md) section 22: given a folder of raw data,
say how its files are named, what each file holds, and which step a lane starts
from. The output is a config fragment that the person confirms before anything
runs. A wrong reading mislabels tilts without any error, because alignment and
reconstruction run on any angle list.

## Start with the main naming schemes

Most folders follow one of the four schemes below. Try them first, in this
order, before reading the names from scratch. The list is not exhaustive. A
folder that fits none of them is read field by field (see
[When no scheme fits](#when-no-scheme-fits)), and that reading is shown to the
person before it is used.

| | Data | Scheme; parts in `{…}` are optional | Example |
|---|---|---|---|
| 1 | frames from SerialEM, one series at a time (TIFF, MRC) | `<PREFIX>_<TS>{_<VIEW>}{_<ANGLE>}{_<DATE>_<TIME>}.<EXT>` | `TS_05_008_-12.0.mrc` (EMPIAR-10164) |
| 2 | Falcon 4 frames (EER) | `<PREFIX>_<TS>{_<VIEW>}[<ANGLE>].<EXT>` | `Position_02_001[-10.00]_EER.eer` (EMPIAR-11306) |
| 3 | several series per grid square, from PACEtomo (TIFF) | `Square<SQUARE>_<PREFIX>_<TS>_<VIEW>_<ANGLE>{_<DATE>_<TIME>}.<EXT>` | `Square3_ts_001_000_-12.0_Feb24_10.12.16.tif` |
| 4 | assembled tilt stacks (MRC) | any unique name; the angles in `<name>.mdoc` beside it | `CTEM_tomo2.mrc` with `CTEM_tomo2.mrc.mdoc` (EMPIAR-10064) |

`<TS>` is the series number. `<VIEW>` is the number of the image within its
series. `<ANGLE>` is the tilt in degrees. `<DATE>_<TIME>` looks like
`Feb24_10.12.16`.

These are the schemes TomoBEAR reads. They come from its wiki page
[Usage → Input Data File Formats](https://github.com/KudryashevLab/TomoBEAR/wiki/Usage#input-data-file-formats)
and were checked on 28 October 2026 against `configurations/defaults.json` and
`utilities/getUnprocessedTomograms.m` on its `main` branch. Three details are
not obvious from the wiki:

- **Scheme 2.** The example ends in `_EER.eer`, which the scheme itself does not
  show. TomoBEAR turns `[` into `_` and drops `]` before reading the name, so
  the angle in brackets is read like any other angle.
- **Scheme 3.** `Square` and the square number have nothing between them. The
  wiki's own example, `Square3_ts_001_000_Feb24_10.12.16.tif`, lacks the angle
  that the scheme requires, and TomoBEAR takes its PACEtomo branch only when it
  finds an angle. The example in the table above has the angle added.
- **Scheme 1 locally.** Our first local dataset follows scheme 1:
  `tomogram_039_001_-42.0.tif` (RyR1 at MDC). Its pattern is in
  `configs/lanes/ryr1-bin8.toml`.

## What each scheme becomes in a lane

**Frames with mdocs.** When the folder has SerialEM mdocs, the angles and the
order come from them (`[data] mdocs`). The frame file names then only have to
match the mdocs' `SubFramePath` lines, and each series id is the mdoc's name
without `.mrc.mdoc` or `.mdoc`. The naming scheme matters much less in that
case.

**Frames without mdocs.** These are read with `[data] frames_pattern`: one Python
regular expression, matched against the whole file name, whose named groups
`series`, `tilt` and `angle` say where each piece sits
([frame_names.py](../../src/cryoet_pipeline/ingest/frame_names.py)).
`cryoet init` refuses a folder in which some file with the frames' suffix does
not match the pattern. It also refuses a series that has the same tilt number
or the same angle twice. Write the pattern in single quotes, so that TOML leaves
the backslashes alone.

The patterns below were each checked against their example on 28 October 2026.
In each, the angle group takes a sign and an optional decimal part, and the date
and time are optional where the scheme has them.

1. **SerialEM frames:**

   ```toml
   frames_pattern = 'TS_(?P<series>\d+)_(?P<tilt>\d+)_(?P<angle>[-+]?\d+(?:\.\d+)?)(?:_[A-Z][a-z]{2}\d{2}_\d{2}\.\d{2}\.\d{2})?\.(?:mrc|tiff?)'
   ```

   Two cases have no pattern:
   - **No angle in the name.** The angles then have to come from mdocs.
   - **No view number in the name.** The pattern then has nothing to put in
     `tilt`. `init` cannot order such a series today, by its angles or by the
     times in its names. Say so, rather than inventing a tilt number.

2. **EER:** the names can be read, but nothing can run on them. EER movies are
   refused as not supported yet (`movies.py`). List the series and tilts found,
   say that EER has no entry, and stop.

   ```text
   Position_(?P<series>\d+)_(?P<tilt>\d+)\[(?P<angle>[-+]?\d+(?:\.\d+)?)\]_EER\.eer
   ```

3. **PACEtomo:** the series id has to be unique in the folder. Put the square in
   the `series` group, because the `ts` numbers may repeat from one square to the
   next. Check this in the listing rather than assuming it.

   ```toml
   frames_pattern = '(?P<series>Square\d+_ts_\d+)_(?P<tilt>\d+)_(?P<angle>[-+]?\d+(?:\.\d+)?)(?:_[A-Z][a-z]{2}\d{2}_\d{2}\.\d{2}\.\d{2})?\.tiff?'
   ```

4. **Assembled stacks:** these make a stack lane:

   ```toml
   [lane]
   input = "stack"

   [data]
   stack = "<folder>/{series}.mrc"
   mdoc = "<folder>/{series}.mrc.mdoc"
   ```

   `import-tilt-stack` needs the mdoc. A stack whose angles exist only as a list
   has no entry today.

Whatever the names, one case has no entry: files that each hold **one summed
image of a tilt** rather than its frames. It would need a stack assembled
without motion correction, which does not exist yet. Here, as for EER, the
assistant says that there is no entry. It does not bend the data into one
([roadmap.md](../roadmap.md) section 22).

## Traps already met

- **Signed angles such as `+03.0`.** The first pattern written by hand for the
  RyR1 folder had no `+`. It matched only the 15 tilts from `-42.0` to `-00.0`,
  and would have reconstructed from half of every series. Playbook D-2 covers this trap and the
  next three.
- **Zero written as `-00.0`, or as `0.0` in one series and `-0.0` in the next.**
  Both are read as 0 here. TomoBEAR's troubleshooting page has users set
  `first_tilt_angle: 0` for this case, because it splits series where the first
  angle recurs.
- **Leading zeros in a series id (`039`).** The id is kept verbatim, so `039` and
  `39` are different series.
- **What the view number means.** In EMPIAR-10164 it is the acquisition order.
  In the RyR1 folder it is the angle order, and the acquisition order is not in
  the names at all. Nothing here needs the dose order today, but do not assume
  either meaning.
- **Angles without a decimal part.** TomoBEAR's angle expression requires a
  point, so it does not read `_-12_`. The patterns above take both forms. If a
  folder mixes the two, say so.
- **A prefix with several parts (`grid2_lamella3_TS_…`).** TomoBEAR's
  troubleshooting page suggests symbolic links that give every file a one-part
  prefix. A pattern does not need that: write the prefix out literally. Never
  rename or move the person's files. A folder of links is something to propose
  to the person, not something to make.
- **Order taken from the file time.** TomoBEAR orders the tilts within a series
  by the date and time in the name, then by the view number, then by the files'
  own timestamps, which a parallel copy changes. Here the order comes from the
  `tilt` group alone. Timestamps can still test what that group means: a time
  in the TIFF tags where one was written, and the file times only if the files
  were not copied.

## Checks before proposing a reading

Run the pattern over every name in the folder with `re.fullmatch`, then show:

- the series found, and per series the number of tilts, the angle range and the
  step;
- the names that did not match, grouped by suffix;
- what one file holds: the TIFF page count or the MRC section count of a few
  files. One file per tilt with several frames in each is movies. One file per
  tilt with a single image is a summed image. One file per series, with as many
  sections as its mdoc has tilts, is an assembled stack.

Then compare these numbers with the acquisition: the tilt scheme on the
instrument card, or what the person says was collected. Some numbers point to a
wrong reading:

- one series with twice the expected tilts, or the same angle twice, is two
  series under one id;
- a range covering only one sign has lost the `+`;
- series of very different lengths can be a wrong field split.

TomoBEAR drops series shorter than 15 files (`minimum_files`). Here nothing is
dropped: a short series is shown and the person is asked about it.

## When no scheme fits

1. List about twenty names (sorted), the count of files per suffix, and a few
   names from the middle and the end of the list.
2. Split the names at `_`, `.`, `[` and `]`. For each field, say whether it is
   constant or varies, and whether it holds digits, a signed decimal, a word or a
   date.
3. Propose which field is the series, which the tilt and which the angle, and
   which fields are noise to skip: a date, or a `_fractions` suffix. Write the
   pattern, and run the checks above.
4. If two readings fit the names, show both and do not pick one. For example, it
   may be unclear which number is the series and which the tilt.
5. Wait for the person to confirm before `init` runs.

## Reference: TomoBEAR's expressions

These are from `configurations/defaults.json`, as JSON with its escaping, read
on 28 October 2026. They are MATLAB regular expressions, which TomoBEAR applies
in `getUnprocessedTomograms.m`. A config can override them in `general`.

```json
{
  "angle_regex": "_([+-]*\\d+[.]+\\d+)[_.]*",
  "name_regex": "([A-Za-z\\d_]+[A-Za-z]+)",
  "number_regex": "[_]*([\\d]+[_[\\d]*]*)_",
  "name_number_regex": "([A-Za-z\\d_]+[A-Za-z]+)[_]*([\\d]+[_[\\d]*]*)",
  "name_number_regex_backup": "([A-Za-z\\d_]+[A-Za-z]+)[_]*([\\d]+[_[\\d]*]*)_",
  "month_date_time_regex": "_([A-Z][a-z]{2}[\\d]{2})_([\\d]{2}.[\\d]{2}.[\\d]{2})[._]"
}
```

- `angle_regex` takes the angle, which must have a decimal part.
- `name_number_regex` takes the prefix and the block of numbers after it. The
  code then works out which number is the series and which the view.
- `month_date_time_regex` takes a date and time like `Feb24_10.12.16`.

TomoBEAR can also be told the position of each field between underscores:
`prefix_position`, `tomogram_number_position`, `tilt_number_position`,
`angle_position`, `date_position` and `time_position`. With
`automatic_filename_parts_recognition`, which is on by default, it works the
positions out from the expressions above. Our `frames_pattern` does the same
job with named groups. These expressions are a
guide to what TomoBEAR expects, not something to copy: a pattern written for
the folder in front of you, and checked against all its names, is what goes
into the config.

**Sources:** the TomoBEAR wiki pages
[Usage](https://github.com/KudryashevLab/TomoBEAR/wiki/Usage#input-data-file-formats)
and
[Troubleshooting](https://github.com/KudryashevLab/TomoBEAR/wiki/Troubleshooting),
and its
[`defaults.json`](https://github.com/KudryashevLab/TomoBEAR/blob/main/configurations/defaults.json)
and
[`getUnprocessedTomograms.m`](https://github.com/KudryashevLab/TomoBEAR/blob/main/utilities/getUnprocessedTomograms.m),
all read on 28 October 2026. The traps come from the RyR1 folder (roadmap
section 22, playbook D-2).
