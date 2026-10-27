"""Do two Dynamo classifications put the same particles in the same classes?

    python scripts/compare_dynamo_classes.py <export>/mra_sbb <export_decoy>/mra_sbb_decoy \\
        --group-first 2,3 --group-second 2,3

compares the classes of two Dynamo projects particle by particle, after the last
iteration each finished (or `--iteration-first/--iteration-second`):

- a table of how many particles each pair of classes shares, with the number two
  independent runs with these class sizes would share by chance;
- the adjusted Rand index (0 at chance, 1 when identical) and the share of
  particles in the same class under the best pairing of classes, each against
  the same statistic over shuffles of the second run's labels;
- with `--group-first` and `--group-second`, the overlap of two groups of
  channels, one per run (say, the channels seeded with blobs or noise), against
  the hypergeometric chance: are the particles one run sends to its junk classes
  the ones the other run sends to its own?

Particles are matched by tag. Tags mean the same particle in two exports only if
both were exported from the same stacks in the same series order, so the export
manifests beside the projects are checked tag by tag (series and stack index)
and a mismatch is refused. Decoys (marked in a manifest) are left out unless
`--include-decoys`. Nothing in the projects is changed. See
src/cryoet_pipeline/class_comparison.py for the statistics.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np

from cryoet_pipeline.class_comparison import (
    GroupOverlap,
    ShuffledNull,
    adjusted_rand_index,
    best_pairing,
    contingency,
    group_overlap,
    paired_share,
    shuffled_null,
)
from cryoet_pipeline.reports.classification import ClassificationRun
from cryoet_pipeline.reports.dynamo_project import read_dynamo_classification


def labels(run: ClassificationRun, iteration: int | None) -> tuple[int, dict[int, int]]:
    """Each tag's class after `iteration` (the last finished one when None)."""

    if not run.iterations:
        raise SystemExit(f"{run.source}: no finished iteration to compare")
    if iteration is None:
        chosen = run.iterations[-1]
    else:
        found = [one for one in run.iterations if one.number == iteration]
        if not found:
            raise SystemExit(f"{run.source}: iteration {iteration} has not finished; finished "
                             f"are {[one.number for one in run.iterations]}")
        chosen = found[0]
    return chosen.number, {int(tag): channel for channel, result in chosen.classes.items()
                           for tag in result.members}


def particle_identity(project: Path) -> dict[int, tuple[str, str, int]] | None:
    """Tag -> (series, stack, index in the stack), from the export manifest beside the project."""

    manifest = project.parent / "export_manifest.json"
    if not manifest.exists():
        return None
    entries = json.loads(manifest.read_text()).get("particles", [])
    return {int(entry["tag"]): (str(entry.get("run")), str(entry.get("source_stack_id")),
                                int(entry.get("source_index_zero_based", -1)))
            for entry in entries}


def check_identity(first: Path, second: Path, tags: list[int]) -> str:
    known_first, known_second = particle_identity(first), particle_identity(second)
    if known_first is None or known_second is None:
        missing = [str(path.parent / "export_manifest.json")
                   for path, known in ((first, known_first), (second, known_second))
                   if known is None]
        return ("NOT checked that a tag means the same particle in both runs: no "
                + ", ".join(missing))
    differing = [tag for tag in tags if known_first.get(tag) != known_second.get(tag)]
    if differing:
        examples = "; ".join(f"tag {tag}: {known_first.get(tag)} vs {known_second.get(tag)}"
                             for tag in differing[:5])
        raise SystemExit(f"{len(differing)} tags name different particles in the two exports, "
                         f"so they cannot be compared by tag: {examples}")
    return "tags checked against both export manifests: same series, stack and index"


def channels(text: str | None) -> set[int] | None:
    if text is None:
        return None
    return {int(one) for one in text.split(",") if one.strip()}


def describe_null(null: np.ndarray, observed: float, label: str, *,
                  percent: bool = False) -> str:
    def shown(value: float) -> str:
        return f"{100 * value:.1f}%" if percent else f"{value:.3f}"

    p = ShuffledNull.p_value(null, observed)
    reached = int(np.count_nonzero(null >= observed))
    tail = (f"none of {len(null)} shuffles reached it (p <= {p:.4f})" if not reached
            else f"{reached} of {len(null)} shuffles reached it (p = {p:.4f})")
    return (f"{label}; shuffled: median {shown(float(np.median(null)))}, "
            f"max {shown(float(np.max(null)))}; {tail}")


def compare_groups(first: np.ndarray, second: np.ndarray, group_first: set[int],
                   group_second: set[int]) -> dict[str, Any] | None:
    """Print and return how one run's group of channels overlaps the other's."""

    overlap: GroupOverlap = group_overlap(np.isin(first, list(group_first)),
                                          np.isin(second, list(group_second)))
    names = (f"channels {','.join(map(str, sorted(group_first)))} of the first run",
             f"channels {','.join(map(str, sorted(group_second)))} of the second")
    print(f"\n{names[0]} ({overlap.first:,}) against {names[1]} ({overlap.second:,}):")
    if not overlap.first or not overlap.second:
        print("  a group holds no particle (its channels were dropped or never kept any), "
              "so there is no overlap to compare")
        return None
    print(f"  in both {overlap.both:,}, first only {overlap.first_only:,}, "
          f"second only {overlap.second_only:,}, in neither {overlap.neither:,}")
    direction = (f"p (this many or more) = {overlap.p_more:.2g}" if overlap.ratio >= 1
                 else f"p (this few or fewer) = {overlap.p_less:.2g}")
    print(f"  by chance {overlap.expected_both:,.1f} in both: x{overlap.ratio:.2f}; {direction}")
    print(f"  {100 * overlap.both / overlap.first:.1f}% of the first group is in the second; "
          f"{100 * overlap.both / overlap.second:.1f}% of the second is in the first")
    return {
        "first": sorted(group_first), "second": sorted(group_second),
        "both": overlap.both, "first_only": overlap.first_only,
        "second_only": overlap.second_only, "neither": overlap.neither,
        "expected_both": overlap.expected_both, "ratio": overlap.ratio,
        "p_more": overlap.p_more, "p_less": overlap.p_less,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("first", type=Path, help="a Dynamo project folder")
    parser.add_argument("second", type=Path, help="another Dynamo project folder")
    parser.add_argument("--iteration-first", type=int, help="default: its last finished one")
    parser.add_argument("--iteration-second", type=int, help="default: its last finished one")
    parser.add_argument("--group-first", help="channels of the first run forming one group, "
                        "e.g. 2,3 for the blob-seeded ones")
    parser.add_argument("--group-second", help="the same for the second run")
    parser.add_argument("--include-decoys", action="store_true",
                        help="compare decoys too (only those both runs hold)")
    parser.add_argument("--shuffles", type=int, default=2000)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--out", type=Path, help="also write the numbers as JSON here")
    args = parser.parse_args()
    if (args.group_first is None) != (args.group_second is None):
        parser.error("give --group-first and --group-second together")

    runs = [read_dynamo_classification(path.resolve(), compute_initial_average=False)
            for path in (args.first, args.second)]
    (first_number, first_labels), (second_number, second_labels) = (
        labels(runs[0], args.iteration_first), labels(runs[1], args.iteration_second))
    decoys = set() if args.include_decoys else set(runs[0].decoys | runs[1].decoys)
    tags = sorted((set(first_labels) & set(second_labels)) - decoys)
    if len(tags) < 2:
        raise SystemExit("fewer than two particles are classified in both runs")
    identity = check_identity(args.first.resolve(), args.second.resolve(), tags)

    first = np.array([first_labels[tag] for tag in tags])
    second = np.array([second_labels[tag] for tag in tags])
    table, left, right = contingency(first, second)
    total = len(tags)
    expected = np.outer(table.sum(axis=1), table.sum(axis=0)) / total

    for name, run, number, values in (("first ", runs[0], first_number, first),
                                      ("second", runs[1], second_number, second)):
        sizes = ", ".join(f"class {k} {int(np.count_nonzero(values == k)):,}"
                          for k in sorted(set(values.tolist())))
        print(f"{name} {run.source} (iteration {number}): {sizes}")
    left_out = len(set(runs[0].decoys | runs[1].decoys)) if not args.include_decoys else 0
    print(f"particles in both: {total:,}" + (f" ({left_out:,} decoys left out)" if left_out
                                            else ""))
    print(identity)

    print("\nparticles per pair of classes (by chance in brackets):")
    width = 18
    print(" " * 10 + "".join(f"{'second ' + str(k):>{width}}" for k in right))
    for row, k in enumerate(left):
        cells = "".join(f"{f'{table[row, column]:,} ({expected[row, column]:,.0f})':>{width}}"
                        for column in range(len(right)))
        print(f"{'first ' + str(k):<10}{cells}")

    index = share = index_p = share_p = None
    pairs: list[tuple[int, int]] = []
    single = [name for name, classes in (("the first run", left), ("the second", right))
              if len(classes) < 2]
    print()
    if single:
        # One class leaves nothing to shuffle: the index is 0/0 against one class on
        # each side, 0 against several, and every shuffle gives the same.
        print(f"one class in {' and '.join(single)}: nothing to compare with chance")
    else:
        null = shuffled_null(first, second, rounds=args.shuffles, seed=args.seed)
        index = adjusted_rand_index(table)
        share = paired_share(table)
        index_p = ShuffledNull.p_value(null.rand_index, index)
        share_p = ShuffledNull.p_value(null.paired_share, share)
        pairs = [(left[row], right[column]) for row, column in best_pairing(table)]
        print(describe_null(null.rand_index, index, f"adjusted Rand index {index:.3f}"))
        print(describe_null(null.paired_share, share,
                            "classes paired " + ", ".join(f"{a}-{b}" for a, b in pairs)
                            + f": {100 * share:.1f}% of particles in the same class",
                            percent=True))

    result: dict[str, Any] = {
        "first": {"project": str(runs[0].source), "iteration": first_number},
        "second": {"project": str(runs[1].source), "iteration": second_number},
        "particles": total,
        "decoys_left_out": left_out,
        "identity": identity,
        "classes_first": left,
        "classes_second": right,
        "table": table.tolist(),
        "expected": expected.round(2).tolist(),
        "adjusted_rand_index": index,
        "adjusted_rand_index_p": index_p,
        "pairing": pairs,
        "paired_share": share,
        "paired_share_p": share_p,
        "shuffles": args.shuffles,
    }

    group_first, group_second = channels(args.group_first), channels(args.group_second)
    if group_first is not None and group_second is not None:
        groups = compare_groups(first, second, group_first, group_second)
        if groups is not None:
            result["groups"] = groups

    if args.out is not None:
        args.out.write_text(json.dumps(result, indent=1) + "\n")
        print(f"\nwrote {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
