#!/usr/bin/env python3
"""Slide figure: the FSC between the RyR1 consensus halves, drawn as the Gag one is.

Reads what `scripts/consensus_halves.py maps` wrote. From the clone on max:

    python scripts/ryr1_fsc_figure.py

With the defaults it reads the bin8 halves as both refiners left them on
2026-11-02, the pipeline's MRA port and Dynamo:

    outputs/ryr1/dynamo/ryr1-consensus-port_fsc.json     pipeline
    outputs/ryr1/dynamo/ryr1-consensus-dynamo_fsc.json   Dynamo (drawn if there)

It draws their masked curves, the "masked" ones `maps` wrote (under the halves'
alignment mask), and leaves out the unmasked ones. Nothing on the slide names the
mask (Anna, 2026-11-06). The phase randomisation of the Dynamo bin8 halves
(2026-10-27) showed that the mask adds nothing measurable: random-phase maps gave
0.02 +- 0.01 over 40-27 A against the measured 0.21.
It writes <pipeline prefix>_fsc_figure.png and .pdf beside them and prints every
crossing, to check against what `maps` printed. The particle counts and the start's
low-pass are read from the half runs the _fsc.json names. When they cannot be read,
the caption goes without them and says why on the terminal.

For bin4, give its `maps` prefix and the bin4 MRA's last low-pass:

    python scripts/ryr1_fsc_figure.py --pipeline outputs/ryr1/mra/ryr1-consensus-bin4 \\
        --dynamo none --shared-lowpass <A>

The caption differs from the Gag one on purpose. There the halves were two
tomograms. Here they are the consensus particles dealt at random within each
tomogram, and they start from poses the MRA refined with all particles together.
That joint fit can raise their FSC down to the MRA's last low-pass, so the curve
measures consistency, not resolution (consensus_halves.py says the same). The
dashed line marks that low-pass.

What only this dataset's records know is in KNOWN, by pixel size: the bin8 MRA's
last low-pass (29.4 A, coarse_to_fine 53.76 -> 29.4 of mra_pad_ssn_g2atan). For
another pixel size the line is drawn and named only when given.

Crossings are interpolated in spatial frequency, as maps and
fsc_phase_randomization.py read them; the resolution is read at the curve's last
fall through 0.143, as for Gag.

Disposable, like the rest of scripts/.
"""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from fsc_phase_randomization import falls
from matplotlib import font_manager

LEVELS = (0.5, 0.143)
CURVE = "masked"  # which of maps' two curves the slide shows
DEFAULT = Path("outputs/ryr1/dynamo")
STACK_PIXEL_ANGSTROM = 1.68  # the motion-corrected stack (0.84 A super-resolution, ftbin 2)
KNOWN = {13.44: {"shared_lowpass": 29.4}}
# Arial as on the Gag slide; Liberation Sans has Arial's letter widths and is the one
# Linux machines often have; then matplotlib's own.
FONTS = ("Arial", "Liberation Sans", "DejaVu Sans")
CANDIDATE_TICKS = (500, 300, 200, 150, 100, 70, 50, 40, 35, 30, 24, 20, 16, 14, 12, 10,
                   8, 7, 6, 5, 4)


def font() -> str:
    """The first of FONTS installed here. Asking for a missing one would print a
    warning for every text (max has no Arial)."""

    names = {entry.name for entry in font_manager.fontManager.ttflist}
    return next((name for name in FONTS if name in names), FONTS[-1])


def prefix_of(path: Path) -> Path:
    """`maps` --out prefix from a prefix or its _fsc.json."""

    name = path.name
    return path.with_name(name[: -len("_fsc.json")]) if name.endswith("_fsc.json") else path


def load(prefix: Path) -> dict:
    """The curve the slide shows, with the pixel size and the half runs `maps` named."""

    path = prefix.with_name(f"{prefix.name}_fsc.json")
    report = json.loads(path.read_text())
    curve = report.get(CURVE) or {}
    if not curve.get("correlation") or len(curve["correlation"]) != len(
        curve.get("resolution_angstrom") or []
    ):
        raise SystemExit(f"{path}: no usable curve")
    if not report.get("pixel_spacing_angstrom"):
        raise SystemExit(f"{path}: no pixel size")
    return {"curve": curve, "pixel": float(report["pixel_spacing_angstrom"]),
            "halves": report.get("halves") or []}


def crossings(curve: dict, level: float) -> tuple[float | None, float | None]:
    """(first fall, last fall). The last is None when the curve ends above the level."""

    found = falls(curve["resolution_angstrom"], curve["correlation"], level)
    if not found:
        return None, None
    return found[0], None if curve["correlation"][-1] >= level else found[-1]


def half_facts(halves: list[str]) -> tuple[list[int], float | None] | None:
    """Each half's particle count and the start's low-pass, from the half runs.

    Read as `maps` reads them (consensus_halves.py), so either layout of run works.
    None, with the reason printed, when the runs cannot be read here."""

    try:
        import consensus_halves

        counts: dict[int, int] = {}
        lowpass = None
        for project in halves:
            run, number, _, record = consensus_halves._half(Path(project))
            counts[int(record["half"])] = len(consensus_halves._moves(run, number)[0])
            lowpass = record.get("start_lowpass_angstrom", lowpass)
    except (Exception, SystemExit) as exc:  # a figure without counts beats no figure
        print(f"particle counts left out: the half runs could not be read ({exc})")
        return None
    if sorted(counts) != [1, 2]:
        print("particle counts left out: the _fsc.json does not name both halves")
        return None
    return [counts[1], counts[2]], None if lowpass is None else float(lowpass)


def ticks_for(nyquist: float, coarsest: float) -> list[float]:
    """Resolution ticks from Nyquist outward, kept apart by 7% of the axis."""

    chosen = [nyquist]
    for value in sorted(CANDIDATE_TICKS):
        if value <= nyquist * 1.08 or value > coarsest:
            continue
        if 1.0 / value <= 1.0 / chosen[-1] - 0.07 / nyquist:
            chosen.append(value)
    return chosen


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--pipeline", type=Path, default=DEFAULT / "ryr1-consensus-port",
                        help="maps --out prefix (or its _fsc.json) of the pipeline's halves")
    parser.add_argument("--dynamo", default=str(DEFAULT / "ryr1-consensus-dynamo"),
                        help="maps --out prefix of Dynamo's halves, drawn if there; "
                        "'none' to leave it out")
    parser.add_argument("--shared-lowpass", type=float, default=None,
                        help="the MRA's last low-pass, A: how far the halves' shared poses "
                        "were refined with all particles (bin8 default from KNOWN)")
    parser.add_argument("--out", type=Path, default=None,
                        help="output path without extension (default <pipeline>_fsc_figure)")
    args = parser.parse_args()

    pipeline_prefix = prefix_of(args.pipeline)
    pipeline = load(pipeline_prefix)
    pixel = pipeline["pixel"]
    resolutions = pipeline["curve"]["resolution_angstrom"]
    shared = args.shared_lowpass if args.shared_lowpass is not None else KNOWN.get(
        round(pixel, 2), {}).get("shared_lowpass")
    curves = [("pipeline", pipeline["curve"])]
    dynamo_prefix = None if args.dynamo.lower() == "none" else prefix_of(Path(args.dynamo))
    if dynamo_prefix is not None:
        if dynamo_prefix.with_name(f"{dynamo_prefix.name}_fsc.json").is_file():
            dynamo = load(dynamo_prefix)
            if not math.isclose(dynamo["pixel"], pixel, rel_tol=1e-3):
                raise SystemExit("the Dynamo halves are at another pixel size")
            if dynamo["curve"]["resolution_angstrom"] != resolutions:
                raise SystemExit("the Dynamo halves have other shells (another box?)")
            curves.append(("dynamo", dynamo["curve"]))
        else:
            print(f"no {dynamo_prefix.name}_fsc.json: Dynamo left out")

    box = round(resolutions[0] / pixel)
    nyquist = 2.0 * pixel
    binning = round(pixel / STACK_PIXEL_ANGSTROM)

    print(f"{'curve':<10} {'0.5 first':>10} {'0.143 first':>12} {'0.143 last':>11}")
    found: dict[str, dict[float, tuple[float | None, float | None]]] = {}

    def text(value: float | None) -> str:
        return "none" if value is None else f"{value:.1f} A"

    for name, curve in curves:
        found[name] = {level: crossings(curve, level) for level in LEVELS}
        half, low = found[name][0.5], found[name][0.143]
        print(f"{name:<10} {text(half[0]):>10} {text(low[0]):>12} {text(low[1]):>11}")

    facts = half_facts(pipeline["halves"])

    chosen_font = font()
    plt.rcParams["font.family"] = chosen_font
    fig = plt.figure(figsize=(10, 8.0), dpi=200)
    ax = fig.add_axes([0.09, 0.40, 0.86, 0.52])
    style = {
        "pipeline": dict(color="#1f4e9c", linewidth=2.4, marker="o", markersize=4.5, zorder=4),
        # On top and dashed: it lies on the pipeline's curve and would vanish under it.
        "dynamo": dict(color="#e07b39", linewidth=1.3, linestyle=(0, (4, 3)), zorder=5),
    }

    def label(name: str) -> str:
        half, low = found[name][0.5][0], found[name][0.143]
        read = low[1] if low[1] is not None else low[0]
        numbers = (f"0.5 at {half:.1f} Å, 0.143 at {read:.1f} Å" if half and read
                   else "does not fall through 0.143")
        title = {"pipeline": "Pipeline",
                 "dynamo": "Dynamo (dashed, on the pipeline's curve)"}[name]
        return f"{title}: {numbers}"

    for name, curve in curves:
        frequency = [1.0 / r for r in curve["resolution_angstrom"]]
        ax.plot(frequency, curve["correlation"], label=label(name), **style[name])

    for level in LEVELS:
        ax.axhline(level, color="#555555", linestyle=":", linewidth=1, zorder=1)
        ax.text(1.0 / nyquist, level + 0.012, f"{level:g}", ha="right", va="bottom",
                fontsize=11, color="#555555")

    span = 1.0 / nyquist
    if shared is not None and shared > nyquist:
        ax.axvline(1.0 / shared, color="#777777", linestyle="--", linewidth=1.1, zorder=2)
        ax.text(1.0 / shared + 0.008 * span, 0.72, f"poses shared\nto {shared:g} Å",
                fontsize=10.5, color="#555555", ha="left", va="center", clip_on=False)
    first, last = found["pipeline"][0.143]
    if last is not None:
        ax.axvline(1.0 / last, color="#c0392b", linestyle="-", linewidth=1.2, zorder=2)
        # Away from the shared line when that is close, so the two labels do not meet.
        near = shared is not None and 0 < 1.0 / shared - 1.0 / last < 0.25 * span
        ax.text(1.0 / last + (-0.008 if near else 0.008) * span, 0.93, f"{last:.1f} Å",
                color="#c0392b", fontsize=13, fontweight="bold",
                ha="right" if near else "left", va="center")

    ax.set_xlim(0, span)
    ax.set_ylim(-0.05, 1.03)
    ticks = ticks_for(nyquist, resolutions[0])
    ax.set_xticks([1.0 / t for t in ticks])
    ax.set_xticklabels([f"{nyquist:.1f}\nNyquist"] + [f"{t:g}" for t in ticks[1:]])
    ax.set_xlabel("Resolution (Å)", fontsize=13)
    ax.set_ylabel("Fourier shell correlation", fontsize=13)
    ax.tick_params(labelsize=11.5)
    ax.spines[["top", "right"]].set_visible(False)
    ax.set_title(f"RyR1, 5 tomograms: FSC between consensus halves, bin{binning}",
                 loc="left", fontsize=15, fontweight="bold", pad=12)
    fig.legend(*ax.get_legend_handles_labels(), loc="upper left",
               bbox_to_anchor=(0.082, 0.305), fontsize=11, frameon=False)

    halves = ("Halves: the particles two MRA runs both put in the RyR1 class, dealt at random "
              "within each tomogram")
    if facts is not None:
        counts, start = facts
        lines = [f"{halves} ({counts[0]} + {counts[1]}),",
                 "and refined apart from one start"
                 + (f" low-passed to {start:g} Å." if start else ".")]
    else:
        lines = [f"{halves}, and refined apart from one start."]
    lines.append("Not independent: both start from poses the MRA refined with all particles"
                 + (f" to {shared:g} Å" if shared is not None else "")
                 + ", so this checks consistency, not resolution.")
    lines.append(f"bin{binning}, {pixel:.2f} Å/px, box {box} ({box * pixel:.0f} Å).")
    lines.append("Read where the curve falls through 0.143 for the last time"
                 + (f"; it first drops below at {first:.1f} Å." if first and last
                    and not math.isclose(first, last) else "."))
    rise = pipeline["curve"].get("rise_toward_nyquist")
    if rise:
        lines.append(f"The curve rises again toward Nyquist, {rise['dip_correlation']:.2f}"
                     f" at {rise['dip_resolution_angstrom']:.1f} Å to "
                     f"{rise['peak_correlation']:.2f} at {rise['peak_resolution_angstrom']:.1f} Å.")
    for row, line in enumerate(lines):
        fig.text(0.09, 0.19 - row * 0.024, line, fontsize=10, color="#555555")

    out = args.out or pipeline_prefix.with_name(f"{pipeline_prefix.name}_fsc_figure")
    for suffix in (".png", ".pdf"):
        fig.savefig(out.with_name(out.name + suffix))
    print(f"font: {chosen_font}")
    print(f"wrote {out.with_name(out.name + '.png')} and .pdf")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
