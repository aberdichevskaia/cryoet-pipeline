#!/usr/bin/env python3
"""Print what a tilt-series has produced so far: artifacts, then every QC record.

    python3 scripts/show_series.py --project outputs/ribo --series TS_0005
    python3 scripts/show_series.py --project outputs/ribo --series TS_0005 --kind alignment

Read-only. Nothing here knows about individual stages: QC documents are rendered
generically, so a stage added tomorrow shows up without touching this file.

Disposable, like scripts/run_series.py. It is a draft of `cryoet describe`
(roadmap section 8) whose shape is being learned by use; promote it once the
fields actually read every day are known, and delete this then.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

# Printed on their own line by the artifact table, so not repeated in QC blocks.
_QC_SKIP = frozenset({"schema_version", "tilt_series_id", "backend", "status", "warnings"})
_NUMERIC = (int, float)

# Suffixes 3dmod can actually open. A Zarr store cannot be opened by IMOD, and
# suggesting it would be a confidently wrong instruction.
_IMOD_SUFFIXES = frozenset({".st", ".mrc", ".rec", ".ali", ".preali", ".map"})


def load(path: Path) -> Any:
    try:
        return json.loads(path.read_text())
    except (OSError, json.JSONDecodeError):
        return None


def belongs_to(artifact: dict[str, Any], series: str) -> bool:
    if artifact.get("parameters", {}).get("tilt_series_id") == series:
        return True
    # Cross-series artifacts (a merged particle set) carry a list instead.
    scope = artifact.get("parameters", {}).get("tilt_series_ids")
    if isinstance(scope, list) and series in scope:
        return True
    return str(artifact.get("id", "")).startswith(f"{series}:")


def render_value(value: Any) -> str | None:
    """One line for a QC field, or None when it is not worth a line."""

    if isinstance(value, bool) or isinstance(value, str):
        return str(value)
    if isinstance(value, _NUMERIC):
        return f"{value:.6g}" if isinstance(value, float) else str(value)
    if isinstance(value, list):
        if not value:
            return "[]"
        if all(isinstance(item, _NUMERIC) and not isinstance(item, bool) for item in value):
            return f"{min(value):.6g}..{max(value):.6g}  (n={len(value)})"
        if all(isinstance(item, str) for item in value):
            return ", ".join(value) if len(value) <= 4 else f"{len(value)} entries"
        return f"{len(value)} entries"
    if isinstance(value, dict):
        return ", ".join(f"{k}={v}" for k, v in value.items()) if len(value) <= 4 else (
            f"{len(value)} keys"
        )
    return None


def print_artifacts(artifacts: list[dict[str, Any]]) -> None:
    header = f"{'kind':<24} {'variant':<10} {'shape':<22} {'apix':>7} {'bin':>4}  id"
    print(header)
    print("-" * len(header))

    def row(art: dict[str, Any], identifier: str) -> str:
        shape = art.get("shape")
        apix = art.get("pixel_spacing_angstrom")
        return (
            f"{str(art.get('kind')):<24} "
            f"{str(art.get('variant') or '-'):<10} "
            f"{str(tuple(shape) if shape else '-'):<22} "
            f"{(f'{apix:.3f}' if isinstance(apix, _NUMERIC) else '-'):>7} "
            f"{str(art.get('binning') or '-'):>4}  "
            f"{identifier}"
        )

    # One artifact per tilt image would otherwise bury everything else. Collapse a
    # run of identical-looking entries of one kind into a single counted line.
    index = 0
    while index < len(artifacts):
        art = artifacts[index]
        signature = (
            art.get("kind"),
            art.get("variant"),
            tuple(art.get("shape") or ()),
            art.get("pixel_spacing_angstrom"),
            art.get("binning"),
        )
        run_end = index + 1
        while run_end < len(artifacts):
            other = artifacts[run_end]
            if (
                other.get("kind"),
                other.get("variant"),
                tuple(other.get("shape") or ()),
                other.get("pixel_spacing_angstrom"),
                other.get("binning"),
            ) != signature:
                break
            run_end += 1

        count = run_end - index
        if count > 3:
            first, last = artifacts[index]["id"], artifacts[run_end - 1]["id"]
            print(row(art, f"{count} x  {first} .. {last}"))
        else:
            for entry in artifacts[index:run_end]:
                print(row(entry, str(entry.get("id"))))
        index = run_end


def qc_documents(artifacts: list[dict[str, Any]]) -> list[tuple[str, Path]]:
    """QC reports, whether registered as their own artifact or pointed at."""

    found: list[tuple[str, Path]] = []
    seen: set[Path] = set()
    for art in artifacts:
        candidates = []
        if art.get("kind") == "qc":
            candidates.append(art.get("path"))
        candidates.append(art.get("parameters", {}).get("qc_path"))
        for candidate in candidates:
            if not isinstance(candidate, str):
                continue
            path = Path(candidate)
            if path in seen or not path.is_file():
                continue
            seen.add(path)
            found.append((str(art.get("id")), path))
    return found


def print_qc(label: str, path: Path) -> None:
    document = load(path)
    if not isinstance(document, dict):
        return
    status = document.get("status", "-")
    print(f"\n{label}")
    print(f"  {'status':<44} {status}")
    for warning in document.get("warnings") or []:
        print(f"  {'warning':<44} {warning}")
    for key, value in document.items():
        if key in _QC_SKIP:
            continue
        rendered = render_value(value)
        if rendered is not None:
            print(f"  {key:<44} {rendered}")
    print(f"  {'(file)':<44} {path}")


def print_alignments(artifacts: list[dict[str, Any]]) -> None:
    """Alignments are not QC documents but carry the numbers worth reading."""

    for art in artifacts:
        if art.get("kind") != "alignment":
            continue
        document = load(Path(str(art.get("path"))))
        if not isinstance(document, dict):
            continue
        transforms = document.get("transforms") or []
        print(f"\n{art.get('id')}")
        print(f"  {'stage':<44} {document.get('stage')}")
        print(f"  {'tilt_axis_angle_deg':<44} {document.get('tilt_axis_angle_deg')}")
        print(f"  {'input_binning':<44} {document.get('input_binning')}")
        print(f"  {'transform_semantics':<44} {document.get('transform_semantics')}")
        print(f"  {'transforms':<44} {len(transforms)}")
        for axis in ("shift_x_px", "shift_y_px"):
            values = [t[axis] for t in transforms if axis in t]
            if values:
                print(f"  {axis:<44} {min(values):+.2f}..{max(values):+.2f} (unbinned)")
        excluded = document.get("excluded_z_values") or []
        print(f"  {'excluded_z_values':<44} {excluded if excluded else '[]'}")


# What a person usually wants to open, and the command that would create it if it
# is not there yet. Ordered as the pipeline runs, so the first missing line is the
# next thing to do.
_EXPECTED: tuple[tuple[str, str, str], ...] = (
    ("coarse-aligned preview", "qc:coarse_alignment:preview", "qc-coarse-alignment"),
    ("fine-aligned stack", "aligned_tilt_stack:final", "build-final-aligned-stack"),
    ("tomogram", "tomogram", "reconstruct-tomogram"),
    ("picks in ChimeraX", "chimerax_script", "export-chimerax"),
    ("particle stack", "particle_stack", "extract-particles"),
    ("average", "subtomogram_average", "average-particles"),
)


def viewable(project: Path, series: str, artifacts: list[dict[str, Any]]) -> None:
    """Name the files worth opening, with the command that opens them.

    Alignment steps produce transforms, not images, which is the usual source of
    'which of these do I look at' — so say plainly when the thing wanted does not
    exist yet and which step would make it.
    """

    print("\nwhat to look at")
    found: set[str] = set()

    for art in artifacts:
        kind = art.get("kind")
        params = art.get("parameters", {})
        path = str(art.get("path", ""))

        if kind in {"tilt_stack", "aligned_tilt_stack"}:
            is_preview = "coarse_alignment:preview" in str(art.get("id"))
            purpose = params.get("purpose") or ("coarse preview" if is_preview else "imported")
            if Path(path).suffix in _IMOD_SUFFIXES:
                print(f"  3dmod {path}")
                print(f"      {kind} ({purpose})")
            else:
                print(f"      {kind} ({purpose}), not openable in 3dmod: {path}")
            if purpose == "final_alignment":
                found.add("aligned_tilt_stack:final")
            if is_preview:
                found.add("qc:coarse_alignment:preview")

        elif kind in {"tomogram", "denoised_tomogram"}:
            # The canonical store is Zarr, which 3dmod cannot open; the IMOD .rec
            # is the compatibility copy, and it is in YZX section order.
            rec = params.get("imod_rec_path")
            if isinstance(rec, str):
                print(f"  3dmod {rec}")
                print(f"      {kind}, IMOD .rec ({params.get('imod_rec_axis_order', '?')})")
            print(f"      canonical Zarr, not openable in 3dmod: {path}")
            found.add("tomogram")

        elif kind in {"template", "subtomogram_average"}:
            opener = "ChimeraX" if Path(path).suffix in _IMOD_SUFFIXES else "(not a map)"
            print(f"  {opener} {path}")
            print(f"      {kind}")
            found.add(kind)

        elif kind == "particle_stack":
            print(f"      particle stack (Zarr): {path}")
            found.add(kind)

        elif kind == "ctf_model":
            diagnostic = params.get("diagnostic_path")
            if isinstance(diagnostic, str):
                print(f"  3dmod {diagnostic}")
                print("      CTF diagnostic: measured Thon rings against the fitted ones")
            print(f"      IMOD .defocus file: {path}")

        elif kind == "qc":
            document = load(Path(path))
            slices = document.get("central_slice_paths") if isinstance(document, dict) else None
            if isinstance(slices, dict):
                for axis, slice_path in sorted(slices.items()):
                    print(f"      central slice {axis}: {slice_path}")

    for script in sorted((project).rglob(f"{series}_open_in_chimerax.cxc")):
        print(f"  chimerax {script}")
        print("      picks over their tomogram (pins the RELION reader)")
        found.add("chimerax_script")

    for label, token, command in _EXPECTED:
        if token not in found:
            print(f"  (no {label} yet — run {command})")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--project", required=True, type=Path)
    parser.add_argument("--series", required=True)
    parser.add_argument("--kind", default=None, help="show only this artifact kind")
    parser.add_argument("--registry", type=Path, default=None)
    args = parser.parse_args()

    registry_path = args.registry or (args.project / "artifacts.json")
    payload = load(registry_path)
    if not isinstance(payload, dict):
        print(f"no readable registry at {registry_path}")
        return 1

    series_artifacts = [a for a in payload.get("artifacts", []) if belongs_to(a, args.series)]
    # The --kind filter narrows the table only. "what to look at" is about the whole
    # series, and filtering it would report things as absent that are merely hidden.
    artifacts = (
        [a for a in series_artifacts if a.get("kind") == args.kind]
        if args.kind
        else series_artifacts
    )
    if not artifacts:
        print(f"no artifacts for {args.series} in {registry_path}")
        return 1

    manifest = load(args.project / "manifests" / f"{args.series}.json")
    if isinstance(manifest, dict):
        angles = [i["tilt_angle_deg"] for i in manifest.get("images", [])]
        if angles:
            print(
                f"{args.series}: {len(angles)} tilts, "
                f"{min(angles):+.2f}..{max(angles):+.2f} deg, "
                f"raw apix {manifest.get('raw_pixel_spacing_angstrom')}"
            )
        for note in manifest.get("notes") or []:
            print(f"  note: {note}")
        print()

    print_artifacts(artifacts)
    print_alignments(artifacts)
    viewable(args.project, args.series, series_artifacts)
    for label, path in qc_documents(artifacts):
        print_qc(label, path)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
