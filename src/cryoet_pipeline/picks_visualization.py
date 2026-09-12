"""Everything one pick-inspection session needs, for ChimeraX and for 3dmod.

Picks get judged in two ways and neither answers the other's question. ChimeraX
with ArtiaX shows the whole set at once, each pick drawn as the template at the
orientation found for it and, when there is one, a segmentation surface under
them: that is where a systematically wrong orientation, or a lattice picked as a
cloud, becomes visible. 3dmod shows one slice with the picks drawn on it, which
is the only way to answer whether a particular pick sits on actual density.

Both need the same tomogram as an MRC, and the tomogram is the largest file in
the project. So this writes one directory with one conversion in it, rather than
two exports each carrying their own copy.

The two viewers disagree with the pipeline about what a coordinate means, and
both disagreements are silent — nothing errors, the picks are simply in the
wrong places:

ArtiaX registers two readers for `.star`: "RELION STAR file" (RELION 3.1
conventions, `rlnCoordinateX/Y/Z` in pixels) and "RELION5 STAR file" (centered
coordinates in Angstrom). Both claim the `.star` suffix and both answer to the
nickname `star`, so the generated ChimeraX script names the format explicitly.
pytom-match-pick writes the RELION 3.1 flavour, and letting ChimeraX guess can
put every particle in the wrong place without any error.

IMOD model coordinates run from 0 to NX in X and Y, but from -0.5 to NZ - 0.5 in
Z, so the centre of voxel i sits at i + 0.5 in X and Y and at i in Z. Picks are
voxel indices — `_crop_start` in the extraction backend rounds them to one — so
X and Y gain half a voxel on the way into the model file and Z does not. Stated
by IMOD's own binary model spec ("Coordinates range from 0 to NX in X and 0 to
NY in Y, but from -0.5 to NZ - 0.5 in Z"), and confirmed against the library
rather than the prose: painting a model sphere onto a volume with `imodmop`
puts the paint on the voxels this offset predicts.
"""

from __future__ import annotations

import shutil
from dataclasses import dataclass
from pathlib import Path
from typing import Any, cast

import mrcfile  # type: ignore[import-untyped]
import numpy as np
import zarr

from cryoet_pipeline.backends.alignment import (
    CommandRunner,
    imod_environment,
    resolve_imod_executable,
    run_command,
)
from cryoet_pipeline.backends.command_log import write_command_log
from cryoet_pipeline.backends.protocols import BackendContext
from cryoet_pipeline.models import Artifact, ArtifactKind
from cryoet_pipeline.particles import read_star_loop

CHIMERAX_STAR_FORMAT = "relion"
"""ArtiaX nickname for the RELION 3.1 star reader, as opposed to `relion5`."""

IMOD_XY_VOXEL_CENTRE_OFFSET = 0.5
"""Half a voxel, added to X and Y but not Z, to reach an IMOD model coordinate.

See the module docstring: IMOD's Z is a section index and its X and Y are not.
"""

_DEFAULT_SPHERE_RADIUS_PX = 5
"""What a pick is drawn as when nothing in the registry says how big it is.

Small enough to stay a marker rather than a blanket over the slice, and the
command prints where the number came from so a wrong one is visible rather than
mistaken for a measurement.
"""


@dataclass(frozen=True)
class SphereRadius:
    """How big to draw a pick in 3dmod, and where that number came from."""

    px: int
    source: str


@dataclass(frozen=True)
class PicksVisualization:
    """Files written for one pick-inspection session."""

    directory: Path
    tomogram_path: Path
    picks_path: Path
    chimerax_script_path: Path
    imod_points_path: Path
    imod_model_path: Path
    imod_log_path: Path
    pick_count: int
    sphere_radius: SphereRadius
    template_path: Path | None = None
    segmentation_path: Path | None = None


def resolve_sphere_radius(
    *,
    requested_px: int | None,
    template_mask: Artifact | None,
    voxel_size_angstrom: float | None,
) -> SphereRadius:
    """Pick the radius to draw picks at, and say what it was derived from.

    The template mask is the one place in the registry that already holds a
    particle radius — it is the radius the search itself was run with — so a
    pick drawn at that size is drawn at the size it was looked for. The mask
    lives at the template's voxel size, which is normally the tomogram's but is
    not guaranteed to be, so it is rescaled rather than assumed.

    A tomogram with no voxel size gets no rescaling rather than an error here:
    the export refuses that tomogram anyway, with a message about the picks
    rather than about their radius.
    """

    if requested_px is not None:
        if requested_px < 1:
            raise ValueError("sphere radius must be at least one voxel")
        return SphereRadius(px=requested_px, source="as given")

    if template_mask is not None:
        mask_radius = template_mask.parameters.get("mask_radius_px")
        mask_spacing = template_mask.pixel_spacing_angstrom
        if isinstance(mask_radius, (int, float)) and not isinstance(mask_radius, bool):
            radius_px = float(mask_radius)
            source = f"the template mask radius, {radius_px:.1f} px"
            if (
                mask_spacing is not None
                and mask_spacing > 0.0
                and voxel_size_angstrom is not None
                and voxel_size_angstrom > 0.0
                and mask_spacing != voxel_size_angstrom
            ):
                radius_px *= mask_spacing / voxel_size_angstrom
                source += (
                    f" at {mask_spacing:.3f} A/px, "
                    f"rescaled to {voxel_size_angstrom:.3f} A/px"
                )
            return SphereRadius(px=max(1, int(round(radius_px))), source=source)

    return SphereRadius(
        px=_DEFAULT_SPHERE_RADIUS_PX,
        source="a default, since no template mask radius was registered",
    )


def write_picks_visualization(
    *,
    tomogram: Artifact,
    picks: Artifact,
    out: Path,
    context: BackendContext,
    sphere_radius: SphereRadius,
    template: Artifact | None = None,
    segmentation: Artifact | None = None,
    overwrite: bool = False,
    command_runner: CommandRunner | None = None,
) -> PicksVisualization:
    """Write everything one inspection session needs, into one directory.

    Points alone answer little. Attaching the template shows each pick with the
    orientation that was found for it, and a segmentation surface gives the
    picks something to sit on or miss — a tomogram slice shows one plane at a
    time and says nothing about the set as a whole. Both are optional, and both
    are converted here because ChimeraX reads MRC, not the canonical Zarr.

    point2model is resolved before anything is written: it is the one step that
    can fail for a reason outside this project, and failing after a tomogram has
    been converted would leave a directory that looks finished and is not.
    """

    if tomogram.kind not in {ArtifactKind.TOMOGRAM, ArtifactKind.DENOISED_TOMOGRAM}:
        raise ValueError(f"expected a tomogram artifact, got {tomogram.kind}")
    if picks.kind is not ArtifactKind.PICKS:
        raise ValueError(f"expected a picks artifact, got {picks.kind}")
    if not picks.path.is_file():
        raise FileNotFoundError(f"picks file not found: {picks.path}")
    voxel_size = tomogram.pixel_spacing_angstrom
    if voxel_size is None or voxel_size <= 0.0:
        raise ValueError("tomogram voxel spacing is required so ArtiaX scales picks correctly")

    runner = command_runner or run_command
    point2model = resolve_imod_executable("point2model", context)

    tilt_series_id = str(tomogram.parameters.get("tilt_series_id", "tomogram"))
    directory = out / "picks_view" / tilt_series_id
    tomogram_path = directory / f"{tilt_series_id}.mrc"
    picks_path = directory / f"{tilt_series_id}_particles.star"
    script_path = directory / f"{tilt_series_id}_open_in_chimerax.cxc"
    points_path = directory / f"{tilt_series_id}_picks.txt"
    model_path = directory / f"{tilt_series_id}_picks.mod"
    log_path = directory / f"{tilt_series_id}_point2model.log"

    template_path = directory / f"{tilt_series_id}_template.mrc" if template else None
    segmentation_path = (
        directory / f"{tilt_series_id}_segmentation.mrc" if segmentation else None
    )

    existing = [
        path
        for path in (
            tomogram_path,
            picks_path,
            script_path,
            points_path,
            model_path,
            template_path,
            segmentation_path,
        )
        if path is not None and path.exists()
    ]
    if existing and not overwrite:
        joined = ", ".join(str(path) for path in existing)
        raise FileExistsError(f"picks visualization already exists: {joined}")

    coordinates = _read_pick_coordinates(picks.path)

    directory.mkdir(parents=True, exist_ok=True)
    _write_tomogram_mrc(tomogram.path, tomogram_path, voxel_size_angstrom=voxel_size)
    shutil.copyfile(picks.path, picks_path)
    template_statistics: _VolumeStatistics | None = None
    if template is not None and template_path is not None:
        template_statistics = _write_tomogram_mrc(
            template.path, template_path, voxel_size_angstrom=voxel_size
        )
    if segmentation is not None and segmentation_path is not None:
        _write_tomogram_mrc(
            segmentation.path, segmentation_path, voxel_size_angstrom=voxel_size
        )
    script_path.write_text(
        _chimerax_script(
            tomogram_path.name,
            picks_path.name,
            voxel_size_angstrom=voxel_size,
            template_name=template_path.name if template_path else None,
            template_statistics=template_statistics,
            segmentation_name=segmentation_path.name if segmentation_path else None,
        )
    )

    # One decision, used by both the file and the command. They cannot be
    # allowed to disagree: point2model reads a four-number line as a value only
    # when `-values` says so, and as a CONTOUR NUMBER followed by X, Y, Z
    # otherwise — silently, exit code 0, with the points scattered across as
    # many contours as there were distinct leading numbers.
    with_values = all(pick.score is not None for pick in coordinates)
    points_path.write_text(_imod_point_file(coordinates, with_values=with_values))
    command = _point2model_command(
        point2model,
        points_name=points_path.name,
        model_name=model_path.name,
        tomogram_name=tomogram_path.name,
        sphere_radius_px=sphere_radius.px,
        with_values=with_values,
    )
    result = runner(command, cwd=directory, env=imod_environment(point2model, context))
    write_command_log(log_path, (command, result))
    if result.returncode != 0:
        detail = (result.stderr or result.stdout or "no process output").strip()
        raise RuntimeError(
            f"point2model failed with exit code {result.returncode}: {detail}; see {log_path}"
        )
    # The exit code is the tool's claim; the file is the deliverable. point2model
    # did exit non-zero on the malformed input it was tried with, so this is a
    # guard rather than a known failure — but a model that is not there should
    # be found here and not by 3dmod.
    if not model_path.is_file():
        raise RuntimeError(
            f"point2model exited 0 but wrote no model at {model_path}; see {log_path}"
        )

    return PicksVisualization(
        directory=directory,
        template_path=template_path,
        segmentation_path=segmentation_path,
        tomogram_path=tomogram_path,
        picks_path=picks_path,
        chimerax_script_path=script_path,
        imod_points_path=points_path,
        imod_model_path=model_path,
        imod_log_path=log_path,
        pick_count=len(coordinates),
        sphere_radius=sphere_radius,
    )


@dataclass(frozen=True)
class _PickCoordinate:
    """One pick, as the STAR file gives it: voxel indices and its score."""

    x_px: float
    y_px: float
    z_px: float
    score: float | None


def _read_pick_coordinates(path: Path) -> list[_PickCoordinate]:
    """Coordinates and scores from a picks STAR, in the file's own order.

    Only the three coordinate columns are required. The score is optional
    because a picks file that came from somewhere other than a template match
    need not carry one, and its absence should cost the colouring rather than
    the whole model.
    """

    columns, rows = read_star_loop(path)
    required = ("rlnCoordinateX", "rlnCoordinateY", "rlnCoordinateZ")
    missing = [name for name in required if name not in columns]
    if missing:
        raise ValueError(f"{path} is missing required STAR columns: {missing}")
    x_index, y_index, z_index = (columns.index(name) for name in required)
    score_index = columns.index("rlnLCCmax") if "rlnLCCmax" in columns else None
    return [
        _PickCoordinate(
            x_px=float(fields[x_index]),
            y_px=float(fields[y_index]),
            z_px=float(fields[z_index]),
            score=float(fields[score_index]) if score_index is not None else None,
        )
        for fields in rows
    ]


def _imod_point_file(coordinates: list[_PickCoordinate], *, with_values: bool) -> str:
    """The point list point2model reads, in IMOD model coordinates.

    Half a voxel is added in X and Y and none in Z; the module docstring says
    why. The score goes last on the line, which is where `-values 1` looks for
    it — and `with_values` is passed in rather than decided here so that the
    line the file has and the line the command expects come from one decision.
    """

    lines = []
    for pick in coordinates:
        fields = [
            f"{pick.x_px + IMOD_XY_VOXEL_CENTRE_OFFSET:.3f}",
            f"{pick.y_px + IMOD_XY_VOXEL_CENTRE_OFFSET:.3f}",
            f"{pick.z_px:.3f}",
        ]
        if with_values and pick.score is not None:
            fields.append(f"{pick.score:.6f}")
        lines.append(" ".join(fields))
    return "\n".join(lines) + "\n"


def _point2model_command(
    executable: Path,
    *,
    points_name: str,
    model_name: str,
    tomogram_name: str,
    sphere_radius_px: int,
    with_values: bool,
) -> list[str]:
    """Build the conversion, with the four options that make it viewable.

    The three files are named without their directory because the process is
    run inside it. Spelling them out would break the moment `--out` is
    relative — which is how the README's own example invokes this: the child
    would resolve `outputs/dev/picks_view/TS_01/...` against a cwd that already
    ends in `outputs/dev/picks_view/TS_01`, and point2model would report a file
    it cannot open at a path that has the same tail twice.

    `-scat` because the picks are separate particles: without it 3dmod joins
    them into one contour and draws the polygon rather than the points.
    `-sphere` because a scattered point with no size is a single pixel, and
    because a sphere is what shows on the neighbouring slices as a shrinking
    circle — which is how a pick that is one slice off becomes visible.
    `-image` so the model carries the volume's size and pixel spacing, which is
    what lets 3dmod load it onto the tomogram without a size mismatch.
    And `-values` with `-flags 3`, when there are scores, so each pick carries
    its own and is coloured by it: a low-scoring pick and a high-scoring one
    otherwise look the same.
    """

    command = [
        str(executable),
        "-input",
        points_name,
        "-output",
        model_name,
        "-scat",
        "-sphere",
        str(sphere_radius_px),
        "-image",
        tomogram_name,
        "-name",
        "picks",
        "-color",
        "255,0,0",
    ]
    if with_values:
        command += ["-values", "1", "-flags", "3"]
    return command


def _chimerax_script(
    tomogram_name: str,
    picks_name: str,
    *,
    voxel_size_angstrom: float,
    template_name: str | None = None,
    template_statistics: _VolumeStatistics | None = None,
    segmentation_name: str | None = None,
) -> str:
    """Open everything at one scale, with each pick shown as the template.

    Four things have to be said explicitly or the result is quietly wrong.
    ArtiaX registers two readers for `.star` and both answer to the suffix, so
    the RELION 3.1 one is named — the RELION5 reader expects centred coordinates
    in Angstrom. ArtiaX places a particle at `coordinate * originScaleFactor`
    with the factor defaulting to one, while the coordinates are in tomogram
    voxels, so without it every pick lands at a fraction of its true position.
    `artiax attach` re-opens the file beneath the particle list rather than
    linking the model that was passed to it, so afterwards the template answers
    to a new number and the one it was opened under no longer exists. And the
    template is inverted to match dark particles, so ChimeraX's positive default
    level wraps the background instead of the density.

    The volumes are opened before ArtiaX starts, because they are the ones the
    later commands have to name and only a plain `open` into an empty session
    gives them predictable numbers.
    """

    extras = [name for name in (template_name, segmentation_name) if name is not None]
    numbers = {name: index + 1 for index, name in enumerate(extras)}
    artiax = len(extras) + 1
    particle_list = f"#{artiax}.2.1"

    lines = [
        "# Open with: chimerax <this file>, in a fresh session: the model numbers",
        "# below assume nothing else is open.",
        "# Requires the ArtiaX bundle (toolshed install ArtiaX).",
    ]
    lines += [f"open {name}" for name in extras]
    lines += [
        "artiax start",
        "# Through ArtiaX rather than as a plain volume, so the tomogram is under",
        "# its control and appears in the Tomograms list.",
        f"artiax open tomo {tomogram_name}",
        "# Two readers claim .star; the RELION5 one expects centred coordinates",
        "# in Angstrom, which these are not.",
        f"open {picks_name} format {CHIMERAX_STAR_FORMAT}",
        "# Coordinates are in tomogram voxels, and ArtiaX scales them by this",
        "# factor, which defaults to one.",
        f"artiax particles {particle_list} originScaleFactor {voxel_size_angstrom:.6f}",
    ]
    if segmentation_name is not None:
        lines += [
            "",
            "# The membranes as a surface, so the picks have something to sit on",
            "# or miss. A tomogram slice shows one plane; this shows the set.",
            f"volume #{numbers[segmentation_name]} level 0.5 style surface transparency 0.5",
        ]
    if template_name is not None:
        # A template with no statistics cannot have its level derived, so the
        # script says so rather than printing a number nothing stands behind.
        if template_statistics is None:
            lines += [
                "",
                "# The template's statistics were not available when this was written,",
                "# so the level below is a placeholder. Set it from the map's own",
                "# histogram: an inverted template's density is its LOW tail, so pick",
                "# a level below the mean, and re-run the attach afterwards.",
                f"volume #{numbers[template_name]} level -1.0 style surface",
                f"artiax attach #{numbers[template_name]} toParticleList {particle_list}",
                f"artiax show {particle_list} surfaces",
                f"artiax hide {particle_list} markers",
                f"artiax hide {particle_list} axes",
            ]
            lines.append("")
            return "\n".join(lines)

        level = template_statistics.level
        template_minimum = template_statistics.minimum
        template_maximum = template_statistics.maximum
        template_mean = template_statistics.mean
        template_sd = template_statistics.sd
        lines += [
            "",
            "# Each pick drawn as the template at the orientation found for it,",
            "# which is what makes a wrong orientation visible.",
            "# Contour first: attaching copies the surface as it stands, so a level",
            "# set afterwards changes the original and not the copies drawn at each",
            "# particle.",
            "#",
            "# The level is NEGATIVE because the template is inverted to match dark",
            "# particles, so its density is the low tail rather than the high one.",
            "# Stated as numbers rather than asserted, because the claim is easy to",
            "# doubt and impossible to check from the script otherwise:",
            f"#   template values run {template_minimum:.5f} to {template_maximum:.5f}",
            f"#   mean {template_mean:.5f}, sd {template_sd:.5f}",
            f"#   level = mean - 2 sd = {level:.5f}",
            "#",
            "# Do NOT flip the map with `volume scale ... factor -1` instead. That",
            "# creates a NEW model rather than changing this one, so the numbering",
            "# every line below depends on shifts, and the attach still copies the",
            "# unflipped surface. The negative level is the inversion.",
            "#",
            "# If the copies drawn on the particles look empty or crumbly, the level",
            "# below did not take effect before the attach: ChimeraX opens a map at",
            "# its own default level, which for an inverted template sits in the",
            "# positive tail where there is almost nothing. Re-run this file in a",
            "# fresh session.",
            f"volume #{numbers[template_name]} level {level:.4f} style surface",
            f"artiax attach #{numbers[template_name]} toParticleList {particle_list}",
            "# Attaching does not display: the list keeps drawing its own markers",
            "# and the attached surface stays hidden.",
            f"artiax show {particle_list} surfaces",
            f"artiax hide {particle_list} markers",
            f"artiax hide {particle_list} axes",
            "# To try another level, set it on the model above and attach again —",
            f"# somewhere between {template_minimum:.5f} and {template_mean:.5f}, since",
            "# further below the mean shows less and above it shows background.",
        ]
    lines.append("")
    return "\n".join(lines)


@dataclass(frozen=True)
class _VolumeStatistics:
    """What a contour level is derived from, kept so the script can show it."""

    minimum: float
    maximum: float
    mean: float
    sd: float

    @property
    def level(self) -> float:
        """Two standard deviations below the mean.

        Below rather than above because an inverted template's density is its
        low tail. Two rather than some other number because it is the
        conventional starting contour and this is a viewing aid, not a
        measurement — the script says how to change it.
        """

        return self.mean - 2.0 * self.sd


def _write_tomogram_mrc(
    source: Path,
    destination: Path,
    *,
    voxel_size_angstrom: float,
) -> _VolumeStatistics:
    """Convert to MRC and return the statistics a contour level comes from.

    ChimeraX picks a positive default from the histogram, which is the wrong
    side for a template inverted to match a tomogram's dark particles: the
    surface then wraps the background instead of the particle.

    The four numbers rather than the level alone, so the written script can show
    its working. A reader who doubts that the template is inverted — a
    reasonable doubt, since nothing in a `.cxc` says so — can otherwise only
    take the level's sign on trust.

    The array is written in the canonical ZYX order it is stored in, which is
    also what makes the picks land: they are indices into this array, and the
    IMOD `.rec` copy the reconstruction leaves behind is in a different order.
    """
    if source.is_dir():
        array = cast(Any, zarr.open(source, mode="r"))
        data = np.asarray(array[:], dtype=np.float32)
    else:
        with mrcfile.open(source, permissive=True) as mrc:
            data = np.array(mrc.data, dtype=np.float32)
    if data.ndim != 3:
        raise ValueError(f"expected a 3D tomogram, got shape {data.shape}: {source}")
    with mrcfile.new(destination, overwrite=True) as mrc:
        mrc.set_data(data)
        mrc.voxel_size = voxel_size_angstrom
    return _VolumeStatistics(
        minimum=float(data.min()),
        maximum=float(data.max()),
        mean=float(data.mean()),
        sd=float(data.std()),
    )
