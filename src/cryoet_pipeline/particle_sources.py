"""Where an MRA run's particles come from: the project's registry (docs/mra_from_registry.md).

`registry_particles(project, label, series, decoy_label)` reads, for each series in
the order given, the one PARTICLE_STACK artifact of `label` in the project's
registry: its Zarr array (one chunk per particle) and its STAR file of picks. It
gives every particle a tag and a row of a Dynamo table, and reads each particle's
voxels from its stack when asked. This is the conversion scripts/export_dynamo.py
made inside the script until 2026-11-05; the script now writes its files from this
module, so an export and a run from the registry cannot convert differently.

The conversion, as the export made it:

- tags run 1..N: the real particles series by series in the order given, each stack
  in its index order, then the decoys (stacks of `decoy_label`) in the same order;
  the series' number in that order is Dynamo's column 20;
- each pick's pose and shift go through `native_to_dynamo`;
- the missing wedge is one y-axis range per series, the views the particles'
  tomogram was built from (`reconstructed_tilt_range`), negated into Dynamo's ytilt
  (`dynamo_ytilt`);
- columns 24-26 hold the pick position plus 0.5, and column 36 the pixel size;
- the voxels are the stack's, unchanged.

The rows keep every digit; the export writes them with 12 significant digits.
"""

from __future__ import annotations

import hashlib
from collections.abc import Callable, Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np

from cryoet_pipeline.artifacts import ArtifactRegistry
from cryoet_pipeline.backends.dynamo_search import dynamo_ytilt, native_to_dynamo
from cryoet_pipeline.models import Artifact, ArtifactKind, TiltSeriesManifest
from cryoet_pipeline.particles import ParticleRecord, read_particle_star
from cryoet_pipeline.rotations import relion_euler_to_matrix
from cryoet_pipeline.runs import RunInput, canonical_json

TABLE_COLUMNS = 42
TAG, SHIFTS, ANGLES, TILTS = 0, slice(3, 6), slice(6, 9), slice(13, 15)


def particle_stack(registry: ArtifactRegistry, series: str, label: str | None) -> Artifact:
    """The one particle stack of `series` with label `label` (None: the unlabelled one)."""

    found = [
        artifact
        for artifact in registry.by_kind(ArtifactKind.PARTICLE_STACK)
        if artifact.covers(series) and artifact.variant == label
    ]
    if len(found) != 1:
        raise ValueError(
            f"{series}: expected one particle stack with variant {label!r}, found "
            f"{[a.id for a in found] or 'none'}"
        )
    return found[0]


def reconstructed_tilt_range(
    registry: ArtifactRegistry, stack: Artifact, manifest: TiltSeriesManifest
) -> tuple[tuple[float, float], str]:
    """The tilt range of the views the particles' tomogram was built from, and its source.

    The manifest lists every view collected, and alignment drops some: on the MDC
    EMPIAR-11538 series min_std_ratio cut the dark extremes, so TS_0306's tomogram
    holds -40..+58 of the manifest's -52..+68. The manifest's extremes would give
    Dynamo a wedge with data where the tomogram has none. Best are the solved
    angles reconstruction recorded (`tilt_angles_path`), which fine alignment has
    also shifted to make the specimen flat (+7 deg on ryr1 039); then the nominal
    angles of the views the tomogram kept (`included_z_values`); then the manifest.
    The caller says which it used. The range is in the tomogram's own angles; a
    Dynamo table gets it as Dynamo's ytilt, through `dynamo_ytilt`.
    """
    angle_by_z = {image.z_value: image.tilt_angle_deg for image in manifest.images}
    tomogram_id = stack.parameters.get("input_tomogram_id")
    if isinstance(tomogram_id, str):
        try:
            tomogram = registry.get(tomogram_id)
        except KeyError:
            tomogram = None
        solved = tomogram.parameters.get("tilt_angles_path") if tomogram is not None else None
        if isinstance(solved, str) and Path(solved).is_file():
            angles = [float(word) for word in Path(solved).read_text().split()]
            if angles:
                return (min(angles), max(angles)), "reconstruction's solved angles"
        included = tomogram.parameters.get("included_z_values") if tomogram is not None else None
        if isinstance(included, list):
            angles = [angle_by_z[z] for z in included if z in angle_by_z]
            if angles:
                return (float(min(angles)), float(max(angles))), "reconstruction"
    angles = manifest.tilt_angles_deg
    return (float(min(angles)), float(max(angles))), "manifest"


def decoy_kind(registry: ArtifactRegistry, stack: Artifact) -> str:
    """What a decoy stack's boxes are, in words for the report, from its picks and tomogram."""

    def recorded(key: str) -> Artifact | None:
        value = stack.parameters.get(key)
        try:
            return registry.get(value) if isinstance(value, str) else None
        except KeyError:
            return None

    tomogram, picks = recorded("input_tomogram_id"), recorded("picks_id")
    if tomogram is not None and tomogram.parameters.get("phase_randomised_from"):
        return "pytom picks in phase-randomised tomograms"
    if picks is not None and picks.parameters.get("picking_method") == "random_positions":
        return "boxes at random positions"
    return "decoy boxes"


def table_row(tag: int, tomogram: int, record: ParticleRecord, box: int,
              tilt_range: tuple[float, float], spacing: float) -> list[float]:
    """One Dynamo table row: pose from the native one, wedge, tomogram, position."""
    angles, shift = native_to_dynamo(
        relion_euler_to_matrix(record.rot_deg, record.tilt_deg, record.psi_deg),
        np.array([record.shift_x_px, record.shift_y_px, record.shift_z_px]),
        box,
    )
    row = [0.0] * TABLE_COLUMNS
    row[0], row[1], row[2] = tag, 1, 1  # tag, aligned, averaged
    row[3:6] = shift.tolist()  # dx dy dz
    row[6:9] = angles.tolist()  # tdrot tilt narot
    row[12] = 1  # ftype: single-axis wedge about y
    # ymintilt ymaxtilt: the tomogram's range negated, which is where Dynamo's
    # wedge has to put it for the volumes this pipeline stores (see dynamo_ytilt).
    row[13], row[14] = dynamo_ytilt(tilt_range)
    row[19] = tomogram
    row[21] = 1  # class
    row[23:26] = [record.x_px + 0.5, record.y_px + 0.5, record.z_px + 0.5]
    row[30] = tag  # original tag
    row[31] = 1
    row[33] = 1  # reference
    row[35] = spacing
    return row


def _file_sha256(path: Path) -> str:
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


@dataclass(frozen=True)
class StackParticle:
    """One particle: its tag, where it sits in its stack, and its pick."""

    tag: int
    tomogram: int
    """The series' number in the order given, Dynamo's column 20."""
    series: str
    stack_id: str
    index: int
    """Its slice in the stack, as the STAR's particleIndex gives it."""
    record: ParticleRecord
    decoy: bool


@dataclass
class RegistryParticles:
    """The particles of a project's registry an MRA run reads, by tag."""

    project: Path
    label: str | None
    series: tuple[str, ...]
    decoy_label: str | None
    table: np.ndarray
    """Dynamo table rows (42 columns), one per particle, by tag, every digit kept."""
    particles: list[StackParticle]
    """By tag."""
    box: int
    pixel_size: float
    stacks: dict[str, Artifact]
    """Each stack read, by id, in the order read."""
    metadata_paths: dict[str, Path]
    wedges: dict[str, list[float]]
    wedge_sources: dict[str, str]
    decoy_kinds: list[str]
    stack_fingerprints: dict[str, str]
    """What each stack says it is: its registry record, its STAR file and the wedge
    read for it; not its voxels."""
    _open: dict[str, Any] = field(default_factory=dict, repr=False)

    def __str__(self) -> str:
        decoys = f", decoys {self.decoy_label}" if self.decoy_label is not None else ""
        return (f"the registry of {self.project}, label {self.label}, series "
                f"{' '.join(self.series)}{decoys}")

    @property
    def tags(self) -> np.ndarray:
        return self.table[:, TAG].astype(int)

    @property
    def angles(self) -> np.ndarray:
        return self.table[:, ANGLES]

    @property
    def shifts(self) -> np.ndarray:
        return self.table[:, SHIFTS]

    @property
    def tilt_ranges(self) -> list[tuple[float, float]]:
        return [(float(low), float(high)) for low, high in self.table[:, TILTS]]

    def volume(self, tag: int) -> np.ndarray:
        """Particle `tag`'s voxels as its stack holds them, (z, y, x) float32."""

        import zarr

        particle = self.particles[int(tag) - 1]
        if particle.tag != int(tag):
            raise KeyError(f"no particle {tag}")
        stack = self._open.get(particle.stack_id)
        if stack is None:
            stack = self._open[particle.stack_id] = zarr.open_array(
                self.stacks[particle.stack_id].path, mode="r")
        return np.asarray(stack[particle.index], dtype=np.float32)

    def load(self, tag: int) -> np.ndarray:
        """Particle `tag` as the port computes with it, float64, as a particle file read
        back from an export would give it."""

        return self.volume(tag).astype(float)

    def fingerprint(self) -> str:
        """Every stack read, in the order read, by what it says it is."""

        return hashlib.sha256(canonical_json(list(self.stack_fingerprints.items()))
                              .encode()).hexdigest()

    def inputs(self) -> list[RunInput]:
        """One input per stack, for a run's record."""

        return [RunInput(slot=f"particle_stack {stack_id}", path=str(stack.path),
                         fingerprint=self.stack_fingerprints[stack_id])
                for stack_id, stack in self.stacks.items()]


def registry_particles(
    project: Path,
    label: str | None,
    series: Sequence[str],
    decoy_label: str | None = None,
    *,
    log: Callable[[str], None] = print,
) -> RegistryParticles:
    """The particles of `series` in `project`'s registry, as the module's header says."""

    project = Path(project).resolve()
    series = tuple(series)
    if not series:
        raise ValueError("no series given")
    if len(set(series)) != len(series):
        raise ValueError("a series is given twice")
    if decoy_label is not None and decoy_label == label:
        raise ValueError("the decoys' label must differ from the particles'")
    import zarr

    registry = ArtifactRegistry.load(project / "artifacts.json")

    rows: list[list[float]] = []
    particles: list[StackParticle] = []
    stacks: dict[str, Artifact] = {}
    metadata_paths: dict[str, Path] = {}
    fingerprints: dict[str, str] = {}
    wedges: dict[str, list[float]] = {}
    wedge_sources: dict[str, str] = {}
    decoy_kinds: set[str] = set()
    box: int | None = None
    spacing: float | None = None
    tag = 0
    # Every real particle first, so they keep the tags a set without decoys gives them.
    passes = [(label, False)] + ([(decoy_label, True)] if decoy_label is not None else [])
    jobs = [(variant, decoy, tomogram, name) for variant, decoy in passes
            for tomogram, name in enumerate(series, start=1)]
    for variant, decoy, tomogram, name in jobs:
        stack = particle_stack(registry, name, variant)
        manifest = TiltSeriesManifest.model_validate_json(
            (project / "manifests" / f"{name}.json").read_text()
        )
        tilt_range, source = reconstructed_tilt_range(registry, stack, manifest)
        if not tilt_range[0] < 0 < tilt_range[1]:
            raise ValueError(f"{name}: tilt range {tilt_range} does not span zero")
        if decoy:
            # Decoys are cut from the same tomogram, or from a phase-randomised copy of
            # it that keeps its tilt angles, so they share its wedge.
            if list(tilt_range) != wedges[name]:
                raise ValueError(f"{name}: the decoys' wedge {tilt_range} differs from the "
                                 f"real particles' {wedges[name]}; not the same tomogram?")
            decoy_kinds.add(decoy_kind(registry, stack))
        else:
            wedges[name] = list(tilt_range)
            wedge_sources[name] = source
        low, high = tilt_range
        log(f"{name} {'decoys' if decoy else 'particles'}: wedge {low:+.1f} to {high:+.1f} "
            f"deg, from the {source}")
        star = Path(str(stack.parameters["metadata_path"]))
        records = read_particle_star(star)
        shape = tuple(zarr.open_array(stack.path, mode="r").shape)
        if shape[0] != len(records):
            raise ValueError(f"{name}: {shape[0]} particles in the stack, {len(records)} rows "
                             f"in {star}")
        this_box, this_spacing = int(shape[1]), float(stack.pixel_spacing_angstrom or 0)
        if box is None:
            box, spacing = this_box, this_spacing
        elif (this_box, this_spacing) != (box, spacing):
            raise ValueError(f"{name}: box {this_box} at {this_spacing} A differs from "
                             f"{box} at {spacing} A of the first series")
        stacks[stack.id] = stack
        metadata_paths[stack.id] = star
        fingerprints[stack.id] = hashlib.sha256(canonical_json({
            "record": stack.model_dump(mode="json"), "star": _file_sha256(star),
            "wedge": list(tilt_range),
        }).encode()).hexdigest()
        for record in records:
            tag += 1
            rows.append(table_row(tag, tomogram, record, this_box, tilt_range, this_spacing))
            particles.append(StackParticle(tag, tomogram, name, stack.id, record.index, record,
                                           decoy))
    if not rows or box is None or spacing is None:
        raise ValueError("no particles in the stacks given")
    if not spacing > 0:
        raise ValueError(f"the stacks record no pixel size ({spacing})")
    return RegistryParticles(
        project=project, label=label, series=series, decoy_label=decoy_label,
        table=np.array(rows, dtype=float), particles=particles, box=box, pixel_size=spacing,
        stacks=stacks, metadata_paths=metadata_paths, wedges=wedges,
        wedge_sources=wedge_sources, decoy_kinds=sorted(decoy_kinds),
        stack_fingerprints=fingerprints,
    )


__all__ = ["RegistryParticles", "StackParticle", "TABLE_COLUMNS", "decoy_kind",
           "particle_stack", "reconstructed_tilt_range", "registry_particles", "table_row"]
