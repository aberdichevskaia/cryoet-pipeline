"""Fine alignment by AreTomo2, without fiducials.

Lamellae carry no gold, and IMOD's patch tracking did not hold on the first one
the pipeline met: on EMPIAR-11830 01082023_BrnoKrios_Arctis_WebUI_Position_1 the
patch route left a mean residual of 18.5 nm (job 3901834, 7 November 2026; 17.7
nm when repeated off the cluster), ten times the 1.5 nm the QC allows. The
authors aligned the same series with AreTomo and reconstructed it with IMOD:
their deposited AreTomo/ folder holds one global `.xf`, the `.tlt` with the tilt
offset AreTomo measured, and the newst.com and tilt.com that used them. This
backend is that route. AreTomo2 solves a global alignment, and the pipeline's
own steps then build the aligned stack, fit the CTF and reconstruct, exactly as
they do after tiltalign.

What AreTomo2 does and writes was read from its source (czimaginginstitute/
AreTomo2, main, version 1.1.3), not measured:

- it sorts the views by tilt itself, and its `.aln` names each view by its
  section in the stack it read (MrcUtil/CLoadMain.cpp, CSaveAlnFile.cpp);
- it drops views it finds dark (`-DarkTol`, 0.7 by default) and lists them in
  the `.aln` header, so they are excluded here too;
- the `.aln` holds per view the tilt axis (ROT), the shift (TX, TY, in pixels of
  the stack it read) and the tilt angle, which includes the measured tilt offset
  when `-TiltCor 1`;
- its IMOD transform, the one RELION 4 reads, is A = R(-ROT) with the shift
  A (-TX, -TY) (ImodUtil/CSaveXF.cpp, mSaveForRelion). The same is computed here
  from the `.aln`, at its precision rather than the `.xf`'s three decimals;
- `-VolZ` is 1200 in its code although its help says 0, so 0 is passed: the
  pipeline reconstructs, and AreTomo2 only aligns;
- projection matching prints, per pass, the shift it measured for each view but
  the reference, the one nearest 0 deg. It runs in rounds, with a search of the
  tilt axis before each (three after a given axis, CProcessThread::mAlign); a
  round passes until the largest shift is under 2 px, or until a pass moves one
  further than the last kept, which is undone though printed (mProjAlign, at most
  10 passes). The last pass is this backend's QC (`AretomoAlignmentQc`);
- the tilt offset is the best of 31 whole degrees, -15 to +15
  (CTiltOffsetMain::mSearch).

Rounds, the undo rule and the offset grid were read on 10 November 2026, after
Position_1's log showed 6 passes in 3 rounds of 2; the rest earlier.
"""

from __future__ import annotations

import json
import math
import os
import re
import shutil
import statistics
import tempfile
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path

from cryoet_pipeline.artifacts import ArtifactRegistry
from cryoet_pipeline.backends.alignment import (
    CommandRunner,
    run_command,
    write_binned_mrc_stack,
    write_tilt_angles_file,
)
from cryoet_pipeline.backends.command_log import write_command_log
from cryoet_pipeline.backends.protocols import (
    BackendContext,
    variant_label,
    variant_suffix,
)
from cryoet_pipeline.models import (
    AlignmentTransform,
    AretomoAlignmentQc,
    Artifact,
    ArtifactKind,
    QcStatus,
    RetentionPolicy,
    StorageRole,
    TiltAlignment,
    TiltSeriesManifest,
)
from cryoet_pipeline.tool_versions import EXTERNAL, aretomo_version_of

# AreTomo2's own default (CInput.cpp, m_iAlignZ), in pixels of the stack it reads.
ARETOMO_DEFAULT_ALIGN_Z_PX = 600
# AreTomo2's own stopping rule for projection matching (CProcessThread.cpp).
ARETOMO_CONVERGED_PX = 2.0
# The aligned stack AreTomo2 writes is thrown away; binning it keeps it small.
_DISCARDED_OUTPUT_BINNING = 8

_ALN_ROW = re.compile(
    r"^\s*(?P<sec>\d+)\s+(?P<rot>\S+)\s+(?P<gmag>\S+)\s+(?P<tx>\S+)\s+(?P<ty>\S+)"
    r"\s+\S+\s+\S+\s+\S+\s+\S+\s+(?P<tilt>\S+)\s*$"
)
_ALN_DARK = re.compile(r"^#\s*DarkFrame\s*=\s*(?P<frame>-?\d+)\s+(?P<sec>\d+)\s+(?P<tilt>\S+)")
_ALN_RAW_SIZE = re.compile(r"^#\s*RawSize\s*=\s*(\d+)\s+(\d+)\s+(\d+)")
_ALN_PATCHES = re.compile(r"^#\s*NumPatches\s*=\s*(\d+)")
_THREE_FLOATS = re.compile(r"^\s*(\S+)\s+(\S+)\s+(\S+)\s*$")
_ROTATION_SCORE = re.compile(r"Rotation align score:\s*(?P<score>[-+.\deE]+)")
_NEW_TILT_AXIS = re.compile(r"New tilt axis:\s*(?P<axis>[-+.\deE]+)")


@dataclass(frozen=True)
class AlnRow:
    """One view of an AreTomo `.aln` global table."""

    section: int
    rotation_deg: float
    shift_x_px: float
    shift_y_px: float
    tilt_deg: float


@dataclass(frozen=True)
class AlnFile:
    """An AreTomo `.aln`: the stack size it read, the views it dropped, the rest."""

    raw_size: tuple[int, int, int]
    dark_sections: dict[int, float]
    rows: list[AlnRow]
    num_patches: int


def parse_aln(text: str) -> AlnFile:
    """Read the header and the global table of an AreTomo `.aln`.

    The local-alignment table that follows `# Local Alignment` is not read: the
    pipeline reconstructs with IMOD from the global alignment, as the authors of
    EMPIAR-11830 did.
    """

    raw_size: tuple[int, int, int] | None = None
    num_patches = 0
    dark: dict[int, float] = {}
    rows: list[AlnRow] = []
    in_global = False
    for line in text.splitlines():
        stripped = line.strip()
        if not stripped:
            continue
        if stripped.startswith("#"):
            if (match := _ALN_RAW_SIZE.match(stripped)) is not None:
                raw_size = (int(match[1]), int(match[2]), int(match[3]))
            elif (match := _ALN_PATCHES.match(stripped)) is not None:
                num_patches = int(match[1])
            elif (match := _ALN_DARK.match(stripped)) is not None:
                dark[int(match["sec"])] = _finite(match["tilt"], "dark-frame tilt")
            elif stripped.lstrip("# ").startswith("SEC"):
                in_global = True
            elif "Local Alignment" in stripped:
                in_global = False
            continue
        if not in_global:
            continue
        match = _ALN_ROW.match(line)
        if match is None:
            raise ValueError(f"unreadable AreTomo alignment row: {line!r}")
        rows.append(
            AlnRow(
                section=int(match["sec"]),
                rotation_deg=_finite(match["rot"], "ROT"),
                shift_x_px=_finite(match["tx"], "TX"),
                shift_y_px=_finite(match["ty"], "TY"),
                tilt_deg=_finite(match["tilt"], "TILT"),
            )
        )
    if raw_size is None:
        raise ValueError("AreTomo alignment file has no '# RawSize' line")
    if not rows:
        raise ValueError("AreTomo alignment file has no global alignment rows")
    sections = [row.section for row in rows]
    if len(set(sections)) != len(sections):
        raise ValueError("AreTomo alignment file names a section twice")
    if set(sections) & set(dark):
        raise ValueError("AreTomo alignment file aligns a section it also called dark")
    return AlnFile(raw_size=raw_size, dark_sections=dark, rows=rows, num_patches=num_patches)


@dataclass(frozen=True)
class ProjectionPass:
    """One projection-matching pass as AreTomo2 printed it."""

    rows: list[tuple[float, float, float]]
    """(tilt, dx, dy) per view but the reference, the view nearest 0 deg."""
    round: int
    """1, 2, ...: a round is one call of CProcessThread::mProjAlign; AreTomo2
    searches the tilt axis again between rounds (mRotAlign)."""
    tilt_axis_deg: float | None
    """The last 'New tilt axis' AreTomo2 printed before this pass."""


def parse_projection_passes(stdout: str) -> list[ProjectionPass]:
    """Every projection-matching pass AreTomo2 printed, with the round it belongs to.

    A pass starts at `# Projection matching measurements`, then the column line,
    then one row per view until a line that is not three numbers. A pass after a
    `New tilt axis:` line starts a new round. Read from AreTomo2's source (main,
    CProcessThread::mAlign and mProjAlign, CProjAlignMain::mDoAll; 10 November
    2026) and matched by its log on EMPIAR-11830 Position_1, 3 rounds of 2:

    - each row is the shift measured for that view in that pass, scaled from the
      binned stack it matches on back to the full stack; the tilt includes the
      tilt offset already applied; the reference view is never printed;
    - a round passes until the largest shift is under 2 px (that pass is kept),
      or until one is larger than the last kept, which is then undone, though
      printed; at most 10 passes.
    """

    passes: list[ProjectionPass] = []
    current: list[tuple[float, float, float]] | None = None
    axis: float | None = None
    rounds = 0
    new_round = True
    for line in stdout.splitlines():
        if (found := _NEW_TILT_AXIS.search(line)) is not None:
            axis = _finite(found["axis"], "tilt axis")
            new_round = True
            current = None
            continue
        if "Projection matching measurements" in line:
            if new_round:
                rounds += 1
                new_round = False
            current = []
            passes.append(ProjectionPass(rows=current, round=rounds, tilt_axis_deg=axis))
            continue
        if current is None:
            continue
        if line.lstrip().startswith("#"):
            continue
        match = _THREE_FLOATS.match(line)
        if match is None:
            current = None
            continue
        try:
            values = tuple(float(value) for value in match.groups())
        except ValueError:
            current = None
            continue
        if not all(math.isfinite(value) for value in values):
            raise ValueError(f"AreTomo printed a non-finite projection shift: {line!r}")
        current.append((values[0], values[1], values[2]))
    return [one for one in passes if one.rows]


def parse_projection_matching(stdout: str) -> list[list[tuple[float, float, float]]]:
    """Every projection-matching pass AreTomo2 printed: (tilt, dx, dy) per view.

    Shifts are in pixels of the stack AreTomo2 read; see `parse_projection_passes`.
    """

    return [one.rows for one in parse_projection_passes(stdout)]


def parse_rotation_scores(stdout: str) -> list[float]:
    return [float(match["score"]) for match in _ROTATION_SCORE.finditer(stdout)]


def imod_transform(row: AlnRow, *, binning: int) -> tuple[float, ...]:
    """The IMOD `.xf` line AreTomo2 writes for RELION, from one `.aln` row.

    A = R(-ROT) turns the tilt axis vertical; the shift A (-TX, -TY) moves the
    view's centre back. Shifts are scaled from the pixels AreTomo2 read to full
    resolution by `binning`.
    """

    theta = math.radians(-row.rotation_deg)
    a11, a12 = math.cos(theta), -math.sin(theta)
    a21, a22 = math.sin(theta), math.cos(theta)
    dx = a11 * -row.shift_x_px + a12 * -row.shift_y_px
    dy = a21 * -row.shift_x_px + a22 * -row.shift_y_px
    return a11, a12, a21, a22, dx * binning, dy * binning


@dataclass(frozen=True)
class _AretomoPaths:
    output_dir: Path
    canonical: Path
    report: Path
    final_transform: Path
    solved_tilt_file: Path
    aln: Path
    log: Path

    @property
    def outputs(self) -> tuple[Path, ...]:
        return (
            self.canonical,
            self.report,
            self.final_transform,
            self.solved_tilt_file,
            self.aln,
            self.log,
        )


def _aretomo_paths(
    output_root: Path, manifest: TiltSeriesManifest, *, label: str | None
) -> _AretomoPaths:
    # The folder and the canonical names are tiltalign's, so that every later
    # step and every reader finds the fine alignment where it always is.
    output_dir = output_root / "alignments" / manifest.tilt_series_id / "fine"
    prefix = f"{manifest.tilt_series_id}_fine" + (f"_{label}" if label else "")
    return _AretomoPaths(
        output_dir=output_dir,
        canonical=output_dir / f"{prefix}_alignment.json",
        report=output_dir / f"{prefix}_alignment_qc.json",
        final_transform=output_dir / f"{prefix}.xf",
        solved_tilt_file=output_dir / f"{prefix}.tlt",
        aln=output_dir / f"{prefix}_aretomo.aln",
        log=output_dir / f"{prefix}_aretomo.log",
    )


class AretomoAlignmentBackend:
    """Solve a global fine alignment with AreTomo2, from the tilt stack itself."""

    name = "aretomo"

    def __init__(self, command_runner: CommandRunner | None = None) -> None:
        self._command_runner = command_runner or run_command

    def align(
        self,
        tilt_stack: Artifact,
        coarse_alignment: Artifact,
        manifest: TiltSeriesManifest,
        context: BackendContext,
    ) -> list[Artifact]:
        """Return the canonical fine alignment and its QC, as tiltalign's backend does."""

        pixel = _validate_tilt_stack(tilt_stack, manifest)
        coarse = _validate_coarse(coarse_alignment, manifest)
        executable = _aretomo_executable(context)
        version = aretomo_version_of(str(executable))
        if version is not None and version.startswith("AreTomo3"):
            raise ValueError(
                f"{executable} is {version}; this backend drives AreTomo2 (or AreTomo "
                "1.x), whose command line AreTomo3 does not share"
            )
        residual_warning_nm = _float(context, "residual_warning_nm", 0.8, minimum=0.0)
        residual_failure_nm = _float(context, "residual_failure_nm", 1.5, minimum=0.0)
        if residual_failure_nm < residual_warning_nm:
            raise ValueError("residual_failure_nm must be at least residual_warning_nm")
        tilt_cor = _bool(context, "aretomo_tilt_cor", True)
        dark_tol = _float(context, "aretomo_dark_tol", 0.7, minimum=0.0, maximum=1.0)
        gpu = _int(context, "aretomo_gpu", 0, minimum=0)
        align_z_nm = context.parameters.get("aretomo_align_z_nm")
        if align_z_nm is None:
            align_z_px = ARETOMO_DEFAULT_ALIGN_Z_PX
        else:
            value = _float(context, "aretomo_align_z_nm", 0.0, minimum=0.0)
            if value <= 0.0:
                raise ValueError("aretomo_align_z_nm must be positive")
            align_z_px = max(1, round(value * 10.0 / pixel))
        overwrite = _bool(context, "overwrite", False)
        paths = _aretomo_paths(context.output_dir, manifest, label=variant_label(context))
        if not overwrite:
            # The log is left out: a failed run writes one, and its rerun replaces it.
            existing = [path for path in paths.outputs if path.exists() and path != paths.log]
            if existing:
                raise FileExistsError(
                    "fine-alignment outputs already exist; pass overwrite to replace: "
                    + ", ".join(str(path) for path in existing)
                )

        excluded_before = sorted(coarse.excluded_z_values)
        left_out = set(excluded_before)
        order = sorted(
            (
                index
                for index, image in enumerate(manifest.images)
                if image.z_value not in left_out
            ),
            key=lambda index: manifest.images[index].tilt_angle_deg,
        )
        if len(order) < 3:
            raise ValueError("AreTomo alignment needs at least three included tilts")
        nominal = [manifest.images[index].tilt_angle_deg for index in order]
        z_by_section = [manifest.images[index].z_value for index in order]
        initial_axis = coarse.tilt_axis_angle_deg

        paths.output_dir.mkdir(parents=True, exist_ok=True)
        with tempfile.TemporaryDirectory(
            prefix=f".{manifest.tilt_series_id}-aretomo-", dir=paths.output_dir.resolve()
        ) as temporary:
            root = Path(temporary)
            input_stack = root / f"{manifest.tilt_series_id}_aretomo_input.mrc"
            angle_file = root / f"{manifest.tilt_series_id}_aretomo_input.rawtlt"
            output_stack = root / f"{manifest.tilt_series_id}_aretomo_aligned.mrc"
            # Where AreTomo2 puts the .aln: the output's folder, the input's name.
            aln_written = root / f"{manifest.tilt_series_id}_aretomo_input.aln"
            write_binned_mrc_stack(
                tilt_stack.path,
                input_stack,
                binning=1,
                pixel_spacing_angstrom=pixel,
                section_indices=order,
            )
            write_tilt_angles_file(angle_file, nominal)
            command = [
                str(executable),
                "-InMrc", str(input_stack),
                "-OutMrc", str(output_stack),
                "-AngFile", str(angle_file),
                # The second value 1 lets AreTomo2 refine the axis, within 6 deg
                # of this one when it is given (CProcessThread.cpp, mAlign).
                "-TiltAxis", f"{initial_axis:.3f}", "1",
                "-AlignZ", str(align_z_px),
                "-VolZ", "0",
                "-OutBin", str(_DISCARDED_OUTPUT_BINNING),
                "-TiltCor", "1" if tilt_cor else "0",
                "-DarkTol", f"{dark_tol:g}",
                "-Gpu", str(gpu),
            ]
            result = self._command_runner(command, cwd=root, env=dict(os.environ))
            write_command_log(paths.log, (command, result))
            if result.returncode != 0:
                raise RuntimeError(
                    f"AreTomo exited with code {result.returncode}; see {paths.log}"
                )
            if not aln_written.is_file():
                # Its name comes from string surgery on the two paths
                # (CSaveAlnFile::DoIt); take the only .aln there if it differs.
                found = sorted(root.glob("*.aln"))
                if len(found) != 1:
                    raise RuntimeError(
                        f"AreTomo wrote no alignment file ({aln_written.name}); "
                        f"see {paths.log}"
                    )
                aln_written = found[0]
            aln = parse_aln(aln_written.read_text())
            shutil.copyfile(aln_written, paths.aln)
        stdout = result.stdout or ""

        if aln.raw_size[2] != len(order):
            raise ValueError(
                f"AreTomo read {aln.raw_size[2]} views, but {len(order)} were given"
            )
        named = [row.section for row in aln.rows] + list(aln.dark_sections)
        if sorted(named) != list(range(len(order))):
            raise ValueError(
                "AreTomo's alignment file does not account for every view given to it "
                f"once: sections {sorted(named)} for {len(order)} views"
            )
        dark_z = sorted(z_by_section[section] for section in aln.dark_sections)
        rows = sorted(aln.rows, key=lambda row: row.tilt_deg)
        transforms = []
        for row in rows:
            values = imod_transform(row, binning=1)
            transforms.append(
                AlignmentTransform(
                    z_value=z_by_section[row.section],
                    tilt_angle_deg=row.tilt_deg,
                    a11=values[0],
                    a12=values[1],
                    a21=values[2],
                    a22=values[3],
                    shift_x_px=values[4],
                    shift_y_px=values[5],
                )
            )
        axes = [row.rotation_deg for row in rows]
        offsets = [row.tilt_deg - nominal[row.section] for row in rows]
        excluded = sorted(set(excluded_before) | set(dark_z))
        alignment = TiltAlignment(
            tilt_series_id=manifest.tilt_series_id,
            backend=self.name,
            stage="fine",
            input_stack_id=tilt_stack.id,
            input_binning=1,
            tilt_axis_angle_deg=statistics.median(axes),
            transform_semantics="global",
            transforms=transforms,
            excluded_z_values=excluded,
        )
        report = _qc(
            alignment_id=f"{manifest.tilt_series_id}:alignment:fine{variant_suffix(context)}",
            manifest=manifest,
            version=version,
            image_count=len(rows),
            excluded_before=excluded_before,
            dark_z=dark_z,
            pixel=pixel,
            align_z_px=align_z_px,
            initial_axis=initial_axis,
            axes=axes,
            tilt_cor=tilt_cor,
            offsets=offsets,
            passes=parse_projection_matching(stdout),
            rotation_scores=parse_rotation_scores(stdout),
            residual_warning_nm=residual_warning_nm,
            residual_failure_nm=residual_failure_nm,
        )

        paths.final_transform.write_text(
            "".join(
                f"{t.a11:.7f} {t.a12:.7f} {t.a21:.7f} {t.a22:.7f} "
                f"{t.shift_x_px:.3f} {t.shift_y_px:.3f}\n"
                for t in transforms
            )
        )
        write_tilt_angles_file(paths.solved_tilt_file, [t.tilt_angle_deg for t in transforms])
        paths.canonical.write_text(json.dumps(alignment.model_dump(mode="json"), indent=2) + "\n")
        paths.report.write_text(json.dumps(report.model_dump(mode="json"), indent=2) + "\n")
        software = {"aretomo": version or EXTERNAL}
        included = [t.z_value for t in transforms]
        alignment_artifact = Artifact(
            id=report.alignment_id,
            variant=variant_label(context),
            kind=ArtifactKind.ALIGNMENT,
            path=paths.canonical,
            parent_ids=[tilt_stack.id, coarse_alignment.id],
            shape=(len(transforms), 6),
            dtype="float64",
            parameters={
                "backend": self.name,
                "stage": "fine",
                "tilt_series_id": manifest.tilt_series_id,
                "transform_semantics": "global",
                "included_z_values": included,
                "excluded_z_values": excluded,
                "aretomo_dark_z_values": dark_z,
                "raw_pixel_spacing_angstrom": pixel,
                "imod_xf_path": str(paths.final_transform),
                "solved_tilt_file_path": str(paths.solved_tilt_file),
                "aretomo_aln_path": str(paths.aln),
                "aretomo_log_path": str(paths.log),
                "aretomo_align_z_px": align_z_px,
                "aretomo_tilt_cor": tilt_cor,
                "aretomo_dark_tol": dark_tol,
                "translation_units": "full_resolution_pixels",
                # AreTomo2 solves no X-axis tilt, and its projection matching is
                # centred on the volume it reconstructs, so nothing is left to
                # shift; the authors' tilt.com says XAXISTILT 0 and SHIFT 0 0.
                "recommended_x_axis_tilt_deg": 0.0,
                "axis_z_shift_applied_in_alignment": True,
                "positioning_source": "aretomo",
                "applied_tilt_angle_offset_deg": report.applied_tilt_angle_offset_deg,
            },
            software_versions=software,
            storage_role=StorageRole.CANONICAL,
            retention_policy=RetentionPolicy.KEEP,
            can_recompute=True,
            size_bytes=sum(
                path.stat().st_size
                for path in (
                    paths.canonical,
                    paths.final_transform,
                    paths.solved_tilt_file,
                    paths.aln,
                    paths.log,
                )
            ),
        )
        report_artifact = Artifact(
            id=f"{manifest.tilt_series_id}:qc:alignment:fine{variant_suffix(context)}",
            variant=variant_label(context),
            kind=ArtifactKind.QC,
            path=paths.report,
            parent_ids=[alignment_artifact.id],
            parameters={
                "qc_type": "fine_alignment",
                "tilt_series_id": manifest.tilt_series_id,
                "status": report.status.value,
                "backend": self.name,
            },
            software_versions=software,
            storage_role=StorageRole.QC,
            retention_policy=RetentionPolicy.KEEP,
            can_recompute=True,
            size_bytes=paths.report.stat().st_size,
        )
        return [alignment_artifact, report_artifact]


def aretomo_align_and_register(
    backend: AretomoAlignmentBackend,
    tilt_stack: Artifact,
    coarse_alignment: Artifact,
    manifest: TiltSeriesManifest,
    context: BackendContext,
    registry: ArtifactRegistry,
    *,
    replace_existing: bool = False,
) -> list[Artifact]:
    """Run AreTomo fine alignment and register the alignment, then its QC."""

    artifacts = backend.align(tilt_stack, coarse_alignment, manifest, context)
    registry.extend(artifacts, replace=replace_existing)
    return artifacts


def _qc(
    *,
    alignment_id: str,
    manifest: TiltSeriesManifest,
    version: str | None,
    image_count: int,
    excluded_before: list[int],
    dark_z: list[int],
    pixel: float,
    align_z_px: int,
    initial_axis: float,
    axes: Sequence[float],
    tilt_cor: bool,
    offsets: Sequence[float],
    passes: list[list[tuple[float, float, float]]],
    rotation_scores: list[float],
    residual_warning_nm: float,
    residual_failure_nm: float,
) -> AretomoAlignmentQc:
    warnings: list[str] = []
    status = QcStatus.PASS
    mean_nm = max_nm = max_px = None
    converged = False
    if not passes:
        status = QcStatus.FAIL
        warnings.append(
            "AreTomo printed no projection-matching pass, so nothing measures how well "
            "the views agree; read the log before using this alignment"
        )
    else:
        last = [math.hypot(dx, dy) for _, dx, dy in passes[-1]]
        max_px = max(last)
        mean_nm = statistics.fmean(last) * pixel / 10.0
        max_nm = max_px * pixel / 10.0
        converged = max_px < ARETOMO_CONVERGED_PX
        if mean_nm > residual_failure_nm:
            status = QcStatus.FAIL
            warnings.append(
                f"last projection-matching pass still moved the views by {mean_nm:.2f} nm "
                f"on average, above the {residual_failure_nm:.2f} nm failure threshold"
            )
        elif mean_nm > residual_warning_nm:
            status = QcStatus.WARNING
            warnings.append(
                f"last projection-matching pass moved the views by {mean_nm:.2f} nm on "
                f"average, above the {residual_warning_nm:.2f} nm warning threshold"
            )
        if not converged:
            if status is QcStatus.PASS:
                status = QcStatus.WARNING
            warnings.append(
                f"AreTomo stopped with a view still moved by {max_px:.1f} px in its last "
                f"pass; its own rule stops under {ARETOMO_CONVERGED_PX:.0f} px"
            )
    if dark_z:
        warnings.append(f"AreTomo left out views it found dark: z {dark_z}")
    spread = max(axes) - min(axes)
    applied_offset = statistics.median(offsets) if tilt_cor else None
    return AretomoAlignmentQc(
        tilt_series_id=manifest.tilt_series_id,
        backend="aretomo",
        alignment_id=alignment_id,
        aretomo_version=version,
        image_count=image_count,
        excluded_before_z_values=excluded_before,
        dark_z_values=dark_z,
        input_pixel_spacing_angstrom=pixel,
        align_z_px=align_z_px,
        initial_tilt_axis_angle_deg=initial_axis,
        tilt_axis_angle_deg=statistics.median(axes),
        tilt_axis_spread_deg=spread,
        tilt_offset_applied=tilt_cor,
        applied_tilt_angle_offset_deg=applied_offset,
        projection_matching_passes=len(passes),
        last_pass_correction_mean_nm=mean_nm,
        last_pass_correction_max_nm=max_nm,
        last_pass_correction_max_px=max_px,
        converged=converged,
        rotation_scores=rotation_scores,
        status=status,
        warnings=warnings,
    )


def _validate_tilt_stack(tilt_stack: Artifact, manifest: TiltSeriesManifest) -> float:
    if tilt_stack.kind is not ArtifactKind.TILT_STACK:
        raise ValueError(f"expected tilt stack artifact, got {tilt_stack.kind}")
    if not tilt_stack.path.exists():
        raise FileNotFoundError(f"tilt stack not found: {tilt_stack.path}")
    if tilt_stack.shape is None or len(tilt_stack.shape) != 3:
        raise ValueError("tilt stack artifact must record shape (tilts, y, x)")
    if int(tilt_stack.shape[0]) != manifest.num_tilts:
        raise ValueError(
            f"tilt stack has {tilt_stack.shape[0]} images, manifest has {manifest.num_tilts}"
        )
    pixel = tilt_stack.pixel_spacing_angstrom
    if pixel is None or not math.isfinite(pixel) or pixel <= 0.0:
        raise ValueError("tilt stack artifact must record its pixel spacing")
    return float(pixel)


def _validate_coarse(artifact: Artifact, manifest: TiltSeriesManifest) -> TiltAlignment:
    if artifact.kind is not ArtifactKind.ALIGNMENT or artifact.parameters.get("stage") != "coarse":
        raise ValueError("AreTomo fine alignment needs the coarse alignment artifact")
    if not artifact.path.is_file():
        raise FileNotFoundError(f"coarse alignment result not found: {artifact.path}")
    coarse = TiltAlignment.model_validate_json(artifact.path.read_text())
    if coarse.stage != "coarse" or coarse.tilt_series_id != manifest.tilt_series_id:
        raise ValueError("coarse alignment is for another stage or tilt series")
    return coarse


def _aretomo_executable(context: BackendContext) -> Path:
    configured = context.parameters.get("aretomo_executable")
    if configured is not None:
        path = Path(os.path.expandvars(str(configured))).expanduser()
        if not path.is_file():
            raise FileNotFoundError(f"AreTomo executable not found: {path}")
        return path
    for name in ("AreTomo2", "AreTomo"):
        found = shutil.which(name)
        if found is not None:
            return Path(found)
    raise FileNotFoundError(
        "no AreTomo executable: pass --aretomo-executable (a lane's [executables] "
        "aretomo), or put AreTomo2 on PATH"
    )


def _finite(text: str, what: str) -> float:
    value = float(text)
    if not math.isfinite(value):
        raise ValueError(f"AreTomo alignment file has a non-finite {what}: {text!r}")
    return value


def _float(
    context: BackendContext,
    key: str,
    default: float,
    *,
    minimum: float | None = None,
    maximum: float | None = None,
) -> float:
    value = context.parameters.get(key, default)
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise TypeError(f"context parameter {key!r} must be numeric")
    value = float(value)
    if not math.isfinite(value):
        raise ValueError(f"context parameter {key!r} must be finite")
    if minimum is not None and value < minimum:
        raise ValueError(f"context parameter {key!r} must be at least {minimum}")
    if maximum is not None and value > maximum:
        raise ValueError(f"context parameter {key!r} must be at most {maximum}")
    return value


def _int(context: BackendContext, key: str, default: int, *, minimum: int) -> int:
    value = context.parameters.get(key, default)
    if isinstance(value, bool) or not isinstance(value, int):
        raise TypeError(f"context parameter {key!r} must be an int")
    if value < minimum:
        raise ValueError(f"context parameter {key!r} must be at least {minimum}")
    return value


def _bool(context: BackendContext, key: str, default: bool) -> bool:
    value = context.parameters.get(key, default)
    if not isinstance(value, bool):
        raise TypeError(f"context parameter {key!r} must be a bool")
    return value
