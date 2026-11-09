"""AreTomo2 as the fine-alignment backend, and the view exclusion it was built with.

What AreTomo2 reads and writes is taken from its source (AreTomo2 main, 1.1.3);
the stand-ins below write its `.aln` and print its projection-matching table in
the formats of CSaveAlnFile.cpp and CProjAlignMain.cpp.
"""

# ruff: noqa: UP031 -- the % formats are AreTomo2's own printf formats, kept verbatim.

from __future__ import annotations

import copy
import math
import subprocess
import sys
import tomllib
from collections.abc import Mapping, Sequence
from pathlib import Path

import mrcfile
import numpy as np
import pytest
import zarr
from typer.testing import CliRunner

from cryoet_pipeline import cli, tool_versions
from cryoet_pipeline.artifacts import ArtifactRegistry
from cryoet_pipeline.backends import aretomo_alignment as at
from cryoet_pipeline.backends.alignment import ImodTiltXcorrAlignmentBackend, _views_at_angles
from cryoet_pipeline.backends.protocols import BackendContext
from cryoet_pipeline.models import (
    AlignmentTransform,
    AretomoAlignmentQc,
    Artifact,
    ArtifactKind,
    AxisOrder,
    QcStatus,
    TiltAlignment,
    TiltImage,
    TiltSeriesManifest,
)
from cryoet_pipeline.runtime import DevicePreference

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
import run_series  # noqa: E402
import submit_lanes  # noqa: E402

PIXEL = 2.0
# Acquisition order, as a dose-symmetric series is stored: z -> nominal angle.
ANGLES = {0: 0.0, 1: 3.0, 2: -3.0, 3: 6.0, 4: -6.0}
COARSE_EXCLUDED = [4]  # the -6 view, left out before AreTomo sees the stack
# Included views sorted by tilt: z 2, 0, 1, 3 -> AreTomo sections 0..3.
Z_BY_SECTION = [2, 0, 1, 3]
DARK_SECTION = 3  # AreTomo drops the +6 view as dark
ROTATION = 84.5
SHIFTS = {0: (10.0, -4.0), 1: (2.5, 1.5), 2: (-7.0, 3.0)}
OFFSET = 8.0
LAST_PASS = [(0.3, 0.4), (0.0, 0.5), (0.6, 0.8)]  # |d| = 0.5, 0.5, 1.0 px
ALN_COLUMNS = (
    "# SEC     ROT         GMAG       TX          TY      SMEAN     SFIT    SCALE     BASE     TILT"
)


# --- parsers -------------------------------------------------------------------


def _aln_text(
    *,
    rows: Sequence[tuple[int, float, float, float, float]],
    dark: Sequence[tuple[int, int, float]] = (),
    raw_size: tuple[int, int, int] = (8, 8, 4),
    local: bool = False,
) -> str:
    """An `.aln` as CSaveAlnFile.cpp prints it."""

    lines = [
        "# AreTomo Alignment / Priims bprmMn ",
        "# RawSize = %d %d %d" % raw_size,
        "# NumPatches = %d" % (2 if local else 0),
    ]
    lines += ["# DarkFrame =  %4d %4d %8.2f" % one for one in dark]
    lines.append(ALN_COLUMNS)
    for section, rotation, tx, ty, tilt in rows:
        lines.append(
            "%5d  %9.4f  %9.5f  %9.3f  %9.3f  %7.2f  %7.2f  %7.2f  %7.2f  %8.2f"
            % (section, rotation, 1.0, tx, ty, 1.0, 1.0, 1.0, 0.0, tilt)
        )
    if local:
        lines.append("# Local Alignment")
        lines.append("   0   0   100.00    200.00      1.50     -0.50   1.0")
    return "\n".join(lines) + "\n"


def _projection_matching(passes: Sequence[Sequence[tuple[float, float, float]]]) -> str:
    """Projection-matching passes as CProjAlignMain.cpp prints them, among other lines."""

    out = ["Remove dark images...", "Rotation align score: 0.912345", ""]
    for one in passes:
        out += ["# Projection matching measurements", "# tilt angle   x shift   y shift"]
        out += ["  %6.2f  %8.2f  %8.2f" % row for row in one]
        out.append("")
    out += ["Rotation align score: 0.987650", "Total time: 12.00 sec"]
    return "\n".join(out) + "\n"


def test_an_aln_is_read_with_its_dark_views_and_without_its_local_table():
    text = _aln_text(
        rows=[(0, 84.5, 10.0, -4.0, 5.0), (2, 84.6, -1.0, 2.0, 11.0)],
        dark=[(1, 1, 8.0)],
        raw_size=(4096, 4096, 3),
        local=True,
    )
    aln = at.parse_aln(text)
    assert aln.raw_size == (4096, 4096, 3)
    assert aln.dark_sections == {1: 8.0}
    assert [row.section for row in aln.rows] == [0, 2]
    assert aln.rows[1].rotation_deg == pytest.approx(84.6)
    assert (aln.rows[0].shift_x_px, aln.rows[0].shift_y_px) == (10.0, -4.0)
    assert aln.num_patches == 2


@pytest.mark.parametrize(
    "text, message",
    [
        (_aln_text(rows=[(0, 84.5, 0, 0, 0)]).replace("# RawSize = 8 8 4\n", ""), "RawSize"),
        (_aln_text(rows=[(0, 84.5, 0, 0, 0), (0, 84.5, 0, 0, 3)]), "names a section twice"),
        (_aln_text(rows=[(0, 84.5, 0, 0, 0)], dark=[(0, 0, 0.0)]), "also called dark"),
        (_aln_text(rows=[]), "no global alignment rows"),
    ],
)
def test_an_aln_that_cannot_be_used_is_refused(text, message):
    with pytest.raises(ValueError, match=message):
        at.parse_aln(text)


def test_every_projection_matching_pass_is_read_and_nothing_else():
    stdout = _projection_matching(
        [[(-3.0, 4.0, -3.0), (0.0, 1.0, 1.0)], [(-3.0, 0.3, 0.4), (0.0, 0.0, 0.5)]]
    )
    passes = at.parse_projection_matching(stdout)
    assert passes == [
        [(-3.0, 4.0, -3.0), (0.0, 1.0, 1.0)],
        [(-3.0, 0.3, 0.4), (0.0, 0.0, 0.5)],
    ]
    assert at.parse_rotation_scores(stdout) == [0.912345, 0.98765]


# --- the transform convention --------------------------------------------------


def test_the_rotation_reproduces_the_authors_deposited_aretomo_xf():
    """EMPIAR-11830 Position_1, AreTomo/..._dose-filt.xf, every line:
    0.0941848 -0.9955547 0.9955547 0.0941848. The authors' tilt axis was -85
    (TOMOMAN tomolist.star), refined by AreTomo; -84.5956 deg is the one that
    matrix encodes. A wrong sign anywhere in R(-ROT) fails this."""

    row = at.AlnRow(section=0, rotation_deg=-84.5956, shift_x_px=0.0, shift_y_px=0.0, tilt_deg=0.0)
    a11, a12, a21, a22, _, _ = at.imod_transform(row, binning=1)
    assert (a11, a12, a21, a22) == pytest.approx(
        (0.0941848, -0.9955547, 0.9955547, 0.0941848), abs=2e-6
    )


@pytest.mark.parametrize("binning", [1, 2])
def test_a_point_on_the_tilt_axis_lands_on_the_vertical_axis(binning):
    """The meaning of the .aln: the axis runs at ROT (CCW from Y) through the
    view's shifted centre (TX, TY). After the transform it must be the line x = 0,
    as IMOD's aligned stacks have it, and the centre must be the origin."""

    row = at.AlnRow(section=0, rotation_deg=84.5, shift_x_px=12.0, shift_y_px=-5.0, tilt_deg=0.0)
    a11, a12, a21, a22, dx, dy = at.imod_transform(row, binning=binning)
    axis = np.array([-math.sin(math.radians(84.5)), math.cos(math.radians(84.5))])
    centre = np.array([12.0, -5.0]) * binning
    for along in (-300.0, 0.0, 250.0):
        point = centre + along * axis
        x = a11 * point[0] + a12 * point[1] + dx
        y = a21 * point[0] + a22 * point[1] + dy
        assert x == pytest.approx(0.0, abs=1e-9)
        assert y == pytest.approx(along, abs=1e-9)


# --- the backend ---------------------------------------------------------------


def _manifest() -> TiltSeriesManifest:
    return TiltSeriesManifest(
        tilt_series_id="TS_A",
        source_mdoc=Path("TS_A.mdoc"),
        raw_pixel_spacing_angstrom=PIXEL,
        images=[
            TiltImage(
                z_value=z,
                tilt_angle_deg=angle,
                subframe_path=f"TS_A_{z:03d}.eer",
                num_subframes=1,
                pixel_spacing_angstrom=PIXEL,
                binning=1,
            )
            for z, angle in ANGLES.items()
        ],
    )


def _inputs(tmp_path: Path) -> tuple[Artifact, Artifact]:
    """A tilt stack whose section z is filled with z + 1, and a coarse alignment."""

    stack_path = tmp_path / "TS_A.mrc"
    data = np.stack([np.full((8, 8), z + 1.0, dtype=np.float32) for z in ANGLES])
    with mrcfile.new(stack_path) as stack:
        stack.set_data(data)
        stack.voxel_size = PIXEL
    tilt_stack = Artifact(
        id="TS_A:tilt_stack",
        kind=ArtifactKind.TILT_STACK,
        path=stack_path,
        shape=(5, 8, 8),
        dtype="float32",
        axis_order=AxisOrder.TYX,
        pixel_spacing_angstrom=PIXEL,
        parameters={"tilt_series_id": "TS_A"},
    )
    coarse = TiltAlignment(
        tilt_series_id="TS_A",
        backend="imod-xcorr",
        stage="coarse",
        input_stack_id=tilt_stack.id,
        input_binning=8,
        tilt_axis_angle_deg=84.1,
        transform_semantics="global",
        transforms=[
            AlignmentTransform(
                z_value=z, tilt_angle_deg=angle, a11=1, a12=0, a21=0, a22=1,
                shift_x_px=0, shift_y_px=0,
            )
            for z, angle in ANGLES.items()
        ],
        excluded_z_values=COARSE_EXCLUDED,
    )
    coarse_path = tmp_path / "TS_A_coarse.json"
    coarse_path.write_text(coarse.model_dump_json())
    coarse_artifact = Artifact(
        id="TS_A:alignment:coarse",
        kind=ArtifactKind.ALIGNMENT,
        path=coarse_path,
        parent_ids=[tilt_stack.id],
        parameters={"stage": "coarse", "tilt_series_id": "TS_A",
                    "excluded_z_values": COARSE_EXCLUDED},
    )
    return tilt_stack, coarse_artifact


def _after(command: Sequence[str], option: str) -> str:
    return command[command.index(option) + 1]


class _FakeAretomo:
    """Stands in for AreTomo2: reads what it is given, writes its .aln, prints passes."""

    def __init__(self, *, passes=None, returncode: int = 0, write_aln: bool = True):
        self.commands: list[list[str]] = []
        self.sections: list[float] = []
        self.angles: list[float] = []
        self.passes = passes
        self.returncode = returncode
        self.write_aln = write_aln

    def __call__(
        self,
        command: Sequence[str],
        *,
        cwd: Path,
        env: Mapping[str, str],
        stdin: str | None = None,
    ) -> subprocess.CompletedProcess[str]:
        del cwd, env, stdin
        self.commands.append(list(command))
        in_mrc = Path(_after(command, "-InMrc"))
        out_mrc = Path(_after(command, "-OutMrc"))
        with mrcfile.open(in_mrc, permissive=True) as stack:
            self.sections = [float(section.mean()) for section in stack.data]
        self.angles = [float(v) for v in Path(_after(command, "-AngFile")).read_text().split()]
        tilt_cor = _after(command, "-TiltCor") == "1"
        rows = [
            (section, ROTATION, *SHIFTS[section],
             self.angles[section] + (OFFSET if tilt_cor else 0.0))
            for section in range(len(self.angles))
            if section != DARK_SECTION
        ]
        if self.write_aln:
            # CSaveAlnFile: the output's folder, the input's name, .mrc -> .aln.
            (out_mrc.parent / f"{in_mrc.stem}.aln").write_text(
                _aln_text(
                    rows=rows,
                    dark=[(DARK_SECTION, DARK_SECTION, self.angles[DARK_SECTION])],
                    raw_size=(8, 8, len(self.angles)),
                )
            )
        last = [(row[4], *d) for row, d in zip(rows, LAST_PASS, strict=True)]
        first = [(row[4], 6.0, -4.0) for row in rows]
        passes = self.passes if self.passes is not None else [first, last]
        return subprocess.CompletedProcess(
            command, self.returncode, stdout=_projection_matching(passes), stderr=""
        )


def _context(tmp_path: Path, **parameters: object) -> BackendContext:
    executable = tmp_path / "AreTomo2"
    executable.touch()
    return BackendContext(
        output_dir=tmp_path / "project",
        device=DevicePreference.CPU,
        parameters={"aretomo_executable": executable, **parameters},
    )


@pytest.fixture(autouse=True)
def _aretomo2(monkeypatch):
    monkeypatch.setattr(at, "aretomo_version_of", lambda executable: "AreTomo2 1.1.3")


def test_aretomo_aligns_the_included_views_and_its_alignment_is_the_pipelines(tmp_path):
    tilt_stack, coarse = _inputs(tmp_path)
    fake = _FakeAretomo()
    registry = ArtifactRegistry.empty()
    registry.extend([tilt_stack, coarse])
    alignment_artifact, qc_artifact = at.aretomo_align_and_register(
        at.AretomoAlignmentBackend(fake), tilt_stack, coarse, _manifest(),
        _context(tmp_path), registry,
    )

    command = fake.commands[0]
    assert _after(command, "-VolZ") == "0"  # its code's default is 1200, not the help's 0
    axis = command.index("-TiltAxis")
    assert command[axis + 1 : axis + 3] == ["84.100", "1"]
    assert _after(command, "-AlignZ") == str(at.ARETOMO_DEFAULT_ALIGN_Z_PX)
    assert _after(command, "-TiltCor") == "1"
    assert _after(command, "-DarkTol") == "0.7"
    assert _after(command, "-Gpu") == "0"
    # The coarse step's excluded view never reaches AreTomo; the rest go in tilt order.
    assert fake.sections == [z + 1.0 for z in Z_BY_SECTION]
    assert fake.angles == [ANGLES[z] for z in Z_BY_SECTION]

    alignment = TiltAlignment.model_validate_json(alignment_artifact.path.read_text())
    assert alignment.stage == "fine" and alignment.backend == "aretomo"
    assert [t.z_value for t in alignment.transforms] == [2, 0, 1]
    assert [t.tilt_angle_deg for t in alignment.transforms] == [-3 + OFFSET, 0 + OFFSET, 3 + OFFSET]
    # The coarse step's exclusion and AreTomo's dark view are both left out.
    assert alignment.excluded_z_values == [3, 4]
    assert alignment.tilt_axis_angle_deg == pytest.approx(ROTATION)
    for section, transform in zip((0, 1, 2), alignment.transforms, strict=True):
        expected = at.imod_transform(
            at.AlnRow(section, ROTATION, *SHIFTS[section], 0.0), binning=1
        )
        got = (transform.a11, transform.a12, transform.a21, transform.a22,
               transform.shift_x_px, transform.shift_y_px)
        assert got == pytest.approx(expected)

    params = alignment_artifact.parameters
    assert alignment_artifact.id == "TS_A:alignment:fine"
    assert params["stage"] == "fine"
    assert params["included_z_values"] == [2, 0, 1]
    assert params["aretomo_dark_z_values"] == [3]
    assert params["recommended_x_axis_tilt_deg"] == 0.0
    assert params["axis_z_shift_applied_in_alignment"] is True
    assert params["applied_tilt_angle_offset_deg"] == pytest.approx(OFFSET)
    assert alignment_artifact.software_versions == {"aretomo": "AreTomo2 1.1.3"}
    xf = Path(params["imod_xf_path"]).read_text().splitlines()
    assert len(xf) == 3 and len(xf[0].split()) == 6
    assert [float(v) for v in Path(params["solved_tilt_file_path"]).read_text().split()] == [
        5.0, 8.0, 11.0
    ]
    assert Path(params["aretomo_aln_path"]).is_file()
    assert "-InMrc" in Path(params["aretomo_log_path"]).read_text()

    qc = AretomoAlignmentQc.model_validate_json(qc_artifact.path.read_text())
    assert qc_artifact.parameters["qc_type"] == "fine_alignment"
    assert qc.projection_matching_passes == 2
    assert qc.last_pass_correction_max_px == pytest.approx(1.0)
    assert qc.last_pass_correction_mean_nm == pytest.approx((0.5 + 0.5 + 1.0) / 3 * PIXEL / 10)
    assert qc.converged and qc.status is QcStatus.PASS
    assert qc.dark_z_values == [3] and any("dark" in warning for warning in qc.warnings)
    assert qc.rotation_scores == [0.912345, 0.98765]
    assert {one.id for one in registry.artifacts} == {
        tilt_stack.id, coarse.id, alignment_artifact.id, qc_artifact.id
    }


@pytest.mark.parametrize(
    "last, status, converged",
    [
        # Small on average, but one view still moved 2.5 px: AreTomo's own rule unmet.
        ([(0.1, 0.0), (0.0, 0.1), (2.5, 0.0)], QcStatus.WARNING, False),
        # 5 px = 1.0 nm on every view: above the 0.8 nm warning, under 1.5.
        ([(3.0, 4.0)] * 3, QcStatus.WARNING, False),
        # 10 px = 2.0 nm on every view: above the 1.5 nm failure threshold.
        ([(6.0, 8.0)] * 3, QcStatus.FAIL, False),
    ],
)
def test_the_last_pass_sets_the_status(tmp_path, last, status, converged):
    tilt_stack, coarse = _inputs(tmp_path)
    passes = [[(0.0, *d) for d in last]]
    _, qc_artifact = at.AretomoAlignmentBackend(_FakeAretomo(passes=passes)).align(
        tilt_stack, coarse, _manifest(), _context(tmp_path)
    )
    qc = AretomoAlignmentQc.model_validate_json(qc_artifact.path.read_text())
    assert (qc.status, qc.converged) == (status, converged)


def test_output_without_projection_matching_fails_the_qc(tmp_path):
    tilt_stack, coarse = _inputs(tmp_path)
    _, qc_artifact = at.AretomoAlignmentBackend(_FakeAretomo(passes=[])).align(
        tilt_stack, coarse, _manifest(), _context(tmp_path)
    )
    qc = AretomoAlignmentQc.model_validate_json(qc_artifact.path.read_text())
    assert qc.status is QcStatus.FAIL and "no projection-matching pass" in qc.warnings[0]


def test_options_reach_aretomo_and_tilt_cor_off_keeps_the_nominal_angles(tmp_path):
    tilt_stack, coarse = _inputs(tmp_path)
    fake = _FakeAretomo()
    alignment_artifact, qc_artifact = at.AretomoAlignmentBackend(fake).align(
        tilt_stack, coarse, _manifest(),
        _context(tmp_path, aretomo_align_z_nm=100.0, aretomo_tilt_cor=False,
                 aretomo_dark_tol=0.5, aretomo_gpu=2),
    )
    command = fake.commands[0]
    assert _after(command, "-AlignZ") == "500"  # 100 nm at 2 A
    assert _after(command, "-TiltCor") == "0"
    assert _after(command, "-DarkTol") == "0.5"
    assert _after(command, "-Gpu") == "2"
    alignment = TiltAlignment.model_validate_json(alignment_artifact.path.read_text())
    assert [t.tilt_angle_deg for t in alignment.transforms] == [-3.0, 0.0, 3.0]
    assert alignment_artifact.parameters["applied_tilt_angle_offset_deg"] is None


def test_a_failed_or_silent_run_registers_nothing(tmp_path):
    tilt_stack, coarse = _inputs(tmp_path)
    registry = ArtifactRegistry.empty()
    registry.extend([tilt_stack, coarse])
    with pytest.raises(RuntimeError, match="exited with code 1"):
        at.aretomo_align_and_register(
            at.AretomoAlignmentBackend(_FakeAretomo(returncode=1)), tilt_stack, coarse,
            _manifest(), _context(tmp_path), registry,
        )
    with pytest.raises(RuntimeError, match="no alignment file"):
        at.aretomo_align_and_register(
            at.AretomoAlignmentBackend(_FakeAretomo(write_aln=False)), tilt_stack, coarse,
            _manifest(), _context(tmp_path), registry,
        )
    assert {one.id for one in registry.artifacts} == {tilt_stack.id, coarse.id}


def test_a_second_run_needs_overwrite(tmp_path):
    tilt_stack, coarse = _inputs(tmp_path)
    backend = at.AretomoAlignmentBackend(_FakeAretomo())
    backend.align(tilt_stack, coarse, _manifest(), _context(tmp_path))
    with pytest.raises(FileExistsError, match="already exist"):
        backend.align(tilt_stack, coarse, _manifest(), _context(tmp_path))
    backend.align(tilt_stack, coarse, _manifest(), _context(tmp_path, overwrite=True))


def test_aretomo3_is_refused(tmp_path, monkeypatch):
    monkeypatch.setattr(at, "aretomo_version_of", lambda executable: "AreTomo3 2.0.10")
    tilt_stack, coarse = _inputs(tmp_path)
    with pytest.raises(ValueError, match="AreTomo3"):
        at.AretomoAlignmentBackend(_FakeAretomo()).align(
            tilt_stack, coarse, _manifest(), _context(tmp_path)
        )


# --- the version and the CLI with a stand-in executable --------------------------

_STAND_IN = r'''#!{python}
import sys
from pathlib import Path
import mrcfile
args = sys.argv[1:]
if args == ["--version"]:
    print("AreTomo2 version 1.1.3\nBuilt on Nov 8 2026")
    sys.exit(0)
def after(option):
    return args[args.index(option) + 1]
inp, out = Path(after("-InMrc")), Path(after("-OutMrc"))
with mrcfile.open(inp, permissive=True) as stack:
    n = stack.data.shape[0]
angles = [float(v) for v in Path(after("-AngFile")).read_text().split()]
lines = ["# AreTomo Alignment / Priims bprmMn ", "# RawSize = 8 8 %d" % n, "# NumPatches = 0",
         "# SEC     ROT         GMAG       TX          TY      SMEAN     SFIT"
         "    SCALE     BASE     TILT"]
for s, a in enumerate(angles):
    lines.append("%5d  %9.4f  %9.5f  %9.3f  %9.3f  %7.2f  %7.2f  %7.2f  %7.2f  %8.2f"
                 % (s, 84.5, 1.0, 1.0 * s, -1.0, 1.0, 1.0, 1.0, 0.0, a + 8.0))
(out.parent / (inp.stem + ".aln")).write_text("\n".join(lines) + "\n")
print("# Projection matching measurements\n# tilt angle   x shift   y shift")
for a in angles:
    print("  %6.2f  %8.2f  %8.2f" % (a + 8.0, 0.2, -0.1))
print()
'''


def _stand_in(tmp_path: Path) -> Path:
    path = tmp_path / "bin" / "AreTomo2"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(_STAND_IN.replace("{python}", sys.executable))
    path.chmod(0o755)
    return path


def test_the_version_names_the_program(tmp_path):
    executable = _stand_in(tmp_path)
    assert tool_versions.aretomo_version_of(str(executable)) == "AreTomo2 1.1.3"
    three = tmp_path / "bin" / "AreTomo3"
    three.write_text("#!/bin/sh\necho 'AreTomo3 version 2.0.10'\n")
    three.chmod(0o755)
    assert tool_versions.aretomo_version_of(str(three)) == "AreTomo3 2.0.10"
    old = tmp_path / "bin" / "AreTomo"
    old.write_text("#!/bin/sh\necho 'AreTomo version 1.3.4'\n")
    old.chmod(0o755)
    assert tool_versions.aretomo_version_of(str(old)) == "AreTomo 1.3.4"


def _project(tmp_path: Path) -> tuple[Path, Path]:
    tilt_stack, coarse = _inputs(tmp_path)
    manifest = tmp_path / "TS_A.json"
    manifest.write_text(_manifest().model_dump_json())
    registry_path = tmp_path / "artifacts.json"
    registry = ArtifactRegistry.empty()
    registry.extend([tilt_stack, coarse])
    registry.commit(registry_path)
    return manifest, registry_path


def test_fine_align_runs_aretomo_from_the_command_line(tmp_path, monkeypatch):
    monkeypatch.undo()  # the real version probe, through the stand-in
    manifest, registry_path = _project(tmp_path)
    result = CliRunner().invoke(cli.app, [
        "fine-align-tilt-series", "--manifest", str(manifest), "--registry",
        str(registry_path), "--out", str(tmp_path / "project"), "--backend", "aretomo",
        "--aretomo-executable", str(_stand_in(tmp_path)),
    ])
    assert result.exit_code == 0, result.output
    assert "QC status: pass" in result.output
    registry = ArtifactRegistry.load(registry_path)
    fine = [a for a in registry.by_kind(ArtifactKind.ALIGNMENT) if a.parameters["stage"] == "fine"]
    assert len(fine) == 1 and fine[0].software_versions == {"aretomo": "AreTomo2 1.1.3"}
    alignment = TiltAlignment.model_validate_json(fine[0].path.read_text())
    assert [t.z_value for t in alignment.transforms] == Z_BY_SECTION


def test_aretomo_options_are_refused_for_tiltalign(tmp_path):
    manifest, registry_path = _project(tmp_path)
    result = CliRunner().invoke(cli.app, [
        "fine-align-tilt-series", "--manifest", str(manifest), "--registry",
        str(registry_path), "--out", str(tmp_path / "project"), "--aretomo-gpu", "1",
    ])
    assert result.exit_code != 0
    assert "only apply to --backend aretomo" in result.output


# --- excluding views by angle in the coarse step ----------------------------------


def test_views_are_named_by_their_nominal_angle_within_half_a_degree():
    manifest = _manifest()
    order = sorted(range(5), key=lambda i: manifest.images[i].tilt_angle_deg)
    # -6.01 is the -6 view (position 0), 3 the +3 view (position 3).
    assert _views_at_angles(manifest, order, [-6.01, 3]) == {0, 3}
    with pytest.raises(ValueError, match="names 0 views"):
        _views_at_angles(manifest, order, [9.0])
    crowded = manifest.model_copy(update={"images": [
        *manifest.images,
        manifest.images[0].model_copy(update={"z_value": 5, "tilt_angle_deg": 0.3}),
    ]})
    with pytest.raises(ValueError, match="names 2 views"):
        _views_at_angles(crowded, list(range(6)), [0.1])


def test_the_coarse_step_skips_and_records_the_excluded_views(tmp_path):
    data = np.stack([np.arange(16, dtype=np.float32).reshape(4, 4) + z for z in range(3)])
    path = tmp_path / "TS_B.zarr"
    zarr.save(path, data)
    stack = Artifact(
        id="TS_B:tilt_stack", kind=ArtifactKind.TILT_STACK, path=path, shape=(3, 4, 4),
        dtype="float32", axis_order=AxisOrder.TYX, pixel_spacing_angstrom=1.0,
        parameters={"tilt_series_id": "TS_B"},
    )
    manifest = TiltSeriesManifest(
        tilt_series_id="TS_B", source_mdoc=Path("TS_B.mdoc"), raw_pixel_spacing_angstrom=1.0,
        images=[
            TiltImage(z_value=z, tilt_angle_deg=angle, subframe_path=f"{z}.mrc",
                      num_subframes=1, pixel_spacing_angstrom=1.0, binning=1)
            for z, angle in enumerate([0.0, -3.0, 3.0])
        ],
    )
    executable = tmp_path / "imod" / "bin" / "tiltxcorr"
    executable.parent.mkdir(parents=True)
    executable.touch()
    (executable.parent / "xftoxg").touch()
    captured: list[str] = []

    def fake_runner(command, *, cwd, env, stdin=None):
        del cwd, env, stdin
        name = "-output" if Path(command[0]).name == "tiltxcorr" else "-goutput"
        if name == "-output":
            captured.extend(command)
        Path(command[command.index(name) + 1]).write_text("1 0 0 1 0 0\n" * 3)
        return subprocess.CompletedProcess(command, 0, stdout="", stderr="")

    artifact = ImodTiltXcorrAlignmentBackend(fake_runner).align(
        stack, manifest,
        BackendContext(
            output_dir=tmp_path / "out", device=DevicePreference.CPU,
            parameters={"binning": 2, "exclude_tilt_angles": [3.0],
                        "tiltxcorr_executable": executable,
                        "imod_dir": executable.parent.parent, "tilt_axis_angle_deg": 84.1},
        ),
    )
    # Tilt order -3, 0, 3: the +3 view is the third, so tiltxcorr skips view 3.
    assert captured[captured.index("-skip") + 1] == "3"
    alignment = TiltAlignment.model_validate_json(artifact.path.read_text())
    assert alignment.excluded_z_values == [2]
    assert artifact.parameters["exclude_tilt_angles"] == [3.0]


# --- the lane runner and the submitter --------------------------------------------


def _aretomo_lane() -> dict:
    config = tomllib.loads((ROOT / "configs" / "lanes" / "ribo-bin8.toml").read_text())
    config = copy.deepcopy(config)
    config.pop("fiducials")
    config["fine_align"] = {"backend": "aretomo", "aretomo_align_z_nm": 200.0,
                            "aretomo_tilt_cor": False}
    config["align"]["exclude_tilt_angles"] = [-75, -66, 6]
    config["executables"]["aretomo"] = "$ARETOMO"
    return config


def test_an_aretomo_lane_skips_the_fiducial_steps_and_renders_valid_commands():
    from test_run_series_lanes import _assert_valid_commands

    config = _aretomo_lane()
    assert run_series.unknown_keys(config) == []
    names = _assert_valid_commands(config, "aretomo")
    assert "generate-fiducial-seed" not in names and "track-fiducials" not in names
    steps = dict(run_series.build_steps(config, Path("p"), "TS_X"))
    fine = steps["fine-align-tilt-series"]
    assert fine[fine.index("--backend") + 1] == "aretomo"
    assert fine[fine.index("--aretomo-executable") + 1] == "$ARETOMO"
    assert fine[fine.index("--aretomo-align-z-nm") + 1] == "200.0"
    assert "--no-aretomo-tilt-cor" in fine
    align = steps["align-tilt-series"]
    assert align[align.index("--exclude-tilt-angles") + 1] == "-75,-66,6"
    assert run_series.combination_text(run_series.lane_combination(config)) == (
        "stack + AreTomo, no membranes"
    )


def test_the_rendered_exclusion_parses_back_to_the_angles():
    """click takes the next token as the value even when it starts with a minus."""

    import typer

    command = typer.main.get_command(cli.app).commands["align-tilt-series"]
    context = command.make_context(
        "align-tilt-series",
        ["--manifest", "m", "--registry", "r", "--out", "o",
         "--exclude-tilt-angles", "-75,-66,6", "--binning", "8"],
    )
    assert context.params["exclude_tilt_angles"] == "-75,-66,6"
    assert context.params["binning"] == 8


@pytest.mark.parametrize(
    "change, message",
    [
        (lambda c: c.update(fiducials={"seed_backend": "imod-patch-seed",
                                       "track_backend": "imod-patch-track"}),
         r"\[fiducials\] is not read"),
        (lambda c: c["fine_align"].update(robust_fitting=True), "does not apply"),
        (lambda c: c["reconstruct"].pop("thickness"), "thickness must be set"),
        (lambda c: c["fine_align"].update(backend="aretomo3"), "must be one of"),
        (lambda c: c["align"].update(exclude_tilt_angles="-75"), "list of angles"),
    ],
)
def test_an_aretomo_lane_that_would_fail_later_is_refused_now(change, message):
    config = _aretomo_lane()
    change(config)
    with pytest.raises(SystemExit, match=message):
        run_series.build_steps(config, Path("p"), "TS_X")


def test_aretomo_keys_are_refused_in_a_tiltalign_lane():
    config = tomllib.loads((ROOT / "configs" / "lanes" / "ribo-bin8.toml").read_text())
    config.setdefault("fine_align", {})["aretomo_gpu"] = 0
    with pytest.raises(SystemExit, match="does not apply"):
        run_series.build_steps(config, Path("p"), "TS_X")


@pytest.mark.parametrize("gpu", [False, True], ids=["cpu-stage", "gpu-stage"])
def test_the_submitter_refuses_aretomo_in_a_stage_without_a_gpu(gpu):
    # ribo-bin8's own stages: "cpu" runs import-tilt-stack .. reconstruct-tomogram.
    config = _aretomo_lane()
    if gpu:
        config["run"]["stage"][0]["sbatch"] = ["--partition=gpu", "--gres=gpu:a40:1"]
    lane = submit_lanes.plan(config, ROOT / "configs" / "lanes" / "ribo-bin8.toml")
    found = [one for one in submit_lanes.problems(config, lane, lane.stages) if "GPU" in one]
    if gpu:
        assert found == []
    else:
        assert found == [
            "stage 'cpu' runs fine-align-tilt-series with AreTomo2, which needs a GPU; "
            "give that stage --partition=gpu and --gres=gpu:..."
        ]
