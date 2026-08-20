from __future__ import annotations

from pathlib import Path

import pytest

from cryoet_pipeline.imod_defocus import (
    CtffindRow,
    parse_ctffind_summary,
    read_imod_defocus_file,
    write_imod_defocus_file,
)


def _row(view: int, defocus1: float = 40000.0, defocus2: float = 39000.0) -> CtffindRow:
    return CtffindRow(
        view=view,
        defocus1_angstrom=defocus1,
        defocus2_angstrom=defocus2,
        astigmatism_azimuth_deg=-43.2,
        phase_shift_rad=1.47,
        cross_correlation=0.12,
        fit_resolution_angstrom=18.0,
    )


def test_write_defocus_header_flags_match_imod_documented_examples(tmp_path: Path) -> None:
    # The IMOD ctfphaseflip(1) man page gives these exact header lines:
    # astigmatism only -> "1 0 0. 0. 0 3"; phase in radians -> "12 ...";
    # both -> "13 ...".
    rows = [_row(1)]
    angles = [-58.94]

    astigmatism_only = tmp_path / "astig.defocus"
    write_imod_defocus_file(
        astigmatism_only, rows, angles, include_astigmatism=True, include_phase_shift=False
    )
    assert astigmatism_only.read_text().splitlines()[0] == "1 0 0.0 0.0 0.0 3"

    phase_only = tmp_path / "phase.defocus"
    write_imod_defocus_file(
        phase_only, rows, angles, include_astigmatism=False, include_phase_shift=True
    )
    assert phase_only.read_text().splitlines()[0] == "12 0 0.0 0.0 0.0 3"

    both = tmp_path / "both.defocus"
    write_imod_defocus_file(
        both, rows, angles, include_astigmatism=True, include_phase_shift=True
    )
    assert both.read_text().splitlines()[0] == "13 0 0.0 0.0 0.0 3"

    plain = tmp_path / "plain.defocus"
    write_imod_defocus_file(
        plain, rows, angles, include_astigmatism=False, include_phase_shift=False
    )
    assert plain.read_text().splitlines()[0] == "0 0 0.0 0.0 0.0 3"


def test_write_defocus_converts_angstrom_to_nanometres_and_keeps_exact_angles(
    tmp_path: Path,
) -> None:
    path = tmp_path / "out.defocus"
    write_imod_defocus_file(
        path,
        [_row(1, defocus1=43760.0, defocus2=43760.0)],
        [-58.937512],
        include_astigmatism=False,
        include_phase_shift=False,
    )

    data_line = path.read_text().splitlines()[1].split()
    assert data_line[0] == "1"
    assert data_line[1] == "1"
    # The man page warns against rounding angles to one decimal place.
    assert data_line[2] == "-58.937512"
    assert data_line[3] == "-58.937512"
    assert float(data_line[4]) == pytest.approx(4376.0)


def test_write_then_read_round_trips_astigmatism(tmp_path: Path) -> None:
    path = tmp_path / "out.defocus"
    rows = [_row(1, 40000.0, 39000.0), _row(2, 41000.0, 40200.0)]
    angles = [-58.94, -56.94]

    write_imod_defocus_file(path, rows, angles, include_astigmatism=True)
    entries = read_imod_defocus_file(path)

    assert len(entries) == 2
    assert entries[0].view_start == 1
    assert entries[0].angle_start_deg == pytest.approx(-58.94)
    assert entries[0].defocus_nm == pytest.approx(4000.0)
    assert entries[0].second_defocus_nm == pytest.approx(3900.0)
    assert entries[0].astigmatism_axis_deg == pytest.approx(-43.2)
    assert entries[0].mean_defocus_nm == pytest.approx(3950.0)
    assert entries[1].defocus_nm == pytest.approx(4100.0)


def test_write_then_read_round_trips_phase_shift(tmp_path: Path) -> None:
    path = tmp_path / "out.defocus"
    write_imod_defocus_file(
        path, [_row(1)], [-58.94], include_astigmatism=True, include_phase_shift=True
    )

    entries = read_imod_defocus_file(path)
    assert entries[0].phase_shift == pytest.approx(1.47)


def test_read_legacy_version_two_file_treats_first_line_as_data(tmp_path: Path) -> None:
    # Version 1/2 files carry data on the first line with the version appended,
    # which is what ctfplotter writes by default.
    path = tmp_path / "legacy.defocus"
    path.write_text("61 61 -58.94 -58.94 4376 2\n60 60 -56.94 -56.94 4599\n")

    entries = read_imod_defocus_file(path)

    assert len(entries) == 2
    assert entries[0].view_start == 61
    assert entries[0].defocus_nm == pytest.approx(4376.0)
    assert entries[0].second_defocus_nm is None
    assert entries[1].defocus_nm == pytest.approx(4599.0)


def test_read_plain_file_without_any_version_marker(tmp_path: Path) -> None:
    path = tmp_path / "plain.defocus"
    path.write_text("20 20 0. 0. 5000\n")

    entries = read_imod_defocus_file(path)

    assert len(entries) == 1
    assert entries[0].defocus_nm == pytest.approx(5000.0)


def test_read_rejects_unsupported_flag_combinations(tmp_path: Path) -> None:
    radians = tmp_path / "radians.defocus"
    radians.write_text("3 0 0.0 0.0 0.0 3\n1 1 -58.94 -58.94 4000.0 3900.0 -0.75\n")
    with pytest.raises(ValueError, match="astigmatism angles in radians"):
        read_imod_defocus_file(radians)

    cut_on = tmp_path / "cuton.defocus"
    cut_on.write_text("37 0 0.0 0.0 0.0 3\n1 1 -58.94 -58.94 4000.0 3900.0 -43.2 1.47 0.17\n")
    with pytest.raises(ValueError, match="cut-on frequency"):
        read_imod_defocus_file(cut_on)


def test_read_rejects_column_count_mismatch(tmp_path: Path) -> None:
    path = tmp_path / "bad.defocus"
    path.write_text("1 0 0.0 0.0 0.0 3\n1 1 -58.94 -58.94 4000.0\n")

    with pytest.raises(ValueError, match="expected 7 columns"):
        read_imod_defocus_file(path)


def test_write_rejects_view_without_tilt_angle(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="no matching tilt angle"):
        write_imod_defocus_file(tmp_path / "out.defocus", [_row(3)], [-58.94, -56.94])


def test_parse_ctffind_summary_reads_all_rows_and_skips_comments(tmp_path: Path) -> None:
    path = tmp_path / "ctffind.txt"
    path.write_text(
        "# Output from CTFFIND version 5.0.2\n"
        "# Columns: micrograph number; defocus 1 [A]; defocus 2 [A]; azimuth\n"
        "1.0 40000.0 39000.0 -43.2 0.0 0.11 18.5\n"
        "2.0 41000.0 40500.0 -41.0 0.0 0.09 19.2\n"
    )

    rows = parse_ctffind_summary(path)

    assert [row.view for row in rows] == [1, 2]
    assert rows[0].defocus1_angstrom == pytest.approx(40000.0)
    assert rows[0].mean_defocus_angstrom == pytest.approx(39500.0)
    assert rows[0].astigmatism_angstrom == pytest.approx(500.0)
    assert rows[0].fit_resolution_angstrom == pytest.approx(18.5)
    assert rows[1].view == 2


def test_parse_ctffind_summary_rejects_bad_input(tmp_path: Path) -> None:
    empty = tmp_path / "empty.txt"
    empty.write_text("# only a comment\n")
    with pytest.raises(ValueError, match="no CTFFIND data rows"):
        parse_ctffind_summary(empty)

    negative = tmp_path / "negative.txt"
    negative.write_text("1.0 -100.0 39000.0 -43.2 0.0\n")
    with pytest.raises(ValueError) as failure:
        parse_ctffind_summary(negative)
    message = str(failure.value)
    assert "view 1" in message, "the failing view is what the user has to act on"
    assert "positive underfocus" in message
    assert "--ctffind-restrain-astigmatism-angstrom" in message

    duplicate = tmp_path / "duplicate.txt"
    duplicate.write_text("1.0 40000.0 39000.0 -43.2 0.0\n1.0 40000.0 39000.0 -43.2 0.0\n")
    with pytest.raises(ValueError, match="duplicate view number"):
        parse_ctffind_summary(duplicate)
