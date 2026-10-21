"""Write a PDF report of a Dynamo multireference classification (MRA) project.

    python scripts/dynamo_classification_report.py <export>/mra_nss

reads the project folder, and the export manifest of scripts/export_dynamo.py
beside it if there is one (pixel size, tilt-series and picking score of every
particle), and writes <export>/mra_nss_classification_report.pdf. It changes
nothing in the project, so it can run while Dynamo is still iterating: it
reports the iterations finished so far.

The pages: parameters; particles at their starting poses; the average of all
of them (computed here in about a minute for 2,700 particles of 48^3, unless
--initial-average names a map); the starting references; class sizes per
iteration; every iteration's classes as z sections, as Dynamo's dslices shows
a map; the last classes by tilt-series and picking score; particles of each
last class at its own poses. See src/cryoet_pipeline/reports/classification.py.

--template names the map the picks were matched with (MRC or EM), for instance
outputs/ribo-mdc/templates/TS_0306/TS_0306_pytom_bin8_template.mrc: every page of
particles or maps then shows it in its top right corner, drawn as that page draws
its own and brought to the particles' box and pixel size. Without it the corner
stays empty.
"""

from __future__ import annotations

import argparse
import subprocess
from datetime import datetime
from importlib.metadata import PackageNotFoundError
from importlib.metadata import version as package_version
from pathlib import Path

from cryoet_pipeline.reports.classification import write_classification_report
from cryoet_pipeline.reports.dynamo_project import read_dynamo_classification


def software() -> str:
    try:
        description = f"cryoet-pipeline {package_version('cryoet-pipeline')}"
    except PackageNotFoundError:
        description = "cryoet-pipeline"
    source = Path(__file__).resolve().parent
    commit = subprocess.run(["git", "-C", str(source), "rev-parse", "--short", "HEAD"],
                            capture_output=True, text=True, timeout=10, check=False)
    if commit.returncode == 0 and commit.stdout.strip():
        description += f" at {commit.stdout.strip()}"
    return description + ", scripts/dynamo_classification_report.py"


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("project", type=Path, help="the Dynamo project folder")
    parser.add_argument("--out", type=Path,
                        help="the PDF; default <project>_classification_report.pdf beside it")
    parser.add_argument("--manifest", type=Path,
                        help="export_manifest.json; default the one beside the project, if any")
    parser.add_argument("--pixel-size", type=float, help="Å per voxel, when no manifest gives it")
    parser.add_argument("--initial-average", type=Path,
                        help="a map (EM or MRC) to show as the average of all particles at their "
                        "starting poses, instead of computing it")
    parser.add_argument("--no-initial-average", action="store_true",
                        help="leave the average of all particles out")
    parser.add_argument("--template", type=Path,
                        help="the template the picks were matched with (MRC or EM), drawn in "
                        "the corner of every page of particles or maps")
    args = parser.parse_args()

    project = args.project.resolve()
    out = args.out or project.parent / f"{project.name}_classification_report.pdf"
    if out.exists():
        print(f"replacing {out}")
    run = read_dynamo_classification(
        project,
        manifest=args.manifest,
        pixel_size_angstrom=args.pixel_size,
        initial_average=args.initial_average,
        compute_initial_average=not args.no_initial_average,
        template=args.template,
    )
    finished = len(run.iterations)
    planned = f" of {run.planned_iterations}" if run.planned_iterations else ""
    print(f"{run.name}: {len(run.tags):,} particles, {len(run.references)} channels, "
          f"{finished}{planned} iterations finished")
    if run.initial_average is not None and args.initial_average is None:
        print("computing the average of all particles at their starting poses")
    outcome = write_classification_report(
        out, run, written=datetime.now().strftime("%Y-%m-%d %H:%M"), software=software()
    )
    print(f"wrote {outcome.path} ({outcome.pages} pages)")
    for problem in outcome.problems:
        print(f"  not read: {problem}")


if __name__ == "__main__":
    main()
