"""Write a PDF report of a multireference classification (MRA): a Dynamo project, or
a run of the pipeline's port.

    python scripts/dynamo_classification_report.py <export>/mra_nss

reads the run's folder through reports.classification_reader, a run of the
run-record layout when it holds a plan.json (docs/run_records.md) and a Dynamo
project otherwise, and the export manifest of scripts/export_dynamo.py beside it if
there is one (pixel size, tilt-series and picking score of every particle), and
writes <export>/mra_nss_classification_report.pdf. It changes nothing in the run,
so it can run while the run is still iterating: it reports the iterations finished
so far.

The pages: parameters; particles at their starting poses; the average of all
of them (computed here in about a minute for 2,700 particles of 48^3, unless
--initial-average names a map); the starting references; class sizes per
iteration; every iteration's classes as z sections, as Dynamo's dslices shows
a map; the last classes by tilt-series and picking score; where each last
class's particles were picked, a page per class and tomogram drawn as the
lane report's picks page (the PI, 2026-10-26); particles of each last class at
its own poses. See src/cryoet_pipeline/reports/classification.py. The
where-picked pages need the pipeline project the export names (its manifest's
source_project) and its tomograms: on the cluster they are there; elsewhere the
pages are left out and the report says why.

--template names the map the picks were matched with (MRC or EM), for instance
outputs/ribo-mdc/templates/TS_0306/TS_0306_pytom_bin8_template.mrc: every page of
particles or maps then shows it in its top right corner, drawn as that page draws
its own and brought to the particles' box and pixel size. Without it the corner
stays empty.

--workers is how many threads compute the average of all particles; the map does
not depend on it. By default the CPUs Slurm gave the job (SLURM_CPUS_PER_TASK), so
the report at the end of scripts/mra_port_run.sbatch uses the job's CPUs, and one
thread outside a job.
"""

from __future__ import annotations

import argparse
import os
import subprocess
from datetime import datetime
from importlib.metadata import PackageNotFoundError
from importlib.metadata import version as package_version
from pathlib import Path

from cryoet_pipeline.reports.classification import write_classification_report
from cryoet_pipeline.reports.classification_reader import read_classification


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
    parser.add_argument("project", type=Path,
                        help="the run: a Dynamo project, or a run of the port (its plan.json)")
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
    parser.add_argument("--workers", type=int,
                        default=int(os.environ.get("SLURM_CPUS_PER_TASK") or 1),
                        help="threads for the average of all particles (default: the job's "
                        "SLURM_CPUS_PER_TASK, else 1); the map does not depend on them")
    args = parser.parse_args()
    if args.workers < 1:
        parser.error(f"--workers must be at least 1, got {args.workers}")

    project = args.project.resolve()
    out = args.out or project.parent / f"{project.name}_classification_report.pdf"
    if out.exists():
        print(f"replacing {out}")
    run = read_classification(
        project,
        manifest=args.manifest,
        pixel_size_angstrom=args.pixel_size,
        initial_average=args.initial_average,
        compute_initial_average=not args.no_initial_average,
        template=args.template,
        workers=args.workers,
    )
    finished = len(run.iterations)
    planned = f" of {run.planned_iterations}" if run.planned_iterations else ""
    print(f"{run.name}: {len(run.tags):,} particles, {len(run.references)} channels, "
          f"{finished}{planned} iterations finished")
    if run.initial_average is not None and args.initial_average is None:
        print(f"computing the average of all particles at their starting poses on "
              f"{args.workers} thread{'s' if args.workers > 1 else ''}")
    outcome = write_classification_report(
        out, run, written=datetime.now().strftime("%Y-%m-%d %H:%M"), software=software()
    )
    print(f"wrote {outcome.path} ({outcome.pages} pages)")
    for problem in outcome.problems:
        print(f"  not read: {problem}")


if __name__ == "__main__":
    main()
