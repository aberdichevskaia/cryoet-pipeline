"""The lineage of a result as a methods record (docs/entry_and_exit.md, Exits).

Everything a result descends from, from the raw data or the imported roots on: each
step with the program that made it, that program's version as the record has it, and
its parameters; what entered from outside, with what was declared about it and the
fingerprints of the files read (the sha256 of their bytes); and for an MRA run its
plan and the records of its nodes. It is what a paper's methods section needs, and it
is read from the registry and the run records only. It holds nothing they do not: a
version a record lacks is written as not recorded, and so is the pipeline's own code
version, which only run records keep.

Artifacts of one step that differ only by a trailing number (one per tilt view, say)
are one step here, their shared parameters once and the ones that vary as a range.
Parameters that are paths stay in the JSON and are left out of the Markdown.
"""

from __future__ import annotations

import json
import re
from collections.abc import Iterable, Sequence
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from cryoet_pipeline.artifacts import ArtifactRegistry
from cryoet_pipeline.models import Artifact, ArtifactKind, Provenance
from cryoet_pipeline.runs import RunInput, RunRecord, files_fingerprint, read_record, writing
from cryoet_pipeline.tool_versions import EXTERNAL

STEP = "methods-record"
STEP_VERSION = 1
BACKEND = "registry-reader"
BACKEND_VERSION = 1
NOT_RECORDED = "not recorded"
LONG_LIST = 8

_NUMBERED = re.compile(r":\d+$")


class MethodsRefused(ValueError):
    """A methods record the registry or the run cannot give."""


@dataclass(frozen=True)
class MethodsRecord:
    folder: Path
    record: dict[str, Any]
    markdown: str


def ordered_lineage(registry: ArtifactRegistry, ids: Sequence[str]) -> list[Artifact]:
    """The artifacts `ids` and all they descend from, each once, every parent before
    its children. A parent the registry no longer holds is skipped."""

    order: list[Artifact] = []
    seen: set[str] = set()
    known = registry.artifact_ids

    def visit(artifact_id: str) -> None:
        # Iterative post-order: lineages of thousands of artifacts stay off the stack.
        stack: list[tuple[str, bool]] = [(artifact_id, False)]
        while stack:
            current, expanded = stack.pop()
            if expanded:
                order.append(registry.get(current))
                continue
            if current in seen or current not in known:
                continue
            seen.add(current)
            stack.append((current, True))
            for parent in reversed(registry.get(current).parent_ids):
                if parent not in seen:
                    stack.append((parent, False))

    for artifact_id in ids:
        visit(artifact_id)
    return order


def _group_name(artifact: Artifact) -> str:
    return _NUMBERED.sub(":*", artifact.id)


def _is_path(key: str) -> bool:
    return key == "path" or key.endswith(("_path", "_paths", "_dir", "_file", "_files", "_mdoc"))


def _backend(artifact: Artifact) -> str | None:
    value = artifact.backend or artifact.parameters.get("backend")
    return str(value) if value is not None else None


def _versions(artifacts: Iterable[Artifact]) -> dict[str, list[str]]:
    found: dict[str, set[str]] = {}
    for artifact in artifacts:
        for program, value in artifact.software_versions.items():
            found.setdefault(program, set()).add(NOT_RECORDED if value == EXTERNAL else value)
    return {program: sorted(values) for program, values in sorted(found.items())}


def _shared_and_varying(artifacts: Sequence[Artifact]) -> tuple[dict[str, Any], dict[str, Any]]:
    keys = sorted({key for artifact in artifacts for key in artifact.parameters})
    shared: dict[str, Any] = {}
    varying: dict[str, Any] = {}
    for key in keys:
        values = [artifact.parameters.get(key) for artifact in artifacts]
        first = json.dumps(values[0], sort_keys=True, default=str)
        if all(json.dumps(value, sort_keys=True, default=str) == first for value in values):
            shared[key] = values[0]
            continue
        numbers = [value for value in values if isinstance(value, (int, float))
                   and not isinstance(value, bool)]
        varying[key] = ({"from": min(numbers), "to": max(numbers)}
                        if len(numbers) == len(values) else "varies")
    return shared, varying


def _run_summary(record: RunRecord, folder: Path) -> dict[str, Any]:
    return {
        "folder": str(folder),
        "step": record.step,
        "step_version": record.step_version,
        "backend": record.backend,
        "backend_version": record.backend_version,
        "declared": record.content_params,
        "inputs": [{"slot": item.slot, "path": item.path, "fingerprint": item.fingerprint}
                   for item in record.inputs],
        "code": record.code or {"package": NOT_RECORDED},
        "ended_at": record.ended_at,
        "host": record.host,
    }


def _run_of(artifact: Artifact) -> dict[str, Any] | None:
    value = artifact.parameters.get("run_path")
    if not isinstance(value, str):
        return None
    record = read_record(Path(value))
    return _run_summary(record, Path(value)) if record is not None else None


def _step(artifacts: list[Artifact], group_of: dict[str, str]) -> dict[str, Any]:
    first = artifacts[0]
    shared, varying = _shared_and_varying(artifacts)
    runs = [run for run in (_run_of(artifact) for artifact in artifacts) if run is not None]
    parents = sorted({group_of[parent] for artifact in artifacts
                      for parent in artifact.parent_ids if parent in group_of}
                     - {_group_name(first)})
    return {
        "name": _group_name(first) if len(artifacts) > 1 else first.id,
        "ids": [artifact.id for artifact in artifacts],
        "kind": first.kind.value,
        "count": len(artifacts),
        "backend": _backend(first),
        "provenance": sorted({artifact.provenance.value for artifact in artifacts}),
        "software_versions": _versions(artifacts),
        "pixel_size_angstrom": sorted({artifact.pixel_spacing_angstrom for artifact in artifacts
                                       if artifact.pixel_spacing_angstrom is not None}),
        "parameters": shared,
        "varying": varying,
        "parents": parents,
        "runs": runs,
    }


def _steps(lineage: list[Artifact]) -> list[dict[str, Any]]:
    """One step per group of the lineage, in the order its first member comes."""

    groups: dict[tuple[str, str, str | None], list[Artifact]] = {}
    for artifact in lineage:
        groups.setdefault((_group_name(artifact), artifact.kind.value, _backend(artifact)),
                          []).append(artifact)
    group_of = {artifact.id: (_group_name(artifact) if len(groups[key]) > 1 else artifact.id)
                for key, members in groups.items() for artifact in members}
    return [_step(members, group_of) for members in groups.values()]


def _mra(run: Path) -> tuple[dict[str, Any], list[str], list[dict[str, Any]]]:
    """An MRA run of the registry: its plan, its stacks' ids, and its nodes' records."""

    from cryoet_pipeline.particle_sources import particle_stack

    plan_path = run / "plan.json"
    if not plan_path.is_file():
        raise MethodsRefused(f"{run} has no plan.json: not an MRA run")
    plan = json.loads(plan_path.read_text())
    particles = plan.get("particles")
    if not isinstance(particles, dict):
        raise MethodsRefused(
            f"{run} read its particles from {particles!r}, an export folder, not from a "
            "registry: their lineage is in the registry the export was made from"
        )
    project = Path(str(particles["project"]))
    registry = ArtifactRegistry.load(project / "artifacts.json")
    stacks = []
    for series in particles["series"]:
        stacks.append(particle_stack(registry, series, particles.get("label")).id)
        if particles.get("decoy_label") is not None:
            stacks.append(particle_stack(registry, series, particles["decoy_label"]).id)
    nodes = []
    for folder in sorted(path for path in run.iterdir() if path.is_dir()):
        record = read_record(folder)
        if record is not None and record.status == "succeeded":
            nodes.append(_run_summary(record, folder))
    return plan, stacks, nodes


def methods_record(project: Path, *, artifact_ids: Sequence[str] = (),
                   run: Path | None = None) -> dict[str, Any]:
    """The methods record of `artifact_ids` of `project`, or of an MRA `run`."""

    if bool(artifact_ids) == (run is not None):
        raise MethodsRefused("give the result: artifacts of the registry, or an MRA run")
    mra: dict[str, Any] | None = None
    ids = list(artifact_ids)
    if run is not None:
        plan, ids, nodes = _mra(Path(run))
        project = Path(str(plan["particles"]["project"]))
        mra = {"run": str(run), "plan": plan, "nodes": nodes}
    registry = ArtifactRegistry.load(Path(project) / "artifacts.json")
    missing = [artifact_id for artifact_id in ids if artifact_id not in registry.artifact_ids]
    if missing:
        raise MethodsRefused(f"{project} has no artifact {missing[0]!r}")
    lineage = [artifact for artifact in ordered_lineage(registry, ids)
               if artifact.kind is not ArtifactKind.QC]
    steps = _steps(lineage)
    programs: dict[str, dict[str, list[str]]] = {}
    for step in steps:
        for program, versions in step["software_versions"].items():
            for value in versions:
                programs.setdefault(program, {}).setdefault(value, []).append(step["name"])
    outside = []
    for artifact in lineage:
        if artifact.provenance is Provenance.DERIVED:
            continue
        run_summary = _run_of(artifact)
        outside.append({
            "id": artifact.id,
            "kind": artifact.kind.value,
            "provenance": artifact.provenance.value,
            "backend": _backend(artifact),
            "source": artifact.parameters.get("source_path"),
            "declared": run_summary["declared"] if run_summary else NOT_RECORDED,
            "inputs": run_summary["inputs"] if run_summary else [],
        })
    return {
        "result": {"artifacts": list(artifact_ids), "run": str(run) if run else None},
        "project": str(project),
        "written_at": datetime.now(UTC).isoformat(timespec="seconds"),
        "programs": programs,
        "not_made_here": outside,
        "steps": steps,
        "mra": mra,
    }


def _value(value: Any) -> str:
    if isinstance(value, list) and len(value) > LONG_LIST:
        return f"{len(value)} values"
    if isinstance(value, float):
        return f"{value:g}"
    if isinstance(value, dict) and set(value) == {"from", "to"}:
        return f"{_value(value['from'])} to {_value(value['to'])}"
    return json.dumps(value, default=str) if isinstance(value, (list, dict)) else str(value)


def _parameters_line(step: dict[str, Any]) -> str:
    parts = [f"{key} {_value(value)}" for key, value in step["parameters"].items()
             if not _is_path(key) and key not in {"backend", "run_fingerprint"}]
    parts += [f"{key} {_value(value)}" for key, value in step["varying"].items()
              if not _is_path(key)]
    return "; ".join(parts) or "none recorded"


def methods_markdown(record: dict[str, Any]) -> str:
    result = record["result"]
    named = result["run"] or ", ".join(result["artifacts"])
    lines = [
        f"# Methods record: {named}",
        "",
        f"Written {record['written_at']} by `cryoet methods-record` from the registry and",
        f"the run records of {record['project']}. It holds what those hold and nothing more:",
        "a version a record lacks is written as not recorded.",
        "",
        "## Programs",
        "",
        "| program | version | steps |",
        "|---|---|---|",
    ]
    for program, versions in record["programs"].items():
        for value, steps in versions.items():
            lines.append(f"| {program} | {value} | {', '.join(steps)} |")
    if not record["programs"]:
        lines.append("| none recorded | | |")
    lines += ["", "## Not made here", ""]
    if not record["not_made_here"]:
        lines.append("Nothing: every step from the raw data on was run in this project.")
    for item in record["not_made_here"]:
        declared = item["declared"]
        said = (", ".join(f"{key} {_value(value)}" for key, value in declared.items())
                if isinstance(declared, dict) else NOT_RECORDED)
        lines.append(f"- `{item['id']}` ({item['kind']}), {item['provenance']} by "
                     f"{item['backend'] or NOT_RECORDED}"
                     + (f" from {item['source']}" if item["source"] else "")
                     + f". Declared: {said}.")
        for entry in item["inputs"]:
            lines.append(f"  - {entry['slot']}: {entry['path']} "
                         f"(fingerprint {entry['fingerprint']})")
    lines += ["", "## Steps, from the data to the result", ""]
    for number, step in enumerate(record["steps"], start=1):
        versions = "; ".join(f"{program} {', '.join(values)}"
                             for program, values in step["software_versions"].items())
        made = "imported" if step["provenance"] == ["imported"] else (
            "attested" if step["provenance"] == ["attested"] else "made here")
        count = f"{step['count']} artifacts, " if step["count"] > 1 else ""
        lines.append(f"{number}. **{step['kind'].replace('_', ' ')}** `{step['name']}` "
                     f"({count}{made}): backend {step['backend'] or NOT_RECORDED}; "
                     f"versions: {versions or NOT_RECORDED}.")
        if step["pixel_size_angstrom"]:
            sizes = ", ".join(f"{value:g}" for value in step["pixel_size_angstrom"])
            lines.append(f"   Pixel {sizes} A.")
        lines.append(f"   Parameters: {_parameters_line(step)}.")
        if step["parents"]:
            lines.append(f"   From: {', '.join(f'`{parent}`' for parent in step['parents'])}.")
        for run in step["runs"]:
            code = ", ".join(f"{key} {value}" for key, value in run["code"].items())
            lines.append(f"   Run record: {run['step']} v{run['step_version']}, "
                         f"{run['backend']} v{run['backend_version']}, code {code}.")
    without = [step["name"] for step in record["steps"] if not step["runs"]]
    if without:
        lines += ["", f"The pipeline's own code version: {NOT_RECORDED} for {len(without)} of "
                  f"{len(record['steps'])} steps, which have no run record; only run records "
                  "keep it."]
    if record["mra"] is not None:
        mra = record["mra"]
        lines += ["", "## MRA run", "", f"`{mra['run']}`, plan.json:", "", "```json",
                  json.dumps(mra["plan"], indent=1, default=str), "```", ""]
        for node in mra["nodes"]:
            code = ", ".join(f"{key} {value}" for key, value in node["code"].items())
            lines.append(f"- {Path(node['folder']).name}: {node['step']} "
                         f"v{node['step_version']}, {node['backend']} "
                         f"v{node['backend_version']}, code {code}, ended {node['ended_at']}")
    lines.append("")
    return "\n".join(lines)


def write_methods_record(project: Path, out: Path, *, artifact_ids: Sequence[str] = (),
                         run: Path | None = None) -> MethodsRecord:
    """Write methods.json and methods.md into `out`, a run folder of its own."""

    record = methods_record(project, artifact_ids=artifact_ids, run=run)
    markdown = methods_markdown(record)
    registry_file = Path(record["project"]) / "artifacts.json"
    inputs = [RunInput(slot="registry", path=str(registry_file.resolve()),
                       fingerprint=files_fingerprint([registry_file]))]
    content = {"artifacts": list(artifact_ids), "run": str(run) if run else None}
    out = Path(out)
    if out.exists():
        raise MethodsRefused(f"{out} is there already; a record is never written over")
    out.parent.mkdir(parents=True, exist_ok=True)
    with writing(out, step=STEP, step_version=STEP_VERSION, backend=BACKEND,
                 backend_version=BACKEND_VERSION, content_params=content,
                 inputs=inputs) as attempt:
        (attempt.folder / "methods.json").write_text(
            json.dumps(record, indent=1, default=str) + "\n")
        (attempt.folder / "methods.md").write_text(markdown)
        attempt.add(ArtifactKind.DATASET_EXPORT, "methods.json")
        attempt.add(ArtifactKind.DATASET_EXPORT, "methods.md")
    return MethodsRecord(folder=out, record=record, markdown=markdown)


__all__ = [
    "MethodsRecord",
    "MethodsRefused",
    "methods_markdown",
    "methods_record",
    "ordered_lineage",
    "write_methods_record",
]
