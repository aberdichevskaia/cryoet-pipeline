#!/usr/bin/env python3
"""A project's results in a browser, served from where the data are.

    python3 scripts/serve_results.py --lane configs/lanes/ryr1-bin8.toml
    python3 scripts/serve_results.py --project outputs/ryr1 --variant bin8 --port 8765

A lane config names its project ([run] project) and the label of its picks ([lane]
variant), as scripts/show_project.py reads them; --project and --variant say the
same by hand, and --variant overrides the lane's label. A project holding several
lanes' picks (outputs/ryr1 holds bin8's and bin4's) needs the label.

Run it on the cluster, in a slogin job or a small job of its own, and open the
link it prints from a laptop through the SSH tunnel it also prints. The tunnel
goes through the login node as Slurm names it, which a laptop outside the site
may reach only by another name or through a gateway: --login (or
CRYOET_SSH_LOGIN) gives the Host of the laptop's ~/.ssh/config to print instead,
for example --login max-login where that Host jumps through MDC's gateway. The page
shows each series' tomogram as slabs on a slider, through z and through y, with
the picks drawn on them, a score histogram with a threshold, and each pick's
views turned into the template's frame. For the classification runs in
<project>/mra (or --mra): every iteration's classes, their sizes, each map's z
sections and central views, the maps in Mol*, and the FSC between half runs.

Run it from the clone's root, like every command: the registry names files from
there. Read-only: it reads the registry, never locks it, and writes only its cache of
pictures (by default <project>/results_cache). Before it prints the link it reads the
project once, as the page's first request does, and says what it found and how long
that took: the first reading imports the pipeline's commands, torch, scipy and
matplotlib among them (7 s on a Mac), and a page opened before then stays empty. The
first look at a series cuts its slabs, a few seconds for a bin8 tomogram; later looks
read the cache, until the tomogram or the picks are newer than it. Each answer is
logged with the time it took.

A prototype of step 1 of docs/dashboard.md.
"""

from __future__ import annotations

import argparse
import getpass
import importlib
import os
import secrets
import socket
import sys
import time
import tomllib
from collections.abc import Mapping
from pathlib import Path

from cryoet_pipeline.dashboard.data import ResultsProject
from cryoet_pipeline.dashboard.mra import ClassificationRuns
from cryoet_pipeline.dashboard.server import LOCAL_ONLY, ResultsServer, tunnel_lines

ROOT = Path(__file__).resolve().parents[1]
ASSETS = ROOT / "vendor"
LANES = ROOT / "configs" / "lanes"


def _from_root(path: str | Path) -> Path:
    """Paths in a lane config are relative to the repository root."""

    path = Path(path)
    return path if path.is_absolute() else ROOT / path


def lane_project(lane: Path) -> tuple[Path, str | None]:
    """A lane config's project folder and the label of its picks."""

    import run_series

    _, config = run_series.read_config(lane)
    name = (config.get("run") or {}).get("project")
    if not name:
        raise SystemExit(f"{lane}: no [run] project")
    return _from_root(name), (config.get("lane") or {}).get("variant")


def registry_hint(project: Path) -> str:
    """What a --project that holds no registry probably meant: the project of a lane
    of that name, and the folders beside it that hold one."""

    hints = []
    lane = LANES / f"{project.name}.toml"
    if lane.is_file():
        try:
            config = tomllib.loads(lane.read_text())
        except tomllib.TOMLDecodeError:
            config = {}
        target = (config.get("run") or {}).get("project")
        label = (config.get("lane") or {}).get("variant")
        if target:
            hints.append(f"{project.name} is a lane, whose project is {target}"
                         + (f" and whose picks are labelled {label}" if label else "")
                         + f": give --lane {lane.relative_to(ROOT)}")
    parent = project.parent
    if parent.is_dir():
        found = sorted(one.name for one in parent.iterdir() if (one / "artifacts.json").is_file())
        if found:
            hints.append(f"the folders in {parent} that hold a registry: {', '.join(found)}")
    return "".join(f"\n  {one}" for one in hints)


def access_lines(host: str, port: int, token: str, *, node: str, user: str,
                 login: str | None, environ: Mapping[str, str]) -> list[str]:
    """What to do on a laptop to open the pages: in a Slurm job, or with the login
    node named, the tunnel first; then the link.

    Without --login the login node is the host Slurm took the job from, under the
    cluster's user name. A laptop may not reach it by that name: from outside MDC
    max-login2 is reached only through MDC's ssh gateway, under a Host of the
    laptop's ~/.ssh/config, and a short name resolves only inside a site.
    """

    submit = environ.get("SLURM_SUBMIT_HOST")
    named = login or (f"{user}@{submit}" if submit else None)
    link = f"  http://localhost:{port}/?token={token}"
    if named is None and not environ.get("SLURM_JOB_ID"):
        return ["open:", link]
    lines = ["on your laptop, open a tunnel first:"]
    lines += [f"  {one}" for one in tunnel_lines(host, port, node, named, user)]
    if login is None:
        lines += [f"  ({named or '<login node>'} is the login node as Slurm names it: where your "
                  "laptop reaches it",
                  "  by another name or through a gateway, write that Host of its ~/.ssh/config "
                  "instead,",
                  "  or name it once with --login or CRYOET_SSH_LOGIN)"]
    if host in LOCAL_ONLY:
        lines += ["  (where the node refuses ssh, start this again with --host 0.0.0.0: the "
                  "tunnel then",
                  "  ends on the login node)"]
    else:
        # Munich, 27 November 2026: hpc-submit02 answered "connect failed: No route to
        # host" for a node's port, and the tunnel through to the node worked.
        lines += ['  (if ssh answers "connect failed", the login node cannot reach this port; '
                  "the tunnel",
                  "  can end on the node instead, where this server listens too:",
                  *(f"  {one})" for one in tunnel_lines("127.0.0.1", port, node, named, user))]
    # Munich, 27 November 2026: VS Code, connected to the login node, had forwarded the
    # port from there and held 127.0.0.1:<port> on the laptop, so the tunnel took only
    # [::1] and a browser met one dead end or the other.
    lines += ["  (from VS Code on the login node: VS Code may forward this port from the login "
              "node,",
              "  where the server is not; if ssh then says the port is in use, stop that forward "
              "in",
              "  VS Code's Ports panel, and open the link in your browser, not in VS Code)"]
    return [*lines, "then open:", link]


def read_once(project: ResultsProject, classification: ClassificationRuns) -> str:
    """Read the project as the page's first request will, and say what was found and
    how long each part took.

    The page's first request resolves each series' results through the pipeline's
    commands (cryoet_pipeline.cli), whose first import brings in torch, scipy and
    matplotlib, and reads every classification run's records. In Munich on 27 November
    2026 a page whose scripts had arrived stayed empty, with no answer to that request
    in the log yet. Read here, before the link is printed, the wait is in the terminal
    with its time, and a page opened from the link is answered at once. A failure is
    said, not raised: the page says the same.
    """

    began = time.monotonic()
    parts: list[str] = []

    def lap(since: float) -> str:
        return f"{time.monotonic() - since:.1f} s"

    try:
        importlib.import_module("cryoet_pipeline.cli")
        parts.append(f"the pipeline's modules {lap(began)}")
        mark = time.monotonic()
        series = project.overview()["series"]
        with_picks = sum(1 for one in series if one["picks"])
        parts.append(f"{len(series)} series, {with_picks} with picks, {lap(mark)}")
        mark = time.monotonic()
        if classification.folder.is_dir():
            listing = classification.listing()
            runs = listing["runs"]
            unread = sum(1 for one in runs if "problem" in one)
            parts.append(
                f"{len(runs)} classification run{'' if len(runs) == 1 else 's'}"
                + (f" ({unread} could not be read)" if unread else "")
                + ("" if runs else " (a run is listed by its plan.json; Dynamo's layout is not)")
                + f" and {len(listing['fsc'])} FSC in {classification.folder}, {lap(mark)}"
            )
        else:
            parts.append(f"no classification runs: no folder {classification.folder}")
    except Exception as exc:  # noqa: BLE001 - the page says the same failure
        parts.append(f"then failed: {type(exc).__name__}: {exc}")
        return f"reading the project took {lap(began)}: " + "; ".join(parts)
    return f"read in {lap(began)}: " + "; ".join(parts)


def arguments() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    where = parser.add_mutually_exclusive_group(required=True)
    where.add_argument("--lane", type=Path, default=None,
                       help="a lane config: its [run] project, and its [lane] variant as the "
                       "label of the picks")
    where.add_argument("--project", type=Path, default=None,
                       help="the project folder, holding artifacts.json and manifests/")
    parser.add_argument("--variant", default=None,
                        help="label of the picks to show, as for cryoet report --variant; "
                        "with --lane, instead of the lane's")
    parser.add_argument("--port", type=int, default=0,
                        help="port to listen on; 0 (the default) takes a free one")
    parser.add_argument("--host", default="127.0.0.1",
                        help="address to listen on; the default lets only this node in")
    parser.add_argument("--login", default=os.environ.get("CRYOET_SSH_LOGIN"),
                        help="the login node as your laptop reaches it, for the tunnel line: a "
                        "Host of the laptop's ~/.ssh/config, or user@host; default "
                        "$CRYOET_SSH_LOGIN, else the host Slurm took the job from")
    parser.add_argument("--cache-dir", type=Path, default=None,
                        help="where pictures are kept; default <project>/results_cache")
    parser.add_argument("--mra", type=Path, default=None,
                        help="the folder of classification runs and FSC curves; default "
                        "<project>/mra, where a lane's MRA writes them")
    parser.add_argument("--assets", type=Path, default=ASSETS,
                        help="the libraries scripts/fetch_viewer_assets.py fetched; "
                        f"default {ASSETS}")
    return parser


def main(argv: list[str] | None = None) -> int:
    parser = arguments()
    args = parser.parse_args(argv)

    if args.lane is not None:
        folder, label = lane_project(args.lane)
        variant = args.variant or label
    else:
        folder, variant = args.project, args.variant
    if not (folder / "artifacts.json").is_file():
        parser.error(f"no registry at {folder / 'artifacts.json'}{registry_hint(folder)}")
    cache = args.cache_dir or folder / "results_cache"
    project = ResultsProject(folder, cache_dir=cache, variant=variant)
    token = secrets.token_urlsafe(24)
    classification = ClassificationRuns(args.mra or folder / "mra", cache_dir=cache)
    print(f"reading {folder.resolve()} once before serving it...", flush=True)
    print(f"  {read_once(project, classification)}", flush=True)
    server = ResultsServer((args.host, args.port), project, token, assets=args.assets,
                           classification=classification)
    node = socket.gethostname()
    print(f"serving {folder.resolve()}" + (f", picks labelled {variant}" if variant else "")
          + f", on {node}, port {server.port}")
    print(f"pictures are kept in {cache.resolve()}")
    if not server.viewer_available:
        print(f"no Mol* in {args.assets}: the 3D view is off until "
              "scripts/fetch_viewer_assets.py has run")
    for line in access_lines(args.host, server.port, token, node=node, user=getpass.getuser(),
                             login=args.login, environ=os.environ):
        print(line)
    print("Ctrl-C stops it.", flush=True)
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        server.server_close()
    return 0


if __name__ == "__main__":
    sys.exit(main())
