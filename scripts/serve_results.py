#!/usr/bin/env python3
"""A project's results in a browser, served from where the data are.

    python3 scripts/serve_results.py --project outputs/ryr1-bin8
    python3 scripts/serve_results.py --project outputs/ryr1-bin8 --variant bin8 --port 8765

Run it on the cluster, in a slogin job or a small job of its own, and open the
link it prints from a laptop through the SSH tunnel it also prints. The page
shows each series' tomogram as slabs on a slider, through z and through y, with
the picks drawn on them, a score histogram with a threshold, and each pick's
views turned into the template's frame. For the classification runs in
<project>/mra (or --mra): every iteration's classes, their sizes, each map's z
sections and central views, the maps in Mol*, and the FSC between half runs.

Run it from the clone's root, like every command: the registry names files from
there. Read-only: it reads the registry, never locks it, and writes only its cache of
pictures (by default <project>/results_cache). The first look at a series cuts
its slabs, a few seconds for a bin8 tomogram; later looks read the cache, until
the tomogram or the picks are newer than it.

A prototype of step 1 of docs/dashboard.md.
"""

from __future__ import annotations

import argparse
import os
import secrets
import socket
import sys
from pathlib import Path

from cryoet_pipeline.dashboard.data import ResultsProject
from cryoet_pipeline.dashboard.mra import ClassificationRuns
from cryoet_pipeline.dashboard.server import ResultsServer, tunnel_lines

ASSETS = Path(__file__).resolve().parents[1] / "vendor"


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--project", type=Path, required=True,
                        help="the project folder, holding artifacts.json and manifests/")
    parser.add_argument("--variant", default=None,
                        help="label of the picks to show, as for cryoet report --variant")
    parser.add_argument("--port", type=int, default=0,
                        help="port to listen on; 0 (the default) takes a free one")
    parser.add_argument("--host", default="127.0.0.1",
                        help="address to listen on; the default lets only this node in")
    parser.add_argument("--cache-dir", type=Path, default=None,
                        help="where pictures are kept; default <project>/results_cache")
    parser.add_argument("--mra", type=Path, default=None,
                        help="the folder of classification runs and FSC curves; default "
                        "<project>/mra, where a lane's MRA writes them")
    parser.add_argument("--assets", type=Path, default=ASSETS,
                        help="the libraries scripts/fetch_viewer_assets.py fetched; "
                        f"default {ASSETS}")
    args = parser.parse_args(argv)

    if not (args.project / "artifacts.json").is_file():
        parser.error(f"no registry at {args.project / 'artifacts.json'}")
    cache = args.cache_dir or args.project / "results_cache"
    project = ResultsProject(args.project, cache_dir=cache, variant=args.variant)
    token = secrets.token_urlsafe(24)
    classification = ClassificationRuns(args.mra or args.project / "mra", cache_dir=cache)
    server = ResultsServer((args.host, args.port), project, token, assets=args.assets,
                           classification=classification)
    node = socket.gethostname()
    login = os.environ.get("SLURM_SUBMIT_HOST")
    print(f"serving {args.project.resolve()} on {node}, port {server.port}")
    print(f"pictures are kept in {cache.resolve()}")
    if not server.viewer_available:
        print(f"no Mol* in {args.assets}: the 3D view is off until "
              "scripts/fetch_viewer_assets.py has run")
    if login is not None or os.environ.get("SLURM_JOB_ID"):
        print("on your laptop, open a tunnel first:")
        for line in tunnel_lines(args.host, server.port, node, login):
            print(f"  {line}")
        print("then open:")
    else:
        print("open:")
    print(f"  http://localhost:{server.port}/?token={token}")
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
