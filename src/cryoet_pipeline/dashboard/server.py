"""A small web server for the result pages, run on the cluster next to the data.

Standard library only. It serves the page (`static/`), the JSON documents and
pictures of `data.ResultsProject`, and nothing else: no request writes to the
project.

A port opened on a compute node can be reached by every user of the cluster, and
the data are unpublished. So every request needs a token: the link the server
prints carries it once, the server answers with a cookie and a redirect to the
same page without it, and the cookie carries it from then on. The cookie's name
holds the port, since a browser keeps cookies per host, not per port, and two
servers on one laptop's localhost would otherwise overwrite each other's. By
default the server listens on 127.0.0.1 only.
"""

from __future__ import annotations

import hmac
import json
import sys
import traceback
from http import HTTPStatus
from http.cookies import SimpleCookie
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Any
from urllib.parse import parse_qs, unquote, urlsplit

from cryoet_pipeline.dashboard.data import ResultsProject

STATIC = Path(__file__).with_name("static")
STATIC_FILES = {
    "app.js": "text/javascript; charset=utf-8",
    "style.css": "text/css; charset=utf-8",
}


class ResultsServer(ThreadingHTTPServer):
    daemon_threads = True

    def __init__(self, address: tuple[str, int], project: ResultsProject, token: str) -> None:
        super().__init__(address, ResultsHandler)
        self.project = project
        self.token = token

    @property
    def port(self) -> int:
        return int(self.server_address[1])

    @property
    def cookie_name(self) -> str:
        return f"cryoet_results_{self.port}"


class ResultsHandler(BaseHTTPRequestHandler):
    server: ResultsServer
    server_version = "cryoet-results"

    def log_message(self, format: str, *args: Any) -> None:  # noqa: A002
        sys.stderr.write(f"{self.address_string()} {format % args}\n")

    def _cookie_token(self) -> str | None:
        header = self.headers.get("Cookie")
        if not header:
            return None
        cookie = SimpleCookie()
        try:
            cookie.load(header)
        except Exception:
            return None
        morsel = cookie.get(self.server.cookie_name)
        return morsel.value if morsel is not None else None

    def _matches(self, value: str | None) -> bool:
        return value is not None and hmac.compare_digest(value, self.server.token)

    def do_GET(self) -> None:  # noqa: N802
        url = urlsplit(self.path)
        query = parse_qs(url.query)
        if "token" in query:
            if not self._matches(query["token"][0]):
                self._text(HTTPStatus.FORBIDDEN, "wrong token\n")
                return
            self.send_response(HTTPStatus.SEE_OTHER)
            self.send_header("Location", url.path or "/")
            self.send_header(
                "Set-Cookie",
                f"{self.server.cookie_name}={self.server.token}; Path=/; HttpOnly; "
                "SameSite=Strict",
            )
            self.send_header("Content-Length", "0")
            self.end_headers()
            return
        if not self._matches(self._cookie_token()):
            self._text(HTTPStatus.FORBIDDEN,
                       "open the link the server printed, with its token\n")
            return
        try:
            self._route(url.path)
        except KeyError as exc:
            self._json(HTTPStatus.NOT_FOUND, {"error": str(exc.args[0] if exc.args else exc)})
        except Exception as exc:
            traceback.print_exc()
            self._json(HTTPStatus.INTERNAL_SERVER_ERROR, {"error": f"{type(exc).__name__}: {exc}"})

    def _route(self, path: str) -> None:
        project = self.server.project
        parts = [unquote(part) for part in path.split("/") if part]
        if not parts:
            self._file(STATIC / "index.html", "text/html; charset=utf-8")
        elif parts[0] == "static" and len(parts) == 2 and parts[1] in STATIC_FILES:
            self._file(STATIC / parts[1], STATIC_FILES[parts[1]])
        elif parts == ["api", "project"]:
            self._json(HTTPStatus.OK, project.overview())
        elif parts[0] == "api" and len(parts) == 3 and parts[1] == "series":
            self._json(HTTPStatus.OK, project.series(parts[2]))
        elif parts[0] == "img" and len(parts) == 4:
            body = project.picture(parts[1], parts[2], parts[3])
            # The key is in the path, so a picture at a path never changes.
            self._bytes(HTTPStatus.OK, body, "image/png",
                        cache="private, max-age=31536000, immutable")
        else:
            raise KeyError(f"nothing at {path}")

    def _file(self, path: Path, content_type: str) -> None:
        self._bytes(HTTPStatus.OK, path.read_bytes(), content_type, cache="no-store")

    def _json(self, status: HTTPStatus, document: Any) -> None:
        self._bytes(status, json.dumps(document).encode(), "application/json", cache="no-store")

    def _text(self, status: HTTPStatus, text: str) -> None:
        self._bytes(status, text.encode(), "text/plain; charset=utf-8", cache="no-store")

    def _bytes(self, status: HTTPStatus, body: bytes, content_type: str, *, cache: str) -> None:
        self.send_response(status)
        self.send_header("Content-Type", content_type)
        self.send_header("Content-Length", str(len(body)))
        self.send_header("Cache-Control", cache)
        self.send_header("X-Content-Type-Options", "nosniff")
        self.end_headers()
        self.wfile.write(body)


def tunnel_lines(host: str, port: int, node: str, login: str | None) -> list[str]:
    """What to type on a laptop to reach a server on a cluster node, by how it listens.

    Listening on 127.0.0.1, only the node itself reaches the port, so the tunnel
    has to end on the node: through the login node with -J, which a site allows
    when it lets a user ssh to a node running their job. Listening on every
    address, the login node can forward to the node's port directly.
    """

    login_name = login or "<login node>"
    if host in {"127.0.0.1", "localhost", "::1"}:
        return [f"ssh -N -L {port}:localhost:{port} -J {login_name} {node}"]
    return [f"ssh -N -L {port}:{node}:{port} {login_name}"]
