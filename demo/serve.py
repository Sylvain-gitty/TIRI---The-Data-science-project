#!/usr/bin/env python3
"""A static server for the TIRI prototype that cannot serve you a stale file.

WHY THIS EXISTS. `python -m http.server` sends `Last-Modified` and nothing else, so
a browser may serve a heuristically-fresh copy without revalidating. Editing a
stylesheet and reloading then shows you the OLD one — silently, no error, no visual
cue. That cost real time repeatedly while building this: a font fix measured as
still-broken for three reloads, and a JS change looked like it had done nothing when
it had simply not been fetched.

TWO MECHANISMS, BECAUSE ONE IS NOT ENOUGH.

  (1) `Cache-Control: no-store` on every response. Necessary, and on its own it
      failed: an entry cached earlier under permissive headers survived a stop, a
      restart, a reload and a brand-new tab, because `no-store` on a response the
      browser never requests cannot evict anything.

  (2) So the HTML is rewritten in flight: `src="app.js"` becomes
      `src="app.js?v=<mtime>"`. A changed file is a changed URL and therefore cannot
      come from cache; an unchanged file keeps its URL and still can. The files on
      disk stay plain, which matters — both pages open straight off the filesystem
      over `file://` with no server at all, and a permanent `?v=` in the markup
      would be a lie there.

    python3 serve.py [port]        # default 8090

Nothing here is required: both pages open straight off the filesystem over `file://`
with no server at all. This exists only so that EDITING them is not a guessing game.
"""
import io
import os
import re
import sys
from functools import partial
from http.server import SimpleHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path


# every local href/src worth stamping: the shared modules and each page's own pair
SUBRESOURCE = re.compile(rb'((?:src|href)=")((?!https?:|//|data:)[^"?#]+\.(?:js|css))(")')


class NoCacheHandler(SimpleHTTPRequestHandler):
    """SimpleHTTPRequestHandler, minus the caching, plus mtime-stamped subresources."""

    def send_head(self):
        """Serve HTML through the rewriter; everything else unchanged.

        Overriding `send_head` rather than `do_GET` keeps HEAD correct for free and
        leaves directory listings, 404s and range handling to the parent.
        """
        path = self.translate_path(self.path)
        if not path.endswith(".html") or not os.path.isfile(path):
            return super().send_head()
        try:
            body = self._stamp(Path(path).read_bytes(), Path(path).parent)
        except OSError:
            return super().send_head()
        self.send_response(200)
        self.send_header("Content-Type", "text/html; charset=utf-8")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        return io.BytesIO(body)

    @staticmethod
    def _stamp(body: bytes, base: Path) -> bytes:
        """Append `?v=<mtime>` to every local .js/.css reference in the markup."""

        def one(m: "re.Match[bytes]") -> bytes:
            rel = m.group(2).decode()
            target = (base / rel).resolve()
            try:
                stamp = str(int(target.stat().st_mtime)).encode()
            except OSError:
                # a reference to a file that is not there: leave it exactly as
                # written, so the 404 in the console still names the real path
                return m.group(0)
            return m.group(1) + m.group(2) + b"?v=" + stamp + m.group(3)

        return SUBRESOURCE.sub(one, body)

    def end_headers(self):
        # no-store is the strong form: do not write it to disk at all, so a
        # back/forward navigation cannot resurrect it either.
        self.send_header("Cache-Control", "no-store, must-revalidate")
        self.send_header("Pragma", "no-cache")
        self.send_header("Expires", "0")
        super().end_headers()

    def log_message(self, fmt, *args):
        # one line per request is noise when a page pulls nine files; errors still
        # surface through the default error handling.
        if not str(args[1]).startswith("2"):
            super().log_message(fmt, *args)


def main() -> None:
    port = int(sys.argv[1]) if len(sys.argv) > 1 else 8090
    # ITS OWN DIRECTORY, not a subdirectory of it. This file used to live one level up
    # and serve `./tiri`; it moved in so that the folder is self-contained — everything
    # needed to run the prototype is beside this script and nothing above it.
    root = Path(__file__).resolve().parent
    handler = partial(NoCacheHandler, directory=str(root))
    with ThreadingHTTPServer(("127.0.0.1", port), handler) as httpd:
        print(f"TIRI prototype on http://127.0.0.1:{port}/landing.html  (no cache)")
        try:
            httpd.serve_forever()
        except KeyboardInterrupt:
            pass


if __name__ == "__main__":
    main()
