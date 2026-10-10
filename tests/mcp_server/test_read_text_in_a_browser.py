"""`browser_read_text` reads what the engine's selector finds, in a browser.

⛔ WHY THIS FILE EXISTS. `read_text` resolved its selector with
`document.querySelector` while every other tool resolved it through the engine,
so the `:nth-match(...)` handles `browser_snapshot` builds to tell identical
elements apart failed here while the same string clicked, and a field inside a
shadow root read as missing. A fake page cannot see that: the defect is in which
resolver runs, and only a real document has both.

The claims, each with its own test:

  * a snapshot's `:nth-match(...)` handle reads the element it names, not the
    first of its kind;
  * an element inside an open shadow root is found;
  * plain CSS still reads, and a selector that matches nothing still says so.
"""
from __future__ import annotations

import pytest

from invisible_playwright_mcp.mcp import actions

PAGE = b"""<!doctype html>
<html><head><title>reading text</title></head><body>
  <ul>
    <li class="item">FIRST-ITEM</li>
    <li class="item">SECOND-ITEM</li>
    <li class="item">THIRD-ITEM</li>
  </ul>
  <p id="prose">PLAIN-PROSE</p>
  <div id="host"></div>
  <script>
    const root = document.getElementById('host').attachShadow({mode: 'open'});
    root.innerHTML = '<p class="inside">INSIDE-A-SHADOW-ROOT</p>';
  </script>
</body></html>"""


def _serve():
    import http.server
    import socket
    import threading

    class H(http.server.BaseHTTPRequestHandler):
        def do_GET(self):
            self.send_response(200)
            self.send_header("Content-Type", "text/html; charset=utf-8")
            self.send_header("Content-Length", str(len(PAGE)))
            self.end_headers()
            self.wfile.write(PAGE)

        def log_message(self, *a):
            pass

    s = socket.socket()
    s.bind(("127.0.0.1", 0))
    port = s.getsockname()[1]
    s.close()
    srv = http.server.HTTPServer(("127.0.0.1", port), H)
    threading.Thread(target=srv.serve_forever, daemon=True).start()
    return srv, "http://127.0.0.1:%d/" % port


SELECTORS = (":nth-match(li.item, 2)", "p.inside", "#prose", "#nothing-here")


@pytest.fixture(scope="module")
def read():
    """One browser for the whole file: reading is a pure observation."""
    import asyncio

    from invisible_playwright_mcp.mcp.plan import plan_session
    from invisible_playwright_mcp.mcp.session import StealthSession

    srv, url = _serve()

    async def run():
        # Through plan_session, as production builds a session (see
        # test_read_html_in_a_browser for what a bare StealthSession costs).
        session = StealthSession(**plan_session().kwargs)
        try:
            await session.start()
            await actions.navigate(session, url)
            out = {}
            for selector in SELECTORS:
                # Each read on its own: a resolver that throws on one selector
                # must fail that selector's test, not the whole file.
                try:
                    out[selector] = await actions.read_text(session, selector)
                except Exception as exc:  # noqa: BLE001 - the test reads it
                    out[selector] = "raised %s: %s" % (type(exc).__name__, exc)
            return out
        finally:
            await session.close()

    try:
        yield asyncio.run(run())
    finally:
        srv.shutdown()


@pytest.mark.e2e
def test_a_snapshot_handle_reads_the_element_it_names(read):
    """Known-bad: resolve with document.querySelector - `:nth-match` is not CSS,
    and the read fails instead of answering SECOND-ITEM."""
    assert read[":nth-match(li.item, 2)"] == "SECOND-ITEM"


@pytest.mark.e2e
def test_an_element_inside_a_shadow_root_is_found(read):
    """Known-bad: document.querySelector stops at the shadow boundary and
    answers that nothing matches."""
    assert read["p.inside"] == "INSIDE-A-SHADOW-ROOT"


@pytest.mark.e2e
def test_plain_css_reads_and_a_miss_still_says_so(read):
    assert read["#prose"] == "PLAIN-PROSE"
    assert "no element matches" in read["#nothing-here"]
