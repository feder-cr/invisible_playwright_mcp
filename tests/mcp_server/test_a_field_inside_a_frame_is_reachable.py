"""Controls inside iframes, cross-origin ones included, are reported and reachable.

Plaid Link (2026-10-27): Square's invoice page opens it as an iframe at
https://cdn.plaid.com, and the bank's sign-in fields are inside it. The
snapshot walked the top-level document only, so it listed none of them, and
every selector tool searched the top-level document, so none could reach
them. A credential filler could not log in at all.

`frame` names an iframe by the selectors of each nested iframe, outermost
first, exactly as the snapshot now reports it on each element.
"""
from __future__ import annotations

import asyncio
import inspect
import json
import socket
import threading

import pytest

from invisible_playwright_mcp.mcp import actions, server


# --- the argument -------------------------------------------------------------

def test_no_frame_is_the_top_level_document():
    assert actions.frame_chain(None) == []
    assert actions.frame_chain([]) == []


@pytest.mark.parametrize("bad", ["#plaid", [""], ["  "], [1], ["#a"] * (actions.FRAME_DEPTH + 1), {"a": 1}])
def test_a_malformed_frame_is_refused_before_the_page_is_asked(bad):
    with pytest.raises(ValueError):
        actions.frame_chain(bad)


def test_every_selector_tool_declares_it():
    for tool in (server.browser_type, server.browser_click, server.browser_select_option,
                 server.browser_evaluate, server.browser_read_html):
        params = inspect.signature(tool).parameters
        assert "frame" in params and params["frame"].default is None, tool.__name__


class _Element:
    def __init__(self, tag, frame):
        self.tag, self.frame, self.disposed = tag, frame, False

    async def evaluate(self, expression):
        return self.tag

    async def content_frame(self):
        return self.frame

    async def dispose(self):
        self.disposed = True


class _Frame:
    def __init__(self, matches):
        self.matches = matches

    async def query_selector_all(self, selector):
        return self.matches.get(selector, [])


class _Page:
    def __init__(self, main):
        self.main_frame = main


class _Session:
    def __init__(self, page):
        self._page = page

    def page(self):
        return self._page


def test_each_step_must_be_exactly_one_iframe():
    inner = _Frame({})
    one = _Element("IFRAME", inner)
    two = [_Element("IFRAME", inner), _Element("IFRAME", inner)]
    div = _Element("DIV", None)
    blank = _Element("IFRAME", None)
    top = _Frame({"#plaid": [one], ".many": two, "#div": [div], "#blank": [blank]})
    session = _Session(_Page(top))
    assert asyncio.run(actions.frame_target(session, ["#plaid"])) is inner
    assert one.disposed
    for chain, said in ((["#none"], "matches 0"), ([".many"], "matches 2"),
                        (["#div"], "not an <iframe>"), (["#blank"], "no document")):
        with pytest.raises(RuntimeError, match=said):
            asyncio.run(actions.frame_target(session, chain))
    assert all(e.disposed for e in two) and div.disposed and blank.disposed
    assert asyncio.run(actions.frame_target(session, None)) is session._page


# --- a real browser -----------------------------------------------------------

FRAME = b"""<!doctype html><html><head><title>bank sign-in</title></head><body>
<input id="email" name="email" type="email">
<input id="password" name="password" type="password">
<input id="token" type="text">
<button id="go" type="button" onclick="document.title='clicked in frame'">Go</button>
</body></html>"""


def _parent(frame_url: str) -> bytes:
    return ("""<!doctype html><html><head><title>invoice</title></head><body>
<input id="top-password" name="top-password" type="password">
<iframe id="link" title="Link" src="%s" style="position:absolute;left:40px;top:120px;width:600px;height:400px;border:3px solid #000"></iframe>
<iframe title="hidden ad" src="%s" style="display:none"></iframe>
</body></html>""" % (frame_url, frame_url)).encode()


def _serve(body_for):
    import http.server

    class H(http.server.BaseHTTPRequestHandler):
        def do_GET(self):
            body = body_for()
            self.send_response(200)
            self.send_header("Content-Type", "text/html; charset=utf-8")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)

        def log_message(self, *a):
            pass

    s = socket.socket()
    s.bind(("127.0.0.1", 0))
    port = s.getsockname()[1]
    s.close()
    srv = http.server.HTTPServer(("127.0.0.1", port), H)
    threading.Thread(target=srv.serve_forever, daemon=True).start()
    return srv, port


@pytest.fixture(scope="module")
def run():
    """A parent on 127.0.0.1 embedding a frame on localhost: two origins."""
    from invisible_playwright_mcp.mcp.plan import plan_session
    from invisible_playwright_mcp.mcp.session import StealthSession

    frame_srv, frame_port = _serve(lambda: FRAME)
    frame_url = "http://localhost:%d/" % frame_port
    parent_srv, parent_port = _serve(lambda: _parent(frame_url))
    url = "http://127.0.0.1:%d/" % parent_port
    loop = asyncio.new_event_loop()
    session = StealthSession(**plan_session().kwargs)
    loop.run_until_complete(session.start())

    def go(fn):
        async def body():
            await actions.navigate(session, url, wait_until="load")
            return await fn(session)
        return loop.run_until_complete(body())

    go.frame_origin = "http://localhost:%d" % frame_port
    go.top_origin = "http://127.0.0.1:%d" % parent_port
    try:
        yield go
    finally:
        loop.run_until_complete(session.close())
        loop.close()
        frame_srv.shutdown()
        parent_srv.shutdown()


@pytest.mark.e2e
def test_the_snapshot_lists_controls_inside_a_cross_origin_frame(run):
    snap = run(lambda s: actions.snapshot(s))
    d = json.loads(snap)
    framed = {e.get("selector"): e for e in d["interactive_elements"] if e.get("frame")}
    for wanted in ("#email", "#password", "#go"):
        assert wanted in framed, sorted(map(str, framed))
        assert framed[wanted]["frame"] == ["#link"]
        assert framed[wanted]["frame_origin"] == run.frame_origin
    top = [e for e in d["interactive_elements"] if not e.get("frame")]
    assert [e.get("selector") for e in top] == ["#top-password"]
    # The hidden frame is not reported; the shown one is, by origin only.
    assert d["frames"] == [{"origin": run.frame_origin, "frame": ["#link"]}]
    # `at` is in the top viewport (frame box at 40+3, 120+3), `frame_at` in the
    # frame's. Within a pixel: a fingerprint's display scale snaps the border,
    # and both coordinates are rounded (measured: 567 for 525 + 43).
    go = framed["#go"]
    for axis, offset in ((0, 43), (1, 123)):
        assert abs(go["at"][axis] - (go["frame_at"][axis] + offset)) <= 1, go


@pytest.mark.e2e
def test_click_and_evaluate_reach_inside_the_frame(run):
    async def body(s):
        await actions.click(s, "#go", frame=["#link"])
        inside = json.loads(await actions.evaluate(s, "document.title", frame=["#link"]))
        outside = json.loads(await actions.evaluate(s, "document.title"))
        origin = json.loads(await actions.evaluate(s, "location.origin", frame=["#link"]))
        return inside, outside, origin
    inside, outside, origin = run(body)
    assert (inside, outside, origin) == ("clicked in frame", "invoice", run.frame_origin)


@pytest.mark.e2e
def test_typing_reaches_the_field_inside_the_frame(run):
    async def body(s):
        out = await actions.type_text(s, "#email", "a@example.com", frame=["#link"])
        inside = json.loads(await actions.evaluate(s, "document.querySelector('#email').value", frame=["#link"]))
        html = await actions.read_html(s, "form", frame=["#link"])
        return out, inside, html
    out, inside, html = run(body)
    assert out.startswith("typed into #email") and inside == "a@example.com"
    assert 'id="email"' in html
