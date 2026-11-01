"""`browser_set_viewport`: a responsive check needs a 390-wide page.

Before it, nothing reached the window size: no tool took one, a browser-chrome
zoom shortcut does not reach Firefox, and a sized iframe is refused by any site
that sends X-Frame-Options (measured 2026-11-01, from an IMPI Foods Manager
session that needed 1440, 1024 and 390-wide screenshots).

The unit tests drive `actions.set_viewport` against a page that behaves like
the engine's: `set_viewport_size` changes what the page reads. The e2e test at
the bottom proves it in a real browser, through owner-mode dispatch.
"""
from __future__ import annotations

import json

import pytest

from invisible_playwright_mcp.mcp import actions

SCREEN = (1920, 1080)
TOOLBAR = 85


class _Page:
    def __init__(self, width=1920, height=947, *, honour=True):
        self._size = {"width": width, "height": height}
        self.honour = honour
        self.calls = []

    @property
    def viewport_size(self):
        return dict(self._size)

    async def set_viewport_size(self, size):
        self.calls.append(dict(size))
        if self.honour:
            self._size = dict(size)

    async def evaluate(self, script):
        assert script == actions._WINDOW_METRICS_JS
        w, h = self._size["width"], self._size["height"]
        return {"inner": [w, h], "outer": [w, h + TOOLBAR], "screen": list(SCREEN),
                "avail": list(SCREEN), "dpr": 1}


class _Session:
    def __init__(self, page):
        self._page = page
        self.own_viewport = None

    def page(self):
        return self._page


async def test_a_size_is_set_and_read_back_from_the_page():
    page = _Page()
    answer = json.loads(await actions.set_viewport(_Session(page), 390, 844))
    assert page.calls == [{"width": 390, "height": 844}]
    assert answer["viewport"] == {"width": 390, "height": 844}
    assert answer["window"] == {"width": 390, "height": 844 + TOOLBAR}
    assert answer["screen"] == {"width": 1920, "height": 1080}
    assert answer["own"] == {"width": 1920, "height": 947}
    assert "warning" not in answer


async def test_neither_size_restores_the_one_the_browser_opened_with():
    """Known-bad: remember the size at every call, and the second resize makes
    390x844 the browser's "own" size."""
    page = _Page()
    session = _Session(page)
    await actions.set_viewport(session, 1024, 768)
    await actions.set_viewport(session, 390, 844)
    answer = json.loads(await actions.set_viewport(session))
    assert page.calls[-1] == {"width": 1920, "height": 947}
    assert answer["viewport"] == {"width": 1920, "height": 947}


async def test_restoring_before_any_change_keeps_the_size():
    page = _Page()
    answer = json.loads(await actions.set_viewport(_Session(page)))
    assert page.calls == [{"width": 1920, "height": 947}]
    assert answer["viewport"] == answer["own"]


@pytest.mark.parametrize("width,height", [(2560, 900), (1440, 1440)])
async def test_larger_than_the_screen_is_refused_and_changes_nothing(width, height):
    page = _Page()
    with pytest.raises(ValueError, match="larger than this browser's own screen"):
        await actions.set_viewport(_Session(page), width, height)
    assert page.calls == []


@pytest.mark.parametrize("width,height,match", [
    (390, None, "both width and height"),
    (None, 844, "both width and height"),
    (100, 844, "at least"),
    (390, 10, "at least"),
    (390.5, 844, "whole number"),
    (True, 844, "whole number"),
    ("390", 844, "whole number"),
])
async def test_a_bad_size_is_refused_before_anything_is_sent(width, height, match):
    page = _Page()
    with pytest.raises(ValueError, match=match):
        await actions.set_viewport(_Session(page), width, height)
    assert page.calls == []


async def test_a_size_the_page_did_not_take_is_said():
    """The answer reports what the page reads, not what was asked: a window
    manager or Firefox's minimum width can keep a window wider than asked."""
    page = _Page(honour=False)
    answer = json.loads(await actions.set_viewport(_Session(page), 390, 844))
    assert answer["viewport"] == {"width": 1920, "height": 947}
    assert "not the 390x844 asked for" in answer["warning"]


# --- in a real browser ------------------------------------------------------

@pytest.mark.e2e
async def test_a_real_page_reads_the_size_and_keeps_its_screen():
    import http.server
    import threading

    from invisible_playwright_mcp.mcp.plan import plan_session
    from invisible_playwright_mcp.mcp.session import StealthSession

    body = b"<!doctype html><title>viewport</title><p>hello</p>"

    class H(http.server.BaseHTTPRequestHandler):
        def do_GET(self):
            self.send_response(200)
            self.send_header("Content-Type", "text/html; charset=utf-8")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)

        def log_message(self, *a):
            pass

    srv = http.server.HTTPServer(("127.0.0.1", 0), H)
    threading.Thread(target=srv.serve_forever, daemon=True).start()
    session = StealthSession(**plan_session().kwargs)
    try:
        await session.start()
        await actions.navigate(session, "http://127.0.0.1:%d/" % srv.server_port)
        before = await actions.window_metrics(session.page())
        for width, height in ((1440, 900), (1024, 768), (390, 844)):
            if width > before["screen"][0] or height > before["screen"][1]:
                continue
            answer = json.loads(await actions.set_viewport(session, width, height))
            assert answer["viewport"] == {"width": width, "height": height}, answer
            now = await actions.window_metrics(session.page())
            assert now["inner"] == [width, height], now
            # The identity is untouched: same screen, same DPR.
            assert now["screen"] == before["screen"] and now["dpr"] == before["dpr"], now
            # ⛔ THE DOCUMENTED TRADE-OFF, PINNED: the engine declares the
            # window maximized, so it does not follow the content area. If an
            # engine change makes it follow, this fails on purpose: update the
            # fingerprint paragraph in docs/mcp-server.md and set_viewport's
            # answer with it.
            assert now["outer"] == before["outer"], now
        restored = json.loads(await actions.set_viewport(session))
        assert restored["viewport"] == restored["own"]
        assert (await actions.window_metrics(session.page()))["inner"] == before["inner"]
        with pytest.raises(ValueError, match="larger than"):
            await actions.set_viewport(session, before["screen"][0] + 1, 600)
    finally:
        await session.close()
        srv.shutdown()
