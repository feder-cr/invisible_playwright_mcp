"""Input to one browser queues, and long text is typed to the end.

Two browser_type calls issued at once drove one page's focus and keyboard
together: the second field ended up holding "BBAAABABBABAB...", the first a
single "A" (measured 2026-09-30). And a long text typed at a person's pace -
this session's hand plans 250-380 ms a character - takes ten minutes for
2,000 characters, so the call was cut off by the client with the field half
full. Text past eighty characters was then sent through `insert_text`, which a
page sees as a composition with no key behind it: no person types like that.

So every character is a key; actions on one browser run one at a time; and a
typing that outlasts the answer goes on in the background, with actions on
that browser refused, saying how far it has got, until it ends.

The unit tests drive the SERVER'S TOOLS with a session that launches nothing,
so the behaviour is what is held, not the spelling of the source; the e2e ones
hold the outcome against a real engine.
"""
from __future__ import annotations

import asyncio
import http.server
import threading
import time

import pytest

from invisible_playwright_mcp.mcp import actions, server, work as work_module
from invisible_playwright_mcp.mcp.work import Work


class _Quiet:
    """A session that launches nothing; every action is replaced below."""

    def __init__(self, **kwargs):
        self.kwargs = kwargs
        self.closed = False

    async def start(self):
        pass

    async def close(self):
        self.closed = True

    def is_usable(self):
        return not self.closed

    def pages(self):
        return ["about:blank"]


@pytest.fixture
def spans(monkeypatch):
    """Every action records when it ran; each takes a little while."""
    w = Work("default", factory=_Quiet)
    w._open["main"] = _Quiet()
    w._launched["main"] = {"seed": 1}
    monkeypatch.setattr(server, "work", w)
    ran = []

    def recorder(name, result="done"):
        async def act(session, *args, **kwargs):
            start = time.monotonic()
            await asyncio.sleep(0.2)
            ran.append((name, start, time.monotonic()))
            return result if result != "png" else b"png"
        return act

    for name in ("click", "type_text", "press_key", "select_option", "navigate",
                 "click_at", "snapshot", "read_text", "read_html", "evaluate"):
        monkeypatch.setattr(actions, name, recorder(name, "png" if name == "click_at" else "done"))
    monkeypatch.setattr(actions, "screenshot_png", recorder("screenshot_png", "png"))
    return ran


def _overlap(ran, a, b):
    (_, a0, a1), = [r for r in ran if r[0] == a]
    (_, b0, b1), = [r for r in ran if r[0] == b]
    return a0 < b1 and b0 < a1


DRIVING = {
    "click": lambda: server.browser_click("#b"),
    "type_text": lambda: server.browser_type("#t", "x"),
    "press_key": lambda: server.browser_press_key("Enter"),
    "select_option": lambda: server.browser_select_option("#s", "a"),
    "navigate": lambda: server.browser_navigate("http://127.0.0.1/"),
    "click_at": lambda: server.browser_click_at(1, 1),
}


@pytest.mark.parametrize("first,second", [
    ("type_text", "type_text"), ("type_text", "click"), ("click", "press_key"),
    ("select_option", "type_text"), ("navigate", "click_at"),
])
async def test_tools_that_drive_one_browser_run_one_at_a_time(spans, first, second):
    """Known-bad: route any of these through `work.reading`, or drop the lock
    from `acting` - the two then overlap, which on a real page is two hands on
    one keyboard."""
    await asyncio.gather(DRIVING[first](), DRIVING[second]())
    names = [r[0] for r in spans]
    if first == second:
        (_, a0, a1), (_, b0, b1) = sorted(spans, key=lambda r: r[1])
        assert b0 >= a1, "two %s calls ran on one browser at once" % first
    else:
        assert sorted(names) == sorted([first, second])
        assert not _overlap(spans, first, second), "%s and %s overlapped" % (first, second)


@pytest.mark.parametrize("reader", [
    lambda: server.browser_snapshot(), lambda: server.browser_read_text(),
    lambda: server.browser_read_html(),
])
async def test_reads_do_not_wait_behind_input(spans, reader):
    await asyncio.gather(server.browser_type("#t", "x"), reader())
    other = next(r[0] for r in spans if r[0] != "type_text")
    assert _overlap(spans, "type_text", other), "%s waited behind typing" % other


async def test_a_script_waits_behind_input_like_an_action(spans):
    """Known-bad: `browser_evaluate` went through `work.reading`, so a script
    calling `focus()` ran in the middle of a typing and the rest of the text
    landed in the field it focused."""
    await asyncio.gather(server.browser_type("#t", "x"), server.browser_evaluate("1"))
    assert not _overlap(spans, "type_text", "evaluate"), "a script ran in the middle of a typing"


async def test_a_script_is_refused_while_a_long_typing_goes_on(slow_typing):
    """A script during a background typing is refused with how far it has got,
    as any action is; a snapshot is still answered."""
    first = await server.browser_type("#t", "x" * 50)
    assert "still typing" in first
    with pytest.raises(RuntimeError, match="still typing"):
        await server.browser_evaluate("document.activeElement.id")
    assert await server.browser_snapshot() == "done"
    slow_typing.set()


@pytest.fixture
def slow_typing(spans, monkeypatch):
    """A typing that takes longer than the answer may wait."""
    monkeypatch.setattr(work_module, "ANSWER_WITHIN_S", 0.2)
    gate = asyncio.Event()

    async def type_text(session, selector, text):
        await gate.wait()
        return "typed into %s" % selector

    monkeypatch.setattr(actions, "type_text", type_text)
    return gate


async def test_a_typing_that_outlasts_its_answer_goes_on_and_says_so(slow_typing):
    said = await server.browser_type("#msg", "y" * 2000)
    assert said.startswith("still typing into #msg: 2000 characters"), said

    # Actions are refused, not queued: a click left waiting would outlive its
    # own call. Reads still answer.
    with pytest.raises(RuntimeError, match="still typing into #msg"):
        await server.browser_click("#send")
    with pytest.raises(RuntimeError, match="still typing"):
        await server.browser_type("#other", "z")
    assert await server.browser_snapshot() == "done"
    assert "still going on in the background, into #msg" in await server.work.status("main")

    slow_typing.set()
    await asyncio.sleep(0.05)
    after = await server.browser_click("#send")
    assert after == ("(the typing that went on in the background has finished: "
                     "typed into #msg)\ndone"), after
    # Said once.
    assert await server.browser_click("#send") == "done"


async def test_a_background_typing_that_fails_is_said_by_the_next_action(spans, monkeypatch):
    monkeypatch.setattr(work_module, "ANSWER_WITHIN_S", 0.05)

    async def type_text(session, selector, text):
        await asyncio.sleep(0.15)
        raise RuntimeError("the page went away")

    monkeypatch.setattr(actions, "type_text", type_text)
    assert (await server.browser_type("#msg", "y" * 500)).startswith("still typing")
    await asyncio.sleep(0.3)
    after = await server.browser_press_key("Enter")
    assert after.startswith("(the typing into #msg that went on in the background "
                            "stopped: the page went away)"), after


async def test_closing_the_browser_stops_a_background_typing(slow_typing):
    await server.browser_type("#msg", "y" * 2000)
    rec = server.work._typing["main"]
    await server.browser_close()
    await asyncio.sleep(0)
    assert rec.task.cancelled() or rec.task.done()
    assert "main" not in server.work._typing


async def test_a_typing_that_finishes_in_time_answers_as_usual(spans):
    assert await server.browser_type("#t", "x") == "done"
    assert server.work._typing == {}


# --- against a real engine -------------------------------------------------

PAGE = b"""<!doctype html><html><body>
<textarea id="comments" maxlength="2000"></textarea>
<input id="first"><input id="second">
<script>
window.__events = [];
for (const k of ['keydown', 'compositionstart', 'beforeinput'])
  document.addEventListener(k, e => __events.push([k, e.inputType || e.key || '']), true);
</script>
</body></html>"""


@pytest.fixture(scope="module")
def url():
    class H(http.server.BaseHTTPRequestHandler):
        def do_GET(self):
            self.send_response(200)
            self.send_header("Content-Type", "text/html; charset=utf-8")
            self.end_headers()
            self.wfile.write(PAGE)

        def log_message(self, *a):
            pass

    srv = http.server.HTTPServer(("127.0.0.1", 0), H)
    threading.Thread(target=srv.serve_forever, daemon=True).start()
    yield "http://127.0.0.1:%d/" % srv.server_port
    srv.shutdown()


async def _with_work(url, body):
    from invisible_playwright_mcp.mcp.plan import plan_session
    from invisible_playwright_mcp.mcp.session import StealthSession
    session = StealthSession(**plan_session().kwargs)
    await session.start()
    try:
        await actions.navigate(session, url)
        w = Work("default")
        w._open["main"] = session
        w._launched["main"] = {}
        return await body(w, session)
    finally:
        await session.close()


@pytest.mark.e2e
def test_two_concurrent_types_each_land_whole(url):
    async def body(w, session):
        await asyncio.wait_for(asyncio.gather(
            w.typing(actions.type_text, "#first", "AAAAAAAAAAAA"),
            w.typing(actions.type_text, "#second", "BBBBBBBBBBBB")), 120)
        page = session.page()
        return (await page.locator("#first").input_value(),
                await page.locator("#second").input_value())

    assert asyncio.run(_with_work(url, body)) == ("AAAAAAAAAAAA", "BBBBBBBBBBBB")


@pytest.mark.e2e
def test_a_long_text_is_typed_key_by_key_to_the_end_in_the_background(url, monkeypatch):
    """The answer is due after two seconds here instead of forty-five, so the
    background path runs on a text that takes a few dozen seconds."""
    monkeypatch.setattr(work_module, "ANSWER_WITHIN_S", 2.0)
    text = ("A message somebody writes in a box, line by line.\n" * 3)[:120]

    async def body(w, session):
        said = await w.typing(actions.type_text, "#comments", text)
        refused = None
        try:
            await w.acting(actions.click, "#first")
        except RuntimeError as exc:
            refused = str(exc)
        while not w._typing["main"].task.done():
            await asyncio.sleep(0.5)
        news = await w.acting(actions.click, "#first")
        page = session.page()
        return (said, refused, news, await page.locator("#comments").input_value(),
                await page.evaluate("__events"))

    said, refused, news, value, events = asyncio.run(_with_work(url, body))
    assert said.startswith("still typing into #comments: 120 characters"), said
    assert refused and "still typing into #comments" in refused and "of 120" in refused, refused
    assert news.startswith("(the typing that went on in the background has finished: "
                           "typed into #comments)"), news
    assert value == text
    kinds = [k for k, _ in events]
    assert kinds.count("keydown") == len(text), "not every character was a key"
    assert "compositionstart" not in kinds, "text went in as a composition"
    assert {t for k, t in events if k == "beforeinput"} <= {"insertText", "insertLineBreak"}


async def test_a_typing_whose_call_the_client_abandoned_is_still_seen(slow_typing, monkeypatch):
    """A client that gives up on the call does not stop the typing it started,
    so the next action must still see it. Known-bad: record the typing only
    when the answer is sent, which a cancelled call never reaches."""
    monkeypatch.setattr(work_module, "ANSWER_WITHIN_S", 30)
    call = asyncio.create_task(server.browser_type("#msg", "y" * 2000))
    await asyncio.sleep(0.05)
    call.cancel()
    with pytest.raises(asyncio.CancelledError):
        await call
    with pytest.raises(RuntimeError, match="still typing into #msg"):
        await server.browser_click("#send")
    slow_typing.set()
    await asyncio.sleep(0.05)
    assert (await server.browser_click("#send")).startswith(
        "(the typing that went on in the background has finished: typed into #msg)")
