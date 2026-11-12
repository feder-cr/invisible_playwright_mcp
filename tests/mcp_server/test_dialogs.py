"""Native dialogs must be answered even when their reporting fails."""
from __future__ import annotations

import asyncio
from collections import defaultdict
import http.server
import os
import sys
import threading

import pytest
from invisible_playwright.async_api import TargetClosedError
from mcp.types import ImageContent

from invisible_playwright_mcp.mcp import actions, server
from invisible_playwright_mcp.mcp import session as session_module
from invisible_playwright_mcp.mcp.session import StealthSession
from invisible_playwright_mcp.mcp.work import Work


class Page:
    url = "https://host/path"

    def __init__(self):
        self.listeners = defaultdict(list)

    def on(self, name, handler):
        self.listeners[name].append(handler)

    def is_closed(self):
        return False

    async def title(self):
        return "Dialogs"


class Context:
    def __init__(self):
        self.pages = [Page()]
        self.listeners = defaultdict(list)

    def on(self, name, handler):
        self.listeners[name].append(handler)

    def remove_listener(self, name, handler):
        self.listeners[name].remove(handler)

    def emit(self, name, value):
        for handler in self.listeners[name]:
            handler(value)

    async def close(self):
        pass


class Dialog:
    def __init__(self, kind="confirm", message="Move these items?", *, error=None, gate=None):
        self.type = kind
        self.message = message
        self.default_value = "original"
        self.page = Page()
        self.calls = []
        self.error = error
        self.gate = gate

    async def _finish(self):
        if self.gate is not None:
            await self.gate.wait()
        if self.error is not None:
            raise self.error

    # The PUBLIC async_api signature (`prompt_text`), not the impl's `promptText`:
    # a stub copying the impl name hid a live TypeError on every prompt.
    async def accept(self, prompt_text=None):
        self.calls.append(("accept", prompt_text))
        await self._finish()

    async def dismiss(self):
        self.calls.append(("dismiss", None))
        await self._finish()


@pytest.fixture
async def opened(monkeypatch):
    class Engine:
        def __init__(self, **kwargs):
            pass

        async def __aenter__(self):
            return Context()

        async def __aexit__(self, *args):
            pass

    monkeypatch.setattr(session_module, "InvisiblePlaywright", Engine)
    session = StealthSession()
    await session.start()
    work = Work("dialogs")
    work._open["main"] = session
    work._launched["main"] = {"seed": 4242}
    monkeypatch.setattr(server, "work", work)
    yield work, session, session._context
    await session.close()


def clicked(monkeypatch, context, dialog):
    async def click(session, selector, **kwargs):
        # Match delivery on the loop after the action's reply.
        asyncio.get_running_loop().call_soon(context.emit, "dialog", dialog)
        return "clicked " + selector

    monkeypatch.setattr(actions, "click", click)


@pytest.mark.parametrize("kind,action", [
    ("alert", "dismissed"), ("confirm", "dismissed"),
    ("prompt", "dismissed"), ("beforeunload", "accepted"),
])
async def test_default_answer_is_reported_in_the_acting_result(opened, monkeypatch, kind, action):
    work, session, context = opened
    dialog = Dialog(kind)
    clicked(monkeypatch, context, dialog)
    said = await server.browser_click("#go")
    assert 'dialog: %s "Move these items?" on https://host/path - %s' % (kind, action) in said
    assert "default; nothing was armed" in said
    assert "clicked #go" in said
    assert dialog.calls == [("accept" if action == "accepted" else "dismiss", None)]
    assert session._dialogs[-1]["succeeded"] is True
    assert session.take_dialog_news() == []
    if action == "dismissed":
        assert "browser_handle_dialog(accept=true) and repeat the action" in said


async def test_one_context_listener_catches_a_site_opened_page(opened):
    work, session, context = opened
    assert len(context.listeners["dialog"]) == 1
    popup = Page()
    context.pages.append(popup)
    context.emit("page", popup)
    assert not popup.listeners["dialog"]
    dialog = Dialog()
    dialog.page = popup
    context.emit("dialog", dialog)
    await session.settle_dialogs()
    assert dialog.calls == [("dismiss", None)]


async def test_accept_is_one_shot_and_a_second_dialog_defaults(opened, monkeypatch):
    work, session, context = opened
    assert "Next dialog armed to accept" in await server.browser_handle_dialog(True)
    first, second = Dialog(), Dialog()
    clicked(monkeypatch, context, first)
    said = await server.browser_click("#go")
    assert "accepted (armed by browser_handle_dialog)" in said
    assert session.dialog_arm() is None
    clicked(monkeypatch, context, second)
    assert "dismissed (default" in await server.browser_click("#go")
    assert first.calls == [("accept", None)]
    assert second.calls == [("dismiss", None)]


@pytest.mark.parametrize("prompt_text", [None, "", "my answer"])
async def test_prompt_accept_uses_native_prompt_text(opened, prompt_text):
    work, session, context = opened
    await server.browser_handle_dialog(True, prompt_text)
    dialog = Dialog("prompt")
    context.emit("dialog", dialog)
    await session.settle_dialogs()
    event, = session.take_dialog_news()
    assert dialog.calls == [("accept", prompt_text)]
    assert event["default_value"] == "original"
    assert event.get("prompt_text") == prompt_text
    assert event["succeeded"] is True


async def test_prompt_text_is_not_sent_to_a_confirm(opened):
    work, session, context = opened
    await server.browser_handle_dialog(True, "not for a confirm")
    dialog = Dialog()
    context.emit("dialog", dialog)
    await session.settle_dialogs()
    assert dialog.calls == [("accept", None)]
    assert "prompt_text" not in session.take_dialog_news()[0]


@pytest.mark.parametrize("inspect_first", [True, False])
async def test_expiry_uses_the_movable_clock_and_clears_the_arm(opened, inspect_first):
    work, session, context = opened
    now = [100.0]
    session._clock = lambda: now[0]
    await server.browser_handle_dialog(True, timeout_seconds=2)
    assert session.dialog_arm()["expires"] == 102
    now[0] = 102
    if inspect_first:
        assert session.dialog_arm() is None
    dialog = Dialog()
    context.emit("dialog", dialog)
    await session.settle_dialogs()
    assert dialog.calls == [("dismiss", None)]
    assert session.take_dialog_news()[0]["source"] == "default"
    assert session.dialog_arm() is None


async def test_a_second_tool_call_replaces_the_arm(opened):
    work, session, context = opened
    await server.browser_handle_dialog(True, "old")
    said = await server.browser_handle_dialog(False, timeout_seconds=600)
    assert "dismiss" in said and "600 seconds" in said
    dialog = Dialog("beforeunload")
    context.emit("dialog", dialog)
    await session.settle_dialogs()
    assert dialog.calls == [("dismiss", None)]
    assert session.take_dialog_news()[0]["source"] == "armed"


@pytest.mark.parametrize("arguments", [
    {"accept": False, "prompt_text": ""},
    {"accept": False, "prompt_text": "no"},
    *[{"accept": True, "timeout_seconds": value}
      for value in (0, -1, 600.01, float("inf"), float("nan"))],
])
async def test_invalid_arms_are_refused_without_replacing_a_valid_one(opened, arguments):
    work, session, context = opened
    await server.browser_handle_dialog(True)
    before = session.dialog_arm()
    with pytest.raises(ValueError, match="refused:"):
        await server.browser_handle_dialog(**arguments)
    assert session.dialog_arm() == before


@pytest.mark.parametrize("accept", [True, False])
async def test_status_shows_arm_and_recent_dialogs_and_drains_news(opened, accept):
    work, session, context = opened
    session._clock = lambda: 10
    context.emit("dialog", Dialog())
    await server.browser_handle_dialog(accept, timeout_seconds=30)
    said = await server.browser_status()
    assert "dialog arm: %s, 30.0 seconds left." % ("accept" if accept else "dismiss") in said
    assert 'confirm "Move these items?"' in said and "dismissed" in said
    assert session.take_dialog_news() == []
    assert 'confirm "Move these items?"' in await server.browser_status()


async def test_status_includes_dialogs_delivered_while_reading_pages(opened, monkeypatch):
    work, session, context = opened

    async def describe():
        asyncio.get_running_loop().call_soon(context.emit, "dialog", Dialog())
        return [{"url": "https://host/path", "title": "", "active": True}]

    monkeypatch.setattr(session, "describe_pages", describe)
    assert 'confirm "Move these items?"' in await server.browser_status()
    assert session.take_dialog_news() == []


@pytest.mark.parametrize("funnel", ["reading", "typing"])
async def test_reads_and_typing_report_news_once_including_during_the_call(opened, funnel):
    work, session, context = opened

    async def operation(session, *args, **kwargs):
        context.emit("dialog", Dialog())
        return "done"

    if funnel == "reading":
        said = await work.reading(operation)
    else:
        said = await work.typing(operation, "#field", "x")
    assert 'confirm "Move these items?"' in said

    async def read(session):
        return "unchanged"

    assert await work.reading(read) == "unchanged"
    assert session.take_dialog_news() == []


async def test_typing_that_outlives_its_call_still_reports_dialogs(opened, monkeypatch):
    from invisible_playwright_mcp.mcp import work as work_module

    work, session, context = opened
    gate = asyncio.Event()
    monkeypatch.setattr(work_module, "ANSWER_WITHIN_S", 0.01)

    async def type_text(session, *args):
        context.emit("dialog", Dialog())
        await gate.wait()
        return "typed"

    try:
        said = await work.typing(type_text, "#field", "long text")
        assert "still typing" in said and "dismissed (default" in said
        assert session.take_dialog_news() == []
    finally:
        gate.set()
        await work._typing["main"].task


@pytest.mark.parametrize("error", [RuntimeError("answer refused"), TargetClosedError("target gone")])
async def test_a_failed_answer_is_reported_as_failed(opened, monkeypatch, error):
    work, session, context = opened
    dialog = Dialog(error=error)
    clicked(monkeypatch, context, dialog)
    said = await server.browser_click("#go")
    assert "answer failed" in said and str(error) in said
    assert session._dialogs[-1]["succeeded"] is False
    assert dialog.calls == [("dismiss", None)]


@pytest.mark.parametrize("broken", ["metadata", "recording", "page"])
async def test_recording_errors_cannot_prevent_answering(opened, monkeypatch, broken, caplog):
    work, session, context = opened

    class BrokenDialog(Dialog):
        @property
        def message(self):
            raise RuntimeError("metadata broke")

        @message.setter
        def message(self, value):
            pass

    dialog = BrokenDialog() if broken == "metadata" else Dialog()
    if broken == "recording":
        def fail(event):
            raise RuntimeError("recording broke")
        monkeypatch.setattr(session, "_queue_dialog", fail)
    if broken == "page":
        dialog.page = object()
    with caplog.at_level("DEBUG", logger="invisible_playwright_mcp"):
        context.emit("dialog", dialog)
        await session.settle_dialogs()
    assert dialog.calls == [("dismiss", None)]
    assert "swallowed:" in caplog.text


async def test_history_and_news_are_bounded_and_dropped_news_is_counted(opened):
    work, session, context = opened
    for index in range(25):
        context.emit("dialog", Dialog(message=str(index) + "\n" + "x" * 600))
    await session.settle_dialogs()
    assert len(session._dialogs) == len(session._dialog_unreported) == 20
    assert all(len(event["message"]) == 500 for event in session._dialogs)
    said = session.dialog_news()
    assert "5 older unreported events dropped" in said
    assert len(said.splitlines()) == 21
    assert session.dialog_news() == ""


@pytest.mark.parametrize("tool", ["browser_click_at", "browser_take_screenshot", "browser_watch"])
async def test_image_wire_results_retain_news_for_the_next_text_tool(opened, monkeypatch, tool):
    work, session, context = opened

    async def picture(*args, **kwargs):
        context.emit("dialog", Dialog())
        return b"picture"

    monkeypatch.setattr(actions, "click_at", picture)
    monkeypatch.setattr(actions, "screenshot_png", picture)
    monkeypatch.setattr(session, "watch_frame", picture)
    arguments = {"x": 1, "y": 2} if tool == "browser_click_at" else {}
    content = await server.mcp.call_tool(tool, arguments)
    assert len(content) == 1 and isinstance(content[0], ImageContent)
    assert len(session._dialog_unreported) == 1
    assert 'confirm "Move these items?"' in await server.browser_status()
    assert session.take_dialog_news() == []


async def test_slow_answers_are_bounded_and_later_failure_is_not_lost(opened, monkeypatch):
    work, session, context = opened
    gate = asyncio.Event()
    dialog = Dialog(gate=gate, error=RuntimeError("late failure"))
    clicked(monkeypatch, context, dialog)
    wait = asyncio.wait
    timeouts = []

    async def measured_wait(tasks, *, timeout):
        timeouts.append(timeout)
        return await wait(tasks, timeout=timeout)

    monkeypatch.setattr(session_module.asyncio, "wait", measured_wait)
    monkeypatch.setattr(StealthSession, "DIALOG_SETTLE_S", 0.05)
    async with asyncio.timeout(1):
        said = await server.browser_click("#go")
    assert timeouts == [0.05]
    assert "answer pending" in said
    assert session._dialog_tasks
    gate.set()
    await session.settle_dialogs()
    assert "answer failed" in session.dialog_news()
    assert session.dialog_news() == ""


async def test_no_dialog_means_no_timed_wait(opened, monkeypatch):
    work, session, context = opened

    async def fail_wait(*args, **kwargs):
        pytest.fail("a quiet action must not pay a fixed dialog delay")

    async def action(session):
        return "done"

    monkeypatch.setattr(session_module.asyncio, "wait", fail_wait)
    assert await work.acting(action) == "done"


async def test_close_drops_arms_history_and_pending_news(opened):
    work, session, context = opened
    gate = asyncio.Event()
    context.emit("dialog", Dialog(gate=gate))
    tasks = tuple(session._dialog_tasks)
    session.take_dialog_news()
    session.arm_dialog(True, None, 60)
    await session.close()
    gate.set()
    await asyncio.gather(*tasks)
    assert session.dialog_arm() is None
    assert not session._dialogs and not session._dialog_tasks
    assert session.take_dialog_news() == []
    assert not context.listeners["dialog"]


async def test_arms_and_news_are_browser_scoped(opened):
    work, main, context = opened
    support = StealthSession()
    await support.start()
    work._open["support"] = support
    work._launched["support"] = {"seed": 4243}
    try:
        await server.browser_handle_dialog(True, browser="support")
        dialog = Dialog(message="main only")
        context.emit("dialog", dialog)
        await main.settle_dialogs()
        assert dialog.calls == [("dismiss", None)]
        assert support.dialog_arm()["accept"] is True
        assert "main only" not in await server.browser_status(browser="support")
        assert "main only" in await server.browser_status()
        dialog = Dialog(message="support only")
        support._context.emit("dialog", dialog)
        await support.settle_dialogs()
        assert dialog.calls == [("accept", None)]
        assert "support only" not in await server.browser_status()
        assert "support only" in await server.browser_status(browser="support")
    finally:
        await support.close()


@pytest.fixture
def dialog_site():
    body = b"""<!doctype html><html><body>
    <button id="confirm" onclick="document.getElementById('out').textContent =
      confirm('Move these items?') ? 'yes' : 'no'">Move</button>
    <button id="prompt" onclick="document.getElementById('out').textContent =
      prompt('New name?', 'original')">Name</button><div id="out"></div>
    </body></html>"""

    class Handler(http.server.BaseHTTPRequestHandler):
        def do_GET(self):
            self.send_response(200)
            self.send_header("Content-Type", "text/html")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)

        def log_message(self, *args):
            pass

    httpd = http.server.ThreadingHTTPServer(("127.0.0.1", 0), Handler)
    thread = threading.Thread(target=httpd.serve_forever, daemon=True)
    thread.start()
    try:
        yield "http://127.0.0.1:%d/" % httpd.server_port
    finally:
        httpd.shutdown()
        httpd.server_close()
        thread.join()


@pytest.mark.e2e
@pytest.mark.skipif(sys.platform not in ("linux", "win32"),
                    reason="the engine supports Linux and Windows, not darwin")
async def test_real_confirm_and_prompt_are_answered_natively(dialog_site, monkeypatch):
    options = {"headless": True}
    if os.environ.get("STEALTHFOX_BINARY"):
        options["binary_path"] = os.environ["STEALTHFOX_BINARY"]
    session = StealthSession(**options)
    work = Work("dialogs-e2e")
    monkeypatch.setattr(server, "work", work)
    await session.start()
    work._open["main"] = session
    try:
        async with asyncio.timeout(120):
            await server.browser_navigate(dialog_site)
            said = await server.browser_click("#confirm")
            assert 'confirm "Move these items?"' in said and "dismissed (default" in said
            assert await session.page().locator("#out").inner_text() == "no"
            await server.browser_handle_dialog(True)
            assert "accepted (armed" in await server.browser_click("#confirm")
            assert await session.page().locator("#out").inner_text() == "yes"
            await server.browser_handle_dialog(True, prompt_text="chosen name")
            assert "accepted (armed" in await server.browser_click("#prompt")
            assert await session.page().locator("#out").inner_text() == "chosen name"
    finally:
        await session.close()
