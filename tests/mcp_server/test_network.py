"""Passive capture with event-emitting doubles; no engine or page script."""
from __future__ import annotations

import asyncio
import gc
import json
import logging
import weakref
from types import SimpleNamespace

import pytest
from mcp import types

from invisible_playwright_mcp.mcp import network, plan, server
from invisible_playwright_mcp.mcp.session import StealthSession
from invisible_playwright_mcp.mcp.work import Work


class Context:
    def __init__(self):
        self.listeners = {}
        self.routes = []
        self.route_calls = []
        self.unroute_calls = []
        self.route_error = None
        self.unroute_error = None

    def on(self, event, callback):
        self.listeners[event] = callback

    def remove_listener(self, event, callback):
        assert self.listeners.pop(event) == callback

    def emit(self, event, value):
        self.listeners[event](value)

    async def route(self, pattern, handler):
        self.route_calls.append((pattern, handler))
        self.routes.append((pattern, handler))
        await asyncio.sleep(0)
        if self.route_error:
            raise RuntimeError(self.route_error)

    async def unroute(self, pattern, handler):
        self.unroute_calls.append((pattern, handler))
        self.routes = [(p, h) for p, h in self.routes if p != pattern or h is not handler]
        await asyncio.sleep(0)
        if self.unroute_error:
            raise RuntimeError(self.unroute_error)


class Request:
    def __init__(self, url="https://site.test/api", *, method="POST", body=b'{"FAST_VERLAST":1}',
                 resource_type="fetch", headers=None):
        self.method = method
        self.url = url
        self.post_data_buffer = body
        self.resource_type = resource_type
        self.headers = headers if headers is not None else {"Content-Type": "application/json"}
        self.frame = SimpleNamespace(url="https://site.test/page")
        self.failure = None
        self.reply = Response(self)

    async def response(self):
        return self.reply


class Response:
    def __init__(self, request, *, body=b'{"ok":true}', status=200, headers=None):
        self.request = request
        self.status = status
        self.status_text = "OK"
        self.headers = headers if headers is not None else {"Content-Type": "application/json"}
        self.raw = body
        self.reads = 0

    async def body(self):
        self.reads += 1
        return self.raw


def recorder(**limits):
    capture = network.Network(network.Limits(**limits))
    context = Context()
    capture.attach(context)
    return capture, context


async def finish(capture, context, request):
    context.emit("response", request.reply)
    assert not request.reply.reads
    context.emit("requestfinished", request)
    await asyncio.gather(*capture._tasks.values())


def rows(capture, **kwargs):
    return capture.entries(include_bodies=True, **kwargs)["entries"]


async def test_full_entry_and_lengths_only_default():
    capture, context = recorder()
    request = Request()
    context.emit("request", request)
    assert rows(capture)[0]["response_body_state"] == "pending"
    await finish(capture, context, request)
    row = rows(capture)[0]
    assert row["id"] == 1 and row["started"] > 0
    assert (row["method"], row["url"], row["resource_type"]) == (
        "POST", request.url, "fetch")
    assert row["post_data"] == request.post_data_buffer.decode()
    assert row["status"] == 200 and row["status_text"] == "OK"
    assert row["response_body"] == '{"ok":true}'
    assert row["duration_ms"] >= 0 and row["page"] == request.frame.url
    assert row["failure"] is None
    out = capture.entries()
    assert out["next_since_id"] == 1 and out["dropped"] == 0
    assert "untrusted page data" in out["note"]
    summary = out["entries"][0]
    assert "post_data" not in summary and "response_body" not in summary
    assert summary["post_data_bytes"] == len(request.post_data_buffer)
    assert summary["response_body_bytes"] == len(request.reply.raw)


async def test_header_redaction_is_at_capture_time_and_case_insensitive():
    capture, context = recorder()
    headers = {"cOoKiE": "cookie-secret", "SET-cookie": "set-secret",
               "AUTHORIZATION": "auth-secret", "Proxy-Authorization": "proxy-secret",
               "X-Keep": "visible", "Content-Type": "text/plain"}
    request = Request(headers=headers)
    request.reply.headers = headers
    context.emit("request", request)
    await finish(capture, context, request)
    stored = json.dumps([row.data for row in capture._rows.values()])
    output = json.dumps(capture.entries(include_bodies=True))
    for secret in ("cookie-secret", "set-secret", "auth-secret", "proxy-secret"):
        assert secret not in stored and secret not in output
    for field in ("request_headers", "response_headers"):
        got = rows(capture)[0][field]
        assert all(got[k] == "[redacted]" for k in list(headers)[:4])
        assert got["X-Keep"] == "visible"


def test_ring_eviction_filters_and_paging_oldest_first():
    capture, context = recorder(entries=3)
    requests = [Request(url="https://site.test/" + str(i),
                        resource_type="xhr" if i % 2 else "fetch") for i in range(5)]
    for request in requests:
        context.emit("request", request)
    out = capture.entries(max_entries=2)
    assert [r["id"] for r in out["entries"]] == [3, 4]
    assert out["next_since_id"] == 4 and out["dropped"] == 2
    assert [r["id"] for r in rows(capture, since_id=4)] == [5]
    assert [r["id"] for r in rows(capture, resource_types=["xhr"])] == [4]
    assert [r["id"] for r in rows(capture, url_contains="/4", since_id=3)] == [5]
    assert rows(capture, resource_types=[]) == []
    assert capture.entries(since_id=5)["next_since_id"] is None
    assert capture.entries()["capacity"] == 3


@pytest.mark.parametrize("kwargs", [{"max_entries": 0}, {"max_entries": -1}, {"since_id": -1}])
def test_invalid_filters_refuse(kwargs):
    capture, _ = recorder()
    with pytest.raises(ValueError):
        capture.entries(**kwargs)


async def test_aggregate_budget_counts_both_bodies_and_evicts_oldest():
    capture, context = recorder(body_bytes=10, total_body_bytes=15)
    first, second = Request(body=b"12345678"), Request(body=b"abcdefgh")
    context.emit("request", first)
    context.emit("request", second)
    assert [r["id"] for r in rows(capture)] == [2]
    assert capture.entries()["stored_body_bytes"] == 8
    second.reply.raw = b"123456789"
    await finish(capture, context, second)
    assert rows(capture) == []
    assert capture.entries()["stored_body_bytes"] == 0
    assert capture.entries()["dropped"] == 2


async def test_body_caps_are_byte_caps_with_original_lengths_and_safe_unicode_boundary():
    capture, context = recorder(body_bytes=5)
    request = Request(body="a\u20ac\u20ac".encode())
    request.reply.raw = b"123456789"
    context.emit("request", request)
    await finish(capture, context, request)
    row = rows(capture)[0]
    assert row["post_data"] == "a\u20ac"
    assert row["post_data_bytes"] == 7 and row["post_data_truncated"]
    assert row["response_body"] == "12345"
    assert row["response_body_bytes"] == 9 and row["response_body_truncated"]
    assert capture.entries()["stored_body_bytes"] == 9


async def test_legacy_charset_expansion_still_respects_stored_byte_cap():
    capture, context = recorder(body_bytes=3)
    request = Request(body=b"\xe9\xe9\xe9", headers={"Content-Type": "text/plain; charset=latin-1"})
    context.emit("request", request)
    row = rows(capture)[0]
    assert row["post_data"] == "\u00e9" and row["post_data_truncated"]
    assert row["post_data_stored_bytes"] == 2


@pytest.mark.parametrize("mime,body", [
    ("application/octet-stream", b"\x00\xff"),
    ("text/plain", b"\xff"),
    ("text/plain; charset=made-up", b"abc"),
])
async def test_binary_and_undecodable_bodies_store_only_size_and_type(mime, body):
    capture, context = recorder()
    request = Request(body=body, headers={"Content-Type": mime})
    request.reply.headers, request.reply.raw = request.headers, body
    context.emit("request", request)
    await finish(capture, context, request)
    row = rows(capture)[0]
    for name in ("post_data", "response_body"):
        assert name not in row
        assert row[name + "_bytes"] == len(body)
        assert row[name + "_content_type"] == mime
        assert "size only" in row[name + "_state"]
    assert capture.entries()["stored_body_bytes"] == 0


@pytest.mark.parametrize("resource_type,reads", [
    ("document", 1), ("xhr", 1), ("fetch", 1), ("image", 0), ("script", 0),
])
async def test_only_selected_response_types_are_read(resource_type, reads):
    capture, context = recorder()
    request = Request(resource_type=resource_type)
    context.emit("request", request)
    await finish(capture, context, request)
    assert request.reply.reads == reads


async def test_clear_keeps_ids_and_ignores_late_events_and_does_not_touch_context():
    capture, context = recorder(entries=1)
    old, pending = Request(), Request()
    context.emit("request", old)
    context.emit("request", pending)
    context.emit("requestfinished", pending)
    cleared = capture.clear()
    assert cleared == {"cleared": 1, "last_id": 2, "dropped": 1}
    await asyncio.gather(*capture._tasks.values(), return_exceptions=True)
    for request in (old, pending):
        context.emit("response", request.reply)
        context.emit("requestfinished", request)
        context.emit("requestfailed", request)
    assert capture.entries()["entries"] == []
    assert set(context.listeners) == {"request", "response", "requestfinished", "requestfailed"}
    current = Request()
    context.emit("request", current)
    assert rows(capture)[0]["id"] == 3
    await capture.close()
    assert context.listeners == {} and capture._tasks == {}


async def test_failures_redirects_and_unreadable_bodies_are_explicit():
    capture, context = recorder()
    failed, redirect, unreadable = Request(), Request(), Request()
    failed.failure = "NS_ERROR_CONNECTION_REFUSED"
    for request in (failed, redirect, unreadable):
        context.emit("request", request)
    context.emit("requestfailed", failed)
    redirect.reply.status = 302
    await finish(capture, context, redirect)

    async def gone():
        raise RuntimeError("page gone with sensitive page data")

    unreadable.reply.body = gone
    await finish(capture, context, unreadable)
    a, b, c = rows(capture)
    assert a["failure"] == failed.failure and a["duration_ms"] >= 0
    assert b["response_body_state"] == "redirect body unavailable"
    assert c["response_body_state"] == "unavailable: RuntimeError"
    assert "sensitive page data" not in json.dumps(c)


async def test_reads_are_bounded_and_timeout_without_blocking_browser(monkeypatch):
    monkeypatch.setattr(network, "BODY_READ_CONCURRENCY", 1)
    monkeypatch.setattr(network, "BODY_READ_TIMEOUT_SECONDS", 0.02)
    capture, context = recorder(entries=2)
    started = []

    async def stall():
        started.append(True)
        await asyncio.Event().wait()

    requests = [Request() for _ in range(2)]
    for request in requests:
        request.reply.body = stall
        context.emit("request", request)
        context.emit("requestfinished", request)
    await asyncio.gather(*capture._tasks.values())
    assert len(started) == 1
    assert all(row["response_body_state"] == "unavailable: TimeoutError" for row in rows(capture))


async def test_eviction_cancels_reads_without_resurrecting_rows():
    capture, context = recorder(entries=1)
    started = asyncio.Event()
    release = asyncio.Event()

    async def late():
        started.set()
        try:
            await release.wait()
        except asyncio.CancelledError:
            await release.wait()
        return b"late"

    first = Request()
    first.reply.body = late
    context.emit("request", first)
    context.emit("requestfinished", first)
    await started.wait()
    second = Request()
    context.emit("request", second)
    release.set()
    await asyncio.gather(*capture._tasks.values(), return_exceptions=True)
    assert [row["id"] for row in rows(capture)] == [2]
    assert capture.entries()["stored_body_bytes"] == len(second.post_data_buffer)


def test_bad_event_cannot_escape_the_listener():
    capture, context = recorder()
    for event in context.listeners:
        context.emit(event, object())
    assert rows(capture) == []


@pytest.mark.parametrize("enabled", [False, True])
async def test_engine_missing_post_data_is_not_reported_as_an_empty_body(enabled):
    capture, context = recorder()
    await capture.capture(request_bodies=enabled)
    request = Request(body=None, headers={"Content-Length": "123"})
    context.emit("request", request)
    row = rows(capture)[0]
    assert row["post_data_bytes"] == 123
    assert row["post_data_state"] == (
        "unavailable: engine did not expose request body" if enabled else network.REQUEST_BODIES_HINT)
    assert "post_data" not in row
    unknown = Request(body=None, headers={})
    context.emit("request", unknown)
    row = rows(capture)[1]
    assert row["post_data_bytes"] is None
    assert row["post_data_state"] == (
        "absent or unavailable from engine" if enabled else network.REQUEST_BODIES_HINT)
    await capture.close()


@pytest.mark.parametrize("method", ["POST", "PUT", "PATCH", "DELETE", "OPTIONS"])
def test_body_capable_methods_without_data_while_off_name_the_opt_in(method):
    capture, context = recorder()
    request = Request(method=method, body=None)
    context.emit("request", request)
    assert rows(capture)[0]["post_data_state"] == network.REQUEST_BODIES_HINT
    assert capture.entries()["request_bodies"] is False


async def test_capture_is_idempotent_concurrent_and_removes_exactly_its_route():
    capture, context = recorder()
    assert not context.routes and capture.entries()["request_bodies"] is False
    assert await capture.capture(request_bodies=False) == {"request_bodies": False}
    assert not context.unroute_calls

    async def unrelated(route):
        await route.fallback()

    await context.route("**/*", unrelated)
    replies = await asyncio.gather(*(capture.capture(request_bodies=True) for _ in range(5)))
    assert all(reply == {"request_bodies": True} for reply in replies)
    assert len(context.route_calls) == 2
    pattern, handler = context.route_calls[-1]
    assert pattern == "**/*" and handler is not unrelated
    assert context.routes == [("**/*", unrelated), (pattern, handler)]
    capture.clear()
    assert capture.entries()["request_bodies"] is True
    assert len(context.routes) == 2
    await asyncio.gather(*(capture.capture(request_bodies=False) for _ in range(5)))
    assert context.unroute_calls == [(pattern, handler)]
    assert context.routes == [("**/*", unrelated)]
    assert capture.entries()["request_bodies"] is False
    await capture.close()


@pytest.mark.parametrize("rollback_fails", [False, True])
async def test_route_failure_rolls_back_partial_registration_and_preserves_error(rollback_fails):
    capture, context = recorder()
    context.route_error = "BrowserContext.route: service_workers='block'\nengine detail"
    if rollback_fails:
        context.unroute_error = "rollback failed"
    with pytest.raises(RuntimeError) as caught:
        await capture.capture(request_bodies=True)
    assert str(caught.value) == context.route_error
    assert capture.entries()["request_bodies"] is False
    assert not context.routes
    assert context.unroute_calls == context.route_calls
    context.route_error = context.unroute_error = None
    assert await capture.capture(request_bodies=True) == {"request_bodies": True}
    assert len(context.routes) == 1
    await capture.close()


async def test_failed_disable_restores_the_registered_handler_and_state():
    capture, context = recorder()
    await capture.capture(request_bodies=True)
    registered = list(context.routes)
    context.unroute_error = "engine refused unroute"
    with pytest.raises(RuntimeError, match="engine refused unroute"):
        await capture.capture(request_bodies=False)
    assert context.routes == registered
    assert capture.entries()["request_bodies"] is True
    context.unroute_error = None
    await capture.capture(request_bodies=False)
    assert not context.routes
    await capture.close()


async def test_cancelled_enable_cleans_up_partial_registration():
    capture, context = recorder()
    entered = asyncio.Event()

    async def wait(pattern, handler):
        context.routes.append((pattern, handler))
        entered.set()
        await asyncio.Event().wait()

    context.route = wait
    task = asyncio.create_task(capture.capture(request_bodies=True))
    await entered.wait()
    task.cancel()
    with pytest.raises(asyncio.CancelledError):
        await task
    assert not context.routes and not capture.entries()["request_bodies"]
    await capture.close()


@pytest.mark.parametrize("fallback_available", [False, True])
@pytest.mark.parametrize("raises", [False, True])
async def test_pass_through_only_falls_back_or_continues_once_and_never_raises(
        fallback_available, raises, caplog):
    capture, context = recorder()
    await capture.capture(request_bodies=True)
    handler = context.routes[0][1]
    called = []

    async def fallback():
        called.append("fallback")
        if raises:
            raise RuntimeError("request already gone")

    async def continue_():
        called.append("continue")
        if raises:
            raise RuntimeError("request already gone")

    route = SimpleNamespace(continue_=continue_)
    if fallback_available:
        route.fallback = fallback
    with caplog.at_level(logging.DEBUG, logger="invisible_playwright_mcp"):
        await asyncio.wait_for(handler(route), 1)
    assert called == ["fallback" if fallback_available else "continue"]
    assert ("request already gone" in caplog.text) is raises
    await capture.close()


@pytest.mark.parametrize("unroute_fails", [False, True])
async def test_close_releases_capture_and_new_browsers_default_to_off(unroute_fails, caplog):
    capture, context = recorder()
    await capture.capture(request_bodies=True)
    registered = context.routes[0]
    if unroute_fails:
        context.unroute_error = "context gone"
    with caplog.at_level(logging.DEBUG, logger="invisible_playwright_mcp"):
        await capture.close()
    assert context.unroute_calls == [registered]
    assert not context.routes and not context.listeners
    assert capture.entries()["request_bodies"] is False
    assert ("context gone" in caplog.text) is unroute_fails
    fresh, _ = recorder()
    assert fresh.entries()["request_bodies"] is False
    with pytest.raises(RuntimeError, match="not attached"):
        await capture.capture(request_bodies=True)


async def test_duplicate_finish_does_not_double_count_body_bytes():
    capture, context = recorder()
    request = Request()
    context.emit("request", request)
    await finish(capture, context, request)
    before = capture.entries()["stored_body_bytes"]
    context.emit("requestfinished", request)
    await asyncio.sleep(0)
    assert request.reply.reads == 1
    assert capture.entries()["stored_body_bytes"] == before


def test_history_does_not_retain_engine_objects_with_unredacted_headers():
    capture, context = recorder()
    request = Request(headers={"Cookie": "secret"})
    ref = weakref.ref(request)
    context.emit("request", request)
    del request
    gc.collect()
    assert ref() is None
    assert rows(capture)[0]["request_headers"]["Cookie"] == "[redacted]"


async def test_task_queue_stays_bounded_even_before_cancelled_tasks_can_exit():
    capture, context = recorder(entries=2)
    requests = [Request() for _ in range(20)]
    for request in requests:
        context.emit("request", request)
        context.emit("requestfinished", request)
        assert len(capture._tasks) <= capture.limits.entries
    assert rows(capture)[-1]["response_body_state"] == "body read queue full"
    await capture.close()
    assert not capture._tasks


def test_returned_headers_cannot_mutate_the_capture():
    capture, context = recorder()
    request = Request()
    context.emit("request", request)
    rows(capture)[0]["request_headers"].clear()
    assert rows(capture)[0]["request_headers"] == request.headers


def test_limits_are_decided_by_plan_once_and_never_passed_to_engine():
    planned = plan.launched_here({
        "STEALTHFOX_NETWORK_ENTRIES": "3",
        "STEALTHFOX_NETWORK_BODY_BYTES": "7",
        "STEALTHFOX_NETWORK_TOTAL_BODY_BYTES": "11",
    })
    session = StealthSession(**planned)
    assert session.network.limits == network.Limits(3, 7, 11)
    assert "network_limits" not in session._kwargs
    assert plan.launched_here({})["network_limits"] == network.Limits()


@pytest.mark.parametrize("name", [
    "STEALTHFOX_NETWORK_ENTRIES", "STEALTHFOX_NETWORK_BODY_BYTES",
    "STEALTHFOX_NETWORK_TOTAL_BODY_BYTES",
])
@pytest.mark.parametrize("bad", ["0", "-1", "abc", "1.5"])
def test_bad_environment_caps_are_refused(name, bad):
    with pytest.raises(ValueError, match=name):
        plan.launched_here({name: bad})


async def test_session_attaches_before_any_navigation_and_closes_capture(monkeypatch):
    from invisible_playwright_mcp.mcp import session as session_mod

    context = Context()

    async def close():
        assert context.listeners == {}

    context.close = close

    class Engine:
        def __init__(self, **kwargs):
            assert "network_limits" not in kwargs

        async def __aenter__(self):
            return context

        async def __aexit__(self, *args):
            pass

    monkeypatch.setattr(session_mod, "InvisiblePlaywright", Engine)
    session = StealthSession()
    await session.start()
    request = Request(resource_type="document")
    context.emit("request", request)
    assert rows(session.network)[0]["id"] == 1
    await session.close()


async def test_tools_preserve_large_json_shape_and_browser_isolation(monkeypatch):
    work = Work("network-test")
    monkeypatch.setattr(server, "work", work)
    for role in ("main", "support"):
        session = StealthSession()
        session._context = Context()
        session.network.attach(session._context)
        request = Request(url="https://" + role + ".test", body=b"x" * 7000)
        session._context.emit("request", request)
        work._open[role] = session
    out = json.loads(await server.browser_network(include_bodies=True))
    assert len(out["entries"][0]["post_data"]) == 7000
    assert out["next_since_id"] == 1 and "preview" not in out
    assert "support.test" not in json.dumps(out)
    assert json.loads(await server.browser_network_capture(request_bodies=True)) == {
        "request_bodies": True}
    assert json.loads(await server.browser_network(browser="support"))["request_bodies"] is False
    await server.browser_network_capture(browser="support", request_bodies=False)
    assert json.loads(await server.browser_network())["request_bodies"] is True
    assert json.loads(await server.browser_network_clear())["cleared"] == 1
    assert json.loads(await server.browser_network())["entries"] == []
    assert len(json.loads(await server.browser_network(browser="support"))["entries"]) == 1
    tools = {tool.name: tool for tool in await server.mcp.list_tools()}
    assert tools["browser_network"].inputSchema["properties"]["max_entries"]["default"] == (
        network.DEFAULT_MAX_ENTRIES)
    assert tools["browser_network_capture"].inputSchema["required"] == ["request_bodies"]
    for name in ("browser_network", "browser_network_clear", "browser_network_capture"):
        assert tools[name].annotations.openWorldHint is False
    for session in work._open.values():
        await session.network.close()


async def test_capture_registration_error_reaches_the_tool_unchanged(monkeypatch):
    work = Work("network-error-test")
    monkeypatch.setattr(server, "work", work)
    session = StealthSession()
    context = Context()
    session._context = context
    session.network.attach(context)
    work._open["main"] = session
    context.route_error = "BrowserContext.route: service_workers='block'\nengine detail"
    request = types.CallToolRequest.model_validate({
        "method": "tools/call",
        "params": {"name": "browser_network_capture", "arguments": {"request_bodies": True}},
    })
    try:
        result = (await server.mcp._mcp_server.request_handlers[types.CallToolRequest](request)).root
        assert result.isError
        text = "\n".join(item.text for item in result.content if item.type == "text")
        assert text == "Error executing tool browser_network_capture: " + context.route_error
        assert not context.routes
        assert json.loads(await server.browser_network())["request_bodies"] is False
    finally:
        await session.network.close()


def test_a_large_answer_stops_early_and_says_there_is_more():
    from invisible_playwright_mcp.mcp import network as net
    capture = net.Network()
    big = "x" * (100 * 1024)
    for i in range(1, 6):
        capture._last_id = i
        capture._rows[i] = net._Entry({"id": i, "url": "https://a/%d" % i,
                                       "resource_type": "xhr", "response_body": big})
    first = capture.entries(include_bodies=True)
    assert first["more"] is True
    assert 1 <= len(first["entries"]) < 5
    rest = capture.entries(include_bodies=True, since_id=first["next_since_id"])
    assert rest["entries"][0]["id"] == first["next_since_id"] + 1
    plain = capture.entries(max_entries=2)
    assert [e["id"] for e in plain["entries"]] == [1, 2] and plain["more"] is True
    assert capture.entries()["more"] is False
