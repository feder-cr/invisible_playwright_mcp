"""Scripted model turns through the product brain and the real HTTP MCP SDK.

Only the HTTP peer is replaced: requests still pass through initialize,
tools/list and tools/call, with their destination, headers and arguments observed.
"""
from __future__ import annotations

import asyncio
from copy import deepcopy
from functools import partial
import json
import inspect
from types import SimpleNamespace
from uuid import UUID

import httpx
import pytest
from mcp.types import CallToolResult, TextContent, Tool

from invisible_playwright_mcp import search
from invisible_playwright_mcp.agent import OpenRouterBrain
from invisible_playwright_mcp.chat import ChatService


class Model:
    def __init__(self, *turns):
        self.turns = list(turns)
        self.requests = []
        self.chat = SimpleNamespace(completions=SimpleNamespace(create=self.create))

    def create(self, **kwargs):
        self.requests.append(deepcopy(kwargs))
        turn = self.turns.pop(0)
        calls = []
        if isinstance(turn, tuple):
            name, arguments = turn
            calls = [SimpleNamespace(id="call-" + str(len(self.requests)),
                                     function=SimpleNamespace(name=name, arguments=json.dumps(arguments)))]
        dumped = {"role": "assistant", "content": None if calls else turn}
        if calls:
            dumped["tool_calls"] = [{"id": c.id, "type": "function", "function": {
                "name": c.function.name, "arguments": c.function.arguments}} for c in calls]
        msg = SimpleNamespace(content=dumped["content"], tool_calls=calls,
                              model_dump=lambda: dumped)
        return SimpleNamespace(choices=[SimpleNamespace(message=msg)], usage=None)


class Browser:
    tools = [Tool(name="browser_navigate", inputSchema={"type": "object", "properties": {"url": {"type": "string"}}})]
    instructions = "Use the browser to inspect pages."

    def __init__(self):
        self.calls = []

    async def call(self, name, arguments):
        self.calls.append((name, arguments))
        return CallToolResult(content=[TextContent(type="text", text="Page inspected")])


@pytest.fixture
def peer(monkeypatch):
    requests, calls = [], []
    state = {"error": False, "block": False, "missing": False, "http_error": False}
    entered = asyncio.Event()

    async def handle(request):
        requests.append(request)
        assert str(request.url) == search.URL
        assert request.headers["user-agent"] == search.USER_AGENT
        assert "authorization" not in request.headers
        assert "x-api-key" not in request.headers
        if request.method != "POST":
            return httpx.Response(405)
        data = json.loads(request.content)
        method = data["method"]
        if "id" not in data:
            return httpx.Response(202)
        if method == "initialize":
            result = {"protocolVersion": data["params"]["protocolVersion"], "capabilities": {"tools": {}},
                      "serverInfo": {"name": "fixture", "version": "1"}}
        elif method == "tools/list":
            result = {"tools": [{"name": name, "description": "Public web " + name,
                                 "inputSchema": {"type": "object", "properties": {
                                     "objective": {"type": "string"}, "search_queries": {"type": "array", "items": {"type": "string"}},
                                     "urls": {"type": "array", "items": {"type": "string"}},
                                     "session_id": {"type": "string"}, "model_name": {"type": "string"}}}}
                                for name in (["web_search"] if state["missing"] else sorted(search.NAMES))]}
        elif method == "tools/call":
            calls.append(data["params"])
            entered.set()
            if state["http_error"]:
                return httpx.Response(503, text="service unavailable")
            if state["block"]:
                await asyncio.Event().wait()
            result = {"content": [{"type": "text", "text": "quota exceeded" if state["error"] else
                                    "Source https://example.org/source: useful excerpt"}], "isError": state["error"]}
        else:
            raise AssertionError(method)
        return httpx.Response(200, json={"jsonrpc": "2.0", "id": data["id"], "result": result})

    def factory(headers=None, timeout=None, auth=None):
        return httpx.AsyncClient(headers=headers, timeout=timeout, auth=auth,
                                 transport=httpx.MockTransport(handle))

    real = search.streamablehttp_client
    if "httpx_client_factory" in inspect.signature(real).parameters:
        monkeypatch.setattr(search, "streamablehttp_client", partial(real, httpx_client_factory=factory))
    else:
        # SDK 1.8 predates the factory argument; replace the same HTTP seam.
        from mcp.client import streamable_http
        monkeypatch.setattr(streamable_http, "create_mcp_http_client", factory)
    # An ambient key is not permission to use it.
    monkeypatch.setenv("PARALLEL_API_KEY", "ambient-key-must-not-leave")
    return SimpleNamespace(requests=requests, calls=calls, state=state, entered=entered)


async def silent(*_):
    pass


def query():
    return ("web_search", {"objective": "Find source", "search_queries": ["source docs"]})


async def test_brain_dispatches_search_fetch_and_browser_and_consumes_urls(peer):
    model = Model(query(), ("web_fetch", {"urls": ["https://example.org/source"]}),
                  ("browser_navigate", {"url": "https://example.org/source"}), "Answer")
    brain, browser = OpenRouterBrain(model, "configured-model", web_search="parallel"), Browser()
    events = []

    async def say(kind, text):
        events.append((kind, text))

    await brain.handle("Find and inspect a source", browser, say)
    assert browser.calls == [("browser_navigate", {"url": "https://example.org/source"})]
    assert [c["name"] for c in peer.calls] == ["web_search", "web_fetch"]
    assert all(c["arguments"]["model_name"] == "configured-model" for c in peer.calls)
    assert {c["arguments"]["session_id"] for c in peer.calls} == {brain.search_session_id}
    assert peer.calls[1]["arguments"]["urls"] == ["https://example.org/source"]
    assert any("https://example.org/source" in m.get("content", "") for m in model.requests[1]["messages"] if m["role"] == "tool")
    assert any(kind == "result" and "useful excerpt" in text for kind, text in events)
    definitions = {t["function"]["name"]: t["function"] for t in model.requests[0]["tools"]}
    assert set(definitions) == {"web_search", "web_fetch", "browser_navigate"}
    assert "session_id" not in definitions["web_search"]["parameters"]["properties"]
    assert "model_name" not in definitions["web_fetch"]["parameters"]["properties"]
    assert "do not silently switch" in model.requests[0]["messages"][0]["content"]


async def test_default_browser_never_contacts_parallel(peer):
    browser = Browser()
    brain = OpenRouterBrain(Model(("browser_navigate", {"url": "https://search.brave.com/search?q=docs"}), "Done"), "model")
    await brain.handle("Find docs", browser, silent)
    assert peer.requests == []
    assert len(browser.calls) == 1
    assert brain._convo.known == ["browser_navigate"]


async def test_followups_retry_restore_interleaving_and_reset_keep_conversation_identity(peer):
    class SharedModel(Model):
        def create(self, **kwargs):
            # Select by transcript rather than scheduling order, since both
            # brains use this same client concurrently.
            last = kwargs["messages"][-1]
            turn = query() if last["role"] == "user" else "Answer"
            single = Model(turn)
            response = single.create(**kwargs)
            self.requests.extend(single.requests)
            return response

    client = SharedModel()
    a = OpenRouterBrain(client, "shared-model", web_search="parallel")
    b = OpenRouterBrain(client, "shared-model", web_search="parallel")
    await asyncio.gather(a.handle("A", Browser(), silent), b.handle("B", Browser(), silent))
    first_ids = {c["arguments"]["session_id"] for c in peer.calls}
    assert first_ids == {a.search_session_id, b.search_session_id}
    assert len(first_ids) == 2
    await a.handle("follow up", Browser(), silent)
    assert peer.calls[-1]["arguments"]["session_id"] == a.search_session_id
    original = a.search_session_id
    UUID(original)
    # A real ChatService writes and restores it, including an empty transcript.
    service = ChatService(Browser(), a, session_id="saved")
    service.save()
    restored = OpenRouterBrain(Model(query(), "Restored", query(), "Reset"), "shared-model", web_search="parallel")
    assert ChatService(Browser(), restored, session_id="saved").restore()
    await restored.handle("continue", Browser(), silent)
    assert peer.calls[-1]["arguments"]["session_id"] == original
    ChatService(Browser(), restored, session_id="saved").reset()
    assert restored.search_session_id != original
    await restored.handle("new", Browser(), silent)
    assert peer.calls[-1]["arguments"]["session_id"] == restored.search_session_id


async def test_quota_errors_reach_model_without_fallback_or_new_id(peer):
    peer.state["error"] = True
    # Forged model metadata must be overwritten at the production call site.
    model = Model(("web_search", {"objective": "docs", "search_queries": ["docs"],
                                  "session_id": "forged", "model_name": "forged"}), "Failed", query(), "Retry failed")
    brain, browser = OpenRouterBrain(model, "configured-model", web_search="parallel"), Browser()
    events = []

    async def say(kind, text):
        events.append((kind, text))

    await brain.handle("Search", browser, say)
    await brain.handle("Retry", browser, say)
    assert any(k == "err" and "quota exceeded" in t for k, t in events)
    assert "quota exceeded" in model.requests[1]["messages"][-1]["content"]
    assert {c["arguments"]["session_id"] for c in peer.calls} == {brain.search_session_id}
    assert {c["arguments"]["model_name"] for c in peer.calls} == {"configured-model"}
    assert browser.calls == []


async def test_timeout_is_reported_and_cancellation_propagates(peer, monkeypatch):
    from datetime import timedelta

    peer.state["block"] = True
    monkeypatch.setattr(search, "TIMEOUT", timedelta(milliseconds=30))
    model = Model(query(), "Timed out")
    brain = OpenRouterBrain(model, "model", web_search="parallel")
    await brain.handle("Search", Browser(), silent)
    assert "Timed out" in model.requests[1]["messages"][-1]["content"]
    monkeypatch.setattr(search, "TIMEOUT", timedelta(seconds=30))
    peer.entered.clear()
    brain = OpenRouterBrain(Model(query(), "Unreachable"), "model", web_search="parallel")
    task = asyncio.create_task(brain.handle("Search", Browser(), silent))
    await asyncio.wait_for(peer.entered.wait(), 2)
    task.cancel()
    with pytest.raises(asyncio.CancelledError):
        await task
    assert brain._convo.client.turns == ["Unreachable"]


async def test_failed_discovery_does_not_silently_run_browser(peer):
    peer.state["missing"] = True
    model, browser = Model("Must not run"), Browser()
    with pytest.raises(RuntimeError, match="did not offer search and fetch"):
        await OpenRouterBrain(model, "model", web_search="parallel").handle("Search", browser, silent)
    assert model.requests == []
    assert browser.calls == []


@pytest.mark.parametrize(("setting", "flag", "expected"), [
    (None, [], "browser"),
    ("parallel", [], "parallel"),
    ("parallel", ["--web-search", "browser"], "browser"),
    ("browser", ["--web-search", "parallel"], "parallel"),
])
def test_ui_resolves_search_choice_before_constructing_brains(monkeypatch, tmp_path, setting, flag, expected):
    from _cli_brake import run_cli
    from invisible_playwright_mcp import cli, llm, sessions

    monkeypatch.chdir(tmp_path)
    monkeypatch.delenv("INVISIBLE_MCP_WEB_SEARCH", raising=False)
    if setting:
        monkeypatch.setenv("INVISIBLE_MCP_WEB_SEARCH", setting)
    monkeypatch.setattr(cli, "engine_on_disk", lambda _: None)
    monkeypatch.setattr(llm, "make_client", lambda _: Model())
    found = []

    def registry(_opts, _key, make_brain, **_kwargs):
        found.append(make_brain().web_search)
        raise RuntimeError("stopped after search resolution")

    monkeypatch.setattr(sessions, "Sessions", registry)
    result = run_cli("ui", "--openrouter-key", "fixture-key", *flag)
    assert str(result.exception) == "stopped after search resolution"
    assert found == [expected]
    help_result = run_cli("ui", "--help")
    assert "--web-search" in help_result.output


def test_invalid_search_choice_is_rejected_before_startup():
    from _cli_brake import run_cli

    result = run_cli("ui", "--web-search", "unknown")
    assert result.exit_code == 2
    assert "Invalid value for '--web-search'" in result.output


async def test_http_failure_is_reported_without_fallback(peer):
    peer.state["http_error"] = True
    brain = OpenRouterBrain(Model(query(), "Unavailable"), "model", web_search="parallel")
    browser = Browser()
    service = ChatService(browser, brain)
    await service.send("Search")
    assert any(event["kind"] == "err" and "503" in event["text"]
               for event in service.history)
    assert browser.calls == []
    assert len(peer.calls) == 1
