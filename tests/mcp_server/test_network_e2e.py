"""Local traffic through the real engine and MCP dispatch."""
from __future__ import annotations

import asyncio
import http.server
import json
import threading

import pytest
from mcp import types

from invisible_playwright_mcp.mcp import network, server
from invisible_playwright_mcp.mcp.work import Work

pytestmark = pytest.mark.e2e

COOKIE = "network-cookie-secret"
AUTHORIZATION = "Bearer network-authorization-secret"
POST = {"FAST_VERLAST": "keep-this-token", "action": "book-then-cancel"}


def success(result):
    text = "\n".join(item.text for item in result.content if item.type == "text")
    assert not result.isError, text
    return text


@pytest.fixture
def site():
    observed = {}
    auth_codes = ",".join(str(ord(char)) for char in AUTHORIZATION)
    page = ("""
<!doctype html><title>Network fixture</title>
<a id="popup" target="_blank" href="/popup">Open site page</a>
<script>
fetch('/EventOccurred', {
  method: 'POST',
  headers: {'Content-Type': 'application/json',
            'Authorization': String.fromCharCode(%s)},
  body: JSON.stringify(%s)
});
const xhr = new XMLHttpRequest();
xhr.open('GET', '/Recalc');
xhr.send();
</script>
""" % (auth_codes, json.dumps(POST))).encode()

    class Handler(http.server.BaseHTTPRequestHandler):
        def reply(self, status, content_type, body, *, cookie=False):
            self.send_response(status)
            self.send_header("Content-Type", content_type)
            if cookie:
                self.send_header("Set-Cookie", "session=%s; Path=/; HttpOnly" % COOKIE)
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)

        def do_GET(self):
            observed.setdefault("requests", []).append(("GET", self.path))
            if self.path == "/":
                self.reply(200, "text/html; charset=utf-8", page, cookie=True)
            elif self.path == "/popup":
                self.reply(200, "text/html", b"<script>fetch('/ExecuteAction')</script>")
            else:
                observed[self.path] = dict(self.headers)
                self.reply(200, "application/json", b'{"result":"read"}')

        def do_POST(self):
            observed.setdefault("requests", []).append(("POST", self.path))
            body = self.rfile.read(int(self.headers["Content-Length"]))
            observed.setdefault(self.path, []).append({"headers": dict(self.headers), "body": body})
            self.reply(201, "application/json", b'{"result":"booked"}', cookie=True)

        def log_message(self, *args):
            pass

    httpd = http.server.ThreadingHTTPServer(("127.0.0.1", 0), Handler)
    thread = threading.Thread(target=httpd.serve_forever, daemon=True)
    thread.start()
    try:
        yield "http://127.0.0.1:%d/" % httpd.server_port, observed
    finally:
        httpd.shutdown()
        httpd.server_close()
        thread.join()


async def test_request_bodies_off_on_off_with_xhr_popup_redaction_and_browser_isolation(
        site, monkeypatch, tmp_path):
    url, observed = site
    for name in ("STEALTHFOX_PROFILE_DIR", "STEALTHFOX_PROXY", "STEALTHFOX_SEED",
                 "STEALTHFOX_NETWORK_ENTRIES", "STEALTHFOX_NETWORK_BODY_BYTES",
                 "STEALTHFOX_NETWORK_TOTAL_BODY_BYTES"):
        monkeypatch.delenv(name, raising=False)
    monkeypatch.setenv("STEALTHFOX_HEADLESS", "1")
    monkeypatch.setattr("tempfile.tempdir", str(tmp_path))
    work = Work("network-e2e")
    monkeypatch.setattr(server, "work", work)

    async def call(browser, name, arguments=None):
        request = types.CallToolRequest.model_validate({
            "method": "tools/call",
            "params": {"name": name, "arguments": {"browser": browser, **(arguments or {})}},
        })
        result = (await server.mcp._mcp_server.request_handlers[types.CallToolRequest](request)).root
        output = result.model_dump_json()
        assert COOKIE not in output and AUTHORIZATION not in output
        return result

    async def read_complete(path, since_id=None):
        output = ""
        try:
            async with asyncio.timeout(30):
                while True:
                    output = success(await call("main", "browser_network", {"include_bodies": True}))
                    assert COOKIE not in output and AUTHORIZATION not in output
                    data = json.loads(output)
                    matches = [row for row in data["entries"] if row["url"] == url + path
                               and (since_id is None or row["id"] > since_id)]
                    if matches and matches[0]["response_body_state"] != "pending":
                        return matches[0], data
                    await asyncio.sleep(0.05)
        except TimeoutError:
            pytest.fail("network capture never completed %s: %s" % (path, output))

    try:
        success(await call("main", "browser_open"))
        success(await call("main", "browser_navigate", {"url": url, "wait_until": "networkidle"}))
        post, _ = await read_complete("EventOccurred")
        xhr, data = await read_complete("Recalc")
        document, _ = await read_complete("")
        assert post["method"] == "POST"
        assert post["status"] == 201 and post["status_text"] == "Created"
        assert post["request_headers"]["authorization"] == "[redacted]"
        assert post["request_headers"]["cookie"] == "[redacted]"
        assert post["response_headers"]["set-cookie"] == "[redacted]"
        assert document["id"] < post["id"]
        assert xhr["method"] == "GET"
        assert observed["/EventOccurred"][0]["headers"]["Authorization"] == AUTHORIZATION
        assert COOKIE in observed["/EventOccurred"][0]["headers"]["Cookie"]
        assert json.loads(observed["/EventOccurred"][0]["body"]) == POST
        assert COOKIE in observed["/Recalc"]["Cookie"]
        assert data["dropped"] == 0 and data["capacity"] == 500
        assert data["request_bodies"] is False
        assert "post_data" not in post
        assert post["post_data_state"] == network.REQUEST_BODIES_HINT
        for row, resource_type in ((document, "document"), (post, "fetch"), (xhr, "xhr")):
            assert row["resource_type"] == resource_type
            assert row["response_body_state"] == "captured", row
        assert json.loads(post["response_body"]) == {"result": "booked"}
        assert json.loads(xhr["response_body"]) == {"result": "read"}
        assert len(observed["/EventOccurred"]) == 1, observed["requests"]

        enabled = success(await call("main", "browser_network_capture", {"request_bodies": True}))
        assert json.loads(enabled) == {"request_bodies": True}
        success(await call("main", "browser_network_capture", {"request_bodies": True}))
        success(await call("main", "browser_navigate", {"url": url, "wait_until": "networkidle"}))
        captured, data = await read_complete("EventOccurred", since_id=post["id"])
        assert data["request_bodies"] is True
        assert captured["post_data"] == observed["/EventOccurred"][1]["body"].decode()
        assert json.loads(captured["post_data"]) == POST
        assert captured["status"] == 201
        assert captured["response_body_state"] == "captured"
        assert json.loads(captured["response_body"]) == {"result": "booked"}
        assert len(observed["/EventOccurred"]) == 2, observed["requests"]

        disabled = success(await call("main", "browser_network_capture", {"request_bodies": False}))
        assert json.loads(disabled) == {"request_bodies": False}
        success(await call("main", "browser_network_capture", {"request_bodies": False}))
        success(await call("main", "browser_navigate", {"url": url, "wait_until": "networkidle"}))
        third, data = await read_complete("EventOccurred", since_id=captured["id"])
        assert data["request_bodies"] is False
        assert "post_data" not in third and third["post_data_state"] == network.REQUEST_BODIES_HINT
        assert third["status"] == 201 and third["response_body_state"] == "captured"
        assert json.loads(third["response_body"]) == {"result": "booked"}
        assert len(observed["/EventOccurred"]) == 3, json.dumps({
            "requests": observed["requests"],
            "entries": [(row["id"], row["method"], row["url"]) for row in data["entries"]],
        })
        assert all(json.loads(item["body"]) == POST for item in observed["/EventOccurred"])
        filtered = json.loads(success(await call("main", "browser_network", {
            "url_contains": "EventOccurred", "resource_types": ["fetch"], "include_bodies": True,
        })))
        assert [row["id"] for row in filtered["entries"]] == [
            post["id"], captured["id"], third["id"]]
        assert json.loads(filtered["entries"][1]["post_data"]) == POST

        missing = await call("support", "browser_network")
        assert missing.isError
        success(await call("support", "browser_open"))
        own = json.loads(success(await call("support", "browser_network")))
        assert not any(row["url"].startswith(url) for row in own["entries"])
        assert own["request_bodies"] is False
        success(await call("main", "browser_network_capture", {"request_bodies": True}))
        success(await call("support", "browser_network_capture", {"request_bodies": False}))
        success(await call("support", "browser_network_clear"))
        own = json.loads(success(await call("main", "browser_network")))
        assert any(row["id"] == post["id"] and row["url"] == post["url"] for row in own["entries"])
        assert own["request_bodies"] is True
        success(await call("support", "browser_close"))

        summary = success(await call("main", "browser_network"))
        assert all("post_data" not in row and "response_body" not in row
                   for row in json.loads(summary)["entries"])
        success(await call("main", "browser_click", {"selector": "#popup"}))
        child, _ = await read_complete("ExecuteAction")
        assert child["page"] == url + "popup"
        assert child["method"] == "GET" and child["status"] == 200
        success(await call("main", "browser_network_clear"))
        assert json.loads(success(await call("main", "browser_network")))["entries"] == []
        success(await call("main", "browser_navigate", {"url": url + "ExecuteAction"}))
        after, _ = await read_complete("ExecuteAction")
        assert after["id"] > child["id"]
        success(await call("main", "browser_network_capture", {"request_bodies": True}))
        success(await call("main", "browser_close"))
        success(await call("main", "browser_open"))
        assert json.loads(success(await call("main", "browser_network")))["request_bodies"] is False
    finally:
        await work.close_all()
