"""Under a host that opens the browser itself, the server says only what is true.

⛔ WHY. invisible_dots runs one server per browser identity, opens it with
browser_open and closes it with browser_close itself, and offers its model only
the page tools. Handed the full instructions, that model read "OPEN `main` WITH
browser_open BEFORE ANYTHING ELSE" about a tool it does not have, and every
schema offered it a `support` browser it cannot open. So the host says so
(`INVISIBLE_MCP_HOST_MANAGED=1`), and the process serves `main` alone with the
page rules alone.

The mode is decided at import, once, as the session id is, so each case runs the
server's import in a process of its own.
"""
from __future__ import annotations

import json
import os
import subprocess
import sys

_DUMP = """
import asyncio, json
from invisible_playwright_mcp.mcp import server
tools = asyncio.run(server.mcp.list_tools())
print(json.dumps({
    "instructions": server.INSTRUCTIONS,
    "page_rules": server.PAGE_RULES,
    "served": server.mcp.instructions,
    "schemas": {t.name: t.inputSchema for t in tools},
}))
"""

_SUPPORT = """
import asyncio
from invisible_playwright_mcp.mcp import server
try:
    asyncio.run(server.mcp.call_tool("browser_snapshot", {"browser": "support"}))
    print("accepted")
except Exception as exc:
    print("refused" if "main" in str(exc) else "other: %s" % exc)
"""


def _run(script: str, host_managed: bool) -> str:
    env = {k: v for k, v in os.environ.items() if k != "INVISIBLE_MCP_HOST_MANAGED"}
    if host_managed:
        env["INVISIBLE_MCP_HOST_MANAGED"] = "1"
    done = subprocess.run([sys.executable, "-c", script], env=env, capture_output=True,
                          text=True, encoding="utf-8", timeout=120, check=True)
    return done.stdout.strip().splitlines()[-1]


def _dump(host_managed: bool) -> dict:
    return json.loads(_run(_DUMP, host_managed))


def test_a_host_that_opens_the_browser_is_told_the_page_rules_alone():
    """Known-bad: serve INSTRUCTIONS whatever the mode - the host's model reads
    about browser_open and `support`."""
    hosted = _dump(host_managed=True)
    assert hosted["served"] == hosted["page_rules"] == hosted["instructions"]
    for absent in ("browser_open", "browser_close", "support", "`main`"):
        assert absent not in hosted["served"], (
            "a host that opens the browser itself is told about %r" % absent)


def test_no_tool_offers_a_browser_to_choose_under_such_a_host():
    """Known-bad: keep the `Literal["main", "support"]` parameter in host mode."""
    hosted = _dump(host_managed=True)
    offering = [name for name, schema in hosted["schemas"].items()
                if "browser" in (schema.get("properties") or {})]
    assert offering == [], "these tools still take `browser`: %s" % offering


def test_the_helper_is_refused_under_such_a_host():
    """The schema hides the argument; a caller that sends it anyway is turned
    back, so a host never gets a second browser it did not open."""
    assert _run(_SUPPORT, host_managed=True) == "refused"


def test_a_standalone_client_still_gets_everything():
    """The default is unchanged: the opening rule first, the page rules, and the
    two browsers - and the page rules are the same words in both modes, so a
    rule written for one is never missing from the other."""
    alone = _dump(host_managed=False)
    hosted = _dump(host_managed=True)
    assert alone["served"] == alone["instructions"]
    assert alone["instructions"].startswith("Two browsers, `main` and `support`.")
    assert hosted["page_rules"] in alone["instructions"]
    assert "support" in alone["instructions"] and "browser_open" in alone["instructions"]
    assert all("browser" in (s.get("properties") or {})
               for name, s in alone["schemas"].items() if name != "browser_list")
