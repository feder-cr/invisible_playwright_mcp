"""The standalone agent's opt-in, anonymous Parallel Search MCP connection."""
from __future__ import annotations

from contextlib import asynccontextmanager
from copy import deepcopy
from datetime import timedelta

from mcp import ClientSession
from mcp.client.streamable_http import streamablehttp_client

URL = "https://search.parallel.ai/mcp"
USER_AGENT = "invisible-playwright-mcp (https://github.com/feder-cr/invisible_playwright_mcp)"
TIMEOUT = timedelta(seconds=30)
NAMES = {"web_search", "web_fetch"}
INSTRUCTIONS = (
    "For public web discovery you may use web_search instead of browsing a "
    "search engine. Its excerpts and web_fetch results are sources: cite their "
    "URLs, and distinguish them from pages you inspected in the browser. Use "
    "web_fetch only when search excerpts are insufficient. These tools do not "
    "share the browser's cookies, proxy or logged-in sessions. Use the browser "
    "for interaction and authenticated pages. Report search failures plainly; "
    "do not silently switch to browser search after a failed Parallel request."
)


@asynccontextmanager
async def parallel_tools(session_id: str, model: str):
    """Own the SDK contexts in the calling task, for exactly one agent turn.

    The transport can be replaced between turns; the conversation supplies the
    identity separately, so reconnecting never changes the search session.
    """
    try:
        async with streamablehttp_client(
            URL, headers={"User-Agent": USER_AGENT}, timeout=TIMEOUT,
            sse_read_timeout=TIMEOUT,
        ) as (read, write, _):
            async with ClientSession(read, write, read_timeout_seconds=TIMEOUT) as session:
                await session.initialize()
                tools = []
                for tool in (await session.list_tools()).tools:
                    if tool.name not in NAMES:
                        continue
                    schema = deepcopy(tool.inputSchema)
                    # Runtime metadata is supplied by the host, never the model.
                    for key in ("session_id", "model_name"):
                        schema.get("properties", {}).pop(key, None)
                        if key in schema.get("required", []):
                            schema["required"].remove(key)
                    tools.append(tool.model_copy(update={"inputSchema": schema}))
                if {tool.name for tool in tools} != NAMES:
                    raise RuntimeError("Parallel Search MCP did not offer search and fetch")

                async def call(name, arguments):
                    if name not in NAMES:
                        raise ValueError("unknown Parallel tool: " + name)
                    return await session.call_tool(name, {
                        **arguments, "session_id": session_id, "model_name": model,
                    })

                yield tools, call
    except ExceptionGroup as exc:
        # SDK task groups wrap HTTP failures. Give the UI the actual cause,
        # rather than the otherwise unhelpful "unhandled errors" wrapper.
        detail = exc
        while isinstance(detail, BaseExceptionGroup):
            detail = detail.exceptions[0]
        raise RuntimeError("Parallel Search MCP: " + str(detail)) from exc
