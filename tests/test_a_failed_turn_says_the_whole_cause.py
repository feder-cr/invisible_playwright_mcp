"""A turn that fails says the whole cause.

The sentence keeps the detail of the exception because it is the only clue
when the cause is real (a provider's error, a broken link). Until 0.72.1 that
detail stopped at 200 characters, and a provider's JSON error is longer than
that: the field naming the cause was in the part that was cut. Known-bad: cut
the detail again.
"""
from __future__ import annotations

from invisible_playwright_mcp.chat import ChatService

from test_web_service import FakeLink

CAUSE = ('{"error": {"message": "Provider returned error", "code": 400, '
         '"metadata": {"raw": "' + "x" * 300 + '", "provider_name": "SomeProvider", '
         '"reason": "the request does not fit this provider"}}}')


class _FailingBrain:
    messages: list = []
    usage: dict = {}

    async def handle(self, text, link, say):
        raise RuntimeError(CAUSE)


async def test_a_failed_turn_says_the_whole_cause():
    svc = ChatService(FakeLink(), _FailingBrain(), model_label="a model")
    await svc.send("go")
    errors = [e["text"] for e in svc.history if e["kind"] == "err"]
    assert errors, "the failed turn said nothing: %r" % svc.history
    assert CAUSE in errors[-1], (
        "the cause was cut: %r" % errors[-1][-80:])
    assert "the request does not fit this provider" in errors[-1]
