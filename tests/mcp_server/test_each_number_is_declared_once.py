"""A number the model is told is the number the code uses, and a measurement
lives beside the code it justifies, not in a tool description.

(This file also tied the 6000-character cap of `browser_read_text` to its
description and schema until 0.71; the cap is gone since 0.72.0, and
`test_answers_come_back_whole.py` holds that it stays gone.)
"""
from __future__ import annotations

import asyncio
import re

from invisible_playwright_mcp.mcp import actions, server


def tools() -> dict:
    return {t.name: t for t in asyncio.run(server.mcp.list_tools())}


def test_a_measurement_is_not_repeated_in_a_tool_description():
    """⛔ THE EVIDENCE BELONGS BESIDE THE CODE IT JUSTIFIES, NOT IN THE PROMPT.

    `browser_snapshot` carried the 958-element selector study and the
    two-hundred-option count, both of which are also written in `actions.py`
    beside the code they explain. A measurement in two places is one that gets
    re-run once and updated once, and the copy left wrong is the one a model
    reads - out of the 1024 characters the description is cut at.

    Held as a property rather than on those two numbers: no tool description
    carries a figure that reads as a study. The rules those paragraphs stated
    are checked below, so this cannot pass by saying less.

    Known-bad: put either measurement back into a description.
    """
    guilty = []
    for name, tool in tools().items():
        text = tool.description or ""
        for m in re.finditer(r"\b(\d{2,4})%|\bacross (\d{3,})\b|\b(\d{3,}) elements\b",
                             text):
            guilty.append("%s: %s" % (name, m.group(0)))
    assert not guilty, (
        "a description carries evidence rather than a rule, which belongs in "
        "the module beside the code it justifies: %s" % guilty)


def test_the_snapshot_still_states_every_rule_it_used_to():
    """The half that stops the cut above from being a loss. Dropping the
    numbers must not drop what a caller has to DO.

    Known-bad: delete any of these sentences along with the evidence.
    """
    text = (tools()["browser_snapshot"].description or "").lower()
    for rule, why in (
            ("verbatim", "pass the selector as given"),
            ("one element", "it is built to reach exactly one"),
            ("first match", "the driver takes the first when it does not"),
            ("`at`", "elements with no selector carry coordinates"),
            ("accessibility tree", "this is not that tree, and why"),
    ):
        assert rule in text, (
            "the snapshot no longer tells a caller that %s (%r missing)"
            % (why, rule))


def test_the_measurements_are_still_written_where_the_code_is():
    """The other half: moved, not deleted. A reader of `actions.py` still finds
    why the selector is built and why the inventory is not the a11y tree.

    Known-bad: delete them from `actions.py` too, and the reason is nowhere.
    """
    source = open(actions.__file__, encoding="utf-8").read()
    assert "958" in source, "the selector study is gone from the module too"
    assert "two hundred" in source, (
        "the option-count measurement is gone from the module too")
