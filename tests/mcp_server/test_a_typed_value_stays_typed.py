"""browser_type waits for the page to answer the focus, and says what the field kept.

Measured on a real form (Angular, a store whose every emission writes the
stored answer into its inputs): "30" typed into a field was gone a second
later, and an email typed after a masked date kept only its last letters, and
both calls answered "typed into". Two causes. The engine's `fill` focuses and
presses the first key in the same breath, so the page's answer to the focus
lands on top of what is arriving; and the answer was a claim about the call,
never a reading of the field.

The first fix retyped whatever looked dropped, and that is not idempotent: a
chip field that turns `a@b.io,` into a chip took the text the first time, and a
retry made a second chip. It also called a one-time code spread over six boxes
an error, because the first box keeps one digit by design. So nothing is
retyped; each shape is named for what it is.

The pause before the first key is the engine's own `fill` since
invisible-playwright 0.25.8, for every caller; this server used to add it in
front of `fill` and adds nothing now. The unit tests hold the sentences and
that the server adds no pause of its own; the e2e ones hold six real shapes -
an ordinary field, a field that keeps digits only, a maxlength, a code split
across boxes, a chip field and a store field - against a real engine.
"""
from __future__ import annotations

import asyncio
import http.server
import threading

import pytest

from invisible_playwright_mcp.mcp import actions


def _kept(text, value, focused=True, secret=False, was_secret=False, selector="#f"):
    return actions.what_the_field_kept(
        selector, text, {"value": "", "focused": True, "secret": was_secret},
        {"value": value, "focused": focused, "secret": secret})


# --- the sentences ------------------------------------------------------------

def test_a_field_that_kept_the_text_is_typed_into():
    assert _kept("30", "30") == "typed into #f"


def test_a_field_the_page_reformatted_says_how_it_shows_it():
    assert _kept("12ab34", "1234") == "typed into #f; the page shows it as '1234'"


def test_a_maxlength_is_said_as_a_cut_not_as_a_success():
    said = _kept("abcdef", "abcd")
    assert said.startswith("#f kept only the first 4 of 6 characters"), said
    assert "typed into" not in said


def test_a_code_split_across_boxes_is_the_page_moving_the_focus_not_an_error():
    """Known-bad, the first version: "kept the first 1 of 6" - a false error
    for the page doing exactly what a code group does."""
    said = _kept("482913", "4", focused=False)
    assert said.startswith("typed into #f until the page moved the focus"), said
    assert "1 of 6" in said and "maxlength" not in said


def test_a_field_the_page_emptied_is_said_and_typing_again_is_warned_against():
    """A chip field empties itself on the comma, having taken the text. Known-
    bad: typing it again on its own, which made a second chip."""
    said = _kept("a@b.io,", "")
    assert said.startswith("#f is empty after typing"), said
    assert "twice" in said


def test_a_field_that_lost_its_start_says_so():
    said = _kept("someone@example.com", "eone@example.com")
    assert said.startswith("#f holds only the last 16 of 19 characters"), said


def test_a_secret_is_never_echoed_even_when_the_page_changed_it():
    assert "hunter" not in _kept("hunter2", "hunter2x", secret=True)
    # A password box the page turned into text while it was typed is still a
    # secret: the reading BEFORE typing says so.
    assert "hunter" not in _kept("hunter2", "hunter2x", was_secret=True)


def test_a_target_with_no_readable_text_says_it_cannot_be_read_back():
    said = _kept("x", None)
    assert "cannot be read back" in said or "can be read back" in said


# --- the pause --------------------------------------------------------------

def test_the_server_adds_no_pause_of_its_own_in_front_of_fill(monkeypatch):
    """The engine's `fill` waits the typist's pause between the focus and the
    first key; a second one here would be a hesitation too many. Known-bad, the
    version before: a `focus`, a pause drawn from private names of the wrapper,
    then `fill`."""
    calls = []

    class _First:
        async def evaluate(self, js, timeout=None):
            calls.append("read")
            return {"value": "", "focused": False, "secret": False}

    class _Page:
        def locator(self, selector):
            return type("L", (), {"first": _First()})()

        async def focus(self, *a, **kw):
            calls.append("focus")

        async def fill(self, selector, text, timeout=None):
            calls.append("fill")

    class _Session:
        seed = 4242

        def page(self):
            return _Page()

    async def slept(s):
        calls.append("sleep")

    monkeypatch.setattr(actions.asyncio, "sleep", slept)
    asyncio.run(actions.type_text(_Session(), "#f", "30"))
    assert calls == ["read", "fill", "read"], calls


# --- against a real engine -------------------------------------------------

PAGE = b"""<!doctype html><html><body>
<input id="normal">
<input id="digits">
<input id="short" maxlength="4">
<div id="code">
  <input id="c1" maxlength="1"><input id="c2" maxlength="1"><input id="c3" maxlength="1">
  <input id="c4" maxlength="1"><input id="c5" maxlength="1"><input id="c6" maxlength="1">
</div>
<div id="chips"></div><input id="chip">
<input id="store">
<script>
window.__events = [];
for (const k of ['focus', 'keydown', 'compositionstart'])
  document.addEventListener(k, e => __events.push([k, e.target.id, performance.now()]), true);
// Keeps digits only, as a phone or a card field does.
digits.addEventListener('input', () => { digits.value = digits.value.replace(/[^0-9]/g, ''); });
// A code split across boxes: each box takes one digit and passes the focus on.
const boxes = [...document.querySelectorAll('#code input')];
boxes.forEach((b, i) => b.addEventListener('input', () => {
  if (b.value && boxes[i + 1]) boxes[i + 1].focus();
}));
// A chip field: a comma turns what is typed into a chip and empties the field.
chip.addEventListener('input', () => {
  if (chip.value.endsWith(',')) {
    const c = document.createElement('span'); c.className = 'chip';
    c.textContent = chip.value.slice(0, -1); chips.appendChild(c); chip.value = '';
  }
});
// A store field: it answers the focus by writing its stored answer, empty,
// back into the input after `answer` ms.
const answer = +(new URLSearchParams(location.search).get('answer') || 900);
store.addEventListener('focus', () => setTimeout(() => { store.value = ''; }, answer));
</script></body></html>"""


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


def _seed_whose_first_pause(longer_than=None, shorter_than=None):
    """A seed whose first pause is known, so the store tests assert the
    mechanism instead of depending on a draw. The engine draws it: the first
    field typed into on a page is its act "field", nonce 1; read from the
    client's internals here (invisible-core since 38.34.0) only to know
    what to expect."""
    from invisible_core.juggler._behaviour import TypingPersona, plan_hesitation

    for seed in range(1, 5000):
        p = plan_hesitation(TypingPersona.from_seed(seed), "field", 1) / 1000.0
        if (longer_than is None or p > longer_than) and (shorter_than is None or p < shorter_than):
            return seed, p
    raise AssertionError("no seed found")


async def _with_browser(url, body, seed=None):
    from invisible_playwright_mcp.mcp.plan import plan_session
    from invisible_playwright_mcp.mcp.session import StealthSession
    session = StealthSession(**plan_session(seed=seed).kwargs)
    await session.start()
    try:
        await actions.navigate(session, url)
        return await body(session)
    finally:
        await session.close()


def _type(url, selector, text, seed=None, extra=None):
    async def body(session):
        said = await actions.type_text(session, selector, text)
        page = session.page()
        got = await page.evaluate(
            "(sel) => ({value: document.querySelector(sel).value,"
            " codes: [...document.querySelectorAll('#code input')].map(b => b.value).join(''),"
            " chips: [...document.querySelectorAll('.chip')].map(c => c.textContent),"
            " events: __events})", selector)
        return said, got
    return asyncio.run(_with_browser(url + (extra or ""), body, seed=seed))


@pytest.mark.e2e
def test_an_ordinary_field_is_typed_into_after_a_pause_a_person_takes(url):
    seed, pause = _seed_whose_first_pause(longer_than=0.3)
    said, got = _type(url, "#normal", "plain words", seed=seed)
    assert said == "typed into #normal" and got["value"] == "plain words"
    focus = next(t for k, i, t in got["events"] if k == "focus" and i == "normal")
    first = next(t for k, i, t in got["events"] if k == "keydown")
    assert (first - focus) / 1000 >= pause * 0.9, (
        "the first key came %.0f ms after the focus, before this session's "
        "pause of %.0f ms" % (first - focus, pause * 1000))
    assert not any(k == "compositionstart" for k, _, _ in got["events"])


@pytest.mark.e2e
def test_a_digits_only_field_is_said_as_the_page_shows_it(url):
    said, got = _type(url, "#digits", "12ab34")
    assert got["value"] == "1234"
    assert said == "typed into #digits; the page shows it as '1234'"


@pytest.mark.e2e
def test_a_maxlength_is_said_as_a_cut(url):
    said, got = _type(url, "#short", "abcdef")
    assert got["value"] == "abcd"
    assert said.startswith("#short kept only the first 4 of 6 characters"), said


@pytest.mark.e2e
def test_a_code_split_across_boxes_lands_whole_and_is_not_an_error(url):
    said, got = _type(url, "#c1", "482913")
    assert got["codes"] == "482913"
    assert said.startswith("typed into #c1 until the page moved the focus"), said


@pytest.mark.e2e
def test_a_chip_field_gets_one_chip_and_the_answer_says_where_the_text_went(url):
    said, got = _type(url, "#chip", "a@b.io,")
    assert got["chips"] == ["a@b.io"], "a retry made a second chip: %r" % got["chips"]
    assert said.startswith("#chip is empty after typing"), said


@pytest.mark.e2e
def test_a_store_that_answers_the_focus_within_the_pause_keeps_the_text(url):
    """The measured shape: the page writes its stored answer 900 ms after the
    focus. A person who pauses longer than that types after it, and so does
    this session's hand."""
    seed, _ = _seed_whose_first_pause(longer_than=1.2)
    said, got = _type(url, "#store", "30", seed=seed)
    assert got["value"] == "30" and said == "typed into #store", (said, got["value"])


@pytest.mark.e2e
def test_a_store_that_answers_after_the_pause_is_said_and_not_retyped(url):
    """Nothing the page shows says a timer is about to fire, so a pause cannot
    always cover it. Then the answer says what is left, and the text is typed
    once: every key went in exactly one time."""
    seed, pause = _seed_whose_first_pause(shorter_than=0.8)
    text = "someone@example.com"
    said, got = _type(url, "#store", text, seed=seed, extra="?answer=%d" % int(pause * 1000 + 1500))
    keys = sum(1 for k, i, _ in got["events"] if k == "keydown")
    assert keys == len(text), "typed %d keys for %d characters" % (keys, len(text))
    assert got["value"] != text
    assert said.startswith("#store holds only the last") or said.startswith("#store is empty"), said
