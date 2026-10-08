"""Controls inside open shadow roots are reported and reachable.

Measured on 2026-09-30 against a real contact form built from custom
elements: `<x-textfield id="firstName">` renders its `<input>` inside an open
shadow root, and so do its `<x-select>` and its option tiles.
`browser_snapshot` walked `document.querySelectorAll`, which stops at every
shadow boundary, so it listed two radios and a privacy link and none of the
fields. The caller wrote `#firstName input` - which Playwright's CSS engine
does pierce - and `browser_type` then waited thirty seconds and reported
`{"matches": 0}`, because the diagnosis also searched with
`document.querySelectorAll` and so contradicted the engine it was explaining.

These run the real functions in a real engine against a page whose only
controls live in shadow roots, nested and not, and one in a CLOSED root, which
a page script cannot reach either and must not be claimed.
"""
from __future__ import annotations

import asyncio
import json
import time

import pytest

from invisible_playwright_mcp.mcp import actions

PAGE = b"""<!doctype html>
<html><head><title>shadow form</title></head><body>
<script>
  class Field extends HTMLElement {
    constructor() {
      super();
      const root = this.attachShadow({mode: 'open'});
      root.innerHTML = '<label for="input">' + this.getAttribute('label') +
        '</label><input id="input" type="text">';
    }
  }
  customElements.define('x-field', Field);
  class Pick extends HTMLElement {
    constructor() {
      super();
      const root = this.attachShadow({mode: 'open'});
      root.innerHTML = '<select id="select"><option value="">-</option>' +
        '<option value="TX">Texas</option><option value="WA">Washington</option></select>';
    }
  }
  customElements.define('x-pick', Pick);
  // A field inside a component inside a component: two boundaries.
  class Card extends HTMLElement {
    constructor() {
      super();
      const root = this.attachShadow({mode: 'open'});
      root.innerHTML = '<x-field id="inner" label="Nested"></x-field>' +
        '<button id="go" type="button">Card button</button>';
    }
  }
  customElements.define('x-card', Card);
  class Sealed extends HTMLElement {
    constructor() {
      super();
      this.attachShadow({mode: 'closed'}).innerHTML = '<input id="sealed-input">';
    }
  }
  customElements.define('x-sealed', Sealed);
  // Three buttons with one handle under one host: A and B in its shadow root,
  // C inside a component nested between them. In tree order C sits between A
  // and B; the engine lists a root's own matches before the roots nested in
  // it, so it says A, B, C. Which of them is `nth=1` is the engine's to say.
  class Deep extends HTMLElement {
    constructor() {
      super();
      this.attachShadow({mode: 'open'}).innerHTML =
        '<button aria-label="Same" type="button" ' +
        'onclick="document.title=\\'C clicked\\'">C</button>';
    }
  }
  customElements.define('x-deep', Deep);
  class Pair extends HTMLElement {
    constructor() {
      super();
      this.attachShadow({mode: 'open'}).innerHTML =
        '<button aria-label="Same" type="button" ' +
        'onclick="document.title=\\'A clicked\\'">A</button><x-deep></x-deep>' +
        '<button aria-label="Same" type="button" ' +
        'onclick="document.title=\\'B clicked\\'">B</button>';
    }
  }
  customElements.define('x-pair', Pair);
</script>
<x-pair id="pair"></x-pair>
<x-field id="firstName" label="First Name"></x-field>
<x-field id="lastName" label="Last Name"></x-field>
<x-pick id="state"></x-pick>
<x-card id="card"></x-card>
<x-sealed id="sealed"></x-sealed>
<input id="plain" name="plain" type="text">
<select id="size" name="size"><option value="s">Small</option><option value="l">Large</option></select>
<div id="slot"></div>
<script>
  // Added by a timer and nothing else: no mutation, request or navigation
  // happens before it, so nothing the page shows says it is coming.
  setTimeout(() => {
    const b = document.createElement('button');
    b.id = 'late'; b.type = 'button'; b.textContent = 'Late';
    b.addEventListener('click', () => { document.title = 'late clicked'; });
    document.getElementById('slot').appendChild(b);
  }, 5000);
</script>
</body></html>"""


def _serve():
    import http.server
    import socket
    import threading

    class H(http.server.BaseHTTPRequestHandler):
        def do_GET(self):
            self.send_response(200)
            self.send_header("Content-Type", "text/html; charset=utf-8")
            self.send_header("Content-Length", str(len(PAGE)))
            self.end_headers()
            self.wfile.write(PAGE)

        def log_message(self, *a):
            pass

    s = socket.socket()
    s.bind(("127.0.0.1", 0))
    port = s.getsockname()[1]
    s.close()
    srv = http.server.HTTPServer(("127.0.0.1", port), H)
    threading.Thread(target=srv.serve_forever, daemon=True).start()
    return srv, "http://127.0.0.1:%d/" % port


@pytest.fixture(scope="module")
def run():
    """One browser for the file; each test hands in a coroutine to run on it."""
    from invisible_playwright_mcp.mcp.plan import plan_session
    from invisible_playwright_mcp.mcp.session import StealthSession

    srv, url = _serve()
    loop = asyncio.new_event_loop()
    session = StealthSession(**plan_session().kwargs)
    loop.run_until_complete(session.start())

    def go(fn):
        async def body():
            await actions.navigate(session, url)
            return await fn(session)
        return loop.run_until_complete(body())

    try:
        yield go
    finally:
        loop.run_until_complete(session.close())
        loop.close()
        srv.shutdown()


def _elements(run):
    async def snap(s):
        return json.loads(await actions.snapshot(s))["interactive_elements"]
    return run(snap)


def _by_selector(elements):
    return {e.get("selector"): e for e in elements}


@pytest.mark.e2e
def test_the_snapshot_lists_controls_inside_open_shadow_roots(run):
    sels = _by_selector(_elements(run))
    for wanted in ("#firstName >> #input", "#lastName >> #input",
                   "#state >> #select", "#card >> #inner >> #input",
                   "#card >> #go", "#plain"):
        assert wanted in sels, "missing %s; got %s" % (wanted, sorted(map(str, sels)))
    assert sels["#state >> #select"]["tag"] == "select"


@pytest.mark.e2e
def test_two_alike_under_one_host_each_get_the_selector_that_reaches_them(run):
    """Known-bad until 0.70.13: the `nth=` of an ambiguous selector inside a
    shadow root came from a hand copy of the engine's search order. Now the
    engine says the order; each selector must click its own button."""
    async def body(s):
        snap = json.loads(await actions.snapshot(s))["interactive_elements"]
        pair = {e["text"]: e.get("selector") for e in snap
                if str(e.get("selector", "")).startswith("#pair >> ")}
        got = {}
        for label, sel in pair.items():
            await actions.click(s, sel)
            got[label] = await s.page().title()
        return pair, got
    pair, got = run(body)
    assert set(pair) == {"A", "B"}, pair
    assert all(" >> nth=" in sel for sel in pair.values()), pair
    assert got == {"A": "A clicked", "B": "B clicked"}, (pair, got)


async def test_the_position_is_the_engines_not_the_documents():
    """`_resolve_nth` takes `k` from the order the engine reports, whatever
    order the document has; an element the engine does not find loses its
    selector rather than getting a wrong one."""
    class Locator:
        def __init__(self, order):
            self.order = order

        async def evaluate_all(self, js):
            assert "pathOf" in js
            return self.order

    class Page:
        def locator(self, sel):
            return Locator(["0/s/3", "1/s/3"] if sel == "#h >> [aria-label='x']" else [])

    elements = [{"selector": "#h >> [aria-label='x']", "_nth_path": "1/s/3"},
                {"selector": "#h >> [aria-label='x']", "_nth_path": "0/s/3"},
                {"selector": "#gone >> [aria-label='y']", "_nth_path": "2/s/1"}]
    await actions._resolve_nth(Page(), elements)
    assert elements[0] == {"selector": "#h >> [aria-label='x'] >> nth=1"}
    assert elements[1] == {"selector": "#h >> [aria-label='x'] >> nth=0"}
    assert elements[2] == {}


@pytest.mark.e2e
def test_a_closed_root_is_not_claimed(run):
    """A closed root is closed to the page's own scripts too; reporting a
    selector into it would be reporting one the engine cannot use."""
    assert not any("sealed" in str(e.get("selector")) for e in _elements(run))


@pytest.mark.e2e
def test_every_shadow_selector_the_snapshot_gives_can_be_used(run):
    async def body(s):
        out = {}
        for sel in ("#firstName >> #input", "#card >> #inner >> #input"):
            await actions.type_text(s, sel, "Plain")
            out[sel] = await s.page().locator(sel).input_value()
        await actions.click(s, "#card >> #go")
        return out
    assert run(body) == {"#firstName >> #input": "Plain",
                         "#card >> #inner >> #input": "Plain"}


@pytest.mark.e2e
def test_a_select_inside_a_shadow_root_can_be_set(run):
    """Up to firefox-34 this could not be done: the driver set the option
    from the page and asked the engine for trusted `input`/`change`
    (`Page.dispatchTrustedInputEvents`), which answered NS_ERROR_UNEXPECTED
    for any node inside a shadow tree. firefox-35 selects through the
    dropdown's own path (`Page.selectOptions`), which a shadow root does not
    stop; the test was a strict xfail until the engine moved."""
    async def body(s):
        await actions.select_option(s, "#state >> #select", "Texas")
        return await s.page().locator("#state >> #select").input_value()
    assert run(body) == "TX"


@pytest.mark.e2e
def test_a_descendant_selector_across_the_boundary_works_too(run):
    """What the caller wrote on the real page. Playwright's CSS pierces open
    roots, so this is reachable and must not be refused."""
    async def body(s):
        await actions.type_text(s, "#firstName input", "Words")
        return await s.page().locator("#firstName >> #input").input_value()
    assert run(body) == "Words"


@pytest.mark.e2e
def test_an_element_a_timer_adds_later_is_still_reached(run):
    """The regression a short presence wait caused, measured: this button,
    added five seconds after load, was clicked at 5.4 s with the full action
    timeout and refused at 3.1 s with a three-second wait. Nothing on the page
    announces it, so no shorter wait can be right for it."""
    async def body(s):
        t = time.monotonic()
        said = await actions.click(s, "#late")
        return time.monotonic() - t, said, await s.page().title()
    took, said, title = run(body)
    assert said == "clicked #late" and title == "late clicked"
    assert took >= 4.5, "the button was there before the timer, the test proves nothing"


@pytest.mark.e2e
def test_a_selector_the_engine_cannot_parse_fails_at_once(run):
    """The early answer that is certain: the engine refuses the syntax, and
    the diagnosis says so instead of calling it a missing element."""
    async def body(s):
        t = time.monotonic()
        with pytest.raises(RuntimeError) as err:
            await actions.click(s, "#plain[")
        return time.monotonic() - t, str(err.value)
    took, message = run(body)
    assert '"bad_selector": true' in message, message
    assert took < 3, "a selector that cannot be parsed took %.1fs to refuse" % took


@pytest.mark.e2e
def test_a_select_is_set_by_the_label_it_shows_in_one_attempt(run):
    """`value=` reaches the driver as value-or-label, so the label a page
    shows is matched without a second attempt."""
    async def body(s):
        said = await actions.select_option(s, "#size", "Large")
        return said, await s.page().locator("#size").input_value()
    assert run(body) == ("selected #size by label: ['l']", "l")
