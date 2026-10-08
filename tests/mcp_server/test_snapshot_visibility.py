"""The snapshot must see what is there and skip what is not.

Measured 2026-09-02 against the version that used `el.offsetParent !== null`: on
a page with five elements it reported three, and it was wrong in BOTH
directions.

  button position:fixed    visible     -> SKIPPED
  visibility:hidden        invisible   -> KEPT
  left:-9999px             invisible   -> KEPT
  div[role=button]         clickable   -> NEVER LOOKED FOR

`offsetParent` is `null` on any `position:fixed` element, and that is not a
laboratory configuration: it is the cookie banner, the sticky bar, the button
inside a modal. When it is the modal that blocks the page, the model does not
see it at all, so the failure is not local, it is terminal.

These tests run against the filter in isolation, without a browser: the logic
lives in a JS string, and to exercise it here it is checked against the shape of
the data that string produces. The end-to-end check with a real browser is in
test_real_launch.py.
"""
import re

import pytest

from invisible_playwright_mcp.mcp import actions


def _code(js: str) -> str:
    """The JS with its comments removed.

    Needed because the comment explaining the incident NAMES `offsetParent`, and
    it is right to name it: it says why that line must not come back. A test
    that read the comments too would go red over the documentation of the very
    defect it protects. It is the same mistake this project has hit several
    times - writing the check against the comment instead of the code - only
    inverted.
    """
    return re.sub(r"//[^\n]*", "", js)


def test_the_filter_no_longer_asks_for_offsetparent():
    """The line that caused the defect must not come back."""
    assert "offsetParent" not in _code(actions.SNAPSHOT_JS), (
        "offsetParent is back in the snapshot: it skips every position:fixed "
        "element, meaning cookie banners, sticky bars and modal buttons"
    )


def test_the_filter_looks_at_what_actually_decides_visibility():
    js = actions.SNAPSHOT_JS
    for expected in ("getBoundingClientRect", "visibility", "display"):
        assert expected in js, f"the snapshot does not look at {expected}"


def test_the_query_reaches_elements_that_are_not_form_tags():
    """Half the buttons on the web are not `<button>`.

    A `div` with `role=button` and a click handler is as clickable as a button,
    and the closed list of tags did not look for it at all.
    """
    js = actions.SNAPSHOT_JS
    assert 'role="button"' in js or "role='button'" in js or "[role=" in js
    assert "onclick" in js
    assert "tabindex" in js


def test_the_snapshot_still_reports_the_fields_a_caller_needs():
    js = actions.SNAPSHOT_JS
    for field in ("tag", "text", "title", "url", "interactive_elements"):
        assert field in js


def test_the_snapshot_does_not_write_to_the_page():
    """Read only, and not as a matter of taste.

    Injecting an attribute to number the elements would mutate the DOM, which
    creates a detection surface inside a product that exists not to have one. If
    a stable index is ever wanted, that choice gets made in the open rather than
    slipped in here.
    """
    js = actions.SNAPSHOT_JS
    for write in ("setAttribute", "dataset.", "innerHTML =", "classList.add"):
        assert write not in js, f"the snapshot writes to the page: {write}"


def test_visibility_rules_are_expressed_once_each():
    """A rule written twice drifts. This is a shape check on the filter being
    one function rather than copied per branch.

    Counted on the CODE. The first version counted the raw string and went red
    the day a comment explained why getBoundingClientRect needs a guard: four
    occurrences, of which two were prose. Every other check in this file already
    strips comments for the same reason, and this one was the exception that
    proved it mattered.
    """
    js = _code(actions.SNAPSHOT_JS)
    assert js.count("getBoundingClientRect") <= 2, (
        f"the rectangle is read {js.count('getBoundingClientRect')} times in "
        "code; the filter is being copied per branch")


def test_the_snapshot_does_not_deduplicate():
    """Measured on the same DOM: deduplicating by text removed 8.1% of the
    elements in exchange for 13% of the weight.

    Two buttons with the same text and no id are not a duplicate: they are two
    different places on the screen, and on a results page they are "add to cart"
    repeated once per product. Hiding one from the model is the character cap
    wearing a different name.
    """
    js = _code(actions.SNAPSHOT_JS)
    for tell in ("doppioni", "firma", "dedup", "signature"):
        assert tell not in js, f"a deduplication is back: {tell}"


# --- a transparent control with a shown label -------------------------------
#
# Measured 2026-10-01 on Vuetify 3.7.5: a labelled, unfocused v-text-field keeps
# its <input> at opacity 0, full size and enabled, and paints the floating
# <label for=...> instead. The snapshot dropped every opacity-0 element, so a
# page with one such field reported `interactive_elements: []`, while typing
# into the same input filled it. The page below reproduces that markup without
# Vuetify, so it runs offline.

LABELLED_PAGE = b"""<!doctype html>
<html><head><title>labelled</title>
<style>
  .field { position: relative; width: 400px; height: 56px; margin: 8px }
  .field label { position: absolute; left: 12px; top: 16px }
  .field input { position: absolute; inset: 0; width: 100%; opacity: 0 }
  .ghost, .field label.ghost { opacity: 0 }
  .ghost-wrap { opacity: 0 }
  .field label.away { left: -9999px }
</style>
</head><body>
  <div class="field">
    <label for="code">Code</label>
    <input id="code" name="code" type="text">
  </div>
  <div class="field">
    <label>Wrapped <input id="wrapped" name="wrapped" type="text"></label>
  </div>

  <input id="honeypot-bare" name="honeypot-bare" class="ghost" type="text">
  <div class="field">
    <label for="honeypot-faded" class="ghost">Leave empty</label>
    <input id="honeypot-faded" name="honeypot-faded" type="text">
  </div>
  <div class="field">
    <span class="ghost-wrap"><label for="honeypot-wrapped">Leave empty</label></span>
    <input id="honeypot-wrapped" name="honeypot-wrapped" type="text">
  </div>
  <div class="field">
    <label for="honeypot-away" class="away">Leave empty</label>
    <input id="honeypot-away" name="honeypot-away" type="text">
  </div>
  <div class="field">
    <label for="honeypot-blank"></label>
    <input id="honeypot-blank" name="honeypot-blank" type="text">
  </div>

  <div id="ghost-button" class="ghost" role="button" tabindex="0"
       style="width:100px;height:30px">GHOST</div>
</body></html>"""

#: Shown labels name these, so a person can see what they are.
LABELLED = ("code", "wrapped")
#: Transparent with no label a person can see: stays out.
UNLABELLED = ("honeypot-bare", "honeypot-faded", "honeypot-wrapped",
              "honeypot-away", "honeypot-blank", "ghost-button")


def test_the_opacity_rule_has_one_exception_and_it_is_shared():
    """The snapshot and the visible-HTML clone answer the same question about
    the same element, so they carry one copy of the exception, not two."""
    from invisible_playwright_mcp.mcp import clean

    assert clean.LABELLED_CONTROL_JS in actions.SNAPSHOT_JS
    assert clean.LABELLED_CONTROL_JS in clean.VISIBLE_HTML_JS
    for js in (_code(actions.SNAPSHOT_JS), _code(clean.VISIBLE_HTML_JS)):
        assert "transparent(s) && !labelledControl(el)" in js


def test_every_script_that_judges_visibility_shares_one_style_rule():
    """Known-bad: four definitions of "visible", and the snapshot called opacity
    0.005 visible where the upload called it hidden. Each script joins
    STYLE_HIDES_JS, and none compares opacity on its own."""
    import re

    from invisible_playwright_mcp.mcp import clean

    scripts = {"SNAPSHOT_JS": actions.SNAPSHOT_JS, "VISIBLE_HTML_JS": clean.VISIBLE_HTML_JS,
               "_FILE_INPUT_JS": actions._FILE_INPUT_JS, "_OPENER_JS": actions._OPENER_JS}
    for name, js in scripts.items():
        assert clean.STYLE_HIDES_JS in js, "%s has its own idea of visible" % name
        own = _code(js.replace(clean.STYLE_HIDES_JS, ""))
        assert not re.search(r"\.opacity\b", own), "%s compares opacity outside the shared rule" % name


def test_the_exception_is_only_for_form_fields():
    """Buttons and links have their own text; a transparent one is hidden."""
    from invisible_playwright_mcp.mcp import clean

    assert "^(input|select|textarea)$" in _code(clean.LABELLED_CONTROL_JS)


def _serve(page):
    import http.server
    import socket
    import threading

    class H(http.server.BaseHTTPRequestHandler):
        def do_GET(self):
            self.send_response(200)
            self.send_header("Content-Type", "text/html; charset=utf-8")
            self.send_header("Content-Length", str(len(page)))
            self.end_headers()
            self.wfile.write(page)

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
def labelled_page():
    """One browser, the snapshot and the form-mode HTML of the same page."""
    import asyncio
    import json

    from invisible_playwright_mcp.mcp.plan import plan_session
    from invisible_playwright_mcp.mcp.session import StealthSession

    srv, url = _serve(LABELLED_PAGE)

    async def run():
        # Through plan_session, the way production builds a session; see
        # test_read_html_in_a_browser.py for why a bare one fails on a runner.
        session = StealthSession(**plan_session().kwargs)
        try:
            await session.start()
            await actions.navigate(session, url)
            snap = json.loads(await actions.snapshot(session))
            html = await actions.read_html(session, "form")
            return snap, html
        finally:
            await session.close()

    try:
        yield asyncio.run(run())
    finally:
        srv.shutdown()


def _ids(snap):
    return {e.get("id") for e in snap["interactive_elements"]}


@pytest.mark.e2e
def test_a_transparent_field_with_a_shown_label_is_in_the_snapshot(labelled_page):
    snap, _ = labelled_page
    ids = _ids(snap)
    for want in LABELLED:
        assert want in ids, (
            "%s is drawn at opacity 0 behind a visible label and is typable, "
            "but the snapshot left it out: %r" % (want, sorted(i for i in ids if i)))


@pytest.mark.e2e
def test_a_transparent_field_with_no_shown_label_stays_out(labelled_page):
    """The other half, so the test above cannot pass by dropping the opacity
    rule altogether."""
    snap, _ = labelled_page
    ids = _ids(snap)
    for unwanted in UNLABELLED:
        assert unwanted not in ids, "%s has no label a person can see" % unwanted


@pytest.mark.e2e
def test_read_html_keeps_the_same_fields_and_drops_the_same_honeypots(labelled_page):
    _, html = labelled_page
    for want in LABELLED:
        assert 'id="%s"' % want in html, "read_html dropped %s" % want
    for unwanted in UNLABELLED:
        assert 'id="%s"' % unwanted not in html, "read_html kept %s" % unwanted
