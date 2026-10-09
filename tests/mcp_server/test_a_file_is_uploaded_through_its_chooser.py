"""browser_upload_files attaches local files the way a person picks them.

Measured 2026-10-01: a credit application's "Supporting Documents" page wanted
four PDFs, and no tool could answer the file chooser "Add Document" opened.
browser_evaluate rightly refuses to build the upload in script. The unit tests
hold the path guard and the shape of the action with a double; the e2e ones
hold the outcome against a real engine: a visible input, a styled button in
front of a hidden one, and a page that listens for a trusted change.
"""
from __future__ import annotations

import asyncio
import http.server
import os
import threading

import pytest

from invisible_playwright_mcp.mcp import actions


@pytest.fixture
def allowed(tmp_path):
    root = tmp_path / "up"
    root.mkdir()
    (root / "a.pdf").write_bytes(b"%PDF-1.4 a")
    (root / "b.pdf").write_bytes(b"%PDF-1.4 b")
    return root


def _env(*dirs):
    return {actions.UPLOAD_DIRS_ENV: os.pathsep.join(str(d) for d in dirs)}


def _link(link, target):
    """A link at `link` to `target`, made the way this system allows.

    A directory gets a symbolic link, or on Windows a junction when the account
    may not make symbolic links: a junction needs no privilege and is resolved
    by `realpath` the same way, so the guard is tested for real. A FILE link on
    such an account has no unprivileged equivalent, and only then is the test
    skipped, saying why.
    """
    try:
        link.symlink_to(target, target_is_directory=target.is_dir())
    except OSError as exc:
        if getattr(exc, "winerror", None) != 1314:
            raise
        if target.is_dir():
            import _winapi
            _winapi.CreateJunction(str(target), str(link))
        else:
            pytest.skip("a file symlink needs SeCreateSymbolicLinkPrivilege, which "
                        "this Windows account does not hold")


# --- the path guard -------------------------------------------------------------

def test_with_no_directory_named_uploads_are_off(allowed):
    with pytest.raises(RuntimeError, match="uploads are off"):
        actions.uploadable([str(allowed / "a.pdf")], env={})


def test_a_file_inside_a_named_directory_is_sent_resolved(allowed, tmp_path):
    link = allowed / "link.pdf"
    _link(link, allowed / "a.pdf")
    assert actions.uploadable([str(link)], env=_env(allowed)) == [
        os.path.realpath(allowed / "a.pdf")]


def test_a_file_outside_every_named_directory_is_refused(allowed, tmp_path):
    other = tmp_path / "secret.txt"
    other.write_text("x")
    with pytest.raises(PermissionError, match="not inside a directory"):
        actions.uploadable([str(other)], env=_env(allowed))


def test_a_link_out_of_the_directory_is_judged_by_where_it_lands(allowed, tmp_path):
    other = tmp_path / "secret.txt"
    other.write_text("x")
    _link(allowed / "innocent.pdf", other)
    with pytest.raises(PermissionError, match="not inside a directory"):
        actions.uploadable([str(allowed / "innocent.pdf")], env=_env(allowed))


def test_dot_dot_does_not_climb_out(allowed, tmp_path):
    other = tmp_path / "secret.txt"
    other.write_text("x")
    with pytest.raises(PermissionError):
        actions.uploadable([str(allowed / ".." / "secret.txt")], env=_env(allowed))


def test_a_hidden_directory_is_not_sent_from(allowed):
    (allowed / ".ssh").mkdir()
    (allowed / ".ssh" / "id_ed25519").write_text("key")
    (allowed / ".env").write_text("TOKEN=1")
    for path in (allowed / ".ssh" / "id_ed25519", allowed / ".env"):
        with pytest.raises(PermissionError, match="hidden"):
            actions.uploadable([str(path)], env=_env(allowed))


def test_what_is_not_a_readable_regular_file_is_refused(allowed):
    with pytest.raises(ValueError, match="not a regular file"):
        actions.uploadable([str(allowed)], env=_env(allowed))
    with pytest.raises(FileNotFoundError):
        actions.uploadable([str(allowed / "missing.pdf")], env=_env(allowed))


def test_relative_paths_and_bad_shapes_are_refused(allowed, monkeypatch):
    monkeypatch.chdir(allowed)
    with pytest.raises(ValueError, match="absolute"):
        actions.uploadable(["a.pdf"], env=_env(allowed))
    with pytest.raises(ValueError, match="list"):
        actions.uploadable(str(allowed / "a.pdf"), env=_env(allowed))
    with pytest.raises(ValueError, match="list"):
        actions.uploadable([], env=_env(allowed))
    with pytest.raises(ValueError, match="twice"):
        actions.uploadable([str(allowed / "a.pdf")] * 2, env=_env(allowed))
    with pytest.raises(RuntimeError, match="not an absolute path"):
        actions.uploadable([str(allowed / "a.pdf")], env={actions.UPLOAD_DIRS_ENV: "up"})


def test_a_root_that_is_a_symlink_turns_uploads_off(allowed, tmp_path):
    """Replacing the staging directory with a link to its parent must not
    widen uploads to the parent's whole tree."""
    link = tmp_path / "staging"
    _link(link, tmp_path)
    secret = tmp_path / "secret.txt"
    secret.write_text("x")
    with pytest.raises(RuntimeError, match="uploads are off"):
        actions.uploadable([str(secret)], env=_env(link))
    with pytest.raises(RuntimeError, match="uploads are off"):
        actions.uploadable([str(allowed / "a.pdf")], env=_env(allowed, link))


def test_a_root_that_does_not_exist_turns_uploads_off(tmp_path):
    with pytest.raises(RuntimeError, match="uploads are off"):
        actions.uploadable(["/x"], env=_env(tmp_path / "missing"))


windows_only = pytest.mark.skipif(
    os.name != "nt", reason="drive letters, case-blind paths and NTFS streams are Windows'")


@windows_only
def test_an_alternate_data_stream_is_not_sent_as_the_file(allowed):
    """`a.pdf:Zone.Identifier` opens, sizes and copies like a file, so the
    guard would send the stream under the file's name. Known-bad: drop
    `_no_stream`, and the stream written below is accepted."""
    stream = str(allowed / "a.pdf") + ":Zone.Identifier"
    with open(stream, "wb") as f:
        f.write(b"[ZoneTransfer]\r\nZoneId=3\r\n")
    with pytest.raises(ValueError, match="alternate data stream"):
        actions.uploadable([stream], env=_env(allowed))


@windows_only
def test_a_directory_named_in_another_case_is_the_same_directory(allowed):
    """`c:\\up` and `C:\\UP` are one directory on Windows. Known-bad, the
    first version: the root was "not a directory at that exact path" and
    uploads were off, or a file in it read as outside it."""
    flipped = str(allowed).swapcase()
    got = actions.uploadable([str(allowed / "a.pdf")], env={actions.UPLOAD_DIRS_ENV: flipped})
    assert os.path.normcase(got[0]) == os.path.normcase(os.path.realpath(allowed / "a.pdf"))
    got = actions.uploadable([str(allowed / "a.pdf").swapcase()], env=_env(allowed))
    assert len(got) == 1


@windows_only
def test_a_file_on_another_drive_is_outside_not_an_error(allowed):
    other = "Z:" + chr(92) + "a.pdf" if not str(allowed).upper().startswith("Z:") else "Y:" + chr(92) + "a.pdf"
    with pytest.raises(PermissionError, match="not inside"):
        actions.uploadable([other], env=_env(allowed))


def test_a_file_over_the_limit_is_refused(allowed, monkeypatch):
    monkeypatch.setattr(actions, "UPLOAD_MAX_BYTES", 4)
    with pytest.raises(ValueError, match="over the"):
        actions.uploadable([str(allowed / "a.pdf")], env=_env(allowed))


# --- the action, against a double -------------------------------------------------

class _Element:
    def __init__(self, page):
        self.page = page

    async def evaluate(self, js):
        return self.page.held


class _Chooser:
    def __init__(self, page, multiple):
        self.page, self.element, self._multiple = page, _Element(page), multiple

    def is_multiple(self):
        return self._multiple

    async def set_files(self, files, timeout=None):
        self.page.held = [os.path.basename(f) for f in files]
        self.page.calls.append(("chooser", files))


class _Page:
    """A page whose click opens a chooser or not, and whose target is `target`.

    `label` is what the page answers about a hidden input's labels."""

    def __init__(self, target, opens=True, label=None):
        self.target, self.opens, self.label = target, opens, label
        self.calls, self.held = [], None
        self._chooser = None

    async def eval_on_selector(self, selector, js):
        if js == actions._OPENER_JS:
            return self.label
        return self.held if "files" in js and "tagName" not in js else self.target

    def wait_for_event(self, event, timeout=None):
        assert event == "filechooser" and timeout == 0
        self._chooser = asyncio.get_running_loop().create_future()
        return self._chooser

    async def click(self, selector, timeout=None):
        self.calls.append(("click", selector))
        if self.opens:
            self._chooser.set_result(_Chooser(self, self.target["multiple"]))


class _Session:
    seed = None

    def __init__(self, page):
        self._page = page
        self.kept = []

    def page(self):
        return self._page

    def keep_until_closed(self, path):
        self.kept.append(path)


def _upload(page, paths, selector="#f", session=None):
    session = session or _Session(page)
    return asyncio.run(actions.upload_files(session, selector, paths))


@pytest.fixture
def env(allowed, tmp_path, monkeypatch):
    monkeypatch.setenv(actions.UPLOAD_DIRS_ENV, str(allowed))
    (tmp_path / "snapshots").mkdir()
    monkeypatch.setattr(actions.tempfile, "tempdir", str(tmp_path / "snapshots"))
    return allowed


def test_a_visible_input_is_clicked_and_its_chooser_answered(env):
    page = _Page({"file": True, "multiple": False, "shown": True})
    out = _upload(page, [str(env / "a.pdf")])
    assert [c[0] for c in page.calls] == ["click", "chooser"]
    assert out == "attached 1 file to #f (picked in the file chooser): a.pdf"


def test_a_button_in_front_of_a_hidden_input_opens_the_chooser(env):
    page = _Page({"file": False, "multiple": True, "shown": True})
    out = _upload(page, [str(env / "a.pdf"), str(env / "b.pdf")], selector="#add")
    assert [c[0] for c in page.calls] == ["click", "chooser"]
    assert out.startswith("attached 2 files to #add") and out.endswith("a.pdf, b.pdf")


@pytest.mark.parametrize("label,clicked", [
    ({"how": "for", "id": "f", "nth": 1, "of": 1}, 'label[for="f"]'),
    ({"how": "for", "id": "f", "nth": 2, "of": 3}, ':nth-match(label[for="f"], 2)'),
    ({"how": "wrap"}, "#f >> xpath=ancestor::label[1]"),
])
def test_a_hidden_input_is_opened_by_clicking_its_label(env, label, clicked):
    """Known-bad, the first version: the files were set on the hidden input
    directly, and the page heard `change` on a control no pointer touched."""
    page = _Page({"file": True, "multiple": False, "shown": False}, label=label)
    out = _upload(page, [str(env / "a.pdf")])
    assert page.calls[0] == ("click", clicked)
    assert [c[0] for c in page.calls] == ["click", "chooser"]
    assert ("its label %s opened" % clicked) in out


def test_a_hidden_input_with_no_label_is_refused_and_nothing_is_copied(env, tmp_path):
    page = _Page({"file": True, "multiple": False, "shown": False}, label=None)
    with pytest.raises(RuntimeError, match="no label on the page opens it"):
        _upload(page, [str(env / "a.pdf")])
    assert page.calls == []
    assert os.listdir(tmp_path / "snapshots") == []


def test_the_chooser_is_answered_through_the_standard_set_files(env):
    """The pause before the files arrive is the wrapper's, inside the standard
    `FileChooser.set_files`: the server draws none of its own, and reaches for
    nothing outside Playwright's contract (decision D82: the wrapper adds no
    helper for it). Known-bad: a pause or a helper import back in the server."""
    import inspect

    page = _Page({"file": True, "multiple": False, "shown": True})
    _upload(page, [str(env / "a.pdf")])
    assert [c[0] for c in page.calls] == ["click", "chooser"]
    source = inspect.getsource(actions)
    assert not hasattr(actions, "hesitation")
    assert "hesitation(" not in source
    assert "_behaviour" not in source


def test_the_copies_belong_to_the_browser_and_go_when_it_closes(env, tmp_path):
    page = _Page({"file": True, "multiple": False, "shown": True})
    session = _Session(page)
    _upload(page, [str(env / "a.pdf")], session=session)
    root, = session.kept
    assert os.path.dirname(root) == str(tmp_path / "snapshots")
    assert os.listdir(root)

    from invisible_playwright_mcp.mcp.session import StealthSession
    real = StealthSession()
    real.keep_until_closed(root)
    real._forget_kept()
    assert not os.path.exists(root)


def test_a_click_that_opens_no_chooser_says_what_to_select(env, monkeypatch):
    monkeypatch.setattr(actions, "ACTION_TIMEOUT_MS", 50)
    page = _Page({"file": False, "multiple": False, "shown": True}, opens=False)
    with pytest.raises(RuntimeError, match="opened no file chooser"):
        _upload(page, [str(env / "a.pdf")], selector="#nothing")


def test_several_files_for_a_single_input_attach_nothing(env):
    two = [str(env / "a.pdf"), str(env / "b.pdf")]
    page = _Page({"file": True, "multiple": False, "shown": True})
    with pytest.raises(RuntimeError, match="one at a time"):
        _upload(page, two)
    assert page.held is None and page.calls == []
    page = _Page({"file": True, "multiple": False, "shown": False},
                 label={"how": "wrap"})
    with pytest.raises(RuntimeError, match="one at a time"):
        _upload(page, two)
    assert page.calls == []


def test_a_refused_path_touches_no_page(env, tmp_path):
    page = _Page({"file": True, "multiple": False, "shown": True})
    (tmp_path / "x").write_text("x")
    with pytest.raises(PermissionError):
        _upload(page, [str(tmp_path / "x")])
    assert page.calls == []


# --- against a real engine ---------------------------------------------------------

PAGE = b"""<!doctype html><html><body>
<input id="one" type="file">
<input id="many" type="file" multiple>
<input id="hidden" type="file" multiple style="display:none">
<label for="behind" id="pick">Choose a file</label>
<input id="behind" type="file" style="display:none">
<button id="add" type="button" onclick="document.getElementById('hidden').click()">Add Document</button>
<button id="inert" type="button">Nothing</button>
<script>
// What a page sees: the change event, whether it was trusted, and the names.
window.seen = {};
window.order = [];
document.addEventListener("click", e => order.push(["click", e.target.id, e.isTrusted,
                                                    performance.now()]), true);
for (const id of ["one", "many", "hidden", "behind"]) {
  document.getElementById(id).addEventListener("change", e => {
    window.seen[id] = {trusted: e.isTrusted, names: Array.from(e.target.files, f => f.name),
                       sizes: Array.from(e.target.files, f => f.size)};
    order.push(["change", id, e.isTrusted, performance.now()]);
  });
}
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


async def _with_browser(url, body):
    from invisible_playwright_mcp.mcp.plan import plan_session
    from invisible_playwright_mcp.mcp.session import StealthSession
    session = StealthSession(**plan_session().kwargs)
    await session.start()
    try:
        await actions.navigate(session, url)
        return await body(session)
    finally:
        await session.close()


@pytest.mark.e2e
def test_files_reach_the_page_with_a_trusted_change(url, env):
    a, b = str(env / "a.pdf"), str(env / "b.pdf")

    async def body(session):
        outs = [await actions.upload_files(session, "#one", [a]),
                await actions.upload_files(session, "#many", [a, b]),
                await actions.upload_files(session, "#add", [b])]
        with pytest.raises(RuntimeError, match="opened no file chooser"):
            await actions.upload_files(session, "#inert", [a])
        return outs, await session.page().evaluate("JSON.stringify(seen)")

    outs, seen = asyncio.run(_with_browser(url, body))
    import json
    seen = json.loads(seen)
    assert outs[0] == "attached 1 file to #one (picked in the file chooser): a.pdf"
    assert outs[1].startswith("attached 2 files to #many (picked in the file chooser)")
    assert outs[2] == "attached 1 file to #add (picked in the file chooser): b.pdf"
    assert seen["one"]["names"] == ["a.pdf"] and seen["one"]["sizes"] == [10]
    assert seen["many"]["names"] == ["a.pdf", "b.pdf"]
    assert seen["hidden"]["names"] == ["b.pdf"]
    assert all(v["trusted"] for v in seen.values()), seen


@pytest.mark.e2e
def test_a_hidden_input_is_opened_by_its_label_after_a_persons_pause(url, env):
    """What the page sees: a trusted click on the label, then, after the time
    this session takes to pick a file, a trusted change. Known-bad, the first
    version: a change on the hidden input with no click before it at all."""
    async def body(session):
        out = await actions.upload_files(session, "#behind", [str(env / "a.pdf")])
        order = await session.page().evaluate("order")
        return out, order, session.seed

    out, order, seed = asyncio.run(_with_browser(url, body))
    assert out == ('attached 1 file to #behind (picked in the file chooser its '
                   'label label[for="behind"] opened): a.pdf'), out
    clicks = [e for e in order if e[0] == "click"]
    change = next(e for e in order if e[0] == "change")
    assert clicks and clicks[0][1] == "pick" and clicks[0][2], order
    assert change[1] == "behind" and change[2], order

    # The wrapper's own pause before the files: two hesitations of the
    # session's hand, the first upload of the first page (nonce 1). Read from
    # the client's internals here (invisible-core since 38.34.0) only to know
    # what to expect.
    from invisible_core.juggler._behaviour import TypingPersona, plan_hesitation
    pause = plan_hesitation(TypingPersona.from_seed(seed), "file", 1, times=2) / 1000.0
    assert (change[3] - clicks[0][3]) / 1000 >= pause * 0.9, (
        "the change came %.0f ms after the click, before this session's pause "
        "of %.0f ms" % (change[3] - clicks[0][3], pause * 1000))


@pytest.mark.e2e
def test_a_hidden_input_with_no_label_is_refused_and_the_page_hears_nothing(url, env):
    async def body(session):
        with pytest.raises(RuntimeError, match="no label on the page opens it"):
            await actions.upload_files(session, "#hidden", [str(env / "a.pdf")])
        return await session.page().evaluate("JSON.stringify([seen, order])")

    assert asyncio.run(_with_browser(url, body)) == "[{},[]]"


# --- the snapshot: what is uploaded is the file that was checked -------------

@pytest.fixture
def private_tmp(tmp_path, monkeypatch):
    base = tmp_path / "tmp"
    base.mkdir()
    monkeypatch.setattr(actions.tempfile, "tempdir", str(base))
    return base


def test_the_engine_gets_a_private_copy_of_what_was_checked(allowed, private_tmp):
    real = actions.uploadable([str(allowed / "a.pdf")], env=_env(allowed))
    copy, = actions.snapshot_files(real, env=_env(allowed))
    assert os.path.basename(copy) == "a.pdf"
    assert copy.startswith(str(private_tmp)) and open(copy, "rb").read() == b"%PDF-1.4 a"
    if os.name != "nt":
        # Owner-only by mode where modes exist; on Windows the directory is
        # made under the user's own temp, which its ACL already keeps private.
        assert os.stat(os.path.dirname(os.path.dirname(copy))).st_mode & 0o077 == 0
    (allowed / "a.pdf").write_bytes(b"changed afterwards")
    assert open(copy, "rb").read() == b"%PDF-1.4 a", "the copy is not the name"


def test_a_name_swapped_for_a_link_after_the_check_is_not_followed(allowed, tmp_path, private_tmp):
    """The race in a shared staging directory: checked as a file, replaced by
    a link to something outside before the chooser is answered."""
    secret = tmp_path / "secret.txt"
    secret.write_text("key")
    real = actions.uploadable([str(allowed / "a.pdf")], env=_env(allowed))
    os.remove(allowed / "a.pdf")
    _link(allowed / "a.pdf", secret)
    with pytest.raises(PermissionError, match="not sent"):
        actions.snapshot_files(real, env=_env(allowed))
    assert os.listdir(private_tmp) == [], "a refused snapshot leaves nothing behind"


def test_a_directory_swapped_out_from_under_the_name_is_caught(allowed, tmp_path, private_tmp):
    sub = allowed / "sub"
    sub.mkdir()
    (sub / "c.pdf").write_bytes(b"ok")
    outside = tmp_path / "outside"
    outside.mkdir()
    (outside / "c.pdf").write_bytes(b"secret")
    real = actions.uploadable([str(sub / "c.pdf")], env=_env(allowed))
    os.rename(sub, allowed / "gone")
    _link(sub, outside)
    with pytest.raises(PermissionError, match="no longer inside"):
        actions.snapshot_files(real, env=_env(allowed))


def test_a_file_that_grew_past_the_limit_is_not_sent(allowed, private_tmp, monkeypatch):
    real = actions.uploadable([str(allowed / "a.pdf")], env=_env(allowed))
    monkeypatch.setattr(actions, "UPLOAD_MAX_BYTES", 4)
    with pytest.raises(ValueError, match="limit"):
        actions.snapshot_files(real, env=_env(allowed))
    assert os.listdir(private_tmp) == []


def test_snapshots_of_a_process_that_is_gone_go_and_live_ones_stay(allowed, private_tmp):
    """A process that ended without closing its browsers leaves its copies;
    the next upload removes them. A running process's copies stay, however
    old: its browser may still send them. Known-bad, the first version:
    anything older than an hour went, from any process."""
    import subprocess
    import sys

    child = subprocess.Popen([sys.executable, "-c", "pass"])
    child.wait()
    gone = private_tmp / ("%s%d-x" % (actions._SNAPSHOT_PREFIX, child.pid))
    mine = private_tmp / ("%s%d-y" % (actions._SNAPSHOT_PREFIX, os.getpid()))
    for d in (gone, mine):
        d.mkdir()
        os.utime(d, (0, 0))
    real = actions.uploadable([str(allowed / "a.pdf")], env=_env(allowed))
    new, = actions.snapshot_files(real, env=_env(allowed))
    assert not gone.exists() and mine.exists() and os.path.exists(new)
    assert os.path.basename(os.path.dirname(os.path.dirname(new))).startswith(
        "%s%d-" % (actions._SNAPSHOT_PREFIX, os.getpid()))


def test_several_files_have_a_budget_together(allowed, private_tmp, monkeypatch):
    two = [str(allowed / "a.pdf"), str(allowed / "b.pdf")]
    monkeypatch.setattr(actions, "UPLOAD_MAX_TOTAL_BYTES", 15)
    with pytest.raises(ValueError, match="together"):
        actions.uploadable(two, env=_env(allowed))
    monkeypatch.setattr(actions, "UPLOAD_MAX_TOTAL_BYTES", 20)
    real = actions.uploadable(two, env=_env(allowed))
    (allowed / "b.pdf").write_bytes(b"%PDF-1.4 b grew")
    with pytest.raises(ValueError, match="together"):
        actions.snapshot_files(real, env=_env(allowed))
    assert os.listdir(private_tmp) == []


@pytest.mark.skipif(not os.path.isdir("/proc/self/fd"), reason="the kernel's own record of an open file")
def test_a_file_unlinked_while_copied_keeps_its_name(allowed, private_tmp, monkeypatch):
    """A cleanup that deletes the staged file between open and copy: the
    descriptor still holds what was checked, and its readback gains the
    kernel's " (deleted)" mark, which must never reach the upload's name."""
    opened = actions._opened_path

    def unlinked_meanwhile(fd, real, st):
        os.unlink(real)
        return opened(fd, real, st)

    monkeypatch.setattr(actions, "_opened_path", unlinked_meanwhile)
    real = actions.uploadable([str(allowed / "a.pdf")], env=_env(allowed))
    copy, = actions.snapshot_files(real, env=_env(allowed))
    assert os.path.basename(copy) == "a.pdf"
    assert open(copy, "rb").read() == b"%PDF-1.4 a"
