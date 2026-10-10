"""One piece of work: the two browsers this process serves, and the file that
says who `main` is.

⛔ OPEN FIRST, AND NOTHING ELSE OPENS A BROWSER. Until 0.53.0 the lifecycle had
seven steps - declare, wake, look, act, rebuild, remember, close - because a
command aimed at a browser that was not running STARTED one, and an action
whose browser had died REBUILT it and ran the action again. Each of those was
a place to be wrong, and both were: the wake reopened pages the file had noted
and blamed the site for them; the rebuild fired on a domain that did not
resolve and threw away a healthy browser with its cookies ([B203]); a rebuild
was silent, then said, then a type. The owner cut it on 2026-09-14: a browser
is opened by `browser_open`, a tool whose browser is not open says so and says
what to call, and a browser that died says that instead and is not brought
back behind the model's back. What is left:

  open      `open(role, ...)` plans a browser and starts it; with no arguments
            it reopens `main` as the person this session already was.
  act       `acting(fn, ...)` runs one action on an open browser. Not open, or
            gone: the sentence, never a start and never a retry.
  look      `listing()` and `status()` describe what is open.
  remember  `open` writes who `main` is; nothing else writes the file.
  close     `close(role)` closes one browser and keeps who it was, so that a
            `browser_open` with no arguments brings the same person back;
            `close_all()` at the end of the process.

There is exactly one of these per process, because a process serves exactly
one piece of work; `server.py` builds it from `INVISIBLE_MCP_SESSION_ID` and every
tool goes through it. A test builds its own with a factory that launches no
browser and installs it in the server's place.
"""
from __future__ import annotations

import asyncio
import os
import time
from pathlib import Path
from dataclasses import dataclass
from typing import Awaitable, Callable, Optional

from invisible_playwright.async_api import TargetClosedError

from . import DEFAULT_BROWSER_ID, GONE, NOT_OPEN, SUPPORT_BROWSER_ID, identity, plan, process, store
from ..quiet import swallow
from .actions import NAVIGATION_TIMEOUT_MS
from .session import StealthSession

#: TWO browsers in one piece of work, with fixed roles, and the number is a
#: decision rather than a measurement - the opposite of what it was until
#: 2026-09-11.
#:
#: It used to be eight, and the eight was measured: 61 processes and 6,515 MB,
#: the eighth taking 13.6 s to start against the first one's 6.8. Nothing about
#: that stopped being true. What changed is what a session MEANS. Identity lives
#: on the browser - seed, fingerprint, profile - so a session holding eight
#: browsers held eight identities while everything above it, the transcript and
#: the saved state and anything a person attaches to a session, was addressed
#: to one. "Whose is this?" had two possible answers and no way to choose.
#:
#: So a session is ONE identity, `main`, plus ONE helper beside it, `support`,
#: for the things that must not touch that identity: a temporary mailbox to
#: receive a verification, a lookup, a second opinion on a page. It is a
#: browser and not a tab because a tab would share cookies and fingerprint with
#: the site the identity is being built on, and the whole point of the helper
#: is that it does not. The helper is not saved - it dies with the process -
#: because a helper that survives IS a second identity, which is the thing this
#: number exists to rule out.
#:
#: The roles are NAMES A CALLER CANNOT INVENT. `browser` on every tool is a
#: closed choice, `main` or `support`, so nothing here ever holds a browser
#: called `b3` or `walmart-jobs` again.
MAX_BROWSERS_PER_SESSION = 2

#: The launch settings that say WHO a browser is, and so the ones the saved
#: file carries. Everything else in the launch kwargs describes THIS LAUNCH -
#: the engine it runs on, whether its window is shown - and is decided by
#: `plan.launched_here` every time, never read back from a file. Both halves
#: were measured by getting them wrong: `binary_path` saved would reopen a
#: browser onto a path that means nothing on another machine, and `headless`
#: saved (until 0.50.0) made one headed run decide for every later process
#: that read the same file.
#:
#: ⛔ THESE ARE THE LAUNCH KWARGS' OWN NAMES, NOT THE TOOL ARGUMENTS' NAMES:
#: this list said `profile` for its first day, the launch kwarg is
#: `profile_dir`, so the filter matched nothing and the profile - the one
#: field that carries the logins a reopened browser is FOR - was never saved.
WHO_A_BROWSER_IS = ("seed", "proxy", "profile_dir")

#: What `browser_open` says about a seed it read back from the file.
REMEMBERED = "the person this session already was"

#: How long `open` gives the engine to say whether it is a cache hit or a
#: download: the core reads its stamp, sweeps orphans and resolves the release
#: asset before it reports the first phase, and the last of those is a network
#: round trip. Past this, the answer is "its download is starting" and the
#: caller asks again; nothing here waits for a transfer.
ENGINE_SETTLE_SECONDS = 10.0

#: The longest a typing call waits before it answers, finished or not; past
#: it the typing goes on in the background (`Work.typing`). The bound
#: `browser_navigate` gives a page to load, which is the longest any tool here
#: already waited inside one call, so no call waits longer than it did before.
ANSWER_WITHIN_S = NAVIGATION_TIMEOUT_MS / 1000


@dataclass
class _Typing:
    """A typing in flight: the task, where, how much, since when, and whether
    it has outlived the call that started it - the only case in which other
    actions are refused rather than queued behind it."""
    task: asyncio.Task
    selector: str
    total: int
    started: float
    outlived: bool = False


def profile_holder(directory) -> Optional[str]:
    """Which other Firefox holds a profile directory right now, or None.

    ⛔ READ FROM THE LOCK FIREFOX ACTUALLY TAKES, AND THE TWO SYSTEMS TAKE
    DIFFERENT ONES. The first version looked for `lock` or `.parentlock` and
    was wrong on both: on Windows the file is `parent.lock`, so a held profile
    was never seen, and on Linux `.parentlock` stays behind in every profile a
    Firefox ever closed, so a closed one read as locked.

    Windows: Firefox keeps `parent.lock` open with no sharing for as long as it
    runs, and the file stays after it exits. So the question is whether the
    file can be opened: measured on firefox-34, PermissionError (13) while a
    browser had the profile, opened after it closed.

    Linux: the lock is the symlink `lock`, pointing at `address:+pid`, which
    Firefox removes when it exits; one left by a crash names a process that is
    gone, and Firefox clears that itself, so it does not hold anything.

    Anywhere else this answers None, and a launch that then fails says what the
    engine said.
    """
    if not directory:
        return None
    folder = Path(directory)
    if os.name == "nt":
        try:
            with open(folder / "parent.lock", "rb"):
                return None
        except PermissionError:
            return "another Firefox"
        except OSError:
            return None
    try:
        target = os.readlink(folder / "lock")
    except OSError:
        return None
    host, _, pid = target.rpartition(":+")
    if not pid.isdigit():
        return "another Firefox (%s)" % target
    if host and not _this_machine(host):
        # Its process cannot be asked from here, so it is taken at its word,
        # which is what Firefox does with it too.
        return "a Firefox on %s" % host
    if not process.alive(int(pid)):
        return None
    return "another Firefox (process %s)" % pid


def _this_machine(address: str) -> bool:
    """Whether a lock's address names this machine, the way Firefox wrote it."""
    import socket

    if address.startswith("127."):
        return True
    with swallow("a host with no resolvable name is not this machine's address"):
        return address == socket.gethostbyname(socket.gethostname())
    return False


def _why_it_did_not_start(settings: dict) -> str:
    """The cause to look at first, named from what this launch was handed.

    ⛔ IT USED TO SAY "A PROXY THAT IS DOWN IS THE USUAL CAUSE" WHATEVER THE
    LAUNCH WAS. On 2026-10-01 a helper with no proxy at all failed on a profile
    another Firefox held, and was told to pass proxy="" - advice about something
    it did not have, and a value some clients cannot even send. A held profile
    is refused before the launch now (`Work.open`), so what is left to name
    here is the exit, when there is one.
    """
    if settings.get("proxy"):
        return (" It was going out through %s; a proxy that is down is a common "
                "cause, so try another exit." % settings["proxy"]["server"])
    return ""


class Work:
    """The one piece of work this process serves. See the module docstring
    for the lifecycle; every method here is one step of it."""

    def __init__(self, session_id: str, *, factory=StealthSession, engine=None) -> None:
        self.session_id = session_id
        self._factory = factory
        #: Where the engine is for this process (`invisible_playwright_mcp.engine.Engine`), or
        #: None for a piece of work that launches nothing and need not ask.
        #: `open` does not start a browser while it is not on disk: it answers
        #: with the download's progress instead, so no tool call waits on it.
        self._engine = engine
        #: The browsers that are open, by role, and what each was launched with.
        self._open: dict[str, StealthSession] = {}
        self._launched: dict[str, dict] = {}
        #: The role the last command was aimed at. See `focused()`, which is
        #: the only reader and which never hands out a browser that is not
        #: open.
        self._acted = DEFAULT_BROWSER_ID
        #: One lock around opening and closing: the live pane and the agent
        #: talk to this server at the same time, and two opens of one role
        #: racing would leave a browser nobody holds a handle to.
        self._lock = asyncio.Lock()
        #: One lock per browser around the commands that drive its input; see
        #: `acting`. Reads and the live pane's frames do not take it.
        self._input_locks: dict[str, asyncio.Lock] = {}
        #: A typing still going on in the background, per browser; see `typing`.
        self._typing: dict[str, _Typing] = {}

    # --- what is here -----------------------------------------------------------

    def roles(self) -> list:
        """The browsers that are open, by role."""
        return sorted(self._open)

    def focused(self) -> str:
        """The browser the agent is working in: the one the last command was
        aimed at, or "" when none is open.

        ⛔ IT IS A FACT NOW, AND UNTIL 0.55.0 IT WAS THE CONSTANT `main`. The
        listing answered `focus: "main"` from a literal, because the tools that
        moved a focus went with the eight-browser session on 2026-09-11 and
        nothing replaced them. The interface believed it: it draws a dot on
        that browser reading `the agent is working here`, so with `support`
        open the dot sat on `main` while the agent typed into the helper, and
        the screen marked current was `main` whatever the agent was doing.
        Saying where the work is happening is the whole job of that dot, and
        it was the one thing it could not do.

        The fact was already in `acting`, which every command that touches a
        page goes through and which knows the role of each one; it was simply
        thrown away. Nothing else had to be built.

        ⛔ AND IT NEVER NAMES A BROWSER THAT IS NOT OPEN. A browser can be
        closed or found gone after it was last acted in, and a focus pointing
        at one would put the dot on a screen the stage no longer draws. The
        fallback is `roles()` in order, so `main` wins whenever it is open -
        which is also where a command that names none goes.

        What this does NOT say is where an unnamed command lands. That is
        always `main`, it is a constant of the two roles rather than a fact
        about this moment, and it is said in `note` and in the tool's own
        description.
        """
        if self._acted in self._open:
            return self._acted
        return next(iter(self.roles()), "")

    def remembered(self) -> Optional[dict]:
        """Who `main` was the last time this session opened it, from the file.

        The identity fields only. A file written by an older build can name
        eight browsers and call them anything: what comes back is the one the
        file says was in focus, else the first by name, and only its identity.
        """
        saved = store.load(self.session_id)
        held = (saved or {}).get("browsers") or {}
        if not held:
            return None
        keep = saved.get("focus") if saved.get("focus") in held else sorted(held)[0]
        return {k: v for k, v in held[keep].items() if k in WHO_A_BROWSER_IS}

    def remember(self) -> None:
        """Write down who `main` is. Called by `open` and by nothing else.

        The helper is never written down: a support browser that came back
        after a restart would be a second identity. A write that fails costs
        the saved file and nothing else - by the time this runs the browser is
        built and correct, and a full disk must not turn a working
        `browser_open` into an error.
        """
        launched = self._launched.get(DEFAULT_BROWSER_ID)
        if launched is None:
            return
        who = {k: v for k, v in launched.items() if k in WHO_A_BROWSER_IS}
        with swallow("a write that fails costs the saved file and nothing else"):
            store.save(self.session_id, {DEFAULT_BROWSER_ID: who}, focus=DEFAULT_BROWSER_ID)

    # --- open and close -------------------------------------------------------

    async def open(self, role: str, *, seed: Optional[int] = None,
                   proxy: Optional[str] = None,
                   profile: Optional[str] = None) -> str:
        """Open one of the two browsers as somebody, or reopen it as somebody
        else. Answers with the plan it made.

        ⛔ WITH NO ARGUMENTS, `main` IS THE PERSON THIS SESSION ALREADY WAS.
        That is the whole of what survives a process: three fields in a file,
        read here and nowhere else. A conversation reopened tomorrow, or an
        assistant reconnecting, calls `browser_open` and gets the same seed,
        the same exit and the same profile without having to know them. Any
        argument means a decision, and a decision goes through the planner,
        which refuses a conflict rather than guessing.
        """
        if role not in (DEFAULT_BROWSER_ID, SUPPORT_BROWSER_ID):
            # ⛔ THE SCHEMA ALREADY REFUSES THIS AND THIS STILL REFUSES IT. The
            # tool's `browser` is a Literal, so a model that invents a name is
            # turned back by the protocol before it reaches here - but the
            # schema is not the only door: the interface and the tests call
            # these functions directly, and a third browser called `b3` is the
            # thing two fixed roles exist to rule out.
            raise ValueError("there are two browsers here: `main`, your own "
                             "identity, and `support`, the helper beside it. "
                             "There is no %r." % role)
        asked = seed is not None or proxy is not None or profile is not None
        remembered = None if asked or role != DEFAULT_BROWSER_ID else self.remembered()
        if remembered:
            # ⛔ THIS LAUNCH DECIDES THE ENGINE AND THE WINDOW, NOT THE FILE.
            # `launched_here` is written OVER the file's word, not under it: a
            # file from before 0.50.0 carries `headless: false`, and a file
            # never carries `binary_path` because a path on this machine means
            # nothing on another.
            settings = dict(remembered, **plan.launched_here())
            seed_from, exit_note, warnings = REMEMBERED, "", ()
        else:
            try:
                chosen = plan.plan_session(seed=seed, proxy=proxy, profile=profile,
                                           helper=role == SUPPORT_BROWSER_ID)
            except (identity.IdentityConflict, ValueError) as exc:
                # Refused, not guessed. Every case here is one where continuing
                # would hand the caller a different person than the one they
                # asked for, and whatever is already running is deliberately
                # left alone: a refusal must not cost somebody the browser they
                # already had.
                raise ValueError("refused: %s" % exc)
            settings = chosen.kwargs
            seed_from, exit_note, warnings = chosen.seed_from, chosen.exit, chosen.warnings

        main_launched = self._launched.get(DEFAULT_BROWSER_ID)
        if role == SUPPORT_BROWSER_ID and proxy is None and main_launched is not None:
            # ⛔ THE HELPER INHERITS THE EXIT, BY DEFAULT AND ON PURPOSE. A
            # helper that came out through a different address than the
            # identity it helps would be the one thing on the wire saying
            # "these two are not the same person, and yet they work together".
            # Its FINGERPRINT is its own - a fresh seed unless given - because
            # the two must not read as one browser either. Same exit, different
            # person: a colleague at the next desk. Copied as the resolved
            # dict, absence included: a `main` that goes out direct has no
            # `proxy` key, so the helper goes out direct as well, rather than
            # picking up an environment proxy `main` never used.
            settings.pop("proxy", None)
            if main_launched.get("proxy"):
                settings["proxy"] = main_launched["proxy"]
            exit_note = "this machine's own address, the same as main"

        self._refuse_a_held_profile(role, settings.get("profile_dir"))

        if self._engine is not None and not self._engine.ready():
            # ⛔ AFTER THE PLAN, AND NOT A LAUNCH: AN ANSWER. A plan that is
            # refused is refused whether or not the engine is here, and a
            # refusal reaches the client as an error; this answer is not one.
            # Launching here would run a second download inside the tool call
            # - the minutes-long call that the README's fetch line existed to
            # avoid, and a race with the one in flight on the same temporary
            # tree. `start` is a no-op while one is in flight and is what
            # retries a failed one. The wait is for the engine to SAY what it
            # is - a cache hit answers in milliseconds and a download announces
            # itself before its first byte - never for the download, so a warm
            # cache opens on the first call and a cold one is answered, not
            # waited on.
            self._engine.start()
            await asyncio.to_thread(self._engine.settle, ENGINE_SETTLE_SECONDS)
            if not self._engine.ready():
                return self._engine.describe()

        async with self._lock:
            # ⛔ The old browser is closed FIRST and unconditionally. Starting
            # the new one first would leave two browsers alive if the second
            # start failed, and the one still holding the profile directory
            # is the one nobody has a handle to any more.
            await self._drop(role)
            session = self._factory(**settings)
            try:
                await session.start()
            except Exception as exc:
                # ⛔ Said plainly, because the dangerous reading is "that
                # failed, carry on". Nothing is running there now, and every
                # later tool will repeat that it is not open rather than
                # quietly starting a browser without the exit that was asked
                # for.
                with swallow("a session that did not start may have nothing to close"):
                    await session.close()
                raise RuntimeError(
                    "the %s browser did NOT start: %s\n"
                    "Nothing is browsing there, and the tools will keep failing "
                    "until browser_open succeeds.%s"
                    % (role, exc, _why_it_did_not_start(settings)))
            self._open[role] = session
            self._launched[role] = settings
            # Opening a browser is working in it: a helper opened mid-task is
            # where the next few commands are going, and the pane should say
            # so before the first of them arrives rather than after.
            self._acted = role
            if role == DEFAULT_BROWSER_ID:
                self.remember()
        return "the %s browser is open. %s" % (role, plan.describe(
            settings, seed_from=seed_from, exit_note=exit_note, warnings=warnings))

    def _refuse_a_held_profile(self, role: str, directory) -> None:
        """Refuse, BEFORE the launch, a profile another browser holds.

        ⛔ BEFORE, NOT DIAGNOSED AFTER. Firefox lets one process use a profile,
        and a second one on the same directory does not fail: measured on
        firefox-34, it did not answer in sixty seconds, and what reached the
        caller of a failed one was the engine's "the pipe is closed". Before
        the old browser of this role is closed, too, so a refusal costs
        nobody the browser they had.

        This role's own browser is the one case that is not a conflict: it is
        closed first, and reopening `main` on its own profile is ordinary.
        The other role is known from this process's own record, which holds
        on every system; anything else is asked of the lock (`profile_holder`).
        """
        if not directory:
            return
        def holds(r):
            return r in self._open and (self._launched.get(r) or {}).get("profile_dir") == directory
        if holds(role):
            return
        other = SUPPORT_BROWSER_ID if role == DEFAULT_BROWSER_ID else DEFAULT_BROWSER_ID
        holder = ("`%s`" % other) if holds(other) else profile_holder(directory)
        if not holder:
            return
        advice = ("Leave profile out for a helper that is not saved, or give it "
                  "another directory." if role == SUPPORT_BROWSER_ID else
                  "Close that browser first, or give main another directory.")
        raise ValueError(
            "refused: %s already has the profile %s open, and Firefox lets one "
            "browser use a profile at a time. %s" % (holder, directory, advice))

    async def close(self, role: str) -> str:
        """Close one browser. Who it was is kept: `browser_open` with no
        arguments brings the same person back. The other browser is not
        touched."""
        async with self._lock:
            existed = role in self._open
            await self._drop(role)
        left = self.roles()
        if not existed:
            return "the %s browser is not open." % role
        return ("the %s browser is closed. Still open: %s."
                % (role, ", ".join(left) if left else "none"))

    async def _drop(self, role: str) -> None:
        rec = self._typing.pop(role, None)
        if rec is not None:
            rec.task.cancel()
        session = self._open.pop(role, None)
        self._launched.pop(role, None)
        if session is not None:
            with swallow("a browser being closed may already be gone, and a "
                         "close that fails must not stop the next open"):
                await session.close()

    async def close_all(self) -> None:
        for role in list(self._open):
            await self._drop(role)

    # --- act ------------------------------------------------------------------

    def gone(self, role: str) -> RuntimeError:
        """Forget this browser, and answer the sentence that says it is gone.

        ⛔ ONE PLACE, BECAUSE IT IS ONE FACT WITH TWO HALVES: the model is
        told what to call, and the dead object is dropped so the next
        `browser_open` starts clean rather than handing the same corpse out
        again. Three callers notice a browser is gone and none of them may
        do only half of this: `session` locally, `acting` when an action
        raises a closed target, and `status` when the question does.
        """
        self._open.pop(role, None)
        self._launched.pop(role, None)
        return RuntimeError(GONE % role)

    def session(self, role: str) -> StealthSession:
        """This browser, open and worth handing out, or the one sentence that
        says why not.

        ⛔ NEVER STARTS ONE AND NEVER BRINGS ONE BACK. A browser that is not
        open is `NOT_OPEN`; one that was open and is unusable is `GONE`. Both
        sentences name `browser_open`, which is the model's next call.

        ⛔ AND WHAT IT CAN SEE IS LOCAL: `is_usable` asks the object, not the
        browser, and a killed engine answers "connected" for seconds. The
        round trip is what notices that, which is why the callers below
        translate a closed target as well rather than trusting this.
        """
        session = self._open.get(role)
        if session is None:
            raise RuntimeError(NOT_OPEN % role)
        if not session.is_usable():
            raise self.gone(role)
        return session

    async def acting(self, fn: Callable[..., Awaitable], *args,
                     role: Optional[str] = None, **kwargs):
        """Run one action that DRIVES a browser: its pointer, its keyboard, its
        navigation. The one funnel every such tool goes through, so "what a
        browser has to be before a tool may use it" is a fact known in one
        place.

        ⛔ ONE AT A TIME PER BROWSER, AND THAT IS THE DEFAULT RATHER THAN A
        FLAG. A page has one focus and one keyboard: two `browser_type` calls
        at once drove both into one field ("BBAAABABBABAB...", the other field
        a single "A", measured 2026-09-30). So these queue behind the browser's
        input lock. It used to be an `exclusive=True` every input tool had to
        remember, so a tool that forgot it raced in silence; now a tool that
        only READS says so by calling `reading`, and forgetting that costs a
        wait instead of a corrupted field.

        A browser that closes UNDER the action - the window shut by hand, the
        engine crashed mid-call - surfaces as a closed target; that is the
        same fact as a browser found dead beforehand, and it gets the same
        sentence, with the browser forgotten so the next open starts clean.
        Any other failure is the page's answer and passes through untouched.
        """
        at = role or DEFAULT_BROWSER_ID
        await self._refuse_while_typing(at)
        news = self._typing_news(at)
        async with self._input_lock(at):
            answer = await self._act(at, fn, *args, **kwargs)
        return news + answer if news and isinstance(answer, str) else answer

    async def reading(self, fn: Callable[..., Awaitable], *args,
                      role: Optional[str] = None, **kwargs):
        """Run one action that only READS a browser. It does not queue behind
        input: a snapshot, a read or a frame of the live pane is answered while
        a long text is still being typed, which is when somebody watching most
        wants to see the page."""
        return await self._act(role or DEFAULT_BROWSER_ID, fn, *args, **kwargs)

    async def typing(self, fn: Callable[..., Awaitable], selector: str, text: str,
                     role: Optional[str] = None) -> str:
        """Type, through the input lock like any action, answering within
        `ANSWER_WITHIN_S` whether or not the typing has finished.

        ⛔ A PERSON TYPES TWO THOUSAND CHARACTERS IN ABOUT TEN MINUTES, AND A
        CLIENT GIVES UP ON ONE CALL LONG BEFORE THAT. This session's hand types
        at 250-380 ms a character (the persona's planned rhythm over twenty
        seeds), so 2,000 characters take 8-13 minutes; measured on main, 252
        characters were in after 90 s, and the client's 300 s cut the call off
        with the field half full. No human rhythm finishes that inside one
        call, and a faster one is a hand nobody has.

        So typing that has not finished when the answer is due goes on in the
        background, still holding the browser's input lock, and the answer says
        so. Until it ends, every action on this browser is refused with how far
        it has got rather than queued - a click left waiting ten minutes would
        outlive its own call and then land on a page the caller has stopped
        thinking about - and reads go on as usual. The outcome, kept or not, is
        said by the first action after it.
        """
        at = role or DEFAULT_BROWSER_ID
        await self._refuse_while_typing(at)
        news = self._typing_news(at)
        lock = self._input_lock(at)

        async def run():
            async with lock:
                return await self._act(at, fn, selector, text)

        task = asyncio.create_task(run())
        rec = _Typing(task, selector, len(text), time.monotonic())
        try:
            done, _ = await asyncio.wait({task}, timeout=ANSWER_WITHIN_S)
        except asyncio.CancelledError:
            # The client gave up on the call; the typing it started has not,
            # and the next action must see it and say how it ended.
            rec.outlived = True
            self._typing[at] = rec
            raise
        if task in done:
            return news + task.result()
        rec.outlived = True
        self._typing[at] = rec
        return news + (
            "still typing into %s: %d characters at this browser's typing pace "
            "take longer than one answer may wait, so the typing goes on after "
            "this answer. Until it is done, actions on the %s browser are "
            "refused with how far it has got; reads work as usual, and "
            "browser_status says when it has finished." % (selector, len(text), at))

    def _input_lock(self, at: str) -> asyncio.Lock:
        return self._input_locks.setdefault(at, asyncio.Lock())

    async def _refuse_while_typing(self, at: str) -> None:
        """Raise, saying how far it has got, while a typing goes on in `at`."""
        rec = self._typing.get(at)
        if rec is None or not rec.outlived or rec.task.done():
            return
        raise RuntimeError("the %s browser is still typing %s. Reads work "
                           "meanwhile; ask for this again when it is done."
                           % (at, await self._typing_progress(at, rec)))

    async def _typing_progress(self, at: str, rec: "_Typing") -> str:
        """`into #x: 812 of 2000 characters so far, about 7 minutes to go`."""
        said = "into %s" % rec.selector
        with swallow("how far it got is a courtesy; the refusal stands without it"):
            held = await self.session(at).page().locator(rec.selector).first.input_value(
                timeout=1_000)
            spent = time.monotonic() - rec.started
            said += ": %d of %d characters so far" % (len(held), rec.total)
            if held and spent > 0:
                left = (rec.total - len(held)) * spent / len(held)
                said += ", about %d minutes to go" % max(1, round(left / 60))
        return said

    def typing_status(self, at: str) -> str:
        """What `browser_status` adds about typing in the background: that it
        goes on, or how it ended. Empty when there is nothing to say."""
        rec = self._typing.get(at)
        if rec is None:
            return ""
        if not rec.task.done():
            return " typing: still going on in the background, into %s." % rec.selector
        return " " + self._typing_news(at).strip()

    def _typing_news(self, at: str) -> str:
        """How a typing that went on in the background ended, said once, by the
        first action after it; empty when there is none to tell."""
        rec = self._typing.get(at)
        if rec is None or not rec.task.done():
            return ""
        del self._typing[at]
        if rec.task.cancelled():
            return ""
        failed = rec.task.exception()
        if failed is not None:
            return "(the typing into %s that went on in the background stopped: %s)\n" % (
                rec.selector, " ".join(str(failed).split()))
        return "(the typing that went on in the background has finished: %s)\n" % (
            rec.task.result())

    async def _act(self, at: str, fn, *args, **kwargs):
        # The browser is looked up after any queue, so one closed while a
        # command waited is reported as gone rather than used.
        session = self.session(at)
        # ⛔ AFTER THE BROWSER ANSWERED FOR ITSELF, NOT BEFORE. `session`
        # refuses a role that is not open and forgets one that is gone, so
        # recording here means the focus only ever moves to a browser that was
        # there to be worked in. Recorded before the action rather than after
        # it because a click that FAILS still happened in that browser, and
        # that is exactly the moment somebody watching wants to be looking at
        # the right screen.
        self._acted = at
        try:
            return await fn(session, *args, **kwargs)
        except TargetClosedError:
            raise self.gone(at) from None

    # --- look -------------------------------------------------------------------

    async def network(self, role: str, **filters) -> dict:
        return self.session(role).network.entries(**filters)

    async def network_clear(self, role: str) -> dict:
        return self.session(role).network.clear()

    async def network_capture(self, role: str, *, request_bodies: bool) -> dict:
        return await self.session(role).network.capture(request_bodies=request_bodies)

    async def listing(self) -> dict:
        """Which browsers are open, where each one is, and which one commands
        that name none go to. Starts nothing.

        One row per open browser: `id`, `url` (the page it is on) and `urls`
        (every page it holds). `urls` is a list, or None for "open and
        unreadable" - the two are different answers and the pane draws them
        differently.

        ⛔ AND THE ROWS CARRY NOTHING DERIVABLE FROM THE ANSWER AROUND THEM.
        `running` went in 0.54.0 saying `true` on every row it could produce;
        `focused` and `limit` went the same way in 0.55.0. `focused` was
        `id == focus` with `focus` named two lines above it, and `limit` was
        the constant 2 that no reader in this product ever read - the
        interface never mentioned the word, and the sentence in `note`
        already says how many of how many. A fact that is a property of the
        answer belongs in the description; a fact that is derivable from
        another field belongs to that field alone, or the two get a chance to
        disagree.
        """
        here = self.focused()
        rows = []
        for name in self.roles():
            session = self._open[name]
            try:
                pages = await session.describe_pages()
            except TargetClosedError:
                # ⛔ NOT A ROW THAT CANNOT BE READ: A BROWSER THAT IS NOT THERE.
                # Everything this answers is open by definition, so a browser
                # whose engine has gone is dropped and forgotten rather than
                # listed as running - the live panes draw this answer, and a
                # pane over a browser that no longer exists is the frozen
                # picture of [B202], one layer up.
                self.gone(name)
                continue
            except Exception:
                # Readable as a state rather than as an absence: a browser whose
                # pages cannot be read is not a browser with no pages, and a pane
                # drawing "nothing open" over a live window would be a lie.
                # `urls` is None for exactly this, and the pane draws it apart.
                rows.append({"id": name, "url": "", "urls": None})
                continue
            urls = [p["url"] or "" for p in pages]
            # The ACTIVE page rather than the first: a site that opens one of
            # its own makes those two different, and the page a command drives
            # is the newest live one.
            shown = next((p for p in pages if p["active"]), pages[0] if pages else None)
            rows.append({"id": name,
                         "url": (shown["url"] or "") if shown else "", "urls": urls})
        return {
            "focus": here,
            "browsers": rows,
            # ⛔ THE NOTE NAMES `main` FROM THE CONSTANT, NEVER FROM `here`.
            # They were the same string until 0.55.0 and this line simply used
            # whichever was to hand. Now `here` is where the agent last acted,
            # so reading it here would answer "commands that name none go to
            # support" the moment the helper had been touched - which is false,
            # and false about the one rule a caller uses to leave `browser` out.
            "note": (NOT_OPEN % DEFAULT_BROWSER_ID if not rows else
                     "%d of %d browsers. Commands that name none go to %s."
                     % (len(rows), MAX_BROWSERS_PER_SESSION, DEFAULT_BROWSER_ID)),
        }

    async def status(self, role: str) -> str:
        """Who this browser is - identity, exit, profile - and the page it is
        on. Starts nothing; not open, or gone, is the sentence."""
        session = self.session(role)
        launched = self._launched[role]
        try:
            rows = await session.describe_pages()
        except TargetClosedError:
            # ⛔ THE QUESTION IS WHAT NOTICED, and it must not answer anyway.
            # The identity below comes from the launch kwargs, which outlive
            # the browser, and the page from a url a dead page still holds in
            # memory - so without this the status of a browser whose engine had
            # been killed read exactly like the status of a healthy one.
            # Measured 2026-09-14 against firefox-30.
            raise self.gone(role) from None
        except Exception:
            # Any other failure is the page being difficult, not the browser
            # being gone: the identity is still worth reporting.
            return (plan.describe(launched) + " page: the page is unreadable."
                    + self.typing_status(role))
        here = next((r for r in rows if r["active"]), rows[0] if rows else None)
        where = (here["url"] or "blank") if here else "no page open yet"
        # ⛔ COUNTED, AND NOT BLAMED ON ANYBODY. A caller cannot make, choose
        # or close a page, so the honest report of a second one is that it is
        # there - not who opened it: a confident wrong cause is the defect this
        # project removed from `navigate`.
        if len(rows) > 1:
            where += " (%d other pages are open in this browser)" % (len(rows) - 1)
        return plan.describe(launched) + " page: %s." % where + self.typing_status(role)
