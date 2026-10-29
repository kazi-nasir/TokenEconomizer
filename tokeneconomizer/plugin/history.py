"""Learning from the sessions a person ran before the plugin was installed.

The plugin learns by folding finished sessions (`fold`), and a hook sees only the sessions that
run with it. The agent, though, keeps its own logs of every earlier session -- Claude Code under
`~/.claude/projects/` -- so the first session in a repository also hands those of the
repository's earlier sessions to a detached `te history`, which folds them in the order they
ran, before what the live sessions add:

  * which repository a saved session belongs to is read as `repos` reads it: from the first
    directory its log names, identified as a live session's (`paths.repo_identity`);
  * each is folded once, keyed by its session id like a live session, so a saved session that
    is resumed later is not counted again, and one the plugin already recorded is left to
    `fold` (a hook saw it);
  * micro-skills and tool pruning learn from it as from a live session; the loop guard, which
    learns from live prompt-turns rather than at the fold, has them replayed from the log
    (`fold._replay_turns`);
  * the proposer, the one paid step, is held to the last of them, so a long history costs at
    most one model call (`microskills=on`), over the window the earlier ones filled;
  * a log written to in the last `fold.STALE_S` seconds may be a session still running in
    another window: it waits, and a later session start folds it once it has gone quiet.

`history.json` in the repository's state records what was done; `history=off` turns it off.
"""
import os
import time

from . import fold
from .paths import repo_identity
from .repos import _transcript_cwd

MARKER = "history.json"
RUNNING_S = 1800        # a `te history` still marked running after this long has died


def _quiet(path, now):
    try:
        return now - os.path.getmtime(path) >= fold.STALE_S
    except OSError:
        return False


def candidates(ctx, exclude=(), now=None):
    """(ready, waiting): this repository's saved sessions not yet folded, as (session_id, path),
    oldest first -- `ready` gone quiet (the most recent `history_max_sessions` of them),
    `waiting` still being written to."""
    now = now or time.time()
    folded = ctx.store.read("folded.json") or {}
    keys = {}
    found = []
    for sid, path in ctx.host.history_transcripts(ctx.env):
        if sid in folded or sid in exclude or os.path.exists(
                ctx.store.path("sessions", f"{sid}.json")):
            continue
        cwd = _transcript_cwd(path)
        if not cwd:
            continue
        if cwd not in keys:
            try:
                keys[cwd] = repo_identity(cwd, ctx.env).key
            except OSError:
                keys[cwd] = None
        if keys[cwd] != ctx.repo.key:
            continue
        try:
            found.append((os.path.getmtime(path), sid, path))
        except OSError:
            continue
    found.sort()
    ready = [(sid, p) for _, sid, p in found if _quiet(p, now)]
    waiting = [(sid, p) for _, sid, p in found if not _quiet(p, now)]
    return ready[-max(1, ctx.config.history_max_sessions):], waiting


def needed(ctx, now=None):
    """Is there history to fold for this repository? Cheap enough for a hook: it reads the
    marker, and looks at the logs the last run left waiting."""
    if ctx.config.history == "off":
        return False
    now = now or time.time()
    m = ctx.store.read(MARKER)
    if not isinstance(m, dict):
        return True
    if m.get("running") and now - float(m["running"]) < RUNNING_S:
        return False
    if not m.get("finished"):
        return True                      # a run that died before it finished
    return any(_quiet(p, now) for _, p in m.get("waiting") or ())


def spawn(ctx, exclude=()):
    """Fold this repository's history in a detached `te history` (inline when TE_FOLD_SYNC=1)."""
    m = ctx.store.read(MARKER) or {}
    ctx.store.write(MARKER, dict(m, running=time.time()))
    if ctx.env.get("TE_FOLD_SYNC") == "1":
        return run(ctx, exclude=exclude)
    return ctx.spawn(["history"] + [a for sid in exclude for a in ("--exclude", sid)],
                     "fold.log")


def run(ctx, exclude=(), now=None):
    """Fold every ready saved session of this repository, oldest first. Returns the marker."""
    now = now or time.time()
    prior = ctx.store.read(MARKER) or {}
    ctx.store.write(MARKER, dict(prior, running=time.time()))
    ready, waiting = candidates(ctx, exclude=exclude, now=now)
    folded, turns = 0, 0
    try:
        for i, (sid, path) in enumerate(ready):
            rec = fold.fold_session(ctx, sid, transcript_path=path,
                                    paid=i == len(ready) - 1, history=True)
            if rec is not None and not rec.get("error"):
                folded += 1
                turns += int(rec.get("replayed_turns") or 0)
    finally:
        m = {"started": prior.get("started") or now, "finished": time.time(),
             "sessions": int(prior.get("sessions", 0)) + folded,
             "turns": int(prior.get("turns", 0)) + turns,
             "waiting": [[sid, p] for sid, p in waiting]}
        ctx.store.write(MARKER, m)
    ctx.store.append("events.jsonl", {"kind": "history", "at": time.time(), "sessions": folded,
                                      "turns": turns, "waiting": len(waiting)})
    return m
