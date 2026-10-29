"""Learning from the sessions saved before the plugin was installed (`plugin/history.py`).

Each test lays out a Claude Code config directory with saved session logs, as the agent keeps
them (`projects/<slug>/<session id>.jsonl`), then drives the hooks: the first session start in a
repository folds its saved sessions (inline here, `TE_FOLD_SYNC=1`), and later sessions find what
they taught.
"""
import json
import os
import time

import pytest

from tokeneconomizer import loopguard, trace
from tokeneconomizer.plugin import cli, history, sidecall
from tokeneconomizer.plugin.context import LiveContext
from tokeneconomizer.plugin.hosts import ClaudeCode

from test_live_hooks import call, run_session, store_for, trap_session, write_session

CLAIM = "`pytest …` fails here with command not found; `python -m pytest …` works."


@pytest.fixture
def env(tmp_path):
    return {"TOKENECONOMIZER_HOME": str(tmp_path / "home"), "TE_FOLD_SYNC": "1",
            "TE_TOOLPRUNE_QUIET_MINUTES": "0", "TE_MICROSKILLS": "envfacts",
            "CLAUDE_CONFIG_DIR": str(tmp_path / "claude"),
            "GIT_CEILING_DIRECTORIES": str(tmp_path), "PATH": os.environ.get("PATH", "")}


def make_repo(tmp_path, name, remote):
    r = tmp_path / name
    (r / ".git" / "info").mkdir(parents=True)
    (r / ".git" / "HEAD").write_text("ref: refs/heads/main\n")
    (r / ".git" / "config").write_text(f'[remote "origin"]\n\turl = {remote}\n')
    (r / "pyproject.toml").write_text("[project]\nname='w'\n")
    return str(r)


@pytest.fixture
def repo(tmp_path):
    return make_repo(tmp_path, "repo", "https://github.com/acme/widget.git")


def saved(env, repo, sid, writer, *args, age_s=3600):
    """A session log Claude Code saved for `repo`, last written `age_s` seconds ago."""
    d = os.path.join(env["CLAUDE_CONFIG_DIR"], "projects",
                     "".join(c if c.isalnum() else "-" for c in repo))
    os.makedirs(d, exist_ok=True)
    path = writer(os.path.join(d, f"{sid}.jsonl"), *args)
    with open(path) as fh:
        lines = [dict(json.loads(ln), cwd=repo, sessionId=sid) for ln in fh if ln.strip()]
    with open(path, "w") as fh:
        fh.write("\n".join(json.dumps(x) for x in lines) + "\n")
    t = time.time() - age_s
    os.utime(path, (t, t))
    return path


def start(env, repo, sid, tmp_path):
    tp = str(tmp_path / f"{sid}.jsonl")
    write_session(tp, [("hello", [])])
    return run_session(env, repo, sid, tp)


def test_the_first_session_learns_from_the_saved_ones_and_the_next_is_told(env, repo, tmp_path):
    for i in range(6):
        saved(env, repo, f"old{i}", trap_session, repo, i, age_s=3600 * (10 - i))
    first = start(env, repo, "new0", tmp_path)
    assert first == {}                       # nothing learned yet when it started
    st = store_for(env, repo)
    assert sorted(st.read("folded.json")) == ["new0"] + [f"old{i}" for i in range(6)]
    assert st.read(history.MARKER)["sessions"] == 6
    second = start(env, repo, "new1", tmp_path)
    assert CLAIM in second["hookSpecificOutput"]["additionalContext"]
    assert "learned from 6 earlier session(s)" in second["systemMessage"]
    third = start(env, repo, "new2", tmp_path)
    assert "earlier session" not in third.get("systemMessage", "")     # said once


def test_only_this_repositorys_sessions_are_read_and_each_once(env, repo, tmp_path):
    other = make_repo(tmp_path, "other", "https://github.com/acme/gadget.git")
    saved(env, repo, "mine", trap_session, repo, 0)
    saved(env, other, "theirs", trap_session, other, 1)
    start(env, repo, "new0", tmp_path)
    st = store_for(env, repo)
    assert "theirs" not in st.read("folded.json")
    ledger = st.read_lines("ledger.jsonl")
    assert [r["session_id"] for r in ledger if r.get("history")] == ["mine"]
    ctx = LiveContext.build(ClaudeCode(), repo, env)
    assert not history.needed(ctx)
    assert history.run(ctx)["sessions"] == 1           # nothing new: still the one
    assert len(st.read_lines("ledger.jsonl")) == len(ledger)


def test_a_log_still_being_written_waits_until_it_goes_quiet(env, repo, tmp_path):
    path = saved(env, repo, "running", trap_session, repo, 0, age_s=5)
    start(env, repo, "new0", tmp_path)
    st = store_for(env, repo)
    assert "running" not in st.read("folded.json")
    assert st.read(history.MARKER)["waiting"] == [["running", path]]
    t = time.time() - 3600
    os.utime(path, (t, t))
    start(env, repo, "new1", tmp_path)
    assert "running" in st.read("folded.json")
    assert st.read(history.MARKER)["waiting"] == []


def test_history_off_learns_only_from_new_sessions(env, repo, tmp_path):
    saved(env, repo, "old", trap_session, repo, 0)
    env_off = dict(env, TE_HISTORY="off")
    tp = str(tmp_path / "n.jsonl")
    write_session(tp, [("hello", [])])
    run_session(env_off, repo, "new0", tp)
    st = store_for(env_off, repo)
    assert "old" not in (st.read("folded.json") or {})
    assert st.read(history.MARKER) is None


def test_saved_prompt_turns_teach_the_loop_guard_its_envelope(env, repo, tmp_path):
    turns = [(f"task {k}", [("Read", {"file_path": f"{repo}/a.py"}, "x", False)] * 3)
             for k in range(4)]
    saved(env, repo, "old", write_session, turns)
    start(env, repo, "new0", tmp_path)
    st = store_for(env, repo)
    tracker = loopguard.load_envelope(st)
    assert tracker.resolved_turns == [3, 3, 3, 3]
    assert st.read(history.MARKER)["turns"] == 4


def test_a_long_history_makes_at_most_one_model_call(env, repo, tmp_path, monkeypatch):
    calls = []
    monkeypatch.setattr(sidecall, "runner_for",
                        lambda *a, **k: lambda prompt, model: calls.append(1) or ("", 0.0))
    for i in range(12):
        saved(env, repo, f"old{i}", trap_session, repo, i, age_s=3600 * (20 - i))
    ctx = LiveContext.build(ClaudeCode(), repo, dict(env, TE_MICROSKILLS="on"))
    history.run(ctx)
    assert len(calls) <= 1


def test_te_history_dry_run_lists_without_folding(env, repo, tmp_path, capsys, monkeypatch):
    saved(env, repo, "old", trap_session, repo, 0)
    for k, v in env.items():
        monkeypatch.setenv(k, v)
    assert cli.main(["--host", "claude", "--cwd", repo, "history", "--dry-run"]) == 0
    assert "1 earlier session(s)" in capsys.readouterr().out
    assert store_for(env, repo).read("folded.json") is None


def test_prompt_turns_skip_subagents_and_mark_interruptions(tmp_path):
    lines = [
        {"type": "user", "message": {"content": "first"}},
        {"type": "assistant", "message": {"content": [
            {"type": "tool_use", "id": "a", "name": "Bash", "input": {"command": "ls"}}]}},
        {"type": "user", "message": {"content": [
            {"type": "tool_result", "tool_use_id": "a", "content": "Exit code 2\nno",
             "is_error": True}]}},
        {"type": "assistant", "isSidechain": True, "message": {"content": [
            {"type": "tool_use", "id": "s", "name": "Read", "input": {}}]}},
        {"type": "user", "message": {"content": [
            {"type": "text", "text": "[Request interrupted by user]"}]}},
        {"type": "user", "message": {"content": "second"}},
    ]
    p = tmp_path / "t.jsonl"
    p.write_text("\n".join(json.dumps(x) for x in lines) + "\n")
    turns = trace.claude_prompt_turns(str(p))
    assert turns == [
        {"calls": [("Bash", {"command": "ls"}, "Exit code 2\nno", False)], "interrupted": True},
        {"calls": [], "interrupted": False}]
