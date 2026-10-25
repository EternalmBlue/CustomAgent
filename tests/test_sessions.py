import sqlite3

import pytest

from custom_agent.store import SCHEMA, SessionError, Store
from tests.support import calls, memory_entry


def seed_session(store, user="alice", session="work"):
    run = store.submit(user, session, "create todo and search")
    store.activate(run["id"])
    slot = store.reserve_decision(run["id"], 8)
    store.save_reply(
        run["id"],
        slot,
        calls(
            ("todo", {"action": "create", "title": "keep"}),
            ("search", {"query": "memory"}),
        ).message,
        {},
        "tool_calls",
    )
    todo, search = store.pending_calls(run["id"])

    def effect(db, scope):
        db.execute(
            "INSERT INTO todos(id,scope,title,created_at) VALUES (?,?,?,?)",
            (todo["id"], scope, "keep", 10),
        )
        return {"ok": True, "data": {}}

    store.complete_sync(todo["id"], effect)
    task = store.ensure_task(search)
    store.task_attempt(task["id"])
    store.complete_task(task["id"], {"ok": True, "data": {"mock": True}})
    store.save_reply(
        run["id"],
        store.reserve_decision(run["id"], 8),
        {"role": "assistant", "content": "done"},
        {},
        "stop",
    )
    source = store.history(run["scope"])[0]["id"]
    store.compact(
        run["scope"],
        1,
        [memory_entry("old", "decision", [source]), memory_entry("new", "decision", [source])],
    )
    return run


def test_upgrade_legacy_sessions_preserves_history_and_is_repeatable(tmp_path):
    path = tmp_path / "legacy.db"
    legacy_schema = SCHEMA.replace(
        " title TEXT NOT NULL DEFAULT '', archived INTEGER NOT NULL DEFAULT 0,\n"
        " created_at REAL NOT NULL DEFAULT 0, updated_at REAL NOT NULL DEFAULT 0,\n",
        "",
    )
    with sqlite3.connect(path) as db:
        db.executescript(legacy_schema)
        assert "title" not in {row[1] for row in db.execute("PRAGMA table_info(sessions)")}
        db.execute("INSERT INTO schema_version VALUES (1)")
        db.execute(
            "INSERT INTO sessions(id,user_id,session_id,compacted_seq) VALUES ('scope','u','s',3)"
        )
        db.execute(
            "INSERT INTO runs(id,scope,seq,input,state,created_at,updated_at) "
            "VALUES ('run','scope',1,'hello','COMPLETED',10,20)"
        )
        db.execute(
            "INSERT INTO messages(scope,run_id,payload,created_at) "
            """VALUES ('scope','run','{"role":"user","content":"hello"}',10)"""
        )
        db.execute("INSERT INTO sessions(id,user_id,session_id) VALUES ('empty','u','empty')")
    store = Store(path)
    try:
        session = store.owned_session("u", "s")
        assert session["title"] == "s" and not session["archived"]
        assert session["created_at"] == 10 and session["updated_at"] == 20
        assert session["compacted_seq"] == 3 and session["message_count"] == 1
        assert store.history("scope")[0]["payload"]["content"] == "hello"
        assert store.owned_session("u", "empty")["created_at"] > 0
        assert store.one("SELECT version FROM schema_version")["version"] == 3
        store.update_session("u", "s", title="Renamed", archived=True)
    finally:
        store.close()
    store = Store(path)
    try:
        assert store.owned_session("u", "s")["title"] == "Renamed"
        assert store.owned_session("u", "s")["archived"] == 1
        assert store.history("scope")[0]["payload"]["content"] == "hello"
    finally:
        store.close()


def test_delete_clears_all_owned_records_and_recreation_starts_clean(tmp_path):
    store = Store(tmp_path / "delete.db")
    try:
        run = seed_session(store)
        other = seed_session(store, user="bob")
        for table in (
            "sessions",
            "runs",
            "messages",
            "invocations",
            "tasks",
            "todos",
            "events",
            "memories",
            "execution_attempts",
        ):
            assert store.one(f"SELECT COUNT(*) n FROM {table}")["n"] > 0
        store.delete_session("alice", "work")
        assert store.run(run["id"]) is None and store.run(other["id"]) is not None
        assert not store.history(run["scope"])
        assert not store.memory_rows(run["scope"], include_superseded=True)
        assert not store.events(run["scope"])
        assert store.one("SELECT COUNT(*) n FROM todos")["n"] == 1
        assert store.one("SELECT COUNT(*) n FROM invocations")["n"] == 2
        assert store.one("SELECT COUNT(*) n FROM tasks")["n"] == 1
        assert not store.attempts(run["id"])
        assert store.attempts(other["id"])
        assert not store.all("PRAGMA foreign_key_check")
        recreated = store.submit("alice", "work", "fresh")
        assert recreated["seq"] == 1 and recreated["scope"] != run["scope"]
    finally:
        store.close()


def test_delete_failure_rolls_back_all_dependent_records(tmp_path):
    store = Store(tmp_path / "rollback.db")
    try:
        run = seed_session(store)
        tables = (
            "sessions",
            "runs",
            "messages",
            "invocations",
            "tasks",
            "todos",
            "events",
            "memories",
            "execution_attempts",
        )
        before = {table: store.all(f"SELECT * FROM {table}") for table in tables}
        store.db.execute(
            "CREATE TRIGGER fail_delete BEFORE DELETE ON sessions "
            "BEGIN SELECT RAISE(ABORT,'injected deletion failure'); END"
        )
        with pytest.raises(sqlite3.IntegrityError, match="injected deletion failure"):
            store.delete_session("alice", "work")
        assert {table: store.all(f"SELECT * FROM {table}") for table in tables} == before
        assert store.run(run["id"])["state"] == "COMPLETED"
        assert not store.all("PRAGMA foreign_key_check")
    finally:
        store.close()


@pytest.mark.parametrize("state", ["QUEUED", "RUNNING", "WAITING_TOOL"])
def test_busy_run_blocks_deletion_without_losing_any_data(tmp_path, state):
    store = Store(tmp_path / "busy.db")
    try:
        run = store.submit("alice", "work", "hello")
        store.db.execute("UPDATE runs SET state=? WHERE id=?", (state, run["id"]))
        with pytest.raises(SessionError, match="SESSION_BUSY"):
            store.delete_session("alice", "work")
        assert store.run(run["id"])["state"] == state
    finally:
        store.close()


def test_archived_queued_work_survives_restart_and_uses_original_scope(tmp_path):
    store = Store(tmp_path / "restart.db")
    run = store.submit("alice", "work", "hello")
    store.update_session("alice", "work", archived=True)
    store.close()
    store = Store(tmp_path / "restart.db")
    try:
        store.recover()
        assert store.runnable()[0]["id"] == run["id"]
        assert store.owned_session("alice", "work")["archived"]
        store.activate(run["id"])
        assert store.history(run["scope"])[0]["payload"]["content"] == "hello"
        with pytest.raises(SessionError, match="SESSION_ARCHIVED"):
            store.submit("alice", "work", "new")
    finally:
        store.close()


def test_unfinished_task_blocks_delete_even_after_inconsistent_terminal_run(tmp_path):
    store = Store(tmp_path / "unfinished-task.db")
    try:
        run = seed_session(store)
        store.db.execute("UPDATE tasks SET state='PENDING'")
        with pytest.raises(SessionError, match="SESSION_BUSY"):
            store.delete_session("alice", "work")
        assert store.run(run["id"]) is not None
    finally:
        store.close()
