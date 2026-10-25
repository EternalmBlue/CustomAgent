from __future__ import annotations

import json
import logging
import sqlite3
import time
import uuid
from contextlib import contextmanager
from pathlib import Path

from .llm import Json, json_text
from .tracing import Redactor, safe_code

log = logging.getLogger(__name__)

TERMINAL = ("COMPLETED", "INCOMPLETE", "FAILED")

SCHEMA = """
CREATE TABLE IF NOT EXISTS schema_version (version INTEGER PRIMARY KEY);
CREATE TABLE IF NOT EXISTS sessions (
 id TEXT PRIMARY KEY, user_id TEXT NOT NULL, session_id TEXT NOT NULL,
 title TEXT NOT NULL DEFAULT '', archived INTEGER NOT NULL DEFAULT 0,
 created_at REAL NOT NULL DEFAULT 0, updated_at REAL NOT NULL DEFAULT 0,
 compacted_seq INTEGER NOT NULL DEFAULT 0,
 UNIQUE(user_id, session_id)
);
CREATE TABLE IF NOT EXISTS runs (
 id TEXT PRIMARY KEY, scope TEXT NOT NULL REFERENCES sessions(id),
 seq INTEGER NOT NULL, request_id TEXT, input TEXT NOT NULL,
 state TEXT NOT NULL DEFAULT 'QUEUED', decisions INTEGER NOT NULL DEFAULT 0,
 answer TEXT, error TEXT, usage TEXT NOT NULL DEFAULT '{}',
 created_at REAL NOT NULL, updated_at REAL NOT NULL,
 UNIQUE(scope, seq), UNIQUE(scope, request_id)
);
CREATE INDEX IF NOT EXISTS runs_queue ON runs(scope, seq, state);
CREATE TABLE IF NOT EXISTS messages (
 id INTEGER PRIMARY KEY AUTOINCREMENT, scope TEXT NOT NULL REFERENCES sessions(id),
 run_id TEXT NOT NULL REFERENCES runs(id), payload TEXT NOT NULL, created_at REAL NOT NULL
);
CREATE INDEX IF NOT EXISTS messages_scope ON messages(scope, id);
CREATE TABLE IF NOT EXISTS invocations (
 id TEXT PRIMARY KEY, run_id TEXT NOT NULL REFERENCES runs(id),
 decision INTEGER NOT NULL, position INTEGER NOT NULL, call_id TEXT NOT NULL,
 name TEXT NOT NULL, arguments TEXT NOT NULL,
 state TEXT NOT NULL DEFAULT 'PENDING', result TEXT, attempts INTEGER NOT NULL DEFAULT 0,
 UNIQUE(run_id, decision, position)
);
CREATE TABLE IF NOT EXISTS tasks (
 id TEXT PRIMARY KEY, invocation_id TEXT NOT NULL UNIQUE REFERENCES invocations(id),
 run_id TEXT NOT NULL REFERENCES runs(id), state TEXT NOT NULL DEFAULT 'PENDING',
 attempts INTEGER NOT NULL DEFAULT 0, result TEXT, error TEXT,
 created_at REAL NOT NULL, updated_at REAL NOT NULL
);
CREATE TABLE IF NOT EXISTS todos (
 id TEXT PRIMARY KEY, scope TEXT NOT NULL REFERENCES sessions(id), title TEXT NOT NULL,
 completed INTEGER NOT NULL DEFAULT 0, created_at REAL NOT NULL
);
CREATE INDEX IF NOT EXISTS todos_scope ON todos(scope, created_at);
CREATE TABLE IF NOT EXISTS events (
 id INTEGER PRIMARY KEY AUTOINCREMENT, scope TEXT NOT NULL REFERENCES sessions(id),
 run_id TEXT REFERENCES runs(id), type TEXT NOT NULL, data TEXT NOT NULL,
 created_at REAL NOT NULL
);
CREATE INDEX IF NOT EXISTS events_scope ON events(scope, id);
CREATE TABLE IF NOT EXISTS memories (
 id INTEGER PRIMARY KEY AUTOINCREMENT, scope TEXT NOT NULL REFERENCES sessions(id),
 kind TEXT NOT NULL, key TEXT NOT NULL, text TEXT NOT NULL,
 entities TEXT NOT NULL, source_ids TEXT NOT NULL, valid_at TEXT,
 observed_at REAL NOT NULL, superseded_by INTEGER REFERENCES memories(id)
);
CREATE INDEX IF NOT EXISTS memories_scope ON memories(scope, key, superseded_by);
CREATE TABLE IF NOT EXISTS execution_attempts (
 id TEXT PRIMARY KEY, run_id TEXT NOT NULL REFERENCES runs(id),
 kind TEXT NOT NULL, name TEXT NOT NULL, decision INTEGER,
 invocation_id TEXT REFERENCES invocations(id), task_id TEXT REFERENCES tasks(id),
 attempt INTEGER NOT NULL, state TEXT NOT NULL DEFAULT 'RUNNING',
 started_at REAL NOT NULL, ended_at REAL, duration_ms REAL, interrupted_at REAL,
 error_code TEXT, request TEXT, response TEXT, usage TEXT, finish_reason TEXT,
 diagnostic TEXT
);
CREATE INDEX IF NOT EXISTS attempts_run ON execution_attempts(run_id, started_at);
CREATE INDEX IF NOT EXISTS attempts_invocation ON execution_attempts(invocation_id, attempt);
"""


class ConflictError(Exception):
    pass


class SessionError(Exception):
    def __init__(self, code: str):
        super().__init__(code)
        self.code = code


SESSION_SUMMARY = """
SELECT s.*,
 (SELECT COUNT(*) FROM runs r WHERE r.scope=s.id) run_count,
 (SELECT COUNT(*) FROM runs r WHERE r.scope=s.id
  AND r.state NOT IN ('COMPLETED','INCOMPLETE','FAILED')) active_runs,
 (SELECT COUNT(*) FROM messages m WHERE m.scope=s.id) message_count,
 (SELECT COUNT(*) FROM todos t WHERE t.scope=s.id) todo_count,
 (SELECT COUNT(*) FROM memories m WHERE m.scope=s.id AND m.superseded_by IS NULL) memory_count,
 (SELECT id FROM runs r WHERE r.scope=s.id ORDER BY seq DESC LIMIT 1) last_run_id,
 (SELECT state FROM runs r WHERE r.scope=s.id ORDER BY seq DESC LIMIT 1) last_run_state
FROM sessions s
"""


class Store:
    def __init__(self, path: Path | str, *, secrets=()):
        self.path = Path(path)
        self.redactor = Redactor(secrets)
        self._attempt_clocks: dict[str, float] = {}
        self._pending_attempt_logs: list[tuple] = []
        self._finished_attempt_clocks: set[str] = set()
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.db = sqlite3.connect(self.path, isolation_level=None)
        self.db.row_factory = sqlite3.Row
        self.db.execute("PRAGMA journal_mode=WAL")
        self.db.execute("PRAGMA foreign_keys=ON")
        self.db.execute("PRAGMA busy_timeout=5000")
        self.db.executescript(SCHEMA)
        try:
            self._upgrade_sessions()
        except BaseException:
            self.db.close()
            raise

    def _upgrade_sessions(self) -> None:
        with self.transaction():
            version = self.one("SELECT MAX(version) version FROM schema_version")["version"]
            if version == 3:
                return
            if version not in (None, 1, 2):
                raise RuntimeError("Unsupported database schema version")
            columns = {row["name"] for row in self.all("PRAGMA table_info(sessions)")}
            for name, definition in (
                ("title", "TEXT NOT NULL DEFAULT ''"),
                ("archived", "INTEGER NOT NULL DEFAULT 0"),
                ("created_at", "REAL NOT NULL DEFAULT 0"),
                ("updated_at", "REAL NOT NULL DEFAULT 0"),
            ):
                if name not in columns:
                    self.db.execute(f"ALTER TABLE sessions ADD COLUMN {name} {definition}")
            self.db.execute("UPDATE sessions SET title=session_id WHERE title=''")
            self.db.execute(
                "UPDATE sessions SET created_at=COALESCE("
                "(SELECT MIN(created_at) FROM runs WHERE scope=sessions.id),?) "
                "WHERE created_at=0",
                (time.time(),),
            )
            self.db.execute(
                "UPDATE sessions SET updated_at=COALESCE("
                "(SELECT MAX(updated_at) FROM runs WHERE scope=sessions.id),created_at) "
                "WHERE updated_at=0"
            )
            self.db.execute("DELETE FROM schema_version")
            self.db.execute("INSERT INTO schema_version VALUES (3)")

    @contextmanager
    def transaction(self):
        self.db.execute("BEGIN IMMEDIATE")
        try:
            yield self.db
            self.db.execute("COMMIT")
        except BaseException:
            self.db.execute("ROLLBACK")
            self._pending_attempt_logs.clear()
            self._finished_attempt_clocks.clear()
            raise
        for attempt_id in self._finished_attempt_clocks:
            self._attempt_clocks.pop(attempt_id, None)
        self._finished_attempt_clocks.clear()
        pending, self._pending_attempt_logs = self._pending_attempt_logs, []
        for level, payload in pending:
            log.log(level, "execution_attempt %s", payload)

    def close(self) -> None:
        self.db.close()

    def one(self, sql: str, args=()) -> Json | None:
        row = self.db.execute(sql, args).fetchone()
        return dict(row) if row else None

    def all(self, sql: str, args=()) -> list[Json]:
        return [dict(row) for row in self.db.execute(sql, args).fetchall()]

    def scope(self, user_id: str, session_id: str, *, create: bool = False) -> str | None:
        row = self.one(
            "SELECT id FROM sessions WHERE user_id=? AND session_id=?", (user_id, session_id)
        )
        if row:
            return row["id"]
        if not create:
            return None
        scope = uuid.uuid4().hex
        now = time.time()
        self.db.execute(
            "INSERT INTO sessions(id,user_id,session_id,title,created_at,updated_at) "
            "VALUES (?,?,?,?,?,?)",
            (scope, user_id, session_id, session_id, now, now),
        )
        self.event(scope, None, "session.created", {"session_id": session_id})
        return scope

    def owned_session(self, user_id: str, session_id: str) -> Json | None:
        return self.one(
            SESSION_SUMMARY + " WHERE s.user_id=? AND s.session_id=?", (user_id, session_id)
        )

    def create_session(self, user_id: str, session_id: str, title: str | None = None) -> Json:
        with self.transaction():
            if self.scope(user_id, session_id):
                raise SessionError("SESSION_EXISTS")
            scope = self.scope(user_id, session_id, create=True)
            if title is not None:
                self.db.execute("UPDATE sessions SET title=? WHERE id=?", (title, scope))
        return self.owned_session(user_id, session_id)

    def list_sessions(
        self, user_id: str, *, status: str = "active", limit: int = 50, offset: int = 0
    ) -> Json:
        if status not in {"active", "archived", "all"}:
            raise ValueError("Invalid session status")
        clause = " WHERE s.user_id=?"
        args: tuple = (user_id,)
        if status != "all":
            clause += " AND s.archived=?"
            args += (int(status == "archived"),)
        total = self.one("SELECT COUNT(*) n FROM sessions s" + clause, args)["n"]
        items = self.all(
            SESSION_SUMMARY + clause + " ORDER BY s.updated_at DESC,s.id LIMIT ? OFFSET ?",
            args + (limit, offset),
        )
        return {"items": items, "total": total, "limit": limit, "offset": offset}

    def update_session(
        self,
        user_id: str,
        session_id: str,
        *,
        title: str | None = None,
        archived: bool | None = None,
    ) -> Json:
        with self.transaction():
            session = self.owned_session(user_id, session_id)
            if session is None:
                raise SessionError("SESSION_NOT_FOUND")
            if title is not None and title != session["title"]:
                self.db.execute("UPDATE sessions SET title=? WHERE id=?", (title, session["id"]))
                self.event(session["id"], None, "session.renamed", {"title": title})
            if archived is not None and archived != bool(session["archived"]):
                self.db.execute(
                    "UPDATE sessions SET archived=? WHERE id=?", (int(archived), session["id"])
                )
                self.event(
                    session["id"], None, "session.archived" if archived else "session.restored"
                )
        return self.owned_session(user_id, session_id)

    def delete_session(self, user_id: str, session_id: str) -> None:
        with self.transaction():
            session = self.owned_session(user_id, session_id)
            if session is None:
                raise SessionError("SESSION_NOT_FOUND")
            scope = session["id"]
            unfinished = self.one(
                "SELECT t.id FROM tasks t JOIN runs r ON r.id=t.run_id "
                "WHERE r.scope=? AND t.state IN ('PENDING','RUNNING') LIMIT 1",
                (scope,),
            )
            if session["active_runs"] or unfinished:
                raise SessionError("SESSION_BUSY")
            self.db.execute("DELETE FROM messages WHERE scope=?", (scope,))
            self.db.execute("DELETE FROM events WHERE scope=?", (scope,))
            self.db.execute(
                "DELETE FROM execution_attempts WHERE run_id IN "
                "(SELECT id FROM runs WHERE scope=?)",
                (scope,),
            )
            self.db.execute(
                "DELETE FROM tasks WHERE run_id IN (SELECT id FROM runs WHERE scope=?)", (scope,)
            )
            self.db.execute(
                "DELETE FROM invocations WHERE run_id IN (SELECT id FROM runs WHERE scope=?)",
                (scope,),
            )
            self.db.execute("DELETE FROM todos WHERE scope=?", (scope,))
            self.db.execute("DELETE FROM memories WHERE scope=?", (scope,))
            self.db.execute("DELETE FROM runs WHERE scope=?", (scope,))
            self.db.execute("DELETE FROM sessions WHERE id=?", (scope,))

    def event(self, scope: str, run_id: str | None, kind: str, data: Json | None = None) -> int:
        now = time.time()
        cursor = self.db.execute(
            "INSERT INTO events(scope,run_id,type,data,created_at) VALUES (?,?,?,?,?)",
            (scope, run_id, kind, json_text(data or {}), now),
        )
        self.db.execute("UPDATE sessions SET updated_at=? WHERE id=?", (now, scope))
        return cursor.lastrowid

    def events(
        self, scope: str, after: int = 0, limit: int = 200, *, through: int = 2**63 - 1
    ) -> list[Json]:
        rows = self.all(
            "SELECT * FROM events WHERE scope=? AND id>? AND id<=? ORDER BY id LIMIT ?",
            (scope, after, through, limit),
        )
        for row in rows:
            row["data"] = json.loads(row["data"])
        return rows

    def attempts(self, run_id: str) -> list[Json]:
        rows = self.all("SELECT * FROM execution_attempts WHERE run_id=? ORDER BY rowid", (run_id,))
        for row in rows:
            for field in ("request", "response", "usage", "diagnostic"):
                row[field] = json.loads(row[field]) if row[field] is not None else None
        return rows

    def _attempt_event(self, attempt: Json, state: str, **data) -> None:
        run = self.run(attempt["run_id"])
        correlation = {
            key: attempt[key]
            for key in ("run_id", "kind", "name", "decision", "invocation_id", "task_id", "attempt")
        }
        payload = self.redactor.clean({"attempt_id": attempt["id"], **correlation, **data})
        self.event(run["scope"], run["id"], f"attempt.{state.lower()}", payload)
        level = logging.WARNING if state in {"FAILED", "INTERRUPTED"} else logging.INFO
        self._pending_attempt_logs.append((level, json_text({"state": state, **payload})))

    def _start_attempt(
        self,
        run_id: str,
        kind: str,
        name: str,
        attempt: int,
        *,
        decision=None,
        invocation_id=None,
        task_id=None,
        request=None,
    ) -> str:
        attempt_id = uuid.uuid4().hex
        self.db.execute(
            "INSERT INTO execution_attempts"
            "(id,run_id,kind,name,decision,invocation_id,task_id,attempt,started_at,request) "
            "VALUES (?,?,?,?,?,?,?,?,?,?)",
            (
                attempt_id,
                run_id,
                kind,
                self.redactor.clean(name),
                decision,
                invocation_id,
                task_id,
                attempt,
                time.time(),
                json_text(self.redactor.clean(request)),
            ),
        )
        self._attempt_clocks[attempt_id] = time.monotonic()
        self._attempt_event(
            self.one("SELECT * FROM execution_attempts WHERE id=?", (attempt_id,)), "STARTED"
        )
        return attempt_id

    def start_model_attempt(self, run_id: str, purpose: str, decision, request: Json) -> str:
        with self.transaction():
            attempt = (
                self.one(
                    "SELECT COUNT(*) n FROM execution_attempts "
                    "WHERE run_id=? AND kind='model' AND name=? AND decision IS ?",
                    (run_id, purpose, decision),
                )["n"]
                + 1
            )
            return self._start_attempt(
                run_id, "model", purpose, attempt, decision=decision, request=request
            )

    def _end_attempt(
        self,
        attempt_id: str,
        state: str,
        *,
        error_code=None,
        diagnostic=None,
        response=None,
        usage=None,
        finish_reason=None,
    ) -> None:
        attempt = self.one("SELECT * FROM execution_attempts WHERE id=?", (attempt_id,))
        if attempt is None or attempt["state"] != "RUNNING":
            return
        clock = self._attempt_clocks.get(attempt_id)
        interrupted = state == "INTERRUPTED"
        duration = (
            None if interrupted or clock is None else max(0, (time.monotonic() - clock) * 1000)
        )
        self.db.execute(
            "UPDATE execution_attempts SET state=?,ended_at=?,duration_ms=?,interrupted_at=?,"
            "error_code=?,response=?,usage=?,finish_reason=?,diagnostic=? WHERE id=?",
            (
                state,
                None if interrupted else time.time(),
                duration,
                time.time() if interrupted else None,
                self.redactor.clean(error_code),
                json_text(self.redactor.clean(response)),
                json_text(self.redactor.clean(usage)),
                self.redactor.clean(finish_reason),
                json_text(self.redactor.clean(diagnostic)),
                attempt_id,
            ),
        )
        self._attempt_event(
            attempt, state, error_code=error_code, duration_ms=duration, diagnostic=diagnostic
        )
        self._finished_attempt_clocks.add(attempt_id)

    def end_attempt(self, attempt_id: str, state: str, **outcome) -> None:
        with self.transaction():
            self._end_attempt(attempt_id, state, **outcome)

    def tool_attempt(self, invocation_id: str) -> Json | None:
        return self.one(
            "SELECT * FROM execution_attempts WHERE invocation_id=? ORDER BY attempt DESC LIMIT 1",
            (invocation_id,),
        )

    def _tool_outcome(self, invocation_id: str, result: Json) -> None:
        attempt = self.tool_attempt(invocation_id)
        if attempt:
            self._end_attempt(
                attempt["id"],
                "SUCCEEDED" if result["ok"] else "FAILED",
                response=result,
                error_code=None if result["ok"] else result["error"]["code"],
            )

    def fail_tool_attempt(self, invocation_id: str, exc: Exception) -> Json:
        attempt = self.tool_attempt(invocation_id)
        code = "TOOL_TIMEOUT" if isinstance(exc, TimeoutError) else safe_code(exc, "TOOL_ERROR")
        self.end_attempt(
            attempt["id"],
            "FAILED",
            error_code=code,
            diagnostic=self.redactor.diagnostic(exc),
            response={"ok": False, "error": {"code": code}},
        )
        return {"attempt_id": attempt["id"], "error_code": code}

    def submit(
        self, user_id: str, session_id: str, text: str, request_id: str | None = None
    ) -> Json:
        with self.transaction():
            scope = self.scope(user_id, session_id, create=True)
            if request_id:
                existing = self.one(
                    "SELECT * FROM runs WHERE scope=? AND request_id=?", (scope, request_id)
                )
                if existing:
                    if existing["input"] != text:
                        raise ConflictError("REQUEST_ID_REUSED")
                    return existing
            if self.one("SELECT archived FROM sessions WHERE id=?", (scope,))["archived"]:
                raise SessionError("SESSION_ARCHIVED")
            seq = self.one("SELECT COALESCE(MAX(seq),0)+1 n FROM runs WHERE scope=?", (scope,))["n"]
            run_id = uuid.uuid4().hex
            now = time.time()
            self.db.execute(
                "INSERT INTO runs(id,scope,seq,request_id,input,created_at,updated_at)"
                " VALUES (?,?,?,?,?,?,?)",
                (run_id, scope, seq, request_id, text, now, now),
            )
            self.event(scope, run_id, "run.queued", {"seq": seq})
        return self.run(run_id)

    def run(self, run_id: str) -> Json | None:
        return self.one("SELECT * FROM runs WHERE id=?", (run_id,))

    def owned_run(self, run_id: str, user_id: str) -> Json | None:
        return self.one(
            "SELECT r.*,s.session_id FROM runs r JOIN sessions s ON s.id=r.scope "
            "WHERE r.id=? AND s.user_id=?",
            (run_id, user_id),
        )

    def runnable(self) -> list[Json]:
        return self.all(
            "SELECT r.* FROM runs r WHERE r.state NOT IN ('COMPLETED','INCOMPLETE','FAILED') "
            "AND NOT EXISTS (SELECT 1 FROM runs older WHERE older.scope=r.scope "
            "AND older.seq<r.seq AND older.state NOT IN ('COMPLETED','INCOMPLETE','FAILED')) "
            "ORDER BY r.created_at,r.seq"
        )

    def message(self, scope: str, run_id: str, payload: Json) -> int:
        return self.db.execute(
            "INSERT INTO messages(scope,run_id,payload,created_at) VALUES (?,?,?,?)",
            (scope, run_id, json_text(payload), time.time()),
        ).lastrowid

    def history(self, scope: str, *, through_seq: int | None = None) -> list[Json]:
        sql = (
            "SELECT m.*,r.seq,r.state FROM messages m JOIN runs r ON r.id=m.run_id WHERE m.scope=?"
        )
        args: tuple = (scope,)
        if through_seq is not None:
            sql += " AND r.seq<=?"
            args += (through_seq,)
        rows = self.all(sql + " ORDER BY m.id", args)
        for row in rows:
            row["payload"] = json.loads(row["payload"])
        return rows

    def activate(self, run_id: str) -> None:
        with self.transaction():
            run = self.run(run_id)
            if run["state"] == "QUEUED":
                self.message(run["scope"], run_id, {"role": "user", "content": run["input"]})
                self.db.execute(
                    "UPDATE runs SET state='RUNNING',updated_at=? WHERE id=?",
                    (time.time(), run_id),
                )
                self.event(run["scope"], run_id, "run.started")

    def reserve_decision(self, run_id: str, limit: int) -> int | None:
        with self.transaction():
            run = self.run(run_id)
            if run["decisions"] >= limit:
                return None
            slot = run["decisions"] + 1
            self.db.execute(
                "UPDATE runs SET decisions=?,updated_at=? WHERE id=?", (slot, time.time(), run_id)
            )
            self.event(run["scope"], run_id, "model.requested", {"decision": slot})
            return slot

    def add_usage(self, run_id: str, usage: Json, purpose: str) -> None:
        run = self.run(run_id)
        totals = json.loads(run["usage"])
        bucket = totals.setdefault(purpose, {})
        for key, value in usage.items():
            if isinstance(value, (int, float)):
                bucket[key] = bucket.get(key, 0) + value
        self.db.execute("UPDATE runs SET usage=? WHERE id=?", (json_text(totals), run_id))

    def save_reply(
        self,
        run_id: str,
        slot: int,
        message: Json,
        usage: Json,
        finish_reason: str,
        *,
        attempt_id: str | None = None,
    ) -> None:
        with self.transaction():
            run = self.run(run_id)
            normalized = {"role": "assistant", "content": message.get("content")}
            calls = message.get("tool_calls", [])
            if calls:
                normalized["tool_calls"] = []
            for position, call in enumerate(calls):
                invocation_id = f"{run_id}:{slot}:{position}"
                call_id = f"call_{run_id}_{slot}_{position}"
                function = call["function"]
                arguments = function["arguments"]
                if not isinstance(arguments, str):
                    arguments = json_text(arguments)
                name = function["name"]
                normalized["tool_calls"].append(
                    {
                        "id": call_id,
                        "type": "function",
                        "function": {"name": name, "arguments": arguments},
                    }
                )
                self.db.execute(
                    "INSERT INTO invocations(id,run_id,decision,position,call_id,name,arguments)"
                    " VALUES (?,?,?,?,?,?,?)",
                    (invocation_id, run_id, slot, position, call_id, name, arguments),
                )
            self.message(run["scope"], run_id, normalized)
            if attempt_id is None:
                self.add_usage(run_id, usage, "agent")
            self.event(
                run["scope"],
                run_id,
                "model.completed",
                {
                    "decision": slot,
                    "attempt_id": attempt_id,
                    "tool_names": [c["function"]["name"] for c in calls],
                },
            )
            if not calls:
                state = "INCOMPLETE" if finish_reason == "length" else "COMPLETED"
                self._finish(run, state, normalized.get("content") or "", None)

    def pending_calls(self, run_id: str) -> list[Json]:
        return self.all(
            "SELECT * FROM invocations WHERE run_id=? AND state!='COMPLETED' "
            "ORDER BY decision,position",
            (run_id,),
        )

    def _finish(self, run: Json, state: str, answer: str | None, error: str | None) -> None:
        self.db.execute(
            "UPDATE runs SET state=?,answer=?,error=?,updated_at=? WHERE id=?",
            (state, answer, error, time.time(), run["id"]),
        )
        self.event(run["scope"], run["id"], f"run.{state.lower()}", {"error": error})

    def finish(
        self,
        run_id: str,
        state: str,
        answer: str | None,
        error: str | None,
        *,
        diagnostic: Json | None = None,
    ) -> None:
        with self.transaction():
            run = self.run(run_id)
            if run["state"] in TERMINAL:
                return
            if diagnostic is not None:
                self.event(run["scope"], run_id, "run.diagnostic", self.redactor.clean(diagnostic))
            for attempt in self.all(
                "SELECT id FROM execution_attempts WHERE run_id=? AND state='RUNNING'", (run_id,)
            ):
                self._end_attempt(attempt["id"], "INTERRUPTED", error_code="RUN_ABORTED")
            # Close abandoned tool groups so later runs keep a valid model protocol.
            for invocation in self.pending_calls(run_id):
                aborted = {
                    "ok": False,
                    "error": {
                        "code": "RUN_ABORTED",
                        "message": "Run ended before this invocation completed",
                    },
                }
                self._complete_call(invocation, aborted)
                task = self.one("SELECT * FROM tasks WHERE invocation_id=?", (invocation["id"],))
                if task and task["state"] not in ("SUCCEEDED", "FAILED"):
                    self.db.execute(
                        "UPDATE tasks SET state='FAILED',result=?,error='RUN_ABORTED',"
                        "updated_at=? WHERE id=?",
                        (json_text(aborted), time.time(), task["id"]),
                    )
                    self.event(
                        run["scope"],
                        run_id,
                        "task.failed",
                        {"task_id": task["id"], "error": "RUN_ABORTED"},
                    )
            if answer:
                self.message(run["scope"], run_id, {"role": "assistant", "content": answer})
            self._finish(run, state, answer, error)

    def _complete_call(self, invocation: Json, result: Json) -> None:
        run = self.run(invocation["run_id"])
        self.db.execute(
            "UPDATE invocations SET state='COMPLETED',result=? WHERE id=?",
            (json_text(result), invocation["id"]),
        )
        self.db.execute("UPDATE runs SET updated_at=? WHERE id=?", (time.time(), run["id"]))
        self.message(
            run["scope"],
            run["id"],
            {
                "role": "tool",
                "tool_call_id": invocation["call_id"],
                "content": json_text(result),
            },
        )
        self.event(
            run["scope"],
            run["id"],
            "tool.completed",
            {
                "invocation_id": invocation["id"],
                "decision": invocation["decision"],
                "name": invocation["name"],
                "ok": result["ok"],
                "attempt_id": (self.tool_attempt(invocation["id"]) or {}).get("id"),
                "task_id": (
                    self.one("SELECT id FROM tasks WHERE invocation_id=?", (invocation["id"],))
                    or {}
                ).get("id"),
            },
        )

    def complete_sync(self, invocation_id: str, handler) -> Json:
        invocation = self.one("SELECT * FROM invocations WHERE id=?", (invocation_id,))
        if invocation["state"] == "COMPLETED":
            return json.loads(invocation["result"])
        # Attempt facts survive rollback; Todo mutation and result remain one commit.
        self.readonly_attempt(invocation_id)
        try:
            with self.transaction():
                run = self.run(invocation["run_id"])
                result = handler(self.db, run["scope"])
                self._complete_call(invocation, result)
                self._tool_outcome(invocation_id, result)
                return result
        except Exception as exc:
            self.fail_tool_attempt(invocation_id, exc)
            raise

    def reject_call(self, invocation_id: str, result: Json) -> None:
        with self.transaction():
            invocation = self.one("SELECT * FROM invocations WHERE id=?", (invocation_id,))
            if invocation["state"] != "COMPLETED":
                self._complete_call(invocation, result)

    def readonly_attempt(self, invocation_id: str) -> int:
        with self.transaction():
            invocation = self.one("SELECT * FROM invocations WHERE id=?", (invocation_id,))
            run = self.run(invocation["run_id"])
            self.db.execute(
                "UPDATE invocations SET attempts=attempts+1 WHERE id=?", (invocation_id,)
            )
            attempt = invocation["attempts"] + 1
            attempt_id = self._start_attempt(
                run["id"],
                "tool",
                invocation["name"],
                attempt,
                decision=invocation["decision"],
                invocation_id=invocation_id,
                request={"arguments": json.loads(invocation["arguments"])},
            )
            self.event(
                run["scope"],
                run["id"],
                "tool.started",
                {
                    "invocation_id": invocation_id,
                    "name": invocation["name"],
                    "attempt": attempt,
                    "attempt_id": attempt_id,
                    "decision": invocation["decision"],
                },
            )
            return attempt

    def complete_readonly(self, invocation_id: str, result: Json) -> None:
        with self.transaction():
            invocation = self.one("SELECT * FROM invocations WHERE id=?", (invocation_id,))
            if invocation["state"] != "COMPLETED":
                self._complete_call(invocation, result)
                self._tool_outcome(invocation_id, result)

    def ensure_task(self, invocation: Json) -> Json:
        with self.transaction():
            task = self.one("SELECT * FROM tasks WHERE invocation_id=?", (invocation["id"],))
            if task:
                return task
            run = self.run(invocation["run_id"])
            task_id = uuid.uuid4().hex
            now = time.time()
            self.db.execute(
                "INSERT INTO tasks(id,invocation_id,run_id,created_at,updated_at)"
                " VALUES (?,?,?,?,?)",
                (task_id, invocation["id"], run["id"], now, now),
            )
            self.db.execute(
                "UPDATE runs SET state='WAITING_TOOL',updated_at=? WHERE id=?",
                (now, run["id"]),
            )
            self.event(
                run["scope"],
                run["id"],
                "task.accepted",
                {
                    "task_id": task_id,
                    "invocation_id": invocation["id"],
                    "decision": invocation["decision"],
                    "name": invocation["name"],
                },
            )
        return self.one("SELECT * FROM tasks WHERE id=?", (task_id,))

    def task_attempt(self, task_id: str) -> int:
        with self.transaction():
            task = self.one("SELECT * FROM tasks WHERE id=?", (task_id,))
            run = self.run(task["run_id"])
            attempt = task["attempts"] + 1
            self.db.execute(
                "UPDATE tasks SET state='RUNNING',attempts=?,updated_at=? WHERE id=?",
                (attempt, time.time(), task_id),
            )
            self.db.execute(
                "UPDATE invocations SET attempts=attempts+1 WHERE id=?", (task["invocation_id"],)
            )
            invocation = self.one("SELECT * FROM invocations WHERE id=?", (task["invocation_id"],))
            attempt_id = self._start_attempt(
                run["id"],
                "tool",
                invocation["name"],
                invocation["attempts"],
                decision=invocation["decision"],
                invocation_id=invocation["id"],
                task_id=task_id,
                request={"arguments": json.loads(invocation["arguments"])},
            )
            self.event(
                run["scope"],
                run["id"],
                "task.started",
                {
                    "task_id": task_id,
                    "attempt": attempt,
                    "attempt_id": attempt_id,
                    "invocation_id": invocation["id"],
                    "decision": invocation["decision"],
                },
            )
        return attempt

    def complete_task(self, task_id: str, result: Json) -> None:
        with self.transaction():
            task = self.one("SELECT * FROM tasks WHERE id=?", (task_id,))
            if task["state"] in ("SUCCEEDED", "FAILED"):
                return
            run = self.run(task["run_id"])
            state = "SUCCEEDED" if result["ok"] else "FAILED"
            self.db.execute(
                "UPDATE tasks SET state=?,result=?,error=?,updated_at=? WHERE id=?",
                (
                    state,
                    json_text(result),
                    None if result["ok"] else result["error"]["code"],
                    time.time(),
                    task_id,
                ),
            )
            invocation = self.one("SELECT * FROM invocations WHERE id=?", (task["invocation_id"],))
            self._complete_call(invocation, result)
            self._tool_outcome(invocation["id"], result)
            self.db.execute(
                "UPDATE runs SET state='RUNNING',updated_at=? WHERE id=?",
                (time.time(), run["id"]),
            )
            self.event(
                run["scope"],
                run["id"],
                "task.completed" if result["ok"] else "task.failed",
                {
                    "task_id": task_id,
                    "invocation_id": invocation["id"],
                    "attempt_id": (self.tool_attempt(invocation["id"]) or {}).get("id"),
                },
            )

    def recover(self) -> None:
        with self.transaction():
            for attempt in self.all("SELECT id FROM execution_attempts WHERE state='RUNNING'"):
                self._end_attempt(attempt["id"], "INTERRUPTED", error_code="PROCESS_INTERRUPTED")
            for task in self.all("SELECT * FROM tasks WHERE state IN ('PENDING','RUNNING')"):
                run = self.run(task["run_id"])
                self.db.execute("UPDATE tasks SET state='PENDING' WHERE id=?", (task["id"],))
                self.event(run["scope"], run["id"], "task.recovered", {"task_id": task["id"]})
            for run in self.all("SELECT * FROM runs WHERE state IN ('RUNNING','WAITING_TOOL')"):
                self.event(
                    run["scope"], run["id"], "run.recovered", {"decisions": run["decisions"]}
                )

    def owned_task(self, task_id: str, user_id: str) -> Json | None:
        return self.one(
            "SELECT t.* FROM tasks t JOIN runs r ON r.id=t.run_id "
            "JOIN sessions s ON s.id=r.scope WHERE t.id=? AND s.user_id=?",
            (task_id, user_id),
        )

    def memory_rows(self, scope: str, *, include_superseded: bool = False) -> list[Json]:
        suffix = "" if include_superseded else " AND superseded_by IS NULL"
        rows = self.all("SELECT * FROM memories WHERE scope=?" + suffix + " ORDER BY id", (scope,))
        for row in rows:
            row["entities"] = json.loads(row["entities"])
            row["source_ids"] = json.loads(row["source_ids"])
        return rows

    def compact(
        self, scope: str, through_seq: int, entries: list[Json], *, run_id: str | None = None
    ) -> None:
        with self.transaction():
            for entry in entries:
                previous = self.all(
                    "SELECT id,text FROM memories WHERE scope=? AND kind=? AND key=? "
                    "AND superseded_by IS NULL",
                    (scope, entry["kind"], entry["key"]),
                )
                new_id = self.db.execute(
                    "INSERT INTO memories(scope,kind,key,text,entities,source_ids,valid_at,"
                    "observed_at) VALUES (?,?,?,?,?,?,?,?)",
                    (
                        scope,
                        entry["kind"],
                        entry["key"],
                        entry["text"],
                        json_text(entry["entities"]),
                        json_text(entry["source_ids"]),
                        entry.get("valid_at"),
                        time.time(),
                    ),
                ).lastrowid
                for old in previous:
                    self.db.execute(
                        "UPDATE memories SET superseded_by=? WHERE id=?", (new_id, old["id"])
                    )
            self.db.execute(
                "UPDATE sessions SET compacted_seq=MAX(compacted_seq,?) WHERE id=?",
                (through_seq, scope),
            )
            self.event(
                scope,
                run_id,
                "memory.compacted",
                {"through_seq": through_seq, "entries": len(entries)},
            )
