import asyncio
import hmac
import json
import uuid
from contextlib import asynccontextmanager
from typing import Annotated, Literal

from fastapi import Depends, FastAPI, Header, HTTPException, Query, Request
from fastapi.responses import JSONResponse, Response, StreamingResponse
from pydantic import BaseModel, ConfigDict, Field, model_validator

from .config import Settings
from .llm import Json, json_text
from .runtime import Runtime
from .store import ConflictError, SessionError


class SubmitRun(BaseModel):
    model_config = ConfigDict(extra="forbid", str_strip_whitespace=True)
    session_id: str = Field(min_length=1, max_length=128)
    input: str = Field(min_length=1, max_length=16000)
    request_id: str | None = Field(default=None, min_length=1, max_length=128)


class CreateSession(BaseModel):
    model_config = ConfigDict(extra="forbid", str_strip_whitespace=True)
    session_id: str = Field(default_factory=lambda: uuid.uuid4().hex, min_length=1, max_length=128)
    title: str | None = Field(default=None, min_length=1, max_length=200)


class UpdateSession(BaseModel):
    model_config = ConfigDict(extra="forbid", str_strip_whitespace=True)
    title: str | None = Field(default=None, min_length=1, max_length=200)
    archived: bool | None = Field(default=None, strict=True)

    @model_validator(mode="after")
    def require_changes(self):
        if not self.model_fields_set or any(
            getattr(self, field) is None for field in self.model_fields_set
        ):
            raise ValueError("Provide a title and/or archived boolean; null is not allowed")
        return self


def public_session(session: Json) -> Json:
    return {
        key: session[key]
        for key in (
            "session_id",
            "title",
            "created_at",
            "updated_at",
            "run_count",
            "active_runs",
            "message_count",
            "todo_count",
            "memory_count",
            "last_run_id",
            "last_run_state",
        )
    } | {"archived": bool(session["archived"])}


def public_run(runtime: Runtime, run: Json) -> Json:
    return {
        key: run.get(key)
        for key in (
            "id",
            "session_id",
            "seq",
            "state",
            "decisions",
            "answer",
            "error",
            "created_at",
            "updated_at",
        )
    } | {
        "run_id": run["id"],
        "usage": json.loads(run["usage"]),
        "task_ids": [
            task["id"]
            for task in runtime.store.all("SELECT id FROM tasks WHERE run_id=?", (run["id"],))
        ],
    }


def create_app(settings: Settings | None = None, runtime: Runtime | None = None) -> FastAPI:
    settings = settings or Settings.from_file()
    runtime = runtime or Runtime(settings)

    @asynccontextmanager
    async def lifespan(app: FastAPI):
        try:
            await runtime.start()
            yield
        finally:
            await runtime.stop()

    app = FastAPI(title="CustomAgent", version="0.1.0", lifespan=lifespan)
    app.state.runtime = runtime

    @app.exception_handler(SessionError)
    async def session_error(request: Request, exc: SessionError):
        return JSONResponse(
            status_code=404 if exc.code == "SESSION_NOT_FOUND" else 409,
            content={"detail": exc.code},
        )

    async def identity(
        authorization: Annotated[str | None, Header()] = None,
        x_user_id: Annotated[str | None, Header()] = None,
    ) -> str:
        if settings.api_tokens:
            token = (
                authorization[7:] if authorization and authorization.startswith("Bearer ") else ""
            )
            for expected, user_id in settings.api_tokens.items():
                if hmac.compare_digest(token, expected):
                    return user_id
            raise HTTPException(401, "AUTH_REQUIRED")
        if not x_user_id or not x_user_id.strip() or len(x_user_id) > 128:
            raise HTTPException(401, "X_USER_ID_REQUIRED")
        return x_user_id

    User = Annotated[str, Depends(identity)]

    def find_run(run_id: str, user: str) -> Json:
        row = runtime.store.owned_run(run_id, user)
        if not row:
            raise HTTPException(404, "RUN_NOT_FOUND")
        return row

    def find_session(session_id: str, user: str) -> Json:
        session = runtime.store.owned_session(user, session_id)
        if session is None:
            raise SessionError("SESSION_NOT_FOUND")
        return session

    @app.get("/health")
    async def health():
        result = runtime.health()
        return JSONResponse(status_code=200 if result["status"] == "ok" else 503, content=result)

    @app.post("/v1/sessions", status_code=201)
    async def create_session(body: CreateSession, user: User):
        return public_session(runtime.store.create_session(user, body.session_id, body.title))

    @app.get("/v1/sessions")
    async def list_sessions(
        user: User,
        status: Literal["active", "archived", "all"] = "active",
        limit: int = Query(default=50, ge=1, le=200),
        offset: int = Query(default=0, ge=0, le=2**63 - 1),
    ):
        result = runtime.store.list_sessions(user, status=status, limit=limit, offset=offset)
        result["items"] = [public_session(session) for session in result["items"]]
        return result

    @app.get("/v1/sessions/{session_id}")
    async def get_session(session_id: str, user: User):
        return public_session(find_session(session_id, user))

    @app.patch("/v1/sessions/{session_id}")
    async def update_session(session_id: str, body: UpdateSession, user: User):
        return public_session(
            runtime.store.update_session(user, session_id, **body.model_dump(exclude_unset=True))
        )

    @app.delete("/v1/sessions/{session_id}", status_code=204)
    async def delete_session(session_id: str, user: User):
        runtime.store.delete_session(user, session_id)
        return Response(status_code=204)

    @app.get("/v1/sessions/{session_id}/messages")
    async def get_messages(
        session_id: str,
        user: User,
        after: int = Query(default=0, ge=0, le=2**63 - 1),
        limit: int = Query(default=100, ge=1, le=200),
    ):
        session = find_session(session_id, user)
        rows = runtime.store.all(
            "SELECT id,run_id,payload,created_at FROM messages "
            "WHERE scope=? AND id>? ORDER BY id LIMIT ?",
            (session["id"], after, limit + 1),
        )
        items = rows[:limit]
        for row in items:
            row["payload"] = json.loads(row["payload"])
        return {
            "items": items,
            "next_after": items[-1]["id"] if items else after,
            "has_more": len(rows) > limit,
        }

    @app.get("/v1/sessions/{session_id}/runs")
    async def get_session_runs(
        session_id: str,
        user: User,
        limit: int = Query(default=50, ge=1, le=200),
        offset: int = Query(default=0, ge=0, le=2**63 - 1),
    ):
        session = find_session(session_id, user)
        rows = runtime.store.all(
            "SELECT * FROM runs WHERE scope=? ORDER BY seq DESC LIMIT ? OFFSET ?",
            (session["id"], limit, offset),
        )
        return {
            "items": [public_run(runtime, row | {"session_id": session_id}) for row in rows],
            "total": session["run_count"],
            "limit": limit,
            "offset": offset,
        }

    @app.post("/v1/runs", status_code=202)
    async def submit(body: SubmitRun, user: User):
        try:
            run = runtime.submit(user, body.session_id, body.input, body.request_id)
        except ConflictError as exc:
            raise HTTPException(409, "REQUEST_ID_REUSED") from exc
        except ValueError as exc:
            raise HTTPException(422, "INVALID_INPUT") from exc
        return public_run(runtime, {**run, "session_id": body.session_id})

    @app.get("/v1/runs/{run_id}")
    async def get_run(run_id: str, user: User):
        return public_run(runtime, find_run(run_id, user))

    @app.get("/v1/tasks/{task_id}")
    async def get_task(task_id: str, user: User):
        task = runtime.store.owned_task(task_id, user)
        if not task:
            raise HTTPException(404, "TASK_NOT_FOUND")
        return {
            key: task[key]
            for key in ("id", "run_id", "state", "attempts", "error", "created_at", "updated_at")
        } | {"result": json.loads(task["result"]) if task["result"] else None}

    @app.get("/v1/runs/{run_id}/trace")
    async def trace(run_id: str, user: User):
        find_run(run_id, user)
        events = runtime.store.all("SELECT * FROM events WHERE run_id=? ORDER BY id", (run_id,))
        for event in events:
            event["data"] = json.loads(event["data"])
            event.pop("scope")
        messages = runtime.store.all(
            "SELECT id,payload,created_at FROM messages WHERE run_id=? ORDER BY id", (run_id,)
        )
        for message in messages:
            message["payload"] = json.loads(message["payload"])
        invocations = runtime.store.all(
            "SELECT * FROM invocations WHERE run_id=? ORDER BY decision,position", (run_id,)
        )
        tasks = runtime.store.all(
            "SELECT * FROM tasks WHERE run_id=? ORDER BY created_at,id", (run_id,)
        )
        for row in invocations + tasks:
            for field in ("arguments", "result"):
                if field in row and row[field] is not None:
                    try:
                        row[field] = json.loads(row[field])
                    except (json.JSONDecodeError, RecursionError):
                        if field != "arguments":
                            raise
                        row["arguments_error"] = "INVALID_JSON"
        return runtime.store.redactor.clean(
            {
                "run_id": run_id,
                "events": events,
                "messages": messages,
                "invocations": invocations,
                "tasks": tasks,
                "attempts": runtime.store.attempts(run_id),
            }
        )

    @app.get("/v1/sessions/{session_id}/memory")
    async def memory(session_id: str, user: User):
        scope = runtime.store.scope(user, session_id)
        if not scope:
            raise HTTPException(404, "SESSION_NOT_FOUND")
        entries = runtime.store.memory_rows(scope)
        for entry in entries:
            entry.pop("scope")
        return {"entries": entries}

    @app.get("/v1/sessions/{session_id}/events")
    async def stream(
        session_id: str,
        request: Request,
        user: User,
        after: int = Query(default=0, ge=0, le=2**63 - 1),
        follow: bool = True,
        last_event_id: Annotated[str | None, Header()] = None,
    ):
        scope = runtime.store.scope(user, session_id)
        if not scope:
            raise HTTPException(404, "SESSION_NOT_FOUND")
        try:
            cursor = int(last_event_id) if last_event_id is not None else after
            if not 0 <= cursor <= 2**63 - 1:
                raise ValueError
        except ValueError as exc:
            raise HTTPException(400, "INVALID_EVENT_CURSOR") from exc

        through = (
            2**63 - 1
            if follow
            else runtime.store.one(
                "SELECT COALESCE(MAX(id),0) id FROM events WHERE scope=?", (scope,)
            )["id"]
        )

        async def generate():
            nonlocal cursor
            idle = 0
            while not await request.is_disconnected():
                if runtime.store.one("SELECT id FROM sessions WHERE id=?", (scope,)) is None:
                    return
                rows = runtime.store.events(scope, cursor, through=through)
                for row in rows:
                    cursor = row["id"]
                    payload = {"run_id": row["run_id"], **row["data"]}
                    yield f"id: {cursor}\nevent: {row['type']}\ndata: {json_text(payload)}\n\n"
                if rows:
                    idle = 0
                    continue
                if not follow:
                    return
                idle += 1
                if idle >= 100:
                    yield ": heartbeat\n\n"
                    idle = 0
                await asyncio.sleep(0.1)

        return StreamingResponse(
            generate(),
            media_type="text/event-stream",
            headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"},
        )

    return app


def app_factory() -> FastAPI:
    return create_app()
