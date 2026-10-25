import asyncio
import json
import sqlite3

import httpx
import pytest

from custom_agent.api import create_app
from custom_agent.config import Settings
from custom_agent.runtime import Runtime
from tests.support import FakeModel, answer, calls, tool_results, wait_for


@pytest.fixture
async def api(tmp_path):
    settings = Settings(db_path=tmp_path / "agent.db", poll_interval=0.005)
    runtime = Runtime(settings, FakeModel())
    app = create_app(settings, runtime)
    async with app.router.lifespan_context(app):
        async with httpx.AsyncClient(
            transport=httpx.ASGITransport(app), base_url="http://test"
        ) as client:
            yield client, runtime


async def test_http_runs_ownership_validation_and_query(api):
    client, runtime = api
    assert (await client.get("/health")).json()["status"] == "ok"
    assert (
        await client.post("/v1/runs", json={"session_id": "s", "input": "hello"})
    ).status_code == 401
    headers = {"X-User-ID": "alice"}
    assert (
        await client.post("/v1/runs", headers=headers, json={"session_id": "s", "input": "  "})
    ).status_code == 422
    response = await client.post(
        "/v1/runs", headers=headers, json={"session_id": "s", "input": "hello", "request_id": "r"}
    )
    assert response.status_code == 202
    receipt = response.json()
    assert receipt["run_id"] and not receipt["task_ids"]
    await runtime.wait(receipt["run_id"])
    url = f"/v1/runs/{receipt['run_id']}"
    assert (await client.get(url, headers={"X-User-ID": "bob"})).status_code == 404
    result = (await client.get(url, headers=headers)).json()
    assert result["state"] == "COMPLETED"
    trace = (await client.get(url + "/trace", headers=headers)).json()
    assert trace["messages"] and trace["events"]
    assert "Authorization" not in json.dumps(trace)
    conflict = await client.post(
        "/v1/runs", headers=headers, json={"session_id": "s", "input": "other", "request_id": "r"}
    )
    assert conflict.status_code == 409


async def test_health_degrades_during_storage_failure_then_recovers(api):
    client, runtime = api
    first = runtime.submit("alice", "s", "first")
    second = runtime.submit("alice", "s", "second")
    runtime.store.db.execute(
        "CREATE TRIGGER reject_events BEFORE INSERT ON events "
        "BEGIN SELECT RAISE(ABORT,'private-health-error-marker'); END"
    )
    await wait_for(lambda: first["id"] in runtime._pending_failures)
    response = await client.get("/health")
    assert response.status_code == 503
    assert response.json() == {"status": "degraded", "error": "STORAGE_UNAVAILABLE"}
    runtime.store.db.execute("DROP TRIGGER reject_events")
    assert (await runtime.wait(second["id"], timeout=3))["state"] == "COMPLETED"
    response = await client.get("/health")
    assert response.status_code == 200 and response.json() == {"status": "ok"}


async def test_health_database_read_fault_is_safe(api, monkeypatch):
    client, runtime = api

    def broken(*args):
        raise sqlite3.OperationalError("private-read-error-marker")

    # Do not yield to the scheduler while its store reader is patched.
    with monkeypatch.context() as patch:
        patch.setattr(runtime.store, "one", broken)
        response = await client.get("/health")
    assert response.status_code == 503
    assert response.json() == {"status": "degraded", "error": "STORAGE_UNAVAILABLE"}
    assert (await client.get("/health")).status_code == 200


async def test_health_requires_a_running_coordinator(tmp_path):
    runtime = Runtime(Settings(db_path=tmp_path / "health.db"), FakeModel())
    app = create_app(runtime.settings, runtime)
    try:
        async with httpx.AsyncClient(
            transport=httpx.ASGITransport(app), base_url="http://test"
        ) as client:
            response = await client.get("/health")
            assert response.status_code == 503
            assert response.json()["error"] == "RUNTIME_NOT_RUNNING"
            await runtime.start()
            assert (await client.get("/health")).status_code == 200
            runtime._scheduler.cancel()
            await asyncio.gather(runtime._scheduler, return_exceptions=True)
            response = await client.get("/health")
            assert response.status_code == 503
            assert response.json()["error"] == "SCHEDULER_STOPPED"
            await runtime.stop()
            response = await client.get("/health")
            assert response.status_code == 503
            assert response.json()["error"] == "RUNTIME_NOT_RUNNING"
    finally:
        await runtime.stop()


@pytest.mark.parametrize(
    "arguments",
    ['{"title":"sk-fixturetrace012345",', "{", "[" * 1100],
    ids=["credential-redaction", "truncated", "over-nested"],
)
async def test_malformed_tool_arguments_trace_retains_safe_evidence_and_ownership(api, arguments):
    client, runtime = api

    def agent(messages):
        if tool_results(messages):
            return answer("handled")
        reply = calls(("todo", {}))
        reply.message["tool_calls"][0]["function"]["arguments"] = arguments
        return reply

    runtime.model.agent = agent
    run = runtime.submit("alice", "s", "malformed JSON")
    assert (await runtime.wait(run["id"], timeout=2))["state"] == "COMPLETED"
    url = f"/v1/runs/{run['id']}/trace"
    assert (await client.get(url, headers={"X-User-ID": "bob"})).status_code == 404
    response = await client.get(url, headers={"X-User-ID": "alice"})
    assert response.status_code == 200
    invocation = response.json()["invocations"][0]
    assert invocation["arguments"] == arguments.replace("sk-fixturetrace012345", "[REDACTED]")
    assert invocation["arguments_error"] == "INVALID_JSON"
    assert invocation["result"]["error"]["code"] == "INVALID_ARGUMENTS"
    assert invocation["attempts"] == 0
    assert "sk-fixturetrace012345" not in response.text


async def test_sse_scoped_replay_and_cursor_precedence(api):
    client, runtime = api
    run = runtime.submit("alice", "s", "hello")
    await runtime.wait(run["id"])
    other = runtime.submit("bob", "s", "private")
    await runtime.wait(other["id"])
    events = runtime.store.events(run["scope"])
    cursor = events[2]["id"]
    response = await client.get(
        "/v1/sessions/s/events?follow=false&after=0",
        headers={"X-User-ID": "alice", "Last-Event-ID": str(cursor)},
    )
    assert response.status_code == 200
    assert response.headers["content-type"].startswith("text/event-stream")
    frames = [block for block in response.text.split("\n\n") if block.startswith("id:")]
    actual_ids = [int(frame.splitlines()[0].removeprefix("id: ")) for frame in frames]
    assert actual_ids == [event["id"] for event in events if event["id"] > cursor]
    assert other["id"] not in response.text
    assert (
        await client.get(
            "/v1/sessions/s/events?follow=false",
            headers={"X-User-ID": "alice", "Last-Event-ID": "bad"},
        )
    ).status_code == 400
    assert (
        await client.get("/v1/sessions/missing/events?follow=false", headers={"X-User-ID": "alice"})
    ).status_code == 404


@pytest.mark.parametrize("event_count", [1, 201])
async def test_bounded_replay_excludes_events_appended_during_delivery(tmp_path, event_count):
    runtime = Runtime(Settings(db_path=tmp_path / "replay.db"), FakeModel())
    app = create_app(runtime.settings, runtime)
    run = runtime.submit("alice", "s", "hello")
    for _ in range(event_count - 1):
        runtime.store.event(run["scope"], run["id"], "test.initial")
    expected = [
        row["id"]
        for row in runtime.store.all(
            "SELECT id FROM events WHERE scope=? ORDER BY id", (run["scope"],)
        )
    ]

    class ConnectedRequest:
        async def is_disconnected(self):
            return False

    endpoint = next(
        route.endpoint for route in app.routes if route.path == "/v1/sessions/{session_id}/events"
    )
    try:
        response = await endpoint(
            "s", ConnectedRequest(), "alice", after=0, follow=False, last_event_id=None
        )
        actual = []
        async for frame in response.body_iterator:
            actual.append(int(frame.splitlines()[0].removeprefix("id: ")))
            runtime.store.event(run["scope"], run["id"], "test.new")
            if len(actual) > len(expected):
                break
        assert actual == expected
    finally:
        await runtime.stop()


async def test_task_query_ownership(api):
    client, runtime = api
    runtime.tools.search_delay = 0
    runtime.model.agent = lambda messages: (
        answer() if tool_results(messages) else calls(("search", {"query": "memory"}))
    )
    run = runtime.submit("alice", "s", "search")
    await runtime.wait(run["id"])
    task_id = runtime.store.one("SELECT id FROM tasks")["id"]
    url = f"/v1/tasks/{task_id}"
    assert (await client.get(url, headers={"X-User-ID": "bob"})).status_code == 404
    task = (await client.get(url, headers={"X-User-ID": "alice"})).json()
    assert task["state"] == "SUCCEEDED"
    assert task["result"]["data"]["mock"]


async def test_bearer_mode_does_not_trust_identity_header(tmp_path):
    settings = Settings(db_path=tmp_path / "auth.db", api_tokens={"test-secret": "alice"})
    runtime = Runtime(settings, FakeModel())
    app = create_app(settings, runtime)
    async with app.router.lifespan_context(app):
        async with httpx.AsyncClient(
            transport=httpx.ASGITransport(app), base_url="http://test"
        ) as client:
            body = {"session_id": "s", "input": "hello"}
            assert (
                await client.post("/v1/runs", headers={"Authorization": "test-secret"}, json=body)
            ).status_code == 401
            assert (
                await client.post("/v1/runs", headers={"X-User-ID": "alice"}, json=body)
            ).status_code == 401
            response = await client.post(
                "/v1/runs",
                headers={"Authorization": "Bearer test-secret", "X-User-ID": "bob"},
                json=body,
            )
            assert response.status_code == 202
            run = response.json()
            await runtime.wait(run["run_id"])
            assert runtime.store.owned_run(run["run_id"], "alice")
            assert not runtime.store.owned_run(run["run_id"], "bob")
            sessions = await client.get(
                "/v1/sessions",
                headers={"Authorization": "Bearer test-secret", "X-User-ID": "bob"},
            )
            assert sessions.status_code == 200 and sessions.json()["total"] == 1
            assert (
                await client.get("/v1/sessions", headers={"X-User-ID": "alice"})
            ).status_code == 401


async def test_session_endpoints_create_list_detail_rename_and_history(api):
    client, runtime = api
    headers = {"X-User-ID": "alice"}
    empty = await client.post("/v1/sessions", headers=headers, json={"title": "Empty"})
    assert empty.status_code == 201
    assert empty.json()["session_id"] and empty.json()["run_count"] == 0
    assert empty.json()["title"] == "Empty"
    created = await client.post(
        "/v1/sessions", headers=headers, json={"session_id": "work", "title": " Work "}
    )
    assert created.status_code == 201
    assert created.json()["title"] == "Work"
    assert "user_id" not in created.json() and "id" not in created.json()
    assert (
        await client.post("/v1/sessions", headers=headers, json={"session_id": "work"})
    ).status_code == 409
    first = runtime.submit("alice", "work", "first")
    second = runtime.submit("alice", "work", "second")
    await runtime.wait(second["id"])
    renamed = await client.patch("/v1/sessions/work", headers=headers, json={"title": " Release "})
    assert renamed.status_code == 200
    detail = (await client.get("/v1/sessions/work", headers=headers)).json()
    assert detail["title"] == "Release" and detail["session_id"] == "work"
    assert detail["run_count"] == 2 and detail["message_count"] == 4
    assert detail["active_runs"] == 0
    assert detail["last_run_id"] == second["id"] and detail["last_run_state"] == "COMPLETED"
    assert detail["created_at"] <= detail["updated_at"]
    listing = (await client.get("/v1/sessions?limit=1", headers=headers)).json()
    assert listing["total"] == 2 and listing["items"][0]["session_id"] == "work"
    page = (await client.get("/v1/sessions?limit=1&offset=1", headers=headers)).json()
    assert page["items"][0]["title"] == "Empty"
    messages = (await client.get("/v1/sessions/work/messages?limit=2", headers=headers)).json()
    assert len(messages["items"]) == 2 and messages["has_more"]
    assert messages["items"][0]["payload"]["content"] == "first"
    next_page = (
        await client.get(
            f"/v1/sessions/work/messages?after={messages['next_after']}", headers=headers
        )
    ).json()
    assert len(next_page["items"]) == 2 and not next_page["has_more"]
    assert next_page["items"][0]["payload"]["content"] == "second"
    runs = (await client.get("/v1/sessions/work/runs?limit=1", headers=headers)).json()
    assert runs["total"] == 2 and runs["items"][0]["run_id"] == second["id"]
    older = (await client.get("/v1/sessions/work/runs?limit=1&offset=1", headers=headers)).json()
    assert older["items"][0]["run_id"] == first["id"]


async def test_session_archive_restore_and_idempotent_submission(api):
    client, runtime = api
    headers = {"X-User-ID": "alice"}
    run = runtime.submit("alice", "work", "hello", "once")
    await runtime.wait(run["id"])
    response = await client.patch("/v1/sessions/work", headers=headers, json={"archived": True})
    assert response.status_code == 200 and response.json()["archived"]
    assert (await client.get("/v1/sessions", headers=headers)).json()["total"] == 0
    archived = (await client.get("/v1/sessions?status=archived", headers=headers)).json()
    assert archived["total"] == 1 and archived["items"][0]["session_id"] == "work"
    assert (await client.get("/v1/sessions?status=all", headers=headers)).json()["total"] == 1
    reject = await client.post(
        "/v1/runs", headers=headers, json={"session_id": "work", "input": "new"}
    )
    assert reject.status_code == 409 and reject.json()["detail"] == "SESSION_ARCHIVED"
    replay = await client.post(
        "/v1/runs",
        headers=headers,
        json={"session_id": "work", "input": "hello", "request_id": "once"},
    )
    assert replay.status_code == 202 and replay.json()["run_id"] == run["id"]
    conflict = await client.post(
        "/v1/runs",
        headers=headers,
        json={"session_id": "work", "input": "different", "request_id": "once"},
    )
    assert conflict.status_code == 409 and conflict.json()["detail"] == "REQUEST_ID_REUSED"
    assert (await client.get("/v1/sessions/work/messages", headers=headers)).status_code == 200
    restored = await client.patch("/v1/sessions/work", headers=headers, json={"archived": False})
    assert restored.status_code == 200 and not restored.json()["archived"]
    next_run = await client.post(
        "/v1/runs", headers=headers, json={"session_id": "work", "input": "new"}
    )
    assert next_run.status_code == 202
    assert (await runtime.wait(next_run.json()["run_id"]))["state"] == "COMPLETED"


async def test_archive_does_not_cancel_accepted_work_and_busy_delete_is_rejected(api):
    client, runtime = api
    headers = {"X-User-ID": "alice"}
    runtime.tools.search_delay = 0.15
    runtime.model.agent = lambda messages: (
        answer() if tool_results(messages) else calls(("search", {"query": "memory"}))
    )
    run = runtime.submit("alice", "work", "search")
    await wait_for(lambda: runtime.store.one("SELECT id FROM tasks WHERE state='RUNNING'"))
    second = runtime.submit("alice", "work", "queued")
    response = await client.patch("/v1/sessions/work", headers=headers, json={"archived": True})
    assert response.json()["active_runs"] == 2
    rejected = await client.delete("/v1/sessions/work", headers=headers)
    assert rejected.status_code == 409 and rejected.json()["detail"] == "SESSION_BUSY"
    assert (await runtime.wait(second["id"]))["state"] == "COMPLETED"
    assert runtime.store.run(run["id"])["state"] == "COMPLETED"
    task_id = runtime.store.one("SELECT id FROM tasks")["id"]
    assert (await client.delete("/v1/sessions/work", headers=headers)).status_code == 204
    assert (await client.get("/v1/runs/" + run["id"], headers=headers)).status_code == 404
    assert (await client.get("/v1/tasks/" + task_id, headers=headers)).status_code == 404
    assert (await client.get("/v1/sessions/work", headers=headers)).status_code == 404
    assert (
        await client.get("/v1/sessions/work/events?follow=false", headers=headers)
    ).status_code == 404
    recreated = await client.post("/v1/sessions", headers=headers, json={"session_id": "work"})
    assert recreated.status_code == 201 and recreated.json()["run_count"] == 0
    assert runtime.store.scope("alice", "work") != run["scope"]


@pytest.mark.parametrize("suffix", ["", "/messages", "/runs", "/memory", "/events?follow=false"])
async def test_session_reads_require_owner(api, suffix):
    client, runtime = api
    runtime.store.create_session("alice", "private", "Secret title")
    url = "/v1/sessions/private" + suffix
    assert (await client.get(url, headers={"X-User-ID": "bob"})).status_code == 404
    assert (await client.get(url)).status_code == 401
    listing = (await client.get("/v1/sessions?status=all", headers={"X-User-ID": "bob"})).json()
    assert listing["items"] == []


async def test_session_mutations_require_owner_and_preserve_other_users(api):
    client, runtime = api
    runtime.store.create_session("alice", "private", "Secret title")
    for body in ({"title": "Stolen"}, {"archived": True}):
        response = await client.patch(
            "/v1/sessions/private", headers={"X-User-ID": "bob"}, json=body
        )
        assert response.status_code == 404
    assert (
        await client.delete("/v1/sessions/private", headers={"X-User-ID": "bob"})
    ).status_code == 404
    assert runtime.store.owned_session("alice", "private")["title"] == "Secret title"
    assert (
        await client.post(
            "/v1/sessions",
            headers={"X-User-ID": "bob"},
            json={"session_id": "private", "title": "Bob"},
        )
    ).status_code == 201
    assert (
        await client.delete("/v1/sessions/private", headers={"X-User-ID": "bob"})
    ).status_code == 204
    assert runtime.store.owned_session("alice", "private")["title"] == "Secret title"


@pytest.mark.parametrize(
    "body",
    [
        {},
        {"title": " "},
        {"title": None},
        {"archived": None},
        {"archived": "yes"},
        {"title": "x" * 201},
        {"unexpected": True},
    ],
)
async def test_session_patch_validation(api, body):
    client, runtime = api
    runtime.store.create_session("alice", "work")
    assert (
        await client.patch("/v1/sessions/work", headers={"X-User-ID": "alice"}, json=body)
    ).status_code == 422


@pytest.mark.parametrize(
    "query", ["status=unknown", "limit=0", "limit=201", "offset=-1", "offset=9223372036854775808"]
)
async def test_session_pagination_validation(api, query):
    client, _ = api
    assert (
        await client.get("/v1/sessions?" + query, headers={"X-User-ID": "alice"})
    ).status_code == 422


async def test_existing_sse_stops_after_session_deleted_and_recreated(tmp_path):
    runtime = Runtime(Settings(db_path=tmp_path / "sse-delete.db"), FakeModel())
    app = create_app(runtime.settings, runtime)
    session = runtime.store.create_session("alice", "work")

    class Connected:
        async def is_disconnected(self):
            return False

    endpoint = next(
        route.endpoint for route in app.routes if route.path == "/v1/sessions/{session_id}/events"
    )
    try:
        response = await endpoint(
            "work", Connected(), "alice", after=0, follow=True, last_event_id=None
        )
        assert "session.created" in await anext(response.body_iterator)
        runtime.store.delete_session("alice", "work")
        new = runtime.store.create_session("alice", "work")
        assert new["id"] != session["id"]
        with pytest.raises(StopAsyncIteration):
            await anext(response.body_iterator)
    finally:
        await runtime.stop()


@pytest.mark.parametrize(
    "body",
    [
        {"session_id": ""},
        {"session_id": " "},
        {"session_id": "x" * 129},
        {"title": " "},
        {"title": "x" * 201},
        {"user_id": "other"},
    ],
)
async def test_session_create_validation_does_not_create_records(api, body):
    client, runtime = api
    response = await client.post("/v1/sessions", headers={"X-User-ID": "alice"}, json=body)
    assert response.status_code == 422
    assert runtime.store.list_sessions("alice")["total"] == 0


async def test_session_mutations_and_create_require_authentication(api):
    client, runtime = api
    runtime.store.create_session("alice", "work")
    assert (await client.post("/v1/sessions", json={})).status_code == 401
    assert (await client.patch("/v1/sessions/work", json={"title": "Changed"})).status_code == 401
    assert (await client.delete("/v1/sessions/work")).status_code == 401
    assert runtime.store.owned_session("alice", "work")["title"] == "work"
