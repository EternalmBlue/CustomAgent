"""Opt-in real provider and real loopback HTTP session lifecycle test."""

import asyncio
import json
import os
import socket
import time
from contextlib import asynccontextmanager
from dataclasses import replace
from datetime import UTC, datetime
from pathlib import Path

import httpx
import pytest
import uvicorn

from custom_agent.api import create_app
from custom_agent.runtime import Runtime

pytestmark = [
    pytest.mark.live,
    pytest.mark.skipif(
        os.getenv("AGENT_LIVE") != "1", reason="Set AGENT_LIVE=1 to call the real API"
    ),
]


@asynccontextmanager
async def serve(settings):
    listener = socket.socket()
    listener.bind(("127.0.0.1", 0))
    runtime = Runtime(settings)
    server = uvicorn.Server(
        uvicorn.Config(create_app(settings, runtime), log_level="error", access_log=False)
    )
    task = asyncio.create_task(server.serve(sockets=[listener]))
    try:
        async with asyncio.timeout(15):
            while not server.started:
                if task.done():
                    await task
                    raise RuntimeError("Test server stopped before startup")
                await asyncio.sleep(0.01)
        url = f"http://127.0.0.1:{listener.getsockname()[1]}"
        async with httpx.AsyncClient(
            base_url=url, headers={"X-User-ID": "live-owner"}, trust_env=False, timeout=15
        ) as client:
            yield client
    finally:
        server.should_exit = True
        try:
            await asyncio.wait_for(task, timeout=15)
        finally:
            if not task.done():
                task.cancel()
                await asyncio.gather(task, return_exceptions=True)
            await runtime.stop()
            listener.close()


async def until(client, path, predicate):
    async with asyncio.timeout(180):
        while True:
            response = await client.get(path)
            assert response.status_code == 200
            result = response.json()
            if predicate(result):
                return result
            await asyncio.sleep(0.02)


async def test_live_http_sessions_archive_recovery_and_followups(tmp_path, provider_settings):
    settings = replace(
        provider_settings,
        db_path=tmp_path / "live-sessions.db",
        workspace_root=Path(__file__).parents[1],
        api_tokens={},
        search_delay=15,
        tool_timeout=30,
    )
    report = {
        "timestamp": datetime.now(UTC).isoformat(),
        "model": settings.model,
        "transport": "real loopback HTTP",
        "provider": "real; no FakeModel",
        "search_data": "local mock fixture",
        "database": "dedicated temporary SQLite",
        "main_server_restarted": False,
        "checks": [],
        "runs": [],
        "passed": False,
    }
    started = time.monotonic()

    async def submit(client, text, request_id):
        body = {"session_id": "work", "input": text, "request_id": request_id}
        response = await client.post("/v1/runs", json=body)
        assert response.status_code == 202
        return response.json(), body

    async def completed(client, receipt):
        result = await until(
            client,
            f"/v1/runs/{receipt['run_id']}",
            lambda row: row["state"] in {"COMPLETED", "FAILED", "INCOMPLETE"},
        )
        report["runs"].append(result)
        assert result["state"] == "COMPLETED", result.get("error")
        trace = (await client.get(f"/v1/runs/{receipt['run_id']}/trace")).json()
        assert any(a["kind"] == "model" and a["usage"] for a in trace["attempts"])
        return trace

    try:
        async with serve(settings) as client:
            created = await client.post(
                "/v1/sessions", json={"session_id": "work", "title": "Live session"}
            )
            assert created.status_code == 201
            receipt, _ = await submit(
                client, "Use todo to create exactly one todo titled live-release.", "create"
            )
            trace = await completed(client, receipt)
            assert any(i["name"] == "todo" for i in trace["invocations"])
            receipt, _ = await submit(
                client, "Use todo to complete the todo you just created.", "complete"
            )
            trace = await completed(client, receipt)
            assert any(
                i["name"] == "todo" and i["arguments"]["action"] == "complete" and i["result"]["ok"]
                for i in trace["invocations"]
            )
            report["checks"].append("real multi-turn Todo creation and completion")
            search, original = await submit(
                client,
                "Use search once to find durable execution; summarize the results.",
                "search",
            )
            waiting = await until(
                client,
                f"/v1/runs/{search['run_id']}",
                lambda row: row["state"] in {"WAITING_TOOL", "FAILED", "COMPLETED", "INCOMPLETE"},
            )
            assert waiting["state"] == "WAITING_TOOL" and waiting["task_ids"]
            task_id = waiting["task_ids"][0]
            await until(client, f"/v1/tasks/{task_id}", lambda row: row["state"] == "RUNNING")
            queued, _ = await submit(client, "Reply with exactly QUEUE_OK.", "queued")
            assert queued["state"] == "QUEUED"
            archived = await client.patch("/v1/sessions/work", json={"archived": True})
            assert archived.status_code == 200 and archived.json()["active_runs"] == 2
            blocked = await client.post("/v1/runs", json={"session_id": "work", "input": "new"})
            assert blocked.status_code == 409 and blocked.json()["detail"] == "SESSION_ARCHIVED"
            busy = await client.delete("/v1/sessions/work")
            assert busy.status_code == 409 and busy.json()["detail"] == "SESSION_BUSY"
            replay = await client.post("/v1/runs", json=original)
            assert replay.status_code == 202 and replay.json()["run_id"] == search["run_id"]
            assert (await client.get("/v1/sessions")).json()["total"] == 0
            assert (await client.get("/v1/sessions?status=archived")).json()["total"] == 1
            report["checks"].append("archive, replay and busy-delete during real async invocation")

        # The first runtime stops with accepted work persisted in its temporary DB.
        async with serve(settings) as client:
            trace = await completed(client, search)
            assert any(i["name"] == "search" for i in trace["invocations"])
            task = (await client.get(f"/v1/tasks/{task_id}")).json()
            assert task["state"] == "SUCCEEDED" and task["attempts"] == 2
            assert any(e["type"] == "task.recovered" for e in trace["events"])
            await completed(client, queued)
            report["checks"].append("archived async work and queued run survive graceful restart")
            renamed = await client.patch(
                "/v1/sessions/work", json={"title": "Recovered session", "archived": False}
            )
            assert renamed.status_code == 200 and renamed.json()["title"] == "Recovered session"
            receipt, _ = await submit(
                client,
                "Use todo to list the todos and say whether live-release is completed.",
                "list",
            )
            trace = await completed(client, receipt)
            listed = [
                i
                for i in trace["invocations"]
                if i["name"] == "todo" and i["arguments"]["action"] == "list"
            ]
            assert listed and listed[-1]["result"]["data"]["items"][0]["completed"]
            detail = (await client.get("/v1/sessions/work")).json()
            assert detail["run_count"] == 5 and detail["active_runs"] == 0
            assert (await client.get("/v1/sessions/work/messages?limit=2")).json()["has_more"]
            assert (await client.get("/v1/sessions/work/runs")).json()["total"] == 5
            report["checks"].append("restore, rename, preserved Todo and paginated history")
            for path in (
                "/v1/sessions/work",
                f"/v1/runs/{search['run_id']}",
                f"/v1/tasks/{task_id}",
            ):
                assert (await client.get(path, headers={"X-User-ID": "other"})).status_code == 404
            events = await client.get("/v1/sessions/work/events?follow=false")
            assert events.status_code == 200
            ids = [
                int(line.removeprefix("id: "))
                for line in events.text.splitlines()
                if line.startswith("id: ")
            ]
            assert ids and ids == sorted(set(ids))
            replay = await client.get(
                "/v1/sessions/work/events?follow=false",
                headers={"Last-Event-ID": str(ids[-1])},
            )
            assert replay.status_code == 200 and not replay.text
            assert (await client.delete("/v1/sessions/work")).status_code == 204
            assert (await client.get(f"/v1/tasks/{task_id}")).status_code == 404
            assert (await client.get("/v1/sessions/work/messages")).status_code == 404
            report["checks"].append("ownership, SSE cursor replay and idle deletion")
            report["passed"] = True
    finally:
        report["seconds"] = round(time.monotonic() - started, 3)
        await asyncio.to_thread(
            Path("reports/live-sessions.json").write_text,
            json.dumps(report, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )
