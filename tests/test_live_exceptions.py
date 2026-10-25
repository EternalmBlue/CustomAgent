"""Opt-in real-provider verification with explicitly injected boundary faults."""

import asyncio
import json
import os
import sqlite3
from contextlib import closing
from dataclasses import replace
from datetime import UTC, datetime
from pathlib import Path
from types import SimpleNamespace

import pytest

from custom_agent import cli
from custom_agent.llm import OpenAICompatibleModel
from custom_agent.runtime import Runtime
from tests.test_live_sessions import serve, until

pytestmark = [
    pytest.mark.live,
    pytest.mark.skipif(
        os.getenv("AGENT_LIVE") != "1", reason="Set AGENT_LIVE=1 to call the real API"
    ),
]


@pytest.fixture
def live_settings(tmp_path, provider_settings):
    settings = replace(
        provider_settings,
        db_path=tmp_path / "live-exceptions.db",
        workspace_root=Path(__file__).parents[1],
        api_tokens={},
        poll_interval=0.01,
        search_delay=2,
        tool_timeout=10,
    )
    return settings


@pytest.fixture
def evidence(request, live_settings):
    report = {
        "timestamp": datetime.now(UTC).isoformat(),
        "test": request.node.name,
        "model": live_settings.model,
        "provider": "real; no FakeModel",
        "database": "dedicated temporary SQLite",
        "main_server_restarted": False,
        "verified": False,
        "runs": [],
    }
    yield report
    filename = request.node.name.replace("[", "-").replace("]", "")
    Path(f"reports/live-exception-{filename}.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8"
    )


def record_trace(evidence, trace, settings):
    serialized = json.dumps(trace)
    if settings.api_key in serialized:
        pytest.fail("Provider credential leaked into trace")
    evidence["runs"].append(
        {
            "run_id": trace["run_id"],
            "attempts": [
                {
                    key: attempt[key]
                    for key in ("kind", "name", "state", "usage", "error_code", "duration_ms")
                }
                for attempt in trace["attempts"]
            ],
        }
    )
    assert any(
        item["kind"] == "model"
        and item["state"] == "SUCCEEDED"
        and item["usage"].get("prompt_tokens", 0) > 0
        for item in trace["attempts"]
        if item["usage"]
    )


async def trace_for(client, run_id):
    response = await client.get(f"/v1/runs/{run_id}/trace")
    assert response.status_code == 200
    return response.json()


async def terminal(client, run_id):
    return await until(
        client,
        f"/v1/runs/{run_id}",
        lambda run: run["state"] in {"COMPLETED", "FAILED", "INCOMPLETE"},
    )


async def test_live_storage_failure_recovers_fifo_without_duplicate_todo(live_settings, evidence):
    evidence["transport"] = "real loopback HTTP"
    evidence["fault"] = "SQLite trigger rejects event writes during local mock search"
    async with serve(live_settings) as client:
        assert (await client.get("/health")).status_code == 200
        response = await client.post(
            "/v1/runs",
            json={
                "session_id": "work",
                "input": (
                    "First use todo to create exactly one todo titled fault-once. "
                    "Then use search exactly once for durable execution. "
                    "Do both operations in that order."
                ),
            },
        )
        assert response.status_code == 202
        first = response.json()["run_id"]
        waiting = await until(
            client,
            f"/v1/runs/{first}",
            lambda run: run["state"] in {"WAITING_TOOL", "FAILED", "COMPLETED", "INCOMPLETE"},
        )
        assert waiting["state"] == "WAITING_TOOL"
        before = await trace_for(client, first)
        assert any(
            call["name"] == "todo"
            and call["arguments"]["action"] == "create"
            and call["result"]["ok"]
            for call in before["invocations"]
        )
        task_id = waiting["task_ids"][0]
        response = await client.post(
            "/v1/runs",
            json={
                "session_id": "work",
                "input": "Use todo to list the existing todos. Do not create any new todo.",
            },
        )
        assert response.status_code == 202 and response.json()["state"] == "QUEUED"
        second = response.json()["run_id"]
        with closing(sqlite3.connect(live_settings.db_path, isolation_level=None)) as db:
            db.execute(
                "CREATE TRIGGER live_reject_events BEFORE INSERT ON events "
                "BEGIN SELECT RAISE(ABORT,'injected live storage fault'); END"
            )
            try:
                async with asyncio.timeout(15):
                    while True:
                        health = await client.get("/health")
                        if health.status_code == 503:
                            break
                        await asyncio.sleep(0.02)
                assert health.json() == {"status": "degraded", "error": "STORAGE_UNAVAILABLE"}
                await asyncio.sleep(0.4)
                assert (await client.get("/health")).status_code == 503
                queued = (await client.get(f"/v1/runs/{second}")).json()
                assert queued["state"] == "QUEUED"
                evidence["health_during_fault"] = 503
            finally:
                db.execute("DROP TRIGGER live_reject_events")
        failed = await terminal(client, first)
        resumed = await terminal(client, second)
        assert failed["state"] == "FAILED" and failed["error"] == "RUNTIME_ERROR"
        assert resumed["state"] == "COMPLETED"
        assert (await client.get("/health")).status_code == 200
        first_trace = await trace_for(client, first)
        second_trace = await trace_for(client, second)
        record_trace(evidence, first_trace, live_settings)
        record_trace(evidence, second_trace, live_settings)
        assert all(call["state"] == "COMPLETED" for call in first_trace["invocations"])
        aborted_search = [call for call in first_trace["invocations"] if call["name"] == "search"]
        assert len(aborted_search) == 1
        assert aborted_search[0]["result"]["error"]["code"] == "RUN_ABORTED"
        task = (await client.get(f"/v1/tasks/{task_id}")).json()
        assert task["state"] == "FAILED" and task["error"] == "RUN_ABORTED"
        listing = [
            call
            for call in second_trace["invocations"]
            if call["name"] == "todo" and call["arguments"]["action"] == "list"
        ]
        assert listing
        items = listing[-1]["result"]["data"]["items"]
        assert len(items) == 1 and items[0]["title"] == "fault-once"
        failed_event = next(
            event["id"] for event in first_trace["events"] if event["type"] == "run.failed"
        )
        started_event = next(
            event["id"] for event in second_trace["events"] if event["type"] == "run.started"
        )
        assert failed_event < started_event
        evidence.update(
            verified=True,
            health_after_recovery=200,
            todo_count=len(items),
            first_state=failed["state"],
            queued_state=resumed["state"],
            failed_event=failed_event,
            next_started_event=started_event,
        )


@pytest.mark.parametrize("arguments", ["{", "[" * 1100], ids=["truncated", "over-nested"])
async def test_live_malformed_arguments_trace_and_provider_correction(
    live_settings, evidence, monkeypatch, arguments
):
    evidence["transport"] = "real loopback HTTP"
    evidence["fault"] = (
        "Mutate calculator arguments in the first actual provider tool reply; "
        "subsequent provider responses remain unchanged"
    )
    original = OpenAICompatibleModel.complete
    injected = False

    async def complete(model, *args, **kwargs):
        nonlocal injected
        reply = await original(model, *args, **kwargs)
        if not injected:
            for call in reply.message.get("tool_calls", []):
                if call["function"]["name"] == "calculator":
                    call["function"]["arguments"] = arguments
                    injected = True
                    break
        return reply

    monkeypatch.setattr(OpenAICompatibleModel, "complete", complete)
    async with serve(live_settings) as client:
        response = await client.post(
            "/v1/runs",
            json={
                "session_id": "work",
                "input": (
                    "Use calculator to compute (17+25)*3. If the tool rejects your "
                    "arguments, correct them and retry calculator, then report the result."
                ),
            },
        )
        assert response.status_code == 202
        run_id = response.json()["run_id"]
        result = await terminal(client, run_id)
        trace = await trace_for(client, run_id)
        record_trace(evidence, trace, live_settings)
        assert injected and result["state"] == "COMPLETED"
        malformed = [
            call for call in trace["invocations"] if call.get("arguments_error") == "INVALID_JSON"
        ]
        assert len(malformed) == 1
        assert malformed[0]["arguments"] == arguments
        assert malformed[0]["result"]["error"]["code"] == "INVALID_ARGUMENTS"
        assert malformed[0]["attempts"] == 0
        assert any(
            call["name"] == "calculator"
            and call["result"]["ok"]
            and call["result"]["data"]["value"] == 126
            for call in trace["invocations"]
        )
        denied = await client.get(f"/v1/runs/{run_id}/trace", headers={"X-User-ID": "other"})
        assert denied.status_code == 404
        evidence.update(
            verified=True,
            state=result["state"],
            trace_status=200,
            arguments_error="INVALID_JSON",
            rejected_attempts=0,
            corrected_calculator_value=126,
            other_user_trace_status=404,
        )


@pytest.mark.parametrize("close_fails", [False, True], ids=["normal-close", "close-raises"])
async def test_live_cli_startup_cleanup_and_real_replacement(
    live_settings, evidence, monkeypatch, close_fails
):
    evidence["transport"] = "CLI entry function and real provider HTTPS"
    evidence["fault"] = "Recovery raises before startup; optional close raises after HTTP close"
    runtime = Runtime(live_settings)
    original_close = runtime.model.close

    async def close():
        await original_close()
        if close_fails:
            raise RuntimeError("injected close error")

    def recover():
        raise sqlite3.OperationalError("injected startup failure")

    monkeypatch.setattr(runtime.model, "close", close)
    monkeypatch.setattr(runtime.store, "recover", recover)
    monkeypatch.setattr(cli, "Runtime", lambda settings: runtime)
    try:
        with pytest.raises(sqlite3.OperationalError, match="injected startup failure"):
            await cli.chat(SimpleNamespace())
        assert runtime._closed and runtime._lock.handle is None
        assert runtime.model._client.is_closed
        with pytest.raises(sqlite3.ProgrammingError):
            runtime.store.db.execute("SELECT 1")
    finally:
        await runtime.stop()
    replacement = Runtime(live_settings)
    try:
        await replacement.start()
        run = replacement.submit("live-user", "replacement", "Reply with exactly READY.")
        result = await replacement.wait(run["id"], timeout=180)
        assert result["state"] == "COMPLETED" and "READY" in result["answer"]
        record_trace(
            evidence,
            {"run_id": run["id"], "attempts": replacement.store.attempts(run["id"])},
            live_settings,
        )
        evidence.update(
            verified=True,
            original_error="OperationalError",
            http_client_closed=True,
            database_closed=True,
            lock_reacquired=True,
            replacement_state=result["state"],
        )
    finally:
        await replacement.stop()
