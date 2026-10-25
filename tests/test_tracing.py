import asyncio
import json
import logging
import sqlite3
from dataclasses import replace

import httpx
import pytest

from custom_agent.api import create_app
from custom_agent.config import Settings
from custom_agent.llm import ModelError, ModelReply, OpenAICompatibleModel
from custom_agent.runtime import Runtime
from custom_agent.store import Store
from custom_agent.tools import ToolError
from custom_agent.tracing import Redactor
from tests.support import FakeModel, answer, calls, tool_results, wait_for


@pytest.fixture
async def runtime(tmp_path):
    runtime = Runtime(
        Settings(db_path=tmp_path / "trace.db", poll_interval=0.005, search_delay=0),
        FakeModel(),
    )
    await runtime.start()
    try:
        yield runtime
    finally:
        await runtime.stop()


async def test_model_retry_preserves_cause_context_usage_and_decision(runtime):
    count = 0

    def agent(messages):
        nonlocal count
        count += 1
        return ModelError("MODEL_TEMP", transient=True) if count == 1 else answer("done")

    runtime.model.agent = agent
    run = runtime.submit("u", "s", "hello")
    result = await runtime.wait(run["id"])
    attempts = runtime.store.attempts(run["id"])
    assert [item["state"] for item in attempts] == ["FAILED", "SUCCEEDED"]
    assert [item["attempt"] for item in attempts] == [1, 2]
    assert {item["decision"] for item in attempts} == {1}
    assert attempts[0]["error_code"] == "MODEL_TEMP" and attempts[0]["usage"] is None
    assert attempts[1]["usage"] == json.loads(result["usage"])["agent"]
    assert attempts[1]["finish_reason"] == "stop"
    assert attempts[1]["response"]["content"] == "done"
    assert attempts[0]["request"] == attempts[1]["request"]
    retry = next(e for e in runtime.store.events(run["scope"]) if e["type"] == "model.retried")
    assert retry["data"]["attempt_id"] == attempts[0]["id"]
    assert retry["data"]["error_code"] == "MODEL_TEMP"
    assert not runtime.store._attempt_clocks


@pytest.mark.parametrize("name", ["weather", "search"])
async def test_tool_retry_retains_each_cause_and_explicit_correlations(runtime, monkeypatch, name):
    count = 0

    def flaky():
        nonlocal count
        count += 1
        if count == 1:
            raise ToolError("TEMP_TRACE", "do not persist raw exception text", transient=True)
        return {"ok": True, "data": {"mock": True}}

    if name == "weather":
        monkeypatch.setattr(runtime.tools, "execute_sync", lambda *args: flaky())
        args = {"city": "Shanghai"}
    else:

        async def execute(*args):
            return flaky()

        monkeypatch.setattr(runtime.tools, "execute_async", execute)
        args = {"query": "memory"}
    runtime.model.agent = lambda messages: (
        answer("done") if tool_results(messages) else calls((name, args))
    )
    run = runtime.submit("u", "s", "tool")
    await runtime.wait(run["id"])
    attempts = [item for item in runtime.store.attempts(run["id"]) if item["kind"] == "tool"]
    assert [item["state"] for item in attempts] == ["FAILED", "SUCCEEDED"]
    assert [item["attempt"] for item in attempts] == [1, 2]
    assert attempts[0]["error_code"] == "TEMP_TRACE"
    invocation = runtime.store.one("SELECT * FROM invocations")
    assert all(item["invocation_id"] == invocation["id"] for item in attempts)
    assert all(item["decision"] == invocation["decision"] for item in attempts)
    assert all(item["request"]["arguments"] == args for item in attempts)
    assert all(item["duration_ms"] >= 0 for item in attempts)
    assert "raw exception text" not in json.dumps(attempts)
    if name == "search":
        task = runtime.store.one("SELECT * FROM tasks")
        assert all(item["task_id"] == task["id"] for item in attempts)
    event_type = "task.retried" if name == "search" else "tool.retried"
    retry = next(e for e in runtime.store.events(run["scope"]) if e["type"] == event_type)
    assert retry["data"]["attempt_id"] == attempts[0]["id"]
    assert retry["data"]["error_code"] == "TEMP_TRACE"


@pytest.mark.parametrize(
    ("name", "args"), [("todo", {"action": "create"}), ("missing", {}), ("todo", "bad json")]
)
async def test_rejected_calls_have_no_executed_attempt(runtime, name, args):
    runtime.model.agent = lambda messages: (
        answer("handled") if tool_results(messages) else calls((name, args))
    )
    run = runtime.submit("u", "s", "invalid")
    await runtime.wait(run["id"])
    invocation = runtime.store.one("SELECT * FROM invocations")
    assert invocation["state"] == "COMPLETED" and invocation["attempts"] == 0
    assert all(item["kind"] == "model" for item in runtime.store.attempts(run["id"]))
    assert not runtime.store.all("SELECT * FROM todos")


async def test_invalid_model_response_keeps_actual_usage(runtime):
    runtime.model.agent = lambda messages: ModelReply(
        {"role": "assistant", "content": None}, {"prompt_tokens": 91}, "stop"
    )
    run = runtime.submit("u", "s", "invalid response")
    result = await runtime.wait(run["id"])
    attempt = runtime.store.attempts(run["id"])[0]
    assert attempt["state"] == "FAILED" and attempt["error_code"] == "MODEL_RESPONSE_INVALID"
    assert attempt["usage"] == {"prompt_tokens": 91}
    assert json.loads(result["usage"])["agent"] == attempt["usage"]
    assert attempt["response"]["content"] is None


async def test_adapter_rejected_response_retains_reported_usage_without_raw_body(tmp_path):
    model = OpenAICompatibleModel("fixture-credential", "https://example.test", "fixture-model")
    await model._client.aclose()
    model._client = httpx.AsyncClient(
        base_url="https://example.test/",
        transport=httpx.MockTransport(
            lambda request: httpx.Response(
                200,
                json={
                    "choices": [],
                    "usage": {
                        "prompt_tokens": 73,
                        "prompt_tokens_details": {"cached_tokens": 12},
                        "private_body": "provider-body-never-record",
                    },
                    "provider_body": "provider-body-never-record",
                },
            )
        ),
    )
    runtime = Runtime(Settings(db_path=tmp_path / "adapter-trace.db", poll_interval=0.005), model)
    await runtime.start()
    try:
        run = runtime.submit("u", "s", "hello")
        result = await runtime.wait(run["id"])
        attempt = runtime.store.attempts(run["id"])[0]
        assert result["error"] == "MODEL_RESPONSE_INVALID"
        assert attempt["request"]["model"] == "fixture-model"
        assert attempt["usage"] == {
            "prompt_tokens": 73,
            "prompt_tokens_details": {"cached_tokens": 12},
        }
        assert json.loads(result["usage"])["agent"]["prompt_tokens"] == 73
        assert "provider-body-never-record" not in json.dumps(runtime.store.attempts(run["id"]))
    finally:
        await runtime.stop()


@pytest.mark.parametrize("failure", ["unexpected", "timeout"])
async def test_async_failures_have_safe_diagnostics_and_timeout_timing(
    runtime, monkeypatch, failure
):
    async def broken(*args):
        if failure == "timeout":
            await asyncio.sleep(1)
        raise RuntimeError("private-provider-body-marker")

    monkeypatch.setattr(runtime.tools, "execute_async", broken)
    runtime.settings = replace(runtime.settings, tool_timeout=0.01)
    runtime.model.agent = lambda messages: (
        answer("handled") if tool_results(messages) else calls(("search", {"query": "memory"}))
    )
    run = runtime.submit("u", "s", "search")
    await runtime.wait(run["id"])
    attempts = [item for item in runtime.store.attempts(run["id"]) if item["kind"] == "tool"]
    assert len(attempts) == (2 if failure == "timeout" else 1)
    assert all(item["state"] == "FAILED" and item["duration_ms"] >= 0 for item in attempts)
    assert "private-provider-body-marker" not in json.dumps(attempts)
    assert attempts[0]["diagnostic"]["exception_type"] in {"RuntimeError", "TimeoutError"}
    assert attempts[0]["diagnostic"]["frames"]


async def test_memory_extraction_snapshot_and_rejected_extraction_usage(tmp_path):
    settings = Settings(db_path=tmp_path / "memory.db", recent_interactions=0)
    model = FakeModel(memory=lambda messages: answer("invalid JSON"))
    runtime = Runtime(settings, model)
    try:
        old = runtime.submit("u", "s", "old history")
        runtime.store.activate(old["id"])
        runtime.store.save_reply(old["id"], 1, answer("ack").message, {}, "stop")
        run = runtime.submit("u", "s", "current")
        runtime.store.activate(run["id"])
        assert not await runtime.memory.compact(run)
        attempt = runtime.store.attempts(run["id"])[0]
        assert attempt["name"] == "memory" and attempt["decision"] is None
        assert attempt["request"]["purpose"] == "memory"
        assert attempt["request"]["context"]["through_seq"] == old["seq"]
        assert json.loads(runtime.store.run(run["id"])["usage"]) == {"memory": attempt["usage"]}
        failure = next(
            e for e in runtime.store.events(run["scope"]) if e["type"] == "memory.failed"
        )
        assert failure["data"]["attempt_id"] == attempt["id"]
        assert runtime.store.one("SELECT compacted_seq FROM sessions")["compacted_seq"] == 0
    finally:
        await runtime.stop()


async def test_model_interruption_retains_uncertain_timing_and_recovers_budget(tmp_path):
    async def slow(messages):
        await asyncio.sleep(10)
        return answer("late")

    settings = Settings(db_path=tmp_path / "interrupt.db", poll_interval=0.005)
    runtime = Runtime(settings, FakeModel(slow))
    await runtime.start()
    run = runtime.submit("u", "s", "hello")
    await wait_for(lambda: runtime.store.attempts(run["id"]))
    await runtime.stop()
    recovered = Runtime(settings, FakeModel())
    await recovered.start()
    try:
        result = await recovered.wait(run["id"])
        attempts = recovered.store.attempts(run["id"])
        assert [item["state"] for item in attempts] == ["INTERRUPTED", "SUCCEEDED"]
        assert [item["decision"] for item in attempts] == [1, 2]
        assert attempts[0]["ended_at"] is None and attempts[0]["duration_ms"] is None
        assert attempts[0]["interrupted_at"] is not None
        assert attempts[0]["error_code"] == "PROCESS_INTERRUPTED"
        assert result["decisions"] == 2
    finally:
        await recovered.stop()


async def test_rich_trace_is_owned_redacted_and_correlated(tmp_path, caplog, monkeypatch):
    settings = Settings(
        db_path=tmp_path / "safe.db",
        api_key="fixture-provider-credential",
        api_tokens={"fixture-auth-credential": "alice"},
        poll_interval=0.005,
    )
    model = FakeModel(
        lambda messages: (
            answer("handled")
            if tool_results(messages)
            else calls(("search", {"query": "fixture-provider-credential"}))
        )
    )
    runtime = Runtime(settings, model)
    app = create_app(settings, runtime)

    async def broken(*args):
        raise RuntimeError("fixture-provider-credential provider-body-never-record")

    monkeypatch.setattr(runtime.tools, "execute_async", broken)
    caplog.set_level(logging.INFO, logger="custom_agent.store")
    async with app.router.lifespan_context(app):
        run = runtime.submit("alice", "s", "hello fixture-auth-credential")
        await runtime.wait(run["id"])
        async with httpx.AsyncClient(
            transport=httpx.ASGITransport(app), base_url="http://test"
        ) as client:
            url = f"/v1/runs/{run['id']}/trace"
            assert (await client.get(url)).status_code == 401
            response = await client.get(
                url, headers={"Authorization": "Bearer fixture-auth-credential"}
            )
            trace = response.json()
            assert response.status_code == 200
            assert trace["events"] and trace["messages"]
            invocation, task = trace["invocations"][0], trace["tasks"][0]
            tool = next(item for item in trace["attempts"] if item["kind"] == "tool")
            assert tool["task_id"] == task["id"]
            assert tool["invocation_id"] == task["invocation_id"] == invocation["id"]
            assert invocation["arguments"] == {"query": "[REDACTED]"}
            assert isinstance(task["result"], dict)
            assert isinstance(tool["response"], dict)
            persisted = runtime.store.all("SELECT * FROM execution_attempts")
            for serialized in (response.text, json.dumps(persisted), caplog.text):
                assert "fixture-provider-credential" not in serialized
                assert "fixture-auth-credential" not in serialized
                assert "provider-body-never-record" not in serialized
            assert tool["id"] in caplog.text and run["id"] in caplog.text
            assert "exception_type" in caplog.text
            other = runtime.submit("bob", "private", "private")
            await runtime.wait(other["id"])
            assert (
                await client.get(
                    f"/v1/runs/{other['id']}/trace",
                    headers={"Authorization": "Bearer fixture-auth-credential"},
                )
            ).status_code == 404


def test_redactor_handles_nested_and_serialized_secrets_and_reasoning():
    synthetic = "sk-" + "x" * 24
    redactor = Redactor(["fixture-secret"])
    value = {
        "reasoning_content": "hidden-one",
        "nested": {"api_key": "sensitive-value", "Authorization": "Bearer opaque-token"},
        "content": json.dumps(
            {"reasoning_details": "hidden-two", "text": synthetic + " fixture-secret"}
        ),
        "ordinary": "observable explanation",
    }
    serialized = json.dumps(redactor.clean(value))
    for forbidden in (
        "reasoning_content",
        "hidden-one",
        "hidden-two",
        synthetic,
        "sensitive-value",
        "opaque-token",
        "fixture-secret",
    ):
        assert forbidden not in serialized
    assert "observable explanation" in serialized


def test_version_two_upgrade_is_additive_without_fabricated_historical_attempts(tmp_path):
    path = tmp_path / "v2.db"
    store = Store(path)
    run = store.submit("u", "s", "history")
    store.activate(run["id"])
    store.save_reply(run["id"], 1, answer("old").message, {}, "stop")
    before = store.history(run["scope"])
    store.close()
    with sqlite3.connect(path) as db:
        db.execute("DROP TABLE execution_attempts")
        db.execute("UPDATE schema_version SET version=2")
    for _ in range(2):
        store = Store(path)
        try:
            assert store.one("SELECT version FROM schema_version")["version"] == 3
            assert store.history(run["scope"]) == before
            assert store.attempts(run["id"]) == []
            assert not store.all("PRAGMA foreign_key_check")
        finally:
            store.close()
