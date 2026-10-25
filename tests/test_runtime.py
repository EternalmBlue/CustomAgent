import asyncio
import json
import sqlite3
from dataclasses import replace

import pytest

from custom_agent.config import Settings
from custom_agent.llm import ModelError
from custom_agent.runtime import Runtime
from custom_agent.store import ConflictError
from custom_agent.tools import ToolError
from tests.support import FakeModel, answer, calls, current_input, tool_results, wait_for


@pytest.fixture
async def runtime(tmp_path):
    settings = Settings(db_path=tmp_path / "agent.db", poll_interval=0.005)
    runtime = Runtime(settings, FakeModel())
    await runtime.start()
    yield runtime
    await runtime.stop()


async def test_submission_idempotency_and_conflict(runtime):
    first = runtime.submit("u", "s", "hello", "client-id")
    repeated = runtime.submit("u", "s", "hello", "client-id")
    assert repeated["id"] == first["id"]
    with pytest.raises(ConflictError):
        runtime.submit("u", "s", "different", "client-id")
    result = await runtime.wait(first["id"])
    assert result["decisions"] == 1
    assert len(runtime.store.all("SELECT * FROM runs")) == 1


async def test_second_runtime_cannot_open_coordinator(runtime):
    other = Runtime(runtime.settings, FakeModel())
    try:
        with pytest.raises(RuntimeError, match="already owns"):
            await other.start()
    finally:
        await other.stop()


async def test_different_database_extensions_have_independent_locks(tmp_path):
    first = Runtime(Settings(db_path=tmp_path / "same.db"), FakeModel())
    second = Runtime(Settings(db_path=tmp_path / "same.sqlite3"), FakeModel())
    try:
        await first.start()
        await second.start()
        assert first._lock.path != second._lock.path
        run = second.submit("u", "s", "ok")
        assert (await second.wait(run["id"], timeout=2))["state"] == "COMPLETED"
    finally:
        await first.stop()
        await second.stop()


async def test_registered_tool_dispatches_without_changing_runtime_loop(runtime):
    runtime.tools.register(
        {
            "name": "test_echo",
            "description": "Test registration only",
            "parameters": {
                "type": "object",
                "properties": {"value": {"type": "string"}},
                "required": ["value"],
                "additionalProperties": False,
            },
        },
        mode="readonly",
        handler=lambda args, db, scope: {"echo": args["value"]},
    )
    runtime.model.agent = lambda messages: (
        answer("done") if tool_results(messages) else calls(("test_echo", {"value": "registered"}))
    )
    run = runtime.submit("u", "s", "echo")
    assert (await runtime.wait(run["id"], timeout=2))["state"] == "COMPLETED"
    result = runtime.store.one("SELECT result FROM invocations WHERE name='test_echo'")
    assert json.loads(result["result"])["data"]["echo"] == "registered"


async def test_transient_readonly_tool_failure_retries_once(runtime, monkeypatch):
    original = runtime.tools._handlers["weather"]
    attempts = 0

    def flaky(args, db, scope):
        nonlocal attempts
        attempts += 1
        if attempts == 1:
            raise ToolError("TEMP", "Temporary failure", transient=True)
        return original(args, db, scope)

    monkeypatch.setitem(runtime.tools._handlers, "weather", flaky)
    runtime.model.agent = lambda messages: (
        answer("done") if tool_results(messages) else calls(("weather", {"city": "Shanghai"}))
    )
    run = runtime.submit("u", "s", "weather")
    assert (await runtime.wait(run["id"], timeout=2))["state"] == "COMPLETED"
    assert attempts == 2
    assert runtime.store.one("SELECT attempts FROM invocations")["attempts"] == 2


async def test_transient_model_retry_uses_one_slot(runtime):
    attempts = 0

    def agent(messages):
        nonlocal attempts
        attempts += 1
        return ModelError("TEMP", transient=True) if attempts == 1 else answer("ok")

    runtime.model.agent = agent
    run = runtime.submit("u", "s", "hello")
    result = await runtime.wait(run["id"])
    assert result["state"] == "COMPLETED"
    assert result["decisions"] == 1 and attempts == 2


async def test_failed_run_unblocks_next_queued_run(runtime):
    runtime.model.agent = lambda messages: ModelError("DENIED")
    first = runtime.submit("u", "s", "first")
    result = await runtime.wait(first["id"])
    assert result["state"] == "FAILED"
    assert result["error"] == "DENIED"
    runtime.model.agent = lambda messages: answer("next")
    second = runtime.submit("u", "s", "next")
    assert (await runtime.wait(second["id"]))["state"] == "COMPLETED"


async def test_tool_timeout_retry_and_recovery_feedback(runtime):
    runtime.settings = replace(runtime.settings, tool_timeout=0.01)
    runtime.tools.search_delay = 1
    runtime.model.agent = lambda messages: (
        answer("tool failed") if tool_results(messages) else calls(("search", {"query": "memory"}))
    )
    run = runtime.submit("u", "s", "search")
    result = await runtime.wait(run["id"])
    task = runtime.store.one("SELECT * FROM tasks")
    assert result["state"] == "COMPLETED"
    assert task["state"] == "FAILED" and task["attempts"] == 2
    assert json.loads(task["result"])["error"]["code"] == "TOOL_TIMEOUT"
    assert tool_results(runtime.model.requests[-1]["messages"])[-1]["ok"] is False


async def test_unexpected_async_tool_exception_finalizes_task_and_returns_feedback(
    runtime, monkeypatch
):
    async def broken(name, args):
        raise RuntimeError("Internal diagnostic must not leak into protocol")

    monkeypatch.setattr(runtime.tools, "execute_async", broken)
    runtime.model.agent = lambda messages: (
        answer("handled") if tool_results(messages) else calls(("search", {"query": "memory"}))
    )
    run = runtime.submit("u", "s", "search")
    result = await runtime.wait(run["id"], timeout=2)
    assert result["state"] == "COMPLETED"
    task = runtime.store.one("SELECT * FROM tasks")
    assert task["state"] == "FAILED"
    assert task["error"] == "TOOL_ERROR"
    assert "Internal diagnostic" not in task["result"]


async def test_failed_tool_run_closes_protocol_before_next_session_input(runtime, monkeypatch):
    def broken(*args):
        raise RuntimeError("Transactional tool failed")

    monkeypatch.setattr(runtime.tools, "execute_sync", broken)

    def agent(messages):
        if current_input(messages) == "second":
            pending = set()
            for message in messages:
                if message.get("tool_calls"):
                    pending.update(call["id"] for call in message["tool_calls"])
                elif message["role"] == "tool":
                    pending.remove(message["tool_call_id"])
            assert not pending
            return answer("second works")
        return calls(
            ("todo", {"action": "create", "title": "must roll back"}),
            ("calculator", {"expression": "2+3"}),
        )

    runtime.model.agent = agent
    first = runtime.submit("u", "s", "first")
    second = runtime.submit("u", "s", "second")
    assert (await runtime.wait(first["id"], timeout=2))["state"] == "FAILED"
    assert (await runtime.wait(second["id"], timeout=2))["state"] == "COMPLETED"
    assert not runtime.store.all("SELECT * FROM todos")
    assert not runtime.store.pending_calls(first["id"])


async def test_repeated_storage_failure_defers_finalization_without_repeating_effects(
    runtime, monkeypatch, caplog
):
    release_search = asyncio.Event()
    entered_search = asyncio.Event()
    executed = []
    original_sync = runtime.tools.execute_sync
    original_finish = runtime.store.finish
    finalizations = 0

    def execute_sync(name, *args):
        executed.append(name)
        return original_sync(name, *args)

    async def execute_async(name, args):
        executed.append(name)
        entered_search.set()
        await release_search.wait()
        return {"ok": True, "data": {"mock": True}}

    def finish(*args, **kwargs):
        nonlocal finalizations
        finalizations += 1
        return original_finish(*args, **kwargs)

    def agent(messages):
        if current_input(messages) == "second":
            pending = set()
            for message in messages:
                if message.get("tool_calls"):
                    pending.update(call["id"] for call in message["tool_calls"])
                elif message["role"] == "tool":
                    pending.remove(message["tool_call_id"])
            assert not pending
            return answer("next works")
        return calls(
            ("todo", {"action": "create", "title": "committed once"}),
            ("search", {"query": "memory"}),
            ("calculator", {"expression": "1+1"}),
        )

    monkeypatch.setattr(runtime.tools, "execute_sync", execute_sync)
    monkeypatch.setattr(runtime.tools, "execute_async", execute_async)
    monkeypatch.setattr(runtime.store, "finish", finish)
    runtime.model.agent = agent
    first = runtime.submit("u", "s", "first")
    second = runtime.submit("u", "s", "second")
    await asyncio.wait_for(entered_search.wait(), timeout=2)
    runtime.store.db.execute(
        "CREATE TRIGGER reject_events BEFORE INSERT ON events "
        "BEGIN SELECT RAISE(ABORT,'private-storage-error-marker'); END"
    )
    release_search.set()
    await wait_for(lambda: finalizations >= 3)
    assert not runtime._scheduler.done()
    assert runtime.health() == {"status": "degraded", "error": "STORAGE_UNAVAILABLE"}
    assert runtime.store.run(second["id"])["state"] == "QUEUED"
    assert not runtime.store.all("SELECT * FROM messages WHERE run_id=?", (second["id"],))
    assert len(runtime.store.all("SELECT * FROM todos")) == 1
    assert executed == ["todo", "search"]
    assert "private-storage-error-marker" not in caplog.text

    runtime.store.db.execute("DROP TRIGGER reject_events")
    assert (await runtime.wait(second["id"], timeout=3))["state"] == "COMPLETED"
    assert runtime.store.run(first["id"])["state"] == "FAILED"
    assert runtime.health() == {"status": "ok"}
    assert len(runtime.store.all("SELECT * FROM todos")) == 1
    assert executed == ["todo", "search"]
    assert not runtime.store.pending_calls(first["id"])
    task = runtime.store.one("SELECT * FROM tasks")
    assert task["state"] == "FAILED" and task["error"] == "RUN_ABORTED"
    invocations = runtime.store.all(
        "SELECT result FROM invocations WHERE run_id=? ORDER BY position", (first["id"],)
    )
    assert json.loads(invocations[0]["result"])["ok"]
    assert all(
        json.loads(item["result"])["error"]["code"] == "RUN_ABORTED" for item in invocations[1:]
    )
    events = runtime.store.events(first["scope"])
    diagnostics = [event for event in events if event["type"] == "run.diagnostic"]
    assert len(diagnostics) == 1
    failed = next(event["id"] for event in events if event["type"] == "run.failed")
    started = next(
        event["id"]
        for event in events
        if event["run_id"] == second["id"] and event["type"] == "run.started"
    )
    assert diagnostics[0]["id"] < failed < started
    assert "private-storage-error-marker" not in json.dumps(events)


@pytest.mark.parametrize(
    ("failure", "code"),
    [(sqlite3.OperationalError, "STORAGE_UNAVAILABLE"), (RuntimeError, "SCHEDULER_ERROR")],
)
async def test_queue_read_failure_keeps_coordinator_alive_and_recovers(
    runtime, monkeypatch, failure, code
):
    original = runtime.store.runnable
    reads = 0
    unavailable = True

    def runnable():
        nonlocal reads
        reads += 1
        if unavailable:
            raise failure("private-queue-error-marker")
        return original()

    monkeypatch.setattr(runtime.store, "runnable", runnable)
    run = runtime.submit("u", "s", "queued during read fault")
    await wait_for(lambda: reads >= 2)
    assert not runtime._scheduler.done()
    assert runtime.health() == {"status": "degraded", "error": code}
    assert runtime.store.run(run["id"])["state"] == "QUEUED"
    unavailable = False
    assert (await runtime.wait(run["id"], timeout=3))["state"] == "COMPLETED"
    assert runtime.health() == {"status": "ok"}


@pytest.mark.parametrize("failure", [RuntimeError, asyncio.CancelledError])
async def test_unhandled_worker_failure_unblocks_same_scope(runtime, monkeypatch, failure):
    original = runtime._drive
    first = runtime.submit("u", "s", "first")
    second = runtime.submit("u", "s", "second")

    async def drive(run_id):
        if run_id == first["id"]:
            raise failure("private-worker-error-marker")
        await original(run_id)

    monkeypatch.setattr(runtime, "_drive", drive)
    assert (await runtime.wait(second["id"], timeout=2))["state"] == "COMPLETED"
    failed = runtime.store.run(first["id"])
    assert failed["state"] == "FAILED" and failed["error"] == "RUNTIME_ERROR"
    assert not runtime._scheduler.done()
    assert "private-worker-error-marker" not in json.dumps(runtime.store.events(first["scope"]))


async def test_shutdown_during_persistent_storage_failure_is_bounded(runtime):
    run = runtime.submit("u", "s", "first")
    runtime.store.db.execute(
        "CREATE TRIGGER reject_events BEFORE INSERT ON events "
        "BEGIN SELECT RAISE(ABORT,'storage unavailable'); END"
    )
    await wait_for(lambda: run["id"] in runtime._pending_failures)
    async with asyncio.timeout(1):
        await runtime.stop()
    assert runtime._closed and runtime._scheduler.done()
    assert runtime._lock.handle is None


@pytest.mark.parametrize("window", ["after-response", "after-effect"])
async def test_restart_at_tool_commit_windows(tmp_path, window):
    settings = Settings(db_path=tmp_path / "agent.db", poll_interval=0.005)
    model = FakeModel(lambda messages: answer("resumed"))
    old = Runtime(settings, model)
    run = old.submit("u", "s", "create a todo")
    old.store.activate(run["id"])
    slot = old.store.reserve_decision(run["id"], 8)
    reply = calls(("todo", {"action": "create", "title": "once"}))
    old.store.save_reply(run["id"], slot, reply.message, reply.usage, "tool_calls")
    if window == "after-effect":
        await old._execute(old.store.pending_calls(run["id"])[0])
    await old.stop()
    recovered = Runtime(settings, model)
    await recovered.start()
    try:
        result = await recovered.wait(run["id"], timeout=3)
        assert result["state"] == "COMPLETED"
        assert result["decisions"] == 2
        assert len(recovered.store.all("SELECT * FROM todos")) == 1
        assert (
            len(
                recovered.store.all(
                    "SELECT * FROM messages WHERE json_extract(payload,'$.role')='tool'"
                )
            )
            == 1
        )
    finally:
        await recovered.stop()


async def test_shutdown_at_completion_event_is_bounded(tmp_path):
    settings = Settings(db_path=tmp_path / "shutdown.db", poll_interval=0.001)
    runtime = Runtime(settings, FakeModel())
    await runtime.start()
    run = runtime.submit("u", "s", "reply")
    await runtime.wait(run["id"], timeout=2)
    # Completion signals the same event that the scheduler is awaiting.
    async with asyncio.timeout(1):
        await runtime.stop()
    assert runtime._scheduler.done()
    assert runtime._closed


async def test_model_close_failure_still_releases_database_and_process_lock(tmp_path):
    settings = Settings(db_path=tmp_path / "close-failure.db")
    model = FakeModel()

    async def broken_close():
        raise RuntimeError("model close failed")

    model.close = broken_close
    runtime = Runtime(settings, model)
    await runtime.start()
    with pytest.raises(RuntimeError, match="model close failed"):
        await runtime.stop()
    try:
        assert runtime._closed
        assert runtime._lock.handle is None
    finally:
        # Also release resources when run against the original buggy implementation.
        runtime.store.close()
        runtime._lock.release()
    replacement = Runtime(settings, FakeModel())
    await replacement.start()
    await replacement.stop()


async def test_model_request_crash_does_not_reset_budget(tmp_path):
    settings = Settings(db_path=tmp_path / "agent.db", max_decisions=1, poll_interval=0.005)
    model = FakeModel()
    old = Runtime(settings, model)
    run = old.submit("u", "s", "hello")
    old.store.activate(run["id"])
    old.store.reserve_decision(run["id"], 1)
    await old.stop()
    recovered = Runtime(settings, model)
    await recovered.start()
    try:
        result = await recovered.wait(run["id"])
        assert result["state"] == "INCOMPLETE"
        assert result["decisions"] == 1
        assert not model.requests
    finally:
        await recovered.stop()


async def test_multiple_calls_execute_sequentially(runtime):
    runtime.tools.search_delay = 0
    runtime.model.agent = lambda messages: (
        answer("done")
        if tool_results(messages)
        else calls(
            ("search", {"query": "memory"}),
            ("todo", {"action": "create", "title": "after search"}),
            ("calculator", {"expression": "3*7"}),
        )
    )
    run = runtime.submit("u", "s", "do tasks")
    assert (await runtime.wait(run["id"]))["state"] == "COMPLETED"
    events = runtime.store.events(run["scope"])
    search_end = next(event["id"] for event in events if event["type"] == "task.completed")
    todo_start = next(
        event["id"]
        for event in events
        if event["type"] == "tool.started" and event["data"]["name"] == "todo"
    )
    assert search_end < todo_start
    tool_messages = [
        row["payload"]
        for row in runtime.store.history(run["scope"])
        if row["payload"]["role"] == "tool"
    ]
    assert json.loads(tool_messages[-1]["content"])["data"]["value"] == 21


async def test_readonly_tool_timeout_is_bounded_and_does_not_block_other_sessions(
    runtime, monkeypatch
):
    import time

    original = runtime.tools.execute_sync

    def slow(name, args, db, scope):
        if name == "read_docs":
            time.sleep(0.10)
        return original(name, args, db, scope)

    monkeypatch.setattr(runtime.tools, "execute_sync", slow)
    runtime.settings = replace(runtime.settings, tool_timeout=0.03)
    runtime.model.agent = lambda messages: (
        answer("handled")
        if current_input(messages) == "fast" or tool_results(messages)
        else calls(("read_docs", {"path": "tests/fixtures/sample.md"}))
    )
    run = runtime.submit("u", "s", "read")
    await wait_for(lambda: runtime.store.one("SELECT id FROM invocations WHERE attempts>0"))
    fast = runtime.submit("u", "other", "fast")
    assert (await runtime.wait(fast["id"], timeout=2))["state"] == "COMPLETED"
    assert runtime.store.run(run["id"])["state"] == "RUNNING"
    result = await runtime.wait(run["id"], timeout=2)
    assert result["state"] == "COMPLETED"
    call = runtime.store.one("SELECT * FROM invocations")
    assert call["attempts"] == 2
    assert json.loads(call["result"])["error"]["code"] == "TOOL_TIMEOUT"
