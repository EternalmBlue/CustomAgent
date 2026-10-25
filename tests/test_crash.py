import asyncio
import subprocess
import sys

import pytest

from custom_agent.config import Settings
from custom_agent.runtime import Runtime
from tests.support import FakeModel, answer, calls, wait_for


async def test_abrupt_process_kill_releases_lock_and_recovers_task(tmp_path):
    db = tmp_path / "kill.db"
    ready = tmp_path / "ready.txt"
    process = await asyncio.to_thread(
        subprocess.Popen,
        [sys.executable, "-m", "tests.crash_worker", str(db), str(ready)],
        stdin=subprocess.DEVNULL,
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
    )
    try:
        await wait_for(ready.exists, deadline_seconds=15)
        run_id = ready.read_text(encoding="ascii")
        process.kill()
        await asyncio.to_thread(process.wait, 5)
    finally:
        if process.poll() is None:
            process.kill()
            await asyncio.to_thread(process.wait, 5)
    runtime = Runtime(
        Settings(db_path=db, search_delay=0, poll_interval=0.005),
        FakeModel(lambda messages: answer("recovered after kill")),
    )
    await runtime.start()
    try:
        result = await runtime.wait(run_id, timeout=5)
        assert result["state"] == "COMPLETED"
        assert result["decisions"] == 2
        assert len(runtime.store.all("SELECT * FROM todos")) == 1
        task = runtime.store.one("SELECT * FROM tasks")
        assert task["state"] == "SUCCEEDED" and task["attempts"] == 2
        attempts = [
            item for item in runtime.store.attempts(run_id) if item["task_id"] == task["id"]
        ]
        assert [item["state"] for item in attempts] == ["INTERRUPTED", "SUCCEEDED"]
        assert attempts[0]["ended_at"] is None and attempts[0]["duration_ms"] is None
        assert attempts[0]["interrupted_at"] >= attempts[0]["started_at"]
    finally:
        await runtime.stop()


async def test_failure_between_todo_mutation_and_result_rolls_back_both(tmp_path, monkeypatch):
    runtime = Runtime(Settings(db_path=tmp_path / "atomic.db"), FakeModel())
    try:
        run = runtime.submit("u", "s", "create")
        runtime.store.activate(run["id"])
        slot = runtime.store.reserve_decision(run["id"], 8)
        reply = calls(("todo", {"action": "create", "title": "atomic"}))
        runtime.store.save_reply(run["id"], slot, reply.message, {}, "tool_calls")
        invocation = runtime.store.pending_calls(run["id"])[0]
        original = runtime.store._complete_call

        def fail(invocation, result):
            raise RuntimeError("Injected failure after write, before result")

        monkeypatch.setattr(runtime.store, "_complete_call", fail)
        with pytest.raises(RuntimeError, match="Injected failure"):
            await runtime._execute(invocation)
        assert not runtime.store.all("SELECT * FROM todos")
        assert runtime.store.pending_calls(run["id"])[0]["attempts"] == 1
        failed = runtime.store.attempts(run["id"])[0]
        assert failed["state"] == "FAILED" and failed["error_code"] == "TOOL_ERROR"
        assert failed["diagnostic"]["exception_type"] == "RuntimeError"
        monkeypatch.setattr(runtime.store, "_complete_call", original)
        await runtime._execute(invocation)
        assert len(runtime.store.all("SELECT * FROM todos")) == 1
        assert not runtime.store.pending_calls(run["id"])
    finally:
        await runtime.stop()
