import json
from dataclasses import replace
from types import SimpleNamespace

import httpx
import pytest
from pytest_bdd import given, scenarios, then, when

from custom_agent.runtime import Runtime
from custom_agent.store import Store
from custom_agent.tools import ToolError
from tests.support import Harness, answer, calls, current_input, tool_results, wait_for

scenarios("features/runtime.feature")


@pytest.fixture
def h(tmp_path):
    harness = Harness(tmp_path)
    yield harness
    harness.close()


@given("an empty runtime")
def empty(h):
    h.start()


@when("event storage fails and then becomes available again")
def failing_storage(h):
    h.run = h.submit("first")
    h.runtime.store.db.execute(
        "CREATE TRIGGER reject_events BEFORE INSERT ON events "
        "BEGIN SELECT RAISE(ABORT,'injected storage failure'); END"
    )
    h.runner.run(
        wait_for(
            lambda: h.runtime._scheduler.done() or getattr(h.runtime, "_scheduler_error", None)
        )
    )
    h.runtime.store.db.execute("DROP TRIGGER reject_events")
    h.second = h.submit("second")


@then("the scheduler remains alive and queued inputs resume in FIFO order")
def storage_recovers(h):
    assert not h.runtime._scheduler.done()
    assert h.finish(h.second)["state"] == "COMPLETED"
    assert h.runtime.store.run(h.run["id"])["state"] == "FAILED"
    events = h.runtime.store.events(h.run["scope"])
    failed = next(e["id"] for e in events if e["type"] == "run.failed")
    second = next(
        e["id"] for e in events if e["run_id"] == h.second["id"] and e["type"] == "run.started"
    )
    assert failed < second


@when("the model sends malformed JSON tool arguments")
def malformed_arguments(h):
    h.bad_arguments = getattr(h, "bad_arguments", "{")

    def agent(messages):
        if tool_results(messages):
            return answer("handled")
        reply = calls(("todo", {}))
        reply.message["tool_calls"][0]["function"]["arguments"] = h.bad_arguments
        return reply

    h.model.agent = agent
    h.run = h.submit("bad JSON")
    h.result = h.finish()


@when("the model sends tool arguments beyond the JSON nesting limit")
def over_nested_arguments(h):
    h.bad_arguments = "[" * 1100
    malformed_arguments(h)


@then("the trace preserves the rejected arguments without a server error")
def malformed_trace(h):
    from custom_agent.api import create_app

    async def query():
        async with httpx.AsyncClient(
            transport=httpx.ASGITransport(
                create_app(h.settings, h.runtime), raise_app_exceptions=False
            ),
            base_url="http://test",
        ) as client:
            return await client.get(f"/v1/runs/{h.run['id']}/trace", headers={"X-User-ID": "alice"})

    response = h.runner.run(query())
    assert response.status_code == 200
    invocation = response.json()["invocations"][0]
    assert h.result["state"] == "COMPLETED"
    assert invocation["arguments"] == h.bad_arguments
    assert invocation["arguments_error"] == "INVALID_JSON"
    assert invocation["result"]["error"]["code"] == "INVALID_ARGUMENTS"


@given("a CLI runtime whose startup fails")
def cli_start_failure(h, monkeypatch):
    from custom_agent import cli

    async def fail():
        raise RuntimeError("injected startup failure")

    monkeypatch.setattr(h.runtime, "start", fail)
    monkeypatch.setattr(cli, "Runtime", lambda settings: h.runtime)


@when("the CLI starts")
def start_cli(h):
    from custom_agent.cli import chat

    with pytest.raises(RuntimeError, match="injected startup failure"):
        h.runner.run(chat(SimpleNamespace()))


@then("its runtime resources are closed")
def cli_closed(h):
    assert h.runtime._closed
    assert h.runtime._lock.handle is None


@given("an empty runtime with a one-decision budget")
def limited(h):
    from dataclasses import replace

    h.runner.run(h.runtime.stop())
    from custom_agent.runtime import Runtime

    h.settings = replace(h.settings, max_decisions=1)
    h.runtime = Runtime(h.settings, h.model)
    h.start()


@when("a user asks for a direct reply")
def direct(h):
    h.run = h.submit("hello")
    h.result = h.finish()


@when("a read-only tool fails transiently before succeeding")
def retry_tool(h):
    original = h.runtime.tools._handlers["weather"]
    count = 0

    def flaky(args, db, scope):
        nonlocal count
        count += 1
        if count == 1:
            raise ToolError("TEMP_WEATHER", "temporary", transient=True)
        return original(args, db, scope)

    h.runtime.tools._handlers["weather"] = flaky
    h.model.agent = lambda messages: (
        answer("done") if tool_results(messages) else calls(("weather", {"city": "Shanghai"}))
    )
    h.run = h.submit("weather")
    h.result = h.finish()


@then("both attempts and the original retry cause remain in the trace")
def retry_trace(h):
    attempts = h.runtime.store.attempts(h.run["id"])
    tools = [item for item in attempts if item["kind"] == "tool"]
    assert [item["state"] for item in tools] == ["FAILED", "SUCCEEDED"]
    assert tools[0]["error_code"] == "TEMP_WEATHER"
    retry = next(e for e in h.runtime.store.events(h.run["scope"]) if e["type"] == "tool.retried")
    assert retry["data"]["attempt_id"] == tools[0]["id"]
    assert retry["data"]["error_code"] == "TEMP_WEATHER"


@then("the model attempt retains context, timing and actual usage")
def model_trace(h):
    attempt = h.runtime.store.attempts(h.run["id"])[0]
    assert attempt["kind"] == "model" and attempt["decision"] == 1
    assert attempt["state"] == "SUCCEEDED"
    assert attempt["duration_ms"] >= 0 and attempt["ended_at"] >= attempt["started_at"]
    assert attempt["request"]["messages"] == h.model.requests[0]["messages"]
    assert attempt["request"]["tools"] == h.model.requests[0]["tools"]
    assert attempt["request"]["context"]["estimated_tokens"] > 0
    assert attempt["usage"] == {"prompt_tokens": 100, "completion_tokens": 10}


@then("the run completes and the reply survives reopening the database")
def durable_reply(h):
    assert h.result["state"] == "COMPLETED"
    assert h.result["answer"] == "reply: hello"
    other = Store(h.settings.db_path)
    try:
        assert other.run(h.run["id"])["answer"] == "reply: hello"
    finally:
        other.close()


@when("the model creates a todo and the invocation is replayed")
def todo_replay(h):
    h.model.agent = lambda messages: (
        answer("created")
        if tool_results(messages)
        else calls(("todo", {"action": "create", "title": "ship"}))
    )
    h.run = h.submit("create todo")
    h.result = h.finish()
    call = h.runtime.store.one("SELECT * FROM invocations WHERE run_id=?", (h.run["id"],))
    result = h.runtime.store.complete_sync(
        call["id"], lambda db, scope: pytest.fail("Replayed effect must not execute")
    )
    assert result["ok"]


@then("exactly one todo exists in the originating scope")
def one_todo(h):
    assert h.result["state"] == "COMPLETED"
    assert len(h.runtime.store.all("SELECT * FROM todos")) == 1


@then("another session cannot read or change it")
def isolation(h):
    store = h.runtime.store
    other = store.scope("alice", "other", create=True)
    todo = store.one("SELECT * FROM todos")
    result = h.runtime.tools.execute_sync("todo", {"action": "list"}, store.db, other)
    assert result["data"]["items"] == []
    for action in ("complete", "delete"):
        result = h.runtime.tools.execute_sync(
            "todo", {"action": action, "id": todo["id"]}, store.db, other
        )
        assert result["error"]["code"] == "TODO_NOT_FOUND"
    bob = store.scope("bob", "main", create=True)
    assert (
        h.runtime.tools.execute_sync("todo", {"action": "list"}, store.db, bob)["data"]["items"]
        == []
    )


def search_script(messages):
    if current_input(messages) == "second":
        return answer("second finished")
    if tool_results(messages):
        return answer("search finished")
    return calls(
        ("todo", {"action": "create", "title": "before-search"}),
        ("search", {"query": "durable execution"}),
    )


@given("a run waiting for an asynchronous search")
@given("a slow asynchronous run in one session")
def waiting(h):
    h.model.agent = search_script
    h.runtime.tools.search_delay = 0.5
    h.start()
    h.run = h.submit("search")
    h.runner.run(
        wait_for(
            lambda: h.runtime.store.one(
                "SELECT * FROM tasks WHERE run_id=? AND state='RUNNING'", (h.run["id"],)
            )
        )
    )


@when("the runtime restarts")
def restart(h):
    h.restart()
    h.result = h.finish()


@then("the search result resumes the original run without repeating prior effects")
def recovered(h):
    assert h.result["state"] == "COMPLETED"
    assert h.result["decisions"] == 2
    assert len(h.runtime.store.all("SELECT * FROM todos")) == 1
    task = h.runtime.store.one("SELECT * FROM tasks")
    assert task["state"] == "SUCCEEDED"
    assert task["attempts"] == 2
    events = h.runtime.store.events(h.run["scope"])
    assert any(event["type"] == "task.recovered" for event in events)


@when("a second input arrives in the same session")
def enqueue(h):
    h.second = h.submit("second")
    assert h.runtime.store.run(h.second["id"])["state"] == "QUEUED"
    assert h.runtime.store.all("SELECT * FROM messages WHERE run_id=?", (h.second["id"],)) == []
    h.finish(h.second)


@then("the second run starts only after the first run finishes")
def fifo(h):
    events = h.runtime.store.events(h.run["scope"])
    first_end = next(
        e["id"] for e in events if e["run_id"] == h.run["id"] and e["type"] == "run.completed"
    )
    second_start = next(
        e["id"] for e in events if e["run_id"] == h.second["id"] and e["type"] == "run.started"
    )
    assert second_start > first_end


@when("another session requests a direct reply")
def parallel_session(h):
    h.second = h.submit("second", session="other")
    h.result = h.finish(h.second)


@then("the other session finishes before the slow task")
def independent(h):
    assert h.result["state"] == "COMPLETED"
    assert h.runtime.store.run(h.run["id"])["state"] == "WAITING_TOOL"


@when("the model supplies invalid todo arguments")
def invalid_args(h):
    h.model.agent = lambda messages: (
        answer("recovered") if tool_results(messages) else calls(("todo", {"action": "create"}))
    )
    h.run = h.submit("bad")
    h.result = h.finish()


@then("the model receives a validation error and no todo is created")
def no_effect(h):
    assert h.result["state"] == "COMPLETED"
    assert not h.runtime.store.all("SELECT * FROM todos")
    result = tool_results(h.model.requests[-1]["messages"])[-1]
    assert result["error"]["code"] == "INVALID_ARGUMENTS"


@when("the model creates a todo")
def creates_todo(h):
    h.model.agent = lambda messages: calls(("todo", {"action": "create", "title": "keep"}))
    h.run = h.submit("create")
    h.result = h.finish()


@then("the run is incomplete and its todo remains")
def incomplete(h):
    assert h.result["state"] == "INCOMPLETE"
    assert h.result["error"] == "DECISION_BUDGET_EXHAUSTED"
    assert len(h.runtime.store.all("SELECT * FROM todos")) == 1


@given("a completed run with persisted events")
def event_run(h):
    h.start()
    h.run = h.submit("hello")
    h.finish()
    other = h.submit("other", user="bob", session="main")
    h.finish(other)


@when("a client reconnects after a known event ID")
def reconnect(h):
    h.cursor = h.runtime.store.events(h.run["scope"])[1]["id"]
    h.restart()
    h.replayed = h.runtime.store.events(h.run["scope"], h.cursor)


@then("only newer events for that user and session are replayed in order")
def ordered_replay(h):
    assert h.replayed
    ids = [event["id"] for event in h.replayed]
    assert ids == sorted(set(ids))
    assert all(event["scope"] == h.run["scope"] and event["id"] > h.cursor for event in h.replayed)
    assert all(event["run_id"] == h.run["id"] for event in h.replayed)
    assert json.dumps(h.replayed)


@given("a runtime with a workspace source file")
def workspace_source(h):
    h.runner.run(h.runtime.stop())
    h.workspace = h.settings.db_path.parent / "workspace"
    h.source = h.workspace / "src" / "example.py"
    h.source.parent.mkdir(parents=True)
    h.source.write_text("VALUE = 126\n", encoding="utf-8", newline="\n")
    h.outside = h.workspace.parent / "outside.txt"
    h.outside.write_text("outside-private-marker", encoding="utf-8")
    h.settings = replace(h.settings, workspace_root=h.workspace)
    h.runtime = Runtime(h.settings, h.model)
    h.start()


@when("the model requests that file by relative and absolute paths")
def read_workspace_source(h):
    h.model.agent = lambda messages: (
        answer("read")
        if tool_results(messages)
        else calls(
            ("read_docs", {"path": "src/example.py"}),
            ("read_docs", {"path": str(h.source)}),
        )
    )
    h.run = h.submit("Read the workspace source file.")
    h.result = h.finish()


@then("both tool results contain the workspace file content")
def workspace_content(h):
    assert h.result["state"] == "COMPLETED"
    results = tool_results(h.model.requests[-1]["messages"])
    assert len(results) == 2
    assert all(result["ok"] and result["data"]["content"] == "VALUE = 126\n" for result in results)
    assert all(result["data"]["path"] == "src/example.py" for result in results)


@when("the model requests a file outside the workspace")
def read_outside_workspace(h):
    h.model.agent = lambda messages: (
        answer("rejected")
        if tool_results(messages)
        else calls(
            ("read_docs", {"path": str(h.outside)}),
            ("read_docs", {"path": "../outside.txt"}),
        )
    )
    h.run = h.submit("Read the outside file.")
    h.result = h.finish()


@then("the tool returns a path error without the outside file content")
def outside_content_rejected(h):
    assert h.result["state"] == "COMPLETED"
    results = tool_results(h.model.requests[-1]["messages"])
    assert len(results) == 2
    assert all(result["error"]["code"] == "PATH_FORBIDDEN" for result in results)
    assert "outside-private-marker" not in json.dumps(results)
