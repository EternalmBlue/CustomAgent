import json

import pytest
from pytest_bdd import given, scenarios, then, when

from custom_agent.llm import json_text
from tests.support import Harness, answer, calls, memory_entry

scenarios("features/memory.feature")


@pytest.fixture
def h(tmp_path):
    harness = Harness(tmp_path, recent_interactions=0)
    yield harness
    harness.close()


def add_completed(h, text):
    run = h.submit(text)
    h.runtime.store.activate(run["id"])
    slot = h.runtime.store.reserve_decision(run["id"], 8)
    reply = answer("acknowledged")
    h.runtime.store.save_reply(run["id"], slot, reply.message, reply.usage, "stop")
    return run


def extract_decisions(messages):
    source = json.loads(messages[-1]["content"])["source_messages"]
    entries = []
    for item in source:
        content = item["message"].get("content") or ""
        if item["message"]["role"] == "user":
            if "choose A" in content:
                entries.append(
                    memory_entry("Choose A for storage", "storage.decision", [item["id"]])
                )
            if "replace" in content:
                entries.append(
                    memory_entry(
                        "Choose B; replace A because recovery is required",
                        "storage.decision",
                        [item["id"]],
                    )
                )
    return answer(json_text({"entries": entries}))


@given("completed interactions choosing A and then replacing it with B")
def decisions(h):
    add_completed(h, "choose A for storage")
    add_completed(h, "replace A with B because recovery is required")
    h.run = h.submit("What is the storage decision?")
    h.runtime.store.activate(h.run["id"])
    h.model.memory = extract_decisions


@when("completed history is compacted")
def compact(h):
    assert h.runner.run(h.runtime.memory.compact(h.run))


@then("a follow-up receives B as the effective decision")
def effective(h):
    entries = h.runtime.memory.retrieve(h.run["scope"], "storage decision")
    assert len(entries) == 1
    assert "Choose B" in entries[0]["text"]
    assert entries[0]["source_ids"]
    assert h.runtime.store.one("SELECT compacted_seq FROM sessions")["compacted_seq"] == 2


@then("a historical question retrieves the reason for replacing A")
def historical(h):
    entries = h.runtime.memory.retrieve(h.run["scope"], "why was A abandoned for storage?")
    assert len(entries) == 2
    assert any("recovery is required" in entry["text"] for entry in entries)
    sources = h.runtime.memory.references(h.run["scope"], "why replace A?", entries)
    assert any("recovery is required" in source["excerpt"] for source in sources)
    assert len(h.runtime.store.history(h.run["scope"])) >= 5


@given("an active run with a persisted tool call and prior compressed history")
def active(h):
    old = add_completed(h, "choose A for storage")
    rows = h.runtime.store.history(old["scope"])
    h.runtime.store.compact(
        old["scope"], 1, [memory_entry("Choose A", "storage.decision", [rows[0]["id"]])]
    )
    h.run = h.submit("calculate")
    h.runtime.store.activate(h.run["id"])
    slot = h.runtime.store.reserve_decision(h.run["id"], 8)
    reply = calls(("calculator", {"expression": "2+3"}))
    h.runtime.store.save_reply(h.run["id"], slot, reply.message, reply.usage, "tool_calls")
    call = h.runtime.store.pending_calls(h.run["id"])[0]
    h.runner.run(h.runtime._execute(call))


@when("a context is assembled under pressure")
def assembled(h):
    h.context = h.runner.run(
        h.runtime.memory.build(h.runtime.store.run(h.run["id"]), h.runtime.tools.definitions)
    )


@then("the tool call and its result remain paired")
def paired(h):
    assistant = next(message for message in h.context.messages if message.get("tool_calls"))
    tool = next(message for message in h.context.messages if message["role"] == "tool")
    assert assistant["tool_calls"][0]["id"] == tool["tool_call_id"]
    assert json.loads(tool["content"])["data"]["value"] == 5


@then("the execution checkpoint and decision count remain exact")
def counters(h):
    assert h.runtime.store.run(h.run["id"])["decisions"] == 1
    assert not h.runtime.store.pending_calls(h.run["id"])
    state = next(
        message["content"]
        for message in h.context.messages
        if "EXACT_EXECUTION_STATE" in (message.get("content") or "")
    )
    assert '"decisions_used":1' in state


@given("memory in one user session")
def private(h):
    old = add_completed(h, "secret-private-value")
    rows = h.runtime.store.history(old["scope"])
    h.runtime.store.compact(
        old["scope"],
        1,
        [memory_entry("secret-private-value", "secret", [rows[0]["id"]], "fact")],
    )


@when("another session assembles context")
def other(h):
    h.other = h.submit("secret", user="bob", session="main")
    h.runtime.store.activate(h.other["id"])
    h.context = h.runner.run(h.runtime.memory.build(h.other, h.runtime.tools.definitions))


@then("the private memory and its evidence are absent")
def no_leak(h):
    assert "secret-private-value" not in json_text(h.context.messages)
    assert h.runtime.memory.retrieve(h.other["scope"], "secret") == []


@given("a model returning invalid memory sources")
def bad_memory(h):
    add_completed(h, "choose A for storage")
    h.run = h.submit("next")
    h.runtime.store.activate(h.run["id"])
    h.model.memory = lambda messages: answer(
        json_text({"entries": [memory_entry("fabricated", "bad", [999999])]})
    )


@when("compaction is attempted")
def attempt(h):
    h.before = h.runtime.store.history(h.run["scope"])
    assert not h.runner.run(h.runtime.memory.compact(h.run))


@then("the summary watermark does not advance")
def watermark(h):
    assert h.runtime.store.one("SELECT compacted_seq FROM sessions")["compacted_seq"] == 0
    assert h.runtime.store.memory_rows(h.run["scope"]) == []


@then("original evidence remains available")
def originals(h):
    assert h.runtime.store.history(h.run["scope"]) == h.before
