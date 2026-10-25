import json

import pytest

from custom_agent.config import Settings
from custom_agent.llm import json_text
from custom_agent.memory import ContextError
from custom_agent.runtime import Runtime
from tests.support import FakeModel, answer, calls, memory_entry


def completed(runtime, text, tool_text=None):
    store = runtime.store
    run = runtime.submit("u", "s", text)
    store.activate(run["id"])
    slot = store.reserve_decision(run["id"], 8)
    if tool_text is not None:
        reply = calls(("read_docs", {"path": "sample.md"}))
        store.save_reply(run["id"], slot, reply.message, {}, "tool_calls")
        call = store.pending_calls(run["id"])[0]
        store.complete_sync(call["id"], lambda db, scope: {"ok": True, "data": tool_text})
        slot = store.reserve_decision(run["id"], 8)
    store.save_reply(run["id"], slot, answer("ack").message, {}, "stop")
    return run


def generic_extraction(messages):
    source = json.loads(messages[-1]["content"])["source_messages"]
    entries = [
        memory_entry(item["message"]["content"][:200], f"fact.{item['id']}", [item["id"]], "fact")
        for item in source
        if item["message"]["role"] == "user"
    ]
    return answer(json_text({"entries": entries}))


async def test_real_pressure_triggers_compaction_and_preserves_active_protocol(tmp_path):
    settings = Settings(
        db_path=tmp_path / "context.db",
        context_tokens=6000,
        reply_tokens=512,
        tool_result_tokens=512,
        compression_ratio=0.55,
        recent_interactions=1,
    )
    model = FakeModel(memory=generic_extraction)
    runtime = Runtime(settings, model)
    try:
        for i in range(12):
            completed(runtime, f"Fact {i}: use storage B. " + "filler " * 100)
        run = runtime.submit("u", "s", "Which storage?")
        runtime.store.activate(run["id"])
        context = await runtime.memory.build(run, runtime.tools.definitions)
        assert context.estimated_tokens <= settings.input_budget
        assert any(request["purpose"] == "memory" for request in model.requests)
        assert runtime.store.one("SELECT compacted_seq FROM sessions")["compacted_seq"] > 0
        assert len(runtime.store.history(run["scope"])) == 25
        assert context.messages[-1]["content"] == "Which storage?"
    finally:
        await runtime.stop()


async def test_observation_masking_keeps_full_evidence(tmp_path):
    settings = Settings(
        db_path=tmp_path / "mask.db",
        context_tokens=6000,
        reply_tokens=512,
        tool_result_tokens=512,
        compression_ratio=0.55,
        recent_interactions=0,
    )
    runtime = Runtime(settings, FakeModel(memory=lambda messages: answer("invalid-json")))
    try:
        old = completed(runtime, "read guide", "large observation " * 1000)
        run = runtime.submit("u", "s", "hello")
        runtime.store.activate(run["id"])
        context = await runtime.memory.build(run, runtime.tools.definitions)
        assert "masked_observation" in json_text(context.messages)
        rows = runtime.store.history(old["scope"])
        assert any(len(json_text(row["payload"])) > 10000 for row in rows)
        assert runtime.store.one("SELECT compacted_seq FROM sessions")["compacted_seq"] == 0
    finally:
        await runtime.stop()


async def test_oversized_required_input_fails_explicitly(tmp_path):
    settings = Settings(
        db_path=tmp_path / "overflow.db",
        context_tokens=3000,
        reply_tokens=512,
        tool_result_tokens=512,
        recent_interactions=0,
    )
    runtime = Runtime(settings, FakeModel())
    try:
        run = runtime.submit("u", "s", "unique tokens " * 1000)
        runtime.store.activate(run["id"])
        with pytest.raises(ContextError, match="CONTEXT_BUDGET_EXCEEDED"):
            await runtime.memory.build(run, runtime.tools.definitions)
        assert runtime.store.history(run["scope"])[0]["payload"]["content"] == run["input"]
    finally:
        await runtime.stop()


async def test_compaction_usage_is_separate_from_agent_decisions(tmp_path):
    runtime = Runtime(
        Settings(db_path=tmp_path / "usage.db", recent_interactions=0),
        FakeModel(memory=generic_extraction),
    )
    try:
        completed(runtime, "Choose storage B")
        run = runtime.submit("u", "s", "next")
        runtime.store.activate(run["id"])
        assert await runtime.memory.compact(run)
        updated = runtime.store.run(run["id"])
        assert updated["decisions"] == 0
        assert json.loads(updated["usage"])["memory"]["prompt_tokens"] == 100
    finally:
        await runtime.stop()


async def test_query_centered_evidence_finds_fact_after_long_tool_output(tmp_path):
    runtime = Runtime(Settings(db_path=tmp_path / "evidence.db"), FakeModel())
    try:
        run = completed(
            runtime, "Read document", "unrelated " * 4000 + "recovery marker ORCHID-742"
        )
        sources = runtime.memory.references(run["scope"], "document recovery marker", [])
        assert any("ORCHID-742" in source["excerpt"] for source in sources)
        assert all(len(source["excerpt"]) <= 1203 for source in sources)
    finally:
        await runtime.stop()


async def test_mandatory_constraints_are_not_displaced_by_relevant_facts(tmp_path):
    runtime = Runtime(Settings(db_path=tmp_path / "constraints.db"), FakeModel())
    try:
        run = completed(runtime, "Keep all constraints")
        source = runtime.store.history(run["scope"])[0]["id"]
        entries = [
            memory_entry(f"Required constraint {i}", f"constraint.{i}", [source], "constraint")
            for i in range(15)
        ] + [memory_entry("storage B", "storage", [source], "fact")]
        runtime.store.compact(run["scope"], 1, entries)
        selected = runtime.memory.retrieve(run["scope"], "storage", limit=4)
        assert len([entry for entry in selected if entry["kind"] == "constraint"]) == 15
    finally:
        await runtime.stop()


async def test_empty_summary_requires_explicit_discard_receipts(tmp_path):
    model = FakeModel(memory=lambda messages: answer('{"entries":[]}'))
    runtime = Runtime(Settings(db_path=tmp_path / "discard.db", recent_interactions=0), model)
    try:
        old = completed(runtime, "Unrelated filler")
        run = runtime.submit("u", "s", "next")
        runtime.store.activate(run["id"])
        assert not await runtime.memory.compact(run)
        ids = [row["id"] for row in runtime.store.history(old["scope"]) if row["seq"] == 1]
        model.memory = lambda messages: answer(
            json_text({"entries": [], "discardable_source_ids": ids})
        )
        assert await runtime.memory.compact(run)
        assert runtime.store.one("SELECT compacted_seq FROM sessions")["compacted_seq"] == 1
        assert len(runtime.store.history(old["scope"])) == 3
    finally:
        await runtime.stop()


async def test_long_complete_interaction_uses_available_extraction_budget(tmp_path):
    settings = Settings(
        db_path=tmp_path / "long-prefix.db",
        context_tokens=6000,
        reply_tokens=512,
        tool_result_tokens=512,
        recent_interactions=0,
    )
    runtime = Runtime(settings, FakeModel(memory=generic_extraction))
    try:
        completed(runtime, "sample " * 2200)
        run = runtime.submit("u", "s", "different " * 1400)
        runtime.store.activate(run["id"])
        context = await runtime.memory.build(run, runtime.tools.definitions)
        assert context.estimated_tokens <= settings.input_budget
        assert runtime.store.one("SELECT compacted_seq FROM sessions")["compacted_seq"] == 1
        assert any(request["purpose"] == "memory" for request in runtime.model.requests)
    finally:
        await runtime.stop()


async def test_extraction_reserves_its_requested_output(tmp_path):
    settings = Settings(
        db_path=tmp_path / "extraction-budget.db",
        context_tokens=3000,
        reply_tokens=700,
        tool_result_tokens=100,
        recent_interactions=0,
    )
    runtime = Runtime(settings, FakeModel(memory=generic_extraction))
    try:
        completed(runtime, "sample " * 600)
        run = runtime.submit("u", "s", "next")
        runtime.store.activate(run["id"])
        await runtime.memory.compact(run)
        for request in runtime.model.requests:
            assert (
                runtime.memory.counter.count(request["messages"]) + request["max_tokens"]
                <= settings.context_tokens
            )
    finally:
        await runtime.stop()


async def test_internal_evidence_does_not_repeat_active_input(tmp_path):
    runtime = Runtime(Settings(db_path=tmp_path / "historical-evidence.db"), FakeModel())
    try:
        for index in range(3):
            completed(runtime, f"storage recovery marker historical-{index}")
        run = runtime.submit("u", "s", "storage recovery marker")
        runtime.store.activate(run["id"])
        reference = runtime.memory._memory_message(run)
        data = json.loads(reference["content"].split(": ", 1)[1])
        assert len(data["evidence"]) == 3
        assert all(source["run_id"] != run["id"] for source in data["evidence"])
    finally:
        await runtime.stop()
