"""Explicitly opt in: AGENT_LIVE=1 plus provider TOML. Never run in ordinary CI."""

import json
import os
import time
from dataclasses import replace
from datetime import UTC, datetime
from pathlib import Path

import pytest

from custom_agent.runtime import Runtime
from tests.support import answer

pytestmark = [
    pytest.mark.live,
    pytest.mark.skipif(
        os.getenv("AGENT_LIVE") != "1", reason="Set AGENT_LIVE=1 to call the real API"
    ),
]


def save_report(name, runtime, results):
    directory = Path("reports")
    directory.mkdir(exist_ok=True)
    runs = runtime.store.all("SELECT id,state,decisions,usage,error FROM runs ORDER BY created_at")
    for run in runs:
        run["usage"] = json.loads(run["usage"])
    report = {
        "timestamp": datetime.now(UTC).isoformat(),
        "model": runtime.settings.model,
        "results": results,
        "runs": runs,
    }
    (directory / f"live-{name}.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8"
    )


async def test_live_five_tools_and_followups(tmp_path, provider_settings):
    settings = replace(
        provider_settings,
        db_path=tmp_path / "live.db",
        workspace_root=Path(__file__).parents[1],
    )
    runtime = Runtime(settings)
    results = []
    await runtime.start()

    async def ask(label, text, session="main"):
        before = time.monotonic()
        run = runtime.submit("live-user", session, text)
        result = await runtime.wait(run["id"], timeout=180)
        results.append(
            {
                "scenario": label,
                "state": result["state"],
                "answer": result["answer"],
                "error": result["error"],
                "seconds": round(time.monotonic() - before, 3),
            }
        )
        assert result["state"] == "COMPLETED", result["error"]
        return result

    try:
        result = await ask("direct", "Reply with exactly READY.")
        assert "READY" in result["answer"]
        result = await ask("calculator", "Use calculator to compute (17+25)*3.")
        invocation = runtime.store.one(
            "SELECT result FROM invocations WHERE run_id=? AND name='calculator'", (result["id"],)
        )
        assert invocation and json.loads(invocation["result"])["data"]["value"] == 126
        await ask("todo-create", "Use todo to create exactly one todo titled release-check.")
        todo = runtime.store.one("SELECT * FROM todos")
        assert todo and todo["title"] == "release-check"
        await ask("todo-followup", "Mark the todo you just created as complete, using todo.")
        assert runtime.store.one("SELECT completed FROM todos")["completed"] == 1
        result = await ask(
            "session-isolation", "Use todo to list all todos in this session.", "other"
        )
        invocation = runtime.store.one(
            "SELECT result FROM invocations WHERE run_id=? AND name='todo'", (result["id"],)
        )
        assert invocation and json.loads(invocation["result"])["data"]["items"] == []
        result = await ask(
            "async-search", "Use search to find durable execution and summarize the local results."
        )
        task = runtime.store.one("SELECT * FROM tasks WHERE run_id=?", (result["id"],))
        assert task and task["state"] == "SUCCEEDED"
        result = await ask(
            "read-docs",
            "Use read_docs to read tests/fixtures/sample.md and tell me the recovery marker.",
        )
        assert "ORCHID-742" in result["answer"]
        result = await ask(
            "weather", "Use weather for Shanghai and say clearly that it is mock data."
        )
        invocation = runtime.store.one(
            "SELECT result FROM invocations WHERE run_id=? AND name='weather'", (result["id"],)
        )
        assert invocation and json.loads(invocation["result"])["data"]["mock"]
        await ask("todo-delete", "Delete the release-check todo using todo.")
        assert not runtime.store.all("SELECT * FROM todos")
    finally:
        save_report("tools", runtime, results)
        await runtime.stop()


async def test_live_compression_decision_change_and_evidence(tmp_path, provider_settings):
    settings = replace(
        provider_settings,
        db_path=tmp_path / "memory.db",
        workspace_root=Path(__file__).parents[1],
        context_tokens=7000,
        reply_tokens=1024,
        tool_result_tokens=1024,
        recent_interactions=1,
        compression_ratio=0.50,
    )
    runtime = Runtime(settings)
    results = []
    # A fixed transcript makes the compression experiment reproducible. Extraction
    # and follow-up answers below use the real provider, not a mocked memory model.
    source_inputs = [
        "部署决策：存储先选择方案A，原因是配置简单。",
        "修改部署决策：放弃方案A，改用方案B，因为必须支持进程重启后的任务恢复。后续以B为准。",
    ] + [
        f"历史记录{i}，仅供背景。 " + "This is unrelated historical background. " * 95
        for i in range(8)
    ]
    for text in source_inputs:
        run = runtime.submit("live-user", "memory", text)
        runtime.store.activate(run["id"])
        slot = runtime.store.reserve_decision(run["id"], 8)
        runtime.store.save_reply(run["id"], slot, answer("已记录。").message, {}, "stop")
    await runtime.start()
    try:
        for label, question, expected in [
            ("effective-decision", "当前部署的存储方案选哪个？只回答方案名。", "B"),
            ("historical-evidence", "当时为什么放弃旧存储方案A？", "恢复"),
        ]:
            before = time.monotonic()
            run = runtime.submit("live-user", "memory", question)
            result = await runtime.wait(run["id"], timeout=180)
            results.append(
                {
                    "scenario": label,
                    "state": result["state"],
                    "answer": result["answer"],
                    "error": result["error"],
                    "seconds": round(time.monotonic() - before, 3),
                }
            )
            assert result["state"] == "COMPLETED", result["error"]
            assert expected in result["answer"]
        scope = runtime.store.scope("live-user", "memory")
        assert (
            runtime.store.one("SELECT compacted_seq FROM sessions WHERE id=?", (scope,))[
                "compacted_seq"
            ]
            >= 2
        )
        assert runtime.store.memory_rows(scope)
        assert len(runtime.store.history(scope)) >= len(source_inputs) * 2
        assert any(
            json.loads(row["usage"]).get("memory", {}).get("prompt_tokens", 0) > 0
            for row in runtime.store.all("SELECT usage FROM runs")
        )
    finally:
        save_report("memory", runtime, results)
        await runtime.stop()
