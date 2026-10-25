"""Reproducible snapshot comparison, with optional real-provider continuations."""

from __future__ import annotations

import argparse
import asyncio
import json
import tempfile
import time
from dataclasses import replace
from pathlib import Path

from .config import Settings
from .llm import Json, ModelReply, OpenAICompatibleModel, json_text
from .memory import SYSTEM, ContextManager
from .store import Store
from .tools import ToolRegistry

QUESTIONS = [
    ("What is the current storage decision? Reply with only A or B.", "B"),
    ("Why was storage A abandoned?", "restart recovery"),
    ("What is the document recovery marker?", "ORCHID-742"),
]


class FixtureExtractor:
    """A fixed extractor for measuring context plumbing, not model quality."""

    async def complete(self, messages, tools, *, purpose="agent", max_tokens=1024):
        if purpose != "memory":
            raise ValueError("Fixture mode only performs memory extraction")
        source = json.loads(messages[-1]["content"])["source_messages"]
        entries = []
        for row in source:
            payload = row["message"]
            if payload["role"] == "user" and payload["content"].startswith("Storage decision:"):
                entries.append(
                    {
                        "kind": "decision",
                        "key": "storage",
                        "text": payload["content"],
                        "entities": ["storage"],
                        "source_ids": [row["id"]],
                        "valid_at": None,
                    }
                )
            if payload["role"] == "tool":
                entries.append(
                    {
                        "kind": "fact",
                        "key": f"document.{row['id']}",
                        "text": f"A document observation is retained at source {row['id']}.",
                        "entities": ["document", "recovery", "marker"],
                        "source_ids": [row["id"]],
                        "valid_at": None,
                    }
                )
        return ModelReply(
            {
                "role": "assistant",
                "content": json_text(
                    {
                        "entries": entries,
                        "discardable_source_ids": [
                            row["id"]
                            for row in source
                            if not any(row["id"] in entry["source_ids"] for entry in entries)
                        ],
                    }
                ),
            },
            {},
        )

    async def close(self):
        pass


def seed(store: Store) -> str:
    inputs = [
        "Storage decision: A because setup is simple.",
        "Read the document.",
        "Storage decision: B, replacing A because restart recovery is required.",
    ] + [f"Unrelated history {i}. " + "Background only. " * 100 for i in range(14)]
    for index, text in enumerate(inputs):
        run = store.submit("eval", "snapshot", text)
        store.activate(run["id"])
        slot = store.reserve_decision(run["id"], 8)
        if index == 1:
            store.save_reply(
                run["id"],
                slot,
                {
                    "role": "assistant",
                    "content": None,
                    "tool_calls": [
                        {
                            "function": {
                                "name": "read_docs",
                                "arguments": '{"path":"tests/fixtures/sample.md"}',
                            },
                        }
                    ],
                },
                {},
                "tool_calls",
            )
            invocation = store.pending_calls(run["id"])[0]
            store.complete_sync(
                invocation["id"],
                lambda db, scope: {
                    "ok": True,
                    "data": {
                        "background": "Verbose document filler. " * 1500,
                        "fact": "The document recovery marker is ORCHID-742.",
                    },
                },
            )
            slot = store.reserve_decision(run["id"], 8)
        store.save_reply(run["id"], slot, {"role": "assistant", "content": "Recorded."}, {}, "stop")
    return run["scope"]


def evidence_probe(messages: list[Json], expected: str, question: str) -> bool:
    text = json_text(messages)
    if expected == "B":
        # Probe that the latest decision and replacement evidence are represented.
        return "Storage decision: B" in text or (
            "storage" in text.lower() and "B" in text and "replac" in text
        )
    return expected in text


def answer_matches_expected(text: str, expected: str) -> bool:
    if expected == "B":
        return text.strip().strip("`* .").casefold() == "b"
    return expected.casefold() in text.casefold()


async def compare(
    output: Path, *, live: bool = False, config_path: Path = Path("config.toml")
) -> Json:
    base = Settings.from_file(config_path)
    model = (
        OpenAICompatibleModel(base.api_key, base.base_url, base.model, base.model_timeout)
        if live
        else FixtureExtractor()
    )
    rows = []
    try:
        with tempfile.TemporaryDirectory(prefix="agent-memory-eval-") as directory:
            root = Path(directory)
            snapshot = Store(root / "snapshot.db")
            scope = seed(snapshot)
            try:
                for policy in ("full_history", "observation_masking", "structured_memory"):
                    for index, (question, expected) in enumerate(QUESTIONS):
                        store = Store(root / f"{policy}-{index}.db")
                        snapshot.db.backup(store.db)
                        settings = replace(
                            base,
                            db_path=store.path,
                            context_tokens=8192,
                            recent_interactions=2,
                            compression_ratio=0.60,
                        )
                        manager = ContextManager(store, model, settings)
                        tools = ToolRegistry(base.workspace_root).definitions
                        run = store.submit("eval", "snapshot", question)
                        store.activate(run["id"])
                        start = time.perf_counter()
                        try:
                            if policy == "structured_memory":
                                context = await manager.build(run, tools)
                                messages = context.messages
                                tokens = context.estimated_tokens
                            else:
                                history = store.history(scope)
                                messages = [{"role": "system", "content": SYSTEM}]
                                for row in history:
                                    payload = dict(row["payload"])
                                    if (
                                        policy == "observation_masking"
                                        and payload["role"] == "tool"
                                    ):
                                        payload["content"] = json_text(
                                            {
                                                "source_id": row["id"],
                                                "masked_observation": True,
                                                "excerpt": payload["content"][:160],
                                            }
                                        )
                                    messages.append(payload)
                                tokens = manager.counter.count(messages) + manager.counter.count(
                                    tools
                                )
                            assembly_ms = (time.perf_counter() - start) * 1000
                            result: Json = {
                                "policy": policy,
                                "question": question,
                                "estimated_input_tokens": tokens,
                                "assembly_ms": round(assembly_ms, 3),
                                "evidence_probe_passed": evidence_probe(
                                    messages, expected, question
                                ),
                                "compression_usage": json.loads(store.run(run["id"])["usage"]),
                            }
                            if live:
                                before = time.perf_counter()
                                reply = await model.complete(messages, [], max_tokens=512)
                                result.update(
                                    answer=reply.message.get("content"),
                                    actual_usage=reply.usage,
                                    answer_seconds=round(time.perf_counter() - before, 3),
                                    answer_contains_expected=answer_matches_expected(
                                        reply.message.get("content") or "", expected
                                    ),
                                )
                            rows.append(result)
                        finally:
                            store.close()
            finally:
                snapshot.close()
    finally:
        await model.close()
    report = {
        "mode": "real_provider" if live else "deterministic_context_probe",
        "note": (
            "Identical SQLite snapshots. Fixture extraction tests retrieval plumbing, "
            "not LLM task success or research superiority. Live mode includes compression usage."
        ),
        "results": rows,
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    await asyncio.to_thread(
        output.write_text, json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    return report


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--live", action="store_true")
    parser.add_argument("--config", type=Path, default=Path("config.toml"))
    parser.add_argument("--output", type=Path, default=Path("reports/memory-comparison.json"))
    args = parser.parse_args()
    report = asyncio.run(compare(args.output, live=args.live, config_path=args.config))
    for row in report["results"]:
        print(
            f"{row['policy']:22} tokens={row['estimated_input_tokens']:5} "
            f"evidence={row['evidence_probe_passed']} {row['question']}"
        )


if __name__ == "__main__":
    main()
