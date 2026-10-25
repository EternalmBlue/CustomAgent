from __future__ import annotations

import asyncio
import inspect
import json
import time
from dataclasses import replace
from pathlib import Path

from custom_agent.config import Settings
from custom_agent.llm import ModelReply, json_text
from custom_agent.runtime import Runtime


def answer(text="done") -> ModelReply:
    return ModelReply(
        {"role": "assistant", "content": text}, {"prompt_tokens": 100, "completion_tokens": 10}
    )


def calls(*items) -> ModelReply:
    return ModelReply(
        {
            "role": "assistant",
            "content": None,
            "tool_calls": [
                {
                    "id": f"provider-{i}",
                    "type": "function",
                    "function": {"name": name, "arguments": json_text(args)},
                }
                for i, (name, args) in enumerate(items)
            ],
        },
        {"prompt_tokens": 100, "completion_tokens": 20},
    )


def current_input(messages) -> str:
    return next(message["content"] for message in reversed(messages) if message["role"] == "user")


def tool_results(messages):
    return [json.loads(message["content"]) for message in messages if message["role"] == "tool"]


class FakeModel:
    def __init__(self, agent=None, memory=None):
        self.agent = agent or (lambda messages: answer("reply: " + current_input(messages)))
        self.memory = memory
        self.requests = []

    async def complete(self, messages, tools, *, purpose="agent", max_tokens=1024):
        self.requests.append(
            {"messages": messages, "tools": tools, "purpose": purpose, "max_tokens": max_tokens}
        )
        handler = self.memory if purpose == "memory" else self.agent
        if handler is None:
            raise AssertionError("Unexpected memory extraction")
        value = handler(messages)
        if inspect.isawaitable(value):
            value = await value
        if isinstance(value, Exception):
            raise value
        return value

    async def close(self):
        pass


async def wait_for(predicate, deadline_seconds=5):
    deadline = time.monotonic() + deadline_seconds
    while time.monotonic() < deadline:
        value = predicate()
        if value:
            return value
        await asyncio.sleep(0.005)
    raise TimeoutError("Expected state was not reached")


class Harness:
    def __init__(self, tmp_path: Path, **overrides):
        self.runner = asyncio.Runner()
        self.model = FakeModel()
        self.settings = Settings(
            db_path=tmp_path / "agent.db",
            workspace_root=Path(__file__).parents[1],
            poll_interval=0.005,
            **overrides,
        )
        self.runtime = Runtime(self.settings, self.model)
        self.run = None

    def start(self):
        self.runner.run(self.runtime.start())

    def submit(self, text="hello", user="alice", session="main"):
        return self.runtime.submit(user, session, text)

    def finish(self, run=None):
        return self.runner.run(self.runtime.wait((run or self.run)["id"], timeout=10))

    def restart(self):
        self.runner.run(self.runtime.stop())
        self.settings = replace(self.settings, search_delay=0.001)
        self.runtime = Runtime(self.settings, self.model)
        self.start()

    def close(self):
        self.runner.run(self.runtime.stop())
        self.runner.close()


def memory_entry(text, key, source_ids, kind="decision"):
    return {
        "kind": kind,
        "key": key,
        "text": text,
        "entities": ["storage"],
        "source_ids": source_ids,
        "valid_at": None,
    }
