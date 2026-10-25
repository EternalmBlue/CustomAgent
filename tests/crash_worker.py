"""Child process used only by the abrupt-restart integration test."""

import argparse
import asyncio
from pathlib import Path

from custom_agent.config import Settings
from custom_agent.runtime import Runtime
from tests.support import FakeModel, calls, wait_for


async def work(db, ready):
    settings = Settings(db_path=db, search_delay=30, tool_timeout=60, poll_interval=0.005)
    model = FakeModel(
        lambda messages: calls(
            ("todo", {"action": "create", "title": "survive-kill"}),
            ("search", {"query": "durable execution"}),
        )
    )
    runtime = Runtime(settings, model)
    await runtime.start()
    run = runtime.submit("crash-user", "crash-session", "create and search", "kill-test")
    await wait_for(lambda: runtime.store.one("SELECT id FROM tasks WHERE state='RUNNING'"))
    ready.write_text(run["id"], encoding="ascii")
    await asyncio.Event().wait()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("db", type=Path)
    parser.add_argument("ready", type=Path)
    args = parser.parse_args()
    asyncio.run(work(args.db, args.ready))
