from __future__ import annotations

import argparse
import asyncio
import json
import logging
from dataclasses import replace
from functools import partial
from pathlib import Path

import uvicorn

from .api import create_app
from .config import Settings
from .runtime import Runtime


def load_settings(args) -> Settings:
    settings = Settings.from_file(getattr(args, "config", Path("config.toml")))
    if getattr(args, "db", None) is not None:
        settings = replace(settings, db_path=args.db.resolve())
    return settings


async def chat(args) -> None:
    runtime = Runtime(load_settings(args))
    try:
        await runtime.start()
    except BaseException:
        try:
            await runtime.stop()
        except Exception as exc:
            logging.getLogger(__name__).error(
                "Startup cleanup failed exception_type=%s", type(exc).__name__
            )
        raise
    try:
        if args.message:
            run = runtime.submit(args.user, args.session, args.message)
            result = await runtime.wait(run["id"], timeout=300)
            print(
                json.dumps(result, ensure_ascii=False, indent=2)
                if args.json
                else result["answer"] or result["error"]
            )
            return
        print(f"CustomAgent | user={args.user} session={args.session} | /exit to stop")
        while True:
            text = await asyncio.to_thread(input, "> ")
            if text.strip() == "/exit":
                return
            if not text.strip():
                continue
            run = runtime.submit(args.user, args.session, text)
            result = await runtime.wait(run["id"], timeout=300)
            print(result["answer"] or result["error"])
    finally:
        await runtime.stop()


def main() -> None:
    parser = argparse.ArgumentParser(prog="custom-agent")
    commands = parser.add_subparsers(dest="command", required=True)
    serve = commands.add_parser("serve")
    serve.add_argument("--host", default="127.0.0.1")
    serve.add_argument("--port", type=int, default=8000)
    talk = commands.add_parser("chat")
    talk.add_argument("--user", default="local")
    talk.add_argument("--session", default="default")
    talk.add_argument("--message")
    talk.add_argument("--json", action="store_true")
    for command in (serve, talk):
        command.add_argument("--config", type=Path, default=Path("config.toml"))
        command.add_argument("--db", type=Path)
        command.add_argument(
            "--log-level", choices=("debug", "info", "warning", "error"), default="warning"
        )
    args = parser.parse_args()
    logger = logging.getLogger("custom_agent")
    logger.setLevel(args.log_level.upper())
    if not logger.handlers:
        handler = logging.StreamHandler()
        handler.setFormatter(logging.Formatter("%(asctime)s %(levelname)s %(name)s %(message)s"))
        logger.addHandler(handler)
    logger.propagate = False
    if args.command == "serve":
        uvicorn.run(
            partial(create_app, load_settings(args)),
            factory=True,
            host=args.host,
            port=args.port,
        )
    else:
        try:
            asyncio.run(chat(args))
        except (EOFError, KeyboardInterrupt):
            pass


if __name__ == "__main__":
    main()
