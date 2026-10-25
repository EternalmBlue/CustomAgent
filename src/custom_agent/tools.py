from __future__ import annotations

import ast
import asyncio
import json
import math
import operator
import re
import sqlite3
import time
import uuid
from collections.abc import Callable
from pathlib import Path, PureWindowsPath
from typing import Any

from jsonschema import Draft202012Validator

from .llm import Json


def ok(data: Any) -> Json:
    return {"ok": True, "data": data}


def error(code: str, message: str) -> Json:
    return {"ok": False, "error": {"code": code, "message": message}}


class ToolError(Exception):
    def __init__(self, code: str, message: str, *, transient: bool = False):
        super().__init__(message)
        self.code = code
        self.transient = transient


def terms(text: str) -> set[str]:
    words = set(re.findall(r"[a-z0-9_-]+", text.casefold()))
    for phrase in re.findall(r"[\u3400-\u9fff]+", text):
        words.update(phrase[i : i + 2] for i in range(len(phrase) - 1))
        if len(phrase) == 1:
            words.add(phrase)
    return words


def schema(properties: Json, required: list[str]) -> Json:
    return {
        "type": "object",
        "properties": properties,
        "required": required,
        "additionalProperties": False,
    }


TEXT = {"type": "string", "minLength": 1, "maxLength": 256}
TOOL_SPECS = [
    {
        "name": "calculator",
        "description": "Calculate bounded arithmetic (+ - * / // % ** and parentheses).",
        "parameters": schema({"expression": TEXT}, ["expression"]),
    },
    {
        "name": "search",
        "description": (
            "Search the deterministic local Agent corpus asynchronously. "
            "This is mock search, not the internet. Wait for its tool result."
        ),
        "parameters": schema({"query": TEXT}, ["query"]),
    },
    {
        "name": "todo",
        "description": (
            "Manage the current user/session's persistent todos. "
            "create requires title; complete/delete require id; list needs only action. "
            "Always use returned IDs. Never invent todo state."
        ),
        "parameters": schema(
            {
                "action": {"type": "string", "enum": ["create", "list", "complete", "delete"]},
                "title": {"type": "string", "minLength": 1, "maxLength": 200},
                "id": {"type": "string", "minLength": 1, "maxLength": 64},
            },
            ["action"],
        )
        | {
            "allOf": [
                {
                    "if": {"properties": {"action": {"const": "create"}}},
                    "then": {"required": ["title"]},
                },
                {
                    "if": {"properties": {"action": {"enum": ["complete", "delete"]}}},
                    "then": {"required": ["id"]},
                },
            ]
        },
    },
    {
        "name": "weather",
        "description": "Return MOCK fixture weather for Beijing/Shanghai/Shenzhen/London/New York.",
        "parameters": schema({"city": TEXT}, ["city"]),
    },
    {
        "name": "read_docs",
        "description": (
            "Read a UTF-8 text file in the workspace, including source and configuration files. "
            "Extract the requested path from the user; pass a workspace-relative path or an "
            "absolute path inside the workspace. Files outside the workspace are forbidden."
        ),
        "parameters": schema(
            {"path": {"type": "string", "minLength": 1, "maxLength": 4096}}, ["path"]
        ),
    },
]

CORPUS = [
    {
        "id": "durable-execution",
        "title": "Durable asynchronous execution / 可恢复异步执行",
        "text": (
            "Persist task checkpoints and tool results in SQLite. Resume interrupted "
            "tasks with stable invocation IDs. Use idempotency for side effects. "
            "SQLite 持久化检查点，重启恢复异步任务，幂等防止重复副作用。"
        ),
    },
    {
        "id": "structured-memory",
        "title": "Structured memory / 结构化记忆与压缩",
        "text": (
            "Store atomic facts with source IDs, validity and supersession links. "
            "Retrieve evidence for the current question within the same session. "
            "原子事实保留来源和替代关系；按问题检索证据，执行状态独立保存。"
        ),
    },
    {
        "id": "bdd",
        "title": "Spec-first BDD",
        "text": (
            "Describe observable behavior with Given, When, Then before implementation. "
            "Test recovery, FIFO ordering, isolation and failure boundaries. "
            "先写行为规范与验收场景，再实现与验证。"
        ),
    },
]


class ToolRegistry:
    def __init__(self, workspace_root: Path, search_delay: float = 0.25):
        self.workspace_root = workspace_root.resolve()
        self.search_delay = search_delay
        self.specs: dict[str, Json] = {}
        self.validators: dict[str, Draft202012Validator] = {}
        self._modes: dict[str, str] = {}
        self._handlers: dict[str, Callable] = {}
        bindings = {
            "calculator": (
                "readonly",
                lambda args, db, scope: {"value": calculate(args["expression"])},
            ),
            "search": ("async", self._search),
            "todo": ("transactional", self._todo),
            "weather": ("readonly", lambda args, db, scope: self._weather(args["city"])),
            "read_docs": ("readonly", lambda args, db, scope: self._read_docs(args["path"])),
        }
        for spec in TOOL_SPECS:
            if spec["name"] == "read_docs":
                spec = spec | {
                    "description": spec["description"] + f" Workspace root: {self.workspace_root}."
                }
            mode, handler = bindings[spec["name"]]
            self.register(spec, mode=mode, handler=handler)

    def register(self, spec: Json, *, mode: str, handler: Callable) -> None:
        name = spec["name"]
        if (
            name in self.specs
            or not re.fullmatch(r"[a-zA-Z0-9_-]{1,64}", name)
            or mode not in {"readonly", "transactional", "async"}
            or not callable(handler)
        ):
            raise ValueError("Invalid or duplicate tool registration")
        Draft202012Validator.check_schema(spec["parameters"])
        self.specs[name] = spec
        self.validators[name] = Draft202012Validator(spec["parameters"])
        self._modes[name] = mode
        self._handlers[name] = handler

    def mode(self, name: str) -> str:
        if name not in self._modes:
            raise ToolError("UNKNOWN_TOOL", "Tool is not registered")
        return self._modes[name]

    @property
    def definitions(self) -> list[Json]:
        return [{"type": "function", "function": spec} for spec in self.specs.values()]

    def parse(self, name: str, arguments: str) -> Json:
        if name not in self.specs:
            raise ToolError("UNKNOWN_TOOL", "Tool is not registered")
        try:
            args = json.loads(arguments)
        except (TypeError, ValueError, RecursionError) as exc:
            raise ToolError("INVALID_ARGUMENTS", "Arguments must be a JSON object") from exc
        if not self.validators[name].is_valid(args):
            raise ToolError("INVALID_ARGUMENTS", "Arguments do not match the tool JSON Schema")
        return args

    def execute_sync(
        self, name: str, args: Json, db: sqlite3.Connection | None, scope: str
    ) -> Json:
        try:
            if self.mode(name) == "async":
                raise ToolError("TOOL_MODE_INVALID", "Tool is not synchronous")
            return ok(self._handlers[name](args, db, scope))
        except ToolError as exc:
            if exc.transient:
                raise
            return error(exc.code, str(exc))

    async def execute_async(self, name: str, args: Json) -> Json:
        if self.mode(name) != "async":
            raise ToolError("TOOL_MODE_INVALID", "Tool is not asynchronous")
        return ok(await self._handlers[name](args))

    async def _search(self, args: Json) -> Json:
        await asyncio.sleep(self.search_delay)
        query_terms = terms(args["query"])
        scored = [
            (len(query_terms & terms(doc["title"] + " " + doc["text"])), doc) for doc in CORPUS
        ]
        hits = [doc for score, doc in sorted(scored, key=lambda item: -item[0]) if score > 0][:3]
        return {"mock": True, "query": args["query"], "hits": hits}

    def _todo(self, args: Json, db: sqlite3.Connection, scope: str) -> Any:
        action = args["action"]
        if action == "list":
            return {
                "items": [
                    dict(row)
                    for row in db.execute(
                        "SELECT id,title,completed,created_at FROM todos WHERE scope=? "
                        "ORDER BY created_at DESC,id",
                        (scope,),
                    )
                ]
            }
        if action == "create":
            title = args["title"].strip()
            if not title:
                raise ToolError("INVALID_ARGUMENTS", "Title must not be blank")
            todo_id = uuid.uuid4().hex
            db.execute(
                "INSERT INTO todos(id,scope,title,created_at) VALUES (?,?,?,?)",
                (todo_id, scope, title, time.time()),
            )
        else:
            todo_id = args["id"]
            if not db.execute(
                "SELECT id FROM todos WHERE id=? AND scope=?", (todo_id, scope)
            ).fetchone():
                raise ToolError("TODO_NOT_FOUND", "Todo does not exist in this session")
            if action == "delete":
                db.execute("DELETE FROM todos WHERE id=? AND scope=?", (todo_id, scope))
                return {"id": todo_id, "deleted": True}
            db.execute("UPDATE todos SET completed=1 WHERE id=? AND scope=?", (todo_id, scope))
        return dict(
            db.execute(
                "SELECT id,title,completed,created_at FROM todos WHERE id=? AND scope=?",
                (todo_id, scope),
            ).fetchone()
        )

    def _weather(self, city: str) -> Json:
        aliases = {
            "北京": "beijing",
            "上海": "shanghai",
            "深圳": "shenzhen",
            "伦敦": "london",
            "纽约": "new york",
        }
        fixtures = {
            "beijing": (20, "clear"),
            "shanghai": (23, "cloudy"),
            "shenzhen": (28, "rain"),
            "london": (14, "cloudy"),
            "new york": (17, "clear"),
        }
        key = aliases.get(city.strip(), city.strip().casefold())
        if key not in fixtures:
            raise ToolError("CITY_NOT_SUPPORTED", "No weather fixture for this city")
        temperature, condition = fixtures[key]
        return {
            "mock": True,
            "city": key,
            "temperature_c": temperature,
            "condition": condition,
            "as_of": "fixture",
        }

    def _read_docs(self, path: str) -> Json:
        portable = path.replace("\\", "/")
        candidate = Path(portable)
        windows = PureWindowsPath(portable)
        if (
            "\x00" in portable
            or ":" in portable[len(windows.drive) :]
            or (windows.drive and not candidate.is_absolute())
            or (candidate.root and not candidate.is_absolute())
        ):
            raise ToolError(
                "PATH_FORBIDDEN", "Use a relative or absolute file path inside the workspace"
            )
        try:
            resolved = (self.workspace_root / candidate).resolve()
        except (OSError, RuntimeError, ValueError) as exc:
            raise ToolError("DOC_UNREADABLE", "File path could not be resolved") from exc
        if not resolved.is_relative_to(self.workspace_root):
            raise ToolError("PATH_FORBIDDEN", "Resolved path escapes the workspace")
        try:
            if not resolved.is_file():
                raise ToolError("DOC_UNREADABLE", "Path must name a readable UTF-8 file")
            with resolved.open("rb") as handle:
                raw = handle.read(65_537)
            if len(raw) > 65_536:
                raise ToolError("DOC_TOO_LARGE", "Document exceeds 64 KiB")
            text = raw.decode("utf-8")
        except (OSError, UnicodeError, ValueError) as exc:
            raise ToolError("DOC_UNREADABLE", "Document is missing or is not UTF-8") from exc
        return {
            "path": resolved.relative_to(self.workspace_root).as_posix(),
            "content": text[:8000],
            "truncated": len(text) > 8000,
            "characters": len(text),
        }


def calculate(expression: str) -> int | float:
    operations = {
        ast.Add: operator.add,
        ast.Sub: operator.sub,
        ast.Mult: operator.mul,
        ast.Div: operator.truediv,
        ast.FloorDiv: operator.floordiv,
        ast.Mod: operator.mod,
        ast.Pow: operator.pow,
    }
    try:
        tree = ast.parse(expression, mode="eval")
        if sum(1 for _ in ast.walk(tree)) > 64:
            raise ValueError

        def visit(node):
            if isinstance(node, ast.Constant) and type(node.value) in (int, float):
                value = node.value
            elif isinstance(node, ast.UnaryOp) and isinstance(node.op, (ast.UAdd, ast.USub)):
                value = visit(node.operand) * (-1 if isinstance(node.op, ast.USub) else 1)
            elif isinstance(node, ast.BinOp) and type(node.op) in operations:
                left, right = visit(node.left), visit(node.right)
                if isinstance(node.op, ast.Pow) and abs(right) > 12:
                    raise ValueError
                value = operations[type(node.op)](left, right)
            else:
                raise ValueError
            if type(value) not in (int, float) or not math.isfinite(value) or abs(value) > 1e100:
                raise ValueError
            return value

        return visit(tree.body)
    except (SyntaxError, ValueError, OverflowError, ZeroDivisionError, TypeError) as exc:
        raise ToolError(
            "INVALID_EXPRESSION", "Expression is invalid or exceeds arithmetic limits"
        ) from exc
