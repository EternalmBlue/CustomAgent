from __future__ import annotations

import asyncio
import json
import logging
import os
import sqlite3
import time
from pathlib import Path

from .config import Settings
from .llm import Json, Model, ModelError, OpenAICompatibleModel
from .memory import ContextError, ContextManager
from .store import TERMINAL, Store
from .tools import ToolError, ToolRegistry, error
from .tracing import model_call, safe_code

log = logging.getLogger(__name__)


class ProcessLock:
    def __init__(self, path: Path):
        self.path = path
        self.handle = None

    def acquire(self) -> None:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.handle = self.path.open("a+b")
        try:
            self.handle.seek(0)
            if os.name == "nt":
                import msvcrt

                if self.path.stat().st_size == 0:
                    self.handle.write(b"0")
                    self.handle.flush()
                self.handle.seek(0)
                msvcrt.locking(self.handle.fileno(), msvcrt.LK_NBLCK, 1)
            else:
                import fcntl

                fcntl.flock(self.handle.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
        except OSError as exc:
            self.handle.close()
            self.handle = None
            raise RuntimeError("Another runtime already owns this database") from exc

    def release(self) -> None:
        if self.handle:
            self.handle.close()
            self.handle = None


class Runtime:
    def __init__(self, settings: Settings, model: Model | None = None):
        self.settings = settings
        self.store = Store(
            settings.db_path, secrets=(settings.api_key, *settings.api_tokens.keys())
        )
        self.model = model or OpenAICompatibleModel(
            settings.api_key, settings.base_url, settings.model, settings.model_timeout
        )
        self.tools = ToolRegistry(settings.workspace_root, settings.search_delay)
        self.memory = ContextManager(self.store, self.model, settings)
        self._lock = ProcessLock(Path(str(settings.db_path.resolve()) + ".runtime.lock"))
        self._scheduler: asyncio.Task | None = None
        self._active: dict[str, tuple[str, asyncio.Task]] = {}
        self._pending_failures: dict[str, tuple[str, Json]] = {}
        self._scheduler_error: str | None = None
        self._wake = asyncio.Event()
        self._stopping = False
        self._closed = False

    async def start(self) -> None:
        if self._closed:
            raise RuntimeError("Runtime is closed")
        if self._scheduler:
            return
        self._lock.acquire()
        try:
            self.store.recover()
            self._scheduler = asyncio.create_task(self._schedule(), name="agent-scheduler")
        except BaseException:
            self._lock.release()
            raise

    async def stop(self) -> None:
        if self._closed:
            return
        self._stopping = True
        self._wake.set()
        if self._scheduler:
            self._scheduler.cancel()
            await asyncio.gather(self._scheduler, return_exceptions=True)
        running = [task for _, task in self._active.values()]
        for task in running:
            task.cancel()
        await asyncio.gather(*running, return_exceptions=True)
        try:
            await self.model.close()
        finally:
            try:
                self.store.close()
            finally:
                self._lock.release()
                self._closed = True

    def submit(
        self, user_id: str, session_id: str, text: str, request_id: str | None = None
    ) -> Json:
        if self._closed:
            raise RuntimeError("Runtime is closed")
        if (
            not user_id.strip()
            or not session_id.strip()
            or not text.strip()
            or (request_id is not None and not request_id.strip())
            or len(text) > 16000
            or max(len(user_id), len(session_id), len(request_id or "")) > 128
        ):
            raise ValueError("Invalid scope, request ID or input (max 16000 characters)")
        run = self.store.submit(user_id, session_id, text, request_id)
        self._wake.set()
        return run

    async def wait(self, run_id: str, timeout: float = 90.0) -> Json:  # noqa: ASYNC109
        deadline = time.monotonic() + timeout
        while time.monotonic() < deadline:
            run = self.store.run(run_id)
            if not run:
                raise KeyError(run_id)
            if run["state"] in TERMINAL:
                return run
            await asyncio.sleep(self.settings.poll_interval)
        raise TimeoutError(f"Run {run_id} did not finish")

    async def _schedule(self) -> None:
        retry_delay = 0.1
        while not self._stopping:
            try:
                for scope, (run_id, task) in list(self._active.items()):
                    if task.done():
                        del self._active[scope]
                        try:
                            task.result()
                        except (asyncio.CancelledError, Exception) as exc:
                            self._fail_run(run_id, "RUNTIME_ERROR", exc)
                self._wake.clear()
                # Terminalize failures before admitting later same-scope work.
                for run_id in list(self._pending_failures):
                    self._finalize_failure(run_id)
                for run in self.store.runnable():
                    if run["scope"] not in self._active:
                        self._active[run["scope"]] = (
                            run["id"],
                            asyncio.create_task(self._drive(run["id"]), name=f"run-{run['id']}"),
                        )
                self._scheduler_error = None
            except Exception as exc:
                self._scheduler_fault(exc)
            if self._scheduler_error:
                await asyncio.sleep(retry_delay)
                retry_delay = min(retry_delay * 2, 1.0)
                continue
            retry_delay = 0.1
            try:
                async with asyncio.timeout(self.settings.poll_interval):
                    await self._wake.wait()
            except TimeoutError:
                pass

    def _scheduler_fault(self, exc: Exception) -> None:
        code = "STORAGE_UNAVAILABLE" if isinstance(exc, sqlite3.Error) else "SCHEDULER_ERROR"
        if code != self._scheduler_error:
            log.error(
                "scheduler error=%s diagnostic=%s",
                code,
                json.dumps(self.store.redactor.diagnostic(exc)),
            )
        self._scheduler_error = code

    def _finalize_failure(self, run_id: str) -> None:
        code, diagnostic = self._pending_failures[run_id]
        self.store.finish(run_id, "FAILED", None, code, diagnostic=diagnostic)
        del self._pending_failures[run_id]

    def _fail_run(self, run_id: str, code: str, exc: BaseException) -> None:
        diagnostic = self.store.redactor.diagnostic(exc)
        log.error("run=%s error=%s diagnostic=%s", run_id, code, json.dumps(diagnostic))
        self._pending_failures[run_id] = (code, diagnostic)
        try:
            self._finalize_failure(run_id)
        except Exception as storage_error:
            self._scheduler_fault(storage_error)

    def health(self) -> Json:
        if self._closed or self._stopping or self._scheduler is None:
            return {"status": "degraded", "error": "RUNTIME_NOT_RUNNING"}
        if self._scheduler.done():
            return {"status": "degraded", "error": "SCHEDULER_STOPPED"}
        code = self._scheduler_error
        if self._pending_failures:
            code = code or "STORAGE_UNAVAILABLE"
        if code is None:
            try:
                self.store.one("SELECT version FROM schema_version")
            except sqlite3.Error:
                code = "STORAGE_UNAVAILABLE"
        return {"status": "degraded", "error": code} if code else {"status": "ok"}

    async def _drive(self, run_id: str) -> None:
        try:
            self.store.activate(run_id)
            while True:
                run = self.store.run(run_id)
                if run["state"] in TERMINAL:
                    return
                for call in self.store.pending_calls(run_id):
                    await self._execute(call)
                run = self.store.run(run_id)
                if run["decisions"] >= self.settings.max_decisions:
                    self.store.finish(
                        run_id,
                        "INCOMPLETE",
                        "The decision budget was exhausted. Completed tool effects are preserved.",
                        "DECISION_BUDGET_EXHAUSTED",
                    )
                    return
                context = await self.memory.build(run, self.tools.definitions)
                with self.store.transaction():
                    self.store.event(
                        run["scope"],
                        run_id,
                        "context.built",
                        {"estimated_tokens": context.estimated_tokens, **context.breakdown},
                    )
                slot = self.store.reserve_decision(run_id, self.settings.max_decisions)
                if slot is None:
                    continue
                reply = None
                for attempt in range(2):
                    try:
                        reply, attempt_id = await model_call(
                            self.store,
                            self.model,
                            self.settings,
                            run_id,
                            context.messages,
                            self.tools.definitions,
                            decision=slot,
                            max_tokens=self.settings.reply_tokens,
                            context={
                                "estimated_tokens": context.estimated_tokens,
                                **context.breakdown,
                            },
                            validate=self._validate_reply,
                        )
                        break
                    except (ModelError, TimeoutError) as exc:
                        transient = isinstance(exc, TimeoutError) or exc.transient
                        if attempt or not transient:
                            raise
                        with self.store.transaction():
                            failed = self.store.one(
                                "SELECT id FROM execution_attempts WHERE run_id=? "
                                "AND kind='model' AND decision=? ORDER BY rowid DESC LIMIT 1",
                                (run_id, slot),
                            )
                            self.store.event(
                                run["scope"],
                                run_id,
                                "model.retried",
                                {
                                    "decision": slot,
                                    "attempt": attempt + 1,
                                    "attempt_id": failed["id"],
                                    "error_code": "MODEL_TIMEOUT"
                                    if isinstance(exc, TimeoutError)
                                    else safe_code(exc, "MODEL_ERROR"),
                                },
                            )
                        await asyncio.sleep(0.1)
                self.store.save_reply(
                    run_id,
                    slot,
                    reply.message,
                    reply.usage,
                    reply.finish_reason,
                    attempt_id=attempt_id,
                )
        except asyncio.CancelledError:
            raise
        except (ModelError, ContextError) as exc:
            code = safe_code(exc, "MODEL_ERROR") if isinstance(exc, ModelError) else str(exc)
            self._fail_run(run_id, code, exc)
        except TimeoutError as exc:
            self._fail_run(run_id, "MODEL_TIMEOUT", exc)
        except Exception as exc:
            self._fail_run(run_id, "RUNTIME_ERROR", exc)
        finally:
            self._wake.set()

    @staticmethod
    def _validate_reply(message: Json) -> None:
        try:
            calls = message.get("tool_calls", [])
            if not isinstance(calls, list) or len(calls) > 16:
                raise ValueError
            if message.get("content") is not None and not isinstance(message["content"], str):
                raise ValueError
            if not calls and not message.get("content"):
                raise ValueError
            for call in calls:
                function = call["function"]
                if not isinstance(function["name"], str) or len(function["name"]) > 128:
                    raise ValueError
                if not isinstance(function["arguments"], (str, dict)):
                    raise ValueError
                if len(json.dumps(function["arguments"])) > 16000:
                    raise ValueError
        except (KeyError, TypeError, ValueError) as exc:
            raise ModelError("MODEL_RESPONSE_INVALID") from exc

    async def _execute(self, call: Json) -> None:
        try:
            args = self.tools.parse(call["name"], call["arguments"])
        except ToolError as exc:
            result = error(exc.code, str(exc))
            self.store.reject_call(call["id"], result)
            return
        mode = self.tools.mode(call["name"])
        if mode == "transactional":
            self.store.complete_sync(
                call["id"],
                lambda db, scope: self.tools.execute_sync(call["name"], args, db, scope),
            )
            return
        if mode == "readonly":
            await self._readonly(call, args)
            return
        task = self.store.ensure_task(call)
        # An interrupted attempt may be retried after recovery without any side effect.
        # The persisted attempt budget also bounds repeated crashes.
        while task["attempts"] < 2:
            attempt = self.store.task_attempt(task["id"])
            try:
                async with asyncio.timeout(self.settings.tool_timeout):
                    result = await self.tools.execute_async(call["name"], args)
                self.store.complete_task(task["id"], result)
                return
            except (ToolError, TimeoutError) as exc:
                failure = self.store.fail_tool_attempt(call["id"], exc)
                transient = isinstance(exc, TimeoutError) or exc.transient
                if attempt < 2 and transient:
                    run = self.store.run(call["run_id"])
                    with self.store.transaction():
                        self.store.event(
                            run["scope"],
                            run["id"],
                            "task.retried",
                            {
                                "task_id": task["id"],
                                "invocation_id": call["id"],
                                "decision": call["decision"],
                                "attempt": attempt,
                                **failure,
                            },
                        )
                    task = self.store.one("SELECT * FROM tasks WHERE id=?", (task["id"],))
                    continue
                self.store.complete_task(
                    task["id"],
                    error(
                        "TOOL_TIMEOUT" if isinstance(exc, TimeoutError) else exc.code,
                        "Tool execution failed; no result is available",
                    ),
                )
                return
            except Exception as exc:
                self.store.fail_tool_attempt(call["id"], exc)
                self.store.complete_task(
                    task["id"], error("TOOL_ERROR", "Asynchronous tool execution failed")
                )
                return
        self.store.complete_task(
            task["id"],
            error("TOOL_ATTEMPTS_EXHAUSTED", "Interrupted attempts exhausted the retry limit"),
        )

    async def _readonly(self, call: Json, args: Json) -> None:
        scope = self.store.run(call["run_id"])["scope"]
        while call["attempts"] < 2:
            attempt = self.store.readonly_attempt(call["id"])
            try:
                async with asyncio.timeout(self.settings.tool_timeout):
                    result = await asyncio.to_thread(
                        self.tools.execute_sync, call["name"], args, None, scope
                    )
                self.store.complete_readonly(call["id"], result)
                return
            except (ToolError, TimeoutError) as exc:
                failure = self.store.fail_tool_attempt(call["id"], exc)
                transient = isinstance(exc, TimeoutError) or exc.transient
                if attempt < 2 and transient:
                    with self.store.transaction():
                        self.store.event(
                            scope,
                            call["run_id"],
                            "tool.retried",
                            {
                                "invocation_id": call["id"],
                                "decision": call["decision"],
                                "attempt": attempt,
                                **failure,
                            },
                        )
                    call = self.store.one("SELECT * FROM invocations WHERE id=?", (call["id"],))
                    continue
                self.store.complete_readonly(
                    call["id"],
                    error(
                        "TOOL_TIMEOUT" if isinstance(exc, TimeoutError) else exc.code,
                        "Read-only tool execution failed",
                    ),
                )
                return
            except Exception as exc:
                self.store.fail_tool_attempt(call["id"], exc)
                self.store.complete_readonly(
                    call["id"], error("TOOL_ERROR", "Read-only tool execution failed")
                )
                return
        self.store.complete_readonly(
            call["id"],
            error("TOOL_ATTEMPTS_EXHAUSTED", "Interrupted attempts exhausted the retry limit"),
        )
