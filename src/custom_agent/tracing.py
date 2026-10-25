from __future__ import annotations

import asyncio
import json
import re
import traceback
from pathlib import Path
from typing import TYPE_CHECKING

from .llm import Json

if TYPE_CHECKING:
    from .config import Settings
    from .llm import Model, ModelReply
    from .store import Store

_HIDDEN = {
    "reasoning",
    "reasoningcontent",
    "reasoningdetails",
    "chainofthought",
    "hiddenreasoning",
}
_SENSITIVE = {
    "authorization",
    "proxyauthorization",
    "apikey",
    "accesstoken",
    "refreshtoken",
    "password",
    "secret",
    "cookie",
    "setcookie",
    "apitokens",
}
_CREDENTIAL = re.compile(r"\bsk-[A-Za-z0-9_-]{8,}|\bBearer\s+[A-Za-z0-9._~+/\-=]+", re.IGNORECASE)


def safe_code(exc: BaseException, fallback: str) -> str:
    code = getattr(exc, "code", None)
    return (
        code if isinstance(code, str) and re.fullmatch(r"[A-Z][A-Z0-9_]{0,79}", code) else fallback
    )


class Redactor:
    def __init__(self, secrets=()):
        self._secrets = tuple(sorted({value for value in secrets if value}, key=len, reverse=True))

    def clean(self, value, _depth: int = 0):
        if _depth > 64:
            return "[OMITTED_DEPTH]"
        if isinstance(value, dict):
            result = {}
            for key, item in value.items():
                normalized = re.sub(r"[^a-z]", "", str(key).lower())
                if normalized in _HIDDEN:
                    continue
                result[self.clean(str(key), _depth + 1)] = (
                    "[REDACTED]" if normalized in _SENSITIVE else self.clean(item, _depth + 1)
                )
            return result
        if isinstance(value, (list, tuple)):
            return [self.clean(item, _depth + 1) for item in value]
        if isinstance(value, str):
            # Tool arguments and protocol contents often contain serialized JSON.
            if value.lstrip().startswith(("{", "[")):
                try:
                    decoded = json.loads(value)
                except (ValueError, RecursionError):
                    pass
                else:
                    return json.dumps(
                        self.clean(decoded, _depth + 1), ensure_ascii=False, separators=(",", ":")
                    )
            for secret in self._secrets:
                value = value.replace(secret, "[REDACTED]")
            return _CREDENTIAL.sub("[REDACTED]", value)
        return value

    def diagnostic(self, exc: BaseException) -> Json:
        return self.clean(
            {
                "exception_type": type(exc).__name__,
                "frames": [
                    {
                        "file": Path(frame.filename).name,
                        "function": frame.name,
                        "line": frame.lineno,
                    }
                    for frame in traceback.extract_tb(exc.__traceback__, limit=12)
                ],
            }
        )


async def model_call(
    store: Store,
    model: Model,
    settings: Settings,
    run_id: str,
    messages: list[Json],
    tools: list[Json],
    *,
    purpose: str = "agent",
    decision: int | None = None,
    max_tokens: int = 1024,
    context: Json | None = None,
    validate=None,
) -> tuple[ModelReply, str]:
    attempt_id = store.start_model_attempt(
        run_id,
        purpose,
        decision,
        {
            "messages": messages,
            "tools": tools,
            "model": getattr(model, "model", settings.model),
            "purpose": purpose,
            "max_tokens": max_tokens,
            "context": context,
        },
    )
    reply = None
    try:
        async with asyncio.timeout(settings.model_timeout):
            reply = await model.complete(messages, tools, purpose=purpose, max_tokens=max_tokens)
        if validate is not None:
            validate(reply.message)
    except asyncio.CancelledError:
        # Persisted RUNNING is reconciled as INTERRUPTED by the next coordinator.
        raise
    except Exception as exc:
        code = "MODEL_TIMEOUT" if isinstance(exc, TimeoutError) else safe_code(exc, "MODEL_ERROR")
        usage = reply.usage if reply is not None else getattr(exc, "usage", None)
        with store.transaction():
            if usage is not None:
                store.add_usage(run_id, usage, purpose)
            store._end_attempt(
                attempt_id,
                "FAILED",
                error_code=code,
                diagnostic=store.redactor.diagnostic(exc),
                response=reply.message if reply is not None else None,
                usage=usage,
                finish_reason=reply.finish_reason if reply is not None else None,
            )
        raise
    with store.transaction():
        store.add_usage(run_id, reply.usage, purpose)
        store._end_attempt(
            attempt_id,
            "SUCCEEDED",
            response=reply.message,
            usage=reply.usage,
            finish_reason=reply.finish_reason,
        )
    return reply, attempt_id
