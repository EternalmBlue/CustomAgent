from __future__ import annotations

import json
from dataclasses import dataclass, field
from typing import Any, Protocol

import httpx

Json = dict[str, Any]


class ModelError(Exception):
    def __init__(self, code: str, *, transient: bool = False, usage: Json | None = None):
        super().__init__(code)
        self.code = code
        self.transient = transient
        self.usage = usage


def numeric_usage(value: Any, depth: int = 0) -> Json | None:
    if not isinstance(value, dict) or depth > 4:
        return None
    result = {}
    for key, item in value.items():
        if type(item) in (int, float):
            result[key] = item
        elif isinstance(item, dict):
            result[key] = numeric_usage(item, depth + 1)
    return result


@dataclass
class ModelReply:
    message: Json
    usage: Json = field(default_factory=dict)
    finish_reason: str = "stop"


class Model(Protocol):
    async def complete(
        self,
        messages: list[Json],
        tools: list[Json],
        *,
        purpose: str = "agent",
        max_tokens: int = 1024,
    ) -> ModelReply: ...

    async def close(self) -> None: ...


class OpenAICompatibleModel:
    """HTTP adapter only; the execution loop belongs to Runtime."""

    def __init__(self, api_key: str, base_url: str, model: str, timeout: float = 60.0) -> None:
        if not api_key or not api_key.strip():
            raise ValueError("Set api_key in the selected TOML configuration")
        self._client = httpx.AsyncClient(
            base_url=base_url.rstrip("/") + "/",
            headers={"Authorization": f"Bearer {api_key}"},
            timeout=timeout,
        )
        self.model = model
        self.deepseek = "deepseek.com" in httpx.URL(base_url).host

    async def complete(
        self,
        messages: list[Json],
        tools: list[Json],
        *,
        purpose: str = "agent",
        max_tokens: int = 1024,
    ) -> ModelReply:
        body: Json = {
            "model": self.model,
            "messages": messages,
            "max_tokens": max_tokens,
            "temperature": 0,
            "stream": False,
        }
        if self.deepseek:
            body["thinking"] = {"type": "disabled"}
        if tools:
            body["tools"] = tools
            body["tool_choice"] = "auto"
        if purpose == "memory":
            body["response_format"] = {"type": "json_object"}
        try:
            response = await self._client.post("chat/completions", json=body)
        except (httpx.TimeoutException, httpx.TransportError) as exc:
            raise ModelError("MODEL_TRANSPORT", transient=True) from exc
        if response.status_code >= 400:
            # Never persist arbitrary provider error bodies or authorization headers.
            raise ModelError(
                f"MODEL_HTTP_{response.status_code}",
                transient=response.status_code in {408, 429} or response.status_code >= 500,
            )
        usage = None
        try:
            payload = response.json()
            usage = numeric_usage(payload.get("usage"))
            choice = payload["choices"][0]
            source = choice["message"]
            message: Json = {"role": "assistant", "content": source.get("content")}
            if source.get("tool_calls"):
                message["tool_calls"] = source["tool_calls"]
            if not message.get("content") and not message.get("tool_calls"):
                raise ValueError("Empty response")
            # Reasoning fields are intentionally neither persisted nor replayed.
            return ModelReply(message, usage or {}, choice.get("finish_reason", "stop"))
        except (KeyError, IndexError, TypeError, ValueError, AttributeError) as exc:
            raise ModelError("MODEL_RESPONSE_INVALID", usage=usage) from exc

    async def close(self) -> None:
        await self._client.aclose()


def json_text(value: Any) -> str:
    return json.dumps(value, ensure_ascii=False, separators=(",", ":"))
