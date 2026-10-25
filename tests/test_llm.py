import json

import httpx
import pytest

from custom_agent.llm import ModelError, OpenAICompatibleModel


async def test_adapter_sends_tool_schema_and_strips_reasoning():
    seen = []

    def handler(request):
        seen.append(json.loads(request.content))
        return httpx.Response(
            200,
            json={
                "choices": [
                    {
                        "message": {
                            "role": "assistant",
                            "content": "ok",
                            "reasoning_content": "private",
                        },
                        "finish_reason": "stop",
                    }
                ],
                "usage": {"prompt_tokens": 12, "completion_tokens": 2},
            },
        )

    model = OpenAICompatibleModel("test-secret", "https://api.deepseek.com", "deepseek-chat")
    await model._client.aclose()
    model._client = httpx.AsyncClient(
        base_url="https://api.deepseek.com/", transport=httpx.MockTransport(handler)
    )
    try:
        result = await model.complete(
            [{"role": "user", "content": "hello"}],
            [{"type": "function", "function": {"name": "test"}}],
        )
        assert result.message == {"role": "assistant", "content": "ok"}
        assert result.usage["prompt_tokens"] == 12
        assert seen[0]["thinking"] == {"type": "disabled"}
        assert seen[0]["tools"]
        assert "test-secret" not in json.dumps(seen)
        await model.complete([{"role": "user", "content": "json"}], [], purpose="memory")
        assert seen[-1]["response_format"] == {"type": "json_object"}
    finally:
        await model.close()


@pytest.mark.parametrize(("status", "transient"), [(401, False), (429, True), (500, True)])
async def test_provider_errors_do_not_leak_response_bodies(status, transient):
    model = OpenAICompatibleModel("private-key", "https://example.test", "model")
    await model._client.aclose()
    model._client = httpx.AsyncClient(
        base_url="https://example.test/",
        transport=httpx.MockTransport(lambda request: httpx.Response(status, text="private-key")),
    )
    try:
        with pytest.raises(ModelError) as captured:
            await model.complete([{"role": "user", "content": "hello"}], [])
        assert captured.value.transient == transient
        assert "private-key" not in str(captured.value)
    finally:
        await model.close()
