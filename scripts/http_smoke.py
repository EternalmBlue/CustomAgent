"""Check a running local server over real HTTP/SSE without a model request."""

import argparse
import asyncio
import json
import uuid
from pathlib import Path

import httpx


async def check(base_url):
    # Windows registry proxies can apply even with no *_PROXY env vars.
    async with httpx.AsyncClient(base_url=base_url, timeout=20, trust_env=False) as client:
        headers = {"X-User-ID": "transport-smoke"}
        session = "socket-" + uuid.uuid4().hex[:12]
        receipt = await client.post(
            "/v1/runs",
            headers=headers,
            json={"session_id": session, "input": "\u9f98" * 14000},
        )
        assert receipt.status_code == 202, receipt.status_code
        run_id = receipt.json()["run_id"]
        ids, kinds = [], []
        async with client.stream(
            "GET", f"/v1/sessions/{session}/events", headers=headers
        ) as stream:
            assert stream.status_code == 200
            async for line in stream.aiter_lines():
                if line.startswith("id: "):
                    ids.append(int(line[4:]))
                if line.startswith("event: "):
                    kinds.append(line[7:])
                    if line[7:] == "run.failed":
                        break
        result = (await client.get("/v1/runs/" + run_id, headers=headers)).json()
        assert result["error"] == "CONTEXT_BUDGET_EXCEEDED", result["error"]
        assert result["decisions"] == 0
        replay = await client.get(
            f"/v1/sessions/{session}/events?follow=false",
            headers={**headers, "Last-Event-ID": str(ids[0])},
        )
        replay_ids = [int(line[4:]) for line in replay.text.splitlines() if line.startswith("id: ")]
        assert replay_ids == ids[1:]
        report = {
            "health": (await client.get("/health")).status_code,
            "post": receipt.status_code,
            "events": kinds,
            "sse_replay": True,
            "expected_error": result["error"],
            "model_decisions": result["decisions"],
        }
        output = Path("reports/http-smoke.json")
        output.parent.mkdir(exist_ok=True)
        await asyncio.to_thread(output.write_text, json.dumps(report, indent=2), encoding="utf-8")
        print(json.dumps(report))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--url", default="http://127.0.0.1:8000")
    args = parser.parse_args()
    asyncio.run(check(args.url))
