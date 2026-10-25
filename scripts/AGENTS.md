# Local Verification Scripts

Applies to this directory. Read ../AGENTS.md and the API/SSE contract in
../src/custom_agent/AGENTS.md.

http_smoke.py checks a running server over actual HTTP/SSE using a fresh scope,
a durable 202 submission and oversized required context. Expect
CONTEXT_BUDGET_EXCEEDED with zero decisions, then verify ordered event replay
using Last-Event-ID. It is transport/error-path evidence, not model success.

Use trust_env=False for local HTTP clients because Windows registry proxies may
otherwise intercept loopback requests. Preserve provider proxy behavior.
Keep --url configurable and write sanitized results to ../reports.

Run from the repository root:

```powershell
.\.venv\Scripts\python.exe scripts/http_smoke.py
```
