# Acceptance And Regression Tests

Applies to this directory. Read ../AGENTS.md and the relevant file-module
contract in ../src/custom_agent/AGENTS.md before changing behavior or assertions.

## BDD

Write observable Given/When/Then scenarios in features/ before implementing
behavior changes. Execute them with pytest-bdd via test_bdd_runtime.py and
test_bdd_memory.py. Do not treat unexecuted Gherkin as acceptance evidence.
Configuration scenarios execute through test_bdd_config.py; they cover CLI
and HTTP loading the same TOML, config-relative roots, and safe rejection before
resource creation. Never require credentials for ordinary configuration tests.

Preserve these acceptance cases:

- Durable direct replies and tool-result follow-ups.
- Todo isolation, replay idempotency and rollback of effect/result together.
- Async restart, same-scope FIFO through waits and other-session progress.
- Invalid arguments/unknown tools never execute; failures become safe feedback.
- Decision exhaustion retains committed effects; model failure unblocks FIFO.
- Scoped ordered SSE replay and cursor precedence.
- Decision supersession, historical reasons and session-local memory/evidence.
- Compression preserves complete tool protocol and exact execution counters.
- Invalid extraction leaves source history and the watermark intact.
- Owned session creation/list/detail/rename, archive/restore and idle deletion.
- Archived sessions reject new messages while accepted tasks keep executing.

## Regression Boundaries

Use FakeModel for deterministic tests; ordinary runs make no paid provider calls.
Use temporary databases, close runtimes, and never overwrite the user's live DB.
Cover hard child-process kill recovery as well as graceful restarts.
Trace coverage must prove rolled-back effects retain failed attempts, retries
retain causes, invalid calls have no executed attempts, model/extraction usage is
recorded per call without double counting, and snapshots/diagnostics are redacted.
Verify trace ownership, task/invocation links, interrupted timing uncertainty,
version-2 migration and transactional deletion of the new attempt records.
Inject repeated event-write failures and queue-read errors: the scheduler must
survive, report degraded health, and finish failed runs before starting queued
same-scope work after storage recovery. Preserve committed Todo effects and close
pending tool protocol without repeating handlers. Check worker exceptions and
shutdown during a continuing fault. Trace malformed raw argument JSON must return
200 with redaction and an explicit parse error. CLI startup failure must release
resources, including when model cleanup raises.
Over-nested argument JSON must produce INVALID_ARGUMENTS feedback and remain
queryable as raw trace evidence, without starting a tool attempt.

Retain targeted tests for persisted attempt limits, commit interruption windows,
closed failed tool-call groups, coordinator exclusion, bounded shutdown and
cleanup after model close errors. SSE replay must stop at its initial cutoff
even when events are appended across the 200-row page boundary.
Exercise version-1 database upgrade and repeat opening, archive/request replay,
busy deletion, full dependent-record deletion and injected deletion rollback.
Check cross-user ownership for all lifecycle/history endpoints, pagination,
validation and SSE termination after deletion/recreation.

Memory pressure tests must exercise an individually valid long interaction,
full extraction input/output accounting, mandatory constraints, current-message
exclusion from historical evidence, and recovery of deep observation evidence.
Test empty summaries' discard receipts and extraction usage independently from
the decision budget.

Keep calculator/JSON Schema/path-boundary tests, including Windows drive paths,
traversal, alternate streams, symlink escapes and file/output size bounds.
Cover model-selected relative and absolute workspace files, custom workspace
configuration and rejection of files outside the root. Use fixtures/ for
project reference fixtures, addressed relative to the project workspace;
path-escape tests use temporary roots. A missing Windows symlink privilege may
skip that case explicitly.

Configuration regressions must cover every Settings field, omitted defaults,
unknown keys/tables, wrong types, invalid ranges, nonfinite numbers, missing and
malformed files, secret-safe errors/repr, path resolution from another working
directory, and legacy environment variables having no effect. Verify --config
and --db reach the actual CLI/HTTP/evaluation entry points.

## Commands And Live Gate

From the repository root:

```powershell
.\.venv\Scripts\python.exe -m pytest -q
.\.venv\Scripts\python.exe -m custom_agent.evaluate
```

Real tests are opt-in: AGENT_LIVE=1 plus a provider key in config.toml.
AGENT_LIVE is a test-only opt-in, not runtime configuration. Use pytest
--agent-config to select a private TOML, shared by all live suites.
test_live.py uses dedicated temporary databases and validates all five model
tool selections, Todo follow-ups/isolation, async continuation, real summary
extraction, effective decisions and historical reasons from seeded history.
Reports include decisions, usage and latency without secrets.
test_live_sessions.py uses a separate loopback HTTP server and temporary database
with the real provider. Cover multi-turn Todo, archive during asynchronous search,
accepted FIFO work, restart recovery, restore, ownership, SSE replay and deletion.
Search results remain local fixtures; the provider selects tools and generates
the follow-up answers. Always stop the test server without touching the main one.
test_live_exceptions.py uses the same explicit live gate and separate loopback
servers/databases. Inject SQLite event-write failure after a real model-selected
Todo and during asynchronous search; verify 503, FIFO, protocol closure and a real
model continuation after recovery. Inject malformed and over-nested argument
strings into an actual provider reply; label this boundary mutation explicitly
and verify trace queries plus real provider correction. Exercise CLI startup
cleanup with a real HTTP client, including close failure, and prove a replacement
runtime can acquire the same database and complete a real model request.
Save sanitized per-test evidence without request snapshots or credentials.

```powershell
.\.venv\Scripts\python.exe -m pytest tests/test_live.py -vv
.\.venv\Scripts\python.exe -m custom_agent.evaluate --live --output reports/live-comparison.json
```

For a private configuration, append --agent-config config.local.toml to pytest
and --config config.local.toml to evaluate. Keep AGENT_LIVE=1 as an explicit
test-process opt-in; the actual provider settings come entirely from TOML.

Provider/auth/funding errors fail real tests; never replace them with FakeModel
passes. Policy comparisons must use identical snapshots and report extraction
costs. Record repeatable validation knowledge here and keep generated evidence
under ../reports; do not create standalone prose reports. Historical successful
runs do not verify a new code revision.

## Retained Validation Knowledge

Historical funded-provider checks on 2026-10-25 passed eight opt-in cases:
five injected-exception cases plus tools, memory and HTTP session lifecycle.
These used actual deepseek-flash calls, temporary databases and independent
loopback test servers. Search/weather remained mock data; memory history was
seeded. Fault tests mutated actual provider replies to truncated/over-nested
JSON and injected SQLite event-write rejection. They did not establish that
the provider naturally emits malformed JSON, or simulate disk-full/corruption.
CLI startup checks called the real chat function in-process; live session
restart was graceful. Hard-kill recovery is separate child-process coverage.
These boundaries matter more than historical suite counts.

An earlier Windows HTTP smoke passed assertions but hit locked-file cleanup.
Always stop test servers, close database handles and await workers before
temporary-directory removal. A missing Windows symlink privilege remains an
explicit skipped boundary rather than a claimed successful path test.
