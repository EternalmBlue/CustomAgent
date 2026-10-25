# CustomAgent

## Scope And Workflow

Python 3.11+ handwritten Agent Runtime with CLI, HTTP, SQLite and exactly five
default public tools. Develop spec-first with executable Given/When/Then
scenarios. Update the owning module contract before changing its behavior.
These files guide development; the application does not load or enforce them.

Before non-trivial coding or bug-fix work, use `$bug-memory-loop` for a preflight
recall scoped to this repository and the relevant paths. After fixing a
non-trivial bug with a clear root cause, capture a reusable repo-scoped lesson.
Promote to global only if it still holds without project-specific nouns.

## Contract Ownership

Read the applicable directory's AGENTS.md before changing its files. Root rules
apply throughout the repository; child files add directory-specific contracts.
The Python package is flat, so its file-module contracts are grouped under
individual headings in src/custom_agent/AGENTS.md.

| Location | Owns |
| --- | --- |
| src/custom_agent/AGENTS.md | runtime, persistence, tools, memory, provider, API and CLI contracts |
| tests/AGENTS.md | BDD acceptance, regression evidence and real-provider gates |
| scripts/AGENTS.md | local HTTP/SSE verification scripts |
| tests/fixtures/AGENTS.md | bounded read_docs fixtures |
| reports/AGENTS.md | generated validation artifacts and their interpretation |

Keep specifications with the owning modules. Do not introduce a separate docs
directory. README.md is the only standalone documentation entry and has exactly
three sections: running, system design, and memory recall timing/placement.
Consolidate valuable design decisions and validation limits into the owning
module AGENTS.md. Do not add standalone design documents, development logs,
acceptance reports or additional READMEs. Generated reports/caches are
intermediate artifacts, not an alternative source of specifications.

## Shared Invariants

- Identity is `(user_id, session_id)`, supplied by the runtime. Tool arguments
  cannot select another identity. Todo, history, memory and events are isolated.
- Same-scope runs execute FIFO, including waits for asynchronous tools.
  Different scopes progress independently.
- Exact persisted execution state and bounded original evidence are authoritative.
  Semantic summaries aid context construction and never drive recovery or Todo truth.
- Support one coordinator per database. No multi-node scheduler or external
  durable worker broker is part of this baseline.
- Do not add an Agent framework, frontend, cross-session memory, token streaming,
  cancellation API, policy DSL or runtime Spec interpreter to this scope.
- Runtime configuration comes from config.toml or an explicit --config path;
  do not load .env files or legacy environment settings. A local TOML can contain
  provider/auth credentials. Keep populated configurations private; never copy
  secrets into source, documentation, messages, traces or reports.
- Search and weather are explicit local mock fixtures; never claim live data.
- Future external side effects require upstream idempotency or reconciliation;
  do not promise universal exactly-once execution.

## Verification

Run commands from the repository root using the project virtual environment:

```powershell
.\.venv\Scripts\python.exe -m pytest -q
.\.venv\Scripts\python.exe -m ruff check src tests scripts
.\.venv\Scripts\python.exe -m ruff format --check src tests scripts
```

Install with `pip --isolated` to avoid the host's global prefix override.
Run tests appropriate to the change. Paid real-provider tests require explicit
opt-in as described in tests/AGENTS.md. Record new results with their actual date;
do not present historical live results as a fresh run.
