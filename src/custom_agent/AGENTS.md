# Agent Runtime Modules

Applies to this package. Read the root AGENTS.md first. Each heading owns the
contract for the named Python file; changes across boundaries must preserve both
contracts. Acceptance and regression coverage lives in ../../tests/AGENTS.md.

## runtime.py: Scheduling And Recovery

- Lifecycle: `QUEUED -> RUNNING <-> WAITING_TOOL -> COMPLETED | INCOMPLETE | FAILED`.
  Recover every nonterminal run from persisted calls, results, counters and FIFO
  order when starting. Never reconstruct execution from semantic memory.
- At most 8 durable model decision slots by default, including recovery calls.
  Reserve a slot before the request. One transient network/model retry shares
  that slot; a crash before response commit can consume a slot.
- Calls in one response execute in order, including across asynchronous waits.
  Dispatch using registered execution metadata, not hardcoded tool names.
  Reuse committed invocation results without repeating effects.
- Commit async task acceptance and the WAITING_TOOL checkpoint before execution.
  Task lifecycle: `PENDING -> RUNNING -> SUCCEEDED | FAILED`. Restart resets
  unfinished tasks to PENDING; successful results are reused.
- Persist read-only and async attempt reservations before work. One retry means
  at most two attempts, including interrupted attempts after a crash. Default
  tool attempt timeout is 5 seconds. Invalid/unknown calls never execute.
- Execute pure/read-only synchronous handlers in threads without database access.
  A timed-out read may finish in its thread but cannot mutate persistent state.
- Exhausted decisions yield INCOMPLETE while retaining committed effects.
  Model failures yield FAILED and unblock the next same-scope run. Tool errors
  become observations for a follow-up model decision.
- One coordinator per full database filename, protected by an OS lock beside
  that database. Shutdown sets a stop condition, wakes/cancels in-process work
  and leaves durable records for restart. Database/lock cleanup must run even
  if the model client's close raises.
- Failure reporting must not terminate the coordinator when storage also fails.
  Retain pending run finalization in memory, retry it with capped backoff before
  scheduling more work, and preserve same-scope FIFO. Do not re-execute a failed
  run merely because its terminal state could not be written. Catch worker-result
  and queue-read failures at the scheduler boundary; keep cancellation effective.
  Storage recovery allows queued work to progress in the same process. This does
  not repair a corrupt/full database or persist unwritten diagnostics across exit;
  process restart still uses the existing exact durable checkpoints.

## store.py: Durable State And Atomicity

SQLite WAL, foreign keys, parameterized SQL and schema version 3. Keep transactions
short; no transaction may cross an await or network request. Upgrade existing
version-1 session records transactionally with titles, archive flags and timestamps,
preserving identities, history, execution checkpoints and compaction watermarks.
Backfill timestamps from existing runs. Upgrade version 2 additively with attempt
storage; do not invent historical attempts. Reopening version 3 is idempotent.

| Table | Owns |
| --- | --- |
| sessions | unique user/session identity, title, archive flag, timestamps and compacted-prefix watermark |
| runs | scope-local FIFO sequence, request ID, input, state, decisions and usage |
| messages | original user, observable assistant and tool protocol messages |
| invocations | planned calls, stable IDs, attempts and committed results |
| tasks | one durable async task per invocation, attempts and outcomes |
| todos | authoritative scoped Todo state |
| events | ordered persisted lifecycle events |
| memories | source-backed atomic entries, timestamps and supersession |
| execution_attempts | durable model/tool attempts, correlation, timing, safe snapshots and diagnostics |

IDs are opaque UUIDs except invocation IDs (`run:decision:position`) and
monotonically increasing message/event IDs.

Preserve these transaction boundaries:

1. Scope lookup/create, FIFO sequence, submitted run and queued event.
2. Activation's original user message, RUNNING state and started event.
3. Decision-slot reservation and model.requested event before network I/O.
4. Normalized model response, all planned invocations, usage and completed event.
   A direct response terminalizes in that same commit.
5. Todo mutation, corresponding tool result and completion event. Failure rolls
   back both effect and result. Committed replay never duplicates the effect.
6. Read-only attempt before execution, then result commit after execution.
7. Async task acceptance/checkpoint before work; outcome, tool result, resumed
   run state and event commit together afterward.
8. Validated semantic entries, supersession, prefix watermark and compaction
   event. Record extraction usage separately even when output validation fails.

Reserve every executed tool attempt in its own commit before handler execution,
including transactional tools. Rollback of Todo/result must retain the failed
attempt, while successful Todo/result/attempt outcome commit together. Validation
rejections produce results without incrementing execution attempts.
Each model network call (agent or memory) has its own attempt and actual usage,
including retries and responses rejected later. Aggregate usage exactly once.
Attempt records link run, decision, invocation and task where applicable. Record
start/end timestamps and monotonic elapsed milliseconds for observed outcomes.
Recovery marks open attempts INTERRUPTED with an observation timestamp, leaving
unknown end/duration null. Timeout ends the observed attempt, not necessarily a
read-only worker thread. Persist safe retry codes and retry links.
Session deletion includes all attempts in the same transaction.

Terminalizing an interrupted call group must commit safe RUN_ABORTED results for
abandoned invocations and fail unfinished associated tasks. A later user message
must not inherit assistant calls without matching tool results.
An optional safe failure diagnostic commits with the terminal state and protocol
closure. Failure of this transaction leaves it eligible for later finalization.

An optional request_id deduplicates within a scope. Identical input returns the
same run; different input with that request ID raises conflict.

Session lifecycle operations are owned by user identity, not model tools.
Explicit creation and automatic creation on first submission coexist. Titles
default to session_id; renaming changes only the title. Archive retains all data
and blocks new submissions with SESSION_ARCHIVED, while accepted/queued work
continues and recovers normally. An identical request_id replay remains readable
while archived. Restoring allows new submissions again.

Deletion requires no nonterminal runs or unfinished tasks; otherwise return
SESSION_BUSY. Check and delete in one transaction, removing associated messages,
events, tasks, invocations, todos, memories, runs and the session. Never affect
another user or scope. Failure must roll back the entire deletion. An old SSE
connection ends when its scope is deleted and never follows a recreated session.

Events: run.queued/started/recovered/completed/incomplete/failed,
model.requested/completed/retried, tool.started/completed,
task.accepted/started/recovered/retried/completed/failed,
context.built and memory.compacted/failed. State changes and their events commit
together. Scoped replay queries must preserve ordering and an optional upper ID.
Session creation, rename, archive and restoration emit session.created/renamed/
archived/restored with run_id=null. Hard deletion purges the session's events.

## tools.py: Five Public Tools

Register each JSON Schema together with a handler and execution mode
(`readonly`, `transactional`, `async`). Validate arguments before execution.
Return safe error codes for invalid arguments, unknown tools and handler errors.
Do not let model arguments override scope.
Malformed JSON, including nesting beyond the decoder limit, returns
INVALID_ARGUMENTS without executing the handler or aborting the run.

| Name | Execution | Contract |
| --- | --- | --- |
| calculator | synchronous readonly | bounded arithmetic AST; no eval or code execution |
| search | asynchronous | deterministic local corpus with configurable simulated delay |
| todo | synchronous transactional | create/list/complete/delete using runtime identity |
| weather | synchronous readonly | explicitly marked mock fixtures for supported cities |
| read_docs | synchronous readonly | bounded UTF-8 workspace files selected by the model's path argument |

Todo titles may repeat. Lists sort newest first, then ID; IDs survive restarts.
Complete/delete use real returned IDs and cannot affect another scope.

read_docs uses config.toml's workspace_root (default the configuration directory).
Advertise the resolved root in its model-facing definition.
Accept workspace-relative paths and absolute paths resolving inside that root,
including source, configuration and reference files. Resolve paths before the
containment check; reject outside paths, drive-relative paths, alternate data
streams, NULs and resolved symlink escapes. The model supplies the path; the root
is the allowed boundary, not a default file or a reference-only subdirectory.
Files are at most 64 KiB; return at most 8000 characters and report truncation.
Return the resolved workspace-relative path. Keep fixture ownership and recovery
marker in ../../tests/fixtures/AGENTS.md.

## memory.py: Evidence And Context

- Retain original messages and complete bounded tool results in SQLite. Exact
  execution checkpoints are independent of compression.
- Store atomic entries with kind (goal/constraint/fact/decision/pending), stable
  key, self-contained text, entities, source message IDs, observed_at, valid_at
  and supersession links. Later decisions reuse keys and preserve older reasons.
- Only complete terminal interactions before the active run are eligible for
  extraction. Advance one compacted prefix atomically. Never split an
  assistant/tool group or use active/queued work as an extraction prefix.
- Use structured output and validate source IDs against the selected prefix.
  Every user message must be cited or explicitly discardable. An empty entry
  batch needs discard receipts for every source ID. Failed/truncated/invalid
  extraction leaves previous memory and watermark intact and records failure.
- Budget the entire extraction request: instructions, existing memory, whole
  interactions and its own requested output must fit the configured context.
  Do not use an arbitrary half-input cutoff that blocks a valid long interaction.
  Large old tool observations may be excerpted with source IDs and an explicit
  full-record-retained marker.
- Retrieve session-local entries by entity/key/lexical relevance. Effective
  goals, constraints and pending items cannot be displaced by relevance limits.
  Historical questions may retrieve superseded decisions.
- Query-centered source excerpts must recover evidence deep in old tool output.
  Internal historical source lookup ends before the active run; its messages
  already appear in the conversation. Internal retrieval adds no sixth tool.
- Assemble system/tool instructions, exact active execution metadata, relevant
  memory, recent complete interactions, current interaction and source excerpts.
  Label memory/evidence as untrusted reference data; this alone is not complete
  prompt-injection protection.
- Near 70% input pressure, first mask old observations, then extract completed
  prefixes, then shrink the recent-window preference (default 6 interactions).
  Reserve reply/tool-result space and include tool schemas in input accounting.
  Do not silently evict history on extraction failure or truncate required
  current messages/effective constraints. If they cannot fit, explicitly fail
  with CONTEXT_BUDGET_EXCEEDED.
- Record conservative tokenizer estimates and their breakdown, plus actual API
  usage with extraction separately under usage.memory. Estimates are not exact
  DeepSeek tokenization. Lexical retrieval and validated citations do not prove
  semantic summary correctness or research-optimal memory.

### Recall Timing And Placement

Runtime calls ContextManager.build before every agent decision, including after
tool completion and recovery. It retrieves memory by the current run input on
each build, even when pressure is too low to trigger extraction. Reassembly after
masking/compaction repeats retrieval; compression is not a prerequisite for recall.

SQLite memories stores semantic entries; messages retains original evidence;
sessions.compacted_seq marks the compacted prefix. runs/invocations/tasks/todos
remain authoritative execution/effect state. No separate memory directory or
cross-session store is involved.

Model messages start with SYSTEM, then EXACT_EXECUTION_STATE, then
SESSION_REFERENCE_DATA. The third system-role message explicitly labels its JSON
memory/evidence as untrusted reference data; it grants no instruction authority.
Recent complete conversation and the current interaction follow it. Current
tool calls/results stay in that conversation rather than being replaced by memory.
Do not recall active-run messages again as historical sources.

## api.py: Identity, Queries And SSE

Optional bearer-token-to-user authentication; configured mode ignores X-User-ID.
Without a token mapping, X-User-ID is trusted-local demo identity. Default server
binds to loopback. Owned resources require the same identity; cross-user lookups
return 404. Health exposes no secrets.
GET /health returns 503 with a safe code while the coordinator is not running,
has stopped, has deferred failures, or observed a storage/scheduler fault. Return
200 again after a successful scheduling pass clears the fault. Probe database
readability without writes; an idle write-only fault is detected on attempted work.

| Method | Path | Behavior |
| --- | --- | --- |
| POST | /v1/runs | session_id, input, optional request_id; durable 202 receipt |
| GET | /v1/runs/{run_id} | state, answer, safe error, task IDs and usage |
| GET | /v1/tasks/{task_id} | state, attempts, result or safe error |
| GET | /v1/runs/{run_id}/trace | ordered lifecycle/messages plus invocations, tasks and execution attempts |
| POST | /v1/sessions | create an empty session; optional session_id/title; 201 |
| GET | /v1/sessions | owned list; status=active/archived/all; limit/offset; total |
| GET | /v1/sessions/{session_id} | metadata, counts and most recent run |
| PATCH | /v1/sessions/{session_id} | title and/or archived boolean; rename/archive/restore |
| DELETE | /v1/sessions/{session_id} | atomic deletion when idle; 204; busy returns 409 |
| GET | /v1/sessions/{session_id}/messages | original messages; after ID/limit pagination |
| GET | /v1/sessions/{session_id}/runs | newest runs; limit/offset pagination |
| GET | /v1/sessions/{session_id}/events | scoped SSE replay and optional follow |
| GET | /v1/sessions/{session_id}/memory | effective entries with source references |
| GET | /health | local health |

POST cannot preallocate a task ID before the model schedules an async tool.
Conflicting request ID returns 409.
Session titles are nonblank and at most 200 characters. Reject empty patches,
null patch values, unknown fields and invalid pagination. Lists default to active
sessions, ordered by updated_at descending then internal ID. Activity and metadata
updates advance updated_at. Resource lookup failures, including cross-user
session mutation attempts, return 404. Duplicate explicit session creation is 409.

SSE uses durable increasing IDs and `id > cursor`, scoped and ordered.
Last-Event-ID takes precedence over after. Delivery is at-least-once; clients
deduplicate by ID. For follow=false, capture the latest scoped event ID before
delivery and constrain every page through it; new events cannot extend replay.
Heartbeats are comments. Disconnect must not cancel execution.

Trace additions preserve events/messages compatibility and ownership checks.
Decode JSON arguments/results/snapshots and expose direct invocation/task/attempt
links. Apply redaction to the entire trace response, including older messages.
Malformed model argument JSON is valid trace evidence: preserve its raw string
with arguments_error=INVALID_JSON instead of failing the query. Valid arguments
remain decoded; do not conceal corruption of runtime-owned result records.

## tracing.py: Safe Observability

Persist redacted model-facing messages, tool schemas, model name, purpose,
requested output limit, observable response and actual per-call usage. Include
context budget estimates for agent requests. Snapshots describe the adapter call,
not raw HTTP headers or provider payloads, and are not execution checkpoints.
Remove hidden reasoning fields recursively, including JSON embedded in content.
Redact configured provider/auth credentials and recognizable credential forms.
Never log exception text, provider error bodies, locals or source lines. Safe
diagnostics contain exception class and bounded stack locations only. Structured
Python logs and durable attempt lifecycle events carry stable correlation IDs.
Snapshot redaction is conservative defense in depth, not a general secrets DLP.

## config.py And llm.py: Provider Boundary

Use immutable validated settings, with hidden secret repr. Settings.from_file
loads config.toml by default, or the explicit path passed by CLI/evaluation.
TOML keys directly match Settings fields; omitted keys use validated defaults.
Reject missing/unreadable files, malformed TOML, unknown fields, wrong types
(including booleans as numbers), nonfinite values and invalid budgets/timeouts.
Configuration errors must not echo TOML content, credential values or decoder
source lines. Do not silently fall back to defaults on an invalid file.
Resolve workspace_root and db_path against the configuration's directory,
including omitted default paths, independent of process working directory.
Programmatic Settings construction remains available for isolated tests.
No .env loading or runtime environment overrides; a single file is authoritative.
Empty api_key is allowed when loading config for deterministic tooling; actual
provider construction requires a nonblank key. api_tokens is a TOML string-to-user
table, not JSON embedded in a string. Normalize trailing slashes in base_url.
Default context is 16384 tokens, reply reserve 1024 and tool-result reserve 2048.
The root config.toml contains blank credentials and complete runnable defaults.
Use a private ignored config.local.toml for populated credentials and --config
to select it. Never copy secrets into templates or diagnostics.

llm.py is an HTTP compatibility adapter, not an execution loop. Retain observable
assistant/tool protocol and actual usage, strip reasoning fields, and surface
safe ModelError codes without persisting arbitrary provider error bodies or
authorization headers. Memory extraction requests structured JSON output.
If a successful HTTP response reports numeric usage but has an invalid assistant
payload, retain those counters on ModelError for per-attempt accounting. Unknown
usage remains null. Never attach arbitrary provider payloads to exceptions.

## cli.py: Entry Points

Support interactive chat, one-shot --message with optional --json, and serve.
Preserve --user/--session identity, /exit termination and durable restart.
Concurrent CLI and HTTP processes need distinct databases. Always close runtime
resources when exiting. A one-shot response reports its durable result.
CLI startup failure must also close the model client/database and release any
acquired lock. Cleanup failure must not mask an original startup exception.
Both chat and serve accept --log-level (debug/info/warning/error, warning by
default). Configure only the custom_agent logger on stderr; do not enable HTTP
client logging or mix execution diagnostics into chat JSON on stdout.
Both accept --config (default config.toml) and pass the selected settings to the
actual runtime/app instance. Evaluation accepts the same option. Config load
failure happens before opening a database or creating a provider client.
CLI --db optionally overrides the selected database for concurrent chat/server
use; resolve that explicit override from the process working directory.

## evaluate.py: Reproducible Measurement

Clone identical original SQLite snapshots for each policy/question. Compare
full history, observation masking and structured memory on latest decisions,
historical reasons and source recovery. Report estimated/actual tokens,
extraction overhead and latency separately.

Default fixture mode measures evidence availability, not real LLM accuracy.
--live performs real continuations and extraction; label fixture-seeded history.
Match normalized complete A/B answers, not letters inside words such as "best".
Phrase checks remain fixture-specific, not a general semantic judge.
Do not infer universal compression ratios, statistical latency wins or research
superiority from a small constructed fixture. Session history/evidence currently
scan the scoped history; very large sessions need bounded/indexed retrieval.

Historical real-provider snapshot experiment (2026-10-23): full history and
structured memory each passed 3/3 fixture answer checks; observation masking
passed 2/3, missing ORCHID-742 deep in a tool observation. Structured answer
inputs were 1764-2024 actual tokens, plus 4145 extraction input and 200-209
extraction output tokens per independent case. Baselines bypassed the simulated
8192-token runtime window. These are bounded fixture observations, not current
revision validation, general accuracy, or statistical latency/billing claims.

### Trace Interpretation Limits

Model request outcome and normalized reply/checkpoint are separate commits.
A crash between them can leave a successful request without a committed call
plan. Recovery follows checkpoints, not request snapshots. Memory-call SUCCEEDED
means the call succeeded; semantic validation may still emit memory.failed with
its attempt_id. Unknown usage is null, not evidence of zero provider billing.
Structured logs are best-effort output; SQLite attempts/events are durable truth.
