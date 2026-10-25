Feature: Durable session-scoped agent execution
  Scenario: Storage failure does not stop the coordinator
    Given an empty runtime
    When event storage fails and then becomes available again
    Then the scheduler remains alive and queued inputs resume in FIFO order

  Scenario: Malformed tool arguments remain queryable as trace evidence
    Given an empty runtime
    When the model sends malformed JSON tool arguments
    Then the trace preserves the rejected arguments without a server error
  Scenario: CLI startup failure still closes its runtime
    Given a CLI runtime whose startup fails
    When the CLI starts
    Then its runtime resources are closed

  Scenario: Tool retry preserves every execution attempt
    Given an empty runtime
    When a read-only tool fails transiently before succeeding
    Then both attempts and the original retry cause remain in the trace

  Scenario: Model requests expose bounded observable evidence
    Given an empty runtime
    When a user asks for a direct reply
    Then the model attempt retains context, timing and actual usage

  Scenario: Direct model reply is durable
    Given an empty runtime
    When a user asks for a direct reply
    Then the run completes and the reply survives reopening the database

  Scenario: Todo writes are isolated and idempotent
    Given an empty runtime
    When the model creates a todo and the invocation is replayed
    Then exactly one todo exists in the originating scope
    And another session cannot read or change it

  Scenario: Async execution resumes after restart
    Given a run waiting for an asynchronous search
    When the runtime restarts
    Then the search result resumes the original run without repeating prior effects

  Scenario: Same-session inputs queue behind an asynchronous tool
    Given a run waiting for an asynchronous search
    When a second input arrives in the same session
    Then the second run starts only after the first run finishes

  Scenario: Different sessions progress independently
    Given a slow asynchronous run in one session
    When another session requests a direct reply
    Then the other session finishes before the slow task

  Scenario: Invalid tool arguments never execute
    Given an empty runtime
    When the model supplies invalid todo arguments
    Then the model receives a validation error and no todo is created

  Scenario: Budget exhaustion preserves successful effects
    Given an empty runtime with a one-decision budget
    When the model creates a todo
    Then the run is incomplete and its todo remains

  Scenario: SSE reconnect replays scoped durable events
    Given a completed run with persisted events
    When a client reconnects after a known event ID
    Then only newer events for that user and session are replayed in order

  Scenario: Model-selected relative and absolute workspace files are readable
    Given a runtime with a workspace source file
    When the model requests that file by relative and absolute paths
    Then both tool results contain the workspace file content

  Scenario: Workspace file reads cannot escape their root
    Given a runtime with a workspace source file
    When the model requests a file outside the workspace
    Then the tool returns a path error without the outside file content

  Scenario: Over-nested argument JSON remains a recoverable tool error
    Given an empty runtime
    When the model sends tool arguments beyond the JSON nesting limit
    Then the trace preserves the rejected arguments without a server error
