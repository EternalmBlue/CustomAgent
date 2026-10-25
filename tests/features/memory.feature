Feature: Evidence-backed session memory
  Scenario: A changed decision supersedes the old fact
    Given completed interactions choosing A and then replacing it with B
    When completed history is compacted
    Then a follow-up receives B as the effective decision
    And a historical question retrieves the reason for replacing A

  Scenario: Compression never loses active tool protocol or execution counters
    Given an active run with a persisted tool call and prior compressed history
    When a context is assembled under pressure
    Then the tool call and its result remain paired
    And the execution checkpoint and decision count remain exact

  Scenario: Memory cannot cross session boundaries
    Given memory in one user session
    When another session assembles context
    Then the private memory and its evidence are absent

  Scenario: Failed extraction preserves history
    Given a model returning invalid memory sources
    When compaction is attempted
    Then the summary watermark does not advance
    And original evidence remains available
