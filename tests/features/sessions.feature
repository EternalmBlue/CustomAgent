Feature: Owned session lifecycle
  Scenario: Sessions can be named and browsed without changing their identity
    Given a session with a completed interaction
    When the owner renames the session
    Then the list and detail show the new title and the original history

  Scenario: Archived sessions reject new work and can be restored
    Given a session with a completed interaction
    When the owner archives the session
    Then the session is hidden from the active list and rejects new work
    When the owner restores the session
    Then a new interaction completes in the same session

  Scenario: Deleting a session clears only its owned data
    Given a session with a completed interaction
    When the owner deletes the idle session
    Then its history disappears and another user's session is preserved

  Scenario: Busy sessions cannot be deleted
    Given a session with a queued interaction
    When the owner tries to delete the session
    Then deletion is rejected and the interaction remains queued
