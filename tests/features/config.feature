Feature: One TOML configuration source
  Scenario: CLI and HTTP share the selected configuration
    Given a TOML configuration with a model and database path
    When chat and serve start with that configuration
    Then both entry points use its model and resolved database path

  Scenario: Paths belong to the configuration directory
    Given a TOML configuration with a model and database path
    When the configuration is loaded from a different working directory
    Then database and workspace paths remain relative to the configuration

  Scenario: Invalid configuration fails before runtime creation
    Given a configuration containing a wrong type and a private marker
    When chat tries to load that configuration
    Then configuration is rejected without exposing the marker or creating a runtime
