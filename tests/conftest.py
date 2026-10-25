from pathlib import Path

import pytest

from custom_agent.config import Settings


def pytest_addoption(parser):
    parser.addoption(
        "--agent-config",
        type=Path,
        default=Path("config.toml"),
        help="TOML configuration for explicitly enabled real-provider tests",
    )


@pytest.fixture
def provider_settings(request):
    settings = Settings.from_file(request.config.getoption("--agent-config"))
    if not settings.api_key.strip():
        pytest.fail("Set api_key in the TOML selected by --agent-config")
    return settings
