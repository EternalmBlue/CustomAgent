import asyncio
import sys
from types import SimpleNamespace

from pytest_bdd import given, scenarios, then, when

from custom_agent import cli
from custom_agent.config import Settings

scenarios("features/config.feature")


@given("a TOML configuration with a model and database path", target_fixture="configuration")
def configuration(tmp_path):
    directory = tmp_path / "configuration"
    directory.mkdir()
    path = directory / "config.toml"
    path.write_text(
        'model = "chosen-model"\ndb_path = "state/agent.db"\nworkspace_root = "workspace"\n',
        encoding="utf-8",
    )
    return SimpleNamespace(path=path, observed=[])


@when("chat and serve start with that configuration")
def entrypoints(configuration, monkeypatch):
    class StubRuntime:
        def __init__(self, settings):
            configuration.observed.append(settings)

        async def start(self):
            pass

        async def stop(self):
            pass

        def submit(self, *args):
            return {"id": "run"}

        async def wait(self, *args, **kwargs):
            return {"answer": "ready"}

    monkeypatch.setattr(cli, "Runtime", StubRuntime)
    monkeypatch.setattr(cli, "create_app", lambda settings: configuration.observed.append(settings))
    monkeypatch.setattr(cli.uvicorn, "run", lambda app, **kwargs: app())
    for command in ("chat", "serve"):
        argv = ["custom-agent", command, "--config", str(configuration.path)]
        if command == "chat":
            argv += ["--message", "hello"]
        monkeypatch.setattr(sys, "argv", argv)
        cli.main()


@then("both entry points use its model and resolved database path")
def shared_settings(configuration):
    assert len(configuration.observed) == 2
    for settings in configuration.observed:
        assert settings.model == "chosen-model"
        assert settings.db_path == configuration.path.parent / "state/agent.db"


@when("the configuration is loaded from a different working directory")
def other_working_directory(configuration, monkeypatch, tmp_path):
    monkeypatch.chdir(tmp_path)
    configuration.settings = Settings.from_file(configuration.path)


@then("database and workspace paths remain relative to the configuration")
def resolved_roots(configuration):
    assert configuration.settings.db_path == configuration.path.parent / "state/agent.db"
    assert configuration.settings.workspace_root == configuration.path.parent / "workspace"


@given(
    "a configuration containing a wrong type and a private marker",
    target_fixture="invalid_configuration",
)
def invalid_configuration(tmp_path):
    path = tmp_path / "invalid.toml"
    path.write_text(
        'api_key = "private-config-marker"\nmodel_timeout = "private-config-marker"\n',
        encoding="utf-8",
    )
    return SimpleNamespace(path=path, created=False, error=None)


@when("chat tries to load that configuration")
def invalid_chat(invalid_configuration, monkeypatch):
    def runtime(settings):
        invalid_configuration.created = True

    monkeypatch.setattr(cli, "Runtime", runtime)
    try:
        asyncio.run(cli.chat(SimpleNamespace(config=invalid_configuration.path)))
    except ValueError as exc:
        invalid_configuration.error = str(exc)


@then("configuration is rejected without exposing the marker or creating a runtime")
def safe_rejection(invalid_configuration):
    assert invalid_configuration.error is not None
    assert "private-config-marker" not in invalid_configuration.error
    assert not invalid_configuration.created
