import json
import os
import subprocess
import sys
import threading
from dataclasses import fields
from http.server import BaseHTTPRequestHandler, HTTPServer
from pathlib import Path

import httpx
import pytest

from custom_agent import api, cli, evaluate
from custom_agent.config import Settings
from custom_agent.runtime import Runtime
from tests.support import FakeModel


def test_toml_loads_every_settings_field_and_resolves_roots(tmp_path, monkeypatch):
    path = tmp_path / "config.toml"
    path.write_text(
        """
db_path = "state/agent.db"
workspace_root = "workspace"
api_key = "private-provider-marker"
base_url = "https://example.test/v1/"
model = "custom-model"
max_decisions = 3
context_tokens = 8000
reply_tokens = 500
tool_result_tokens = 1000
compression_ratio = 0.6
recent_interactions = 2
tool_timeout = 8
search_delay = 0
poll_interval = 0.03
model_timeout = 90
[api_tokens]
private-auth-marker = "alice"
""",
        encoding="utf-8",
    )
    monkeypatch.setenv("AGENT_MODEL", "ignored")
    monkeypatch.setenv("DEEPSEEK_API_KEY", "ignored")
    monkeypatch.setenv("AGENT_API_TOKENS", "invalid json")
    settings = Settings.from_file(path)
    expected = dict(
        db_path=tmp_path / "state/agent.db",
        workspace_root=tmp_path / "workspace",
        api_key="private-provider-marker",
        base_url="https://example.test/v1",
        model="custom-model",
        max_decisions=3,
        context_tokens=8000,
        reply_tokens=500,
        tool_result_tokens=1000,
        compression_ratio=0.6,
        recent_interactions=2,
        tool_timeout=8,
        search_delay=0,
        poll_interval=0.03,
        model_timeout=90,
        api_tokens={"private-auth-marker": "alice"},
    )
    assert {item.name for item in fields(settings)} == set(expected)
    assert all(getattr(settings, key) == value for key, value in expected.items())
    assert settings.input_budget == 6500
    assert "private-provider-marker" not in repr(settings)
    assert "private-auth-marker" not in repr(settings)


def test_omitted_defaults_and_legacy_environment_are_independent(tmp_path, monkeypatch):
    path = tmp_path / "config.toml"
    path.write_text("", encoding="utf-8")
    monkeypatch.chdir(tmp_path)
    monkeypatch.setenv("AGENT_WORKSPACE", "ignored")
    monkeypatch.setenv("AGENT_DB", "ignored.db")
    monkeypatch.setenv("OPENAI_API_KEY", "ignored")
    settings = Settings.from_file()
    assert settings.db_path == tmp_path / "data/agent.sqlite3"
    assert settings.workspace_root == tmp_path
    assert settings.api_key == ""
    assert settings.model == Settings().model
    assert not (tmp_path / "data").exists()


@pytest.mark.parametrize(
    "content",
    [
        'unknown_field = "private-marker"',
        '[unknown_table]\napi_key = "private-marker"',
        'context_tokens = "private-marker"',
        "context_tokens = true",
        "model = 12",
        'api_key = ["private-marker"]',
        "workspace_root = 123",
        'db_path = ["private-marker"]',
        'db_path = ""',
        'api_tokens = "private-marker"',
        "api_tokens = { token = 123 }",
        'api_tokens = { token = "" }',
        'model = ""',
        'base_url = "private-marker"',
        'base_url = "https://private-marker@example.test"',
        "max_decisions = 0",
        "recent_interactions = -1",
        "reply_tokens = 0",
        "tool_result_tokens = -1",
        "context_tokens = 1000",
        "compression_ratio = 1.0",
        "poll_interval = 0",
        "tool_timeout = true",
        "model_timeout = -1",
        "search_delay = -1",
        "search_delay = nan",
        "tool_timeout = inf",
        "compression_ratio = nan",
    ],
)
def test_invalid_configuration_is_rejected_without_value_echo(tmp_path, content):
    path = tmp_path / "config.toml"
    path.write_text(content, encoding="utf-8")
    with pytest.raises(ValueError) as error:
        Settings.from_file(path)
    assert "private-marker" not in str(error.value)


@pytest.mark.parametrize(
    "content", ['api_key = "private-marker"\ninvalid = [', 'api_key = "private-marker\n']
)
def test_toml_syntax_error_suppresses_source_and_exception_chain(tmp_path, content):
    path = tmp_path / "config.toml"
    path.write_text(content, encoding="utf-8")
    with pytest.raises(ValueError) as error:
        Settings.from_file(path)
    assert "private-marker" not in str(error.value)
    assert error.value.__suppress_context__


def test_missing_file_is_an_explicit_error_without_side_effects(tmp_path):
    with pytest.raises(FileNotFoundError):
        Settings.from_file(tmp_path / "missing.toml")
    assert not list(tmp_path.iterdir())


def test_utf8_bom_and_absolute_paths(tmp_path):
    path = tmp_path / "config.toml"
    absolute_db = tmp_path / "absolute.db"
    path.write_text(
        f'db_path = {json.dumps(str(absolute_db))}\nmodel = "configured"\n',
        encoding="utf-8-sig",
    )
    assert Settings.from_file(path).db_path == absolute_db


def test_cli_explicit_database_overrides_file_relative_to_working_directory(tmp_path):
    path = tmp_path / "config.toml"
    path.write_text('db_path = "file.db"\n', encoding="utf-8")
    args = type("Args", (), {"config": path, "db": Path("cli.db")})()
    assert cli.load_settings(args).db_path == Path("cli.db").resolve()


def test_evaluate_main_passes_selected_config(monkeypatch, tmp_path):
    selected = tmp_path / "config.toml"
    observed = {}

    async def compare(output, *, live, config_path):
        observed.update(config_path=config_path, live=live)
        return {"results": []}

    monkeypatch.setattr(evaluate, "compare", compare)
    monkeypatch.setattr(sys, "argv", ["evaluate", "--config", str(selected)])
    evaluate.main()
    assert observed == {"config_path": selected, "live": False}


async def test_evaluation_reads_toml_with_legacy_environment_ignored(tmp_path, monkeypatch):
    selected = tmp_path / "config.toml"
    selected.write_text("reply_tokens = 512\n", encoding="utf-8")
    monkeypatch.setenv("AGENT_MODEL", "ignored")
    report = await evaluate.compare(tmp_path / "comparison.json", config_path=selected)
    assert len(report["results"]) == 9


async def test_http_default_app_loads_toml_identity_and_database(tmp_path, monkeypatch):
    (tmp_path / "config.toml").write_text(
        'model = "toml-model"\ndb_path = "state/http.db"\n[api_tokens]\naccess = "alice"\n',
        encoding="utf-8",
    )
    monkeypatch.chdir(tmp_path)
    monkeypatch.setattr(api, "Runtime", lambda settings: Runtime(settings, FakeModel()))
    app = api.app_factory()
    runtime = app.state.runtime
    assert runtime.settings.model == "toml-model"
    assert runtime.settings.db_path == tmp_path / "state/http.db"
    async with app.router.lifespan_context(app):
        async with httpx.AsyncClient(
            transport=httpx.ASGITransport(app), base_url="http://test"
        ) as client:
            body = {"session_id": "work", "input": "hello"}
            denied = await client.post("/v1/runs", headers={"X-User-ID": "alice"}, json=body)
            assert denied.status_code == 401
            accepted = await client.post(
                "/v1/runs", headers={"Authorization": "Bearer access"}, json=body
            )
            assert accepted.status_code == 202
            result = await runtime.wait(accepted.json()["run_id"])
            assert result["state"] == "COMPLETED"
            assert runtime.store.owned_run(result["id"], "alice")


def test_cli_child_process_uses_toml_provider_from_another_directory(tmp_path):
    observed = []

    class Provider(BaseHTTPRequestHandler):
        def do_POST(self):
            body = json.loads(self.rfile.read(int(self.headers["Content-Length"])))
            observed.append((self.path, self.headers["Authorization"], body["model"]))
            payload = json.dumps(
                {
                    "choices": [
                        {
                            "message": {"role": "assistant", "content": "READY"},
                            "finish_reason": "stop",
                        }
                    ],
                    "usage": {"prompt_tokens": 100, "completion_tokens": 1},
                }
            ).encode()
            self.send_response(200)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(payload)))
            self.end_headers()
            self.wfile.write(payload)

        def log_message(self, *args):
            pass

    server = HTTPServer(("127.0.0.1", 0), Provider)
    worker = threading.Thread(target=server.serve_forever, kwargs={"poll_interval": 0.01})
    worker.start()
    try:
        directory = tmp_path / "configuration"
        directory.mkdir()
        selected = directory / "config.toml"
        selected.write_text(
            'api_key = "toml-private-marker"\n'
            f'base_url = "http://127.0.0.1:{server.server_port}/v1/"\n'
            'model = "selected-model"\ndb_path = "state/cli.db"\n',
            encoding="utf-8",
        )
        result = subprocess.run(
            [
                sys.executable,
                "-m",
                "custom_agent.cli",
                "chat",
                "--config",
                str(selected),
                "--message",
                "hello",
                "--json",
            ],
            cwd=tmp_path,
            env=dict(os.environ, NO_PROXY="*", DEEPSEEK_API_KEY="ignored"),
            capture_output=True,
            text=True,
            timeout=15,
        )
        assert result.returncode == 0, result.stderr
        run = json.loads(result.stdout)
        assert run["state"] == "COMPLETED" and run["answer"] == "READY"
        assert "toml-private-marker" not in result.stdout + result.stderr
        assert observed == [
            ("/v1/chat/completions", "Bearer toml-private-marker", "selected-model")
        ]
        assert (directory / "state/cli.db").is_file()
        assert not (tmp_path / "state").exists()
    finally:
        server.shutdown()
        worker.join(timeout=2)
        server.server_close()
