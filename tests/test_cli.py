import logging
import sqlite3
import sys
from types import SimpleNamespace

import pytest

from custom_agent import cli
from custom_agent.config import Settings
from custom_agent.runtime import Runtime
from tests.support import FakeModel


@pytest.mark.parametrize("close_fails", [False, True])
async def test_startup_failure_closes_resources_and_preserves_original_error(
    tmp_path, monkeypatch, caplog, close_fails
):
    settings = Settings(db_path=tmp_path / "startup.db")
    model = FakeModel()
    closed = False

    async def close():
        nonlocal closed
        closed = True
        if close_fails:
            raise RuntimeError("private-close-error-marker")

    model.close = close
    runtime = Runtime(settings, model)

    def fail_recovery():
        raise sqlite3.OperationalError("startup recovery failed")

    monkeypatch.setattr(runtime.store, "recover", fail_recovery)
    monkeypatch.setattr(cli, "Runtime", lambda settings: runtime)
    with pytest.raises(sqlite3.OperationalError, match="startup recovery failed"):
        await cli.chat(SimpleNamespace())
    assert closed and runtime._closed
    assert runtime._lock.handle is None
    with pytest.raises(sqlite3.ProgrammingError, match="closed"):
        runtime.store.db.execute("SELECT 1")
    assert "private-close-error-marker" not in caplog.text
    replacement = Runtime(settings, FakeModel())
    await replacement.start()
    await replacement.stop()


async def test_cli_lock_conflict_cleanup_preserves_existing_coordinator(tmp_path, monkeypatch):
    settings = Settings(db_path=tmp_path / "lock.db", poll_interval=0.005)
    owner = Runtime(settings, FakeModel())
    rejected = Runtime(settings, FakeModel())
    await owner.start()
    monkeypatch.setattr(cli, "Runtime", lambda settings: rejected)
    try:
        with pytest.raises(RuntimeError, match="already owns"):
            await cli.chat(SimpleNamespace())
        assert rejected._closed and rejected._lock.handle is None
        assert owner._lock.handle is not None
        run = owner.submit("u", "s", "still works")
        assert (await owner.wait(run["id"], timeout=2))["state"] == "COMPLETED"
        third = Runtime(settings, FakeModel())
        try:
            with pytest.raises(RuntimeError, match="already owns"):
                await third.start()
        finally:
            await third.stop()
    finally:
        await rejected.stop()
        await owner.stop()


@pytest.mark.parametrize("command", ["chat", "serve"])
def test_entrypoint_log_level_is_scoped_to_package_and_stderr(monkeypatch, capsys, command):
    package_log = logging.getLogger("custom_agent")
    previous = (package_log.level, package_log.propagate, package_log.handlers[:])
    root_level = logging.getLogger().level
    package_log.handlers = []
    monkeypatch.setattr(sys, "argv", ["custom-agent", command, "--log-level", "info"])

    def emit():
        logging.getLogger("custom_agent.store").info('execution_attempt {"state":"STARTED"}')
        print("result-only")

    async def fake_chat(args):
        emit()

    monkeypatch.setattr(cli, "chat", fake_chat)
    monkeypatch.setattr(cli.uvicorn, "run", lambda *args, **kwargs: emit())
    try:
        cli.main()
        output = capsys.readouterr()
        assert output.out == "result-only\n"
        assert "execution_attempt" in output.err
        assert logging.getLogger().level == root_level
        assert not package_log.propagate
    finally:
        for handler in package_log.handlers:
            handler.close()
        package_log.level, package_log.propagate, package_log.handlers = previous
