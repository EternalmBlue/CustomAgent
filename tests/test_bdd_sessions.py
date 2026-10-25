import pytest
from pytest_bdd import given, scenarios, then, when

from custom_agent.store import SessionError
from tests.support import Harness

scenarios("features/sessions.feature")


@pytest.fixture
def h(tmp_path):
    harness = Harness(tmp_path)
    yield harness
    harness.close()


@given("a session with a completed interaction")
def completed_session(h):
    h.start()
    h.run = h.submit("remember our discussion")
    h.finish()


@when("the owner renames the session")
def rename(h):
    h.runtime.store.update_session("alice", "main", title="Release planning")


@then("the list and detail show the new title and the original history")
def renamed(h):
    store = h.runtime.store
    assert store.owned_session("alice", "main")["title"] == "Release planning"
    assert store.list_sessions("alice")["items"][0]["title"] == "Release planning"
    assert store.scope("alice", "main") == h.run["scope"]
    assert store.history(h.run["scope"])[0]["payload"]["content"] == "remember our discussion"


@when("the owner archives the session")
def archive(h):
    h.runtime.store.update_session("alice", "main", archived=True)


@then("the session is hidden from the active list and rejects new work")
def archived(h):
    assert h.runtime.store.list_sessions("alice")["items"] == []
    assert h.runtime.store.list_sessions("alice", status="archived")["total"] == 1
    with pytest.raises(SessionError, match="SESSION_ARCHIVED"):
        h.submit("should not start")
    assert len(h.runtime.store.history(h.run["scope"])) == 2


@when("the owner restores the session")
def restore(h):
    h.runtime.store.update_session("alice", "main", archived=False)


@then("a new interaction completes in the same session")
def restored(h):
    second = h.submit("continue")
    assert second["scope"] == h.run["scope"] and second["seq"] == 2
    assert h.finish(second)["state"] == "COMPLETED"


@when("the owner deletes the idle session")
def delete(h):
    other = h.submit("other user", user="bob")
    h.finish(other)
    h.other = other
    h.runtime.store.delete_session("alice", "main")


@then("its history disappears and another user's session is preserved")
def deleted(h):
    store = h.runtime.store
    assert store.scope("alice", "main") is None
    assert store.history(h.run["scope"]) == []
    assert store.run(h.run["id"]) is None
    assert store.run(h.other["id"])["state"] == "COMPLETED"


@given("a session with a queued interaction")
def queued(h):
    h.run = h.submit("queued")


@when("the owner tries to delete the session")
def delete_busy(h):
    with pytest.raises(SessionError, match="SESSION_BUSY"):
        h.runtime.store.delete_session("alice", "main")


@then("deletion is rejected and the interaction remains queued")
def still_queued(h):
    assert h.runtime.store.scope("alice", "main") == h.run["scope"]
    assert h.runtime.store.run(h.run["id"])["state"] == "QUEUED"
