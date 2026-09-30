"""The bar above the chat names the conversation the chat belongs to.

A saved conversation shows its own name; an unsaved one says what it is
instead of inventing a title. The pencil renames the conversation in llm's
own database, so the sidebar, the title bar and `llm logs` all read the
same string - a rename that only reached one of them would be the same
kind of drift this project exists to prevent.
"""

import pytest
from starlette.testclient import TestClient

from native_api_chat import store
from native_api_chat.app import SESSIONS, app

SONNET = "claude-sonnet-5"

pytestmark = pytest.mark.usefixtures("key")


@pytest.fixture
def client():
    SESSIONS.clear()
    yield TestClient(app)
    SESSIONS.clear()


def send(client, text):
    return client.post(
        "/api/chat", json={"session_id": "default", "text": text, "model": SONNET}
    ).json()


# --- the rename itself -------------------------------------------------------


def test_a_rename_reaches_every_reader_of_the_threads_row(
    client, fake_provider, transports
):
    saved = send(client, "a conversation about thresholds")

    renamed = client.post(
        f"/api/conversations/{saved['record']['conversation_id']}/name",
        json={"name": "threshold tuning"},
    )

    assert renamed.status_code == 200
    assert renamed.json()["name"] == "threshold tuning"
    # The sidebar list, the opened conversation and the stored row are one fact.
    listing = client.get("/api/conversations").json()["items"]
    assert listing[0]["name"] == "threshold tuning"
    opened = client.get(f"/api/conversations/{saved['record']['conversation_id']}").json()
    assert opened["name"] == "threshold tuning"
    assert store.load_conversation(saved["record"]["conversation_id"])["name"] == (
        "threshold tuning"
    )


def test_renaming_an_unknown_conversation_is_a_404_not_a_quiet_noop(client):
    response = client.post(
        "/api/conversations/01jno-such-thread/name", json={"name": "ghost"}
    )

    assert response.status_code == 404
    assert "not found" in response.json()["error"]


@pytest.mark.parametrize("name", ["", "   ", " \t "])
def test_an_empty_name_is_refused_not_stored(client, fake_provider, transports, name):
    """A conversation with no name looks unsaved; the old name stays."""
    saved = send(client, "keep my name")
    conversation_id = saved["record"]["conversation_id"]
    before = store.load_conversation(conversation_id)["name"]

    response = client.post(
        f"/api/conversations/{conversation_id}/name", json={"name": name}
    )

    assert response.status_code == 400
    assert store.load_conversation(conversation_id)["name"] == before


def test_a_rename_is_stored_the_way_names_are_written(client, fake_provider, transports):
    """Whitespace inside the label is collapsed, as the first-message title is."""
    saved = send(client, "spacing")

    renamed = client.post(
        f"/api/conversations/{saved['record']['conversation_id']}/name",
        json={"name": "  spaced   out  "},
    )

    assert renamed.json()["name"] == "spaced out"


# --- the page ----------------------------------------------------------------


def test_the_bar_above_the_chat_names_the_conversation(static_page):
    flat = " ".join(static_page.split())

    assert 'id="chatTitleBar"' in flat
    assert 'id="chatTitle"' in flat
    assert "function renderChatTitle()" in flat
    # An unsaved conversation has no name to show: the bar says what it is.
    assert "New conversation" in flat
    # The first saved turn's name arrives with the refreshed list, so the
    # bar reads it there too, not only when a conversation is opened.
    assert "state.conversationName = current.name;" in flat


def test_the_pencil_edits_the_name_in_place(static_page):
    flat = " ".join(static_page.split())

    assert 'id="renameConversation"' in flat
    assert 'id="chatTitleInput"' in flat
    assert "function startRename()" in flat
    assert "function commitRename()" in flat
    # Enter or leaving the field keeps the new name; Escape keeps the old.
    assert "event.key === 'Enter'" in flat
    assert "event.key === 'Escape'" in flat
    # One write, in llm's own database - not a label kept only on the page.
    assert "+ '/name'" in flat
