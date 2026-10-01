"""Conversations that survive a restart, stored in llm's own SQLite.

These tests are the ones the round asked to be able to prove offline: that a
saved turn can be read back as itself, that a resumed conversation carries the
real history, that two conversations never bleed into each other, and that a
turn which could not be written is never reported as saved.

The database is a temporary file every time, so a run can neither read nor
damage the history `llm` keeps for everything else.
"""

import json

import pytest
from starlette.testclient import TestClient

from native_api_chat import rates_openrouter, store
from native_api_chat.app import SESSIONS, app

SONNET = "claude-sonnet-5"
HAIKU = "claude-haiku-4-5-20251001"

# Sending requires a key; a machine without one must say so instead of
# quietly failing, and these tests are about storage, not about keys.
pytestmark = pytest.mark.usefixtures("key")


@pytest.fixture
def client():
    SESSIONS.clear()
    yield TestClient(app)
    SESSIONS.clear()


@pytest.fixture
def database(isolated_history):
    return store.connect()


def send(client, text, **kwargs):
    payload = {"session_id": "default", "text": text, "model": SONNET}
    payload.update(kwargs)
    return client.post("/api/chat", json=payload).json()


def restart(client):
    """Forget everything held in memory, as a restarted server would."""
    SESSIONS.clear()


# --- one turn, written and read back ----------------------------------------


def test_a_saved_turn_comes_back_as_itself(client, fake_provider, transports):
    saved = send(client, "first question")
    loaded = client.get(f"/api/conversations/{saved['record']['conversation_id']}").json()

    assert loaded["turns"][0]["turn_id"] == saved["turn_id"]
    assert loaded["turns"][0]["user_input"] == "first question"
    assert loaded["turns"][0]["request_kwargs"] == saved["kwargs"]
    assert loaded["turns"][0]["rendered_code"] == saved["code"]
    assert loaded["turns"][0]["response"]["text"] == saved["text"]


def test_the_database_holds_what_the_page_showed(client, fake_provider, transports):
    """Storage does not keep a second, differently-shaped version of a turn."""
    saved = send(client, "same request either way")
    conversation_id = saved["record"]["conversation_id"]
    loaded = client.get(f"/api/conversations/{conversation_id}").json()

    # The timestamp differs by design: once saved, llm's own clock for the
    # turn is the one every surface reports. Everything else is identical.
    page = dict(saved["record"])
    page.pop("timestamp")
    stored = dict(loaded["turns"][0])
    stored.pop("timestamp")
    assert page == stored
    assert loaded["turns"][0]["timestamp"]


def test_the_title_is_the_first_message(client, fake_provider, transports):
    saved = send(client, "Explain prompt caching to me")
    listed = client.get("/api/conversations").json()

    assert listed["items"][0]["name"] == "Explain prompt caching to me"
    assert listed["items"][0]["turns"] == 1
    assert saved["record"]["conversation_id"] == listed["items"][0]["id"]


def test_a_long_title_is_trimmed_not_truncated_mid_word(client, fake_provider, transports):
    send(client, "word " * 40)
    name = client.get("/api/conversations").json()["items"][0]["name"]

    assert name.endswith("…")
    assert len(name) <= 61
    assert not name[:-1].endswith(" ")


# --- resuming ---------------------------------------------------------------


def test_a_restarted_server_resumes_a_conversation(client, fake_provider, transports):
    """Nothing is needed from memory: history and messages both come from disk."""
    first = send(client, "remember this question")
    conversation_id = first["record"]["conversation_id"]
    restart(client)

    listed = client.get("/api/conversations").json()
    assert listed["items"][0]["id"] == conversation_id

    loaded = client.get(f"/api/conversations/{conversation_id}").json()
    assert [turn["user_input"] for turn in loaded["turns"]] == ["remember this question"]


def test_a_third_turn_carries_the_whole_history(client, fake_provider, transports):
    """Restored then extended: turn three must still contain turns one and two."""
    first = send(client, "one")
    conversation_id = first["record"]["conversation_id"]
    send(client, "two", conversation_id=conversation_id)
    restart(client)

    third = send(client, "three", conversation_id=conversation_id)
    messages = third["kwargs"]["messages"]

    assert [message["role"] for message in messages] == [
        "user",
        "assistant",
        "user",
        "assistant",
        "user",
    ]
    texts = [
        block.get("text")
        for message in messages
        for block in message.get("content", [])
        if isinstance(block, dict)
    ]
    assert texts[0] == "one"


def test_the_history_is_the_stored_messages_not_a_rebuild(
    client, fake_provider, transports
):
    """The restored chain comes from llm's message store, so it is exact."""
    sent = send(client, "preserved question")
    conversation_id = sent["record"]["conversation_id"]
    restart(client)

    messages = store.thread_messages(conversation_id)
    roles = [getattr(message, "role", None) for message in messages]

    assert roles == ["user", "assistant"]


# --- two conversations never mix --------------------------------------------


def test_two_conversations_are_independent(client, fake_provider, transports):
    first = send(client, "conversation one")
    restart(client)
    second = send(client, "conversation two")

    first_id = first["record"]["conversation_id"]
    second_id = second["record"]["conversation_id"]
    assert first_id != second_id

    one = client.get(f"/api/conversations/{first_id}").json()
    two = client.get(f"/api/conversations/{second_id}").json()
    assert [t["user_input"] for t in one["turns"]] == ["conversation one"]
    assert [t["user_input"] for t in two["turns"]] == ["conversation two"]

    listed = client.get("/api/conversations").json()["items"]
    assert len(listed) == 2


def test_switching_back_and_forth_uses_the_right_history(
    client, fake_provider, transports
):
    first = send(client, "alpha")
    first_id = first["record"]["conversation_id"]
    second = send(client, "beta", new_conversation=True)
    second_id = second["record"]["conversation_id"]

    resumed = send(client, "alpha again", conversation_id=first_id)

    texts = [
        block.get("text")
        for message in resumed["kwargs"]["messages"]
        for block in message.get("content", [])
        if isinstance(block, dict)
    ]
    assert "alpha" in texts
    assert "beta" not in texts
    assert second_id != first_id


# --- each model keeps its own options ---------------------------------------


def test_options_are_stored_with_the_turn(client, fake_provider, transports):
    saved = client.post(
        "/api/chat",
        json={
            "session_id": "default",
            "text": "recorded with options",
            "model": HAIKU,
            "max_tokens": 512,
            "thinking": "off",
        },
    ).json()
    loaded = client.get(f"/api/conversations/{saved['record']['conversation_id']}").json()

    assert loaded["turns"][0]["options"]["model"] == HAIKU
    assert loaded["turns"][0]["options"]["max_tokens"] == 512


# --- failures are failures ---------------------------------------------------


def test_a_failed_request_records_no_success(client, fake_provider, transports, monkeypatch):
    """A rejected send must not leave a turn behind claiming it answered."""
    import llm_anthropic

    def explode(**kwargs):
        raise RuntimeError("provider rejected the request")

    monkeypatch.setattr(llm_anthropic, "Anthropic", explode)

    response = client.post(
        "/api/chat", json={"session_id": "default", "text": "this fails", "model": SONNET}
    )
    body = response.json()

    assert response.status_code == 502
    assert "failed" in body["error"]
    assert client.get("/api/conversations").json()["items"] == []


def test_the_same_failure_reaches_the_streaming_route(
    client, fake_provider, transports, monkeypatch
):
    """Streaming reports the failure too, instead of closing silently."""
    import llm_anthropic

    def explode(**kwargs):
        raise RuntimeError("provider refused the stream")

    monkeypatch.setattr(llm_anthropic, "Anthropic", explode)

    response = client.post(
        "/api/chat/stream",
        json={"session_id": "default", "text": "this fails", "model": SONNET},
    )

    assert "the request failed" in response.text
    assert "record" not in response.text
    assert client.get("/api/conversations").json()["items"] == []


def test_a_save_that_fails_says_so(client, fake_provider, transports, monkeypatch):
    """Billed but unsaved is a loss the page has to report as one."""

    def broken(value):
        raise OSError("disk full")

    # Break what is being written rather than the writer, so this exercises
    # the path a real storage failure takes.
    monkeypatch.setattr(store, "_json", broken)

    result = send(client, "this will not be saved")

    assert result["saved"] is False
    assert "could not be saved" in result["error"]
    assert client.get("/api/conversations").json()["items"] == []


def test_a_failed_save_leaves_no_partial_turn(client, fake_provider, transports, monkeypatch):
    """One transaction: a written turn always has every surface of itself."""

    def broken(value):
        raise OSError("disk full")

    monkeypatch.setattr(store, "_json", broken)
    send(client, "half written")

    database = store.connect()

    assert database["turns"].count == 0
    assert database[store.SIDECAR_TABLE].count == 0
    assert database["threads"].count == 0


def test_an_unknown_conversation_is_not_an_empty_one(client, fake_provider):
    response = client.get("/api/conversations/does-not-exist")

    assert response.status_code == 404


# --- who sent a stored turn -------------------------------------------------


def test_a_row_that_predates_the_provider_field_is_priced_from_the_right_one(
    client, database, openrouter_registry, fake_openrouter
):
    """The provider is read back off the row when the row does not carry one.

    A response records the provider that sent it, and that field arrived with
    the OpenRouter prices - so every conversation already on disk has none.
    A stored OpenRouter turn's model is the catalogue slug, which no provider
    claims, so the cost fell through to Anthropic's rates whatever the model
    was: a turn priced in the catalogue came back "no estimate", and the page
    blamed rates that had the model all along.

    The id the turn was *sent with* is in the row and answers the same
    question - it is the id ``provider_for`` exists to judge - so it is read
    rather than guessed at.
    """
    (model,) = openrouter_registry("openai/gpt-5.4")
    fake_openrouter.answers(
        "hello",
        usage={
            "input_tokens": 20,
            "output_tokens": 639,
            "input_tokens_details": {"cached_tokens": 0},
        },
    )

    saved = send(client, "hello", model=model)
    conversation_id = saved["record"]["conversation_id"]
    stored = json.loads(
        database[store.SIDECAR_TABLE].get(saved["turn_id"])["response_json"]
    )
    assert stored["provider"] == "openrouter"
    # Exactly what a row written before the field existed looks like.
    del stored["provider"]
    database[store.SIDECAR_TABLE].update(
        saved["turn_id"], {"response_json": json.dumps(stored)}
    )

    loaded = client.get(f"/api/conversations/{conversation_id}").json()
    response = loaded["turns"][0]["response"]

    assert response["provider"] == "openrouter"
    # The Anthropic page this used to be asked cannot price this model at
    # all, so naming the document is what says the right one was asked.
    assert response["cost"]["rates_source"] == rates_openrouter.RATES_SOURCE
    assert response["cost"]["total"] > 0


def test_a_record_that_does_say_which_provider_sent_it_keeps_its_word(
    client, database, openrouter_registry, fake_openrouter
):
    """The stored value wins; the recovery only fills a gap."""
    (model,) = openrouter_registry("vendor/model")
    fake_openrouter.answers("hello")

    saved = send(client, "hello", model=model)
    stored = json.loads(
        database[store.SIDECAR_TABLE].get(saved["turn_id"])["response_json"]
    )
    stored["provider"] = "someone-else"
    database[store.SIDECAR_TABLE].update(
        saved["turn_id"], {"response_json": json.dumps(stored)}
    )

    loaded = client.get(f"/api/conversations/{saved['record']['conversation_id']}").json()

    assert loaded["turns"][0]["response"]["provider"] == "someone-else"


# --- llm's own tooling can read it ------------------------------------------


def test_llm_logs_can_see_the_conversation(client, fake_provider, transports, database):
    """Same schema, same tables: `llm logs` lists what this app wrote."""
    from llm.logs import LogStore, merged_log_rows

    saved = send(client, "visible to llm logs")
    rows = list(merged_log_rows(LogStore(database)))

    assert len(rows) == 1
    assert rows[0]["id"] == saved["turn_id"]
    assert rows[0]["conversation_id"] == saved["record"]["conversation_id"]
    assert rows[0]["conversation_name"] == "visible to llm logs"
    assert "sonnet-5" in rows[0]["model"]


def test_the_write_is_marked_as_ours_without_new_columns(
    client, fake_provider, transports, database
):
    """Provenance rides in a table of its own, not someone else's column."""
    send(client, "who wrote this")

    row = list(
        database.query(
            "select source from {table} limit 1".replace("{table}", store.SIDECAR_TABLE)
        )
    )[0]
    upstream = set(database["turns"].columns_dict)

    assert row["source"] == "native-api-chat"
    assert "source" not in upstream


def test_the_sidecar_from_before_the_rename_is_carried_over(isolated_history):
    """A database written as llm-sdk-view keeps its turns under the new name.

    The rows are this app's own, so the rename moves them rather than leaving
    them in a table nothing reads. Their ``source`` is not rewritten: that
    column says who wrote the row, and llm-sdk-view is who wrote these.
    """
    path = store.database_path()
    old = store.connect(path)
    old[store.SIDECAR_TABLE].drop()
    old[store.LEGACY_SIDECAR_TABLE].create(
        {"turn_id": str, "source": str, "user_input": str},
        pk="turn_id",
        foreign_keys=(("turn_id", "turns", "id"),),
    )
    old["turns"].insert({"id": "t1"}, pk="id", alter=True)
    old[store.LEGACY_SIDECAR_TABLE].insert(
        {"turn_id": "t1", "source": "llm-sdk-view", "user_input": "written before"}
    )
    old.conn.commit()
    old.conn.close()

    db = store.connect(path)
    rows = list(db[store.SIDECAR_TABLE].rows)

    assert not db[store.LEGACY_SIDECAR_TABLE].exists()
    assert [row["user_input"] for row in rows] == ["written before"]
    assert rows[0]["source"] == "llm-sdk-view"
    assert db[store.SIDECAR_TABLE].foreign_keys[0].other_table == "turns"

    # Opening it again must find the new table and do nothing.
    db.conn.close()
    assert len(list(store.connect(path)[store.SIDECAR_TABLE].rows)) == 1


def test_the_store_never_uses_the_users_own_database(isolated_history):
    """Every test talks to a temporary file, never~/Library."""
    path = store.database_path()

    assert path.is_relative_to(isolated_history)


def test_streaming_writes_once_the_stream_finishes(client, fake_provider, transports):
    """The streaming route saves the same single turn, not one per chunk."""
    response = client.post(
        "/api/chat/stream",
        json={"session_id": "default", "text": "streamed turn", "model": SONNET},
    )
    body = response.text
    records = [
        json.loads(part[6:])
        for part in body.split("\n\n")
        if part.startswith("data: ") and '"record"' in part
    ]

    assert len(records) == 1
    assert records[0]["saved"] is True
    assert client.get("/api/conversations").json()["items"][0]["turns"] == 1
