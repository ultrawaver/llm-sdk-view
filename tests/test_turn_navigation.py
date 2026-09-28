"""A chat bubble is a link into the pane that carries it.

What the user sent is the Request; what Claude answered is the Response.
Both ends of a turn are one record, so the click only decides which of the
two panes shows - and it must be able to decide, because a bubble that
jumps to the request no matter which end was clicked makes the Response
pane unreachable from the conversation.

The time above a bubble is the same turn's own clock, shown in the
computer's zone. llm stamps UTC, so the page converts; it never prints the
stored string as if that string were local time.
"""

from datetime import datetime, timedelta, timezone

import pytest
from starlette.testclient import TestClient

from llm_sdk_view import store
from llm_sdk_view.app import SESSIONS, app
from llm_sdk_view.records import ResponseView, TurnRecord

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


# --- the clock the bubbles show --------------------------------------------


def test_a_turn_is_stamped_in_utc_so_the_page_can_convert_it(record):
    """Local time is a rendering decision; the stored fact is UTC.

    A naive stamp would be read as the browser's own zone and show a turn
    hours away from when it happened.
    """
    stamp = datetime.fromisoformat(record["timestamp"])

    assert stamp.utcoffset() == timedelta(0)


def test_every_stored_turn_carries_the_clock_its_bubble_shows(
    client, fake_provider, transports
):
    saved = send(client, "what time is this")
    loaded = client.get(f"/api/conversations/{saved['record']['conversation_id']}").json()

    for turn in loaded["turns"]:
        stamp = datetime.fromisoformat(turn["timestamp"])
        assert stamp.tzinfo is not None
        assert stamp.utcoffset() == timedelta(0)


def test_a_restored_turn_keeps_llms_own_stamp(client, fake_provider, transports):
    """Saved and reloaded are the same instant.

    llm stamps the turn while it writes it; that is the one clock every
    surface reports, rather than a second timestamp taken on the way out.
    """
    saved = send(client, "same instant either way")
    conversation_id = saved["record"]["conversation_id"]

    loaded = store.load_conversation(conversation_id)
    row = store.connect()["turns"].get(saved["turn_id"])

    assert loaded["turns"][0].timestamp == row["datetime_utc"]


# --- the page ----------------------------------------------------------------


def test_each_end_of_a_turn_opens_the_pane_that_carries_it(static_page):
    flat = " ".join(static_page.split())

    # One click handler per end, and each names the pane it belongs to.
    assert "function wireBubble(div, turnIndex, tab, label)" in flat
    assert "wireBubble(user.parentElement, index, 'request'" in flat
    assert "wireBubble(assistant.parentElement, index, 'response'" in flat
    # ...and the pane the click asked for is the one that opens, not the
    # one the inspector happened to be showing.
    assert "function showTurn(index, tab)" in flat
    assert "setTab(tab || state.tab);" in flat


def test_the_bubble_shows_the_computers_own_clock(static_page):
    flat = " ".join(static_page.split())

    # Rendered through the platform's own locale, not a hand-rolled format.
    assert "Intl.DateTimeFormat(undefined" in flat
    assert 'class="msg-time"' in flat
    # llm's stamp is UTC; one without a zone is UTC too, never local.
    assert "ZONED_STAMP" in flat
    assert "text + 'Z'" in flat


def test_a_bubble_without_a_stamp_shows_nothing_rather_than_now(static_page):
    """No clock, no time: "now" is not when an unsaved message happened."""
    flat = " ".join(static_page.split())

    assert "if (!moment) {" in flat
    assert "el.hidden = true;" in flat


def test_the_conversation_list_shows_the_same_local_clock(static_page):
    """The sidebar used to print llm's UTC stamp as if it were local."""
    flat = " ".join(static_page.split())

    assert "const when = formatMoment(momentOf(item.last_turn));" in flat


def _empty_response():
    return ResponseView(
        response_id=None, message_id=None, model=SONNET, content_blocks=[],
        text="", thinking=None, citations=[], server_tool_blocks=[],
        stop_reason=None, usage={}, response_json=None,
    )


@pytest.fixture
def record():
    """A finished turn as the page receives it, with nothing sent to anyone."""
    return TurnRecord(
        conversation_id="01jtest",
        turn_id="01jturn",
        user_input="hello",
        options={},
        request_kwargs={},
        rendered_code="",
        response=_empty_response(),
        timestamp=datetime.now(timezone.utc).isoformat(),
    ).as_dict()


def test_the_click_is_answered_by_the_list_not_by_each_bubble(static_page):
    """One listener on the list; the pane a bubble opens is written on it.

    A handler per bubble can go missing - wired twice, or wired after the
    fact - and a dead click is indistinguishable from a page that ignores
    the user, which is exactly what a bubble that does nothing looked like.
    """
    flat = " ".join(static_page.split())
    assert "messages.addEventListener('click'" in flat
    assert "event.target.closest('.msg[data-pane]')" in flat
    assert "div.dataset.pane = tab;" in flat
    # The click outranks a preview the last keystroke left queued, which
    # would otherwise land a moment later and put the draft back.
    assert "cancelPreview();" in flat
    assert "function cancelPreview()" in flat


def test_a_bubble_is_wired_to_its_own_turn(static_page):
    """The turn is written where the bubble is wired.

    It used to be left to the 1-based number addMessage was handed, so a
    bubble could carry a turn no click could resolve - and a click that
    resolves nothing looks like a click that was ignored.
    """
    assert "div.dataset.turn = String(turnIndex);" in static_page


def test_a_click_answers_even_when_nothing_else_moves(static_page):
    """Clicking the turn already on the right changes no other pixel, so the
    bubble acknowledges it: without that, the most recent turn - the one a
    hand reaches for - is indistinguishable from a dead one."""
    flat = " ".join(static_page.split())
    assert "flash(bubble);" in flat
    assert "function flash(bubble)" in flat
    assert "bubble.classList.add('hit');" in flat


def test_the_settings_card_never_swallows_a_click(static_page):
    """It describes a bubble and can be drawn over one; a read-only card has
    no business being a click target."""
    flat = " ".join(static_page.split())
    assert "pointer-events: none;" in flat
