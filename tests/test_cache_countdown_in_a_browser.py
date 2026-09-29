"""The cache countdown, measured in a real page.

A turn whose usage says the cache caught something must start the five
minute countdown immediately - that is the whole feature - and the badge
must change face as the window closes. None of that is wording: the state
is a computed colour and a running animation, so it is measured here with
getComputedStyle, not grepped for.

No key reaches the network: the turn the first test sends runs through the
conftest's fake transport, and the final Message the test installs is the
only thing the assertions depend on.
"""

import re

import pytest

pytestmark = pytest.mark.browser

# The finished Message of a turn that wrote the cache. The fake transport
# hands this back; nothing about it is invented in this file.
CACHE_WRITING_TURN = {
    "id": "msg_fake",
    "type": "message",
    "role": "assistant",
    "model": "claude-sonnet-5",
    "content": [{"type": "text", "text": "Hello world"}],
    "stop_reason": "end_turn",
    "stop_sequence": None,
    "usage": {
        "input_tokens": 2048,
        "output_tokens": 3,
        "cache_creation_input_tokens": 2048,
        "cache_read_input_tokens": 0,
    },
}

BADGE = """() => {
    const el = byId('cacheState');
    const cs = getComputedStyle(el);
    return { text: el.textContent, face: el.dataset.state,
             color: cs.color, bg: cs.backgroundColor,
             anim: cs.animationName };
}"""

AT_REMAINING = """(remaining) => {
    state.cache.anchor = {
        at: Date.now() - (CACHE_TTL_MS - remaining),
        kind: 'read', tokens: 2048,
    };
    renderCacheStatus();
    const el = byId('cacheState');
    const cs = getComputedStyle(el);
    return { text: el.textContent, face: el.dataset.state,
             color: cs.color, bg: cs.backgroundColor,
             anim: cs.animationName };
}"""


def test_a_cache_writing_turn_starts_the_countdown(
    page, key, fake_provider, response_scenario
):
    """The user's core ask: the moment a response reports cache activity,
    the clock is running. No clicking anything, no waiting for a refresh."""
    response_scenario(CACHE_WRITING_TURN)
    page.fill("#prompt", "hi")
    page.click("#send")
    page.wait_for_function("() => state.cache.anchor !== null")
    badge = page.evaluate(BADGE)
    # Five minutes, minus however long the fake turn itself took.
    assert re.match(r"^TTL [45]:", badge["text"]), badge
    assert badge["face"] == "live"
    # The live face is a real badge, not bare text: it has a background.
    assert badge["bg"] != "rgba(0, 0, 0, 0)"


def test_the_badge_escalates_as_the_window_closes(page):
    """One element, four faces: live > 60s, warn inside the minute, urgent
    inside thirty seconds, expired at zero."""
    ladder = page.evaluate(
        f"""() => {{
            const at = {AT_REMAINING};
            return {{
                live: at(180000),
                warn: at(45000),
                urgent: at(12000),
                expired: at(-5000),
            }};
        }}"""
    )
    assert ladder["live"]["face"] == "live"
    assert ladder["warn"]["face"] == "warn"
    assert ladder["urgent"]["face"] == "urgent"
    assert ladder["expired"]["face"] == "expired"
    assert ladder["expired"]["text"] == "cache expired"
    assert re.match(r"^TTL 0:", ladder["urgent"]["text"]), ladder["urgent"]
    # The faces are visibly different, not just differently named.
    colours = {
        step["color"] for step in (ladder["live"], ladder["warn"], ladder["urgent"])
    }
    assert len(colours) == 3
    # The urgent face moves.
    assert ladder["urgent"]["anim"] != "none"


def test_the_pulse_dies_when_the_user_asked_for_no_motion(page):
    """The reminder is the red, not the blink: prefers-reduced-motion kills
    the animation and keeps the state."""
    page.emulate_media(reduced_motion="reduce")
    urgent = page.evaluate(f"({AT_REMAINING})(12000)")
    assert urgent["face"] == "urgent"
    assert urgent["anim"] == "none"


def test_a_naive_utc_stamp_is_not_read_as_local_time(page):
    """llm's older stored stamps are UTC with no zone marker. Read as local
    time the anchor moves by the whole timezone offset, and a live cache
    would instantly report itself hours expired."""
    drift = page.evaluate(
        """() => {
            const end = new Date().toISOString();
            const naive = end.replace('Z', '');
            const anchor = cacheAnchorOf({
                timestamp: naive,
                response: { usage: { cache_read_input_tokens: 10 },
                            duration_ms: 2000 },
            });
            if (!anchor) return null;
            return Math.abs((Date.parse(end) - 2000) - anchor.at);
        }"""
    )
    assert drift is not None
    # The anchor must land within a few seconds of the request's start,
    # never hours away.
    assert drift < 5000


def test_a_conversation_reopened_days_later_reads_like_information(page):
    """The expired note used to count in minutes forever: a three day old
    conversation was "4320:00 ago", which is a number, not information."""
    result = page.evaluate(
        """() => {
            // cacheNote lives in the cost popover, which needs a paid turn
            // to exist; stand one in so the note has somewhere to land.
            const note = document.createElement('p');
            note.id = 'cacheNote';
            document.body.appendChild(note);
            state.cache.anchor = {
                at: Date.now() - 3 * 24 * 3600 * 1000,
                kind: 'read', tokens: 100,
            };
            renderCacheStatus();
            const out = { bar: byId('cacheState').textContent,
                          note: note.textContent };
            note.remove();
            return out;
        }"""
    )
    assert result["bar"] == "cache expired"
    assert result["note"].startswith("Expired 72h ago"), result["note"]
