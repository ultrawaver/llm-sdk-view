"""The two cost scopes, measured in a real page.

The footer used to describe whichever turn was selected, and both bubbles
hovered up the same settings card. So the conversation had no total on screen
and the answer looked like a second copy of the request. The split is: the
footer totals the conversation, hovering an answer shows that turn's receipt,
hovering the question shows the settings that went out.

None of that is wording. Where a card lands, whether it is on screen at all,
which width it wears, whether the footer's number really is the sum of the
turns, and whether every figure left in the cost bar is drawn whole are
measured here with getBoundingClientRect and getComputedStyle.

No key reaches the network: every turn runs through the conftest fake
transport, and the finished Message each test installs is the only thing the
assertions depend on.
"""

import json
import re

import pytest

pytestmark = pytest.mark.browser

# One turn's finished Message. Sonnet 5 is on the pricing page the suite
# installs, and the usage carries all four counters a receipt can show, so the
# card has input, cache-write, cache-read and output lines to draw.
PRICED_TURN = {
    "id": "msg_fake",
    "type": "message",
    "role": "assistant",
    "model": "claude-sonnet-5",
    "content": [{"type": "text", "text": "Hello world"}],
    "stop_reason": "end_turn",
    "stop_sequence": None,
    "usage": {
        "input_tokens": 1000,
        "output_tokens": 200,
        "cache_creation_input_tokens": 500,
        "cache_read_input_tokens": 4000,
    },
}

# A second turn that costs a different amount, so "the footer did not move"
# cannot be true by accident of the two turns being identical.
SECOND_TURN = {
    **PRICED_TURN,
    "usage": {**PRICED_TURN["usage"], "output_tokens": 900},
}

SEARCHING_TURN = {
    **PRICED_TURN,
    "usage": {
        **PRICED_TURN["usage"],
        "server_tool_use": {"web_search_requests": 2},
    },
}

CARD = """() => {
    const el = byId('settingsCard');
    const cs = getComputedStyle(el);
    const r = el.getBoundingClientRect();
    return {
        cls: el.className,
        text: el.textContent,
        display: cs.display,
        pointer: cs.pointerEvents,
        width: Math.round(r.width),
        left: Math.round(r.left),
        top: Math.round(r.top),
        bottom: Math.round(r.bottom),
        lines: el.querySelectorAll('.cost-line').length,
        segments: el.querySelectorAll('.cost-stack span').length,
        vw: window.innerWidth,
        vh: window.innerHeight,
    };
}"""

FOOTER = """() => ({
    total: byId('costTotal').textContent,
    tokens: byId('costTokens').textContent,
    hit: byId('costHit').textContent,
    perTurn: state.turns.map((turn) => turn.response.cost && turn.response.cost.total),
    sum: money(state.turns.reduce(
        (total, turn) => total + ((turn.response.cost && turn.response.cost.total) || 0), 0)),
})"""

# The bar as the browser laid it out. Nothing here reads a stylesheet: it is
# widths and rects, because that is the only place "it fits" exists.
BAR = """() => {
    const bar = byId('costBar');
    const outer = bar.getBoundingClientRect();
    const cells = ['costTotal', 'costTokens', 'costHit', 'cacheState'];
    const toggle = byId('costToggle').getBoundingClientRect();
    return {
        text: bar.textContent,
        tokens: byId('costTokens').textContent,
        hit: byId('costHit').textContent,
        // A hidden cell has no box at all, which is the difference between
        // "dropped" and "drawn too small to read".
        hitShown: byId('costHit').offsetWidth > 0,
        // scrollWidth past clientWidth is the browser having had to cut the
        // cell, ellipsis or no ellipsis. A display:none cell measures 0 and 0
        // and so is not a cut.
        cut: cells.filter((id) => {
            const el = byId(id);
            return el.scrollWidth > el.clientWidth + 1;
        }),
        overflow: bar.scrollWidth > bar.clientWidth + 1,
        buttonWhole: toggle.left >= outer.left - 1
            && toggle.right <= outer.right + 1,
        widths: {
            bar: Math.round(outer.width),
            pane: Math.round(
                byId('chatPane').getBoundingClientRect().width),
        },
    };
}"""


def settle(page):
    """Wait for the pane to stop moving.

    The grid columns are animated, so a measurement taken straight after a
    resize measures a pane that is still changing - which is how the first
    version of this check reported a button 4px outside a bar it was inside.
    """
    page.wait_for_function(
        """() => new Promise((done) => {
            const width = () => byId('costBar').getBoundingClientRect().width;
            const first = width();
            requestAnimationFrame(() => requestAnimationFrame(
                () => done(Math.abs(width() - first) < 0.5)));
        })"""
    )


def send(page, text, expect, install, message=PRICED_TURN):
    """Send a turn and wait for it to be recorded and for Send to come back.

    The finished Message is installed before every send, because the plugin
    consumes - and mutates - the one it is handed: a scenario installed once
    serves one turn only, and the second turn of a shared install dies with
    KeyError('usage') inside the plugin rather than in anything this file
    asserts.
    """
    install(message)
    page.fill("#prompt", text)
    page.click("#send")
    page.wait_for_function(
        f"() => state.turns.length === {expect} && !streaming && !byId('send').disabled"
    )


def show_card(page, selector):
    page.locator(selector).first.hover()
    page.wait_for_selector("#settingsCard:not([hidden])")
    return page.evaluate(CARD)


def test_the_footer_totals_the_conversation(
    page, key, fake_provider, response_scenario
):
    """The number under the composer belongs to the conversation, and it is
    the sum of the turns - not the turn the pane happens to be showing."""
    send(page, "first", 1, response_scenario)
    send(page, "second", 2, response_scenario)
    footer = page.evaluate(FOOTER)
    assert footer["tokens"] == "2 turns", footer["tokens"]
    assert footer["perTurn"][0] == footer["perTurn"][1]
    assert footer["perTurn"][0] is not None
    assert footer["total"] == footer["sum"]
    # Two identical turns: the sum is twice one of them, so a footer that had
    # kept describing a single turn could not produce this number.
    assert footer["total"] == page.evaluate(
        "(one) => money(one * 2)", footer["perTurn"][0]
    )


def test_clicking_a_bubble_does_not_move_the_total(
    page, key, fake_provider, response_scenario
):
    """Picking a turn decides what the right pane shows and nothing else.

    The two turns are priced differently on purpose: a footer that still
    described the selection would have been moved by this click.
    """
    send(page, "first", 1, response_scenario)
    send(page, "second", 2, response_scenario, SECOND_TURN)
    footer = page.evaluate(FOOTER)
    assert footer["perTurn"][0] != footer["perTurn"][1]
    before = page.evaluate("() => byId('costTotal').textContent")
    page.locator("#messages .msg.user").first.click()
    page.wait_for_function("() => state.selected === 0")
    after = page.evaluate("() => byId('costTotal').textContent")
    assert before == after == footer["sum"]


def test_the_details_popover_totals_the_conversation(
    page, key, fake_provider, response_scenario
):
    send(page, "first", 1, response_scenario)
    send(page, "second", 2, response_scenario)
    page.click("#costToggle")
    pop = page.evaluate(
        """() => {
            const el = byId('costPop');
            const r = el.getBoundingClientRect();
            const bar = byId('costBar').getBoundingClientRect();
            return {
                hidden: el.hidden,
                text: el.textContent,
                total: el.querySelector('.cost-total-row .num').textContent,
                heading: el.querySelector('h3').textContent,
                lines: el.querySelectorAll('.cost-line').length,
                // The wrapper is also the bar's query container now, which
                // applies layout containment to it: the popover must still
                // float above the bar, in the window, where it was.
                width: Math.round(r.width),
                left: Math.round(r.left),
                right: Math.round(r.right),
                bottom: Math.round(r.bottom),
                barTop: Math.round(bar.top),
                vw: window.innerWidth,
            };
        }"""
    )
    assert pop["hidden"] is False
    assert pop["heading"].startswith("Conversation")
    assert "Priced turns" in pop["text"]
    assert "2 of 2" in pop["text"]
    footer = page.evaluate(FOOTER)
    assert pop["total"] == footer["sum"]
    assert pop["width"] > 200, pop
    assert pop["bottom"] <= pop["barTop"] + 1, pop
    assert 0 <= pop["left"] and pop["right"] <= pop["vw"], pop


def test_hovering_an_answer_shows_that_turns_own_receipt(
    page, key, fake_provider, response_scenario
):
    """The receipt the footer used to show for the selected turn, on the
    bubble whose turn it is - and readable: four columns need the wider card,
    and a card that lands off screen is not a card."""
    send(page, "hi", 1, response_scenario)
    card = show_card(page, "#messages .msg.assistant")
    assert "costcard" in card["cls"]
    assert "turn 1 · cost" in card["text"]
    # One row per counter the Message reported.
    for label in ("Uncached input", "Cache write · 5m TTL",
                  "Cache read (hit)", "Output"):
        assert label in card["text"], card["text"]
    # The composition bar is drawn from the same lines.
    assert card["segments"] == 4
    assert "Total" in card["text"]
    assert card["display"] != "none"
    # Read-only: it must never swallow the click that picks the turn.
    assert card["pointer"] == "none"
    assert 0 <= card["top"] and card["bottom"] <= card["vh"], card
    assert 0 <= card["left"] and card["left"] + card["width"] <= card["vw"], card
    # The receipt's own width, not the settings card's narrower one.
    assert card["width"] > 300, card


def test_hovering_a_question_shows_the_settings_not_the_receipt(
    page, key, fake_provider, response_scenario
):
    """The complaint: both bubbles hovered up the same card. The settings are
    what the request asked for, so they stay on the question; the answer
    carries its own figures."""
    send(page, "hi", 1, response_scenario)
    card = show_card(page, "#messages .msg.user")
    assert "turn 1 · as sent" in card["text"]
    assert "costcard" not in card["cls"]
    assert card["lines"] == 0
    assert card["width"] < 300, card


def test_the_footer_bar_keeps_only_what_fits(
    page, key, fake_provider, response_scenario
):
    """The complaint: the bar's figures were squeezed until they could not be
    read. The token totals were the widest thing it carried, so they are gone
    from it - and so is the tool-call count, which would have put the same
    squeeze back the first time a conversation used web search.

    What replaced "let one cell shrink" is a rule that can be measured: every
    figure the bar shows, it shows whole, and the cache hit rate - the one
    figure the receipt already derives in full - is absent rather than
    abbreviated when it cannot be. Making the hit rate the elastic cell instead
    would only have moved the defect one cell to the right: at 1100px it was
    drawn 82px wide for 104px of text, and at 960 all 12 of those pixels.

    Nothing here reads the source. `scrollWidth <= clientWidth` is the test for
    "whole" - it holds whether or not the browser drew an ellipsis - and the
    button's rect against the bar's says whether a clipped bar ate it.
    """
    send(page, "first", 1, response_scenario)
    send(page, "second", 2, response_scenario)

    # 1280 is a window this app is used at. The two panes take half of what is
    # left of it, so 960 leaves the bar 338px to say all of this in - and 314
    # of that is the money, the scope, the countdown badge and the button.
    for width in (1280, 1100, 1024, 960):
        page.set_viewport_size({"width": width, "height": 900})
        settle(page)
        seen = page.evaluate(BAR)
        assert seen["tokens"] == "2 turns", (width, seen)
        assert seen["cut"] == [], (width, seen)
        assert not seen["overflow"], (width, seen)
        assert seen["buttonWhole"], (width, seen)

    # The default window has room for the whole bar, hit rate included, so the
    # drop rule cannot pass by degenerating into "the hit rate is always gone".
    page.set_viewport_size({"width": 1280, "height": 900})
    settle(page)
    roomy = page.evaluate(BAR)
    assert roomy["hitShown"] is True, roomy
    assert re.match(r"cache [\d.]+% hit$", roomy["hit"]), roomy["hit"]

    # The narrowest pane has not, and there the hit rate is absent - not halved.
    page.set_viewport_size({"width": 960, "height": 900})
    settle(page)
    tight = page.evaluate(BAR)
    assert tight["hitShown"] is False, tight

    # Nothing was lost by dropping it: the receipt prints the same arithmetic,
    # line by line, one click away.
    page.click("#costToggle")
    receipt = page.evaluate("() => byId('costPop').textContent")
    for text in ("Uncached input", "Cache write · 5m TTL",
                 "Cache read (hit)", "Output", "Total"):
        assert text in receipt, receipt

    # The shapes the removed figures wore, at any abbreviation: "5,000 in",
    # "5k in", "1.5M out" - and a tool count, which is supposed to be one click
    # away as well.
    for text in (tight["text"], roomy["text"]):
        assert not re.search(r"\d[\d.,]*[kM]? (in|out)\b", text), text
        assert not re.search(r"\bsearch(es)?\b", text), text


def test_a_conversations_tool_calls_are_counted_in_the_details(
    page, key, fake_provider, response_scenario
):
    """A count that fits in "2 turns · 2 searches" at 1280 and does not at 960
    cannot be allowed in the bar at all, so it moved to the popover's header
    line - beside the other things the bar does not have room to say."""
    send(page, "search for it", 1, response_scenario, SEARCHING_TURN)
    bar = page.evaluate(BAR)
    assert bar["tokens"] == "1 turn", bar
    assert not re.search(r"\bsearch(es)?\b", bar["text"]), bar["text"]

    page.click("#costToggle")
    pop = page.evaluate("() => byId('costPop').textContent")
    assert "2 searches" in pop, pop
    # The rows behind the count are there too, with the price per search.
    assert "Web search" in pop and "$0.01 / search" in pop, pop


def test_a_turn_that_could_not_be_priced_is_named_not_dropped(
    page, key, fake_provider, response_scenario
):
    """A model the pricing page does not cover gets no estimate. The sum then
    covers only what it could, and has to say so - a total quietly smaller
    than the conversation is the failure this guards."""
    send(page, "unpriced", 1, response_scenario,
         {**PRICED_TURN, "model": "claude-imaginary-9"})
    send(page, "priced", 2, response_scenario)

    footer = page.evaluate(FOOTER)
    assert footer["perTurn"][0] is None
    assert footer["perTurn"][1] is not None
    # The sum is the priced turn alone, and the count in front of it says the
    # conversation is two turns long.
    assert footer["total"] == footer["sum"] != "$0.0000"
    assert footer["tokens"] == "2 turns", footer["tokens"]

    page.click("#costToggle")
    pop = page.evaluate("() => byId('costPop').textContent")
    assert "1 of 2 turns have no estimate" in pop
    assert "Priced turns" in pop and "1 of 2" in pop

    card = show_card(page, "#messages .msg.assistant")
    assert "no estimate for this turn" in card["text"]


def test_the_footer_says_why_there_is_no_estimate(
    page, key, fake_provider, response_scenario
):
    """"No estimate" has more than one reason, and the reason is what makes it
    honest: an unreachable pricing page is not the same as a model the page
    does not cover.

    The rates answer is read under the names the server sends it with. This
    check exists because the page read `state`/`error`, which matched nothing
    in `{rates_state, rates_error}` - so the first reason could never appear.
    """
    page.route(
        "**/api/rates",
        lambda route: route.fulfill(
            status=200,
            content_type="application/json",
            body=json.dumps({
                "rates_state": "unavailable",
                "rates_error": "the pricing page could not be read",
            }),
        ),
    )
    send(page, "hi", 1, response_scenario,
         {**PRICED_TURN, "model": "claude-imaginary-9"})
    # The page fetched the rates before the route existed.
    page.evaluate("() => loadRates()")
    page.wait_for_function(
        "() => byId('costTokens').textContent.startsWith('no rates:')"
    )
    assert page.evaluate("() => byId('costTokens').textContent") == (
        "no rates: the pricing page could not be read"
    )

    # At 1280 that sentence fits. At 960 it cannot - the bar is one line - and
    # this is the only place the reason is ever said, because the popover hides
    # itself when there is nothing to total. So the full sentence has to stay
    # reachable, and what is drawn has to be a cut of it rather than a
    # different, vaguer claim.
    page.set_viewport_size({"width": 960, "height": 900})
    settle(page)
    kept = page.evaluate(
        """() => {
            const el = byId('costTokens');
            return {
                title: el.title,
                text: el.textContent,
                cut: el.scrollWidth > el.clientWidth + 1,
            };
        }"""
    )
    assert kept["cut"] is True, kept
    assert kept["title"] == kept["text"], kept
    assert kept["title"].endswith("could not be read"), kept
