"""Where a cost figure belongs: the turn that produced it, or the conversation.

The two scopes used to be one. The footer described whichever turn was
selected, so the conversation's own total existed only as a row inside a
popover, and both bubbles hovered up the *same* settings card - which made the
answer look like a second copy of the request and left the answer's own
figures with nowhere to live.

The invariants below are the split. They are string assertions, and a string
assertion protects the wording, not the behaviour: what must actually be
visible is measured in test_cost_scopes_in_a_browser.py.
"""


def test_the_footer_totals_the_conversation(static_page):
    assert "function conversationTotals()" in static_page
    assert "const totals = conversationTotals();" in static_page
    # The selection is no longer what the footer reads: clicking a bubble
    # changes the right pane, never the total under the composer.
    assert "selectedCost" not in static_page
    assert "sessionCost" not in static_page


def test_both_scopes_draw_the_same_receipt(static_page):
    """One definition of a line, so the turn's receipt and the conversation's
    cannot disagree about what a cache read cost."""
    assert "function costReceiptHtml(cost, { note = false } = {})" in static_page
    assert "costReceiptHtml(totals, { note: true })" in static_page
    assert "costReceiptHtml(cost)" in static_page


def test_the_total_is_the_sum_of_the_rows_above_it(static_page):
    """A receipt whose rows do not add up to its total is not a receipt."""
    assert (
        "const total = round6(sums.reduce((sum, line) => sum + line.amount, 0));"
        in static_page
    )


def test_an_unpriced_turn_is_named_rather_than_dropped(static_page):
    """The sum covers the turns that could be priced. The ones that could not
    must be counted out loud - a total quietly smaller than the conversation
    is the failure mode this exists for."""
    assert "have no estimate and are not included above" in static_page
    assert "Priced turns" in static_page


def test_only_the_answer_bubble_carries_the_turns_receipt(static_page):
    assert "function costCardHtml(record, index)" in static_page
    assert "wireTurnCostCard(assistant.parentElement, index);" in static_page
    assert "wireTurnCostCard(body.parentElement, state.selected);" in static_page


def test_the_countdown_sentence_has_exactly_one_home(static_page):
    """The popover is where the long "send now" sentence is written. A second
    element with the same id would be a second writer to it."""
    assert static_page.count('id="cacheNote"') == 1


def test_the_hover_card_is_built_when_it_is_read(static_page):
    """A card built at wiring time would describe the record that existed
    then; the record is replaced by every later turn."""
    assert "function hoverCard(div, build, extraClass)" in static_page
    assert "const html = build();" in static_page
    assert "card.className = 'scard' + (extraClass ? ' ' + extraClass : '');" in static_page
