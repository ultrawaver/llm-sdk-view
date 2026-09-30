"""A conversation owns its settings, its draft, and its warning.

The browser file next to this one measures the behaviour end to end; these
assertions are the cheap guard on the wiring, so a refactor cannot quietly
stop calling the pieces. They name stable names, not wording.
"""


def test_switching_conversations_moves_the_draft_with_it(static_page):
    """Typed text belongs to the conversation it was typed into, so leaving
    saves it and returning restores it. A new conversation is keyed apart
    from `null`, which is also "no conversation yet"."""
    flat = " ".join(static_page.split())

    assert "drafts: {}" in flat
    assert "const NEW_DRAFT_KEY = '__new__'" in flat
    assert "function saveDraft()" in flat
    assert "function restoreDraft()" in flat
    assert "function clearDraft()" in flat
    # Wired on both exits: opening a conversation and starting a new one.
    assert "async function openConversation(id) { saveDraft();" in flat
    assert "async function startNewConversation() { saveDraft();" in flat
    # Sent, so no longer waiting in the composer.
    assert "// Sent, so it is no longer a draft waiting in this conversation." in flat


def test_a_reopened_conversation_continues_its_own_settings(static_page):
    """The topbar used to restore `model` only, and from the first turn; the
    request underneath a reopened conversation was therefore not the one it
    was built with."""
    flat = " ".join(static_page.split())

    assert "function applyStoredOptions(options)" in flat
    # Read off the last turn, and written after the model's own defaults
    # have been put on the form: a value is only restorable while the model
    # still offers it.
    assert "const lastOptions = (lastTurn && lastTurn.options) || {};" in flat
    # Written after the model's own defaults have been put on the form, and
    # the model first: it decides which defaults and capabilities apply()
    # writes. A value is only restorable while the model still offers it.
    assert "await loadForm(); applyStoredOptions(lastOptions);" not in flat
    assert "const wantedModel = lastOptions.model;" in flat
    assert "await loadForm();" in flat
    # A stored value this model cannot send is not written. The rule itself
    # is measured in a browser by
    # test_a_value_the_model_no_longer_offers_is_not_written; only the seam
    # is named here, because the sentence it used to quote moved into the
    # helper when restoring was split per provider.
    assert "function restoreChoice(id, value)" in flat
    # Each provider restores its own fields. Both sets used to be written,
    # which ended in Anthropic's state functions reading controls no
    # OpenRouter schema carries - and the throw took the transcript with it.
    assert "if (providerId() === 'openrouter') restoreOpenRouterOptions(options);" in flat
    assert "else restoreAnthropicOptions(options);" in flat
    # The list narrows to one model per series, so a conversation older than
    # the release now points outside it: that id is added back, marked, not
    # silently swapped for whatever superseded it.
    assert "function ensureModelOption(id)" in flat
    assert "ensureModelOption(wantedModel);" in flat
    assert "' · legacy'" in flat
    # A brand-new conversation is the one case that starts from defaults.
    assert "await loadForm(); restoreDraft();" in flat


def test_a_difference_from_the_last_turn_is_always_named(static_page):
    """Two classes, because only one of them can cost a cache hit: prefix
    fields invalidate the cached prefix, the rest merely differ. Silence
    about the second kind was the complaint; calling it a cache miss would
    be a lie."""
    flat = " ".join(static_page.split())

    assert "const CACHE_PREFIX_FIELDS = [" in flat
    assert "const OTHER_REQUEST_FIELDS = [" in flat
    for field in ("thinking", "effort", "max_tokens", "allowed_callers"):
        assert f"['{field}'," in flat
    # The two reports are worded differently on purpose.
    assert "differ from the last turn" in flat
    assert "reuse the cached prefix" in flat


def test_the_composer_does_not_scroll_at_three_lines(static_page):
    flat = " ".join(static_page.split())

    assert "#prompt { min-height: 132px; }" in flat
