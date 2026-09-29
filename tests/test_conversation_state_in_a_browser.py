"""What belongs to a conversation: its settings, its draft, its warning.

Three things the user lost by switching conversations, measured in a browser
because none of them is a matter of wording:

- the settings a conversation ended with, which the topbar used to leave at
  the form's defaults;
- the draft typed into it, which used to be gone on return;
- the "this differs from your last turn" warning, which used to be silent
  for every field that is not part of the cached prefix.

No key and no network: the turn the assertions compare against is one the
test supplies, never a live one.
"""

import pytest

pytestmark = pytest.mark.browser


def test_a_conversation_reopens_with_its_own_settings(page):
    """Reopening a conversation must not quietly change the request.

    The old code restored one field - `model`, and from the *first* turn.
    Everything else stayed at the selected model's defaults.
    """
    restored = page.evaluate(
        """async () => {
            // Values this model really offers, so restoring them is legal.
            const pick = (id) => {
                const options = Array.from(byId(id).options).map((o) => o.value);
                return options.find((v) => v !== byId(id).value) || options[0];
            };
            const stored = {
                system: 'Answer in one sentence.',
                max_tokens: 4096,
                web_search: false,
                thinking: pick('thinking'),
                effort: pick('effort'),
            };
            await loadForm();                 // model defaults first
            applyStoredOptions(stored);       // then the conversation's own
            return {
                system: byId('system').value,
                maxTokens: byId('maxTokens').value,
                webSearch: byId('webSearch').value,
                thinking: byId('thinking').value,
                effort: byId('effort').value,
            };
        }"""
    )
    assert restored["system"] == "Answer in one sentence."
    assert restored["maxTokens"] == "4096"
    assert restored["webSearch"] == "false"
    assert restored["thinking"]
    assert restored["effort"]


def test_a_value_the_model_no_longer_offers_is_not_written(page):
    """Honest restore: a stored option this model cannot send keeps the
    model's own default instead of showing a request it cannot make."""
    outcome = page.evaluate(
        """async () => {
            await loadForm();
            const before = byId('thinking').value;
            applyStoredOptions({ thinking: 'a-mode-this-model-does-not-have' });
            return { before, after: byId('thinking').value };
        }"""
    )
    assert outcome["after"] == outcome["before"]


def test_a_draft_stays_with_the_conversation_it_was_typed_into(page):
    """Leaving a conversation and coming back must not cost what was typed."""
    round_trip = page.evaluate(
        """() => {
            byId('prompt').value = 'draft for the first conversation';
            state.conversationId = 'conv-a';
            saveDraft();

            state.conversationId = 'conv-b';
            restoreDraft();
            const elsewhere = byId('prompt').value;

            byId('prompt').value = 'draft for the second';
            saveDraft();

            state.conversationId = 'conv-a';
            restoreDraft();
            const back = byId('prompt').value;
            return { elsewhere, back };
        }"""
    )
    assert round_trip["elsewhere"] == ""
    assert round_trip["back"] == "draft for the first conversation"


def test_sending_clears_the_draft_it_came_from(page):
    """A sent message is not a draft still waiting in the composer."""
    assert page.evaluate(
        """() => {
            state.conversationId = 'conv-a';
            byId('prompt').value = 'about to be sent';
            saveDraft();
            clearDraft();
            byId('prompt').value = '';
            restoreDraft();
            return byId('prompt').value === '' && !state.drafts['conv-a'];
        }"""
    )


def test_a_field_outside_the_cache_prefix_still_gets_reported(page):
    """max output tokens cannot cost a cache hit, so it must not be reported
    as one - but the request does differ from the last turn, and silence
    about that was the complaint."""
    warned = page.evaluate(
        """async () => {
            await loadForm();
            state.turns = [{
                options: currentFormOptions(),
            }];
            byId('maxTokens').value = '4096';
            byId('maxTokens').dispatchEvent(new Event('change'));
            const el = byId('cacheWarn');
            return {
                hidden: el.hidden,
                text: byId('cacheWarnText').textContent,
            };
        }"""
    )
    assert warned["hidden"] is False
    assert "max output tokens" in warned["text"]
    # Honest about the cause: not a prefix field, so not a cache miss.
    assert "won't reuse the cached prefix" not in warned["text"]


def test_a_prefix_field_is_reported_as_what_it_costs(page):
    """The model is part of the prefix, so changing it says so."""
    warned = page.evaluate(
        """async () => {
            await loadForm();
            state.turns = [{ options: currentFormOptions() }];
            byId('system').value = 'a different system prompt';
            byId('system').dispatchEvent(new Event('change'));
            return {
                hidden: byId('cacheWarn').hidden,
                text: byId('cacheWarnText').textContent,
            };
        }"""
    )
    assert warned["hidden"] is False
    assert "system" in warned["text"]
    assert "won't reuse the cached prefix" in warned["text"]


def test_a_model_the_list_no_longer_offers_can_still_be_selected(page):
    """The list narrowed to one model per series, and a stored conversation
    can point at a member that has since been superseded.

    "Superseded" is not "unusable": that turn was sent with it, and applying
    another model's capabilities to it would misreport what happens next. So
    the id must come back selectable and marked, and the form for it must
    load.
    """
    outcome = page.evaluate(
        """async () => {
            const unlisted = 'claude-sonnet-5-5';
            const before = Array.from(byId('model').options).map((o) => o.value);
            const added = ensureModelOption(unlisted);
            byId('model').value = unlisted;
            await loadForm();
            const option = byId('model').selectedOptions[0];
            return {
                wasOffered: before.includes(unlisted),
                added: added,
                value: byId('model').value,
                marked: option.dataset.superseded === 'true',
                count: before.length,
                after: Array.from(byId('model').options).length,
                loaded: state.data.model.id,
                blocked: !byId('send').disabled,
                pill: PILL_DEFS.find((p) => p.id === 'model').value(),
            };
        }"""
    )
    assert outcome["wasOffered"] is False, "the fixture has to start without it"
    assert outcome["added"] is True
    assert outcome["value"] == "claude-sonnet-5-5", "the stored model must win"
    assert outcome["marked"] is True, "it is legacy, not a current model"
    assert outcome["after"] == outcome["count"] + 1
    # The form for it really loaded: what the right pane shows must be the
    # model the conversation used, not whichever one superseded it.
    assert outcome["loaded"] == "claude-sonnet-5-5"
    assert outcome["blocked"] is True, "a legacy model is still sendable"
    assert "legacy" in outcome["pill"], "the pill has to say it is legacy"


def test_adding_a_model_twice_does_not_grow_the_list(page):
    """Opening the same old conversation repeatedly must not keep stacking a
    duplicate option onto the select."""
    outcome = page.evaluate(
        """() => {
            const id = 'claude-opus-4-6';
            const first = ensureModelOption(id);
            const second = ensureModelOption(id);
            const count = Array.from(byId('model').options)
                .filter((o) => o.value === id).length;
            return { first, second, count };
        }"""
    )
    assert outcome["first"] is True
    assert outcome["second"] is False
    assert outcome["count"] == 1


def test_matching_the_last_turn_keeps_the_warning_away(page):
    """No difference, no warning: the same settings are not news."""
    assert page.evaluate(
        """async () => {
            await loadForm();
            state.turns = [{ options: currentFormOptions() }];
            updateCacheWarn();
            return byId('cacheWarn').hidden;
        }"""
    ) is True


def test_the_composer_holds_more_than_three_lines(page):
    """Three lines used to be enough to start scrolling."""
    height = page.evaluate(
        "() => parseFloat(getComputedStyle(byId('prompt')).minHeight)"
    )
    assert height >= 120, f"the composer is back to {height}px"
