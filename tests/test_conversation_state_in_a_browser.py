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

    The effort level is asserted by value, never by truthiness. This test
    used to ask only whether the control held *something*, and `'default'`
    is something - which is how a restore that was wiped dry passed as a
    restore that worked.
    """
    outcome = page.evaluate(
        """async () => {
            // A model that has effort levels to restore at all: the page
            // starts on Haiku, whose only level is 'default', so the old
            // version of this test had nothing to lose in the first place.
            byId('model').value = 'claude-fable-5-1';
            await loadForm();
            const stored = {
                system: 'Answer in one sentence.',
                max_tokens: 4096,
                web_search: false,
                thinking: 'on',
                effort: 'xhigh',
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
    assert outcome["system"] == "Answer in one sentence."
    assert outcome["maxTokens"] == "4096"
    assert outcome["webSearch"] == "false"
    assert outcome["thinking"] == "on"
    assert outcome["effort"] == "xhigh", "the stored level, not 'default'"


def test_reopening_a_conversation_does_not_report_a_change_nobody_made(page):
    """The complaint, measured: switching back to an old conversation showed
    "effort differ from the last turn" for a change the user never made.

    `applyStoredOptions()` ended by rebuilding the effort list, and replacing
    a select's options clears its selection - so the level it had just
    restored was wiped to "default" and the composer then reported that
    difference as if the user had caused it.
    """
    outcome = page.evaluate(
        """async () => {
            byId('model').value = 'claude-fable-5-1';
            await loadForm();
            const stored = Object.assign(currentFormOptions(), {
                system: 'the system prompt this conversation used',
                max_tokens: 64000,
                effort: 'xhigh',
            });
            // What openConversation() does, in its own order.
            state.turns = [{ options: stored }];
            await loadForm();
            applyStoredOptions(stored);
            updateCacheWarn();
            return {
                effort: byId('effort').value,
                system: byId('system').value,
                maxTokens: byId('maxTokens').value,
                warned: !byId('cacheWarn').hidden,
                warnText: byId('cacheWarnText').textContent,
            };
        }"""
    )
    assert outcome["effort"] == "xhigh"
    assert outcome["system"] == "the system prompt this conversation used"
    assert outcome["maxTokens"] == "64000"
    assert outcome["warned"] is False, outcome["warnText"]
    assert outcome["warnText"] == "", "a withdrawn warning must not linger"


def test_a_rebuilt_effort_list_keeps_a_level_it_still_offers(page):
    """A rebuild is not a reset.

    The effort list is rebuilt from the model's capabilities whenever the
    form reloads or the thinking mode changes, and a rebuild clears the
    selection. A level the new list still offers belongs to the user; only
    one that cannot be sent falls back to the provider's own level.
    """
    outcome = page.evaluate(
        """async () => {
            byId('model').value = 'claude-fable-5-1';
            await loadForm();
            const set = (value) => { byId('effort').value = value; };

            set('max');
            applyEffortState();                 // rebuild, nothing asked
            const kept = byId('effort').value;

            // This model rejects every explicit level with thinking off, so
            // none of them may survive it.
            byId('thinking').value = 'off';
            applyEffortState();
            const blocked = byId('effort').value;

            byId('thinking').value = 'on';
            set('high');
            applyEffortState();
            const legal = byId('effort').value;

            set('default');
            applyEffortState('xhigh');          // asked for, and offered
            const asked = byId('effort').value;

            // A model with no effort parameter at all must not be handed a
            // level it cannot send.
            byId('model').value = 'claude-haiku-4-5-20251001';
            await loadForm();
            applyEffortState('xhigh');
            return {
                kept, blocked, legal, asked,
                unsupported: byId('effort').value,
                disabled: byId('effort').disabled,
            };
        }"""
    )
    assert outcome["kept"] == "max", "a still-offered level is the user's"
    assert outcome["blocked"] == "default", "an unsendable level is refused"
    assert outcome["legal"] == "high"
    assert outcome["asked"] == "xhigh"
    assert outcome["unsupported"] == "default"
    assert outcome["disabled"] is True, "Haiku has no effort parameter"


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


def test_a_new_conversation_still_offers_the_system_prompt(page):
    """No turns yet: the runtime honours system=, so the menu stays open for
    editing and keeps the cache-prefix wording it always had."""
    outcome = page.evaluate(
        """async () => {
            await loadForm();
            state.turns = [];
            const menu = document.createElement('div');
            PILL_DEFS.find((p) => p.id === 'system').menu(menu);
            const input = menu.querySelector('input');
            return {
                disabled: input.disabled,
                hint: menu.querySelector('.hint').textContent,
            };
        }"""
    )
    assert outcome["disabled"] is False
    assert "cache prefix" in outcome["hint"]


def test_a_conversation_under_way_locks_the_system_prompt(page):
    """From turn 2 on, llm carries the first turn's system forward and reads
    a later system= from nowhere - the request the user thought they changed
    is not the request that gets built. So the menu shows the conversation's
    own value, disabled, with the reason the schema words; the verify refusal
    that used to be the only answer is the backstop, not the UX."""
    outcome = page.evaluate(
        """async () => {
            await loadForm();
            const stored = Object.assign(currentFormOptions(), {
                system: 'the system prompt this conversation started with',
            });
            state.turns = [{ options: stored }];
            applyStoredOptions(stored);
            const menu = document.createElement('div');
            PILL_DEFS.find((p) => p.id === 'system').menu(menu);
            const input = menu.querySelector('input');
            return {
                disabled: input.disabled,
                value: input.value,
                hint: menu.querySelector('.hint').textContent,
            };
        }"""
    )
    assert outcome["disabled"] is True, (
        "a change the runtime would silently ignore must not be offerable"
    )
    assert outcome["value"] == "the system prompt this conversation started with"
    assert "first turn" in outcome["hint"]
    assert "ignored" in outcome["hint"]


def test_the_composer_holds_more_than_three_lines(page):
    """Three lines used to be enough to start scrolling."""
    height = page.evaluate(
        "() => parseFloat(getComputedStyle(byId('prompt')).minHeight)"
    )
    assert height >= 120, f"the composer is back to {height}px"
