"""The settings live in the topbar as pills, over one hidden value store.

The pills are a view, never a second source of truth: every menu writes back
into the hidden form controls and fires their change events, so the payload,
the validation and the preview read exactly what they always read. A pill the
model or the runtime cannot honour is dashed and grey, not a fake control.

History shows its own settings three ways, all read from the turn's stored
effective_options: a hover card on each bubble, a dashed divider where the
settings changed between turns, and a cache hint that only fires for fields
that are actually part of the cache prefix.
"""

import re


def css_rule(page: str, selector: str) -> str:
    """The declarations of one CSS rule.

    A test about layout has to read the property rather than find the string
    somewhere in the file, because the defect the rules below guard against
    was a perfectly correct declaration (``position: absolute``) sitting in
    the wrong containing block.
    """
    match = re.search(rf"^{re.escape(selector)}\s*{{([^}}]*)}}", page, re.M)
    assert match, f"no CSS rule for {selector!r}"
    return match.group(1)


# --- the pills are a view over the hidden controls ---------------------------


def test_the_topbar_carries_the_pills_and_the_controls_stay(static_page):
    """The value store survives; the pills only mirror it."""
    assert 'id="settingsPills"' in static_page
    assert 'id="paramsPanel" hidden' in static_page
    # Every control the payload reads is still in the document.
    for control in ["model", "maxTokens", "system", "thinking", "effort",
                    "webSearch", "webSearchType", "allowedCallers",
                    "responseInclusion", "maxUses", "cacheControl"]:
        assert f'id="{control}"' in static_page


def test_the_collapsed_params_bar_is_gone(static_page):
    """There is no second settings surface left to drift from the pills."""
    assert "paramsToggle" not in static_page
    assert "paramsSummary" not in static_page
    assert "toggleParams" not in static_page


def test_a_menu_write_goes_through_the_control_and_its_change_event(static_page):
    """One write path: setControl sets the hidden control's value and fires
    the same change/input events a hand on the old form would have fired."""
    assert "function setControl(id, value)" in static_page
    assert "el.dispatchEvent(new Event('change'))" in static_page
    assert "el.dispatchEvent(new Event('input'))" in static_page


def test_every_pill_is_derived_from_controls_and_capabilities(static_page):
    """The eight blocks the user specified, in order."""
    flat = static_page.replace("\n", " ")
    for pill in ["'model'", "'thinking'", "'effort'", "'maxTokens'",
                 "'system'", "'webSearch'", "'cacheControl'", "'streaming'"]:
        assert f"id: {pill}," in flat


def test_streaming_is_a_fixed_pill_not_a_choice(static_page):
    """llm-anthropic always streams, so the pill shows On and opens nothing."""
    flat = static_page.replace("\n", " ")
    assert "id: 'streaming', label: 'Streaming'," in flat
    assert "fixed: () => true," in flat
    assert "fixedTag: () => 'runtime fixed'," in flat


# --- effort: no extra Default row, the documented level carries the tag ------


def test_the_effort_menu_has_no_default_row(static_page):
    """The user rejected a separate Default entry: the menu lists only the
    model's levels, low to max, and the documented default level carries the
    tag. Picking it stores 'default', so no effort field goes on the wire."""
    assert "control.options.filter((o) => o.value !== 'default')" in static_page
    assert "def: isDefault, selected, disabled: blocked," in static_page
    assert "setControl('effort', isDefault ? 'default' : option.value);" in static_page


def test_effort_pill_shows_the_effective_level_not_the_word_default(static_page):
    assert "cap1(state.caps.default_effort || 'default')" in static_page


def test_effort_is_unsupported_for_models_without_it(static_page):
    """Haiku has no effort parameter; the pill greys out, never fakes one."""
    assert "unsupported: () => state.caps && !state.caps.supports_effort," in static_page


# --- max tokens and the thinking budget floor --------------------------------


def test_max_tokens_menu_enforces_the_thinking_budget_floor(static_page):
    """With thinking enabled the minimum is the budget + 1 (1025 for Haiku);
    the budget itself is never shown as a setting - it is hard-coded."""
    assert "const min = control.min_by_thinking[thinking] || 1;" in static_page
    assert "value = Math.max(min, max ? Math.min(max, value) : value);" in static_page
    assert "the legal minimum is" in static_page


def test_budget_tokens_are_not_a_pill(static_page):
    """The runtime fixes the budget, so it appears nowhere as a control."""
    flat = static_page.replace("\n", " ")
    assert "id: 'budgetTokens'" not in flat
    assert "id: 'budget'" not in flat


# --- history: hover card, divider, cache hint --------------------------------


def test_bubbles_offer_the_turns_settings_card(static_page):
    assert "function wireSettingsCard(div, turnIndex)" in static_page
    assert "wireSettingsCard(user.parentElement, index);" in static_page
    assert "wireSettingsCard(assistant.parentElement, index);" in static_page


def test_a_turn_without_a_sidecar_row_says_not_recorded(static_page):
    """llm -c conversations have no options; the card must not invent them."""
    assert "settings were not recorded" in static_page


def test_the_card_footer_states_the_absence_rule(static_page):
    """Settled in docs/per-turn-settings.md §1b: absent means default."""
    assert "absent fields = provider default" in static_page


def test_a_divider_marks_turns_whose_settings_changed(static_page):
    assert "function settingsDiffLines(prev, cur)" in static_page
    assert "settings-divider" in static_page
    assert "settings changed" in static_page
    # History render and the live record event both insert it.
    assert "messages.appendChild(dividerFor(diff));" in static_page
    assert "messages.insertBefore(dividerFor(diff), userBody.parentElement);" in static_page


def test_the_cache_hint_only_fires_for_prefix_fields(static_page):
    """system + tools + messages is the prefix; max_tokens, effort and
    thinking are not part of it and must not warn."""
    flat = static_page.replace("\n", " ")
    assert "const CACHE_PREFIX_FIELDS = [" in flat
    for field in ["'model'", "'system'", "'web_search'", "'web_search_type'",
                  "'max_uses'", "'response_inclusion'"]:
        assert field in flat.split("const CACHE_PREFIX_FIELDS = [")[1].split("];")[0]
    for field in ["'max_tokens'", "'effort'", "'thinking'"]:
        assert field not in flat.split("const CACHE_PREFIX_FIELDS = [")[1].split("];")[0]


def test_cache_off_says_wont_read_or_write_not_invalidated(static_page):
    assert "won't read or" in static_page
    assert "write the cache" in static_page


# --- model display names are formatting, not facts ----------------------------


def test_short_model_names_are_parsed_not_looked_up(static_page):
    """No model id may be hard-coded in the UI (the capability-matrix test
    enforces this); the short name is derived from the id's own shape."""
    assert "function shortModelName(id)" in static_page
    assert "cannot parse is shown as it is" in static_page


# --- a menu has to be able to appear -----------------------------------------
#
# "Clicking a pill does nothing" survived three repairs because each of them
# searched the js, and the js was right: the handler ran, the menu was built,
# inserted, and reported display:block, visibility:visible, z-index 70. It was
# inserted inside the pill - and the pill row is a scroller, where `overflow-x:
# auto` computes `overflow-y` to `auto` as well, so a 28px-tall row clipped a
# menu beginning 4px below it. The menu was in the DOM and painted nowhere.
# These are the invariants that failure turned into. None of them can be seen
# by reading the js, which is exactly why it was not found there.


def test_the_pill_row_is_a_scroller_and_therefore_clips(static_page):
    """The premise the rest of this section rests on."""
    assert "overflow-x: auto" in css_rule(static_page, "#settingsPills")


def test_a_menu_is_mounted_in_the_overlay_layer_not_inside_its_pill(static_page):
    assert 'id="pillLayer"' in static_page
    assert "byId('pillLayer').appendChild(menu)" in static_page
    # The one line that made every menu invisible.
    assert "pill.appendChild(menu)" not in static_page


def test_a_menu_is_placed_like_the_status_card_not_like_a_child(static_page):
    """#whyTip is already fixed-position for this exact reason, and says so
    in its own comment; the menu it sits beside was not."""
    rule = css_rule(static_page, ".pill-menu")
    assert "position: fixed" in rule
    assert "position: absolute" not in rule
    assert "position: fixed" in css_rule(static_page, "#pillLayer")


def test_the_overlay_layer_cannot_swallow_the_page_behind_it(static_page):
    """It covers the viewport, so only the menu inside it may be a target."""
    assert "pointer-events: none" in css_rule(static_page, "#pillLayer")
    assert "pointer-events: auto" in css_rule(static_page, ".pill-menu")


def test_one_routine_places_every_overlay_the_topbar_owns(static_page):
    """Two copies of the same flip-and-clamp arithmetic is how one of them
    gets fixed and the other does not."""
    assert "function anchorOverlay(card, anchor," in static_page
    assert "anchorOverlay(card, el, { nudge: -12 })" in static_page
    assert "anchorOverlay(pillMenu.el, pillMenu.pill, { align: 'right' })" in static_page


def test_a_menu_field_is_mounted_before_it_is_built(static_page):
    """focus() on a field in a detached node does nothing, which is why the
    Max tokens menu opened with no caret and typing appended to the old
    value (16384 + "4096" = "163844096") instead of replacing it."""
    flat = static_page.replace("\n", " ")
    mount = flat.index("byId('pillLayer').appendChild(menu)")
    assert mount < flat.index("build(menu);", mount - 400)
    assert "input.focus({ preventScroll: true })" in static_page


# --- a press has to survive the render it sets off ----------------------------
#
# The second half of the same defect, and the one met most often: pressing a
# pill blurs an open menu's field, the field commits, the commit re-renders the
# row - and the node the press landed on is detached before the button comes
# back up. The browser then dispatches no click at all, so the pill that was
# aimed at does nothing and the one the user had open just closes.


def test_the_pill_row_is_updated_in_place_never_rebuilt(static_page):
    assert "const pillNodes = new Map();" in static_page
    assert "if (pill.parentElement !== host) host.appendChild(pill);" in static_page
    # Emptying the row is what replaced the nodes mid-gesture.
    assert "host.innerHTML = ''" not in static_page


def test_only_the_pill_itself_is_a_pointer_target(static_page):
    """A pill's contents are rewritten on every render, so a press that
    landed on the label would still be holding a detached node."""
    assert "pointer-events: none" in css_rule(static_page, ".pill > *")


def test_a_pill_reads_its_own_state_at_click_time(static_page):
    """One listener per pill for the life of the page: a pill gains and
    loses its menu as the model changes, so the handler cannot close over
    which kind it was when it was built."""
    assert "if (pill.dataset.menu === 'yes')" in static_page
    assert "pill.dataset.menu = fixed || unsupported ? 'no' : 'yes';" in static_page


def test_an_open_menu_survives_a_render_and_follows_its_pill(static_page):
    """Max uses lives inside the Web Search menu, so writing it re-renders
    the row: the menu the user is standing in must not vanish, and must not
    be rebuilt either - that would wipe the field being typed into."""
    assert "if (pillMenu.pill.dataset.menu !== 'yes') closePillMenu();" in static_page
    assert "else anchorPillMenu();" in static_page
    assert "window.addEventListener('resize', anchorPillMenu);" in static_page
