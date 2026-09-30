"""Switching provider, measured in a browser instead of read.

The form is no longer one provider's. ``apply()`` used to reach straight into
``data.controls.cache_control.ttl`` and ``data.controls.thinking``, and an
OpenRouter schema carries neither - so the page threw halfway through
applying it and then sat there looking applied. That is the whole class of
defect this file is aimed at, and none of it is visible in the source: the
exception is asynchronous, nothing is logged where a test can read it, and
the controls that were written before the throw are correct.

So the questions here are the ones only a browser can answer. Did the page
raise while switching. Is the pill the selected provider owns actually
painted at its own rectangle. Is the pill the other provider owns gone from
the row rather than merely styled away. Does a pick reach the handler and
change what the next request would carry.

No key and no network: ``llm-openrouter`` registers models from a seeded
model list and a fake key, the catalogue is this project's own disk cache in
a throwaway directory, and the browser only ever loads localhost.
"""

import pytest

pytestmark = pytest.mark.browser

#: The pills each provider owns, and the heading the menu carries. Web Search
#: is in both lists under two ids: the tool is the same idea, the fields are
#: not, so each provider has its own pill for it.
ANTHROPIC_PILLS = {
    "thinking": "Thinking",
    "webSearch": "Web search",
    "cacheControl": "Prompt caching",
}
OPENROUTER_PILLS = {
    "reasoning": "Reasoning",
    "orWebSearch": "Web search",
    "transport": "OpenRouter API",
    "routing": "Provider routing",
}
#: Neither provider's: the three controls every form has, plus the one both
#: plugins fix.
SHARED_PILLS = ("provider", "model", "maxTokens", "system", "streaming")
#: Of those, the ones that open a menu. Streaming is runtime-fixed on both.
SHARED_MENUS = {
    "provider": "Provider",
    "model": "Model",
    "maxTokens": "Max output tokens",
    "system": "System prompt",
}


@pytest.fixture
def sending(openrouter_registry, fake_openrouter, browser, live_app):
    """The same page, with the OpenRouter transport scripted.

    Real turns, stored in the real sidecar, without a key or a network: only
    the OpenAI client is replaced, so ``llm-openrouter``'s own ``execute``
    still builds and reads the request.
    """
    openrouter_registry("openai/gpt-5.4")
    fake_openrouter.answers("Hi there.")
    context = browser.new_context(viewport={"width": 1500, "height": 820})
    page = context.new_page()
    errors: list[str] = []
    page.on("pageerror", lambda error: errors.append(str(error)))
    page.errors = errors
    page.goto(live_app)
    page.wait_for_selector('#settingsPills [data-pill="model"]')
    page.wait_for_function("() => (state.data.provider.options || []).length > 1")
    try:
        yield page
    finally:
        context.close()


@pytest.fixture
def switchable(openrouter_registry, browser, live_app):
    """The live page with both providers usable.

    It opens its own page rather than using the suite's ``page`` fixture,
    because the models have to be registered before the page loads: the form
    is fetched while the page is loading, and ``openrouter_registry`` is a
    factory whose ``install`` runs in the test body - by which time the
    suite's page has already been told OpenRouter has nothing to offer.
    """
    openrouter_registry("openai/gpt-5.4", "anthropic/claude-sonnet-5")
    context = browser.new_context(viewport={"width": 1500, "height": 820})
    page = context.new_page()
    # Collected rather than asserted here: several tests below want to say
    # "and it did not throw while doing that", and an uncaught error in an
    # async handler reaches no assertion on its own. Attached before the
    # navigation, because applying the first schema is already a chance to
    # throw.
    errors: list[str] = []
    page.on("pageerror", lambda error: errors.append(str(error)))
    page.errors = errors
    page.goto(live_app)
    page.wait_for_selector('#settingsPills [data-pill="model"]')
    page.wait_for_function("() => (state.data.provider.options || []).length > 1")
    try:
        yield page
    finally:
        context.close()


def painted_at_its_own_rect(page, selector: str) -> bool:
    """Whether a hit test inside the element's box actually finds it."""
    return page.evaluate(
        """(selector) => {
            const el = document.querySelector(selector);
            if (!el) return false;
            const rect = el.getBoundingClientRect();
            if (rect.width === 0 || rect.height === 0) return false;
            const hit = document.elementFromPoint(
                rect.x + rect.width / 2, rect.y + Math.min(10, rect.height / 2));
            return !!hit && el.contains(hit);
        }""",
        selector,
    )


def pills_on_screen(page) -> list:
    """The pills a user can actually see, in the order they are painted."""
    return page.evaluate(
        """() => Array.from(document.querySelectorAll('#settingsPills [data-pill]'))
            .filter((pill) => pill.getBoundingClientRect().width > 0)
            .map((pill) => pill.dataset.pill)"""
    )


def switch_to(page, provider: str) -> None:
    """Pick a provider through the menu, the way a user does, and wait.

    The pick sets off a fetch, so the assertion has to wait for the schema
    rather than for the click: reading the row straight after the click reads
    the provider the page is still on.
    """
    page.click('[data-pill="provider"]')
    page.locator("#pillLayer .mi", has_text=LABELS[provider]).first.click()
    # `state` and not `window.state`: a top-level const in a classic script is
    # a global binding without being a property of window, so the window form
    # is undefined forever and the wait only ever times out.
    page.wait_for_function(
        "(id) => state.data && state.data.provider.id === id", arg=provider
    )
    # And for the render the schema sets off, which is one frame later.
    page.wait_for_timeout(50)


LABELS = {"anthropic": "Anthropic", "openrouter": "OpenRouter"}


# --- the switch happens at all ------------------------------------------------


def test_the_provider_pill_offers_both_and_the_menu_is_visible(switchable):
    page = switchable
    page.click('[data-pill="provider"]')

    assert painted_at_its_own_rect(page, "#pillLayer .pill-menu")
    items = page.locator("#pillLayer .mi").all_text_contents()
    assert any("Anthropic" in item for item in items)
    assert any("OpenRouter" in item for item in items)
    assert page.errors == []


def test_picking_openrouter_loads_openrouters_form_without_throwing(switchable):
    """The defect: the page threw partway through applying a schema whose
    controls it assumed, and what had already been written looked right."""
    page = switchable
    switch_to(page, "openrouter")

    assert page.errors == []
    assert page.evaluate("() => state.data.provider.sdk") == "openai-python"
    assert page.input_value("#model").startswith("openrouter/")


def test_the_model_list_holds_only_the_selected_providers_models(switchable):
    """A list carrying both providers' models offers a model the loaded
    controls do not belong to."""
    page = switchable
    before = page.evaluate(
        "() => Array.from(document.getElementById('model').options)"
        ".map((option) => option.value)"
    )
    assert before and all(not id.startswith("openrouter/") for id in before)

    switch_to(page, "openrouter")
    after = page.evaluate(
        "() => Array.from(document.getElementById('model').options)"
        ".map((option) => option.value)"
    )

    assert after and all(id.startswith("openrouter/") for id in after)


def test_switching_back_restores_anthropics_form(switchable):
    """Both directions, because the pills are kept across renders rather than
    rebuilt: a pill hidden on the way out has to come back."""
    page = switchable
    switch_to(page, "openrouter")
    switch_to(page, "anthropic")

    assert page.errors == []
    assert page.evaluate("() => state.data.provider.id") == "anthropic"
    assert page.input_value("#model").startswith("claude-")
    assert set(ANTHROPIC_PILLS) <= set(pills_on_screen(page))


# --- and the row is the selected provider's -----------------------------------


def test_only_the_selected_providers_pills_are_on_screen(switchable):
    page = switchable
    on_anthropic = pills_on_screen(page)
    assert set(ANTHROPIC_PILLS) <= set(on_anthropic)
    assert set(OPENROUTER_PILLS).isdisjoint(on_anthropic)

    switch_to(page, "openrouter")
    on_openrouter = pills_on_screen(page)

    assert set(OPENROUTER_PILLS) <= set(on_openrouter)
    assert set(ANTHROPIC_PILLS).isdisjoint(on_openrouter)
    assert set(SHARED_PILLS) <= set(on_openrouter)


def test_a_hidden_pill_keeps_its_node_so_it_can_come_back(switchable):
    """Removing it would be the pointer-gesture defect again: the pills are
    updated in place precisely so a press survives the render it sets off."""
    page = switchable
    page.evaluate(
        """() => { window.__watched =
             document.querySelector('[data-pill="cacheControl"]'); }"""
    )
    switch_to(page, "openrouter")

    assert page.evaluate("() => window.__watched.hidden") is True
    switch_to(page, "anthropic")
    assert page.evaluate(
        """() => window.__watched
             === document.querySelector('[data-pill="cacheControl"]')
             && !window.__watched.hidden"""
    )


@pytest.mark.parametrize(
    "width,height", [(1500, 820), (900, 700)], ids=["wide", "narrow"]
)
def test_every_pill_on_an_openrouter_model_opens_a_visible_menu(
    switchable, width, height
):
    """The shared pills are in here with OpenRouter's own, and they are the
    ones that quietly assume Anthropic: Max tokens read `min_by_thinking`,
    which only Anthropic's control carries, so opening it threw and the menu
    never appeared. The narrow case is when the row really does scroll.
    """
    page = switchable
    page.set_viewport_size({"width": width, "height": height})
    switch_to(page, "openrouter")

    for pill, heading in {**SHARED_MENUS, **OPENROUTER_PILLS}.items():
        page.click(f'[data-pill="{pill}"]')
        menu = page.locator("#pillLayer .pill-menu")
        assert menu.count() == 1, f"{pill} opened no menu at {width}x{height}"
        assert page.text_content("#pillLayer .pill-menu .mhead").startswith(heading)
        assert painted_at_its_own_rect(page, "#pillLayer .pill-menu"), (
            f"{pill}'s menu is in the DOM but painted nowhere at {width}x{height}"
        )
        box = menu.bounding_box()
        assert box["x"] >= 0 and box["x"] + box["width"] <= width
        page.keyboard.press("Escape")
    assert page.errors == []


def test_no_pill_is_stranded_off_the_edge_after_a_switch(switchable):
    """A hidden pill that still takes part in the layout pushes the row past
    the start edge, where the overflow is not scrollable."""
    page = switchable
    page.set_viewport_size({"width": 900, "height": 700})
    switch_to(page, "openrouter")

    stranded = page.evaluate(
        """() => {
            const host = document.getElementById('settingsPills');
            host.scrollLeft = 0;
            const reach = host.scrollWidth - host.clientWidth;
            const left = host.getBoundingClientRect().left;
            return Array.from(host.children)
                .filter((pill) => !pill.hidden)
                .filter((pill) => pill.getBoundingClientRect().left < left - reach - 1)
                .map((pill) => pill.dataset.pill);
        }"""
    )

    assert stranded == []


# --- a pick has to change what the next request would carry -------------------


def test_the_transport_pill_writes_the_field_that_picks_the_api(switchable):
    """The two transports are the one OpenRouter choice with no Anthropic
    analogue, and it is chosen per request rather than per model."""
    page = switchable
    switch_to(page, "openrouter")
    page.click('[data-pill="transport"]')
    page.locator("#pillLayer .mi", has_text="chat.completions.create").first.click()

    assert page.input_value("#chatCompletions") == "true"
    assert "chat.completions" in page.text_content('[data-pill="transport"]')


def test_a_reasoning_field_commits_to_its_own_hidden_control(switchable):
    """Four API fields, four controls. A menu field that kept its own value
    would be a second source for what the request carries."""
    page = switchable
    switch_to(page, "openrouter")
    page.click('[data-pill="reasoning"]')
    page.select_option("#menuEffort", "high")

    assert page.input_value("#reasoningEffort") == "high"
    assert "high" in page.text_content('[data-pill="reasoning"]')
    assert page.errors == []


def test_the_preview_follows_the_switch_to_the_other_sdk(switchable):
    """The right pane is the product's own claim, so it has to change hands
    with the form rather than keep rendering the previous provider's call."""
    page = switchable
    page.fill("#prompt", "hello")
    page.wait_for_function(
        "() => document.getElementById('code').textContent.includes('anthropic')"
    )

    switch_to(page, "openrouter")
    page.wait_for_function(
        "() => document.getElementById('code').textContent.includes('openrouter.ai')"
    )
    code = page.text_content("#code")

    assert "client.messages.stream" not in code
    assert "OPENROUTER_KEY" in code
    assert page.errors == []


# --- and nothing may claim a control the selected provider does not have ------


def test_the_cache_readout_does_not_claim_a_control_openrouter_has_not_got(
    switchable,
):
    """"Cache off" is a statement about a parameter. On a provider whose form
    does not carry it, it reads as a setting the user could go and change."""
    page = switchable
    switch_to(page, "openrouter")

    assert page.text_content("#cacheState") == "no cache control"
    assert page.locator("#cacheWarn").is_hidden()


def test_the_context_header_names_its_source_rather_than_saying_undefined(
    switchable,
):
    """The header prints the capability record's own words for where the
    ceiling came from. OpenRouter's record did not carry the field the page
    reads, and a missing key is not an empty one: every OpenRouter model read
    "context window 262,144 tokens · undefined".
    """
    page = switchable
    switch_to(page, "openrouter")
    header = page.text_content("#contextWindow")

    assert "undefined" not in header
    assert "OpenRouter" in header


def test_send_is_available_on_an_openrouter_model(switchable):
    """Anthropic's dynamic-filtering block is one provider's rule. Leaving it
    applied would refuse a request OpenRouter would have accepted."""
    page = switchable
    switch_to(page, "openrouter")
    page.fill("#prompt", "hello")

    assert page.is_enabled("#send")
    assert page.text_content("#blockedNote") == ""


# --- reopening one, which is where the assumption bit hardest ----------------


def test_reopening_an_openrouter_conversation_shows_it(sending):
    """The reported defect: a stored OpenRouter conversation opened empty.

    ``applyStoredOptions`` ended by calling the two Anthropic state
    functions, which read ``controls.thinking`` and ``controls.budget_tokens``
    - fields no OpenRouter schema carries. It threw, and the throw was
    *before* the loop that appends the bubbles, so the transcript never
    rendered and the settings were left half-applied. Nothing said so: the
    exception is asynchronous and the page just sat there looking loaded.
    """
    page = sending
    switch_to(page, "openrouter")
    page.fill("#prompt", "hello")
    page.click("#send")
    page.wait_for_function("() => state.turns.length === 1")
    page.click("#newConversation")
    page.wait_for_function("() => state.turns.length === 0")

    page.click(".conversation")
    page.wait_for_function("() => state.turns.length === 1")

    assert page.errors == []
    assert page.locator("#messages .msg").count() == 2
    assert "Hi there." in page.text_content("#messages")
    assert page.evaluate("() => state.data.provider.id") == "openrouter"


def test_a_reopened_conversation_gets_its_own_providers_settings_back(sending):
    """Settings belong to the conversation, so reopening must not quietly
    change the request - and an OpenRouter conversation's settings are
    OpenRouter's fields, none of which were restored."""
    page = sending
    switch_to(page, "openrouter")
    page.evaluate(
        """() => {
            setControl('reasoningEffort', 'high');
            setControl('chatCompletions', 'true');
            setControl('system', 'Answer in one sentence.');
            setControl('maxTokens', 4096);
        }"""
    )
    page.fill("#prompt", "hello")
    page.click("#send")
    page.wait_for_function("() => state.turns.length === 1")
    page.click("#newConversation")
    page.wait_for_function("() => state.turns.length === 0")

    page.click(".conversation")
    page.wait_for_function("() => state.turns.length === 1")

    assert page.input_value("#reasoningEffort") == "high"
    assert page.input_value("#chatCompletions") == "true"
    assert page.input_value("#system") == "Answer in one sentence."
    assert page.input_value("#maxTokens") == "4096"
    # And the pills read the same thing, since the row is what the user sees.
    assert "high" in page.text_content('[data-pill="reasoning"]')
    assert "chat.completions" in page.text_content('[data-pill="transport"]')


def test_reopening_hands_the_form_back_to_the_conversations_own_provider(sending):
    """A conversation carries its provider, because it carries its model.

    Reopening one sent through the other provider has to change the whole
    form back - controls, model list and pills - or the next turn would
    continue it under settings it was never sent with.
    """
    page = sending
    switch_to(page, "openrouter")
    page.fill("#prompt", "hello")
    page.click("#send")
    page.wait_for_function("() => state.turns.length === 1")
    switch_to(page, "anthropic")

    page.click(".conversation")
    page.wait_for_function("() => state.data.provider.id === 'openrouter'")
    page.wait_for_timeout(50)

    assert page.errors == []
    assert page.locator("#messages .msg").count() == 2
    assert set(OPENROUTER_PILLS) <= set(pills_on_screen(page))
    assert set(ANTHROPIC_PILLS).isdisjoint(pills_on_screen(page))
    # The model it was sent with, offered as a current model rather than
    # marked legacy: it is the other provider's, not a superseded one.
    assert page.input_value("#model") == "openrouter/openai/gpt-5.4"
    assert page.evaluate(
        "() => byId('model').selectedOptions[0].dataset.superseded"
    ) is None
    assert "legacy" not in page.text_content('[data-pill="model"]')
