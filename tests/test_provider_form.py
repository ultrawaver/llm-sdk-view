"""The chat form, asked for each provider in turn.

`/api/form` used to reach for `claude_model_id` on whatever model it was
given, so selecting an OpenRouter model crashed the page before it could draw
anything. These checks are about the seam rather than either provider's
controls: that each provider answers for its own form, that neither is handed
the other's vocabulary, and that the words a control is greyed out with are the
same words on both sides.

The per-provider control rules live in `test_form_controls.py` and
`test_openrouter_turn.py`. Nothing here needs a key or a network.
"""

import pytest

from native_api_chat.chat import form_schema
from native_api_chat.providers import PROVIDERS
from native_api_chat.providers.openrouter import (
    EDITABLE,
    NOT_SENT_BY_PLUGIN,
    STATUS_WORDS,
    UNKNOWN_TO_CATALOGUE,
    UNSUPPORTED_BY_MODEL,
    OpenRouterProvider,
    slug_for,
)
from native_api_chat.turn import STATUSES

ANTHROPIC_ONLY = ("thinking", "budget_tokens", "effort", "cache_control", "allowed_callers")
OPENROUTER_ONLY = (
    "chat_completions",
    "reasoning_effort",
    "reasoning_max_tokens",
    "reasoning_enabled",
    "reasoning_summary",
    "routing",
)
# Controls both forms carry, because both APIs really have the field. Only the
# first three come from the shared skeleton; each provider settles the rest
# itself, and they agree on the name rather than on the rules.
SHARED = ("model", "max_tokens", "system", "stream", "web_search", "max_uses")


@pytest.fixture
def openrouter_form(openrouter_registry):
    """The form for a registered OpenRouter model."""

    def build(*models):
        ids = openrouter_registry(*(models or ("openai/gpt-5.4",)))
        return form_schema(ids[0])

    return build


# --- each provider answers for its own form ---------------------------------


def test_the_form_answers_for_an_openrouter_model(openrouter_form):
    """The defect this file exists for: /api/form read Anthropic's capability
    record whatever the model was, and died on claude_model_id."""
    schema = openrouter_form()

    assert schema["provider"]["id"] == "openrouter"
    assert schema["provider"]["sdk"] == "openai-python"
    assert schema["transport"]["sdk_method"] == "responses.create"


def test_each_form_carries_only_its_own_controls(openrouter_form):
    """A form assembled from both providers' fields would offer the selected
    model controls it has never heard of, which is the one thing a form must
    not do. Checked in both directions, because either omission is the bug."""
    anthropic = form_schema("claude-sonnet-5")["controls"]
    openrouter = openrouter_form()["controls"]

    for control in ANTHROPIC_ONLY:
        assert control in anthropic, control
        assert control not in openrouter, control
    for control in OPENROUTER_ONLY:
        assert control in openrouter, control
        assert control not in anthropic, control
    for control in SHARED:
        assert control in anthropic and control in openrouter, control


def test_the_overlap_is_only_fields_both_apis_really_have(openrouter_form):
    """Whatever both forms hold has to be a field both APIs have. Anything
    else appearing in both is a control one of them cannot send.

    Sharing a name is not sharing rules: both have a web search switch, and
    each still settles its own status, note and tool version.
    """
    anthropic = set(form_schema("claude-sonnet-5")["controls"])
    openrouter = set(openrouter_form()["controls"])

    assert anthropic & openrouter == set(SHARED)


def test_both_forms_name_the_transport_that_will_be_called(openrouter_form):
    """The status line naming the call and the code pane rendering it are one
    fact read once, or the page describes a request nobody makes."""
    anthropic = form_schema("claude-sonnet-5")
    openrouter = openrouter_form()

    assert anthropic["transport"]["sdk_method"] == "messages.stream"
    assert anthropic["controls"]["stream"]["sdk_method"].startswith(
        "client.messages.stream"
    )
    assert openrouter["transport"]["sdk_method"] == "responses.create"
    assert openrouter["controls"]["stream"]["sdk_method"].startswith(
        "client.responses.create"
    )


# --- one vocabulary for a refusal -------------------------------------------


def test_every_control_status_is_from_the_shared_vocabulary(openrouter_form):
    """A greyed-out control must say why in the same words on both sides, or
    the page teaches two dialects of one idea. OpenRouter settles its controls
    with four internal codes; those are for deciding, not for showing."""
    for schema in (form_schema("claude-sonnet-5"), openrouter_form()):
        for name, control in schema["controls"].items():
            assert control["status"] in STATUSES, (schema["provider"]["id"], name)


def test_every_way_a_control_can_be_refused_has_a_shared_word_for_it():
    """OpenRouter settles its controls with four internal codes, and each is a
    different fact the user would act on differently. The form has to have a
    shared word for all four: a code with no translation would reach the page
    as its own private vocabulary the first time that state came up."""
    assert set(STATUS_WORDS) == {
        EDITABLE,
        UNSUPPORTED_BY_MODEL,
        NOT_SENT_BY_PLUGIN,
        UNKNOWN_TO_CATALOGUE,
    }
    assert set(STATUS_WORDS.values()) <= set(STATUSES)


def test_the_form_does_not_hand_the_page_the_code_it_decided_with(openrouter_form):
    """The four codes are for deciding which answer this is. A page given them
    would branch on them, which puts a capability decision in a UI
    conditional - where the next model's rules cannot reach it."""
    for control in openrouter_form()["controls"].values():
        assert "reason" not in control
        assert control["status"] not in STATUS_WORDS


# --- the switcher -----------------------------------------------------------


def test_the_switcher_offers_every_provider_from_either_side(openrouter_form):
    """The switcher is the same list whichever provider is selected: a page
    that could only see the provider it was already on could not leave it."""
    for schema in (form_schema("claude-sonnet-5"), openrouter_form()):
        offered = [option["id"] for option in schema["provider"]["options"]]
        assert offered == [provider.id for provider in PROVIDERS]


def test_a_provider_with_no_models_is_offered_and_says_why(openrouter_registry):
    """Hiding it would look like a shorter list; an empty list would look like
    a broken page. "No models available" is the thing the user can act on."""
    schema = form_schema("claude-sonnet-5")
    openrouter = next(
        option
        for option in schema["provider"]["options"]
        if option["id"] == "openrouter"
    )

    assert openrouter["available"] is False
    assert openrouter["models"] == 0
    assert openrouter["default_model"] is None
    assert "no models are available from OpenRouter" in openrouter["unavailable_note"]


def test_the_default_model_reads_the_same_in_both_places(openrouter_form):
    """The schema names a default twice - once for the selected provider and
    once per switcher entry - and two readings of one fact can disagree."""
    schema = openrouter_form()
    selected = next(
        option
        for option in schema["provider"]["options"]
        if option["id"] == schema["provider"]["id"]
    )

    assert selected["default_model"] == schema["default_model"]


def test_switching_to_a_provider_lands_on_a_model_it_really_offers(openrouter_form):
    """The switcher hands the page a model id to select, and a default that is
    not in the list would leave the dropdown showing nothing."""
    schema = openrouter_form("openai/gpt-5.4", "anthropic/claude-sonnet-5")

    assert schema["default_model"] in schema["models"]


def test_the_openrouter_default_is_not_a_pricing_tier(openrouter_registry):
    """`:batch` is an asynchronous endpoint and `:free` is rate limited by
    whoever else is using it. Neither is what "just open OpenRouter" should
    mean, though both stay selectable."""
    openrouter_registry(
        {"id": "openai/gpt-5.4", "created": 1_800_000_000, "pricing": {}},
        {"id": "openai/gpt-5.4:free", "created": 1_900_000_000, "pricing": {}},
        {"id": "openai/gpt-5.4:batch", "created": 1_900_000_000, "pricing": {}},
    )
    default = OpenRouterProvider().default_model

    assert ":" not in slug_for(default)
    assert default == "openrouter/openai/gpt-5.4"


def test_the_newest_standard_model_is_the_default(openrouter_registry):
    """Newest by publication date is a fact the catalogue reports. Ranking 356
    models by merit is not this project's business."""
    openrouter_registry(
        {"id": "vendor/old-1", "created": 1_700_000_000, "pricing": {}},
        {"id": "vendor/new-1", "created": 1_900_000_000, "pricing": {}},
    )

    assert OpenRouterProvider().default_model == "openrouter/vendor/new-1"
