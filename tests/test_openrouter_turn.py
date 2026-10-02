"""The OpenRouter form, checked against the request it produces.

Every option here is settled by the provider, sent through the plugin's own
builders and then read back off the built request. The form value and the
request field are therefore two independently produced things, which is the
only arrangement in which "the pane shows what was sent" is a check rather
than a hope.

Nothing reaches the network, and no test needs a key.
"""

import llm
import pytest
from conftest import reasoning_entry

from native_api_chat.providers.openrouter import (
    OMITTED,
    REASONING_EFFORTS,
    OpenRouterOptions,
    OpenRouterProvider,
)
from native_api_chat.turn import TurnOptions, UnsupportedOptionError

pytest.importorskip("llm_openrouter")

PROVIDER = OpenRouterProvider()


@pytest.fixture
def build(openrouter_registry):
    """Settle options, build the turn through the plugin, hand both back."""

    def factory(**fields):
        (model_id,) = openrouter_registry("openai/gpt-5.4")
        model = llm.get_model(model_id)
        capabilities = PROVIDER.capabilities_for(model_id)
        options = OpenRouterOptions(model=model_id, **fields)
        settled = PROVIDER.accept(options, capabilities)
        conversation = llm.Conversation(model=model)
        response = conversation.prompt(
            "hello",
            system=settled.system or None,
            options=PROVIDER.plugin_options(settled, capabilities),
            tools=PROVIDER.tools(model, settled, capabilities),
            stream=True,
        )
        kwargs = PROVIDER.assemble(
            model, response.prompt, conversation, response.prompt.options
        )
        PROVIDER.verify(settled, kwargs)
        return settled, kwargs

    return factory


@pytest.fixture
def reasoning_model(openrouter_catalogue):
    """Install a catalogue entry whose model states its own reasoning rules."""

    def install(slug, efforts=None, mandatory=False, default=None):
        openrouter_catalogue(reasoning_entry(slug, efforts, mandatory, default))
        return f"openrouter/{slug}"

    return install


# --- the shape of the options -----------------------------------------------


def test_the_options_share_three_fields_with_every_provider_and_no_more():
    """The overlap is a model, a reply ceiling and a system prompt. Hoisting
    anything else into the shared object puts a control on the page that the
    selected model has never heard of."""
    shared = set(TurnOptions.__dataclass_fields__)
    mine = set(OpenRouterOptions.__dataclass_fields__)

    assert shared == {"model", "max_tokens", "system"}
    assert shared < mine
    assert "thinking" not in mine
    assert "cache_control" not in mine


@pytest.mark.parametrize(
    "fields",
    [
        {"reasoning_effort": "enormous"},
        {"reasoning_summary": "verbose"},
        {"reasoning_max_tokens": 0},
        {"max_uses": 0},
        {"search_context_size": "enormous"},
        {"routing": "fp8"},
        {"max_tokens": 0},
    ],
)
def test_a_value_no_api_would_take_is_refused_on_the_way_in(fields):
    with pytest.raises(ValueError):
        OpenRouterOptions(model="openrouter/openai/gpt-5.4", **fields)


def test_every_default_is_expressed_by_omission():
    """Absence on the wire means the provider's own default applied - never
    that a value was lost on the way there."""
    model_id = "openrouter/openai/gpt-5.4"
    options = OpenRouterOptions(model=model_id)
    capabilities = PROVIDER.capabilities_for(model_id)

    assert PROVIDER.plugin_options(options, capabilities) == {"max_tokens": 16384}
    assert options.reasoning_effort == OMITTED
    assert options.reasoning_enabled is None


# --- the four reasoning fields stay four ------------------------------------


def test_each_reasoning_field_lands_in_the_reasoning_block(build):
    """They are four separate API fields, so they are four separate controls:
    one slider writing to whichever seemed to fit would be inventing a
    parameter OpenRouter does not have."""
    _, kwargs = build(reasoning_effort="high", reasoning_summary="detailed")

    assert kwargs["reasoning"] == {"effort": "high", "summary": "detailed"}


@pytest.mark.parametrize(
    "fields,expected",
    [
        ({"reasoning_effort": "high"}, {"effort": "high"}),
        ({"reasoning_max_tokens": 2048}, {"max_tokens": 2048}),
        ({"reasoning_enabled": True}, {"enabled": True}),
        ({"reasoning_summary": "detailed"}, {"summary": "detailed"}),
    ],
)
def test_each_of_the_four_reaches_the_request_on_its_own(build, fields, expected):
    """One at a time, because two of them can no longer be asked for together:
    the point of the four is that each is its own field, and that is settled
    by each arriving alone."""
    _, kwargs = build(**fields)

    assert kwargs["reasoning"] == expected


def test_an_effort_and_a_budget_together_are_refused_rather_than_sent(build):
    """Measured, not read: OpenRouter answers the combination with 400 "Only
    one of \\"reasoning.effort\\" and \\"reasoning.max_tokens\\" can be
    specified". A form that offered both fields could otherwise draw - and
    send - a request the API refuses."""
    with pytest.raises(UnsupportedOptionError, match="never both"):
        build(reasoning_effort="high", reasoning_max_tokens=2048)


def test_the_budget_alone_is_still_sent(build):
    """The rule is about the pair, not about the field."""
    _, kwargs = build(reasoning_max_tokens=2048)

    assert kwargs["reasoning"] == {"max_tokens": 2048}


def test_an_effort_the_serving_model_does_not_list_is_refused(reasoning_model):
    """Those levels are accepted and silently mapped down to the nearest one
    the model does take, which would make the pane show one setting and the
    turn run at another."""
    model_id = reasoning_model("vendor/three-levels-1", ["high", "medium", "low"])
    capabilities = PROVIDER.capabilities_for(model_id)

    with pytest.raises(UnsupportedOptionError, match="maps"):
        PROVIDER.accept(
            OpenRouterOptions(model=model_id, reasoning_effort="max"), capabilities
        )
    assert PROVIDER.accept(
        OpenRouterOptions(model=model_id, reasoning_effort="medium"), capabilities
    ).reasoning_effort == "medium"


def test_a_model_that_requires_reasoning_refuses_both_ways_of_turning_it_off(
    reasoning_model,
):
    """Two different refusals, both measured. The model's own metadata says
    never to send `effort: "none"`; and `reasoning.enabled=false` comes back
    400 "Reasoning is mandatory for this endpoint and cannot be disabled."
    """
    model_id = reasoning_model(
        "vendor/mandatory-1", ["max", "high", "low"], mandatory=True
    )
    capabilities = PROVIDER.capabilities_for(model_id)

    with pytest.raises(UnsupportedOptionError, match="mandatory"):
        PROVIDER.accept(
            OpenRouterOptions(model=model_id, reasoning_effort="none"), capabilities
        )
    with pytest.raises(UnsupportedOptionError, match="cannot be disabled"):
        PROVIDER.accept(
            OpenRouterOptions(model=model_id, reasoning_enabled=False), capabilities
        )


def test_a_model_reasoning_is_optional_for_still_takes_off(reasoning_model):
    model_id = reasoning_model("vendor/optional-1", ["high", "low"])
    capabilities = PROVIDER.capabilities_for(model_id)

    assert PROVIDER.accept(
        OpenRouterOptions(model=model_id, reasoning_enabled=False), capabilities
    ).reasoning_enabled is False


def test_an_omitted_reasoning_field_is_absent_from_the_request(build):
    _, kwargs = build(reasoning_effort="low")

    assert kwargs["reasoning"] == {"effort": "low"}
    assert "summary" not in kwargs["reasoning"]
    assert "max_tokens" not in kwargs["reasoning"]


def test_reasoning_can_be_asked_for_without_an_effort(build):
    """`enabled` is its own field and means "reason with the defaults"."""
    _, kwargs = build(reasoning_enabled=True)

    assert kwargs["reasoning"] == {"enabled": True}


def test_the_reasoning_block_moves_under_extra_body_on_chat_completions(build):
    """Not cosmetic: the same setting is a different field on each path, so a
    verifier that assumed one shape would pass a request it never read."""
    _, kwargs = build(chat_completions=True, reasoning_effort="high")

    assert kwargs["extra_body"]["reasoning"] == {"effort": "high"}
    assert "reasoning" not in kwargs


# --- the transport switch is a constraint, not a rendering choice -----------


def test_web_search_is_refused_while_chat_completions_is_selected(build):
    """The plugin raises on that path, so the form refuses first and says
    which of the two the user would have to change."""
    with pytest.raises(UnsupportedOptionError, match="chat.completions"):
        build(chat_completions=True, web_search=True)


def test_a_summary_the_plugin_would_drop_is_refused_rather_than_sent(build):
    """Measured, not read: on the Chat path the mixin never copies
    reasoning_summary into `reasoning`, so the control would do nothing."""
    with pytest.raises(UnsupportedOptionError, match="reasoning_summary"):
        build(chat_completions=True, reasoning_summary="concise")


def test_the_same_summary_is_accepted_on_the_responses_path(build):
    _, kwargs = build(reasoning_summary="concise")

    assert kwargs["reasoning"]["summary"] == "concise"


# --- the system prompt ------------------------------------------------------


def test_the_system_prompt_lands_in_whichever_field_the_path_uses(build):
    _, on_responses = build(system="be brief")
    _, on_chat = build(chat_completions=True, system="be brief")

    assert on_responses["instructions"] == "be brief"
    assert on_chat["messages"][0] == {"role": "system", "content": "be brief"}


def test_an_empty_system_prompt_is_omitted(build):
    _, on_responses = build()
    _, on_chat = build(chat_completions=True)

    assert "instructions" not in on_responses
    assert [message["role"] for message in on_chat["messages"]] == ["user"]


# --- web search -------------------------------------------------------------


def test_the_tool_is_the_plugins_own_and_carries_only_what_was_set(build):
    """Hand-writing the spec here would be the second copy of the request
    this project exists to avoid."""
    _, kwargs = build(web_search=True, max_uses=3, search_context_size="medium")
    (tool,) = kwargs["tools"]

    assert tool == {
        "type": "openrouter:web_search",
        "parameters": {"max_uses": 3, "search_context_size": "medium"},
    }


def test_no_limit_is_expressed_by_omitting_max_uses(build):
    _, kwargs = build(web_search=True)
    (tool,) = kwargs["tools"]

    assert tool == {"type": "openrouter:web_search"}


def test_search_off_means_no_tool_at_all(build):
    _, kwargs = build()

    assert not kwargs.get("tools")


# --- the ceiling ------------------------------------------------------------


def test_max_tokens_is_held_to_the_endpoints_own_ceiling(build):
    with pytest.raises(ValueError, match="at most 128000"):
        build(max_tokens=200_000)


def test_the_ceiling_is_not_enforced_against_a_model_nobody_described(
    openrouter_registry,
):
    """With no catalogue there is no ceiling to enforce, and inventing one
    would refuse a request the endpoint would have accepted."""
    (model_id,) = openrouter_registry("openai/gpt-5.4")
    capabilities = PROVIDER.capabilities_for("openrouter/vendor/undescribed-1")
    options = OpenRouterOptions(model=model_id, max_tokens=200_000)

    assert PROVIDER.accept(options, capabilities).max_tokens == 200_000


def test_no_effort_level_is_refused_on_behalf_of_a_model_nobody_described(
    openrouter_registry,
):
    """The catalogue is the only source for what a model accepts, so a model
    it has never heard of has not refused anything. Turned into a refusal,
    "I do not know" would lock out every model on a machine that is offline
    the first time the page is opened."""
    (model_id,) = openrouter_registry("openai/gpt-5.4")
    capabilities = PROVIDER.capabilities_for("openrouter/vendor/undescribed-1")

    for level in ("none", "minimal", "low", "medium", "high", "xhigh", "max"):
        assert capabilities.effort_refusal(level) == ""
        assert capabilities.enabled_refusal(False) == ""
    assert capabilities.offered_efforts() == REASONING_EFFORTS
    assert PROVIDER.accept(
        OpenRouterOptions(model=model_id, reasoning_effort="xhigh"), capabilities
    ).reasoning_effort == "xhigh"


def test_a_model_that_lists_no_reasoning_refuses_the_reasoning_controls(
    openrouter_registry,
):
    (model_id,) = openrouter_registry(
        {
            "id": "vendor/plain-1",
            "supported_parameters": ["max_tokens"],
            "pricing": {},
            "top_provider": {},
        }
    )
    capabilities = PROVIDER.capabilities_for(model_id)

    with pytest.raises(UnsupportedOptionError, match="reasoning_effort"):
        PROVIDER.accept(
            OpenRouterOptions(model=model_id, reasoning_effort="high"), capabilities
        )


# --- what the pane can say --------------------------------------------------


def test_the_reported_facts_are_read_off_the_request(build):
    _, kwargs = build(
        reasoning_effort="high", web_search=True, routing={"quantizations": ["fp8"]}
    )
    facts = PROVIDER.request_facts(kwargs)

    assert facts["reasoning"] == {"effort": "high"}
    assert facts["routing"] == {"quantizations": ["fp8"]}
    assert facts["web_search"] == "openrouter:web_search"


def test_an_option_enum_is_reported_as_the_value_on_the_wire(build):
    """The plugin's options are str enums: json writes "high", but str()
    writes ReasoningEffortEnum.high, and a record is read by more than one
    thing."""
    _, kwargs = build(reasoning_effort="high")
    effort = PROVIDER.request_facts(kwargs)["reasoning"]["effort"]

    assert effort == "high"
    assert str(effort) == "high"


def test_a_turn_that_asked_for_nothing_reports_nothing(build):
    _, kwargs = build()

    assert PROVIDER.request_facts(kwargs) == {
        "reasoning": None,
        "routing": None,
        "web_search": None,
    }


# --- the verifier is not a formality ----------------------------------------


def test_the_verifier_catches_a_request_that_lost_the_setting(build):
    """A verifier that cannot fail is not checking anything, so this proves
    it reads the request rather than the options it was handed."""
    settled, kwargs = build(reasoning_effort="high")
    kwargs["reasoning"] = {}

    with pytest.raises(ValueError, match="reasoning effort"):
        PROVIDER.verify(settled, kwargs)


def test_the_verifier_catches_a_request_that_gained_a_tool(build):
    settled, kwargs = build()
    kwargs["tools"] = [{"type": "openrouter:web_search"}]

    with pytest.raises(ValueError, match="web_search is off"):
        PROVIDER.verify(settled, kwargs)


def test_the_verifier_reads_the_field_this_path_actually_uses(build):
    """max_output_tokens on one path, max_tokens on the other. Checking one
    name would silently pass every request on the other transport."""
    settled, kwargs = build(chat_completions=True, max_tokens=512)
    kwargs["max_tokens"] = 511

    with pytest.raises(ValueError, match="max_tokens"):
        PROVIDER.verify(settled, kwargs)
