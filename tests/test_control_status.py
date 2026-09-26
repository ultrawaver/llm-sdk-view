"""The independent re-review, in test form.

Most of these tests are not about one feature: they sweep every model and every
legal combination of the controls, and then assert that the request that comes
out could not be rejected by the API, that nothing the UI offers is quietly
dropped, and that every control says which of the four meanings it has.

Four meanings, one vocabulary:

    Editable                            the form sets it and the request carries it
    API supported · runtime fixed       the API has it, llm-anthropic does not
    Unsupported by selected model       the API has it, this model does not
    Unsupported by current tool version the API has it, this tool version does not
    Provider default                    no value is sent; the API decides
    Fallback capability data            the number came from the offline profile
"""

import pytest

from llm_sdk_view.capabilities import (
    DEFAULT_EFFORT,
    THINKING_OFF,
    THINKING_ON,
    capabilities_for,
)
from llm_sdk_view.chat import (
    EDITABLE,
    FALLBACK_DATA,
    PROVIDER_DEFAULT,
    RUNTIME_FIXED,
    UNSUPPORTED_BY_MODEL,
    UNSUPPORTED_BY_TOOL,
    ChatOptions,
    UnsupportedOptionError,
    form_schema,
)

FOUR_MODELS = (
    "claude-fable-5-1",
    "claude-opus-5-5",
    "claude-sonnet-5",
    "claude-haiku-4-5-20251001",
)

VOCABULARY = {
    EDITABLE,
    RUNTIME_FIXED,
    UNSUPPORTED_BY_MODEL,
    UNSUPPORTED_BY_TOOL,
    PROVIDER_DEFAULT,
    FALLBACK_DATA,
}


# --- 1. every control says which meaning it has --------------------------------


@pytest.mark.parametrize("model_id", FOUR_MODELS)
def test_every_control_reports_a_known_status(model_id):
    controls = form_schema(model_id)["controls"]

    assert controls, "the form must describe its controls"
    for name, control in controls.items():
        assert "status" in control, f"{name} has no status"
        assert control["status"] in VOCABULARY, f"{name}: {control['status']}"


@pytest.mark.parametrize("model_id", FOUR_MODELS)
def test_a_control_marked_editable_really_is_selectable(model_id):
    controls = form_schema(model_id)["controls"]

    for name, control in controls.items():
        if control["status"] != EDITABLE:
            continue
        options = control.get("options")
        if options is None:
            continue
        assert any(
            not option["disabled"] for option in options
        ), f"{name} is editable but every option is disabled"


@pytest.mark.parametrize("model_id", FOUR_MODELS)
def test_a_runtime_fixed_control_never_claims_to_be_a_choice(model_id):
    """Runtime fixed means the request cannot carry another value."""
    controls = form_schema(model_id)["controls"]
    fixed = {
        "web_search_type",
        "allowed_callers",
        "stream",
        "budget_tokens",
    }

    for name in fixed:
        control = controls[name]
        if control["status"] == UNSUPPORTED_BY_MODEL:
            continue
        assert control["status"] == RUNTIME_FIXED
        assert control.get("editable", False) is False


def test_the_ttl_is_only_ever_a_provider_default():
    for model_id in FOUR_MODELS:
        ttl = form_schema(model_id)["controls"]["cache_control"]["ttl"]

        assert ttl["status"] == PROVIDER_DEFAULT
        assert ttl["editable"] is False
        assert ttl["supported"] is False


def test_fallback_capability_data_is_labelled_as_such():
    """When the Models API has not answered, the label says so."""
    data = form_schema("claude-haiku-4-5-20251001")

    assert data["model_data"]["source"] in {"fallback", "models-api"}
    if data["model_data"]["source"] == "fallback":
        assert data["context"]["limit_data_source"] == FALLBACK_DATA


# --- 2. no legal combination can produce a 400 ---------------------------------


def _attempt(make_session, model_id, **overrides):
    """Build a turn, or return the refusal instead of sending it."""
    try:
        chat = make_session(model=model_id, **overrides)
    except (UnsupportedOptionError, ValueError) as ex:
        return None, ex
    return chat.prepare("Hi"), None


@pytest.mark.parametrize("model_id", FOUR_MODELS)
@pytest.mark.parametrize("thinking", (THINKING_ON, THINKING_OFF))
@pytest.mark.parametrize("effort", (DEFAULT_EFFORT, "low", "high", "xhigh", "max"))
def test_no_thinking_and_effort_combination_produces_an_illegal_request(
    model_id, thinking, effort, make_session, fake_provider
):
    """Either the UI refuses before sending, or the request is coherent."""
    prepared, refusal = _attempt(make_session, model_id, thinking=thinking, effort=effort)

    if refusal is not None:
        assert fake_provider == [], "a refused combination must not be sent"
        return

    kwargs = prepared.kwargs
    capabilities = capabilities_for(model_id)
    thinking_sent = kwargs.get("thinking")

    if thinking == THINKING_OFF:
        if capabilities.thinking_off_request == "disabled":
            assert thinking_sent == {"type": "disabled"}
        else:
            assert thinking_sent is None
        # Anthropic rejects thinking disabled at the highest effort levels.
        assert (kwargs.get("output_config") or {}).get("effort") not in ("xhigh", "max")
    else:
        assert thinking_sent is not None
        assert thinking_sent.get("type") in ("adaptive", "enabled")

    if capabilities.thinking_mode == "extended" and thinking == THINKING_ON:
        assert thinking_sent["budget_tokens"] < kwargs["max_tokens"]


@pytest.mark.parametrize("model_id", FOUR_MODELS)
def test_no_model_accepts_a_max_tokens_above_its_ceiling(
    model_id, make_session, fake_provider
):
    ceiling = capabilities_for(model_id).max_output_tokens

    prepared, refusal = _attempt(make_session, model_id, max_tokens=ceiling + 1)

    assert refusal is not None, "an output ceiling is a hard limit"
    assert fake_provider == []


@pytest.mark.parametrize("model_id", FOUR_MODELS)
@pytest.mark.parametrize("thinking", (THINKING_ON, THINKING_OFF))
def test_every_model_accepts_its_own_default_max_tokens(
    model_id, thinking, make_session
):
    prepared, refusal = _attempt(make_session, model_id, thinking=thinking)

    if refusal is None:
        assert prepared.kwargs["max_tokens"] == 16384


@pytest.mark.parametrize("model_id", FOUR_MODELS)
def test_the_tool_version_is_always_the_one_the_model_really_emits(
    model_id, make_session
):
    prepared, refusal = _attempt(make_session, model_id)
    if refusal is not None:
        pytest.skip("this combination is refused before a request is built")

    tool = next(t for t in prepared.kwargs["tools"] if t["name"] == "web_search")

    assert tool["type"] == capabilities_for(model_id).web_search_type
    assert "allowed_callers" not in tool
    if tool["type"] != "web_search_20260318":
        assert "response_inclusion" not in tool


@pytest.mark.parametrize("model_id", FOUR_MODELS)
def test_no_request_carries_an_effort_the_model_does_not_have(
    model_id, make_session
):
    capabilities = capabilities_for(model_id)

    prepared, refusal = _attempt(make_session, model_id, effort="high")

    if not capabilities.supports_effort:
        assert refusal is not None
        return
    assert prepared.kwargs["output_config"] == {"effort": "high"}


# --- 3. thinking, effort and budget land in the right fields -------------------


def test_thinking_is_never_expressed_through_effort(make_session):
    """The two are separate controls and separate request fields."""
    prepared = make_session(model="claude-sonnet-5", thinking=THINKING_ON, effort="low")
    kwargs = prepared.prepare("Hi").kwargs

    assert kwargs["thinking"]["type"] == "adaptive"
    assert kwargs["output_config"] == {"effort": "low"}


def test_the_budget_only_appears_on_extended_thinking(make_session):
    haiku = make_session(
        model="claude-haiku-4-5-20251001", thinking=THINKING_ON
    ).prepare("Hi")
    sonnet = make_session(model="claude-sonnet-5", thinking=THINKING_ON).prepare("Hi")

    assert "budget_tokens" in str(haiku.kwargs["thinking"])
    assert "budget_tokens" not in str(sonnet.kwargs["thinking"])


def test_no_effort_field_exists_on_a_model_without_effort(make_session):
    prepared = make_session(model="claude-haiku-4-5-20251001").prepare("Hi")

    assert "output_config" not in prepared.kwargs
    assert "effort" not in prepared.code


# --- 7. nothing official is hidden, nothing shown is impossible ----------------


def test_no_sampling_parameters_were_added():
    """The scope of this round: no temperature, top_p or top_k."""
    fields = set(ChatOptions.__dataclass_fields__)

    assert not {"temperature", "top_p", "top_k", "stop_sequences"} & fields


def test_the_scope_did_not_grow():
    """Only the controls this round authorised, plus the ones already there."""
    fields = set(ChatOptions.__dataclass_fields__)

    assert fields == {
        "model",
        "max_tokens",
        "system",
        "thinking",
        "effort",
        "web_search",
        "web_search_type",
        "allowed_callers",
        "response_inclusion",
        "max_uses",
        "cache_control",
    }


def test_haiku_shows_the_search_behaviour_it_really_has():
    """No UI tidying: Haiku's tool has no dynamic filtering and no inclusion."""
    controls = form_schema("claude-haiku-4-5-20251001")["controls"]

    assert controls["web_search_type"]["value"] == "web_search_20250305"
    assert controls["allowed_callers"]["value"] == "direct"
    assert controls["dynamic_filtering"]["value"] == "not-supported"
    assert controls["dynamic_filtering"]["status"] == UNSUPPORTED_BY_TOOL
    assert controls["response_inclusion"]["status"] == UNSUPPORTED_BY_TOOL
    assert controls["effort"]["status"] == UNSUPPORTED_BY_MODEL


def test_the_newer_models_show_dynamic_filtering_as_active():
    for model_id in ("claude-fable-5-1", "claude-opus-5-5", "claude-sonnet-5"):
        controls = form_schema(model_id)["controls"]

        assert controls["web_search_type"]["value"] == "web_search_20260318"
        assert controls["allowed_callers"]["value"] == "code_execution_20260120"
        assert controls["dynamic_filtering"]["value"] == "active"
