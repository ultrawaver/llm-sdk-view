"""The model capability matrix, checked against both sources.

The matrix in ``native_api_chat/models.json`` carries the numbers llm-anthropic
does not publish (context window, default effort). Everything else is
re-derived from the installed plugin at runtime. These tests exist so that a
drift between the two, or a capability the form claims but the request cannot
carry, fails here instead of in the UI.
"""

import pytest
from starlette.testclient import TestClient

from native_api_chat.app import app
from native_api_chat.capabilities import (
    DEFAULT_EFFORT,
    THINKING_OFF,
    THINKING_ON,
    capabilities_for,
    model_ids,
    plugin_thinking_budget,
    profile,
)
from native_api_chat.chat import ChatOptions, UnsupportedOptionError

FOUR_MODELS = (
    "claude-fable-5-1",
    "claude-opus-5-5",
    "claude-sonnet-5",
    "claude-haiku-4-5-20251001",
)

# Read from https://platform.claude.com/docs/en/about-claude/models/overview
# and the per-model thinking docs on 2026-09-26.
EXPECTED = {
    "claude-fable-5-1": {
        "context_window": 1_000_000,
        "max_output_tokens": 128_000,
        "thinking_mode": "adaptive",
        "thinking_default": "on",
        "thinking_always_on": True,
        "thinking_editable": False,
        "thinking_off_request": "unsupported",
        "budget_tokens": None,
        "supports_effort": True,
        "default_effort": "high",
        "web_search_type": "web_search_20260318",
        "dynamic_filtering": "active",
        "effective_allowed_callers": "code_execution_20260120",
    },
    "claude-opus-5-5": {
        "context_window": 1_000_000,
        "max_output_tokens": 128_000,
        "thinking_mode": "adaptive",
        "thinking_default": "on",
        "thinking_always_on": True,
        "thinking_editable": False,
        "thinking_off_request": "unsupported",
        "budget_tokens": None,
        "supports_effort": True,
        "default_effort": "medium",
        "web_search_type": "web_search_20260318",
        "dynamic_filtering": "active",
        "effective_allowed_callers": "code_execution_20260120",
    },
    "claude-sonnet-5": {
        "context_window": 1_000_000,
        "max_output_tokens": 128_000,
        "thinking_mode": "adaptive",
        "thinking_default": "on",
        "thinking_always_on": False,
        "thinking_editable": True,
        "thinking_off_request": "disabled",
        "budget_tokens": None,
        "supports_effort": True,
        "default_effort": "high",
        "web_search_type": "web_search_20260318",
        "dynamic_filtering": "active",
        "effective_allowed_callers": "code_execution_20260120",
    },
    "claude-haiku-4-5-20251001": {
        "context_window": 200_000,
        "max_output_tokens": 64_000,
        "thinking_mode": "extended",
        "thinking_default": "off",
        "thinking_always_on": False,
        "thinking_editable": True,
        "thinking_off_request": "omitted",
        "budget_tokens": 1024,
        "supports_effort": False,
        "default_effort": None,
        "web_search_type": "web_search_20250305",
        "dynamic_filtering": "not-supported",
        "effective_allowed_callers": "direct",
    },
}


# --- the matrix itself ------------------------------------------------------


def test_matrix_lists_exactly_the_four_form_models():
    assert set(model_ids()) == set(FOUR_MODELS)


def test_fallback_profile_is_versioned_and_labelled():
    data = profile()

    assert data["profile_version"] == "2026-09-26"
    assert "platform.claude.com" in data["profile_source"]
    assert "fallback" in data["profile_role"]


@pytest.mark.parametrize("model_id", FOUR_MODELS)
def test_matrix_matches_the_published_specs(model_id):
    capabilities = capabilities_for(model_id)
    expected = EXPECTED[model_id]

    for key, value in expected.items():
        assert getattr(capabilities, key) == value, f"{model_id}: {key}"


@pytest.mark.parametrize("model_id", FOUR_MODELS)
def test_matrix_agrees_with_the_installed_plugin(model_id):
    """The second source: llm-anthropic's own model objects."""
    import llm
    from llm_anthropic import WebSearch

    capabilities = capabilities_for(model_id)
    model = llm.get_model(capabilities.llm_id)

    assert capabilities.api_model_id == model.claude_model_id
    assert capabilities.max_output_tokens == model.default_max_tokens
    assert capabilities.thinking_always_on == bool(model.always_thinks)
    assert capabilities.thinking_default == (
        "on" if (model.always_thinks or model.thinks_by_default) else "off"
    )
    assert capabilities.supports_effort == bool(model.supports_thinking_effort)
    assert capabilities.can_disable_thinking == (
        bool(model.supports_thinking) and not model.always_thinks
    )
    assert capabilities.web_search_type == WebSearch().tool_spec(model)["type"]


@pytest.mark.parametrize("model_id", FOUR_MODELS)
def test_every_matrix_model_resolves_through_llm(model_id):
    import llm

    llm.get_model(capabilities_for(model_id).llm_id)


def test_thinking_capability_matrix():
    assert capabilities_for("claude-fable-5-1").thinking_editable is False
    assert capabilities_for("claude-opus-5-5").thinking_editable is False
    assert capabilities_for("claude-sonnet-5").thinking_editable is True
    assert capabilities_for("claude-haiku-4-5-20251001").thinking_editable is True

    assert capabilities_for("claude-haiku-4-5-20251001").supports_effort is False
    assert capabilities_for("claude-haiku-4-5-20251001").effort_levels == ()


def test_effort_levels_come_from_the_plugin():
    assert capabilities_for("claude-sonnet-5").effort_levels == (
        "low",
        "medium",
        "high",
        "xhigh",
        "max",
    )
    # Effort has no "off" value any more: that is the thinking control's job.
    assert capabilities_for("claude-sonnet-5").effort_options() == (
        "default",
        "low",
        "medium",
        "high",
        "xhigh",
        "max",
    )


def test_response_inclusion_is_only_real_on_20260318():
    assert capabilities_for("claude-sonnet-5").response_inclusion is True
    # The older tool version has no response_inclusion field at all.
    assert capabilities_for("claude-haiku-4-5-20251001").response_inclusion is False


# --- thinking ----------------------------------------------------------------


@pytest.mark.parametrize("model_id", FOUR_MODELS)
def test_thinking_default_is_the_official_state(model_id, make_session):
    chat = make_session(model=model_id)
    expected = EXPECTED[model_id]["thinking_default"]

    assert chat.options.thinking == expected


@pytest.mark.parametrize("model_id", ("claude-fable-5-1", "claude-opus-5-5"))
def test_models_that_always_think_refuse_thinking_off(
    model_id, make_session, fake_provider
):
    with pytest.raises(UnsupportedOptionError, match="thinking cannot be turned off"):
        make_session(model=model_id, thinking=THINKING_OFF)

    assert fake_provider == []


@pytest.mark.parametrize("model_id", FOUR_MODELS)
def test_thinking_on_sends_the_official_value(model_id, make_session):
    prepared = make_session(model=model_id, thinking=THINKING_ON).prepare("Hi")
    expected = EXPECTED[model_id]

    if expected["thinking_mode"] == "adaptive":
        assert prepared.kwargs["thinking"]["type"] == "adaptive"
        assert "budget_tokens" not in prepared.kwargs["thinking"]
    else:
        assert prepared.kwargs["thinking"]["type"] == "enabled"
        assert (
            prepared.kwargs["thinking"]["budget_tokens"] == expected["budget_tokens"]
        )


def test_thinking_off_on_sonnet_sends_disabled(make_session):
    prepared = make_session(model="claude-sonnet-5", thinking=THINKING_OFF).prepare("Hi")

    assert prepared.kwargs["thinking"] == {"type": "disabled"}
    assert "thinking={" in prepared.code
    assert '"type": "disabled"' in prepared.code


def test_thinking_off_on_haiku_omits_the_field(make_session):
    """Haiku's official default is not to think, so OFF sends no thinking at all.

    Sending thinking={"type": "disabled"} to a model whose documented default
    is "off" is not verified, so the request relies on the documented default
    instead of on a value that might be rejected.
    """
    prepared = make_session(
        model="claude-haiku-4-5-20251001", thinking=THINKING_OFF
    ).prepare("Hi")

    assert "thinking" not in prepared.kwargs
    assert "thinking=" not in prepared.code


# --- budget_tokens ------------------------------------------------------------


def test_the_plugin_exposes_no_budget_option():
    """budget_tokens is a runtime fact, not a setting."""
    assert capabilities_for("claude-haiku-4-5-20251001").budget_tokens_editable is False
    assert capabilities_for("claude-haiku-4-5-20251001").budget_tokens == (
        plugin_thinking_budget()
    )


def test_budget_reaches_the_request_and_the_code(make_session):
    prepared = make_session(
        model="claude-haiku-4-5-20251001", thinking=THINKING_ON
    ).prepare("Hi")
    budget = capabilities_for("claude-haiku-4-5-20251001").budget_tokens

    assert prepared.kwargs["thinking"]["budget_tokens"] == budget
    assert f'"budget_tokens": {budget}' in prepared.code


def test_budget_is_not_sent_when_thinking_is_off(make_session):
    prepared = make_session(
        model="claude-haiku-4-5-20251001", thinking=THINKING_OFF
    ).prepare("Hi")

    assert "budget_tokens" not in prepared.code


def test_max_tokens_must_stay_above_the_budget(make_session, fake_provider):
    budget = capabilities_for("claude-haiku-4-5-20251001").budget_tokens

    with pytest.raises(ValueError, match="greater than the thinking budget"):
        make_session(
            model="claude-haiku-4-5-20251001", thinking=THINKING_ON, max_tokens=budget
        )
    assert fake_provider == []

    # One token more is enough: Anthropic requires budget_tokens < max_tokens.
    prepared = make_session(
        model="claude-haiku-4-5-20251001", thinking=THINKING_ON, max_tokens=budget + 1
    ).prepare("Hi")
    assert prepared.kwargs["max_tokens"] == budget + 1


def test_adaptive_models_have_no_budget_at_all(make_session):
    capabilities = capabilities_for("claude-sonnet-5")

    assert capabilities.budget_tokens is None
    assert capabilities.min_max_tokens(THINKING_ON) == 1


# --- max_tokens ------------------------------------------------------------


@pytest.mark.parametrize("model_id", FOUR_MODELS)
def test_max_tokens_at_the_ceiling_is_allowed(model_id, make_session):
    ceiling = capabilities_for(model_id).max_output_tokens

    chat = make_session(model=model_id, max_tokens=ceiling)
    assert chat.prepare("Hi").kwargs["max_tokens"] == ceiling


@pytest.mark.parametrize("model_id", FOUR_MODELS)
def test_max_tokens_above_the_ceiling_is_refused(model_id, make_session, fake_provider):
    ceiling = capabilities_for(model_id).max_output_tokens

    with pytest.raises(ValueError, match="max_tokens"):
        make_session(model=model_id, max_tokens=ceiling + 1)
    assert fake_provider == []


# --- effort -----------------------------------------------------------------


@pytest.mark.parametrize("model_id", FOUR_MODELS)
def test_default_effort_is_left_off_the_request(model_id, make_session):
    prepared = make_session(model=model_id).prepare("Hi")

    assert "effort" not in prepared.kwargs.get("output_config", {})
    assert "effort" not in prepared.code


@pytest.mark.parametrize("model_id", ("claude-fable-5-1", "claude-opus-5-5", "claude-sonnet-5"))
def test_effort_level_lands_in_output_config(model_id, make_session):
    prepared = make_session(model=model_id, effort="high").prepare("Hi")

    assert prepared.kwargs["output_config"] == {"effort": "high"}
    assert prepared.kwargs["thinking"]["type"] == "adaptive"
    assert '"effort": "high"' in prepared.code


def test_effort_is_independent_of_thinking_on_sonnet(make_session):
    """Anthropic documents effort as working with or without thinking."""
    off = make_session(model="claude-sonnet-5", thinking=THINKING_OFF, effort="high")
    kwargs = off.prepare("Hi").kwargs

    assert kwargs["thinking"] == {"type": "disabled"}
    assert kwargs["output_config"] == {"effort": "high"}


@pytest.mark.parametrize("level", ("xhigh", "max"))
def test_effort_levels_that_need_thinking_are_refused(make_session, fake_provider, level):
    """Anthropic rejects thinking disabled at xhigh and max effort."""
    with pytest.raises(ValueError, match="rejected when thinking is off"):
        make_session(model="claude-sonnet-5", thinking=THINKING_OFF, effort=level)

    assert fake_provider == []


def test_those_levels_are_allowed_with_thinking_on(make_session):
    prepared = make_session(
        model="claude-sonnet-5", thinking=THINKING_ON, effort="max"
    ).prepare("Hi")

    assert prepared.kwargs["output_config"] == {"effort": "max"}


def test_a_model_without_effort_refuses_effort_levels(make_session, fake_provider):
    with pytest.raises(UnsupportedOptionError, match="does not support effort"):
        make_session(model="claude-haiku-4-5-20251001", effort="high")

    assert fake_provider == []


def test_an_unknown_effort_level_is_refused(make_session):
    with pytest.raises(ValueError, match="effort must be one of"):
        make_session(model="claude-sonnet-5", effort="ultra")


def test_default_effort_is_the_form_default():
    assert ChatOptions().effort == DEFAULT_EFFORT
    assert ChatOptions().thinking is None, "None means the official default"


def test_no_form_control_touches_reasoning_display():
    """Reasoning visibility is display only, so it is not a form field."""
    fields = set(ChatOptions.__dataclass_fields__)

    assert not {"display", "reasoning", "hide_reasoning"} & fields
    assert not {"stream", "temperature", "top_p", "top_k"} & fields


# --- per-model request and code agreement -----------------------------------


@pytest.mark.parametrize("model_id", FOUR_MODELS)
def test_right_pane_matches_the_request_for_every_model(model_id, make_session, fake_provider):
    chat = make_session(model=model_id)
    result = chat.run_turn("Does the code match the request?")

    assert fake_provider == [result["kwargs"]]
    assert '"name": "web_search"' in result["code"]


@pytest.mark.parametrize("model_id", FOUR_MODELS)
def test_the_code_pane_reports_the_real_caller(model_id, make_session):
    prepared = make_session(model=model_id).prepare("Hi")

    assert prepared.allowed_callers == EXPECTED[model_id]["effective_allowed_callers"]
    assert prepared.dynamic_filtering == EXPECTED[model_id]["dynamic_filtering"]


def test_haiku_request_has_no_response_inclusion(make_session):
    prepared = make_session(model="claude-haiku-4-5-20251001").prepare("Hi")
    tool = next(t for t in prepared.kwargs["tools"] if t["name"] == "web_search")

    assert tool["type"] == "web_search_20250305"
    assert "response_inclusion" not in tool
    assert "response_inclusion" not in prepared.code


def test_haiku_shows_the_version_it_really_emits(make_session):
    """No UI tidying: the form shows 20250305, not the newer version."""
    data = TestClient(app).get(
        "/api/form", params={"model": "claude-haiku-4-5-20251001"}
    ).json()

    assert data["model"]["web_search_type"] == "web_search_20250305"
    assert data["controls"]["web_search_type"]["value"] == "web_search_20250305"
    assert data["controls"]["dynamic_filtering"]["value"] == "not-supported"
    assert data["controls"]["response_inclusion"]["status"] == (
        "Unsupported by current tool version"
    )


# --- the form surface --------------------------------------------------------


@pytest.mark.parametrize("model_id", FOUR_MODELS)
def test_form_endpoint_describes_every_model(model_id):
    data = TestClient(app).get("/api/form", params={"model": model_id}).json()
    expected = EXPECTED[model_id]

    assert data["defaults"]["model"] == model_id
    assert data["defaults"]["effort"] == DEFAULT_EFFORT
    assert data["defaults"]["thinking"] == expected["thinking_default"]
    assert data["model"]["context_window"] == expected["context_window"]
    assert data["model"]["max_tokens"] == expected["max_output_tokens"]
    assert data["capabilities"]["supports_effort"] == expected["supports_effort"]
    assert data["capabilities"]["thinking_default"] == expected["thinking_default"]
    assert data["capabilities"]["budget_tokens"] == expected["budget_tokens"]
    assert data["cache_ttl"]["supported"] is False


def test_form_endpoint_lists_the_models():
    data = TestClient(app).get("/api/form").json()

    assert set(data["models"]) == set(FOUR_MODELS)
    assert data["default_model"] == "claude-haiku-4-5-20251001"


def test_form_endpoint_rejects_a_model_llm_cannot_send():
    """What it refuses is a model that cannot be sent, not one merely absent
    from the dropdown: an older member of a series still has to load, because
    a stored conversation keeps pointing at it.
    """
    response = TestClient(app).get(
        "/api/form", params={"model": "claude-not-registered-anywhere"}
    )

    assert response.status_code == 400


def test_form_endpoint_accepts_a_superseded_model_for_old_conversations():
    """Superseded is not the same as unusable: Sonnet 5 is still sendable."""
    response = TestClient(app).get("/api/form", params={"model": "claude-sonnet-5"})

    assert response.status_code == 200
    assert response.json()["capabilities"]["id"] == "claude-sonnet-5"


def test_the_ui_holds_no_model_specific_logic(static_page):
    """Capabilities must come from the matrix, not from UI conditionals."""
    lowered = static_page.lower()

    for model_id in FOUR_MODELS:
        assert model_id not in static_page, f"{model_id} must not be hard-coded in the UI"
    for marker in ("always_thinks", "supports_adaptive_thinking", "default_max_tokens"):
        assert marker not in lowered, f"{marker} must not appear in the UI"


# --- prompt caching per model -----------------------------------------------


@pytest.mark.parametrize("model_id", FOUR_MODELS)
def test_cache_control_on_marks_the_request_for_every_model(model_id, make_session):
    prepared = make_session(model=model_id, cache_control=True).prepare("Hi")

    assert prepared.kwargs["messages"][-1]["content"][-1]["cache_control"] == {
        "type": "ephemeral"
    }
    assert '"cache_control"' in prepared.code


@pytest.mark.parametrize("model_id", FOUR_MODELS)
def test_cache_control_off_marks_nothing_for_every_model(model_id, make_session):
    prepared = make_session(model=model_id, cache_control=False).prepare("Hi")

    for message in prepared.kwargs["messages"]:
        for block in message["content"]:
            assert "cache_control" not in block
    assert "cache_control" not in prepared.code


# --- the documented minimum cacheable prompt length ---------------------------
# The Models API does not expose it, so it is a hard-coded narrow rule keyed
# by model family - the one place allowed to know these numbers.


@pytest.mark.parametrize(
    ("model_id", "expected"),
    [
        ("claude-haiku-4-5-20251001", 4096),
        ("claude-sonnet-5", 1024),
        ("claude-opus-5-5", 512),
    ],
)
def test_the_documented_cache_minimum_is_a_capability_flag(model_id, expected):
    capabilities = capabilities_for(model_id)

    assert capabilities.min_cacheable_tokens == expected
    assert capabilities.min_cacheable_tokens_source.startswith("Anthropic documented")


def test_an_unknown_family_reports_none_rather_than_guessing():
    from native_api_chat.capabilities import min_cacheable_tokens_for

    assert min_cacheable_tokens_for("claude-something-new-1") is None
