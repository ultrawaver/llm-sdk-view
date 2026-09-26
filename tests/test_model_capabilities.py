"""The model capability matrix, checked against both sources.

The matrix in ``llm_sdk_view/models.json`` carries the numbers llm-anthropic
does not publish (context window). Everything else is re-derived from the
installed plugin at runtime. These tests exist so that a drift between the two,
or a capability the form claims but the request cannot carry, fails here
instead of in the UI.
"""

import pytest
from starlette.testclient import TestClient

from llm_sdk_view.app import app
from llm_sdk_view.capabilities import (
    DEFAULT_EFFORT,
    DISABLED_EFFORT,
    capabilities_for,
    model_ids,
    profile,
)
from llm_sdk_view.chat import ChatOptions, UnsupportedOptionError

FOUR_MODELS = (
    "claude-fable-5-1",
    "claude-opus-5-5",
    "claude-sonnet-5",
    "claude-haiku-4-5-20251001",
)

# Read from https://platform.claude.com/docs/en/about-claude/models/overview
# on 2026-09-26.
EXPECTED = {
    "claude-fable-5-1": {
        "context_window": 1_000_000,
        "max_output_tokens": 128_000,
        "thinking_mode": "adaptive",
        "thinking_always_on": True,
        "supports_effort": True,
        "default_effort": "high",
        "web_search_type": "web_search_20260318",
    },
    "claude-opus-5-5": {
        "context_window": 1_000_000,
        "max_output_tokens": 128_000,
        "thinking_mode": "adaptive",
        "thinking_always_on": True,
        "supports_effort": True,
        "default_effort": "medium",
        "web_search_type": "web_search_20260318",
    },
    "claude-sonnet-5": {
        "context_window": 1_000_000,
        "max_output_tokens": 128_000,
        "thinking_mode": "adaptive",
        "thinking_always_on": False,
        "supports_effort": True,
        "default_effort": "high",
        "web_search_type": "web_search_20260318",
    },
    "claude-haiku-4-5-20251001": {
        "context_window": 200_000,
        "max_output_tokens": 64_000,
        "thinking_mode": "extended",
        "thinking_always_on": False,
        "supports_effort": False,
        "default_effort": None,
        "web_search_type": "web_search_20250305",
    },
}


# --- the matrix itself ------------------------------------------------------


def test_matrix_lists_exactly_the_four_form_models():
    assert tuple(model_ids()) == FOUR_MODELS


def test_fallback_profile_is_versioned_and_labelled():
    data = profile()

    assert data["profile_version"] == "2026-09-26"
    assert "platform.claude.com" in data["profile_source"]
    assert "fallback" in data["profile_role"]


@pytest.mark.parametrize("model_id", FOUR_MODELS)
def test_matrix_matches_the_published_specs(model_id):
    capabilities = capabilities_for(model_id)
    expected = EXPECTED[model_id]

    assert capabilities.context_window == expected["context_window"]
    assert capabilities.max_output_tokens == expected["max_output_tokens"]
    assert capabilities.thinking_mode == expected["thinking_mode"]
    assert capabilities.thinking_always_on == expected["thinking_always_on"]
    assert capabilities.supports_effort == expected["supports_effort"]
    assert capabilities.default_effort == expected["default_effort"]
    assert capabilities.web_search_type == expected["web_search_type"]


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
    assert capabilities.supports_effort == bool(model.supports_thinking_effort)
    assert capabilities.can_disable_thinking == (
        bool(model.supports_thinking) and not model.always_thinks
    )
    assert capabilities.web_search_type == WebSearch().tool_spec(model)["type"]


@pytest.mark.parametrize("model_id", FOUR_MODELS)
def test_every_matrix_model_resolves_through_llm(model_id):
    import llm

    llm.get_model(capabilities_for(model_id).llm_id)


def test_effort_capability_matrix():
    assert capabilities_for("claude-fable-5-1").can_disable_thinking is False
    assert capabilities_for("claude-opus-5-5").can_disable_thinking is False
    assert capabilities_for("claude-sonnet-5").can_disable_thinking is True
    assert capabilities_for("claude-haiku-4-5-20251001").can_disable_thinking is True

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
    assert capabilities_for("claude-sonnet-5").effort_options() == (
        "default",
        "low",
        "medium",
        "high",
        "xhigh",
        "max",
        "off",
    )


def test_response_inclusion_is_only_real_on_20260318():
    assert capabilities_for("claude-sonnet-5").response_inclusion is True
    # The older tool version has no response_inclusion field at all.
    assert capabilities_for("claude-haiku-4-5-20251001").response_inclusion is False


# --- max_tokens -------------------------------------------------------------


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


@pytest.mark.parametrize("model_id", ("claude-sonnet-5", "claude-haiku-4-5-20251001"))
def test_effort_off_disables_thinking(model_id, make_session):
    prepared = make_session(model=model_id, effort=DISABLED_EFFORT).prepare("Hi")

    assert prepared.kwargs["thinking"] == {"type": "disabled"}
    assert "effort" not in prepared.kwargs.get("output_config", {})


@pytest.mark.parametrize("model_id", ("claude-fable-5-1", "claude-opus-5-5"))
def test_models_that_always_think_refuse_effort_off(model_id, make_session, fake_provider):
    with pytest.raises(UnsupportedOptionError, match="always thinking"):
        make_session(model=model_id, effort=DISABLED_EFFORT)

    assert fake_provider == []


def test_a_model_without_effort_refuses_effort_levels(make_session, fake_provider):
    with pytest.raises(UnsupportedOptionError, match="does not support effort"):
        make_session(model="claude-haiku-4-5-20251001", effort="high")

    assert fake_provider == []


def test_an_unknown_effort_level_is_refused(make_session):
    with pytest.raises(ValueError, match="effort must be one of"):
        make_session(model="claude-sonnet-5", effort="ultra")


def test_default_effort_is_the_form_default():
    assert ChatOptions().effort == DEFAULT_EFFORT


def test_no_form_control_touches_reasoning_display():
    """Reasoning visibility is display only, so it is not a form field."""
    fields = set(ChatOptions.__dataclass_fields__)

    assert not {"display", "reasoning", "hide_reasoning"} & fields


# --- per-model request and code agreement -----------------------------------


@pytest.mark.parametrize("model_id", FOUR_MODELS)
def test_right_pane_matches_the_request_for_every_model(model_id, make_session, fake_provider):
    chat = make_session(model=model_id)
    result = chat.run_turn("Does the code match the request?")

    assert fake_provider == [result["kwargs"]]
    assert '"name": "web_search"' in result["code"]


def test_haiku_request_has_no_response_inclusion(make_session):
    prepared = make_session(model="claude-haiku-4-5-20251001").prepare("Hi")
    tool = next(t for t in prepared.kwargs["tools"] if t["name"] == "web_search")

    assert tool["type"] == "web_search_20250305"
    assert "response_inclusion" not in tool
    assert "response_inclusion" not in prepared.code


# --- the form surface --------------------------------------------------------


@pytest.mark.parametrize("model_id", FOUR_MODELS)
def test_form_endpoint_describes_every_model(model_id):
    data = TestClient(app).get("/api/form", params={"model": model_id}).json()

    assert data["defaults"]["model"] == model_id
    assert data["defaults"]["effort"] == DEFAULT_EFFORT
    assert data["model"]["context_window"] == EXPECTED[model_id]["context_window"]
    assert data["model"]["max_tokens"] == EXPECTED[model_id]["max_output_tokens"]
    assert data["capabilities"]["supports_effort"] == EXPECTED[model_id]["supports_effort"]
    assert data["capabilities"]["can_disable_thinking"] is (
        not EXPECTED[model_id]["thinking_always_on"]
    )
    assert data["cache_ttl"]["supported"] is False


def test_form_endpoint_lists_the_models():
    data = TestClient(app).get("/api/form").json()

    assert data["models"] == list(FOUR_MODELS)
    assert data["default_model"] == "claude-sonnet-5"


def test_form_endpoint_rejects_a_model_outside_the_matrix():
    response = TestClient(app).get("/api/form", params={"model": "claude-opus-4.1"})

    assert response.status_code == 400


def test_the_ui_holds_no_model_specific_logic():
    """Capabilities must come from the matrix, not from UI conditionals."""
    from pathlib import Path

    import llm_sdk_view

    html = (Path(llm_sdk_view.__file__).parent / "static" / "index.html").read_text("utf-8")
    lowered = html.lower()

    for model_id in FOUR_MODELS:
        assert model_id not in html, f"{model_id} must not be hard-coded in the UI"
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
