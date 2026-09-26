"""Offline proof that every chat form control drives the real request.

Nothing here touches the network: the Anthropic transport is faked, while
``llm.Conversation``, ``llm-anthropic`` and ``model.build_kwargs()`` are the
real installed code. The point of each test is the same: the value the form
shows has to be the value that ends up in the prepared request, and the code in
the right pane has to be rendered from that request.
"""

import llm
import pytest
from starlette.testclient import TestClient

from llm_sdk_view.app import app
from llm_sdk_view.capabilities import capabilities_for, model_ids
from llm_sdk_view.chat import (
    ALLOWED_CALLERS,
    RESPONSE_INCLUSIONS,
    WEB_SEARCH_TYPES,
    ChatOptions,
    UnsupportedOptionError,
    form_schema,
)

HAIKU = "claude-haiku-4-5-20251001"
SONNET = "claude-sonnet-5"


def _web_search_tool(kwargs: dict) -> dict:
    return next(t for t in kwargs["tools"] if t["name"] == "web_search")


def _last_block(kwargs: dict) -> dict:
    return kwargs["messages"][-1]["content"][-1]


# --- defaults -------------------------------------------------------------


def test_form_defaults():
    options = ChatOptions()
    assert options.model == HAIKU
    assert options.max_tokens == 16384
    assert options.system == ""
    # None means "the official default for this model", resolved per model.
    assert options.thinking is None
    assert options.effort == "default"
    assert options.web_search is True
    # None means "the version llm-anthropic gives this model".
    assert options.web_search_type is None
    assert options.allowed_callers == "code_execution_20260120"
    assert options.response_inclusion == "excluded"
    assert options.max_uses == 1
    assert options.cache_control is True


def test_there_is_no_stream_option_to_choose():
    """Streaming is fixed by the runtime, so it is not a form value."""
    fields = set(ChatOptions.__dataclass_fields__)

    assert "stream" not in fields


def test_form_schema_defaults_match_the_options():
    schema = form_schema(SONNET)

    assert schema["defaults"] == {
        "model": SONNET,
        "max_tokens": 16384,
        "system": "",
        "thinking": "on",
        "effort": "default",
        "web_search": True,
        "web_search_type": "web_search_20260318",
        "allowed_callers": "code_execution_20260120",
        "response_inclusion": "excluded",
        "max_uses": 1,
        "cache_control": True,
    }
    assert schema["transport"]["sdk_method"] == "stream"
    assert schema["web_search_types"] == list(WEB_SEARCH_TYPES)
    assert schema["response_inclusions"] == list(RESPONSE_INCLUSIONS)
    assert schema["allowed_callers"]["options"] == list(ALLOWED_CALLERS)
    assert schema["allowed_callers"]["api_default"] == "code_execution_20260120"
    assert schema["model"]["id"] == SONNET
    assert schema["model"]["sends"] == SONNET
    assert schema["model"]["max_tokens"] == 128000
    assert schema["model"]["context_window"] == 1000000
    assert schema["models"] == list(model_ids())


def test_default_request_carries_the_default_values(make_session):
    prepared = make_session().prepare("Hello")

    assert prepared.kwargs["model"] == HAIKU
    assert prepared.kwargs["max_tokens"] == 16384
    assert "system" not in prepared.kwargs
    # Haiku's official default is not to think, so there is no thinking field.
    assert "thinking" not in prepared.kwargs
    tool = _web_search_tool(prepared.kwargs)
    assert tool["type"] == "web_search_20250305"
    assert tool["max_uses"] == 1
    assert "allowed_callers" not in tool
    assert _last_block(prepared.kwargs)["cache_control"] == {"type": "ephemeral"}


def test_model_id_is_a_real_registered_model():
    llm.get_model(capabilities_for(ChatOptions.model).llm_id)


# --- every control reaches the request ------------------------------------


def test_every_form_field_reaches_the_prepared_request(make_session, capabilities):
    chat = make_session(
        model=SONNET,
        max_tokens=1024,
        system="Be terse",
        web_search_type="web_search_20260318",
        response_inclusion="full",
        max_uses=3,
        effort="low",
    )
    prepared = chat.prepare("Hi")
    kwargs = prepared.kwargs
    tool = _web_search_tool(kwargs)

    assert kwargs["model"] == SONNET
    assert kwargs["max_tokens"] == 1024
    assert kwargs["system"] == "Be terse"
    assert kwargs["output_config"] == {"effort": "low"}
    assert kwargs["thinking"]["type"] == "adaptive"
    assert tool["type"] == "web_search_20260318"
    assert tool["max_uses"] == 3
    assert _last_block(kwargs)["cache_control"] == {"type": "ephemeral"}
    if not capabilities(SONNET).response_inclusion:
        pytest.skip(
            "installed llm-anthropic has no response_inclusion; "
            "see docs/upstream-contributions.md"
        )
    assert tool["response_inclusion"] == "full"

    # The right pane renders the same values, because it renders this dict.
    for fragment in ("max_tokens=1024", 'system="Be terse"', '"max_uses": 3'):
        assert fragment in prepared.code


def test_right_pane_code_is_rendered_from_the_captured_request(
    make_session, fake_provider
):
    chat = make_session(max_tokens=2048)
    result = chat.run_turn("Does the code match?")

    assert fake_provider == [result["kwargs"]]
    assert "max_tokens=2048" in result["code"]


# --- system ---------------------------------------------------------------


def test_system_is_omitted_when_blank(make_session):
    prepared = make_session(system="").prepare("No system please")

    assert "system" not in prepared.kwargs
    assert "system=" not in prepared.code


def test_whitespace_only_system_is_also_omitted(make_session):
    prepared = make_session(system="   ").prepare("No system please")

    assert "system" not in prepared.kwargs


def test_system_is_sent_when_set(make_session):
    prepared = make_session(system="Answer in one line").prepare("Hi")

    assert prepared.kwargs["system"] == "Answer in one line"
    assert 'system="Answer in one line"' in prepared.code


# --- web search on / off --------------------------------------------------


def test_web_search_off_sends_no_tool(make_session, fake_provider):
    chat = make_session(web_search=False)
    prepared = chat.prepare("No search needed")

    assert "tools" not in prepared.kwargs
    assert "web_search" not in prepared.code

    chat.run_turn("No search needed")
    assert "tools" not in fake_provider[0]


def test_web_search_on_sends_the_tool(make_session):
    prepared = make_session(web_search=True).prepare("Search please")

    assert _web_search_tool(prepared.kwargs)["name"] == "web_search"


# --- web search type ------------------------------------------------------


def test_web_search_type_is_the_explicit_version(make_session):
    prepared = make_session(model=SONNET).prepare("Hi")
    tool = _web_search_tool(prepared.kwargs)

    assert tool["type"] == "web_search_20260318"
    assert "latest" not in tool["type"]


def test_a_type_the_model_cannot_emit_is_refused(make_session, fake_provider):
    """claude-sonnet-5 emits web_search_20260318, so 20250305 is rejected."""
    chat = make_session(model=SONNET, web_search_type="web_search_20250305")

    with pytest.raises(ValueError, match="web_search_20260318"):
        chat.prepare("Wrong tool version")

    assert fake_provider == []


def test_an_older_model_emits_the_older_version(make_session):
    """claude-haiku-4-5-20251001 has no adaptive thinking, so the plugin emits
    the older tool version - and response_inclusion does not exist on it."""
    chat = make_session(model=HAIKU, web_search_type="web_search_20250305")
    prepared = chat.prepare("Hi")

    tool = _web_search_tool(prepared.kwargs)
    assert tool["type"] == "web_search_20250305"
    # response_inclusion only exists on web_search_20260318.
    assert "response_inclusion" not in tool


# --- max_uses --------------------------------------------------------------


def test_max_uses_zero_is_omitted_rather_than_sent(make_session, fake_provider):
    chat = make_session(max_uses=0)
    prepared = chat.prepare("Unlimited searches")

    assert "max_uses" not in _web_search_tool(prepared.kwargs)
    assert "max_uses" not in prepared.code

    chat.run_turn("Unlimited searches")
    assert "max_uses" not in _web_search_tool(fake_provider[0])


def test_max_uses_is_sent_when_positive(make_session):
    prepared = make_session(max_uses=7).prepare("Seven searches at most")

    assert _web_search_tool(prepared.kwargs)["max_uses"] == 7
    assert '"max_uses": 7' in prepared.code


def test_negative_max_uses_is_rejected():
    with pytest.raises(ValueError):
        ChatOptions(max_uses=-1)


# --- allowed_callers / dynamic filtering -----------------------------------


def test_dynamic_filtering_is_active_on_the_newer_tool(make_session, capabilities):
    chat = make_session(model=SONNET)
    prepared = chat.prepare("Filter this")

    assert prepared.dynamic_filtering == "active"
    assert prepared.allowed_callers == "code_execution_20260120"
    # Nothing is sent, so the API default code_execution_20260120 applies.
    assert "allowed_callers" not in _web_search_tool(prepared.kwargs)


def test_the_older_tool_has_no_dynamic_filtering(make_session):
    prepared = make_session(model=HAIKU).prepare("Filter this")

    assert prepared.dynamic_filtering == "not-supported"
    assert prepared.allowed_callers == "direct"


def test_direct_callers_is_refused_rather_than_silently_ignored(
    make_session, fake_provider
):
    """llm-anthropic has no allowed_callers parameter at all."""
    chat = make_session(model=SONNET, allowed_callers="direct")

    with pytest.raises(UnsupportedOptionError, match="allowed_callers"):
        chat.prepare("Direct search")
    assert fake_provider == []


def test_the_form_shows_direct_as_an_official_but_unavailable_option():
    control = form_schema(SONNET)["controls"]["allowed_callers"]
    options = {option["value"]: option for option in control["options"]}

    assert control["editable"] is False
    assert options["code_execution_20260120"]["disabled"] is False
    assert options["code_execution_20260120"]["note"] == "API default · runtime fixed"
    assert options["direct"]["disabled"] is True
    assert options["direct"]["note"] == (
        "API supported · not exposed by llm-anthropic"
    )


def test_unknown_allowed_callers_is_rejected():
    with pytest.raises(ValueError):
        ChatOptions(allowed_callers="code_execution")


# --- response_inclusion -----------------------------------------------------


def test_response_inclusion_excluded_and_full(make_session, capabilities):
    if not capabilities(SONNET).response_inclusion:
        pytest.skip(
            "installed llm-anthropic has no response_inclusion; "
            "see docs/upstream-contributions.md"
        )
    excluded = make_session(model=SONNET, response_inclusion="excluded").prepare("Hi")
    full = make_session(model=SONNET, response_inclusion="full").prepare("Hi")

    assert _web_search_tool(excluded.kwargs)["response_inclusion"] == "excluded"
    assert '"response_inclusion": "excluded"' in excluded.code
    assert _web_search_tool(full.kwargs)["response_inclusion"] == "full"
    assert '"response_inclusion": "full"' in full.code


def test_response_inclusion_is_dropped_when_the_plugin_cannot_send_it(
    make_session, capabilities
):
    if capabilities(SONNET).response_inclusion:
        pytest.skip("installed llm-anthropic can send response_inclusion")
    prepared = make_session(model=SONNET, response_inclusion="excluded").prepare("Hi")

    assert "response_inclusion" not in _web_search_tool(prepared.kwargs)


def test_unknown_response_inclusion_is_rejected():
    with pytest.raises(ValueError):
        ChatOptions(response_inclusion="partial")


# --- cache_control ----------------------------------------------------------


def test_cache_control_on_marks_the_last_block(make_session):
    prepared = make_session(cache_control=True).prepare("Cache me")

    assert _last_block(prepared.kwargs)["cache_control"] == {"type": "ephemeral"}
    assert '"cache_control"' in prepared.code


def test_cache_control_off_sends_nothing(make_session, fake_provider):
    chat = make_session(cache_control=False)
    prepared = chat.prepare("Do not cache me")

    for message in prepared.kwargs["messages"]:
        for block in message["content"]:
            assert "cache_control" not in block
    assert "cache_control" not in prepared.code

    chat.run_turn("Do not cache me")
    assert "cache_control" not in str(fake_provider[0])


# --- max_tokens --------------------------------------------------------------


def test_max_tokens_above_the_model_ceiling_is_rejected(make_session, capabilities):
    ceiling = capabilities(HAIKU).max_output_tokens

    with pytest.raises(ValueError, match="max_tokens"):
        make_session(max_tokens=ceiling + 1)


def test_max_tokens_must_be_positive():
    with pytest.raises(ValueError):
        ChatOptions(max_tokens=0)


# --- TTL ----------------------------------------------------------------------


def test_the_installed_plugin_has_no_cache_ttl(capabilities):
    """There is no TTL control because llm-anthropic cannot send a TTL."""
    assert capabilities(HAIKU).cache_ttl is False


def test_form_schema_offers_no_ttl_control():
    schema = form_schema(HAIKU)

    assert schema["cache_ttl"]["supported"] is False
    assert schema["capabilities"]["cache_ttl"] is False
    assert "ttl" not in schema["defaults"]
    # The only thing shown is the provider default, and it is not editable.
    ttl = schema["controls"]["cache_control"]["ttl"]
    assert ttl == {
        "status": "Provider default",
        "value": "5m",
        "editable": False,
        "supported": False,
        "note": ttl["note"],
    }


def test_no_ttl_reaches_the_request(make_session):
    prepared = make_session().prepare("Any TTL?")

    assert "ttl" not in prepared.kwargs
    assert "ttl" not in str(prepared.kwargs["messages"])


# --- HTTP surface -------------------------------------------------------------


def test_form_endpoint_reports_defaults_and_capabilities():
    data = TestClient(app).get("/api/form").json()

    assert data["defaults"]["model"] == HAIKU
    assert data["defaults"]["max_tokens"] == 16384
    assert data["defaults"]["max_uses"] == 1
    assert data["defaults"]["response_inclusion"] == "excluded"
    assert data["defaults"]["allowed_callers"] == "code_execution_20260120"
    assert data["cache_ttl"]["supported"] is False
    assert data["model"]["web_search_type"] == "web_search_20250305"


def test_form_endpoint_rejects_an_unknown_model():
    response = TestClient(app).get("/api/form", params={"model": "not-a-model"})

    assert response.status_code == 400
    assert "error" in response.json()


def test_chat_endpoint_rejects_an_impossible_request(monkeypatch, fake_provider):
    """A form value the plugin cannot honour is a 400, not a silent change."""
    monkeypatch.setenv("ANTHROPIC_API_KEY", "fake-key-for-tests")
    from llm_sdk_view.app import SESSIONS

    SESSIONS.clear()
    response = TestClient(app).post(
        "/api/chat",
        json={"text": "Hi", "model": SONNET, "allowed_callers": "direct"},
    )

    assert response.status_code == 400
    assert "allowed_callers" in response.json()["error"]
    assert fake_provider == []


def test_chat_endpoint_applies_web_search_off(monkeypatch, fake_provider):
    monkeypatch.setenv("ANTHROPIC_API_KEY", "fake-key-for-tests")
    from llm_sdk_view.app import SESSIONS

    SESSIONS.clear()
    response = TestClient(app).post(
        "/api/chat", json={"text": "Hi", "web_search": False}
    )

    assert response.status_code == 200
    assert "tools" not in response.json()["kwargs"]
    assert "web_search" not in response.json()["code"]
    assert "tools" not in fake_provider[0]


def test_chat_endpoint_applies_thinking_off(monkeypatch, fake_provider):
    monkeypatch.setenv("ANTHROPIC_API_KEY", "fake-key-for-tests")
    from llm_sdk_view.app import SESSIONS

    SESSIONS.clear()
    response = TestClient(app).post(
        "/api/chat", json={"text": "Hi", "model": SONNET, "thinking": "off"}
    )

    assert response.status_code == 200
    assert response.json()["kwargs"]["thinking"] == {"type": "disabled"}


# --- the verification itself -------------------------------------------------


def test_a_request_that_drifts_from_the_form_is_refused(make_session, monkeypatch):
    """The build_kwargs() check has to bite, not just exist."""
    chat = make_session()
    original = chat.model.build_kwargs

    def drop_cache_control(prompt, conversation):
        kwargs = original(prompt, conversation)
        for message in kwargs["messages"]:
            for block in message["content"]:
                block.pop("cache_control", None)
        return kwargs

    monkeypatch.setattr(chat.model, "build_kwargs", drop_cache_control)
    with pytest.raises(ValueError, match="cache_control"):
        chat.prepare("Hi")


def test_a_request_that_drops_thinking_is_refused(make_session, monkeypatch):
    """Thinking off has to mean thinking off on the wire."""
    chat = make_session(model=SONNET, thinking="off")
    original = chat.model.build_kwargs

    def keep_thinking(prompt, conversation):
        kwargs = original(prompt, conversation)
        kwargs["thinking"] = {"type": "adaptive"}
        return kwargs

    monkeypatch.setattr(chat.model, "build_kwargs", keep_thinking)
    with pytest.raises(ValueError, match="thinking"):
        chat.prepare("Hi")


def test_a_request_that_invents_a_budget_is_refused(make_session, monkeypatch):
    """The budget is the runtime's number; a different one is a contradiction."""
    chat = make_session(model=HAIKU, thinking="on")
    original = chat.model.build_kwargs

    def bigger_budget(prompt, conversation):
        kwargs = original(prompt, conversation)
        kwargs["thinking"]["budget_tokens"] = 4096
        return kwargs

    monkeypatch.setattr(chat.model, "build_kwargs", bigger_budget)
    with pytest.raises(ValueError, match="budget_tokens"):
        chat.prepare("Hi")


def test_a_request_with_the_wrong_model_is_refused(make_session, monkeypatch):
    chat = make_session()
    original = chat.model.build_kwargs

    def wrong_model(prompt, conversation):
        kwargs = original(prompt, conversation)
        kwargs["model"] = "claude-something-else"
        return kwargs

    monkeypatch.setattr(chat.model, "build_kwargs", wrong_model)
    with pytest.raises(ValueError, match="model does not match"):
        chat.prepare("Hi")


def test_no_hand_maintained_request_renderer_survives():
    """The renderer must take build_kwargs() output and nothing else."""
    from llm_sdk_view import codegen

    assert not hasattr(codegen, "anthropic_kwargs")
    assert not hasattr(codegen, "anthropic_messages")
    assert not hasattr(codegen, "render_anthropic_python")
