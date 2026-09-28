"""The Response pane has to be the SDK's answer, not a paraphrase of it.

Everything here runs against the fake transport, so nothing is paid for and
nothing leaves the machine; the point is that a reassembled response would
differ from the real one in exactly the ways these tests look for.
"""


from llm_sdk_view.records import (
    NAME_MAX_CHARS,
    RAW_MESSAGE_KEYS,
    RAW_USAGE_KEYS,
    ResponseView,
    TurnRecord,
    build_response_view,
    conversation_name,
    format_duration,
)

SONNET = "claude-sonnet-5"

# A Message with everything the pane claims to show. Nothing here can be
# recovered from the streamed text: citations, thinking, tool results and the
# cache counters live only on the Message.
RICH = {
    "id": "msg_rich_1",
    "type": "message",
    "role": "assistant",
    "model": "claude-sonnet-5-real",
    "content": [
        {"type": "thinking", "thinking": "considering whether to search"},
        {"type": "server_tool_use", "id": "srv_1", "name": "web_search", "input": {"query": "x"}},
        {
            "type": "web_search_tool_result",
            "tool_use_id": "srv_1",
            "content": [
                {
                    "type": "web_search_result",
                    "url": "https://example.com",
                    "encrypted_content": "abc",
                }
            ],
        },
        {
            "type": "text",
            "text": "It works.",
            "citations": [
                {
                    "type": "web_search_result_location",
                    "url": "https://example.com",
                    "cited_text": "It works.",
                }
            ],
        },
    ],
    "stop_reason": "end_turn",
    "usage": {
        "input_tokens": 1200,
        "output_tokens": 32,
        "cache_creation_input_tokens": 900,
        "cache_read_input_tokens": 150,
        "server_tool_use": {"web_search_requests": 2},
    },
}


def test_the_view_is_read_off_the_finished_message(make_session, response_scenario):
    """Nothing in the pane is reconstructed from streamed text."""
    response_scenario(RICH)
    prepared = make_session(model=SONNET).prepare("does it work?")
    list(prepared.response.stream_events())
    view = build_response_view(prepared.response)

    assert view.message_id == "msg_rich_1"
    assert view.stop_reason == "end_turn"
    # The model that actually answered, not the one that was asked.
    assert view.model == "claude-sonnet-5-real"
    assert view.text == "It works."


def test_blocks_the_streamed_text_cannot_express_survive(
    make_session, response_scenario
):
    """Thinking, citations and tool results are kept whole."""
    response_scenario(RICH)
    prepared = make_session(model=SONNET).prepare("does it work?")
    list(prepared.response.stream_events())
    view = build_response_view(prepared.response)

    assert view.thinking == "considering whether to search"
    assert view.citations[0]["url"] == "https://example.com"
    assert [block["type"] for block in view.server_tool_blocks] == [
        "server_tool_use",
        "web_search_tool_result",
    ]
    assert len(view.content_blocks) == 4


def test_usage_carries_cache_and_search_counts(make_session, response_scenario):
    """The numbers the pane promises are the numbers the API reported."""
    response_scenario(RICH)
    prepared = make_session(model=SONNET).prepare("does it work?")
    list(prepared.response.stream_events())
    view = build_response_view(prepared.response)

    assert view.usage["cache_creation_input_tokens"] == 900
    assert view.usage["cache_read_input_tokens"] == 150
    assert view.usage["web_search_requests"] == 2
    assert view.summary() == (
        "input 1200 · output 32 · cache creation 900 · cache read 150 · web searches 2"
    )


def test_a_plain_response_says_what_is_unreported(make_session):
    """No invention: what the API did not report says so."""
    result = make_session(model=SONNET).run_turn("hello")

    assert result["record"]["response"]["summary"].endswith("web searches unreported")


def test_the_record_ties_one_turn_together(make_session):
    """Request, response, code and options all belong to one turn."""
    session = make_session(model=SONNET)
    result = session.run_turn("hello")
    record = result["record"]

    assert record["conversation_id"] == session.conversation_id
    assert record["turn_id"] == session.last_response.id
    assert record["request_kwargs"] == result["kwargs"]
    assert record["rendered_code"] == result["code"]
    assert record["user_input"] == "hello"
    assert record["options"]["model"] == SONNET
    assert record["source"] == "llm-sdk-view"


def test_the_bubble_and_the_pane_are_the_same_message(make_session, response_scenario):
    """Two views of one object, never two objects."""
    response_scenario(RICH)
    events = list(make_session(model=SONNET).stream_turn("does it work?"))
    done = next(event for event in events if event["type"] == "done")
    record = next(event for event in events if event["type"] == "record")

    assert done["text"] == record["record"]["response"]["text"] == "It works."


def test_no_response_is_invented_before_the_stream_finishes(make_session):
    """A half-streamed answer is not a response yet, so nothing claims one."""
    session = make_session(model=SONNET)
    seen = []
    for event in session.stream_turn("hello"):
        seen.append(event["type"])
        if event["type"] == "prepared":
            break

    assert "record" not in seen
    assert "done" not in seen


def test_the_record_round_trips_without_losing_anything(make_session, response_scenario):
    """What the UI shows is also exactly what the database keeps."""
    response_scenario(RICH)
    result = make_session(model=SONNET).run_turn("does it work?")
    record = TurnRecord.from_dict(result["record"])

    assert record.as_dict() == result["record"]
    assert record.response.citations[0]["cited_text"] == "It works."
    assert record.response.usage["cache_read_input_tokens"] == 150


def test_a_view_missing_fields_still_loads():
    """Rows written before a field existed must not make the history unreadable."""
    view = ResponseView.from_dict({})

    assert view.text == ""
    assert view.usage == {}
    assert view.content_blocks == []


# --- latency: measured, never invented ----------------------------------------


def test_latency_comes_from_llms_own_measurement():
    """llm times the call; the view carries that number untouched."""
    class _Timed:
        def duration_ms(self):
            return 3400

    view = build_response_view(_Timed(), ttft_ms=210)

    assert view.duration_ms == 3400
    assert view.ttft_ms == 210
    assert view.latency_summary() == (
        "latency 3.40 s (client-measured, includes streaming)"
    )


def test_an_unmeasured_response_invents_no_latency():
    """No duration on the response means the pane says nothing about it."""

    class _Untimed:
        pass

    view = build_response_view(_Untimed())

    assert view.duration_ms is None
    assert view.ttft_ms is None
    assert view.latency_summary() is None


def test_format_duration_matches_the_console_style():
    assert format_duration(340) == "0.34 s"
    assert format_duration(3954) == "3.95 s"
    assert format_duration(None) is None
    assert format_duration(-5) is None


def test_the_streamed_turn_records_ttft(make_session):
    """The first-chunk time is measured on the way through, not reconstructed."""
    result = make_session(model=SONNET).run_turn("hello")
    ttft = result["record"]["response"]["ttft_ms"]

    assert isinstance(ttft, int) and ttft >= 0


# --- names come from the first message, not from a model ---------------------


def test_a_short_first_message_is_the_name():
    assert conversation_name("Tell me about caching") == "Tell me about caching"


def test_a_long_first_message_is_trimmed_at_a_word():
    name = conversation_name("word " * 40)

    assert len(name) <= NAME_MAX_CHARS + 1
    assert not name.rstrip("…").endswith("word ")  # not cut mid-word


def test_an_empty_first_message_gets_a_usable_name():
    assert conversation_name("   ") == "Untitled conversation"


def test_a_name_is_the_same_every_time():
    """Deterministic, because history has to survive being looked at twice."""
    text = "word " * 40

    assert conversation_name(text) == conversation_name(text)


# --- the official Raw shape -------------------------------------------------


def _view(message: dict | None, usage: dict) -> ResponseView:
    return ResponseView(
        response_id="resp_1",
        message_id=(message or {}).get("id"),
        model=(message or {}).get("model"),
        content_blocks=[],
        text="",
        thinking=None,
        citations=[],
        server_tool_blocks=[],
        stop_reason=(message or {}).get("stop_reason"),
        usage=usage,
        response_json=message,
    )


OFFICIAL_USAGE = {
    "input_tokens": 2227,
    "cache_creation_input_tokens": 0,
    "cache_read_input_tokens": 0,
    "cache_creation": {
        "ephemeral_5m_input_tokens": 0,
        "ephemeral_1h_input_tokens": 0,
    },
    "output_tokens": 18,
    "service_tier": "standard",
    "inference_geo": "not_available",
}


def test_raw_reattaches_usage_that_llm_anthropic_popped_off():
    message = {
        "id": "msg_1",
        "type": "message",
        "role": "assistant",
        "model": "claude-haiku-4-5-20251001",
        "content": [{"type": "text", "text": "hi"}],
        "container": None,
        "stop_reason": "end_turn",
        "stop_sequence": None,
        "stop_details": None,
        "diagnostics": None,
    }
    raw = _view(message, OFFICIAL_USAGE).raw

    assert raw["usage"]["input_tokens"] == 2227
    assert raw["usage"]["cache_creation"]["ephemeral_5m_input_tokens"] == 0


def test_raw_has_the_official_keys_in_the_official_order():
    """Every official key, and in the order the Playground prints them."""
    message = {
        "diagnostics": None,
        "stop_details": None,
        "stop_sequence": None,
        "stop_reason": "end_turn",
        "container": None,
        "content": [{"citations": None, "text": "hi", "type": "text"}],
        "role": "assistant",
        "type": "message",
        "id": "msg_1",
        "model": "claude-haiku-4-5-20251001",
    }
    raw = _view(message, OFFICIAL_USAGE).raw

    assert list(raw) == list(RAW_MESSAGE_KEYS)
    assert list(raw["usage"]) == list(RAW_USAGE_KEYS)
    # Block keys are put back too: a stored block comes out alphabetical.
    assert list(raw["content"][0]) == ["type", "text", "citations"]


def test_raw_leaves_out_everything_this_app_derived():
    """text, thinking, summary and the latency figures are ours, not the API's."""
    usage = dict(OFFICIAL_USAGE, web_search_requests=3)
    raw = _view({"model": "m"}, usage).raw

    assert "web_search_requests" not in raw["usage"]
    for key in ("response_id", "text", "thinking", "summary", "duration_ms", "ttft_ms"):
        assert key not in raw


def test_raw_keeps_any_key_the_api_adds_later():
    """The order list is presentation: an unknown key is appended, not dropped."""
    raw = _view({"model": "m", "brand_new_field": 1}, OFFICIAL_USAGE).raw

    assert raw["brand_new_field"] == 1


def test_raw_never_invents_a_count_the_provider_did_not_report():
    """With caching off llm-anthropic drops counters; a zero would be a guess."""
    raw = _view({"model": "m"}, {"input_tokens": 10, "output_tokens": 2}).raw

    assert "cache_read_input_tokens" not in raw["usage"]
    assert "cache_creation" not in raw["usage"]
    assert raw["usage"]["input_tokens"] == 10


def test_raw_is_none_when_there_is_no_message():
    assert _view(None, {}).raw is None


# --- the cost estimate rides the record -----------------------------------------


def test_the_cost_is_derived_like_raw_and_carries_its_rates_label():
    view = _view({"model": "claude-fable-5-1"}, {
        "input_tokens": 900,
        "cache_read_input_tokens": 8300,
        "cache_creation_input_tokens": 0,
        "output_tokens": 900,
        "web_search_requests": 1,
        "server_tool_use": {"web_search_requests": 1},
    })
    data = view.as_dict()

    cost = data["cost"]
    assert cost["total"] > 0
    assert cost["rates_source"] == "Anthropic pricing"
    assert cost["estimated"] is True
    # The cost is derived, not read back: it is never stored on the view.
    assert "cost" not in view.__dict__ or view.__dict__.get("cost") is None


def test_an_unknown_model_turn_carries_no_cost():
    view = _view({"model": "claude-imaginary-9"}, {"input_tokens": 1, "output_tokens": 1})

    assert view.as_dict()["cost"] is None
