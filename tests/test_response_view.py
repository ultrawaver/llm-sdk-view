"""The Response pane has to be the SDK's answer, not a paraphrase of it.

Everything here runs against the fake transport, so nothing is paid for and
nothing leaves the machine; the point is that a reassembled response would
differ from the real one in exactly the ways these tests look for.
"""


from llm_sdk_view.records import (
    NAME_MAX_CHARS,
    ResponseView,
    TurnRecord,
    build_response_view,
    conversation_name,
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
