import json
from pathlib import Path

from starlette.applications import Starlette
from starlette.requests import Request
from starlette.responses import HTMLResponse, JSONResponse, StreamingResponse
from starlette.routing import Route

from .chat import ChatOptions, ChatSession, MissingKeyError
from .codegen import anthropic_kwargs, render_anthropic_python
from .models import AnthropicTurn, Message

STATIC = Path(__file__).parent / "static"

# In-memory sessions. Nothing is persisted: a page reload starts a new
# conversation rather than resurrecting one from SQLite.
SESSIONS: dict[str, ChatSession] = {}


def _turn_from_payload(payload: dict) -> AnthropicTurn:
    messages = tuple(
        Message(role=item["role"], content=item["content"])
        for item in payload.get("messages", [])
    )
    return AnthropicTurn(
        model=payload.get("model", "claude-sonnet-5"),
        max_tokens=int(payload.get("max_tokens", 4096)),
        messages=messages,
        web_search=bool(payload.get("web_search", True)),
        max_searches=int(payload.get("max_searches", 5)),
        response_inclusion=payload.get("response_inclusion", "excluded"),
        prompt_cache=bool(payload.get("prompt_cache", True)),
    )


def _options_from_payload(payload: dict) -> ChatOptions:
    return ChatOptions(
        model=payload.get("model", "claude-sonnet-5"),
        max_tokens=int(payload.get("max_tokens", 4096)),
        web_search=bool(payload.get("web_search", True)),
        max_searches=int(payload.get("max_searches", 5)),
        response_inclusion=payload.get("response_inclusion", "excluded"),
        prompt_cache=bool(payload.get("prompt_cache", True)),
    )


def _session(payload: dict) -> ChatSession:
    session_id = payload.get("session_id", "default")
    if session_id not in SESSIONS:
        SESSIONS[session_id] = ChatSession(_options_from_payload(payload))
    return SESSIONS[session_id]


async def index(request: Request) -> HTMLResponse:
    return HTMLResponse((STATIC / "index.html").read_text("utf-8"))


async def preview(request: Request) -> JSONResponse:
    try:
        turn = _turn_from_payload(await request.json())
    except (KeyError, TypeError, ValueError) as ex:
        return JSONResponse({"error": str(ex)}, status_code=400)
    return JSONResponse(
        {
            "sdk": "anthropic-python",
            "code": render_anthropic_python(turn),
            "kwargs": anthropic_kwargs(turn),
            "execution": "disabled-in-initial-scaffold",
        }
    )


async def chat(request: Request) -> JSONResponse:
    payload = await request.json()
    if not payload.get("text"):
        return JSONResponse({"error": "text must not be empty"}, status_code=400)
    session = _session(payload)
    try:
        result = session.run_turn(payload["text"])
    except MissingKeyError as ex:
        return JSONResponse({"error": "missing_api_key", "detail": str(ex)}, status_code=400)
    return JSONResponse({"sdk": "anthropic-python", **result})


async def chat_stream(request: Request) -> StreamingResponse:
    payload = await request.json()
    if not payload.get("text"):
        return JSONResponse({"error": "text must not be empty"}, status_code=400)
    session = _session(payload)

    def events():
        try:
            for event in session.stream_turn(payload["text"]):
                yield f"data: {json.dumps(event)}\n\n"
        except MissingKeyError:
            yield f"data: {json.dumps({'type': 'error', 'error': 'missing_api_key'})}\n\n"

    return StreamingResponse(events(), media_type="text/event-stream")


async def health(request: Request) -> JSONResponse:
    return JSONResponse({"ok": True, "project": "llm-sdk-view"})


def create_app() -> Starlette:
    return Starlette(
        debug=False,
        routes=[
            Route("/", index),
            Route("/api/preview", preview, methods=["POST"]),
            Route("/api/chat", chat, methods=["POST"]),
            Route("/api/chat/stream", chat_stream, methods=["POST"]),
            Route("/health", health),
        ],
    )


app = create_app()
