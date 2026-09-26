import json
from pathlib import Path

import llm
from starlette.applications import Starlette
from starlette.requests import Request
from starlette.responses import HTMLResponse, JSONResponse, StreamingResponse
from starlette.routing import Route

from .chat import ChatOptions, ChatSession, MissingKeyError, form_schema
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
    """Map the chat form onto ChatOptions.

    Unknown or malformed values raise, and the routes turn that into a 400, so
    a bad form never falls back to a different request than the one shown.
    """
    defaults = ChatOptions()
    return ChatOptions(
        model=payload.get("model", defaults.model),
        max_tokens=int(payload.get("max_tokens", defaults.max_tokens)),
        system=str(payload.get("system", defaults.system) or ""),
        web_search=bool(payload.get("web_search", defaults.web_search)),
        web_search_type=payload.get("web_search_type", defaults.web_search_type),
        allowed_callers=payload.get("allowed_callers", defaults.allowed_callers),
        response_inclusion=payload.get("response_inclusion", defaults.response_inclusion),
        max_uses=int(payload.get("max_uses", defaults.max_uses)),
        cache_control=bool(payload.get("cache_control", defaults.cache_control)),
    )


def _session(payload: dict) -> ChatSession:
    """Reuse the conversation, but only while the form has not changed it."""
    session_id = payload.get("session_id", "default")
    options = _options_from_payload(payload)
    existing = SESSIONS.get(session_id)
    # Switching model means a new llm.Conversation; changing any other value
    # can be applied to the session that is already holding the history.
    if existing is None or existing.options.model != options.model:
        SESSIONS[session_id] = ChatSession(options)
    else:
        existing.options = options
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
    try:
        session = _session(payload)
        result = session.run_turn(payload["text"])
    except MissingKeyError as ex:
        return JSONResponse({"error": "missing_api_key", "detail": str(ex)}, status_code=400)
    except (KeyError, TypeError, ValueError) as ex:
        return JSONResponse({"error": str(ex)}, status_code=400)
    return JSONResponse({"sdk": "anthropic-python", **result})


async def chat_stream(request: Request) -> StreamingResponse:
    payload = await request.json()
    if not payload.get("text"):
        return JSONResponse({"error": "text must not be empty"}, status_code=400)
    try:
        session = _session(payload)
    except (KeyError, TypeError, ValueError) as ex:
        return JSONResponse({"error": str(ex)}, status_code=400)

    def events():
        try:
            for event in session.stream_turn(payload["text"]):
                yield f"data: {json.dumps(event)}\n\n"
        except MissingKeyError:
            yield f"data: {json.dumps({'type': 'error', 'error': 'missing_api_key'})}\n\n"
        except (KeyError, TypeError, ValueError) as ex:
            yield f"data: {json.dumps({'type': 'error', 'error': str(ex)})}\n\n"

    return StreamingResponse(events(), media_type="text/event-stream")


async def form(request: Request) -> JSONResponse:
    """Defaults, limits and plugin capabilities for the chat form."""
    model_id = request.query_params.get("model") or ChatOptions.model
    try:
        return JSONResponse(form_schema(model_id))
    except (llm.UnknownModelError, ValueError) as ex:
        return JSONResponse({"error": str(ex)}, status_code=400)


async def health(request: Request) -> JSONResponse:
    return JSONResponse({"ok": True, "project": "llm-sdk-view"})


def create_app() -> Starlette:
    return Starlette(
        debug=False,
        routes=[
            Route("/", index),
            Route("/api/form", form),
            Route("/api/preview", preview, methods=["POST"]),
            Route("/api/chat", chat, methods=["POST"]),
            Route("/api/chat/stream", chat_stream, methods=["POST"]),
            Route("/health", health),
        ],
    )


app = create_app()
