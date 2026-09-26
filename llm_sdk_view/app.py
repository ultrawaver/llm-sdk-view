import json
from pathlib import Path

import llm
from starlette.applications import Starlette
from starlette.requests import Request
from starlette.responses import HTMLResponse, JSONResponse, StreamingResponse
from starlette.routing import Route

from . import store
from .chat import ChatOptions, ChatSession, MissingKeyError, form_schema
from .records import TurnRecord

STATIC = Path(__file__).parent / "static"

# Live sessions. The durable copy of every conversation is in llm's own SQLite;
# this dict only keeps the objects a page is currently talking through.
SESSIONS: dict[str, ChatSession] = {}


def _options_from_payload(payload: dict) -> ChatOptions:
    """Map the chat form onto ChatOptions.

    Unknown or malformed values raise, and the routes turn that into a 400, so
    a bad form never falls back to a different request than the one shown.
    """
    defaults = ChatOptions()
    thinking = payload.get("thinking", defaults.thinking)
    return ChatOptions(
        model=payload.get("model", defaults.model),
        max_tokens=int(payload.get("max_tokens", defaults.max_tokens)),
        system=str(payload.get("system", defaults.system) or ""),
        # None means "the official default for this model"; anything else is
        # validated against the model's real thinking capability.
        thinking=str(thinking) if thinking is not None else None,
        # The form always sends the effort key; an empty value is "nothing
        # selected" (a greyed-out select after a model switch), i.e. the
        # default, never a level to validate against the model.
        effort=str(payload.get("effort") or defaults.effort),
        web_search=bool(payload.get("web_search", defaults.web_search)),
        web_search_type=payload.get("web_search_type", defaults.web_search_type),
        allowed_callers=payload.get("allowed_callers", defaults.allowed_callers),
        response_inclusion=payload.get("response_inclusion", defaults.response_inclusion),
        max_uses=int(payload.get("max_uses", defaults.max_uses)),
        cache_control=bool(payload.get("cache_control", defaults.cache_control)),
    )


def _session(payload: dict) -> ChatSession:
    """The session this turn belongs to.

    Four cases, all decided here so that the history and the options cannot
    disagree about which conversation is open:

    * same tab, same model, no other conversation named - reuse it;
    * same tab but the payload names a stored conversation - open that one
      with its stored messages;
    * ``new_conversation`` - start another, leaving the previous saved;
    * nothing to reuse - a new conversation under llm's own ULID.
    """
    session_id = payload.get("session_id", "default")
    conversation_id = payload.get("conversation_id")
    options = _options_from_payload(payload)
    existing = SESSIONS.get(session_id)
    if existing is not None and not payload.get("new_conversation"):
        same_model = existing.options.model == options.model
        same_conversation = not conversation_id or conversation_id == existing.conversation_id
        if same_model and same_conversation:
            # New form values, resolved and checked the same way a new session
            # would be, so "use the model default" is never left as None on a
            # session that is about to build a request.
            existing.update_options(options)
            return existing
    history = store.thread_messages(conversation_id) if conversation_id else None
    SESSIONS[session_id] = ChatSession(
        options, conversation_id=conversation_id, history=history
    )
    return SESSIONS[session_id]


def _preview_session(payload: dict) -> ChatSession:
    """A session to build a request with, without disturbing the real one.

    Looking at a request must not be able to destroy the conversation: a model
    change here builds on a throwaway session, while the stored session keeps
    its history until the user actually sends something.
    """
    session_id = payload.get("session_id", "default")
    options = _options_from_payload(payload)
    existing = SESSIONS.get(session_id)
    if existing is not None and existing.options.model == options.model:
        existing.update_options(options)
        return existing
    return ChatSession(options)


def _keep(session: ChatSession, record: dict) -> dict:
    """Write a finished turn to llm's own database.

    The words already cost money by the time this runs, so a failure has to
    reach the page as a failure: a turn that silently failed to be saved is
    a lost reply the user still believes is there.
    """
    try:
        turn_id = store.save_turn(session.last_response, TurnRecord.from_dict(record))
    except store.PersistenceError as ex:
        return {"record": record, "saved": False, "error": str(ex)}
    except Exception as ex:  # noqa: BLE001 - surfaced, never swallowed
        return {"record": record, "saved": False, "error": f"not saved: {ex}"}
    return {"record": record, "saved": True, "turn_id": turn_id}


async def index(request: Request) -> HTMLResponse:
    return HTMLResponse((STATIC / "index.html").read_text("utf-8"))


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
    except Exception as ex:  # noqa: BLE001 - a provider failure is still an answer
        # Nothing was stored, and that is the point: a request that never got
        # an answer must not leave a turn claiming it did.
        return JSONResponse({"error": f"the request failed: {ex}"}, status_code=502)
    record = result.pop("record")
    return JSONResponse({"sdk": "anthropic-python", **result, **_keep(session, record)})


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
                if event["type"] == "record":
                    # Nothing to store until the stream finishes: only then
                    # does the full Message, and therefore the response, exist.
                    saved = {"type": "record", **_keep(session, event["record"])}
                    yield f"data: {json.dumps(saved)}\n\n"
                    continue
                yield f"data: {json.dumps(event)}\n\n"
        except MissingKeyError:
            yield f"data: {json.dumps({'type': 'error', 'error': 'missing_api_key'})}\n\n"
        except (KeyError, TypeError, ValueError) as ex:
            yield f"data: {json.dumps({'type': 'error', 'error': str(ex)})}\n\n"
        except Exception as ex:  # noqa: BLE001 - reported, never silently saved
            yield f"data: {json.dumps({'type': 'error', 'error': f'the request failed: {ex}'})}\n\n"

    return StreamingResponse(events(), media_type="text/event-stream")


async def preview(request: Request) -> JSONResponse:
    """The exact request Send would build, built without sending it.

    This is not a second model of the request: it runs the same
    :meth:`ChatSession.prepare` and the same renderer as a real turn. The only
    difference is that the Response is never executed, so nothing reaches the
    provider, nothing is billed and nothing is recorded in the conversation.
    That is the whole point - the right pane has to be readable *before* the
    user pays for a call.
    """
    payload = await request.json()
    if not payload.get("text"):
        return JSONResponse({"error": "text must not be empty"}, status_code=400)
    try:
        prepared = _preview_session(payload).prepare(payload["text"])
    except (KeyError, TypeError, ValueError) as ex:
        # Refusing early is the feature: an illegal combination is caught while
        # it is still free to fix.
        return JSONResponse({"error": str(ex)}, status_code=400)
    return JSONResponse(
        {
            "sdk": "anthropic-python",
            "code": prepared.code,
            "kwargs": prepared.kwargs,
            "dynamic_filtering": prepared.dynamic_filtering,
            "allowed_callers": prepared.allowed_callers,
            "transport": prepared.transport,
            "context": prepared.context.as_dict(),
            "sent": False,
        }
    )


async def conversations(request: Request) -> JSONResponse:
    """Every saved conversation, newest first, for the sidebar."""
    try:
        items = store.list_conversations()
    except Exception as ex:  # noqa: BLE001 - the list must be able to fail loudly
        return JSONResponse({"error": f"cannot read history: {ex}"}, status_code=500)
    return JSONResponse({"items": items})


async def conversation(request: Request) -> JSONResponse:
    """One conversation with both panes of every turn it holds.

    Turns come back exactly as they were recorded - the same request kwargs
    the runtime built and the same response the SDK finished with - because
    rebuilding either from chat text would lose citations, thinking and
    provider metadata.
    """
    conversation_id = request.path_params["id"]
    try:
        loaded = store.load_conversation(conversation_id)
    except Exception as ex:  # noqa: BLE001 - read failures are not missing ids
        return JSONResponse({"error": f"cannot read history: {ex}"}, status_code=500)
    if loaded is None:
        return JSONResponse({"error": "conversation not found"}, status_code=404)
    return JSONResponse(
        {
            "id": loaded["id"],
            "name": loaded["name"],
            "model": loaded["model"],
            "turns": [turn.as_dict() for turn in loaded["turns"]],
        }
    )


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
            Route("/api/chat", chat, methods=["POST"]),
            Route("/api/chat/stream", chat_stream, methods=["POST"]),
            Route("/api/preview", preview, methods=["POST"]),
            Route("/api/conversations", conversations),
            Route("/api/conversations/{id}", conversation),
            Route("/health", health),
        ],
    )


app = create_app()
