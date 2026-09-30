import hashlib
import json
from dataclasses import replace
from pathlib import Path

import llm
from starlette.applications import Starlette
from starlette.requests import Request
from starlette.responses import HTMLResponse, JSONResponse, Response, StreamingResponse
from starlette.routing import Route

from . import export_md, rates_page, store, token_count
from .chat import ChatOptions, ChatSession, MissingKeyError, form_schema
from .providers import provider_for
from .records import TurnRecord
from .turn import TurnOptions

STATIC = Path(__file__).parent / "static"

# Live sessions. The durable copy of every conversation is in llm's own SQLite;
# this dict only keeps the objects a page is currently talking through.
SESSIONS: dict[str, ChatSession] = {}


def _options_from_payload(payload: dict) -> TurnOptions:
    """Map the chat form onto the options of whichever provider will send it.

    The form is the provider's own, so reading it is too. Nothing here knows
    which fields exist; the model id decides who is asked.
    """
    model_id = payload.get("model") or ChatOptions.model
    return provider_for(model_id).options_from({**payload, "model": model_id})


def _prior_usage(conversation_id: str | None) -> dict | None:
    """The provider's own token counts for the last turn of a conversation.

    A cheap and exact answer to "how full is this conversation", and the one
    the meter should show the moment a stored conversation is opened: the API
    already reported it, so nothing has to be estimated or fetched. Context
    after a turn is the whole input (uncached + cache write + cache read) plus
    the reply it produced, which becomes part of the next request.

    Returns None when there is no stored turn or no counts on it, so a caller
    falls through to the counter and then to a labelled estimate.
    """
    if not conversation_id:
        return None
    try:
        stored = store.load_conversation(conversation_id)
    except Exception:  # noqa: BLE001 - a baseline is not worth failing a preview
        return None
    turns = (stored or {}).get("turns") or []
    if not turns:
        return None
    usage = getattr(turns[-1].response, "usage", None) or {}
    counts = {
        "input": usage.get("input_tokens"),
        "output": usage.get("output_tokens"),
        "cache_creation": usage.get("cache_creation_input_tokens"),
        "cache_read": usage.get("cache_read_input_tokens"),
    }
    return counts if any(isinstance(value, int) for value in counts.values()) else None


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
        options,
        conversation_id=conversation_id,
        history=history,
        baseline_usage=_prior_usage(conversation_id),
    )
    return SESSIONS[session_id]


def _preview_session(payload: dict) -> ChatSession:
    """A session to build a request with, without disturbing the real one.

    Looking at a request must not be able to destroy the conversation: a model
    change here builds on a throwaway session, while the stored session keeps
    its history until the user actually sends something.

    The throwaway still carries the stored history, because the request Send
    would build has it. Without it the pane showed only the message being
    typed for a conversation that already had turns - a second, quieter model
    of the request, and a context figure that counted one message.
    """
    session_id = payload.get("session_id", "default")
    conversation_id = payload.get("conversation_id")
    options = _options_from_payload(payload)
    existing = SESSIONS.get(session_id)
    if existing is not None and existing.options.model == options.model:
        existing.update_options(options)
        return existing
    history = None
    if conversation_id:
        try:
            history = store.thread_messages(conversation_id)
        except Exception as ex:  # noqa: BLE001 - refusing beats a wrong request
            raise ValueError(
                f"cannot read the conversation to preview it: {ex}"
            ) from ex
    return ChatSession(
        options, history=history, baseline_usage=_prior_usage(conversation_id)
    )


def _no_key(ex: MissingKeyError) -> JSONResponse:
    """One answer for a missing key, so every route says the same thing.

    Building a session can now fail this way, not just sending from one: a
    plugin that registers no models without a key makes a missing key look
    like an unknown model, and the provider is asked which it is while the
    session is still being built. A route that only guarded the send saw it
    as an unhandled error and answered 500.
    """
    return JSONResponse({"error": "missing_api_key", "detail": str(ex)}, status_code=400)


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


def asset_version(name: str) -> str:
    """Fingerprint an asset so a replaced file is never served from cache.

    The page is a local tool that changes several times a day, and no cache
    header alone settles whether the browser keeps its copy: a version in the
    URL makes a stale file unreachable instead of merely discouraged.
    """
    stat = (STATIC / name).stat()
    return hashlib.sha1(f"{stat.st_mtime_ns}:{stat.st_size}".encode()).hexdigest()[:8]


async def index(request: Request) -> HTMLResponse:
    html = (STATIC / "index.html").read_text("utf-8")
    for name in ("app.css", "app.js"):
        html = html.replace(f'"/static/{name}"', f'"/static/{name}?v={asset_version(name)}"')
    return HTMLResponse(html, headers={"Cache-Control": "no-store"})


async def asset(request: Request) -> Response:
    """Serve the page's own css/js (and the vendored highlighter).

    Only files under ``static/`` with a stylesheet or script suffix are
    served; anything else is a 404, so this route cannot read outside the
    asset directory.
    """
    name = request.path_params["name"]
    media = {".css": "text/css; charset=utf-8", ".js": "text/javascript; charset=utf-8"}.get(
        Path(name).suffix
    )
    path = (STATIC / name).resolve()
    if media is None or not path.is_file() or not path.is_relative_to(STATIC.resolve()):
        return JSONResponse({"error": "not found"}, status_code=404)
    return Response(
        path.read_bytes(),
        media_type=media,
        headers={
            # Revalidate, and let the version in the page's URL decide when
            # the bytes really changed.
            "Cache-Control": "no-cache",
            "ETag": f'"{asset_version(name)}"',
        },
    )


async def chat(request: Request) -> JSONResponse:
    payload = await request.json()
    if not payload.get("text"):
        return JSONResponse({"error": "text must not be empty"}, status_code=400)
    try:
        session = _session(payload)
        result = session.run_turn(payload["text"])
    except MissingKeyError as ex:
        return _no_key(ex)
    except (KeyError, TypeError, ValueError) as ex:
        return JSONResponse({"error": str(ex)}, status_code=400)
    except Exception as ex:  # noqa: BLE001 - a provider failure is still an answer
        # Nothing was stored, and that is the point: a request that never got
        # an answer must not leave a turn claiming it did.
        return JSONResponse({"error": f"the request failed: {ex}"}, status_code=502)
    record = result.pop("record")
    return JSONResponse({"sdk": session.provider.sdk, **result, **_keep(session, record)})


async def chat_stream(request: Request) -> StreamingResponse:
    payload = await request.json()
    if not payload.get("text"):
        return JSONResponse({"error": "text must not be empty"}, status_code=400)
    try:
        session = _session(payload)
    except MissingKeyError as ex:
        return _no_key(ex)
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

    An empty draft is not an empty request: the stored history, the system
    prompt and the tools all ride along no matter what is typed next. So an
    empty ``text`` is allowed and returns only the context figure - the
    baseline the context meter shows before a word is typed. There is no
    request code to show for a message that does not exist yet.

    The context figure is measured, not guessed, wherever that is possible: the
    prepared request is looked up in the counter's cache, and a missing count is
    asked for in the background (``pending``) rather than waited for. The count
    is what sees the ~2,200 tokens Anthropic adds for the ``web_search`` tool,
    which no character count in this process can.
    """
    payload = await request.json()
    text = payload.get("text") or ""
    try:
        session = _preview_session(payload)
        prepared = session.prepare(text)
    except MissingKeyError as ex:
        # A preview sends nothing, but it cannot be built either: without a key
        # llm-openrouter offers no models, so there is no model to build with.
        return _no_key(ex)
    except (KeyError, TypeError, ValueError) as ex:
        # Refusing early is the feature: an illegal combination is caught while
        # it is still free to fix.
        return JSONResponse({"error": str(ex)}, status_code=400)

    # The provider has already answered this question when the conversation's
    # last turn reported its usage and nothing new has been typed: that usage
    # is the conversation's own size. Counting again would spend a round trip
    # to be told the same number.
    already_answered = session.baseline_usage is not None and not text.strip()
    counted = None
    pending = False
    # Only the provider that owns the counter may be asked for a count. The
    # payload handed over is the request just built, so counting an OpenRouter
    # turn here would mean posting it to Anthropic.
    if not already_answered and session.provider.counts_tokens:
        counted = token_count.lookup(prepared.kwargs)
        if counted is None:
            pending = token_count.kick(prepared.kwargs)
    context = session.measure(prepared.kwargs, text, counted=counted)
    if pending:
        context = replace(context, pending=True)
    body = {
        "sdk": session.provider.sdk,
        "context": context.as_dict(),
        "sent": False,
    }
    if not text.strip():
        return JSONResponse({**body, "baseline": True})
    return JSONResponse(
        {
            **body,
            "code": prepared.code,
            "kwargs": prepared.kwargs,
            "transport": prepared.transport.name,
            # Whatever this provider can say about the request. Naming the
            # fields here would make the route know one provider's words.
            **prepared.facts,
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


async def rename_conversation(request: Request) -> JSONResponse:
    """Rename a conversation, in llm's own database like everything else.

    The sidebar, the chat title and `llm logs` all read the threads row, so
    one write keeps them in agreement. An unknown id is a 404 and an empty
    name a 400: neither may come back looking like a rename that happened.
    """
    payload = await request.json()
    try:
        renamed = store.rename_conversation(
            request.path_params["id"], payload.get("name", "")
        )
    except KeyError:
        return JSONResponse({"error": "conversation not found"}, status_code=404)
    except ValueError as ex:
        return JSONResponse({"error": str(ex)}, status_code=400)
    return JSONResponse(renamed)


async def export_conversation(request: Request) -> Response:
    """The whole conversation as one downloadable Markdown document.

    Built from the same stored records the panes read, so what leaves the
    app is what the app showed - the file is for sending to another AI or
    keeping, and an unknown id is a 404 rather than an empty document.
    """
    conversation_id = request.path_params["id"]
    try:
        loaded = store.load_conversation(conversation_id)
    except Exception as ex:  # noqa: BLE001 - read failures are not missing ids
        return JSONResponse({"error": f"cannot read history: {ex}"}, status_code=500)
    if loaded is None:
        return JSONResponse({"error": "conversation not found"}, status_code=404)
    filename = export_md.export_filename(loaded["name"], loaded["turns"])
    return Response(
        export_md.conversation_markdown(loaded),
        media_type="text/markdown; charset=utf-8",
        headers={"Content-Disposition": export_md.content_disposition(filename)},
    )


async def delete_conversation(request: Request) -> JSONResponse:
    """Delete a conversation from llm's own database, like everything else.

    The client asks twice before this is called; an unknown id is still a
    404, because a delete that found nothing must not look like one that
    happened.
    """
    conversation_id = request.path_params["id"]
    try:
        store.delete_conversation(conversation_id)
    except KeyError:
        return JSONResponse({"error": "conversation not found"}, status_code=404)
    return JSONResponse({"deleted": conversation_id})


async def form(request: Request) -> JSONResponse:
    """Defaults, limits and plugin capabilities for the chat form."""
    model_id = request.query_params.get("model") or ChatOptions.model
    try:
        return JSONResponse(form_schema(model_id))
    except (llm.UnknownModelError, ValueError) as ex:
        return JSONResponse({"error": str(ex)}, status_code=400)


async def health(request: Request) -> JSONResponse:
    return JSONResponse({"ok": True, "project": "native-api-chat"})


async def version(request: Request) -> JSONResponse:
    """The build the server would serve right now.

    A tab keeps its js until it is reloaded, and a local tool rebuilt several
    times a day means a tab can easily run a build the server has already
    replaced - reporting bugs that no longer exist. The page compares this
    answer against the build it loaded and says so on itself when they
    differ, instead of letting a stale tab impersonate the current one.
    """
    return JSONResponse(
        {"version": asset_version("app.js")},
        headers={"Cache-Control": "no-store"},
    )


async def rates(request: Request) -> JSONResponse:
    """Where the unit prices came from, and how many models they cover."""
    from .pricing import rates_status

    status = rates_status()
    status["models"] = len(rates_page.snapshot()["rates"])
    return JSONResponse(status)


def create_app() -> Starlette:
    # Unit prices come from the pricing page, so a start is also a refresh.
    # It runs in the background: the first cost shown may come from the disk
    # cache, and it says so.
    rates_page.kick_refresh(force=True)
    return Starlette(
        debug=False,
        routes=[
            Route("/", index),
            Route("/static/{name:path}", asset),
            Route("/api/form", form),
            Route("/api/chat", chat, methods=["POST"]),
            Route("/api/chat/stream", chat_stream, methods=["POST"]),
            Route("/api/preview", preview, methods=["POST"]),
            Route("/api/conversations", conversations),
            Route("/api/conversations/{id}", conversation),
            Route("/api/conversations/{id}", delete_conversation, methods=["DELETE"]),
            Route("/api/conversations/{id}/name", rename_conversation, methods=["POST"]),
            Route("/api/conversations/{id}/export.md", export_conversation),
            Route("/api/rates", rates),
            Route("/health", health),
            Route("/api/version", version),
        ],
    )


app = create_app()
