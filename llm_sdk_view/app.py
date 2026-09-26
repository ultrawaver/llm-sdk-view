from pathlib import Path

from starlette.applications import Starlette
from starlette.requests import Request
from starlette.responses import HTMLResponse, JSONResponse
from starlette.routing import Route

from .codegen import anthropic_kwargs, render_anthropic_python
from .models import AnthropicTurn, Message

STATIC = Path(__file__).parent / "static"


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


async def health(request: Request) -> JSONResponse:
    return JSONResponse({"ok": True, "project": "llm-sdk-view"})


def create_app() -> Starlette:
    return Starlette(
        debug=False,
        routes=[
            Route("/", index),
            Route("/api/preview", preview, methods=["POST"]),
            Route("/health", health),
        ],
    )


app = create_app()
