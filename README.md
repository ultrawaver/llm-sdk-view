# LLM SDK View

Chat on the left. Equivalent provider SDK code on the right.

LLM SDK View is a local web interface for [LLM](https://llm.datasette.io/) that
is meant to keep a real conversation beside the provider SDK code that represents
each turn. The first provider target is Anthropic through
[`llm-anthropic`](https://github.com/simonw/llm-anthropic), which uses
Anthropic's official Python SDK.

> **Status: first conversation slice.** The chat pane now runs a real
> multi-turn conversation through `llm.Conversation` and `llm-anthropic`, and
> the right pane renders the parameters that path actually built. **No test in
> this repository contacts a provider**: the Anthropic transport is faked, so
> no API key is needed and no paid call is made. Running the app against the
> real API requires your own key, configured the usual LLM way.

## What actually runs today

- `llm sdk-view` starts a local ASGI server on `127.0.0.1`.
- The left pane sends a message into an `llm.Conversation`; replies stream back
  over `/api/chat/stream` as server-sent events.
- The right pane has two views of the same turn. **Request** renders the
  `anthropic` Python SDK call from `model.build_kwargs()` — the dictionary
  `llm-anthropic` hands to the SDK. There is no second copy of the request: the
  renderer takes that dictionary and nothing else. **Response** shows what came
  back, read off the Message the SDK finished with: content blocks, thinking,
  citations, server tool blocks, stop reason, input/output/cache tokens and web
  search requests, as formatted JSON plus a one-line usage summary. The left
  bubble and the Response pane are projections of the same record, so they
  cannot disagree: clicking the user's bubble opens the Request that went out,
  and clicking the assistant's opens the Response that came back.
- Each bubble shows the time of its turn in your computer's own zone, rendered
  through the browser's locale. The stored stamp is UTC, so the page converts
  it; a turn with no stamp shows no time rather than "now".
- Conversations are saved and can be reopened. Storage goes through LLM's own
  SQLite schema (`llm.logs.LogStore`), so these conversations are not a second
  chat database and they show up in `llm logs` unchanged. Titles come from the
  first message; no model call is made to name a conversation. `llm-anthropic` always opens `client.messages.stream()`, so
  that is what the pane shows; a non-streaming `messages.create()` is never
  rendered as if it happened.
- The request's settings are eight pills in the topbar — Model, Thinking,
  Effort, Max tokens, System, Web Search, Caching, Streaming — each opening a
  small menu that writes back into one hidden form, so what Send builds always
  matches what the pills show. A pill the model or the runtime cannot honour
  is dashed and grey with the reason attached, never a fake control.
- Every form control reports a status, so a greyed-out value says why:
  `Editable`, `API supported · runtime fixed`, `Unsupported by selected model`,
  `Unsupported by current tool version`, `Provider default`, or
  `Fallback capability data`.
- Model facts come from the [Anthropic Models
  API](https://platform.claude.com/docs/en/about-claude/models/overview) when a
  key is present (cached on disk, refreshed in the background), and from a
  versioned fallback profile when it is not. The UI always says which one it is
  showing.
- If no Anthropic key is available, the turn fails locally and nothing is sent.
- Offline tests drive the real LLM Python API with a faked transport: they
  assert history, streaming, the four Anthropic behaviours, and that the code
  matches the request that was sent.

Not implemented: any provider other than Anthropic, any paid or authenticated
call in the test suite, hand-written conversation titles, and history
management beyond opening a saved conversation (no search, delete or export).

## Storage

Every completed turn is written to LLM's own database (the one `llm logs`
reads) through `llm.logs.LogStore`:

| Endpoint | Purpose |
|---|---|
| `GET /api/conversations` | saved conversations, newest first |
| `GET /api/conversations/{id}` | one conversation, with each turn's request and response |

Resuming passes llm's stored messages back in, so history keeps its original
content blocks instead of being rebuilt from chat text. The request this app
built and the code it rendered are not part of llm's schema, so they go in one
sidecar table keyed by turn id — written in the same transaction, so a failed
save leaves nothing behind and is reported as a failure rather than as success.

## Target behaviour

The first Anthropic integration is expected to prove these four behaviours:

1. `web_search_20260318`
2. Dynamic filtering, without forcing `allowed_callers=["direct"]`
3. `response_inclusion="excluded"`
4. Prompt caching

Their current status against `llm-anthropic` is recorded in
[`docs/upstream-contributions.md`](docs/upstream-contributions.md). Items 1, 2
and 4 are already properties of the released plugin and are asserted by offline
tests. Item 3 needs a plugin that accepts `response_inclusion`; the released
`llm-anthropic` does not, so CI on this branch installs a pinned commit of the
contribution branch and the assertion is skipped when that capability is
absent.

## Requirements

- Python 3.10 or later
- `llm>=0.36`
- `llm-anthropic>=0.29`

Verified on Python 3.13.12 with `llm==0.36` and `llm-anthropic==0.29`.

## Install for development

With `uv`:

```bash
uv sync --extra test
source .venv/bin/activate
```

Or with `pip`:

```bash
python -m venv .venv
source .venv/bin/activate
pip install -e '.[test]'
```

## Run

```bash
llm sdk-view
```

Then open `http://127.0.0.1:8000`. The server binds to `127.0.0.1` by default
and warns if told otherwise.

Without activating the environment, use `uv run --extra test llm sdk-view`.

The same application can be started directly as a module:

```bash
python -m llm_sdk_view
```

## Test

```bash
just
```

Or:

```bash
pytest
ruff check .
```

Part of the suite drives the page in a real browser, because reading the
source cannot see a clipped menu or a click the browser never dispatched.
Those checks skip themselves unless the extra is installed; Playwright uses
an installed Google Chrome, so there is usually nothing to download:

```bash
pip install -e '.[test,browser]'
pytest -m browser          # only the real-page checks
pytest -m "not browser"    # skip them
```

On a machine with no Chrome, `playwright install chromium` provides one.

## Relationship to LLM, llm-anthropic and Anthropic

This is an independent, unofficial project. It is **not** maintained, endorsed,
affiliated with, or sponsored by Simon Willison, the LLM project, or Anthropic.
It is a plugin and a local UI built on top of their published, open-source
Python packages. "LLM", "llm-anthropic" and "Anthropic" are the marks of their
respective owners and are used here only to describe what this project depends
on.

## Why build on LLM?

LLM already provides the provider plugin model, conversations, streaming
abstractions, SQLite logging, key management, structured messages, and usage
tracking. `llm-anthropic` uses Anthropic's official Python SDK. This project
should contribute generally useful provider capabilities upstream instead of
maintaining a broad private fork.

## Upstream-first plan

Checked against the installed `llm==0.36` and `llm-anthropic==0.29` source,
three of the four target behaviours already work upstream and one does not:

- `web_search_20260318` — already emitted for adaptive-thinking models;
- dynamic filtering — already preserved, `allowed_callers` is never sent;
- prompt caching — already applied to the final message with `options.cache`;
- `response_inclusion` — **not exposed**, and the only upstream change needed.

The first upstream contribution is therefore narrowly scoped to a single
optional argument:

```text
WebSearch(response_inclusion="full" | "excluded")
```

It has been submitted as
[`simonw/llm-anthropic` PR #95](https://github.com/simonw/llm-anthropic/pull/95)
and this project waits for it rather than carrying a private fork. Status and
follow-up steps are in
[`docs/upstream-contributions.md`](docs/upstream-contributions.md).

## Repository layout

```text
llm_sdk_view/       Python package and LLM plugin
  app.py            Local ASGI application
  capabilities.py   Model capability matrix, from three ranked sources
  chat.py           The conversation path and the form schema
  cli.py            `llm sdk-view` command
  codegen.py        Provider SDK code generation from build_kwargs()
  model_api.py      Optional Anthropic Models API layer with a disk cache
  models.json       Versioned fallback capability profile
  static/index.html Minimal two-pane UI
tests/              Focused unit tests
docs/               Architecture, contribution and upstream notes
```

## Security

- No API key is required to install, test, or run this project today.
- When model execution is added, keys must stay in LLM's key store or in
  environment variables. Generated code reads `ANTHROPIC_API_KEY` from the
  environment and never embeds a secret.
- The development server binds to `127.0.0.1` by default.
- Model execution will not be enabled until request construction can be
  inspected and tested without accidental paid calls.
- To report a vulnerability, see [`SECURITY.md`](SECURITY.md).

## License

Apache License 2.0, matching LLM and `llm-anthropic`. See [`LICENSE`](LICENSE).
