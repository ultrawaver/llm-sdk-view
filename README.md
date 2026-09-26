# LLM SDK View

Chat on the left. Equivalent provider SDK code on the right.

LLM SDK View is a local web interface for [LLM](https://llm.datasette.io/) that
is meant to keep a real conversation beside the provider SDK code that represents
each turn. The first provider target is Anthropic through
[`llm-anthropic`](https://github.com/simonw/llm-anthropic), which uses
Anthropic's official Python SDK.

> **Status: early scaffold.** What works today is the local server, the two-pane
> shell, the canonical turn model, and a deterministic Anthropic Python SDK code
> generator. **There is no working chat pane and no model execution yet**: the
> right-hand pane previews the request that would be sent, and no paid API call
> is made anywhere in this repository or its tests.

## What actually runs today

- `llm sdk-view` starts a local ASGI server on `127.0.0.1`.
- The left pane edits a turn; the right pane renders the equivalent
  `anthropic` Python SDK call as text.
- `/api/preview` returns the generated code and the request dictionary. It
  never contacts a provider.
- Offline tests assert the request shape against the installed
  `llm-anthropic`, with no network and no API key.

Not implemented: real multi-turn conversation, streaming into the left pane,
SQLite-backed history in the UI, and any paid or authenticated model call.

## Target behaviour

The first Anthropic integration is expected to prove these four behaviours:

1. `web_search_20260318`
2. Dynamic filtering, without forcing `allowed_callers=["direct"]`
3. `response_inclusion="excluded"`
4. Prompt caching

Their current status against `llm-anthropic` is recorded in
[`docs/upstream-contributions.md`](docs/upstream-contributions.md). Items 1, 2
and 4 are already properties of the plugin and are asserted by offline tests.
Item 3 is expressed in generated code but is **not sendable yet**: the installed
plugin has no way to emit it, so it is unverified at runtime.

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

The first proposed upstream contribution is therefore narrowly scoped to a
single optional argument:

```text
WebSearch(response_inclusion="full" | "excluded")
```

Product-specific work stays here. See
[`docs/upstream-contributions.md`](docs/upstream-contributions.md).

## Repository layout

```text
llm_sdk_view/       Python package and LLM plugin
  app.py            Local ASGI application
  cli.py            `llm sdk-view` command
  codegen.py        Provider SDK code generation
  models.py         Canonical turn specification
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
