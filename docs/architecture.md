# Architecture

## Objective

LLM SDK View is a thin visual layer over LLM and provider plugins.

```text
Browser UI
  -> local ASGI application
  -> LLM conversation and provider abstractions
  -> llm-anthropic
  -> Anthropic official Python SDK
  -> Anthropic Messages API
```

## One source of request truth

The form, generated SDK code, and execution path must share one request: the
dictionary `model.build_kwargs()` returns. Code examples must never be
maintained as independent templates that can drift from execution, so:

- `llm_sdk_view/codegen.py` renders a `build_kwargs()` result and cannot build
  a request of its own;
- every form value is checked against the request that was actually built, and
  a turn is refused when the two disagree;
- the static `/api/preview` endpoint was removed for exactly this reason: it
  rendered a hand-maintained request shape rather than the real one.

## Model capability matrix

Model facts come from three ranked sources, resolved in
`llm_sdk_view/capabilities.py`:

1. The **Anthropic Models API** — model list, `max_input_tokens` (context
   window), `max_tokens` (output ceiling), and the thinking/effort
   capabilities. Read through the Anthropic SDK (never through `llm`, so
   nothing lands in the prompt log), cached on disk, refreshed in the
   background so the page never waits for the network.
2. The **fallback profile** `llm_sdk_view/models.json` — a versioned snapshot
   of Anthropic's published specs for the four current models, used only when
   the API is unavailable and always labelled as a fallback.
3. The **installed llm-anthropic** — what can actually go on the wire: tool
   version per model, whether thinking can be disabled, which tool options
   exist.

Only narrow rules the Models API does not expose are hard-coded: default effort
per model, whether thinking can be switched off, and how dynamic filtering maps
onto `allowed_callers`. Nothing in the UI branches on a model id.

## Request path

The chat pane runs a real `llm.Conversation` through `llm-anthropic` and streams
the response; the right pane shows the request that run was about to send.

## First provider contract

The Anthropic implementation must prove:

```text
web_search_20260318
Dynamic filtering enabled
response_inclusion = excluded
Prompt caching enabled
```

Dynamic filtering is treated as enabled when the current web search version supports it and the request does not force direct calling.

## Verified status and known limits

Checked against `llm-anthropic` 0.29 by reading the installed plugin source.
Three of the four contract items are already properties of the plugin, not of
this repository:

- `web_search_20260318` is emitted when the model reports
  `supports_adaptive_thinking`;
- dynamic filtering is the default for that tool version and is preserved by
  never sending `allowed_callers`;
- prompt caching is applied by `options.cache` to the last content block of the
  final message.

`response_inclusion` is not exposed by the plugin at all. It is expressed in
generated code but is not yet sendable; see
[upstream contributions](upstream-contributions.md).

The right-hand pane is the request `llm-anthropic` built, not a
reconstruction of it: `llm-anthropic` adds `extra_body.temperature` and, for
thinking models, a `thinking` block, and both appear in the rendered code
because they are in the dictionary that was sent. Tests assert the rendered
call evaluates back to exactly that dictionary.

## Upstream boundary

- Provider behavior useful to every `llm-anthropic` user belongs upstream.
- Provider-neutral LLM hooks useful to multiple plugins belong in LLM core.
- UI, code presentation, and product statistics belong here.
