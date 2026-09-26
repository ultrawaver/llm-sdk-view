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

## Canonical turn specification

The form, generated SDK code, and future execution path must share one typed turn specification. Code examples must never be maintained as independent templates that can drift from execution.

## Initial request path

The first scaffold only generates equivalent Anthropic Python SDK code. Model execution is intentionally deferred until the provider-facing values can be observed and tested.

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

The right-hand pane is a request preview, not a byte-for-byte capture of what
the plugin sends. `llm-anthropic` also adds `extra_body.temperature` and a
`thinking` block, which the canonical turn specification does not model yet.
Tests assert equality only on the four contract fields.

## Upstream boundary

- Provider behavior useful to every `llm-anthropic` user belongs upstream.
- Provider-neutral LLM hooks useful to multiple plugins belong in LLM core.
- UI, code presentation, and product statistics belong here.
