# Changelog

## 0.1a0 - Unreleased

- Initial repository scaffold.
- Added the `llm sdk-view` plugin command.
- Added a local two-pane UI shell.
- Added a canonical Anthropic turn specification.
- Added an Anthropic Python SDK code generator.
- Added focused tests and CI.

### Fixed after verification against llm 0.36 / llm-anthropic 0.29

- Removed the deprecated license classifier that made `pip install` fail under
  PEP 639 metadata.
- Moved prompt caching from a top-level `cache_control` argument, which the
  Messages API does not accept, onto the last content block of the final
  message, matching `llm-anthropic`.
- Removed `cache_ttl`; `llm-anthropic` has no way to send a TTL, so the option
  produced SDK code that could not match execution.
- Replaced the placeholder GitHub account in project URLs with the real owner.
- Made `uvicorn` import lazy so loading the plugin for any `llm` command stays
  cheap.
- Added offline contract tests that assert the four required Anthropic
  behaviours against the installed `llm-anthropic`.
- Bumped `actions/checkout` to v7 and `actions/setup-python` to v7 in CI.

### Added on the API form controls step

- The chat form now drives `model`, `max_tokens`, `system`, `web_search`,
  `web_search_type`, `allowed_callers`, `response_inclusion`, `max_uses` and
  `cache_control`. Defaults: `claude-sonnet-5`, `16384`, empty system, search
  on, `web_search_20260318`, `code_execution_20260120`, `excluded`, `1`, cache
  on.
- `GET /api/form?model=...` reports those defaults, the model's output-token
  ceiling, the web search tool version the plugin will emit for that model, and
  what the installed `llm-anthropic` can actually send.
- Every form value is checked against the built request before a turn streams:
  a value the plugin cannot honour is refused instead of silently changed.
- Empty `system` is omitted from the request; `max_uses=0` (unlimited) is
  omitted rather than sent as `0`; `allowed_callers=direct` is blocked because
  the installed plugin cannot send `allowed_callers`.
- No prompt cache TTL control: `llm-anthropic` hard-codes
  `cache_control={"type": "ephemeral"}` and exposes no TTL option.

### Added on the conversation branch

- `llm_sdk_view/chat.py`: a `ChatSession` that runs turns through
  `llm.Conversation` and `llm-anthropic`, streaming text events out.
- `/api/chat` and `/api/chat/stream` (server-sent events).
- The right pane now renders `model.build_kwargs()` output instead of a second,
  hand-maintained rendering of the turn.
- Offline tests covering multi-turn history, streaming, missing-key handling
  and the four Anthropic behaviours, using a faked Anthropic transport.
