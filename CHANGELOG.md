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

### Added on the model capability and streaming step

- `stream` control (ON/OFF). ON shows text as it arrives; OFF buffers the turn
  and delivers the finished reply. Both modes send identical provider
  parameters, and `stream` never reaches the request.
- Completion now reads the reply from the final accumulated Message, so the
  bubble, the returned text and the recorded conversation agree even when
  streaming was used.
- The right pane renders the transport `llm-anthropic` really uses,
  `client.messages.stream(...)`, in both modes. A buffered turn is not rendered
  as a `messages.create()` that never happens; the reason is stated in the code.
- Model facts come from the Anthropic Models API when a key is present -
  model list, `max_input_tokens`, `max_tokens`, thinking and effort
  capabilities - cached on disk and refreshed in the background so the page
  never waits for the network. Without it, the versioned fallback profile in
  `llm_sdk_view/models.json` is used and labelled as a fallback.
- `llm_sdk_view/capabilities.py` holds one `ModelCapabilities` structure per
  model; the UI branches on capability flags, never on a model id.
- Model dropdown, context window display, per-model output ceiling validation,
  and an `effort` control (`default`, the plugin's levels, `off`) that is
  disabled where the plugin cannot honour it.

### Added on the form honesty step

- Default model is `claude-haiku-4-5-20251001`; `max_tokens` still defaults to
  `16384` and shows the selected model's output ceiling.
- `thinking` and `effort` are now separate controls, because they are separate
  API fields: `thinking` produces the `thinking` block, `effort` produces
  `output_config.effort`. `effort=off` no longer exists.
  - `claude-fable-5-1` / `claude-opus-5-5`: adaptive thinking, locked ON and
    greyed, because both the API and `llm-anthropic` reject disabling it.
  - `claude-sonnet-5`: ON/OFF, default ON (the official default); OFF sends
    `thinking={"type": "disabled"}`. Effort stays usable, but `xhigh` and
    `max` are refused while thinking is off.
  - `claude-haiku-4-5-20251001`: ON/OFF, default OFF (the official default);
    OFF omits the field entirely rather than sending an unverified `disabled`.
- Thinking budget: `llm-anthropic` hard-codes `DEFAULT_THINKING_TOKENS` and
  exposes no option, so `budget_tokens` is read off the installed plugin and
  shown read-only as `API supported · runtime fixed`. `max_tokens` is still
  validated against it, because Anthropic requires `budget_tokens < max_tokens`.
- Context meter: shows `current or estimated tokens / limit`, a percentage and
  the source of both numbers (`estimated` before a turn, `API usage` after one,
  `unknown` when a block cannot be measured). It refuses to send when the
  estimate plus the reserved output cannot fit, instead of trimming, and names
  the two ways out.
- `stream` is read-only `ON`, marked `API supported · runtime fixed by
  llm-anthropic`. The fake "OFF" that only buffered text in the UI is gone; the
  right pane renders `client.messages.stream(...)` because that is what runs.
- `allowed_callers` shows the effective caller for the model's tool version
  (`code_execution_20260120` on `web_search_20260318`, `direct` on
  `web_search_20250305`) and lists the other official value greyed as
  `API supported · not exposed by llm-anthropic`.
- Haiku honestly shows `web_search_20250305`, `direct`,
  `Dynamic filtering: Not supported` and `response_inclusion: Not available`.
- Every control reports one status: `Editable`,
  `API supported · runtime fixed`, `Unsupported by selected model`,
  `Unsupported by current tool version`, `Provider default` or
  `Fallback capability data`.
- Prompt cache TTL is still unsupported by `llm-anthropic`; the form shows only
  `Provider default: 5m`, with no editable control.

### Fixed after re-reviewing the previous step

- Removed `/api/preview`, `AnthropicTurn` and the hand-maintained
  `anthropic_kwargs()` renderer. They were a second request model that could
  drift from `build_kwargs()`: they hard-coded `web_search_20260318` for models
  that send `web_search_20250305`, ignored `system`, `effort` and
  `max_uses=0`, and used stale defaults. The renderer now takes a
  `build_kwargs()` result and nothing else.
- `prepare()` now also verifies the model id, `max_tokens` and prompt caching
  against the built request, not just the tool fields.

### Added on the conversation branch

- `llm_sdk_view/chat.py`: a `ChatSession` that runs turns through
  `llm.Conversation` and `llm-anthropic`, streaming text events out.
- `/api/chat` and `/api/chat/stream` (server-sent events).
- The right pane now renders `model.build_kwargs()` output instead of a second,
  hand-maintained rendering of the turn.
- Offline tests covering multi-turn history, streaming, missing-key handling
  and the four Anthropic behaviours, using a faked Anthropic transport.
