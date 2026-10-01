# AGENTS.md

This project is an LLM plugin and local web application.

## Development environment

Install the project with test dependencies:

```bash
pip install -e '.[test]'
```

Run tests:

```bash
pytest
```

Run lint:

```bash
ruff check .
```

The browser checks are part of `pytest` and skip themselves when no browser
is installed. Do not leave them skipped while changing the UI:

```bash
pip install -e '.[test,browser]'   # uses an installed Chrome if there is one
pytest -m browser                  # just the real-page checks
```

## Diagnosing UI defects

Three separate attempts to fix "clicking a topbar pill does nothing" failed,
and all three failed the same way: they searched the JavaScript, and the
JavaScript was correct. The handler ran, the menu was built and inserted, and
it reported `display: block`, `visibility: visible`, `z-index: 70`. An
ancestor's `overflow` was clipping it away. Two more clicks-that-do-nothing
were hiding behind that one. So:

- **Never conclude that a UI behaviour works, or is broken, by reading the
  source.** Open the page in a browser and measure it: `elementFromPoint`
  inside the element's own rectangle, `getComputedStyle` up the ancestor
  chain, `getBoundingClientRect`. `tests/test_topbar_in_a_browser.py` has the
  helpers and the harness.
- **A test of the form `assert "..." in static_page` protects the wording,
  not the behaviour.** It cannot see a containing block, a computed style, a
  hit test or an event that was never dispatched - which is exactly why this
  defect shipped past a green suite. Anything positional or interactive needs
  a browser check as well.
- **An overlay must never be a descendant of an element inside a scroll
  container.** `overflow-x: auto` computes `overflow-y` to `auto` too, so a
  horizontal scroller clips vertically as well. Menus, popovers and tooltips
  belong in `#pillLayer` or an equivalent top-level layer, placed from the
  anchor's rect by `anchorOverlay()`.
- **Never rebuild a container's children from a handler that can fire during
  a pointer gesture.** If the element that received `mousedown` is detached
  before `mouseup`, the browser dispatches no `click` at all and the control
  is simply dead. Update nodes in place and keep them across renders.
- **`justify-content: flex-end` on a scroller strands its own content.** The
  overflow goes past the start edge, where it is not scrollable; the row
  reports `scrollWidth === clientWidth` while items sit at negative x. Use
  `safe flex-end`.

## Project boundaries

- Keep the product small: chat on the left, equivalent provider SDK code on the right.
- Reuse LLM and provider plugins instead of rebuilding conversations, key storage, or provider adapters.
- Submit generally useful provider behavior upstream.
- Never make paid API calls in tests.
- Never store, print, serialize, or render API keys.
- Generate SDK code from the request `model.build_kwargs()` returns, never from a second template.
- Read model facts from the Anthropic Models API first; fall back to the versioned profile in `native_api_chat/models.json` only when it is unavailable, and always say which one is in use.
- Read unit prices from whoever publishes them, at runtime, never from a table committed here: Anthropic's pricing page (`native_api_chat/rates_page.py`) and OpenRouter's own model catalogue (`native_api_chat/rates_openrouter.py`), both background-fetched, disk-cached under `~/.cache/native-api-chat/`, never blocking. Never commit a rate table - a price change upstream must not need a commit here. Label a cached rate as cached, and treat an unfetched source, or a model that source does not cover, as "no estimate", never as a rate borrowed from a neighbour. Price a turn by its own provider's arithmetic, because the counters differ: Anthropic reports the *uncached* part of the prompt as `input_tokens` and bills cache writes and reads beside it, while OpenRouter reports the *whole* prompt with the cache read nested inside it - charging that prompt at the input rate and the read at its own rate bills the cached tokens twice.
- Keep model capability decisions in `native_api_chat/capabilities.py`, not in UI conditionals.
- Keep `thinking` and `effort` separate: they are separate API fields, and no control may express one through the other.
- Never offer a control the runtime cannot honour. Label it `API supported · runtime fixed`, `Unsupported by selected model` or `Unsupported by current tool version`, and refuse the value rather than sending something else.
- Never render an SDK call that does not happen: `llm-anthropic` always opens `client.messages.stream()`, so `messages.create()` must not appear.
- Never trim, summarise or compact the conversation. Refuse to send and tell the user to lower `max_tokens` or start a new conversation.
- Measure the context figure rather than guessing it, in this order: the API's own `count_tokens` for the request just built (`native_api_chat/token_count.py` - free, background, cached, never blocking), then the API's own `usage` from the last turn in the conversation, then a character estimate labelled as one. A character count cannot see the ~2,200 tokens Anthropic adds for the `web_search` stub, and must never be preferred over a number the provider reported.
- Do not add agents, RAG, memory, MCP, file upload, or additional providers without a separately accepted objective.

## Current Anthropic acceptance gates

1. `web_search_20260318`
2. Dynamic filtering remains enabled
3. `response_inclusion="excluded"`
4. Prompt caching is enabled

Read `docs/architecture.md` and `docs/upstream-contributions.md` before changing
the active implementation objective.
