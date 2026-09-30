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

### What reaches the dropdown

The API supplies every model, but the dropdown offers one model per series,
newest first: `llm_sdk_view/model_series.py` reads the family and version out
of each id, so when a new member of an existing series arrives it replaces the
old one on the next read, with nothing hand-maintained here. Two further gates
decide what is offered: the installed plugin must be able to resolve the id
(otherwise this project cannot send the request), and what the narrowing left
out is reported back to the page rather than silently dropped. A stored
conversation whose model has since been superseded adds it back, marked
`legacy` — superseded means no longer listed, not unusable.

Only narrow rules the Models API does not expose are hard-coded: default effort
and default thinking state per model, whether thinking can be switched off and
how, which effort levels Anthropic rejects when thinking is disabled, and how
dynamic filtering maps onto `allowed_callers`. Nothing in the UI branches on a
model id.

Thinking and effort are separate controls, because they are separate API
fields: `thinking` produces the `thinking` block, `effort` produces
`output_config.effort`. The budget inside an extended-thinking block is a
runtime fact, not a setting: `llm-anthropic` hard-codes it, so it appears
nowhere as a control - what the UI enforces instead is the floor it puts under
`max_tokens` (the budget plus one while thinking is enabled).

## The settings surface

The request's settings are eight pills in the topbar (Model, Thinking, Effort,
Max tokens, System, Web Search, Caching, Streaming). The pre-pills form
controls survive as a hidden value store, and that store is the only one:
every pill menu writes back into a control and fires its change event, so the
payload, the validation and the preview read exactly what they read before the
pills existed. A pill is therefore never able to drift from what Send builds.

The row itself has two structural rules, both paid for by the same defect -
pills whose menus opened invisibly, and pills that ignored a click outright:

- **A menu is never a descendant of its pill.** The row is a horizontal
  scroller, and `overflow-x: auto` computes `overflow-y` to `auto` too, so a
  28px-tall row clips everything hanging below it. A menu positioned inside a
  pill is in the DOM, reports `display: block`, and is painted nowhere. Menus
  and the status card are both fixed-position overlays in `#pillLayer`, placed
  from their pill's own rect by one routine, `anchorOverlay`.
- **The row is updated in place, never rebuilt.** It re-renders whenever any
  control changes, including a control one of its own menus just wrote.
  Replacing the nodes did that in the middle of the user's gesture: the press
  that blurs a menu field commits it, the commit re-renders, and the node the
  press landed on is detached before the button comes back up - at which point
  the browser dispatches no click at all. The eight buttons outlive every
  render, only their contents change, and `.pill > * { pointer-events: none }`
  keeps the button itself the thing the pointer lands on.

The row aligns right with `justify-content: safe flex-end`, not `flex-end`:
plain end alignment pushes the overflow past the start edge, where it is not
scrollable, and three pills sat at negative x in a 900px window with the row
reporting no scrollable overflow at all.

Because none of this is visible in the source - a clipped element keeps its
box, its computed style and its place in the DOM - the topbar is the one
surface with tests that drive a real browser
(`tests/test_topbar_in_a_browser.py`): a hit test inside the element's own
rectangle, a walk up the ancestor chain for `overflow`, and clicks that have
to reach a handler. They run as part of `pytest` and skip without a browser;
CI sets `LLM_SDK_VIEW_REQUIRE_BROWSER=1` so a skip there is a failure.

Two display rules fall out of the honesty rules:

- A pill the model or the runtime cannot honour is dashed and grey with the
  reason attached (Effort on a model with no effort parameter, Thinking on an
  always-on model, Streaming). It is never a clickable fake.
- The effort menu lists only the model's real levels; the level Anthropic
  documents as the default carries a `default` tag and stores `default`, so
  picking it sends no effort field at all. Absence on the wire means the
  provider default applied - never that a value was lost (see
  `docs/per-turn-settings.md` §1b).
- The effort value is decided where its list is rebuilt, never written onto
  the control from somewhere else. Replacing a select's options clears its
  selection, so a value written first and rebuilt after is a value lost, and
  the composer would then report the loss as a change the user had made. The
  rebuild keeps a level the new list still offers and thinking still allows,
  and falls back to the provider's own level for one this model cannot send.

History shows each turn's settings through the same honesty lens: a hover card
on the request bubble read from the turn's stored `effective_options` (a turn
with no sidecar row says "not recorded"), a dashed divider between turns whose
stored settings differ, and a composer hint naming what the next turn changes
about the last one. The hint has two classes, because only one of them can cost
something: a field inside the cache prefix (model, system, the web-search
tool's shape) says the cached prefix will not be reused, while `max_tokens`,
`effort`, `thinking` and `allowed_callers` are named as a difference and never
as a cache miss. Reopening a conversation puts its own stored settings back on
the form first, so the hint stays silent about differences the user did not
cause.

## Two cost scopes, one receipt

A cost figure belongs either to a turn or to the conversation, and the two are
shown in different places so that neither has to guess which it is:

- **The footer is the conversation's.** `conversationTotals()` sums every turn
  the session holds, and the bar leads with the turn count for exactly that
  reason. Clicking a bubble decides what the right pane shows and moves
  nothing here.
- **A turn's receipt is on its answer.** Hovering the assistant bubble shows
  that turn's own composition bar, its Input/Output/Tools rows with unit
  rates, its total and its saving against no cache. Hovering the request
  bubble shows the settings that went out - the same card for both was two
  copies of one fact and left the answer with nowhere to put its figures.

Both scopes are drawn by one `costReceiptHtml()`, and the conversation's total
is built from the turns' receipt *lines* rather than re-derived from their
totals: one definition of a line, so the two cannot disagree about what a cache
read cost, and a total is always the sum of the rows printed above it.

Two honesty rules survive the summing. A turn whose model the pricing page does
not cover gets no estimate, so the popover counts those turns out loud ("1 of 2
turns have no estimate and are not included above") rather than reporting a
total quietly smaller than the conversation; and a turn that reported no cache
counters is named for the same reason.

The bar carries only figures it can show whole, and that is a rule rather than
a hope. The token totals left it: they were the widest thing there, so they
were the first to be cut when the pane narrowed, and a half-drawn
`5.2k in · 81…` claims a precision the bar does not have. The tool-call count
went with them, for the same reason and into the same place - the popover,
where the tokens are the receipt's own rows and the searches are on its header
line. What is left is the scope (`2 turns`), the total, the cache hit rate and
the countdown.

Making one cell elastic is what this replaced, and it only moved the defect
along: with the hit rate as the shrinkable cell it was drawn 82px wide for
104px of text at 1100px, 44px at 1024px and 12px at 960px. Now the bar's
wrapper is a query container (`container-type: inline-size`, so it follows the
pane rather than the window), and below the width at which every cell fits at
its natural width (418px, measured) the hit rate is dropped instead of halved.
It is the one figure the bar can do without - it is derived from the receipt's
own lines, which print the arithmetic in full.

## Control status

Every form control reports one status, so a greyed-out value always says why:

| Status | Meaning |
|---|---|
| `Editable` | the form sets it and the request carries it |
| `API supported · runtime fixed` | the API has it, `llm-anthropic` does not expose it |
| `Unsupported by selected model` | the API has it, this model does not |
| `Unsupported by current tool version` | the API has it, this tool version does not |
| `Provider default` | nothing is sent; the API decides |
| `Fallback capability data` | the number came from the offline profile, not the Models API |

## Context

The context meter shows `tokens / limit`, a percentage, and the source of both
numbers. Three sources, in the order they are trusted:

1. `API count` — `client.messages.count_tokens()`, which Anthropic documents as
   free, against the exact request `prepare()` built
   (`llm_sdk_view/token_count.py`: background thread, in-memory cache, never
   blocking, never raising into a request path).
2. `API usage` — the provider's own counts from the last turn in the
   conversation, which is why reopening a two-turn conversation shows its real
   weight instead of a character count. `API usage + estimated draft` is that
   figure plus the message being typed.
3. `estimated` — a character count, and only that. Measured against the API's
   counter: Latin text is ~4 characters per token, CJK ~1 character per token.

The counter is worth the round trip because the largest cost in this app's
requests is invisible locally: Anthropic expands the 70-character `web_search`
stub into roughly 2,200 input tokens (measured 2026-09-27: haiku-4-5 went from
12 tokens to 2,220 when the stub was added, sonnet-5 from 12 to 2,806). Nothing
in the request body shows them.

When the figure plus the reserved output cannot fit, sending is refused with the
two ways out — lower `max_tokens` or start a new conversation. The same figure
feeds that check, so the meter and the refusal never disagree. Nothing compacts,
summarises or silently trims the history.

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

## One record per turn

Every view of a turn is a projection of one object, `llm_sdk_view/records.py`:

```text
ChatOptions          -> what the form asked for
build_kwargs()       -> the request that was built
render_kwargs()      -> the code the right pane showed
finished Message     -> the response the SDK answered with
        |
        v
    TurnRecord  ->  chat bubble, Response pane, rows in SQLite
```

The Response is read off the Message the provider finished with, never
reassembled from streamed text: thinking blocks, citations, tool results and
usage counters only exist there. Nothing that surface shows is derived from
the bubble, so the two cannot disagree.

The record is also what a bubble is a handle on. Both ends of a turn select
the same record and differ only in which pane eats the click: the user's
bubble opens the Request, the assistant's opens the Response. The time above
each bubble is that record's own stamp — llm's `datetime_utc` for the turn —
converted to the computer's zone in the page, with no second clock taken
anywhere along the way. A turn with no stamp shows no time rather than the
time it was looked at.

## History lives in llm's database

`llm_sdk_view/store.py` writes through `llm.logs.LogStore` — llm's own
read/write API for its SQLite schema and the same object its CLI builds on.
There is no second conversation or response table, and no subprocess call to
the CLI:

- `threads` and `turns` carry the conversation id, timings, usage and provider
  response, so `llm logs` sees these conversations unchanged;
- conversation ids are llm's own ULIDs, reused rather than reinvented;
- resuming passes the stored messages back in as `Conversation.loaded_messages`
  so the original content blocks - not a transcript - become the history;

One narrow sidecar table, `llm_sdk_view_turns`, holds what upstream has no
column for: the request this app built and the code it rendered. Both writes
happen in one transaction; if the sidecar cannot be written the turn is not
saved and the page is told so. Provenance rides in that table's own `source`
column instead of being smuggled into an upstream one.

## Upstream boundary

- Provider behavior useful to every `llm-anthropic` user belongs upstream.
- Provider-neutral LLM hooks useful to multiple plugins belong in LLM core.
- UI, code presentation, and product statistics belong here.
