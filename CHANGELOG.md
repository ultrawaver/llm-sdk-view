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

### Added on the SDK response and persistence step

- The right pane has two views of one turn: `Request` keeps rendering
  `model.build_kwargs()`, and `Response` shows what the SDK answered with -
  message id, resolved model, content blocks, final text, thinking, citations,
  server tool blocks, stop reason, input/output/cache tokens and web search
  requests - as formatted JSON plus a one-line usage summary.
- Added `TurnRecord` (`llm_sdk_view/records.py`): one object per turn holding
  the conversation id, turn id, user input, effective options, request kwargs,
  rendered code, response view, context and timestamp. The chat bubble, the
  Response pane and the stored row are all projections of it, so there is no
  second response model. The response is read off the finished Message, never
  reassembled from streamed text.
- Conversations are stored in LLM's own SQLite through `llm.logs.LogStore`: the
  same schema the `llm` CLI writes, using LLM's own ULIDs as conversation ids
  and its stored messages as the history a resumed conversation starts from.
  One sidecar table keeps the request and rendered code upstream has no column
  for; both writes happen in one transaction.
- Added `GET /api/conversations` and `GET /api/conversations/{id}`, plus a
  conversation sidebar with `+ New conversation`. Titles come from the first
  user message, trimmed at a word boundary; no model call is made to name a
  conversation.
- A turn is only recorded once its stream has finished, and a save that fails
  is reported as a failure: the completed turn said `saved: false` with the
  reason instead of pretending to be in history.

### Fixed after re-reviewing the previous step

- Removed `/api/preview`, `AnthropicTurn` and the hand-maintained
  `anthropic_kwargs()` renderer. They were a second request model that could
  drift from `build_kwargs()`: they hard-coded `web_search_20260318` for models
  that send `web_search_20250305`, ignored `system`, `effort` and
  `max_uses=0`, and used stale defaults. The renderer now takes a
  `build_kwargs()` result and nothing else.
- `prepare()` now also verifies the model id, `max_tokens` and prompt caching
  against the built request, not just the tool fields.

### Fixed after the first live smoke

Found by running the page against a real key and a real Models API response,
which the offline suite could not see:

- The Models API reports capabilities as `{"name": {"supported": bool}}`, not
  as a list of names. Reading presence instead of the flag made every model
  look adaptive, so Haiku was labelled adaptive, thinking OFF sent the
  unverified `{"type": "disabled"}`, and thinking ON was refused outright.
  `_thinking_mode_from_api()` and `_effort_levels_from_api()` now read the
  flag, so Haiku is `extended` again, OFF omits the field, and only effort
  levels the API marks supported are offered.
- The page read the shown `allowed_callers` value as a user choice and
  disabled Send whenever it was `direct`. For `web_search_20250305` the
  effective caller really is `direct`, so the default model could not send
  anything. The control is now disabled when it is runtime fixed, and only a
  value the user could actually select can block sending.
- `/api/preview` (POST) is back, but not as the old second request model: it
  runs the same `ChatSession.prepare()` and the same renderer, is never
  executed, and returns the request Send would build. The right pane refreshes
  on every form change, so the SDK call is readable before the call is paid
  for, and an illegal combination is refused while it is still free to fix.
  A preview never records a turn and never discards the conversation.
- Changing a form value on a live session left "use this model's default" as
  `None`, which the request builder read as "thinking off" and sent
  `{"type": "disabled"}` on a model that always thinks. Form changes now go
  through `ChatSession.update_options()`, which resolves and checks exactly
  like a new session.
- The test suite is now hermetic: every test starts with no key and an empty
  Models API cache, so a developer's own key or warm cache can no longer decide
  what the assertions see.

### Added on the conversation branch

- `llm_sdk_view/chat.py`: a `ChatSession` that runs turns through
  `llm.Conversation` and `llm-anthropic`, streaming text events out.
- `/api/chat` and `/api/chat/stream` (server-sent events).
- The right pane now renders `model.build_kwargs()` output instead of a second,
  hand-maintained rendering of the turn.
- Offline tests covering multi-turn history, streaming, missing-key handling
  and the four Anthropic behaviours, using a faked Anthropic transport.

### Changed on the UI overhaul step

No behaviour change: same endpoints, same payload, same request truth. Only
how the page looks and reads.

- The single-file page is split into `static/index.html`, `static/app.css`
  and `static/app.js`, served by a `/static/{name}` route that only serves
  stylesheets and scripts from the asset directory. `pyproject.toml`
  package data covers the new files.
- Light and dark themes via `prefers-color-scheme`, all colours as CSS custom
  properties; the page was dark-only with unstyled native controls before.
- The flat thirteen-control form is now a collapsible `Build your request`
  panel in the centre column: a one-line config summary when collapsed, five
  accordion groups (Model & limits, Thinking, Web search, Caching,
  Transport) with descriptions when expanded. Form and code stay visible
  together, so the knob-to-code feedback loop survives.
- The six control statuses render as badges (colour + icon + label) instead
  of bracketed text, with a worst-status roll-up badge per group.
- Boolean controls are Playground-style switches; the hidden select remains
  the only value the payload reads.
- The context meter is a bar with ok/warn/danger thresholds; the refusal
  state stays a refusal, never a trim.
- Inspector: segmented Request | Response control, syntax highlighting with
  line numbers (vendored Prism 1.29.0: core, python, json, line-numbers - no
  CDN, works offline), a copy button, a language chip, and a centred empty
  state per tab.
- Streaming shows a caret; a finished turn switches the inspector to the
  Response tab; bubbles carry turn numbers; Send disables into `Sending…`
  while a turn runs.
- Keyboard: ⌘Enter send, ⌘B sidebar, ⌘, parameters, ⌘1/⌘2 tabs.
- Tests that grepped `index.html` source now share a `static_page` fixture
  that concatenates the page's own assets (vendored code excluded); the
  assertions themselves are unchanged.

### Added on the Playground-parity pass

- The Request pane drops its explanatory comment. A request carries no
  commentary; why streaming is fixed is now only the greyed-out stream
  control's status.
- The rendered layout follows the official Playground: a value stays on one
  line when it fits (88 columns, the usual Python line length - the official
  sample inlines 58 columns and explodes 104, so any threshold between them
  reproduces it) and explodes when it does not. Short dicts and lists are
  therefore no longer spread over a dozen lines.
- The call ends the way the Playground ends it - `for text in
  stream.text_stream: print(text, end="", flush=True)` - instead of a lone
  `stream.get_final_message()`, which read like a one-shot fetch.
- Key order is still exactly the order `llm-anthropic` assembled the request
  in; nothing sorts or regroups it.
- The Response pane shows the official Raw shape: the Message the API returned,
  with `usage` re-attached (llm-anthropic pops it off while consuming the
  stream). Official keys in the official order; a key the API adds later is
  appended, never dropped, and a count it did not report stays absent rather
  than being zero-filled. Everything this app derived - `text`, `thinking`,
  `citations`, `summary`, `duration_ms`, `ttft_ms` and the hoisted
  `web_search_requests` - is out of the JSON and stays on the summary line and
  the bubble instead.

### The context figure is measured, not guessed

- The meter takes the provider's own numbers wherever it can, in this order:
  `API count` (the free `count_tokens` endpoint, asked about the exact request
  `prepare()` built), then `API usage` (the last turn's reported counts), then a
  character estimate that says it is one. New module
  `llm_sdk_view/token_count.py`: one background thread, an in-memory cache keyed
  by request signature, a 60-second cooldown after a failure, and no way to
  block or raise into a request path.
- Fixed: reopening a stored conversation showed a figure out by a factor of 47.
  A two-turn conversation the API had reported 2,265 input and 70 output tokens
  for showed `49 tokens`, because the baseline was a character count that never
  saw the history. The baseline now comes from the stored `usage`.
- Fixed: the estimate read every language as English. Measured against the
  API's own counter, Chinese is about one token per character rather than one
  per four, so a Chinese conversation was under-counted four-fold.
- The count is what sees the cost nothing local can: Anthropic expands this
  app's 70-character `web_search` stub into roughly 2,200 input tokens
  (measured 2026-09-27 - haiku-4-5 went 12 to 2,220, sonnet-5 12 to 2,806).
  A brand-new conversation with `你好` typed now reads 2,216 tokens, not 2.
- The refusal that guards the context window is computed from the same figure
  the meter shows, and names its source, so the two cannot disagree.
- The page looks again for a count that is still running - a bounded number of
  times - so a fallback figure becomes the API's own number when it lands.

### A bubble opens the pane that carries it

- Clicking a bubble now decides which pane opens: the user's bubble is the
  request that went out, the assistant's is the answer that came back. Before,
  both ends opened the Request, which left the Response pane unreachable from
  the conversation. A turn whose pane is open is marked, so the click has a
  visible result.
- Fixed: a turn sent in this page had no click handler at all - only turns
  loaded from history were wired, so the reply that had just arrived was the
  one turn whose response could not be opened from its bubble.
- Each bubble carries the turn's own time above it, in the computer's zone,
  the way a chat client shows it: time alone for today, day and time earlier
  this year, the full date beyond that, with the un-ellipsised local stamp in
  the tooltip. `Intl.DateTimeFormat` renders it, so the browser's locale
  decides whether that is `6:41 PM` or `18:41`.
- llm stores `datetime_utc`; the page converts instead of printing it. A stored
  stamp with no zone on it is UTC here too, never the browser's zone - which is
  how the sidebar used to misread it, printing UTC as if it were local time.
- A bubble with no stamp shows nothing rather than "now": a turn that has not
  been recorded has no clock, and inventing one would date a message to when it
  was looked at. The live turn's bubbles take the record's stamp over from the
  page's own send time the moment the record arrives, so the bubble and history
  agree.

### Settings moved to the topbar as pills

- The collapsible "Build your request" panel is gone. The request's settings
  are eight pills in the header - Model, Thinking, Effort, Max tokens, System,
  Web Search, Caching, Streaming - each opening a small menu anchored to the
  pill. The hidden form controls stay the single source of truth: every menu
  writes back into a control and fires its change event, so the payload, the
  validation and the preview read exactly what they always read.
- A pill the model or the runtime cannot honour is dashed and grey with the
  reason attached: Effort on Haiku (the model has no effort parameter),
  Thinking on always-on models (Adaptive, fixed), Streaming (llm-anthropic
  always streams). None of them is a fake control.
- The effort menu lists only the model's levels, low to max: no synthetic
  Default row. The level Anthropic documents as the default carries a
  `default` tag and is what a fresh form selects; picking it stores "default",
  so no effort field goes on the request at all.
- The thinking budget is not shown anywhere as a setting: llm-anthropic
  hard-codes it. What the Max tokens menu does enforce is the legal floor it
  creates - with thinking enabled the minimum is the budget plus one (1025 on
  Haiku), and the menu's input clamps to it.
- History shows each turn's settings on hover: one card shared with the
  pills, read from the turn's stored `effective_options`, with the fields
  that differ from the previous turn dotted amber. A turn with no sidecar
  row (a conversation written elsewhere) says its settings were not recorded
  rather than guessing.
- A dashed divider between two turns names the settings that changed there -
  `thinking off → on · max_tokens 1024 → 16384` - so a conversation shows
  where its own configuration moved.
- Changing anything that is part of the cache prefix (model, system, the web
  search tool's shape) warns above the composer that the next turn will not
  reuse the cached prefix. max_tokens, effort and thinking are not part of
  the prefix and never warn; with caching off the message is "won't read or
  write", not "invalidated".
- Keyboard: ⌘, is retired with the panel it toggled.

### Added on the hover-status pass

- A control status is a reason, not a reading: "runtime fixed" and
  "Unsupported by selected model" no longer sit in the row. Each is a 13px
  dot in the colour the badge had, and the whole sentence - status plus the
  note that explains it - appears on hover in one fixed card (#whyTip), so
  the pills scroller cannot clip it. One delegated mouseover pair covers
  every dot, including ones rendered later; keyboard focus asks for it too.
- A fixed or unsupported pill carries its reason on the whole pill
  (data-tip), not on pill.title, and opens nothing as before.
- Bubble clicks are answered by one listener on #messages reading the pane
  off the bubble (data-pane), not by a handler per bubble, and a click
  cancels a preview the last keystroke left queued - a dead-looking click
  and a pane that reverted to the draft are both structurally impossible
  now rather than merely unreproducible.

### Added on the "the click does nothing" fix

- The page's own css and js are served with a version in the URL
  (`?v=<sha1(mtime, size)>`), `/` is `no-store`, and assets carry
  `no-cache` + ETag. None of them had a cache header at all, so a browser
  could keep a file this project had already replaced - which is how a fix
  could ship twice and appear neither time.
- A click on the turn already shown on the right changes nothing else on
  screen, so the bubble now acknowledges it (a 300ms ring, `.msg.hit`). The
  selected assistant bubble gets the accent background too; before, the only
  sign of selection was a 1px border recolour.
- The settings card is `pointer-events: none`: it describes a bubble and can
  be drawn over one, and a read-only card has no business being a click
  target.
- `wireBubble()` writes the turn it was given. It used to be left to the
  1-based number `addMessage` was handed, so the pane a bubble opens and the
  turn it carried were written in two different places.

### Fixed on the grey pill step

- A fixed or unsupported pill (Effort on a model without the parameter,
  Streaming always) opens no menu by design, but its click was met with
  silence: no handler at all, so the one control that most needs to explain
  itself looked simply dead. The click is now answered - unmistakably. The
  first answer (re-show the reason card plus a background wash) still read
  as dead, because hover had already put that exact card on screen and the
  wash was near-invisible: freeze-framed screenshots of the click showed a
  delta too small to perceive. The answer is now visibly NEW: the pill
  shakes "no" (motion survives the scroller's shadow clipping and the human
  eye) and the reason card pops back in an answered state - accent border,
  full-strength text - that hover alone never gets.
- `flash()` outlives the animation it starts: it used to remove the class
  at 320ms, cutting a longer animation mid-frame.

### Added on the stale build step

- `GET /api/version` names the build the server would serve right now. The
  page reads its own build off its script URL, stamps it in the sidebar
  ("build 9e152046"), and whenever the tab regains focus it compares the
  two. A mismatch raises a banner across the top - "This tab runs build X
  but the server is on Y - click to reload" - because a tab that kept
  yesterday's js reports bugs that no longer exist; one click bug was
  reported three times before the page learned to say this itself.

### Added on the conversation title step

- The bar above the chat names the conversation the chat belongs to. An
  unsaved conversation says "New conversation" instead of inventing a title;
  the first saved turn's name arrives with the refreshed sidebar list.
- The pencil next to the title renames the conversation in place (Enter or
  leaving the field keeps the new name, Escape keeps the old, an empty name
  is refused). The rename is one write to llm's own `threads` row through
  `POST /api/conversations/{id}/name`, so the sidebar, the title bar and
  `llm logs` all read the same string.

### Fixed: the topbar pills opened nothing

Three independent defects wore the same symptom - a pill that answers a click
with nothing - and all three were in the layout, which is why three passes
through the js found none of them. Measured in a real browser, not read:

- **A menu was a child of its pill, and the pill row clips.** `#settingsPills`
  is a horizontal scroller, and `overflow-x: auto` computes `overflow-y` to
  `auto` as well, so the 28px-tall row clipped a menu that begins 4px below
  it. Every js-level check passed - the handler ran, `stopPropagation` was
  right, the node existed with `display: block`, `visibility: visible`,
  `z-index: 70` - and the menu was painted nowhere. The page already knew
  this: `#whyTip` is fixed-position with a comment saying the pills scroller
  would clip an absolutely positioned child, written directly above the rule
  that positioned the menus absolutely inside one. Menus now mount in a
  `#pillLayer` overlay and are placed from the pill's rect by the same
  routine that places the status card.
- **The row was rebuilt under the user's own hand.** The pills re-render on
  every control change, and the row was emptied and recreated each time. So
  pressing a pill while a menu field held focus blurred the field, the field
  committed, the commit re-rendered the row, and the element the press landed
  on was detached before the button came back up - the browser then dispatched
  no click at all. Editing Max uses, which lives inside the Web Search menu,
  destroyed that menu for the same reason. The eight buttons now outlive every
  render and only their contents change; an open menu is re-anchored rather
  than discarded, and never rebuilt, so a field being typed into survives.
- **Three pills were unreachable in a narrow window.** Found by the first run
  of the new browser checks, not by a person. `justify-content: flex-end` on a
  scroller sends the overflow past the *start* edge, and overflow in that
  direction is not scrollable: at 900px the row reported `scrollWidth ===
  clientWidth` while Model, Thinking and Effort sat at negative x. Not
  clipped, not scrollable to - gone. `justify-content: safe flex-end` falls
  back to start alignment the moment the pills stop fitting, so the row
  scrolls and every pill stays right-aligned when there is room.
- `.pill > * { pointer-events: none }`: a pill's contents are rewritten on
  every render, so a press that landed on the label rather than the button
  would still be left holding a detached node.
- A menu is mounted before it is built. `focus()` on a field inside a detached
  node does nothing, so the Max tokens menu opened with no caret and no
  selection, and typing appended to the ceiling instead of replacing it
  (16384 + "4096" = "163844096").
- A pill that opens no menu now shows `cursor: help` rather than `default`: it
  opens nothing, but it does answer a click with the reason it is grey.
- Pills carry `aria-haspopup` and `aria-expanded`.
### Added: the page is now tested in a browser

The existing topbar suite is string matching over the concatenated assets,
and it passed throughout all three defects. It cannot see a containing block,
a computed style, a hit test, or an event the browser never dispatched -
which is the whole reason they shipped and were then looked for in the wrong
file three times.

- `tests/test_topbar_in_a_browser.py` runs the real application on a real
  port and drives the real page: whether a menu is the thing painted at its
  own rectangle, whether any ancestor clips it, whether every pill is
  reachable at 900px as well as 1500px, and whether pressing one pill while
  another's field has focus both commits the field and opens the pill. It
  found the third defect above on its first run.
- The checks are marked `browser`, run as part of plain `pytest`, and skip
  themselves when no browser is installed. `pip install -e '.[test,browser]'`
  enables them; Playwright drives an installed Google Chrome, so there is
  usually nothing to download.
- CI sets `LLM_SDK_VIEW_REQUIRE_BROWSER=1`, which turns "no browser" from a
  skip into a failure. A check that quietly skips is a check that is not
  running, and that is the state this suite was already in.
- The source-level regression tests now read CSS declarations rather than
  searching for a string, because the defect was a correct declaration
  (`position: absolute`) in the wrong containing block. Each was verified by
  reintroducing the defect it names and confirming it fails.
- `AGENTS.md` carries the five rules this cost: measure UI behaviour in a
  browser instead of reading it; a string-matching test protects wording, not
  behaviour; an overlay is never a descendant of anything inside a scroller;
  never rebuild children from a handler that can fire mid-gesture; and
  `justify-content: flex-end` on a scroller strands its own content.

### Added on the reading-experience and conversation-state step

- A turn shows its reasoning as it arrives: the `reasoning` events stream into a
  strip above the answer with a live clock, which folds itself away when the
  first text lands and reports `Thought for 12.4s · 431 words`. A turn read back
  from storage names no duration, because the stored row carries none - a clock
  added later would be a number about the file, not about the answer.
- The finished answer renders as Markdown through `safeMarkdown()`, which builds
  DOM nodes and never assigns model output to `innerHTML`: headings, lists, bold
  and italic, inline code, fenced blocks highlighted by the vendored Prism with
  a copy button, and tables. The official Playground renders a table, so a table
  in the answer is not a surprise here either.
- The chat follows the stream only while it is already at the bottom. Scrolling
  up to read detaches it and offers a `↓ New messages` pill back to the end.
  Only a *decrease* in `scrollTop` unpins follow-mode, because a growing document
  moves `scrollTop` up without the user touching anything; reading that as intent
  unpinned the page several times a second. Reduced motion turns the smooth ride
  off without turning the following off: `scrollChatToBottom()` asks for `smooth`
  only when `prefers-reduced-motion` has not asked for less.
- Fixed: every assistant bubble carried two blank lines at the top and one at
  the bottom. The regions that are `display: flex` beat the browser's own
  `[hidden]` rule - an author style wins over the UA sheet - so a region that was
  hidden still took its gap. Each hidden region now states `display: none`
  itself. A screenshot showed it; reading the js had not, in three passes.
- A draft belongs to the conversation it was typed in: switching away and back
  restores what was left in the composer, and sending clears only the draft it
  sent.
- Opening an older conversation restores the settings that conversation last
  used - but only the values the selected model still offers. A value the model
  does not have is left at the model's own default rather than written back as a
  setting the request would then have to refuse.
- Changing a setting above the composer says what it costs. A field that is part
  of the cache prefix - model, system, the web search tool's shape - warns that
  the next turn will not reuse the cached prefix; every other field says only
  that the next turn is a new request. "Cache invalidated" about `max_tokens`
  would have been a lie with a plausible sound to it.
- The composer is 132px tall instead of one line high.
- The Model menu lists one model per series, newest first, and names the count of
  the earlier ones it is not showing. The rule is derived, not maintained:
  `model_series.py` reads a family and a version out of an id - both the current
  `claude-<family>-<major>[-<minor>]` form and the older
  `claude-<major>[-<minor>]-<family>` one, with date suffixes and `-latest`
  stripped - groups by family and keeps the newest version in each. A version the
  rule cannot parse is grouped alone and so is never hidden: the fallback for
  "we do not know what this is" is to show it.
- A superseded model is still selectable and still sends. It is marked `legacy`
  in the menu and answers as before, because a conversation pinned to it has to
  keep working; only a model the installed plugin cannot resolve at all is
  refused.
- An unknown model no longer raises in `capabilities_for()`. It reports what it
  cannot know - `Provider default`, no verified flag - instead of taking the page
  down with it, which is what the previous `entry[...]` lookups would have done
  for any id the profile had not seen.

### The cache window is counted down, and a split can no longer hide a write

Two independent things were wrong about the prompt-cache figures, and both were
visible on one screen at the same time.

- Fixed: a turn that used web search showed `Cache write 0` while its own response
  JSON said `cache_creation_input_tokens: 8940`. The API had returned a
  self-contradicting usage - a real 8,940-token write at the top level and
  `cache_creation: {ephemeral_5m: 0, ephemeral_1h: 0}` underneath.
  `pricing.py` trusted the nested split whenever it was present, so two zeros
  covered the total: the write was billed at the uncached rate and the turn was
  reported at $0.0137 instead of $0.0249. The top-level total is now the
  authoritative fact and the split only carves the 1-hour part out of it, which is
  the one thing the split is for. Cost is recomputed when it is read, so every
  stored turn corrected itself on the next load rather than needing a migration.
- The cost bar counts the prompt-cache window down. The moment a turn reports a
  cache read or write, a badge shows the time left - green above a minute, amber
  below it, red and bold with a slow pulse under thirty seconds, grey
  `cache expired` at zero - with the long explanation ("send now - after 0:24 the
  prefix is written again at 1.25× the input price") in the cost popover.
  `prefers-reduced-motion` removes the pulse; the red is the information and the
  pulse was only emphasis.
- The countdown starts when the **answer landed**, which is not what Anthropic
  documents. The prompt-caching page says the lifetime is measured "from the start
  of the request that writes or reads the cache entry, not from the end of its
  response", so a long streaming answer eats its own window - and with that anchor
  a cache the model was plainly still reading showed as nearly expired while it
  was still generating. This account's own usage counters disagree with the
  documented rule: in a 13-turn conversation, turns starting 335.6s and 338.9s
  after the previous request started still read the cache (196,792 and 242,523
  tokens), past the 300s the docs describe, and both were long answers (167s,
  163s) whose responses had landed only 168.2s and 175.7s earlier. Twelve reads in
  that conversation, and only "the answer's end" explains all twelve. The badge
  carries its basis in its tooltip, because a countdown that will not say where it
  starts cannot be checked. The evidence is a lower bound - no miss was ever
  observed, so the true deadline is still unproven - but the documented one is
  ruled out.
- The countdown stops in exactly two places: at zero, and when Send is pressed.
  Sending freezes the badge at the margin the send had left and says so; the new
  turn's record clears the freeze and counts five minutes from that answer's end.
  A send that failed or had no key unfreezes too - no request, no new cache.
- Fixed: the countdown read a stored stamp with `Date.parse`, which reads a
  zone-less UTC stamp as local time and moves the anchor by the whole offset -
  eight hours here, so a live cache read as `Expired 8:00 ago`. It uses the same
  `momentOf()` the rest of the page uses.
- Fixed: an old conversation said `Expired 4320:00 ago`, which is a duration, not
  information. Past an hour the note counts in hours instead.
- The countdown had no behavioural test at all: one assertion on
  `'id="cacheState"' in static_page` protected the id, and the faked transport
  produced no cache tokens, so nothing ever walked from a usage record to a badge -
  the exact trap `AGENTS.md` names. There are now eighteen checks across
  `tests/test_cache_countdown.py` and `tests/test_cache_countdown_in_a_browser.py`,
  the latter measuring the badge's computed colour, weight and animation in every
  state in a real page, a frozen badge that does not move for two seconds, and a
  page under `prefers-reduced-motion` that does not pulse.

### Each conversation can leave the app

- Added `llm_sdk_view/export_md.py` and `GET /api/conversations/{id}/export.md`:
  the whole conversation as one Markdown document - title, model, turn count,
  total cost, the system prompt, then each turn's user message, thinking, answer,
  sources and a one-line figure strip (latency, in/out tokens, cache activity,
  cost). It is built from the same stored records the panes read, so what leaves
  the app is what the app showed, and it is readable by another model without
  drowning it in provider JSON. An unknown id is a 404 that writes nothing.
- The download name keeps the conversation's own title, CJK included, through
  RFC 5987's `filename*`. Fixed: the plain-ASCII fallback was the title with every
  non-ASCII byte dropped, so a Chinese title downloaded as ` -2026-09-29.md`.
  When nothing but digits and separators survives, the fallback now names the file
  for what it is (`conversation-2026-09-29.md`) instead of shipping a mystery slug.
- Each conversation in the sidebar carries a `…` on hover: **Export Markdown**
  downloads the document, **Delete** removes the conversation. Delete is two
  clicks on purpose - the first turns the item into its own confirmation - so a
  misclick cannot cost a conversation, and deleting the conversation on screen
  returns the page to a new conversation.
- Added `DELETE /api/conversations/{id}` and `store.delete_conversation()`: the
  thread, its turns and the sidecar row go in one transaction, in foreign-key
  order. `turns` is also referenced by `turn_fragments`, `turn_search`,
  `turn_tools` and `tool_instantiations`, and llm ships no delete API of its own,
  so the cascade is written out here rather than assumed. llm's
  content-addressed `messages` rows are shared by design and stay, exactly as
  they would after `llm logs` pruning. An unknown id is a 404: a delete that
  found nothing must not report itself as a delete that happened.
- The menu mounts in the top-level overlay layer rather than inside the sidebar's
  scroller - the rule the topbar pills had already paid for.

### Fixed: a reopened conversation reported a change nobody had made

Switching back to an old conversation showed `effort differ from the last turn`
above the composer and `High` in the effort pill, for a conversation that had
been sent at `xhigh` thirteen turns running. The user had changed nothing.

- `applyStoredOptions()` ended by calling `applyEffortState()`, which rebuilds
  the effort `<option>` list from the model's capabilities. Replacing a
  select's options clears its selection, and the guard that followed turned a
  cleared value into `default` - so the level `applyStoredOptions()` had just
  restored was wiped by the next line, and `updateCacheWarn()` then compared
  that `default` against the turn's stored `xhigh` and reported the difference
  as the user's own doing. The restore had been written before the rebuild and
  destroyed by it.
- The rebuild now owns the decision. `applyEffortState(chosen)` reads the level
  the control must end up holding *before* the list is replaced, keeps it when
  the rebuilt list still offers it and thinking still allows it, and falls back
  to the provider's own level when it does not - which is the single case the
  old blanket clearing was ever there for. `apply()` asks for `default`
  explicitly, so changing the model still starts at the provider's own level;
  the restore path hands the rebuild its stored level instead of writing the
  select first.
- The same clearing was dropping a legal level whenever the thinking mode
  changed: `high` with thinking off was reset to `default` even though nothing
  about thinking makes `high` unsendable.
- `updateCacheWarn()` now clears its sentence when it hides itself. A hidden
  node still holding a withdrawn claim about the current form reads as a live
  warning to anything that inspects the DOM, which is exactly how the stale
  text announced itself.
- The test that should have caught this asserted `restored["effort"]`, which
  `'default'` satisfies - and it ran on a page whose model was Haiku, whose
  only effort level *is* `default`, so it had nothing to lose in the first
  place. It now asserts the restored level by value on a model that has levels,
  and two browser checks were added: the reported symptom (a conversation that
  used `xhigh` reopens with the control holding it, the warning hidden and
  empty) and the rebuild invariant (a level the new list still offers survives;
  one this model cannot send does not). All three fail against the previous
  implementation. The string assertion that pinned the old guard's literal text
  - a page cannot be asked what a select ends up holding - is replaced by one
  that pins the wiring, with the behaviour measured in the browser.



### The footer totals the conversation, and a turn's receipt is on its answer

- The cost footer described whichever turn was selected, so the conversation's
  own total existed only as a row inside the popover and moved every time a
  bubble was clicked. It is now the conversation's: `conversationTotals()`
  sums every turn the session holds, and the bar reads `2 turns` in front of
  the total, the cache-hit rate and the countdown.
  `selectedCost()` and `sessionCost()` are gone; selecting a turn changes the
  right pane and nothing else.
- The sum is built from the turns' own receipt *lines*, not re-derived from
  their totals, so the answer's receipt and the footer's cannot disagree about
  what a cache read cost. Both scopes are drawn by one `costReceiptHtml()`, and
  a total is always the sum of the rows printed above it.
- The turns it cannot price are counted out loud. A model the pricing page does
  not cover has no estimate, and the old session row silently dropped those
  turns from both the money and the count. The popover now says `1 of 2 turns
  have no estimate and are not included above`, and names how many turns
  reported no cache counters at all.
- Hovering an answer now shows that turn's own receipt - the composition bar,
  the Input/Output/Tools rows with unit rates, the total, the hit-rate formula
  and the saving against no cache - instead of a second copy of the request's
  settings. The settings card stays on the question bubble, where the settings
  came from, and the receipt wears the wider card its four columns need.
- Only the answer bubble gets a cost card; both cards are the same fixed,
  `pointer-events: none` overlay, so neither can swallow the click that picks
  the turn it describes, and the build function runs at hover time so a card
  always describes the record it is over.
- The bar carries only what it can show whole, and what that is was measured
  rather than guessed. The conversation's token totals left it - they were the
  widest thing there, and a half-drawn `5.2k in · 81…` claims a precision the
  bar does not have - and so did the tool-call count, which would have put the
  same squeeze back the first time a conversation searched. Both are in the
  popover: the tokens as the receipt's own rows, the searches on its header
  line beside the model.
- Making one cell elastic was tried first and only moved the defect along:
  with the hit rate as the shrinkable cell, at 1100px it was drawn 82px wide
  for 104px of text, at 1024px 44px, and at 960px all twelve of them. So the
  rule is now "whole or absent": the bar's wrapper is a query container
  (`container-type: inline-size`, so it follows the pane and not the window),
  and below the width at which every cell fits at its natural width the cache
  hit rate is dropped rather than drawn in half. The one figure the bar can do
  without is the one that goes - it is derived from the receipt's own lines,
  and the popover prints that arithmetic in full.
- Tests: `tests/test_cost_scopes.py` and `tests/test_cost_scopes_in_a_browser.py`
  (nine browser checks - the footer equals the sum of the turns, clicking a
  bubble does not move it, the popover totals the conversation and still floats
  above the bar, the answer's hover card is its receipt while the question's is
  its settings, an unpriced turn is named rather than dropped, an unreachable
  pricing page says so, a conversation's searches are counted in the details,
  and the bar keeps only
  what fits: no cell truncated, no overflow and the button still inside it at
  1280, 1100, 1024 and 960px, with the hit rate present at the first and gone
  at the last). They fail against the previous implementation - removing the
  drop rule alone fails the bar check at 1100px, where the hit rate stays rigid
  and the scope label is what gets truncated instead. The test that pinned
  "typing repaints the cost footer" now pins the opposite, because the footer
  no longer describes the selection.
- Fixed on the way past: "there are no rates" could never be shown. The page
  read `state.rates.state` and `state.rates.error`, while `/api/rates` sends
  `rates_state` and `rates_error` - so the reason for a missing estimate always
  fell through to the vaguer "no cost estimate" instead of naming the
  unreachable pricing page.


### Renamed to Native API Chat

- `llm-sdk-view` described the right-hand pane, not the product. The name is
  now `native-api-chat`: *native* because the request carries the provider's
  own API rather than a lowest-common-denominator compatibility layer, *API*
  because every parameter, request, response and usage figure is visible, and
  *chat* because this is a conversation you keep, not a one-shot inspector.
  LLM stays in the description ("built on LLM and llm-anthropic") instead of
  being pushed into the brand.
- Everything mechanical moved with it: the package is `native_api_chat`, the
  plugin command is `llm native-chat`, the standalone script is
  `native-api-chat`, the disk cache is `~/.cache/native-api-chat/`, and the
  three environment variables are `NATIVE_API_CHAT_CACHE_DIR`,
  `NATIVE_API_CHAT_LOGS_DB` and `NATIVE_API_CHAT_REQUIRE_BROWSER`.
- The sidecar table `llm_sdk_view_turns` becomes `native_api_chat_turns`, and
  an existing one is carried over by `ALTER TABLE ... RENAME TO` rather than
  left behind for nothing to read. The rows keep their primary key and their
  foreign key into `turns`, and their `source` values are **not** rewritten: a
  turn written by llm-sdk-view was written by llm-sdk-view, and that column
  exists to say so. `tests/test_conversations.py` pins the carry-over, that
  the legacy table is gone afterwards, and that reopening the database is a
  no-op.
- Entries above this one keep the paths and names the files really had at the
  time. Rewriting them would make the log claim a history it does not have.
