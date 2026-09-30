# UI/UX improvement plan

Status: **P0 mostly landed, P1 partial** (updated 2026-09-27, branch
`conversation-vertical-slice`). The plan itself below is unchanged.

Landed: assets split into `index.html` + `app.css` + `app.js` with a vendored
Prism, light and dark themes both honoured, one status-badge language for the
six statuses, the context meter bar with its refusal messages, syntax
highlighting with a copy button, turn numbers on bubbles, the keyboard map,
and the floating cost/cache footer (a receipt with unit rates, prompt-cache
hit rate and TTL countdown). The footer totals the **conversation**; a
**turn's** own receipt is the card hovering its answer puts up, and hovering
the request bubble shows the settings that went out. Each bubble is also a
link into the pane it carries — the user's bubble to the Request, the
assistant's to the Response — with the turn's own time above it in the
computer's zone (job 4, "turn↔inspector association").

Landed 2026-09-27 (second pass, decision D in §5): the settings moved from
the collapsible panel to **eight pills in the topbar**, each opening an
anchored menu over the same hidden value store; the effort menu lists only
real levels with the documented default tagged; the thinking budget appears
nowhere as a control (its floor is enforced in the Max tokens menu); history
shows each turn's settings on the request bubble's hover card, marks settings changes
between turns with a dashed divider, and warns above the composer when a
change invalidates the cache prefix (only model/system/tools fields warn).

Still open: the Response tab renders raw JSON only (no per-block sections from
`content_blocks`), chat messages do not render fenced code, no elapsed time
while a turn streams (the `.streaming` caret itself is styled), no "jump to
latest" pill, no sidebar loading skeleton, no title filter, and
`prefers-reduced-motion` is not honoured.

This plan was written from first principles: what this product is, what job the
user hires it for, what existing LLM chat UIs have already figured out, and
where this app must deliberately differ from them.

---

## 1. What this product is — and what it is not

LLM SDK View is **an inspection and teaching tool with a chat attached**, not a
chat client with a code view attached.

```
ChatGPT / Claude / LibreChat:   chat is the hero, code is an afterthought
LLM SDK View:                   the request and the response are the hero;
                                chat is how you produce them
```

Every design decision below follows from three product principles that already
govern the codebase, now applied to pixels:

1. **The wire is the truth.** The user must always be able to answer "what
   exactly will be / was sent, and what exactly came back" without scrolling,
   guessing, or trusting a reconstruction.
2. **Every status is legible.** A value the runtime fixed, a model that cannot
   do something, a fallback number — each must *look* different, not just read
   different. The six control statuses are this app's signature feature; today
   they are bracketed plain text, which buries them.
3. **No fake controls, no fake states.** A Stop button that cannot cancel the
   provider stream, a token count that is secretly an estimate, a `create()`
   render of a `stream()` call — none of these may appear. The UI states
   matrix below only contains states the backend can actually produce.

## 2. Jobs to be done, ranked

| # | Job | Primary surface |
|---|---|---|
| 1 | "Tweak a knob and see the wire-level effect *before* it costs anything" | form + Request pane, simultaneously visible |
| 2 | "See exactly what came back, with honest provenance" | Response pane |
| 3 | "Ask something and read the reply comfortably" | chat |
| 4 | "Revisit and compare past turns/conversations" | sidebar + turn↔inspector association |
| 5 | "Know at a glance what config is active and whether it fits" | config summary + context meter |

The ranking matters for layout: jobs 1–2 are why this app exists, so the
inspector must never be the thing that collapses first.

## 3. What existing LLM chat UIs have already solved (don't reinvent)

| Reference | What it does well | What to borrow | What to deliberately avoid |
|---|---|---|---|
| **Anthropic Console Workbench** | closest product analog: prompt pane, parameters rail, "Get code" export | parameters as a rail/drawer; code as first-class export | code that goes stale (ours live-previews — keep that advantage) |
| **OpenAI Playground** | center chat + right parameter rail; session-level settings | rail grouping by concern (model / generation / tools) | settings that hide while you type — our knob→code loop needs form and code co-visible |
| **Vercel AI Chatbot / Chatbot UI** | collapsible history sidebar (⌘B-style), message bubbles, code blocks with copy, streaming caret | sidebar collapse, copy buttons, streaming affordance | chat-first information architecture; regenerate/edit-message buttons that rewrite history (violates our "one record per turn, never trimmed" rule) |
| **LM Studio / Jan** | per-chat parameter sidebar, dense developer density | 13px base density, developer-grade information packing | desktop-app chrome (title bars, native menus) — this is a browser page |
| **LibreChat** | breadth | nothing in particular | its weight; also impractical to run here |

Common patterns worth taking wholesale: keyboard-first operation (⌘Enter send,
⌘B sidebar), code blocks with a copy affordance, empty states that teach,
sticky composer, "jump to latest" pill when scrolled up, segmented tabs.

## 4. Current-state audit (index.html, 650 lines)

Concrete findings, grouped:

**Layout**
- Form is 13 controls in a flat 2-column grid *below* the chat input: the most
  important job (knob→code) is buried at the bottom of the middle column, and
  the chat log is squeezed to `min-height: 200px`.
- Sidebar is fixed 260px, not collapsible; weak "current" affordance.
- Inspector tabs are two buttons with an outline; there is dead code
  (`index === len-1 ? 'request' : 'request'`) and the pane never auto-switches
  to Response when a turn completes — the newest information requires a manual
  click.

**Theme / Mac feel**
- Dark-only: `body` hardcodes `#10141b` while `color-scheme: light dark` is
  declared but never honoured. On a light-mode Mac, native inputs/buttons
  render light-on-dark and look broken.
- Buttons and inputs are browser-default styled; no focus rings beyond the
  platform default; no keyboard shortcuts at all (Enter in the textarea
  inserts a newline; Send requires a mouse).
- Code font is the default `pre` font, not SF Mono/ui-monospace.

**Honesty / status legibility**
- Six statuses are rendered as `[Status] note` plain text — same color, same
  weight, no iconography; the legend lives only in docs.
- Context "meter" is a text line; the refusal state ("does not fit") is a
  sentence, not a visual state.
- Model-data source and dynamic-filtering lines are also plain text.

**Code pane**
- No syntax highlighting; no copy button; `pre-wrap` with no max-height
  management.
- Response tab dumps raw JSON — no folding, no per-block rendering, even
  though `ResponseView` already carries structured `content_blocks`.

**States**
- Loading: none anywhere (form load, preview build, conversation load, send).
- Streaming: raw text append; no caret, no elapsed time, no per-turn status.
- Errors: inline text only; the "sent but NOT saved" state (a real, designed
  state) looks identical to a minor notice.
- Empty: chat is a blank bordered box; conversation list is blank on first
  run; Response tab has a one-liner.

**Chat**
- Messages are plain blocks; assistant replies that contain fenced code render
  as unformatted text — in an app whose entire point is reading code.
- No association between a chat turn and the inspector: clicking an old bubble
  shows its request, but there is no turn numbering, no visible link, and the
  Response tab can't be navigated per-turn except by clicking bubbles.

## 5. Target information architecture

Three regions, sized by the job ranking (sidebar collapses first, chat second,
inspector never):

```
┌─────────────────────────────────────────────────────────────────────────┐
│ ☰ LLM SDK View   [Model▾][Thinking▾][Effort▾][Max tokens▾][System▾]      │
│                  [Web Search▾][Caching▾][Streaming 🔒]   ← topbar pills  │
├────────────┬──────────────────────────────┬───────────────────────────┤
│ ☰ Convos   │  Chat                        │  Inspector                │
│ + New      │  ┌────────────────────────┐  │ ┌─────────┬────────────┐  │
│ • title    │  │ turn 1  user           │  │ │ Request │  Response  │  │
│   2t 19:59 │  │ turn 1  assistant  ▸i  │  │ └─────────┴────────────┘  │
│ • title    │  │ ─ settings changed ──  │  │ status line · [⧉ copy]    │
│            │  │ …(scroll, caret)       │  │ ┌───────────────────────┐   │
│ ⌘B         │  └────────────────────────┘  │ │ syntax-highlighted    │   │
│ collapses  │  ⚡ cache-prefix warning      │ │ code / response JSON  │   │
│            │  [ textarea        ] [Send ⌘⏎]│ └───────────────────────┘   │
│            │  ▓▓▓▓▓░░░ 1.2k / 200k        │ usage · stop reason         │
└────────────┴──────────────────────────────┴───────────────────────────┘
```

**The one consequential IA decision: where the form lives.** Candidates:

- A. Right rail as a 4th column — rejected: inspector would shrink below
  usefulness on a 1440px screen; breaks the "inspector never collapses" rule.
- B. A third inspector tab (Request | Response | Parameters) — rejected:
  kills the knob→code feedback loop, the #1 job.
- C. Collapsible panel in the center column — **superseded** (shipped 2026-09,
  replaced 2026-09-27): even collapsed it spent a permanent bar on a summary
  nobody read, and expanded it pushed the conversation around.
- D. **Pills in the topbar — chosen 2026-09-27.** The header was empty space;
  eight pills (Model · Thinking · Effort · Max tokens · System · Web Search ·
  Caching · Streaming) fill it, each opening a small anchored menu. The
  collapsed panel's controls survive as a hidden value store, so the payload,
  validation and preview keep one source of truth; the pills are a view over
  it, never a second one. A pill the model or runtime cannot honour is dashed
  and grey with the reason, never a fake control.

**Form grouping** is gone with the panel; each pill is one API field (or one
field family for Web Search). Status language is unchanged: a dashed grey
pill is the old greyed control, the menu's note line carries what the group
note used to say.

**Inspector behavior**: segmented tabs; auto-switch rules — typing/preview
shows Request, a completed turn switches to Response (the new information is
there), clicking any historical bubble shows that turn. The dead ternary goes
away.

## 6. Backlog by area (acceptance criteria in brackets)

### P0 — structure & honesty

1. **Split static assets**: `static/index.html` + `static/app.css` +
   `static/app.js` (+ vendored highlighter later). No framework, no build
   step. Keep every existing element `id`. [page behaves identically; the six
   HTML-grepping tests are re-pointed to the new files or to ID-level checks]
2. **Light + dark theme** via `prefers-color-scheme`, all colors as CSS
   custom properties; fully styled controls (no browser defaults).
   [screenshot in both themes; every interactive element has a visible focus
   ring]
3. **Parameters → topbar pills**: eight pills in the header, each opening an
   anchored menu that writes back into the hidden form controls.
   [shipped 2026-09-27: effort menu has no synthetic Default row (the
   documented level carries the tag); thinking budget shown nowhere as a
   control, its floor enforced in the Max tokens menu; ⌘, retired]
4. **Status badge component**: one visual language for the six statuses —
   color + icon + label, identical meaning everywhere (form notes, context
   meter, model data, dynamic filtering). "?" legend popover.
   [every place a status appears uses the badge; no `[Status]` bracket text
   remains]
5. **Context meter bar**: horizontal bar with used/reserved segments,
   thresholds (ok < 70% / amber / red), source caption, red refusal state with
   the two ways out inline. [estimate always labelled estimated; refusal
   disables Send exactly as today]
6. **States matrix** implemented for every region:

   | Region | Loading | Empty | Error |
   |---|---|---|---|
   | sidebar | skeleton rows | "No conversations yet" + note where they live | fetch error inline |
   | chat | (n/a) | one-paragraph explainer of the two panes | provider error bubble, verbatim, with the "nothing was sent" fact |
   | composer | Send disabled + "Sending…" + elapsed | Send disabled | refusal reason inline (exists, restyle) |
   | code pane | subtle "building…" in status line | "Type to preview" (exists, restyle) | refusal text (exists, restyle) |
   | response pane | — | "No response yet…" (exists) | "sent but NOT saved" as a distinct warning banner, not a note |

### P1 — the reading experience

7. **Syntax highlighting + copy**: vendor Prism (core + python + json,
   ~8 KB, no CDN, no build); copy button with transient "Copied" state.
   [works offline; copied text is byte-identical to the pane]
8. **Response view structure**: render `content_blocks` as titled sections
   (text / thinking / tool use / citations) with raw JSON behind a toggle —
   the data already exists in `ResponseView`, this is presentation only.
   [JSON view remains byte-identical to today]
9. **Chat reading**: fenced-code and inline-code rendering in messages
   (markdown-lite: code only, no tables/HTML injection); streaming caret with
   elapsed time; "jump to latest" pill when scrolled up.
   [textContent-based, no innerHTML of model output]
10. **Turn ↔ inspector association**: turn numbers on bubbles; clicking a
    bubble highlights it and loads that turn into the inspector; per-turn
    one-line usage summary under assistant messages (tokens · stop reason)
    that deep-links to the Response tab.
11. **Keyboard map**: ⌘Enter send · ⌘B sidebar · ⌘1/⌘2 tabs.
    [shortcuts shown in tooltips; no shortcut conflicts with text editing]

### P2 — polish

12. Sidebar: collapse to icon rail; client-side title filter (presentation
    only — not the "search history" feature, which is a declared non-goal).
13. Code pane: line-wrap toggle; line numbers.
14. `prefers-reduced-motion` honoured; scroll shadows; SubtleMac scrollbars.
15. Revisit dead `session_id: 'default'` in the payload (dead field today).

## 7. Explicit non-goals (guardrails)

- **No Stop/cancel button** unless the cancellation semantics of
  `llm-anthropic`'s stream are verified — a client abort that keeps billing
  would be a fake control.
- **No regenerate / edit-message / delete-turn buttons**: they rewrite or
  prune history, which this product refuses to do.
- **No rename/delete/export conversations**: declared "not implemented" in the
  README; that is a backend objective, not a UI pass.
- **No suggested-prompt gallery that auto-sends**; click-to-fill text only.
- **No framework, no build step, no npm**: single-page vanilla + vendored
  Prism. The project is a thin honest layer; the frontend must stay legible
  enough to audit in one sitting.
- **No other providers, agents, RAG, uploads** (AGENTS.md boundaries).
- **No mobile investment** beyond the existing breakpoints: this is a Mac
  desktop browser tool.

## 8. Test impact

Six tests grep `index.html` source for implementation strings
(`test_control_status.py`, `test_streaming.py`, `test_preview.py`,
`test_context.py`, `test_model_capabilities.py`, `test_models_api.py`). They
encode real regressions (the runtime-fixed-caller Send bug, the
no-`messages.create()` rule), so they must not be deleted — re-point them at
`app.js` or, better, convert them to assert on stable element IDs/attributes
so the next visual iteration doesn't touch them again.

Sequencing: steps 1–2 first (pure structure, behaviour-identical), then 3–6,
then P1/P2 — each step independently shippable with the suite green.
