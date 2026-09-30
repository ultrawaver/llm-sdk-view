# Per-turn settings: what is recorded, and what a past turn can honestly say

Status: **investigation closed, nothing implemented** (2026-09-27, branch
`conversation-vertical-slice`). The chip strip proposed in earlier drafts was
withdrawn — see §4; §1b records why it is unnecessary.

Question this answers: the form can be changed before every turn, so what is
recorded per turn, and how much of a past turn's configuration can be shown?

Decision already made by the user: **history is read-only.** A past turn's
settings are never written back into the form; "restore" is not a feature.

---

## 1. Short answer

Every turn already records both halves of the story:

| What | Where it lives |
|---|---|
| what the form asked for | `native_api_chat_turns.effective_options` = `asdict(ChatOptions)`, 11 fields |
| what actually went out | `native_api_chat_turns.request_kwargs` = `model.build_kwargs()` output |
| what the pane rendered | `native_api_chat_turns.rendered_code` |
| what answered | `turns.response_json` (model served, stop reason) + `turns.input/output_tokens` |

`/api/conversations/<id>` already returns `options` and `request_kwargs` for
every turn. Nothing needs to be added to storage; showing it is a read-only
presentation job. The UI uses exactly one of those fields today — `model`, from
**turn 1**, when a conversation is opened.

**Settled in §1b: the Request pane already answers the question.** Everything on
the wire is visible in it verbatim, and anything absent from it was not a live
choice at that moment, so the provider default applied. A separate settings
strip is therefore not being built.

## 1b. "Isn't the Request itself the settings?" — yes

The user's objection, which is correct: **an absent parameter is not "lost", it
is the provider default.** Saying otherwise was wrong, and here is why it is
wrong in this codebase specifically.

**The Request is a faithful projection.** `render_kwargs()` is deliberately
dumb: it iterates `kwargs.items()` and emits every top-level key as-is, with no
filtering, no defaults and no reordering (`native_api_chat/codegen.py`). `model`,
`max_tokens`, `system`, `tools[].type` (the real web-search version),
`max_uses`, `thinking` in its actual shape, `output_config.effort`,
`output_config.response_inclusion`, `extra_body`, `betas` and
`messages[-1].content[-1].cache_control` are all readable from it. Verified in
this app's own stored rows.

**Absence means default, and it cannot mean "substituted".** This project's
rule is *a control the provider cannot honour is refused, not faked*, and that
rule has teeth in code, not just in intent:

- `capabilities.py` names how "off" is expressed per model
  (`thinking_off_request` = `disabled` / `omitted` / `unsupported`) and states
  why: a model whose documented default is not to think is turned off **by
  omitting the field**, which is the default and cannot be rejected.
- `ChatSession._check_effort()` **raises** rather than dropping a value: if the
  model does not support effort, or the level is not offered, or the level is
  rejected with thinking off → `UnsupportedOptionError` / `ValueError`.
- The form greys the same things out before you can pick them
  (`effort_disabled_with`, surfaced as status badges).

So there is no path by which "I set X" becomes "Y went out". Either X went out
and is visible, or X could not be sent and was refused before the request was
built. Reading the Request therefore answers the question completely:

> Everything on the wire is in it; everything not in it was not a live choice
> at that moment, so the provider default applied.

**The one distinction worth keeping** (not "loss", just provenance): the default
can come from two different places.

| Absent field | The default belongs to | Example |
|---|---|---|
| `thinking` | Anthropic — the model's documented default | no key ⇒ that model does not think; `capabilities.thinking_default` names it |
| `output_config` | Anthropic — effort is not a live dimension without thinking | no key ⇒ effort default |
| cache TTL | **llm-anthropic, not Anthropic** | the API accepts `cache_control.ttl`; this plugin has no such parameter, so requests always go out at the provider's 5-minute default |
| `allowed_callers` | **llm-anthropic, not Anthropic** | same shape: never expressible by the installed plugin |

In every row the *effect on that turn* was fully determined, so nothing was
lost — but for the last two the reason is the tool, not the API. If someone
copied the rendered code into their own SDK usage, "absent" would no longer
imply the same thing. That is a note about provenance, not a missing value.

**Conclusion.** No chip strip is needed. It would only restate what the Request
already says, plus one fact — *whose* default an absence is — that the form's
existing `Provider default` badge already shows for the live controls. The one
cheap thing this document recommends instead is §4.

## 2. Per-field: what can be shown, and from where

`intent` = `options.*` (what the form said). `wire` = `request_kwargs` (what
was sent). Where they can differ, the difference is the interesting part and
must be shown as a difference, not papered over.

| Setting | From | Can show | Note |
|---|---|---|---|
| model | intent + wire + response | ✅ all three | The response can name the snapshot that actually served it. |
| max_tokens | intent + wire | ✅ always | The wire value is authoritative; llm-anthropic may substitute its own default when no value was set. |
| system | intent + wire | ✅ | `kwargs.system` is absent when empty — absence means "not sent", not "unknown". |
| thinking | intent + wire | ✅ read from the wire | The form's `on/off/default` becomes `{"type": "disabled"}`, `{"type": "adaptive"}` or `{"type": "enabled", "budget_tokens": N}` depending on the model and the plugin, plus a `display` flag. Only the wire value says what happened. Can also arrive as `extra_body.thinking` when the 128K-output beta applies. |
| effort | wire | ✅ | Lands in `output_config.effort` only when the model supports effort and thinking is on. Otherwise the key does not exist, which means the provider default — `_check_effort()` refuses every level that could not be expressed, so "my value was silently dropped" cannot happen. |
| web_search on/off | intent + wire | ✅ | Truth = whether `tools` carries a web search entry. |
| tool version | intent + wire | ✅ | The stored intent can be null ("plugin decides"); only `tools[].type` says which version went out (`web_search_20250305` vs `web_search_20260318`). Real: this app's own history already contains both. |
| max_uses | intent + wire | ✅ | `0` (unlimited) is sent as *omission*, so absence is the value. |
| response_inclusion | intent + wire | ✅ | Only exists on `web_search_20260318` and only when the installed plugin can express it; otherwise the form marks it `Unsupported by current tool version` and it never reaches the wire. |
| allowed_callers | intent **only** | ⚠️ intent only | `llm-anthropic` has no such parameter and never sends it. This field must never be presented as part of the request. |
| cache_control on/off | intent + wire | ✅ | Truth = `messages[-1].content[-1].cache_control == {"type": "ephemeral"}` (verified in the stored kwargs). It sits inside `messages`, not at the top level. |
| cache TTL | — | provider default | Not a wire fact: every request goes out at the provider's 5-minute default, and the control already says so. |
| temperature / top_p / top_k | wire only | ✅ | `extra_body.temperature: 1.0` is added by `llm-anthropic`, not set by this form — show it as "added by the plugin". |
| betas | wire only | ✅ | `betas` (`effort-2025-11-24`, `output-128k-…`) is part of what went out. |
| streaming | — | constant | `llm-anthropic` always opens `messages.stream()`; there is no per-turn choice to display (the ON/OFF switch was removed). |

## 3. What cannot be shown — and must not be guessed

1. **Turns with no sidecar row.** `load_conversation` LEFT JOINs the sidecar, so
   a conversation written elsewhere (`llm -c`, another plugin) has turns with
   empty `options` *and* empty `request_kwargs`. Those turns get "this turn's
   settings were not recorded", not inferred values.
2. **What Anthropic expanded server-side.** This app's 70-character
   `web_search` stub becomes ~2,200 input tokens in the service; nothing in the
   stored kwargs shows that. Only `count_tokens`/`usage` measures it.
3. **The plugin and SDK versions in force at the time.** Not recorded. The tool
   version in `tools[].type` is evidence, not a version number.
4. **`allowed_callers` and the cache TTL as wire facts.** Neither is ever sent;
   they exist only as labelled intent / provider default. Not "missing" — just
   not expressible by this runtime.
5. **Anything derived from a re-resolution against today's model.** A past
   turn's thinking mode must be read from that turn's stored kwargs, never
   recomputed with the currently installed plugin — the record *is* the truth
   for that moment.

## 4. Decision: do not build the chip strip

Withdrawn, for the reason in §1b: every chip would either repeat a value the
Request already shows verbatim, or report absence where absence already means
*provider default* — information the form's own `Provider default` badge carries
for the live controls. Shipping it would add a second rendering of the same
truth, and a second rendering is exactly how panes drift.

The cheap alternative, if anything at all is wanted: make the Request pane's
status line state the reading convention once, instead of leaving the user to
infer it. Today it reads `request · turn 2 of this conversation, as it was
sent`. Appending "— anything absent here went out at the provider default"
costs one sentence and answers the question the chips were going to answer.

Nothing here writes to the form, changes what Send will build, or claims a
value that was not recorded.

## 5. Not implemented, and why it needed no schema

Had it shipped, it would have been read-only derived data — this much is still
worth recording, because it is the reason the option stays cheap later:

- One derived reader in `native_api_chat/records.py` — a `TurnRecord.settings()`
  in the same spirit as `ResponseView.raw` and `ResponseView.cost`: derived on
  read, never stored, so an old row and a new row are read through the same
  code and no schema or migration is needed.
- `/api/conversations/<id>` already returns both halves for every turn, so no
  endpoint change is required either — only whatever renders it.
