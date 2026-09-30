/* LLM SDK View — page logic.
   One request truth: the right pane renders model.build_kwargs() output and
   nothing else. One record per turn: bubble, Response pane and stored row are
   projections of the same TurnRecord. */

const byId = (id) => document.getElementById(id);
const log = byId('log');
// Bubbles live in #messages; #log is only the scroll container.
const messages = byId('messages');
const state = {
  data: null, caps: null, blocked: false, fits: null,
  conversationId: null, conversationName: null,
  turns: [], tab: 'request', selected: null,
  cache: { anchor: null, lastInput: null, frozen: null },
  // Draft text, kept per conversation: leaving a conversation and coming
  // back must not cost the user what they had typed into it.
  drafts: {},
  // Unit prices are fetched, not shipped, so "no estimate" needs a reason.
  rates: null,
  // The meter's empty-draft baseline, stashed from the form payload.
  formContext: null
};

/* Prompt-cache lifetime. llm-anthropic sends the default ephemeral marker,
   so the TTL is the provider default of 5 minutes; the runtime exposes no
   setting for it. The anchor is the moment the answer FINISHED landing -
   not the start of the request that Anthropic's documentation names. That
   documented rule was measured against this account's own history and lost:
   turns that began 335.6s and 338.9s after the previous request started
   still read the cache (12 hits, 0 misses), which a 5-minute clock from
   request start says cannot happen, while a clock from the end of the
   answer fits every one of them. It is also the only anchor the user can
   act on: they cannot send the follow-up before they have read the answer.
   No miss has ever been observed, so this is the best-fitting model, not a
   verified one - the note under the countdown says where it counts from. */
const CACHE_TTL_MS = 5 * 60 * 1000;

/* The countdown's faces: green while the window is comfortable, amber in
   the last minute, red and moving in the last thirty seconds, grey once it
   is gone. Escalation by colour is the convention every countdown timer
   uses, so the state registers before the number does - which is the whole
   point: send the next message before this cache is lost. */
const TTL_WARN_MS = 60 * 1000;
const TTL_URGENT_MS = 30 * 1000;

function cacheTouchOf(response) {
  const usage = (response && response.usage) || {};
  const read = usage.cache_read_input_tokens;
  const write = usage.cache_creation_input_tokens;
  // A read is the more interesting event (it means the cache paid off), so
  // it wins when a turn both read and extended the entry.
  if (typeof read === 'number' && read > 0) return { kind: 'read', tokens: read };
  if (typeof write === 'number' && write > 0) return { kind: 'write', tokens: write };
  return null;
}

function cacheAnchorOf(record) {
  const touch = cacheTouchOf(record.response);
  if (!touch) return null;
  // momentOf, not Date.parse: a stored stamp can be llm's naive UTC form,
  // which Date.parse would read as local time and move the anchor by the
  // whole timezone offset. The stamp is the turn's end - llm files the
  // turn when the stream closes - and the end is the anchor (see above).
  const end = momentOf(record.timestamp);
  if (!end) return null;
  return { at: end.getTime(), kind: touch.kind, tokens: touch.tokens };
}

function clock(ms) {
  const total = Math.max(0, Math.round(ms / 1000));
  return Math.floor(total / 60) + ':' + String(total % 60).padStart(2, '0');
}

/* How long ago something happened, small enough for a footer. clock() has
   no unit above minutes, and a conversation reopened days later is not
   carrying "4320:00" of information. */
function ago(ms) {
  if (ms >= 3600000) return Math.round(ms / 3600000) + 'h';
  return clock(ms);
}

/* --- when a turn happened ---------------------------------------------------
   The record carries UTC; a chat shows the computer's own clock, so the
   stamp is converted here rather than printed as it is stored. A stamp with
   no zone on it is llm's UTC, never the browser's zone, so it gets one
   before it is parsed. */
const ZONED_STAMP = /(?:Z|[+-]\d{2}:?\d{2})$/;
const TIME_TODAY = new Intl.DateTimeFormat(undefined,
  { hour: '2-digit', minute: '2-digit' });
const TIME_THIS_YEAR = new Intl.DateTimeFormat(undefined,
  { month: 'numeric', day: 'numeric', hour: '2-digit', minute: '2-digit' });
const TIME_FULL = new Intl.DateTimeFormat(undefined,
  { year: 'numeric', month: 'numeric', day: 'numeric',
    hour: '2-digit', minute: '2-digit' });

function momentOf(value) {
  if (value instanceof Date) return value;
  if (!value) return null;
  const text = String(value);
  const date = new Date(ZONED_STAMP.test(text) ? text : text + 'Z');
  return Number.isNaN(date.getTime()) ? null : date;
}

/* Today is the time alone; a full date only appears once it is needed. */
function formatMoment(moment) {
  if (!moment) return '';
  const now = new Date();
  if (moment.toDateString() === now.toDateString()) return TIME_TODAY.format(moment);
  if (moment.getFullYear() === now.getFullYear()) return TIME_THIS_YEAR.format(moment);
  return TIME_FULL.format(moment);
}

function renderCacheStatus() {
  const stateEl = byId('cacheState');
  const noteEl = byId('cacheNote');
  // The bar gets the short state; the long explanation lives in the cost
  // popover's cacheNote slot, which only exists while the popover is built.
  // data-state drives the colour ladder, so the urgency is visible before
  // any of this text is read.
  const say = (label, note, face) => {
    stateEl.textContent = label;
    stateEl.dataset.state = face || 'idle';
    if (noteEl) noteEl.textContent = note;
  };
  if (byId('cacheControl').value !== 'true') {
    say('cache off', 'Prompt cache is off — every turn pays the full input price.');
    return;
  }
  const minimum = state.caps ? state.caps.min_cacheable_tokens : null;
  const anchor = state.cache.anchor;
  if (anchor) {
    const frozen = state.cache.frozen;
    const remaining = frozen
      ? frozen.remaining : anchor.at + CACHE_TTL_MS - Date.now();
    const face = remaining > TTL_WARN_MS ? 'live'
      : remaining > TTL_URGENT_MS ? 'warn'
        : remaining > 0 ? 'urgent' : 'expired';
    // A number that will not say where it counts from cannot be checked,
    // so the anchor rides on the badge as its tooltip.
    stateEl.title = 'counted from when the answer landed · the next hit refreshes it';
    stateEl.dataset.frozen = frozen ? 'yes' : 'no';
    if (frozen) {
      // Frozen by the send key: the margin the send had stays up while the
      // answer streams, then the record event starts a fresh countdown.
      say('TTL ' + clock(remaining),
        'Sent with ' + clock(remaining)
        + ' to spare — the countdown resumes from the answer\u2019s end when it lands',
        face);
      stateEl.title = 'cache was alive when this turn was sent · counting resumes when the answer lands';
      return;
    }
    if (face === 'expired') {
      say('cache expired',
        'Expired ' + ago(-remaining)
        + ' ago — the next turn writes a fresh cache at 1.25× the input price',
        face);
      return;
    }
    const touched = (anchor.kind === 'read' ? 'Read ' : 'Written ')
      + fmt(anchor.tokens) + ' tok';
    if (face === 'urgent') {
      say('TTL ' + clock(remaining),
        touched + ' · send now — after ' + clock(remaining)
        + ' the prefix is written again at 1.25× the input price',
        face);
      return;
    }
    say('TTL ' + clock(remaining),
      touched + ' · expires in ' + clock(remaining)
      + ' · counted from the answer\u2019s end; the next hit refreshes it for free',
      face);
    return;
  }
  stateEl.title = '';
  stateEl.dataset.frozen = 'no';
  const input = state.cache.lastInput;
  if (typeof input === 'number' && minimum && input < minimum) {
    say('not cached',
      'Input ' + fmt(input) + ' tok is below the ' + fmt(minimum)
      + ' tok minimum for this model — shorter prompts are silently not '
      + 'cached, no error is returned');
  } else if (typeof input === 'number' && minimum) {
    say('not cached', 'Both cache counters were 0 on the last turn');
  } else {
    say('no cache data', minimum
      ? 'This model caches only from ' + fmt(minimum) + ' input tokens up'
      : 'Send a turn to see whether the cache caught it');
  }
}

setInterval(renderCacheStatus, 1000);
let streaming = false;

/* --- following the stream without stealing the reading place ----------------
   New content scrolls the chat only while the user is already at the
   bottom; once they scroll up to read, the stream keeps arriving but the
   page stays put and offers the way back. Reduced motion turns the smooth
   ride off: the jump is instant rather than animated. */
const reducedMotion = window.matchMedia
  && window.matchMedia('(prefers-reduced-motion: reduce)').matches;
const jumpPill = byId('jumpPill');
let pinnedToBottom = true;
/* While the pill's own smooth ride is in progress, the intermediate scroll
   positions are not the user scrolling away: without this flag the ride
   would unpin itself on its first frame and the stream would pull the pill
   back out from under the click. */
let ridingToBottom = false;

function scrollChatToBottom(smooth) {
  log.scrollTo({
    top: log.scrollHeight,
    behavior: smooth && !reducedMotion ? 'smooth' : 'auto'
  });
}

/* Every append to the chat comes through here: follow when pinned, offer
   the pill when not. */
function followStream() {
  if (pinnedToBottom) scrollChatToBottom(false);
  else jumpPill.hidden = false;
}

let lastScrollTop = 0;
log.addEventListener('scroll', () => {
  const atBottom = log.scrollHeight - log.scrollTop - log.clientHeight <= 48;
  if (atBottom) ridingToBottom = false;
  // Only a hand moving up unpins. A programmatic follow-scroll's event
  // fires after the stream has already grown past the position it scrolled
  // to, and reading that as "the user left the bottom" would drop the pin
  // in the middle of a turn the page is supposed to be following.
  const movedUp = log.scrollTop < lastScrollTop;
  lastScrollTop = log.scrollTop;
  if (movedUp && !ridingToBottom) pinnedToBottom = false;
  if (atBottom) pinnedToBottom = true;
  if (pinnedToBottom) jumpPill.hidden = true;
});

/* A hand on the wheel or the glass cancels the ride: the user is driving
   again, so their position wins over the pill's destination. */
log.addEventListener('wheel', () => { ridingToBottom = false; }, { passive: true });
log.addEventListener('touchstart', () => { ridingToBottom = false; }, { passive: true });

jumpPill.addEventListener('click', () => {
  pinnedToBottom = true;
  ridingToBottom = true;
  jumpPill.hidden = true;
  scrollChatToBottom(true);
});

/* --- cost ---------------------------------------------------------------------
   One receipt shape, two scopes. The footer is the CONVERSATION's own total -
   every turn it holds, summed - and a turn's own receipt is shown by hovering
   its answer, where it describes that one turn. The settings card stays on the
   user's bubble, because the settings are what the request asked for; showing
   them on both bubbles made the answer look like a second copy of the request.

   The numbers are ResponseView.cost, computed server-side from rates the
   pricing page publishes and labelled as estimates. This file only sums and
   formats them; it never prices anything itself. */

function money(usd) {
  if (typeof usd !== 'number') return '–';
  const abs = Math.abs(usd);
  const digits = abs >= 1 ? 2 : 4;
  return (usd < 0 ? '-$' : '$') + abs.toFixed(digits);
}

const round6 = (value) => Math.round(value * 1e6) / 1e6;

function costLine(cost, key) {
  return (cost.lines || []).find((line) => line.key === key) || null;
}

/* The conversation's totals, summed from the turns' own receipts.
   Summing the lines rather than re-deriving keeps one definition of a line, so
   the answer's receipt and the footer's cannot disagree about what a cache
   read cost. Only priced turns are in the money: a turn whose model the
   pricing page does not cover has no estimate at all, and the count of those
   travels with the total so the popover can say the sum is short of the
   conversation instead of quietly reporting a smaller one. */
function conversationTotals() {
  const turns = state.turns.length;
  if (!turns) return null;
  const lines = new Map();
  const models = new Set();
  let priced = 0;
  let missingCounters = 0;
  let inputTotal = 0;
  let noCacheTotal = 0;
  let provenance = null;
  state.turns.forEach((record) => {
    const cost = record.response && record.response.cost;
    if (!cost || typeof cost.total !== 'number') return;
    priced += 1;
    if (cost.cache_counters_reported === false) missingCounters += 1;
    inputTotal += cost.input_total_tokens || 0;
    noCacheTotal += cost.no_cache_total || 0;
    if (cost.model) models.add(cost.model);
    provenance = provenance || cost;
    (cost.lines || []).forEach((line) => {
      const kept = lines.get(line.key);
      if (!kept) { lines.set(line.key, { ...line }); return; }
      kept.quantity += line.quantity;
      kept.amount += line.amount;
      // A line is "reported" only if every turn it came from reported it.
      kept.reported = kept.reported && line.reported;
    });
  });
  if (!priced) return null;
  const sums = Array.from(lines.values())
    .map((line) => ({ ...line, amount: round6(line.amount) }));
  // The total is the sum of the rows above it, exactly as a per-turn receipt
  // is: a receipt whose rows do not add up to its total is not a receipt.
  const total = round6(sums.reduce((sum, line) => sum + line.amount, 0));
  const read = lines.get('cache_read');
  const output = lines.get('output');
  const search = lines.get('web_search');
  return {
    turns,
    priced,
    unpriced: turns - priced,
    missingCounters,
    models: Array.from(models),
    lines: sums,
    total,
    input_total_tokens: inputTotal,
    output_tokens: output ? output.quantity : null,
    searches: search ? search.quantity : 0,
    no_cache_total: round6(noCacheTotal),
    savings: round6(noCacheTotal - total),
    cache_hit_rate: inputTotal ? (read ? read.quantity : 0) / inputTotal : null,
    rates_state: provenance.rates_state,
    rates_date: provenance.rates_date,
    rates_source: provenance.rates_source,
    rates_error: provenance.rates_error
  };
}

const COST_COLORS = {
  output: 'out', uncached_input: 'uncached', cache_write_5m: 'cw',
  cache_write_1h: 'cw', cache_read: 'cr', web_search: 'tool'
};
const COST_STACK_ORDER = [
  'output', 'uncached_input', 'cache_write_5m', 'cache_write_1h',
  'cache_read', 'web_search'
];
const COST_GROUP_NAMES = { input: 'Input', output: 'Output', tools: 'Tools' };

/* Why a figure is missing, in the terms the surface asking needs. Prices are
   fetched rather than shipped, so "no estimate" has more than one honest
   reason: the page could not be read, or this model is not on it. The rates
   answer is read under the names the server sends - ``rates_state`` and
   ``rates_error``; reading ``state``/``error`` here silently matched nothing
   and left the first reason unreachable. */
function noCostReason(scope) {
  if (!state.turns.length) return 'send a turn to see what it cost';
  if (state.rates && state.rates.rates_state === 'unavailable') {
    return 'no rates: '
      + (state.rates.rates_error || 'the pricing page could not be read');
  }
  return scope === 'turn'
    ? 'the pricing page does not cover this turn\u2019s model'
    : 'no cost estimate for this conversation';
}

async function loadRates() {
  try {
    state.rates = await (await fetch('/api/rates')).json();
  } catch (ex) {
    state.rates = null;
  }
  renderCostBar();
}

function renderCostBar() {
  const totals = conversationTotals();
  const toggle = byId('costToggle');
  const scope = byId('costTokens');
  const hit = byId('costHit');
  toggle.disabled = !totals;
  toggle.title = totals
    ? 'Cost breakdown for this conversation · ' + totals.turns
      + (totals.turns === 1 ? ' turn' : ' turns')
    : 'Cost breakdown for this conversation';
  if (!totals) {
    byId('costTotal').textContent = '–';
    // The reason is a sentence, not a figure, and this is the only place it is
    // ever said - the popover hides itself when there is nothing to total. So
    // the whole of it rides along as the title: "no rates: the pricing page
    // co…" is not a reason anyone can act on.
    scope.textContent = noCostReason('conversation');
    scope.title = scope.textContent;
    hit.textContent = '';
    return;
  }
  byId('costTotal').textContent = money(totals.total);
  // The turn count leads: it is what says this figure is the conversation's
  // and not the turn the right pane happens to be showing.
  //
  // The token totals and the tool-call count are deliberately absent. The bar
  // is one line that cannot wrap, and each of them was long enough to be the
  // thing that got squeezed when the pane narrowed - a half-drawn
  // "5.2k in · 81…" claims a precision this bar does not have, and it was
  // never the figure that told you anything the receipt could not. Both are in
  // the popover: the token totals as the receipt's own rows, the tool calls on
  // its header line. What is left here is short enough to stay whole.
  scope.textContent = totals.turns + (totals.turns === 1 ? ' turn' : ' turns');
  scope.title = '';
  hit.textContent = totals.cache_hit_rate === null
    ? ''
    : 'cache ' + (totals.cache_hit_rate * 100).toFixed(1) + '% hit';
}

/* One receipt, drawn from the same line shape for a single turn and for the
   whole conversation: the stacked bar, the grouped rows and the hit-rate
   arithmetic are written once, so the two scopes cannot drift apart. */
function costReceiptHtml(cost, { note = false } = {}) {
  const stack = COST_STACK_ORDER.map((key) => {
    const line = costLine(cost, key);
    if (!line || line.amount <= 0 || cost.total <= 0) return '';
    return '<span class="c-' + COST_COLORS[key] + '" style="width:'
      + (100 * line.amount / cost.total).toFixed(1) + '%" title="'
      + esc(line.label) + ' ' + money(line.amount) + '"></span>';
  }).join('');

  const groups = {};
  (cost.lines || []).forEach((line) => {
    (groups[line.group] = groups[line.group] || []).push(line);
  });
  const receipt = ['input', 'output', 'tools'].map((group) => {
    const lines = groups[group];
    if (!lines) return '';
    const tokenTotal = group === 'input'
      ? cost.input_total_tokens
      : lines.reduce((sum, line) => sum + line.quantity, 0);
    const heading = COST_GROUP_NAMES[group]
      + (group === 'tools' ? '' : ' · ' + fmt(tokenTotal) + ' tokens');
    const rows = lines.map((line) => (
      '<div class="cost-line">'
      + '<span class="what d-' + COST_COLORS[line.key] + '">' + esc(line.label)
      + (line.note ? '<small>' + esc(line.note) + '</small>' : '')
      + '</span>'
      + '<span class="toks num">' + fmt(line.quantity) + '</span>'
      + '<span class="rate num">' + esc(line.rate_label) + '</span>'
      + '<span class="amt num">' + money(line.amount) + '</span>'
      + '</div>'
    )).join('');
    return '<div class="cost-group-name">' + esc(heading) + '</div>' + rows;
  }).join('');

  let cache = '';
  if (cost.cache_hit_rate !== null) {
    const pct = cost.cache_hit_rate * 100;
    const read = costLine(cost, 'cache_read');
    const savings = cost.savings >= 0
      ? '<span class="cost-savings">saved ' + money(cost.savings)
        + ' vs no-cache (' + money(cost.no_cache_total) + ')</span>'
      : '<span class="cost-extra">cache cost ' + money(-cost.savings)
        + ' extra vs no-cache (' + money(cost.no_cache_total) + ')</span>';
    cache = '<div class="cost-hitstrip"><span class="pct num">'
      + pct.toFixed(1) + '%</span>'
      + '<span class="meter"><i style="width:' + pct.toFixed(1) + '%"></i></span>'
      + '<span>prompt cache hit rate</span></div>'
      + '<div class="cost-formula">hit rate = cache read '
      + fmt(read ? read.quantity : 0) + ' ÷ total input '
      + fmt(cost.input_total_tokens) + ' · ' + savings + '</div>';
  }
  if (cost.cache_counters_reported === false) {
    cache += '<div class="cost-formula">cache counters were not reported '
      + '(caching off) — the cache lines read $0</div>';
  }

  return '<div class="cost-stack">' + stack + '</div>'
    + receipt
    + '<div class="cost-total-row"><span>Total</span><span class="num">'
    + money(cost.total) + '</span></div>'
    + cache
    // The countdown's long sentence has one home, in the popover that is open.
    // A second element with the same id would be a second writer to it.
    + (note ? '<p class="cost-note" id="cacheNote"></p>' : '');
}

function renderCostPop() {
  const pop = byId('costPop');
  const totals = conversationTotals();
  if (!totals) {
    pop.hidden = true;
    byId('costBar').classList.remove('open');
    byId('costToggle').setAttribute('aria-expanded', 'false');
    return;
  }

  // The rates are fetched, not shipped, so a total priced from a stale cache
  // has to say which one it is standing on.
  const ratesLine = totals.rates_state === 'cached'
    ? 'rates: cached copy from ' + esc(totals.rates_date)
      + (totals.rates_error ? ' — ' + esc(String(totals.rates_error)) : '')
    : 'rates: ' + esc(totals.rates_source) + ' ' + esc(totals.rates_date);
  // A conversation can change models between turns, so the header only names
  // one when there is one.
  const models = totals.models.length === 1
    ? totals.models[0] : totals.models.length + ' models';

  // The sum covers the turns that could be priced; the ones that could not are
  // named rather than left to make the total look smaller than the truth.
  const caveats = [];
  if (totals.unpriced) {
    caveats.push(totals.unpriced + ' of ' + totals.turns
      + ' turns have no estimate and are not included above');
  }
  if (totals.missingCounters) {
    caveats.push(totals.missingCounters
      + (totals.missingCounters === 1 ? ' turn' : ' turns')
      + ' reported no cache counters \u2014 '
      + (totals.missingCounters === 1 ? 'its' : 'their')
      + ' cache lines read $0');
  }

  pop.innerHTML = '<h3>Conversation — <span class="num">' + money(totals.total)
    + '</span></h3>'
    + '<div class="cost-src">' + ratesLine
    + ' · estimated locally · ' + esc(models)
    // The tool calls the bar used to count. They are counted here because a
    // "12 searches" beside the total is the sort of figure that gets squeezed
    // into nonsense on a narrow pane, and this is where the rows behind it are.
    + (totals.searches
      ? ' · ' + totals.searches + ' search' + (totals.searches === 1 ? '' : 'es')
      : '')
    + '</div>'
    + costReceiptHtml(totals, { note: true })
    + '<div class="cost-session"><span>Priced turns</span><span class="num">'
    + totals.priced + ' of ' + totals.turns + '</span></div>'
    + caveats.map((text) => '<div class="cost-caveat">' + esc(text)
      + '</div>').join('');
  renderCacheStatus();
}

/* One turn's own receipt, shown by hovering its answer. The card is the same
   fixed overlay #settingsCard uses - never a descendant of a bubble, and
   pointer-events: none, so it cannot steal the click that picks the turn. */
function costCardHtml(record, index) {
  const head = '<h5>turn ' + (index + 1) + ' · cost</h5>';
  const cost = record.response && record.response.cost;
  if (!cost) {
    return head + '<div class="cost-src">no estimate for this turn</div>'
      + '<div class="sfoot">' + esc(noCostReason('turn')) + '</div>';
  }
  const provenance = cost.rates_state === 'cached'
    ? ' · rates cached from ' + esc(cost.rates_date)
    : '';
  return head + '<div class="cost-src">estimated locally · '
    + esc(cost.model || 'unknown model') + provenance + '</div>'
    + costReceiptHtml(cost);
}

function toggleCost() {
  const toggle = byId('costToggle');
  if (toggle.disabled) return;
  const pop = byId('costPop');
  const open = pop.hidden;
  if (open) renderCostPop();
  pop.hidden = !open;
  byId('costBar').classList.toggle('open', open);
  toggle.setAttribute('aria-expanded', open ? 'true' : 'false');
}

function refreshCost() {
  renderCostBar();
  if (!byId('costPop').hidden) renderCostPop();
}

byId('costToggle').addEventListener('click', toggleCost);

const esc = (text) => String(text).replace(/[&<>"]/g, (ch) => (
  { '&': '&amp;', '<': '&lt;', '>': '&gt;', '"': '&quot;' }[ch]
));

/* One visual language for every control status, so a greyed value says why.
   A status is a reason, not a reading: it costs one dot of width and says
   its sentence on hover, where the whole sentence fits. */
const STATUS_RULES = [
  ['not exposed', 'fixed'], ['runtime fixed', 'fixed'],
  ['Unsupported by selected model', 'fixed'],
  ['Unsupported by current tool version', 'warn'],
  ['Provider default', 'info'],
  ['Fallback capability data', 'warn'],
  ['Editable', 'ok']
];

function badgeClass(status) {
  for (const [key, value] of STATUS_RULES) {
    if (status.includes(key)) return value;
  }
  return 'info';
}

/* The dot that stands in for a status. The sentence it carries rides along
   as data, so nothing has to be re-read to show it. */
function whyMark(status, text) {
  const tip = [status, text].filter(Boolean).join(' · ');
  return '<span class="why-dot why-' + badgeClass(status) + '" tabindex="0"'
    + ' role="note" aria-label="' + esc(tip) + '" data-tip="' + esc(tip)
    + '"></span>';
}

function setNote(id, status, text) {
  byId(id).innerHTML = whyMark(status, text);
}

/* The worst status in a group is the one the group summary shows. */
function groupBadge(id, statuses) {
  const rank = (status) => {
    if (status.includes('Unsupported by current tool version')) return 4;
    if (status.includes('Fallback capability data')) return 3;
    if (status.includes('not exposed') || status.includes('runtime fixed')) return 2;
    if (status.includes('Unsupported by selected model')) return 2;
    if (status.includes('Provider default')) return 1;
    return 0;
  };
  let worst = null;
  statuses.forEach((status) => {
    if (status && (worst === null || rank(status) > rank(worst))) worst = status;
  });
  byId(id).innerHTML = worst ? whyMark(worst, '') : '';
}

/* --- the status sentence, on demand -----------------------------------------
   A status can be a whole clause ("Unsupported by current tool version",
   "the API has it, llm-anthropic does not") and the pills live in a
   horizontal scroller that would clip it, so the sentence is shown in one
   fixed card anchored to the dot that stands for it. One delegated pair of
   listeners covers every dot, including the ones rendered later. */

function hideWhyTip() {
  const card = byId('whyTip');
  card.hidden = true;
  card.classList.remove('answered');
}

/* Place a fixed overlay against the element it belongs to.
   Every overlay the topbar owns has to be positioned this way rather than
   nested in its anchor, because the pill row clips its descendants (see
   #pillLayer in the stylesheet). One function does the placing, so the
   status card and the pill menus cannot drift apart. `align` picks which
   edge lines up with the anchor's; both flip above the anchor rather than
   run off the bottom of the window. */
function anchorOverlay(card, anchor, { align = 'left', nudge = 0, gap = 6 } = {}) {
  const rect = anchor.getBoundingClientRect();
  const width = card.offsetWidth;
  const height = card.offsetHeight;
  let top = rect.bottom + gap;
  if (top + height > window.innerHeight - 8) top = rect.top - height - gap;
  const left = (align === 'right' ? rect.right - width : rect.left) + nudge;
  const rightmost = Math.max(8, window.innerWidth - width - 8);
  card.style.left = Math.min(Math.max(8, left), rightmost) + 'px';
  card.style.top = Math.max(8, top) + 'px';
}

/* answered = the card is the reply to a click, not a hover echo: it gets the
   accent edge and the pop, so the user sees their click produced something
   new instead of the card that was already sitting there. */
function showWhyTip(el, answered = false) {
  const card = byId('whyTip');
  const tip = el.dataset.tip;
  if (!tip) { hideWhyTip(); return; }
  card.textContent = tip;
  card.classList.toggle('answered', answered);
  card.hidden = false;
  anchorOverlay(card, el, { nudge: -12 });
}

/* Hover and keyboard focus both ask for it; anything that moves the page
   under the card - scroll, click, Escape - takes it away. One exception: a
   click on a fixed or unsupported pill re-shows it on purpose (see
   renderPills), because that click is the user asking why. */
document.addEventListener('mouseover', (event) => {
  const el = event.target.closest ? event.target.closest('[data-tip]') : null;
  if (el) showWhyTip(el); else hideWhyTip();
});
document.addEventListener('focusin', (event) => {
  const el = event.target.closest ? event.target.closest('[data-tip]') : null;
  if (el) showWhyTip(el);
});
document.addEventListener('focusout', hideWhyTip);
document.addEventListener('mousedown', hideWhyTip);
window.addEventListener('scroll', hideWhyTip, true);

/* --- topbar settings pills --------------------------------------------------
   The pills are a view over the hidden form controls, never a second store:
   every menu writes back into a control and fires its change event, so the
   payload, the validation and the preview keep exactly one source. */

/* Display formatting of the model id, the way fmtLimit abbreviates a number:
   the family name is capitalised and a date suffix is dropped, so an id like
   "claude-<family>-<major>-<minor>-<date>" reads as "Family Major.Minor". It
   invents no fact - an id it cannot parse is shown as it is. */
function shortModelName(id) {
  const match = /^claude-([a-z]+)-(\d+)(?:-(\d+))?(?:-\d{8})?$/.exec(id || '');
  if (!match) return id || '?';
  const name = match[1][0].toUpperCase() + match[1].slice(1);
  return name + ' ' + match[2] + (match[3] ? '.' + match[3] : '');
}

const cap1 = (text) => text ? text[0].toUpperCase() + text.slice(1) : text;

/* Write one hidden control and let its existing listeners do the rest. */
function setControl(id, value) {
  const el = byId(id);
  if (el.value === String(value)) return;
  el.value = value;
  el.dispatchEvent(new Event('change'));
  el.dispatchEvent(new Event('input'));
}

/* The open menu, or null. The id is what a second click on the same pill is
   recognised by, and what tells a re-render which pill is currently open. */
let pillMenu = null;

function anchorPillMenu() {
  if (pillMenu) anchorOverlay(pillMenu.el, pillMenu.pill, { align: 'right' });
}

function markPillOpen(pill, open) {
  pill.classList.toggle('menu-open', open);
  // A pill that has just lost its menu is not collapsed - it is not
  // expandable at all, and must not claim to be.
  if (pill.dataset.menu === 'yes') pill.setAttribute('aria-expanded', String(open));
}

function closePillMenu() {
  if (!pillMenu) return;
  markPillOpen(pillMenu.pill, false);
  pillMenu.el.remove();
  pillMenu = null;
}

function openPillMenu(pill, build) {
  const id = pill.dataset.pill;
  if (pillMenu && pillMenu.id === id) { closePillMenu(); return; }
  closePillMenu();
  const menu = document.createElement('div');
  menu.className = 'pill-menu';
  menu.setAttribute('role', 'menu');
  // Clicks inside belong to the menu; the document listener below is what
  // closes it, and it must not hear them.
  menu.addEventListener('click', (event) => event.stopPropagation());
  // Mounted before it is built, because a field inside a detached node
  // cannot take focus: the Max tokens menu used to open with the caret
  // nowhere and no selection, so typing appended to the old value
  // (16384 + "4096" = "163844096") instead of replacing it.
  byId('pillLayer').appendChild(menu);
  pillMenu = { el: menu, pill, id };
  build(menu);
  // Build and placement are one task, so the menu is never painted at the
  // layer's origin on its way to the pill.
  anchorPillMenu();
  markPillOpen(pill, true);
}

function menuItem(menu, { label, sub = '', selected = false, disabled = false,
                          def = false, onPick = null }) {
  const item = document.createElement('button');
  item.className = 'mi' + (selected ? ' selected' : '') + (disabled ? ' disabled' : '');
  item.innerHTML = '<span class="check">' + (selected ? '✓' : '') + '</span>'
    + '<span class="grow">' + esc(label) + (sub ? '<span class="sub">' + esc(sub) + '</span>' : '')
    + '</span>' + (def ? '<span class="def">default</span>' : '');
  if (onPick && !disabled) {
    item.addEventListener('click', () => { onPick(); });
  }
  menu.appendChild(item);
  return item;
}

function menuNote(menu, html) {
  const note = document.createElement('div');
  note.className = 'mnote';
  note.innerHTML = html;
  menu.appendChild(note);
}

function menuHead(menu, text) {
  const head = document.createElement('div');
  head.className = 'mhead';
  head.textContent = text;
  menu.appendChild(head);
}

/* What the thinking control's on/off is called for this model: the API's own
   type names, since that is what the request will carry. */
function thinkingStateName(value) {
  if (value === 'off') return 'Disabled';
  return state.caps && state.caps.thinking_mode === 'adaptive' ? 'Adaptive' : 'Enabled';
}

const PILL_DEFS = [
  {
    id: 'model', label: 'Model',
    value: () => {
      const select = byId('model');
      const chosen = select.selectedOptions[0];
      // A conversation can be older than the list: say so rather than let a
      // superseded model look like a current one.
      return chosen && chosen.dataset.superseded
        ? shortModelName(select.value) + ' · legacy'
        : shortModelName(select.value);
    },
    menu: (menu) => {
      menuHead(menu, 'Model');
      (state.data.models || []).forEach((id) => {
        menuItem(menu, {
          label: shortModelName(id), sub: id,
          selected: id === byId('model').value,
          onPick: () => { closePillMenu(); setControl('model', id); }
        });
      });
      // The list narrowed on purpose, and a short list that lost nine models
      // without a word reads like missing data rather than like a rule.
      const superseded = (state.data.superseded || []).length;
      if (superseded) {
        menuNote(menu, esc(superseded + ' earlier models not listed · one '
          + 'model per series, newest first'));
      }
    }
  },
  {
    id: 'thinking', label: 'Thinking',
    fixed: () => state.caps && !state.caps.thinking_editable,
    fixedValue: () => cap1(state.caps.thinking_mode),
    fixedTag: () => 'always on',
    fixedTitle: () => state.data.controls.thinking.note,
    value: () => thinkingStateName(byId('thinking').value),
    menu: (menu) => {
      const control = state.data.controls.thinking;
      menuHead(menu, 'Thinking · ' + shortModelName(byId('model').value));
      ['off', 'on'].forEach((value) => {
        if (!control.options.some((o) => o.value === value)) return;
        const sub = value === 'off'
          ? (state.caps.thinking_off_request === 'omitted'
              ? 'the field is omitted - the model default'
              : 'sent as {"type": "disabled"}')
          : (state.caps.thinking_mode === 'adaptive'
              ? 'sent as {"type": "adaptive"}'
              : 'sent as {"type": "enabled", "budget_tokens": '
                + (state.caps.budget_tokens || '?') + '}');
        menuItem(menu, {
          label: thinkingStateName(value), sub,
          selected: byId('thinking').value === value,
          onPick: () => { closePillMenu(); setControl('thinking', value); }
        });
      });
      if (state.caps.thinking_mode === 'extended' && state.caps.budget_tokens) {
        menuNote(menu, 'budget_tokens = ' + fmt(state.caps.budget_tokens)
          + ' — hard-coded by llm-anthropic, so there is nothing to set.');
      }
    }
  },
  {
    id: 'effort', label: 'Effort',
    unsupported: () => state.caps && !state.caps.supports_effort,
    value: () => {
      const value = byId('effort').value;
      return value === 'default' ? cap1(state.caps.default_effort || 'default') : cap1(value);
    },
    menu: (menu) => {
      const control = state.data.controls.effort;
      const off = byId('thinking').value === 'off';
      menuHead(menu, 'Effort · low → max');
      control.options.filter((o) => o.value !== 'default').forEach((option) => {
        const blocked = off && option.blocked_without_thinking;
        const isDefault = option.value === state.caps.default_effort;
        // Choosing the level Anthropic documents as the default stores
        // 'default', so no effort field goes on the request at all.
        const selected = byId('effort').value === 'default' ? isDefault
          : byId('effort').value === option.value;
        menuItem(menu, {
          label: cap1(option.value),
          sub: blocked ? 'rejected by the API with thinking off' : '',
          def: isDefault, selected, disabled: blocked,
          onPick: () => {
            closePillMenu();
            setControl('effort', isDefault ? 'default' : option.value);
          }
        });
      });
      menuNote(menu, 'The level tagged <b>default</b> is this model\'s documented '
        + 'default (' + esc(state.caps.default_effort || 'unknown')
        + '): picking it sends no effort field, so the provider default applies.');
    }
  },
  {
    id: 'maxTokens', label: 'Max tokens',
    value: () => fmt(Number(byId('maxTokens').value))
      + '<span class="sub-max">/' + fmtLimit(state.caps ? state.caps.max_output_tokens : null) + '</span>',
    menu: (menu) => {
      const control = state.data.controls.max_tokens;
      const thinking = byId('thinking').value;
      const min = control.min_by_thinking[thinking] || 1;
      const max = state.caps.max_output_tokens;
      menuHead(menu, 'Max output tokens');
      const field = document.createElement('div');
      field.className = 'field';
      field.innerHTML = '<label>[ value ] / ' + fmt(max) + ' ceiling</label>'
        + '<input type="number" min="' + min + '" max="' + (max || '') + '" value="'
        + esc(byId('maxTokens').value) + '">'
        + '<div class="hint">' + esc(control.note)
        + (min > 1 ? ' · with thinking enabled the legal minimum is ' + fmt(min)
          + ' (above the ' + fmt(state.caps.budget_tokens) + ' budget)' : '')
        + '</div>';
      menu.appendChild(field);
      const input = field.querySelector('input');
      input.addEventListener('change', () => {
        let value = Math.round(Number(input.value) || min);
        value = Math.max(min, max ? Math.min(max, value) : value);
        closePillMenu();
        setControl('maxTokens', value);
      });
      // Selected, so the first keystroke replaces the ceiling rather than
      // extending it. preventScroll: the menu is fixed and already placed.
      input.focus({ preventScroll: true });
      input.select();
    }
  },
  {
    id: 'system', label: 'System',
    value: () => byId('system').value.trim() ? 'On' : 'Off',
    menu: (menu) => {
      menuHead(menu, 'System prompt');
      const field = document.createElement('div');
      field.className = 'field';
      field.innerHTML = '<input type="text" placeholder="empty = Off" value="'
        + esc(byId('system').value) + '">'
        + '<div class="hint">part of the cache prefix - changing it invalidates '
        + 'the cached prefix</div>';
      menu.appendChild(field);
      const input = field.querySelector('input');
      input.addEventListener('change', () => {
        closePillMenu();
        setControl('system', input.value);
      });
      input.focus({ preventScroll: true });
    }
  },
  {
    id: 'webSearch', label: 'Web Search',
    value: () => byId('webSearch').value === 'true' ? 'On' : 'Off',
    menu: (menu) => {
      const on = byId('webSearch').value === 'true';
      menuHead(menu, 'Web search');
      const toggle = document.createElement('button');
      toggle.className = 'mi';
      toggle.innerHTML = '<span class="grow">Enabled</span>'
        + '<span class="switch' + (on ? ' on' : '') + '" role="switch" aria-checked="'
        + on + '"><span class="knob"></span></span>';
      toggle.addEventListener('click', () => {
        closePillMenu();
        setControl('webSearch', on ? 'false' : 'true');
      });
      menu.appendChild(toggle);
      const typeControl = state.data.controls.web_search_type;
      const fields = document.createElement('div');
      fields.innerHTML = '<div class="field"><label>Version (derived from the model)</label>'
        + '<input type="text" value="' + esc(typeControl.value || 'none') + '" disabled></div>'
        + '<div class="field"><label>Max uses · 0 = unlimited (field omitted)</label>'
        + '<input type="number" id="menuMaxUses" min="0" value="' + esc(byId('maxUses').value) + '"></div>'
        + '<div class="field"><label>Allowed callers</label>'
        + '<input type="text" value="' + esc((state.data.controls.allowed_callers.value || 'direct')
          + ' · not exposed by llm-anthropic') + '" disabled></div>'
        + '<div class="field"><label>Response inclusion</label>'
        + '<select id="menuRespIncl"' + (state.data.controls.response_inclusion.status !== 'Editable'
            ? ' disabled' : '') + '>'
        + '<option value="excluded"' + (byId('responseInclusion').value === 'excluded' ? ' selected' : '') + '>excluded</option>'
        + '<option value="full"' + (byId('responseInclusion').value === 'full' ? ' selected' : '') + '>full</option>'
        + '</select>'
        + (state.data.controls.response_inclusion.status !== 'Editable'
            ? '<div class="hint">' + esc(state.data.controls.response_inclusion.note) + '</div>' : '')
        + '</div>';
      menu.appendChild(fields);
      fields.querySelector('#menuMaxUses').addEventListener('change', (event) => {
        setControl('maxUses', Math.max(0, Math.round(Number(event.target.value) || 0)));
      });
      const incl = fields.querySelector('#menuRespIncl');
      if (!incl.disabled) {
        incl.addEventListener('change', () => setControl('responseInclusion', incl.value));
      }
    }
  },
  {
    id: 'cacheControl', label: 'Caching',
    value: () => byId('cacheControl').value === 'true' ? '5m' : 'Off',
    menu: (menu) => {
      const on = byId('cacheControl').value === 'true';
      menuHead(menu, 'Prompt caching');
      menuItem(menu, {
        label: 'On', sub: '5m TTL · the provider default', selected: on,
        onPick: () => { closePillMenu(); setControl('cacheControl', 'true'); }
      });
      menuItem(menu, {
        label: 'Off', sub: "this turn won't read or write the cache", selected: !on,
        onPick: () => { closePillMenu(); setControl('cacheControl', 'false'); }
      });
      menuNote(menu, esc(state.data.controls.cache_control.ttl.note));
    }
  },
  {
    id: 'streaming', label: 'Streaming',
    fixed: () => true,
    fixedValue: () => 'On',
    fixedTag: () => 'runtime fixed',
    fixedTitle: () => state.data ? state.data.controls.stream.note : '',
    value: () => 'On',
    menu: null
  }
];

/* The row is built once and updated in place.

   It used to be thrown away and rebuilt on every change, and because the
   pills re-render whenever any control moves, that happened in the middle
   of the user's own gesture: pressing a pill blurs an open menu's field,
   the field commits, the commit re-renders - and the node the press landed
   on is detached before the button comes back up. The browser then
   dispatches no click at all, so the pill that was aimed at did nothing.
   Rebuilding cost a real interaction and bought nothing, so the eight
   buttons now outlive every render and only their contents change. */
const pillNodes = new Map();

function pillNode(def) {
  const existing = pillNodes.get(def.id);
  if (existing) return existing;
  const pill = document.createElement('button');
  pill.type = 'button';
  pill.dataset.pill = def.id;
  // Bound once, and it reads the pill's current state rather than closing
  // over it, because a pill can lose or gain its menu when the model
  // changes.
  pill.addEventListener('click', (event) => {
    // The document listener below closes menus; opening one must not
    // reach it.
    event.stopPropagation();
    if (pill.dataset.menu === 'yes') {
      openPillMenu(pill, def.menu);
    } else {
      // A pill that cannot open a menu still owes its click an answer - a
      // grey value must say why, never go silent. The answer must read as
      // NEW: the hover already shows this card, so a bare re-show is
      // invisible. The pill shakes "no" and the card pops back answered.
      flash(pill);
      showWhyTip(pill, true);
    }
  });
  pillNodes.set(def.id, pill);
  return pill;
}

function renderPills() {
  const host = byId('settingsPills');
  if (!host || !state.data) return;
  PILL_DEFS.forEach((def) => {
    const pill = pillNode(def);
    const fixed = def.fixed && def.fixed();
    const unsupported = def.unsupported && def.unsupported();
    pill.className = 'pill' + (fixed ? ' fixed' : '') + (unsupported ? ' unsupported' : '');
    let value;
    if (unsupported) value = '—';
    else if (fixed) value = def.fixedValue();
    else value = def.value();
    const tag = unsupported ? 'Unsupported by selected model'
      : fixed && def.fixedTag ? def.fixedTag() : null;
    // Why a pill is the way it is belongs on hover, not in the row: the row
    // is a reading, the reason is a sentence.
    const reason = unsupported
      ? 'this model has no such parameter'
      : fixed && def.fixedTitle ? def.fixedTitle() : '';
    pill.innerHTML = '<span class="plabel">' + def.label + '</span>'
      + '<span class="pvalue">' + value + '</span>'
      + (tag ? whyMark(tag, reason) : '')
      + (!fixed && !unsupported ? '<span class="caret">▾</span>' : '');
    pill.dataset.menu = fixed || unsupported ? 'no' : 'yes';
    if (fixed || unsupported) {
      // The whole pill carries the reason: the dot is the affordance, and a
      // pill with no menu has nothing else to hover.
      pill.dataset.tip = [tag, reason].filter(Boolean).join(' · ');
      pill.removeAttribute('aria-haspopup');
      pill.removeAttribute('aria-expanded');
    } else {
      delete pill.dataset.tip;
      pill.setAttribute('aria-haspopup', 'menu');
      pill.setAttribute('aria-expanded', String(!!pillMenu && pillMenu.id === def.id));
    }
    if (pill.parentElement !== host) host.appendChild(pill);
  });
  if (!pillMenu) return;
  // The model changed and this pill no longer has a menu to be open.
  if (pillMenu.pill.dataset.menu !== 'yes') closePillMenu();
  // A neighbour's value changed width, so the anchor moved under the menu.
  else anchorPillMenu();
}

document.addEventListener('click', () => { closePillMenu(); closeConvMenu(); });
document.addEventListener('keydown', (event) => {
  if (event.key === 'Escape') { closePillMenu(); closeConvMenu(); }
});
// A context menu belongs to the row it opened from; the row scrolling away
// under it is the end of the conversation between them.
window.addEventListener('scroll', closeConvMenu, true);
// A fixed overlay has to be told when its anchor moves. Capture, so the
// pill row's own horizontal scrolling counts too.
window.addEventListener('resize', anchorPillMenu);
window.addEventListener('scroll', anchorPillMenu, true);

/* Client-side latency: llm measured the whole call; ttft is this page's
   time to first word. Nothing here is the Console's server-side figure -
   that is not available through the API, so it is never implied. */
function latencyLabel(response) {
  const ms = response && response.duration_ms;
  if (typeof ms !== 'number' || ms < 0) return null;
  let label = 'latency ' + (ms / 1000).toFixed(2) + ' s (client-measured, includes streaming)';
  const out = response.usage && response.usage.output_tokens;
  if (typeof out === 'number' && out > 0) {
    label += ' · ' + out + ' tok · ' + Math.round(out / (ms / 1000)) + ' tok/s';
  }
  return label;
}

/* The line under an assistant bubble: how long the turn took, how much it
   said, why it stopped, and a copy of everything it said. The figures are
   llm's own measurement and the provider's own usage; this page adds
   nothing to them. */
function attachMeta(bodyEl, response) {
  const parts = [];
  const ms = response && response.duration_ms;
  if (typeof ms === 'number' && ms >= 0) parts.push((ms / 1000).toFixed(2) + ' s');
  const out = response.usage && response.usage.output_tokens;
  if (typeof out === 'number' && out > 0) parts.push(out + ' out');
  if (response.stop_reason) parts.push(response.stop_reason);
  if (!parts.length) return;
  const meta = document.createElement('div');
  meta.className = 'msg-meta';
  const label = document.createElement('span');
  label.textContent = parts.join(' · ');
  meta.appendChild(label);
  const copy = document.createElement('button');
  copy.type = 'button';
  copy.className = 'copy-all';
  copy.title = 'Copy this answer (and its thinking, if any)';
  copy.textContent = '⧉ copy all';
  meta.appendChild(copy);
  meta.title = 'client-measured wall time from dispatch to stream end';
  bodyEl.parentElement.appendChild(meta);
}

function setCode(text, lang) {
  const el = byId('codeContent');
  el.textContent = text;
  el.className = 'language-' + (lang || 'none');
  byId('codeLang').textContent = { python: 'Python', json: 'JSON' }[lang] || 'Text';
  if (window.Prism) Prism.highlightElement(el);
}

function setInspectorEmpty(title, sub) {
  byId('inspector').classList.add('is-empty');
  byId('emptyTitle').textContent = title;
  byId('emptySub').textContent = sub;
}

function clearInspectorEmpty() {
  byId('inspector').classList.remove('is-empty');
}

/* Boolean controls are switches; the hidden select stays the single value
   the payload and every existing code path reads. */
const switchSyncers = [];

function wireSwitch(selectId, switchId) {
  const select = byId(selectId);
  const sw = byId(switchId);
  const sync = () => {
    const on = select.value === 'true';
    sw.classList.toggle('on', on);
    sw.setAttribute('aria-checked', on ? 'true' : 'false');
    sw.disabled = select.disabled;
  };
  sw.addEventListener('click', () => {
    if (select.disabled) return;
    select.value = select.value === 'true' ? 'false' : 'true';
    select.dispatchEvent(new Event('change'));
    select.dispatchEvent(new Event('input'));
    sync();
  });
  switchSyncers.push(sync);
  sync();
}

function syncSwitches() {
  switchSyncers.forEach((sync) => sync());
}

function setBlocked(text, isError) {
  const el = byId('blockedNote');
  el.textContent = text || '';
  el.className = 'blocked' + (text ? ' on' : '') + (isError ? ' error' : '');
}

/* --- markdown-lite, built as DOM ------------------------------------------
   Model output is text, never HTML: every element below is created and
   filled with textContent, so a reply can contain anything and still only
   ever render as the markup this function built. Headings, lists, bold /
   italic / inline code, fenced code with highlighting and a copy button,
   tables, and links - http(s) only, anything else stays literal text. */

const INLINE_MARK = /(`[^`\n]+`)|(\[([^\]]+)\]\((https?:\/\/[^)\s]+)\))|(\*\*([^*]+)\*\*)|(\*([^*\n]+)\*)|(^|[^A-Za-z0-9])_([^_\n]+)_(?![A-Za-z0-9])/g;

function appendInline(parent, text) {
  // The recursion into bold/italic needs its own regex instance: a shared
  // /g regex carries lastIndex across calls, and a nested scan would reset
  // the outer one's position mid-string - matching the same span forever.
  const mark = new RegExp(INLINE_MARK.source, 'g');
  let last = 0;
  let match;
  while ((match = mark.exec(text)) !== null) {
    // An underscore emphasis keeps its delimiter prefix as plain text.
    const start = match[10] ? match.index + match[9].length : match.index;
    if (start > last) {
      parent.appendChild(document.createTextNode(text.slice(last, start)));
    }
    if (match[1]) {
      const code = document.createElement('code');
      code.textContent = match[1].slice(1, -1);
      parent.appendChild(code);
    } else if (match[2]) {
      const link = document.createElement('a');
      link.href = match[4];
      link.target = '_blank';
      link.rel = 'noopener noreferrer';
      link.textContent = match[3];
      parent.appendChild(link);
    } else if (match[5]) {
      const strong = document.createElement('strong');
      appendInline(strong, match[6]);
      parent.appendChild(strong);
    } else {
      const em = document.createElement('em');
      appendInline(em, match[8] || match[10]);
      parent.appendChild(em);
    }
    last = mark.lastIndex;
  }
  if (text.length > last) {
    parent.appendChild(document.createTextNode(text.slice(last)));
  }
}

/* A fenced block: the language label, a copy button, and the code itself,
   highlighted only when the vendored Prism actually speaks the language. */
function codeBlockElement(lang, codeText) {
  const wrap = document.createElement('div');
  wrap.className = 'codeblock';
  const head = document.createElement('div');
  head.className = 'codehead';
  const name = document.createElement('span');
  name.className = 'codelang';
  name.textContent = lang || 'code';
  const copy = document.createElement('button');
  copy.type = 'button';
  copy.className = 'copycode';
  copy.title = 'Copy this code block';
  copy.textContent = '⧉';
  head.appendChild(name);
  head.appendChild(copy);
  const pre = document.createElement('pre');
  const code = document.createElement('code');
  code.textContent = codeText;
  if (lang && window.Prism && Prism.languages[lang]) {
    code.className = 'language-' + lang;
    Prism.highlightElement(code);
  }
  pre.appendChild(code);
  wrap.appendChild(head);
  wrap.appendChild(pre);
  return wrap;
}

function splitRow(line) {
  let cells = line.trim();
  if (cells.startsWith('|')) cells = cells.slice(1);
  if (cells.endsWith('|')) cells = cells.slice(0, -1);
  return cells.split('|').map((cell) => cell.trim());
}

/* The divider row under a table header: dashes with optional colons. */
function isTableDivider(line) {
  const cells = splitRow(line);
  return cells.length > 0 && cells.every((cell) => /^:?-+:?$/.test(cell));
}

function tableElement(headerLine, rowLines) {
  const wrap = document.createElement('div');
  wrap.className = 'tablewrap';
  const table = document.createElement('table');
  const thead = document.createElement('thead');
  const headRow = document.createElement('tr');
  splitRow(headerLine).forEach((cell) => {
    const th = document.createElement('th');
    appendInline(th, cell);
    headRow.appendChild(th);
  });
  thead.appendChild(headRow);
  const tbody = document.createElement('tbody');
  rowLines.forEach((line) => {
    const tr = document.createElement('tr');
    splitRow(line).forEach((cell) => {
      const td = document.createElement('td');
      appendInline(td, cell);
      tr.appendChild(td);
    });
    tbody.appendChild(tr);
  });
  table.appendChild(thead);
  table.appendChild(tbody);
  wrap.appendChild(table);
  return wrap;
}

const MD_HEADING = /^(#{1,6})\s+(.*)$/;
const MD_UL = /^\s*[-*+]\s+(.*)$/;
const MD_OL = /^\s*\d+[.)]\s+(.*)$/;
const MD_QUOTE = /^>\s?(.*)$/;
const MD_FENCE = /^```(\w*)\s*$/;

/* A line starts a table when it is a row and the next line is the divider. */
function startsTable(lines, i) {
  return lines[i].trim().startsWith('|') && lines[i].includes('|', 1)
    && i + 1 < lines.length && isTableDivider(lines[i + 1]);
}

function safeMarkdown(text) {
  const frag = document.createDocumentFragment();
  const lines = String(text).split('\n');
  let i = 0;
  while (i < lines.length) {
    const line = lines[i];
    if (!line.trim()) { i += 1; continue; }
    const fence = MD_FENCE.exec(line);
    if (fence) {
      const buf = [];
      i += 1;
      while (i < lines.length && !MD_FENCE.test(lines[i])) { buf.push(lines[i]); i += 1; }
      i += 1; // the closing fence, or the end of an unterminated block
      frag.appendChild(codeBlockElement(fence[1], buf.join('\n')));
      continue;
    }
    if (startsTable(lines, i)) {
      const rows = [];
      const header = line;
      i += 2;
      while (i < lines.length && lines[i].trim().startsWith('|')) {
        rows.push(lines[i]);
        i += 1;
      }
      frag.appendChild(tableElement(header, rows));
      continue;
    }
    const heading = MD_HEADING.exec(line);
    if (heading) {
      // A bubble already sits inside a page with an h1/h2, so the reply's
      // own outline starts at h3.
      const level = Math.min(heading[1].length + 2, 6);
      const h = document.createElement('h' + level);
      appendInline(h, heading[2]);
      frag.appendChild(h);
      i += 1;
      continue;
    }
    if (MD_UL.test(line) || MD_OL.test(line)) {
      const ordered = !MD_UL.test(line);
      const itemRe = ordered ? MD_OL : MD_UL;
      const list = document.createElement(ordered ? 'ol' : 'ul');
      while (i < lines.length) {
        const item = itemRe.exec(lines[i]);
        if (!item) break;
        const li = document.createElement('li');
        appendInline(li, item[1]);
        list.appendChild(li);
        i += 1;
      }
      frag.appendChild(list);
      continue;
    }
    if (MD_QUOTE.test(line)) {
      const bq = document.createElement('blockquote');
      while (i < lines.length) {
        const quoted = MD_QUOTE.exec(lines[i]);
        if (!quoted) break;
        const p = document.createElement('p');
        appendInline(p, quoted[1]);
        bq.appendChild(p);
        i += 1;
      }
      frag.appendChild(bq);
      continue;
    }
    const p = document.createElement('p');
    const buf = [line];
    i += 1;
    // A paragraph runs until a blank line or the start of another block.
    while (i < lines.length && lines[i].trim()
           && !MD_HEADING.test(lines[i]) && !MD_UL.test(lines[i])
           && !MD_OL.test(lines[i]) && !MD_QUOTE.test(lines[i])
           && !MD_FENCE.test(lines[i]) && !startsTable(lines, i)) {
      buf.push(lines[i]);
      i += 1;
    }
    appendInline(p, buf.join('\n'));
    frag.appendChild(p);
  }
  return frag;
}

function renderMarkdown(container, text) {
  container.replaceChildren(safeMarkdown(text));
}

/* One bubble frame for both roles: the head (who · when) over a body. */
function msgShell(role, turnNo, when) {
  const div = document.createElement('div');
  div.className = 'msg ' + role;
  div.innerHTML = '<div class="msg-head"><span class="who"></span>'
    + '<time class="msg-time"></time></div><div class="body"></div>';
  if (turnNo) div.dataset.turn = String(turnNo - 1);
  div.querySelector('.who').textContent = turnNo ? role + ' · turn ' + turnNo : role;
  setMessageTime(div, when);
  messages.appendChild(div);
  followStream();
  return div;
}

function addMessage(role, text, turnNo, when) {
  const div = msgShell(role, turnNo, when);
  const body = div.querySelector('.body');
  body.textContent = text;
  return body;
}

/* An assistant bubble is regions over one body: the thinking strip, the
   tool strip, the answer, the sources. What a turn never produced stays
   hidden rather than drawn blank. */
function addAssistantMessage(turnNo, when) {
  const div = msgShell('assistant', turnNo, when);
  const body = div.querySelector('.body');
  body.innerHTML = '<button type="button" class="think-strip" hidden>'
    + '<span class="chev">▸</span><span class="think-label"></span></button>'
    + '<div class="think-body" hidden></div>'
    + '<button type="button" class="tool-strip" hidden>'
    + '<span class="chev">▸</span><span class="tool-label"></span></button>'
    + '<div class="tool-list" hidden></div>'
    + '<div class="answer"></div>'
    + '<button type="button" class="sources-head" hidden>'
    + '<span class="chev">▸</span><span class="sources-label"></span></button>'
    + '<div class="sources-list" hidden></div>';
  return body;
}

function assistantParts(body) {
  return {
    thinkStrip: body.querySelector('.think-strip'),
    thinkLabel: body.querySelector('.think-label'),
    thinkBody: body.querySelector('.think-body'),
    toolStrip: body.querySelector('.tool-strip'),
    toolLabel: body.querySelector('.tool-label'),
    toolList: body.querySelector('.tool-list'),
    answer: body.querySelector('.answer'),
    sourcesHead: body.querySelector('.sources-head'),
    sourcesLabel: body.querySelector('.sources-label'),
    sourcesList: body.querySelector('.sources-list')
  };
}

function wordCount(text) {
  return text.trim().split(/\s+/).filter(Boolean).length;
}

/* The label a finished thinking strip carries: how long it took when this
   page measured it, how much there is either way. A stored turn has no
   clock of its own, so it names none instead of inventing one. */
function thoughtLabel(thinking, seconds) {
  const words = wordCount(thinking) + ' words';
  return typeof seconds === 'number'
    ? 'Thought for ' + seconds.toFixed(1) + 's · ' + words
    : 'Thought · ' + words;
}

function showFinishedThinking(parts, thinking, seconds) {
  parts.thinkStrip.hidden = false;
  parts.thinkStrip.classList.remove('open');
  parts.thinkStrip.querySelector('.chev').textContent = '▸';
  parts.thinkBody.hidden = true;
  parts.thinkBody.classList.remove('live');
  parts.thinkBody.textContent = thinking;
  parts.thinkLabel.textContent = thoughtLabel(thinking, seconds);
}

/* What the model looked up, from the server tool blocks on the Message:
   one line per query behind a strip that names the first. */
function showToolUse(parts, serverToolBlocks) {
  const queries = (serverToolBlocks || [])
    .filter((b) => b && b.type === 'server_tool_use' && b.name === 'web_search')
    .map((b) => b.input && b.input.query)
    .filter((q) => typeof q === 'string' && q);
  if (!queries.length) return;
  parts.toolStrip.hidden = false;
  parts.toolLabel.textContent = queries.length === 1
    ? 'Searched the web: "' + queries[0] + '"'
    : 'Searched the web ' + queries.length + ' times';
  queries.forEach((q) => {
    const row = document.createElement('div');
    row.textContent = '"' + q + '"';
    parts.toolList.appendChild(row);
  });
}

/* The pages an answer cites, deduplicated by URL; a citation with no
   linkable address is not a source and is not listed as one. */
function showSources(parts, citations) {
  const seen = new Set();
  const links = [];
  (citations || []).forEach((citation) => {
    const url = citation && citation.url;
    if (typeof url !== 'string' || !/^https?:\/\//.test(url) || seen.has(url)) return;
    seen.add(url);
    links.push(citation);
  });
  if (!links.length) return;
  parts.sourcesHead.hidden = false;
  parts.sourcesLabel.textContent = 'Sources (' + links.length + ')';
  links.forEach((citation) => {
    const a = document.createElement('a');
    a.href = citation.url;
    a.target = '_blank';
    a.rel = 'noopener noreferrer';
    a.textContent = citation.title || citation.url;
    parts.sourcesList.appendChild(a);
  });
}

/* A stored turn redraws the same regions the stream filled live, read off
   the record alone. */
function renderAssistantRecord(body, response) {
  const parts = assistantParts(body);
  if (response.thinking) showFinishedThinking(parts, response.thinking, null);
  showToolUse(parts, response.server_tool_blocks);
  renderMarkdown(parts.answer, response.text || '');
  showSources(parts, response.citations);
}

/* The time above a bubble. The record's own stamp is the only clock a turn
   has, so a bubble with no stamp to show says nothing instead of guessing:
   there is no moment at which "now" was this message. */
function setMessageTime(div, when) {
  const el = div.querySelector('.msg-time');
  if (!el) return;
  const moment = momentOf(when);
  if (!moment) {
    el.hidden = true;
    el.removeAttribute('datetime');
    el.textContent = '';
    return;
  }
  el.hidden = false;
  el.dateTime = moment.toISOString();
  el.textContent = formatMoment(moment);
  el.title = TIME_FULL.format(moment);
}

/* A bubble is a link into the pane that carries it: what the user sent is
   the Request, what Claude answered is the Response. Both ends are one
   record, so a click only says which of its two panes to show. The pane is
   written on the bubble and one listener on the list answers every click -
   a bubble wired after that listener exists (or wired twice) still works,
   which a handler per bubble could not promise. */
function wireBubble(div, turnIndex, tab, label) {
  if (!div) return;
  div.classList.add('linked');
  div.dataset.pane = tab;
  // The turn is written where the bubble is wired, not left to the 1-based
  // number addMessage happened to be given: one writer for both, so a click
  // cannot read a different turn than the one this bubble was made for.
  div.dataset.turn = String(turnIndex);
  div.title = label;
}

/* A click on the turn already shown on the right changes nothing else on
   screen, so the bubble answers itself - without this, clicking the most
   recent turn looks exactly like clicking nothing. */
function flash(bubble) {
  bubble.classList.remove('hit');
  void bubble.offsetWidth; // restart the animation on a repeated click
  bubble.classList.add('hit');
  // Outlives the longest .hit animation (pill-shake, 360ms): removing the
  // class early would cut the motion mid-frame.
  setTimeout(() => bubble.classList.remove('hit'), 400);
}

async function copyText(text) {
  try {
    await navigator.clipboard.writeText(text);
    return true;
  } catch (ex) {
    return false;
  }
}

/* A copy control inside a bubble: a code block's ⧉ copies that block; the
   meta row's ⧉ copy all copies the turn's whole answer, thinking included.
   Both say ✓ for a moment, and neither is the bubble asking for its pane. */
function copyControl(button) {
  let text = '';
  if (button.classList.contains('copycode')) {
    const block = button.closest('.codeblock');
    text = block ? block.querySelector('code').textContent : '';
  } else {
    const msg = button.closest('.msg[data-turn]');
    const record = msg && state.turns[Number(msg.dataset.turn)];
    if (!record) return;
    const thinking = record.response.thinking;
    text = thinking ? thinking + '\n\n' + record.response.text : record.response.text;
  }
  copyText(text);
  const label = button.textContent;
  button.textContent = '✓';
  button.classList.add('done');
  setTimeout(() => {
    button.textContent = label;
    button.classList.remove('done');
  }, 1200);
}

/* The collapsible regions of an assistant bubble - thinking, tool use,
   sources - share one toggle: open shows the region, closed hides it, and
   the chevron says which. */
function toggleRegion(button) {
  const body = button.parentElement;
  const region = button.classList.contains('think-strip')
    ? body.querySelector('.think-body')
    : button.classList.contains('tool-strip')
      ? body.querySelector('.tool-list')
      : body.querySelector('.sources-list');
  const open = region.hidden;
  region.hidden = !open;
  button.classList.toggle('open', open);
  button.querySelector('.chev').textContent = open ? '▾' : '▸';
}

messages.addEventListener('click', (event) => {
  if (!(event.target instanceof Element)) return;
  // Controls inside a bubble answer their own click first: copy buttons,
  // the region toggles, links out. None of them is the bubble asking for
  // its pane, so the pane logic below never sees them.
  const copy = event.target.closest('.copycode, .copy-all');
  if (copy) { copyControl(copy); return; }
  const toggle = event.target.closest('.think-strip, .tool-strip, .sources-head');
  if (toggle) { toggleRegion(toggle); return; }
  if (event.target.closest('a')) return;
  const bubble = event.target.closest('.msg[data-pane]');
  if (!bubble) return;
  const index = Number(bubble.dataset.turn);
  if (!Number.isInteger(index) || !state.turns[index]) return;
  // The click wins over a preview the last keystroke left queued, which
  // would otherwise land a moment later and put the draft back.
  cancelPreview();
  flash(bubble);
  showTurn(index, bubble.dataset.pane);
});

function markSelected() {
  messages.querySelectorAll('.msg').forEach((div) => {
    div.classList.toggle('selected',
      state.selected !== null && Number(div.dataset.turn) === state.selected);
  });
}

/* --- a past turn's settings: card, divider, cache hint ----------------------
   Everything here reads the turn's stored effective_options - what the form
   asked for that turn. Absence in the rendered request means the provider
   default applied, never that a value was lost, so the card says so once in
   its footer instead of repeating it per row. */

function currentFormOptions() {
  return {
    model: byId('model').value,
    max_tokens: Number(byId('maxTokens').value),
    system: byId('system').value,
    thinking: byId('thinking').value,
    effort: byId('effort').value,
    web_search: byId('webSearch').value === 'true',
    web_search_type: byId('webSearchType').value,
    allowed_callers: byId('allowedCallers').value,
    response_inclusion: byId('responseInclusion').value,
    max_uses: Number(byId('maxUses').value),
    cache_control: byId('cacheControl').value === 'true'
  };
}

const trim40 = (v) => {
  const text = String(v ?? '');
  return text.length > 40 ? text.slice(0, 40) + '…' : text;
};

const CARD_FIELDS = [
  ['model', 'Model', (v) => shortModelName(v)],
  ['max_tokens', 'Max tokens', (v) => fmt(v)],
  ['system', 'System', (v) => trim40(v) || '—'],
  ['thinking', 'Thinking', (v) => v || '—'],
  ['effort', 'Effort', (v) => v || '—'],
  ['web_search', 'Web search', (v, o) => v ? 'on · ' + (o.web_search_type || '?') : 'off'],
  ['max_uses', 'Max uses', (v) => String(v)],
  ['cache_control', 'Prompt cache', (v) => v ? 'on · 5m' : 'off']
];

const DIFF_FIELDS = [
  ['model', (v) => shortModelName(v)],
  ['max_tokens', (v) => fmt(v)],
  ['system', null],
  ['thinking', null],
  ['effort', null],
  ['web_search', (v) => v ? 'on' : 'off'],
  ['web_search_type', null],
  ['response_inclusion', null],
  ['max_uses', null],
  ['cache_control', (v) => v ? 'on' : 'off']
];

function settingsDiffLines(prev, cur) {
  const parts = [];
  DIFF_FIELDS.forEach(([key, format]) => {
    const a = prev[key];
    const b = cur[key];
    if (String(a ?? '') === String(b ?? '')) return;
    if (key === 'system') { parts.push('system changed'); return; }
    const fa = format ? format(a) : String(a ?? '—');
    const fb = format ? format(b) : String(b ?? '—');
    parts.push(key.replace(/_/g, ' ') + ' ' + fa + ' → ' + fb);
  });
  return parts;
}

/* The dashed rule between two turns whose settings differ, naming exactly
   which fields moved - inserted before the later turn's user bubble. */
function dividerFor(diff) {
  const divider = document.createElement('div');
  divider.className = 'settings-divider';
  divider.innerHTML = 'settings changed<span class="diff">' + esc(diff.join(' · ')) + '</span>';
  return divider;
}

function diffBetweenTurns(prevRecord, record) {
  const prev = (prevRecord && prevRecord.options) || {};
  const cur = (record && record.options) || {};
  if (!Object.keys(prev).length || !Object.keys(cur).length) return [];
  return settingsDiffLines(prev, cur);
}

/* One hover card, two readers. Both bubbles answer a hover - the user's with
   the settings that went out, the answer's with what it cost - and both are
   read-only: #settingsCard is fixed and pointer-events: none, so it can never
   swallow the click that picks the turn it is describing. The build function
   is asked at hover time, so a card always describes the record it is over
   rather than the one that existed when the bubble was wired. */
let settingsCardTimer = null;

function placeHoverCard(card, anchor) {
  const rect = anchor.getBoundingClientRect();
  const width = card.offsetWidth;
  const height = card.offsetHeight;
  // The current bubble's own left edge, never off the side of the window.
  card.style.left = Math.min(Math.max(8, rect.left),
    Math.max(8, window.innerWidth - width - 8)) + 'px';
  let top = rect.top - height - 8;
  if (top < 8) top = rect.bottom + 8;
  // A receipt is taller than a settings card; past the bottom of the window
  // it is clamped rather than pushed out of reach.
  card.style.top = Math.min(Math.max(8, top),
    Math.max(8, window.innerHeight - height - 8)) + 'px';
}

function hoverCard(div, build, extraClass) {
  if (!div) return;
  div.addEventListener('mouseenter', () => {
    clearTimeout(settingsCardTimer);
    const html = build();
    if (!html) return;
    const card = byId('settingsCard');
    // The class is set, not toggled: whichever card this is replaces the last
    // one whole, so a receipt can never keep the settings card's width.
    card.className = 'scard' + (extraClass ? ' ' + extraClass : '');
    card.innerHTML = html;
    card.hidden = false;
    placeHoverCard(card, div);
  });
  div.addEventListener('mouseleave', () => {
    settingsCardTimer = setTimeout(() => { byId('settingsCard').hidden = true; }, 120);
  });
}

/* What the request asked for, on the bubble that made it. */
function settingsCardHtml(turnIndex) {
  const record = state.turns[turnIndex];
  if (!record) return '';
  const options = record.options || {};
  const prev = turnIndex > 0 ? (state.turns[turnIndex - 1].options || {}) : null;
  let html = '<h5>turn ' + (turnIndex + 1) + ' · as sent</h5>';
  if (!Object.keys(options).length) {
    // A conversation written elsewhere (llm -c, another plugin) has no
    // sidecar row; the card says so instead of inventing values.
    html += '<div class="srow"><span class="slabel">settings were not recorded'
      + ' for this turn</span></div>';
  } else {
    html += CARD_FIELDS.map(([key, label, format]) => {
      const value = format(options[key], options);
      const diff = prev && Object.keys(prev).length
        && String(options[key] ?? '') !== String(prev[key] ?? '');
      return '<div class="srow' + (diff ? ' diff' : '') + '"><span class="slabel">'
        + esc(label) + '</span><span class="svalue">' + esc(value) + '</span></div>';
    }).join('');
  }
  html += '<div class="sfoot">absent fields = provider default · the Request'
    + ' pane shows exactly what went out</div>';
  return html;
}

function wireSettingsCard(div, turnIndex) {
  hoverCard(div, () => settingsCardHtml(turnIndex), '');
}

function wireTurnCostCard(div, turnIndex) {
  hoverCard(div, () => {
    const record = state.turns[turnIndex];
    return record ? costCardHtml(record, turnIndex) : '';
  }, 'costcard');
}

/* The cache prefix is system + tools + messages, so only a change to the
   model, the system prompt or the web-search tool's shape invalidates the
   cached prefix. max_tokens, effort, thinking and allowed_callers are not
   part of it, so they are reported as a difference and never as a cache
   miss - a warning that blames the cache for a change that cannot cost a
   hit is a wrong warning. With caching off the message is not "invalidated"
   - the turn simply will not read or write the cache. */
const CACHE_PREFIX_FIELDS = [
  ['model', 'model'], ['system', 'system'],
  ['web_search', 'web search'], ['web_search_type', 'search version'],
  ['max_uses', 'max uses'], ['response_inclusion', 'response inclusion']
];

/* Not part of the prefix, so they must never be reported as a cache miss -
   but the request they produce does differ from the last turn, and the user
   is told so. `allowed_callers` is here because llm-anthropic cannot send
   it at all: the value in effect is the API default on every request. */
const OTHER_REQUEST_FIELDS = [
  ['thinking', 'thinking'], ['effort', 'effort'],
  ['max_tokens', 'max output tokens'],
  ['allowed_callers', 'allowed callers']
];

function updateCacheWarn() {
  const el = byId('cacheWarn');
  const text = byId('cacheWarnText');
  // Hiding is not enough: a sentence left behind in the node is a claim
  // about this form that nothing has withdrawn, and reading the DOM then
  // shows a warning the page is not making.
  const hide = () => { el.hidden = true; text.textContent = ''; };
  const last = state.turns[state.turns.length - 1];
  const lastOptions = last && last.options;
  if (!lastOptions || !Object.keys(lastOptions).length) {
    hide();
    return;
  }
  if (byId('cacheControl').value !== 'true') {
    text.innerHTML = 'prompt cache is <b>off</b> — this turn won\'t read or'
      + ' write the cache';
    el.hidden = false;
    return;
  }
  const form = currentFormOptions();
  const differs = (fields) => fields
    .filter(([key]) => String(form[key] ?? '') !== String(lastOptions[key] ?? ''))
    .map(([, label]) => label);
  const prefixChanged = differs(CACHE_PREFIX_FIELDS);
  const otherChanged = differs(OTHER_REQUEST_FIELDS);
  if (!prefixChanged.length && !otherChanged.length) {
    hide();
    return;
  }
  // Only a prefix change can cost a cache hit, so only a prefix change says
  // so. The rest still differs from the last turn, and still gets named.
  if (!prefixChanged.length) {
    text.innerHTML = '<b>' + esc(otherChanged.join(', ')) + '</b> differ from'
      + ' the last turn <span class="why">· none of these touch the cached'
      + ' prefix</span>';
    el.hidden = false;
    return;
  }
  text.innerHTML = '<b>' + esc(prefixChanged.join(', ')) + '</b> changed since the'
    + ' last turn — this turn won\'t reuse the cached prefix'
    + (otherChanged.length
      ? ' <span class="why">· also ' + esc(otherChanged.join(', '))
        + ', which don\'t affect the cache</span>'
      : ' <span class="why">· max output tokens, effort, thinking and allowed'
        + ' callers don\'t affect the cache</span>');
  el.hidden = false;
}

function payload(text) {
  return {
    session_id: 'default',
    text,
    // Set once the first turn has been saved, empty while this really is
    // a new conversation: the server then issues llm's own ULID.
    conversation_id: state.conversationId,
    new_conversation: state.fresh === true,
    model: byId('model').value,
    max_tokens: Number(byId('maxTokens').value),
    system: byId('system').value,
    thinking: byId('thinking').value,
    effort: byId('effort').value,
    web_search: byId('webSearch').value === 'true',
    web_search_type: byId('webSearchType').value,
    allowed_callers: byId('allowedCallers').value,
    response_inclusion: byId('responseInclusion').value,
    max_uses: Number(byId('maxUses').value),
    cache_control: byId('cacheControl').value === 'true'
  };
}

function setOptions(select, options, labels) {
  select.innerHTML = '';
  options.forEach((option) => {
    const el = document.createElement('option');
    el.value = option.value;
    el.textContent = labels[option.value] || option.value;
    el.disabled = !!option.disabled;
    select.appendChild(el);
  });
}

const fmt = (n) => (n === null || n === undefined) ? 'unknown' : n.toLocaleString('en-US');

function effortLabels() {
  const control = state.data.controls.effort;
  const off = byId('thinking').value === 'off';
  const labels = {};
  control.options.forEach((option) => {
    const blocked = off && option.blocked_without_thinking;
    labels[option.value] = option.value === 'default'
      ? (state.caps.default_effort
          ? 'default (model default: ' + state.caps.default_effort + ')'
          : 'default')
      : option.value + (blocked ? ' (rejected with thinking off)' : '');
  });
  return labels;
}

/* The effort control, rebuilt from the loaded model's own capabilities.
 *
 * `chosen` is the level the control should end up holding; omitted, it keeps
 * the level it is already showing. It is read before the list is replaced,
 * because replacing a select's options clears its selection - a rebuild that
 * discarded a value the new list still offers would silently change the
 * request, and the composer's "differs from the last turn" warning would
 * then blame the user for it. A level this model does not offer - a leftover
 * from the previous one, or one thinking has since made illegal - falls back
 * to the provider's own level rather than being sent. */
function applyEffortState(chosen) {
  const control = state.data.controls.effort;
  const off = byId('thinking').value === 'off';
  const labels = effortLabels();
  const select = byId('effort');
  const wanted = chosen === undefined ? select.value : chosen;
  const options = control.options.map((option) => ({
    value: option.value,
    // A level Anthropic rejects with thinking off is not selectable.
    disabled: !state.caps.supports_effort || (off && option.blocked_without_thinking)
  }));
  setOptions(select, options, labels);
  select.disabled = !state.caps.supports_effort;
  const kept = Array.from(select.options)
    .find((option) => option.value === wanted && !option.disabled);
  select.value = kept ? wanted : 'default';
  // Replacing the options clears the value to '' rather than picking the
  // first one. "default" is the provider's own level and thinking never
  // blocks it, so this is only a guard against a future model that does.
  if (!select.value || (select.selectedOptions[0] && select.selectedOptions[0].disabled)) {
    const first = Array.from(select.options).find((option) => !option.disabled);
    if (first) select.value = first.value;
  }
  byId('effortNote').innerHTML = whyMark(control.status, state.caps.supports_effort
    ? 'sent as output_config.effort'
    : control.note);
}

function applyThinkingState() {
  const thinking = state.data.controls.thinking;
  const budget = state.data.controls.budget_tokens;
  const on = byId('thinking').value === 'on';

  byId('budgetTokens').value = budget.value === null ? '' : budget.value;
  byId('budgetTokens').disabled = true;
  byId('budgetNote').innerHTML = whyMark(budget.status,
    (budget.value === null || !on)
      ? budget.note
      : 'sent as thinking.budget_tokens; ' + budget.note);

  const control = state.data.controls.max_tokens;
  byId('maxTokens').min = control.min_by_thinking[byId('thinking').value] || 1;
  byId('maxTokensHint').innerHTML = whyMark(control.status, control.note
    + (on && budget.value !== null
        ? '; must stay above the ' + budget.value + ' token thinking budget'
        : ''));
}

/* The window size reads better abbreviated: 1,000,000 -> 1M, 200,000 -> 200k. */
function fmtLimit(n) {
  if (n === null || n === undefined) return '?';
  if (n >= 1000000) return (n % 1000000 === 0 ? n / 1000000 : (n / 1000000).toFixed(1)) + 'M';
  if (n >= 1000) return Math.round(n / 1000) + 'k';
  return String(n);
}

/* One decimal while it is small, whole numbers once it matters: 0.8% -> 74%.
   A value that rounds to nothing says so instead of claiming zero. */
function fmtPct(p) {
  if (p === null || p === undefined) return null;
  if (p === 0) return '0';
  if (p < 0.1) return '<0.1';
  if (p < 10) return String(Math.round(p * 10) / 10);
  return String(Math.round(p));
}

/* Green while there is room, amber past 70%, red past 90% or when the
   request no longer fits - the same escalation Claude's own UI uses. */
function contextLevel(context) {
  if (context.fits === false) return 'danger';
  const pct = context.percent;
  if (pct === null || pct === undefined) return '';
  if (pct >= 90) return 'danger';
  if (pct >= 70) return 'warn';
  return '';
}

/* A count and a usage report are the provider's own numbers and need no
   qualification. Everything else is partly ours, and says which part. */
const CONTEXT_LABEL = {
  'estimated': ' · estimated',
  'API usage + estimated draft': ' · draft estimated',
  'unknown': ' · cannot be measured'
};

/* The API's counter answers in the background, so the first preview of a new
   request shows a labelled fallback and says a count is coming. Look again for
   it - a bounded number of times, so a slow or unreachable API cannot turn the
   meter into a polling loop. */
const CONTEXT_LOOKS = 3;
let contextLooksLeft = CONTEXT_LOOKS;
let contextLookTimer = null;

function lookAgainForTheCount(context) {
  if (!context || !context.pending) return;
  if (contextLookTimer !== null || contextLooksLeft <= 0) return;
  contextLooksLeft -= 1;
  contextLookTimer = setTimeout(() => {
    contextLookTimer = null;
    updatePreview();
  }, 1200);
}

function showContext(context) {
  if (!context) return;
  const level = contextLevel(context);
  const pctText = fmtPct(context.percent);
  const tokens = context.tokens === null || context.tokens === undefined
    ? 'unknown' : fmt(context.tokens);
  let html = esc(tokens) + ' / ' + esc(fmtLimit(context.limit)) + ' tokens';
  if (pctText !== null) {
    html += ' · <span class="pct' + (level ? ' ' + level : '') + '">'
      + pctText + '%</span>';
  }
  byId('contextUsage').innerHTML = html;
  const note = byId('contextNote');
  byId('contextLabel').closest('.meter-block').classList.toggle('over', context.fits === false);
  if (context.fits === false) {
    note.textContent = ' · does not fit: lower max_tokens or start a new conversation';
    note.className = 'ctx-danger';
  } else {
    // An estimated figure says what about it is estimated; the API's own
    // count and the API's own usage need no such label.
    note.textContent = CONTEXT_LABEL[context.source] || '';
    note.className = '';
  }
  // The full provenance stays one hover away, including whether a count is on
  // its way to replacing a figure that is only partly the API's.
  byId('contextLabel').title = 'Context: ' + tokens + ' / ' + fmt(context.limit)
    + ' tokens' + (pctText !== null ? ' · ' + pctText + '%' : '')
    + ' — source: ' + context.source
    + (context.pending ? " (asking the API's own counter…)" : '')
    + ' · limit source: ' + context.limit_source
    + ' · ' + fmt(context.reserved) + ' reserved for the reply'
    + (context.fits === false ? ' · does not fit' : '');
  const fill = byId('contextFill');
  const pct = context.percent;
  fill.style.width = (pct === null || pct === undefined) ? '0%' : Math.min(100, pct) + '%';
  fill.className = 'meter-fill' + (level ? ' ' + level : '');
  if (context.fits !== undefined) state.fits = context.fits;
  byId('send').disabled = state.blocked || state.fits === false;
  lookAgainForTheCount(context);
}

/* The meter answers "what would the next request carry?" - and the stored
   history is part of that answer even before a word is typed. An empty
   draft therefore still asks the server for the baseline. */
async function refreshContext() {
  if (!state.conversationId) return;
  try {
    const response = await fetch('/api/preview', {
      method: 'POST', headers: {'content-type': 'application/json'},
      body: JSON.stringify(payload(''))
    });
    const data = await response.json();
    // A non-empty draft owns the meter now; this baseline is already stale.
    if (byId('prompt').value.trim()) return;
    if (!data.error) showContext(data.context);
  } catch (ex) {
    // Keep the last honest figure rather than blanking the meter.
  }
}

let previewTimer = null;

/* A pending preview describes a draft; an explicit click on a turn replaces
   the draft, so the queued look must not fire after it. */
function cancelPreview() {
  if (previewTimer) {
    clearTimeout(previewTimer);
    previewTimer = null;
  }
}

function schedulePreview() {
  cancelPreview();
  // A new request earns a fresh set of looks for its count, and a look
  // already queued describes a draft this change has just replaced.
  if (contextLookTimer !== null) {
    clearTimeout(contextLookTimer);
    contextLookTimer = null;
  }
  contextLooksLeft = CONTEXT_LOOKS;
  previewTimer = setTimeout(updatePreview, 200);
}

// The right pane is built before anything is sent: same prepare() path,
// same renderer, but the Response is never executed.
async function updatePreview() {
  const text = byId('prompt').value.trim();
  if (!text) {
    // An empty prompt while a stored turn is selected means "show that
    // turn's request", not "show nothing" - opening a conversation races
    // this timer, and the turn would otherwise blank out again.
    if (state.selected !== null && state.turns[state.selected]) {
      showTurn(state.selected);
    } else {
      state.request = null;
      renderPane();
    }
    // The meter still tells the truth about the whole conversation: an
    // empty draft does not make the stored history weigh nothing.
    refreshContext();
    return;
  }
  byId('codeStatus').textContent = 'building…';
  let data;
  try {
    const response = await fetch('/api/preview', {
      method: 'POST', headers: {'content-type': 'application/json'},
      body: JSON.stringify(payload(text))
    });
    data = await response.json();
  } catch (ex) {
    byId('codeStatus').textContent = 'preview unavailable';
    return;
  }
  if (data.error) {
    state.request = { error: data.error };
    renderPane();
    return;
  }
  state.request = {
    code: data.code,
    status: 'preview · built by model.build_kwargs() · not sent yet',
    dynamic_filtering: data.dynamic_filtering,
    allowed_callers: data.allowed_callers
  };
  showContext(data.context);
  renderPane();
}

function setTab(tab) {
  state.tab = tab;
  byId('tabRequest').classList.toggle('current', tab === 'request');
  byId('tabResponse').classList.toggle('current', tab === 'response');
  renderPane();
}

// One pane, two views of the same turn. The Response is never assembled
// from the chat text: it is the JSON the SDK Message carried.
function renderPane() {
  if (state.tab === 'response') {
    const record = state.turns[state.selected];
    if (!record) {
      setInspectorEmpty('Press Send ⌘⏎', 'The response will appear here.');
      byId('codeStatus').textContent = 'response · not available';
      byId('responseSummary').textContent = '';
      return;
    }
    clearInspectorEmpty();
    // The official Raw shape: the Message as the API returned it, with usage
    // back on it. Our own reading of the record is the summary line below.
    setCode(JSON.stringify(record.response.raw || record.response, null, 2), 'json');
    byId('codeStatus').textContent =
      'response · raw · ' + (record.response.message_id || 'unknown')
      + ' · usage re-attached from the SDK message';
    const summaryParts = [
      'model ' + (record.response.model || 'unreported'),
      record.response.summary,
      'stop reason ' + (record.response.stop_reason || 'unreported')
    ];
    const latency = latencyLabel(record.response);
    if (latency) summaryParts.push(latency);
    byId('responseSummary').textContent = summaryParts.join(' · ');
    return;
  }
  if (!state.request) {
    setInspectorEmpty('Type a prompt',
      'The exact request Send will build appears here.');
    byId('codeStatus').textContent = 'not built yet';
    byId('responseSummary').textContent = '';
    return;
  }
  clearInspectorEmpty();
  if (state.request.error) {
    byId('codeStatus').textContent = 'cannot be sent';
    setCode('This combination cannot be sent:\n\n' + state.request.error, 'none');
    byId('responseSummary').textContent = '';
    return;
  }
  setCode(state.request.code, 'python');
  byId('codeStatus').textContent = state.request.status;
  byId('responseSummary').textContent = '';
}

function refresh() {
  const caps = state.caps;
  const on = byId('webSearch').value === 'true';
  ['webSearchType', 'allowedCallers', 'responseInclusion', 'maxUses'].forEach(
    (id) => { byId(id).disabled = !on; }
  );
  byId('responseInclusion').disabled = !on
    || state.data.controls.response_inclusion.status !== 'Editable';

  const callerControl = state.data.controls.allowed_callers;
  const chosen = byId('allowedCallers').value;
  // Only a choice the user could actually make can be blocked. When the
  // control is runtime-fixed the value is a fact about the request - for
  // web_search_20250305 the effective caller really is "direct" - and the
  // request stays legal, so Send must stay available.
  const off = !!callerControl.editable && chosen === 'direct';
  // The blocked message is kept as a string: the note is now a dot, and the
  // composer's warning cannot be read back out of it.
  const blockedMessage = off
    ? 'sending is blocked: the request would still use the API default' : '';
  byId('allowedCallersNote').innerHTML = whyMark(off
    ? 'API supported · not exposed by llm-anthropic' : callerControl.status,
    off ? blockedMessage : callerControl.note);
  state.blocked = off;
  setBlocked(blockedMessage);
  byId('send').disabled = state.blocked || state.fits === false;
  syncSwitches();
  renderPills();
  updateCacheWarn();
  renderCacheStatus();
}

function apply(data) {
  state.data = data;
  const caps = data.capabilities;
  state.caps = caps;
  state.fits = data.context ? data.context.fits : null;

  byId('contextWindow').textContent = 'context window '
    + fmt(caps.context_window) + ' tokens · ' + caps.context_window_source;
  byId('maxTokens').max = caps.max_output_tokens || '';
  byId('maxTokens').value = data.defaults.max_tokens;

  setNote('ttlNote', data.controls.cache_control.ttl.status,
    data.controls.cache_control.ttl.value + ' · ' + data.controls.cache_control.ttl.note);

  // Streaming is not a choice here: llm-anthropic always opens a stream.
  const streamControl = data.controls.stream;
  setOptions(byId('stream'), streamControl.options, {
    ON: 'ON', OFF: 'OFF (API supported · not used by llm-anthropic)'
  });
  byId('stream').value = 'ON';
  byId('stream').disabled = true;
  setNote('streamNote', streamControl.status, streamControl.note);

  // Thinking: its own control, with the official mode and default state.
  const thinkingControl = data.controls.thinking;
  setOptions(byId('thinking'), thinkingControl.options, {
    on: 'ON (' + caps.thinking_mode + ')', off: 'OFF'
  });
  byId('thinking').value = thinkingControl.value;
  byId('thinking').disabled = thinkingControl.status !== 'Editable';
  setNote('thinkingNote', thinkingControl.status, thinkingControl.note);

  // A model change starts at the provider's own level: the effort another
  // model was given is not carried onto one that tunes differently. A
  // conversation reopening is the path that restores its own.
  applyEffortState('default');
  applyThinkingState();

  const typeControl = data.controls.web_search_type;
  setOptions(byId('webSearchType'), typeControl.options, {});
  byId('webSearchType').value = typeControl.value || '';
  byId('webSearchType').disabled = true;
  setNote('webSearchTypeNote', typeControl.status, typeControl.note);

  const callerControl = data.controls.allowed_callers;
  const callerLabels = {};
  callerControl.options.forEach((option) => {
    callerLabels[option.value] = option.value + ' (' + option.note + ')';
  });
  setOptions(byId('allowedCallers'), callerControl.options, callerLabels);
  byId('allowedCallers').value = callerControl.value || '';
  // A control the runtime cannot honour is shown, not offered as a choice.
  byId('allowedCallers').disabled = !callerControl.editable;

  const inclusion = data.controls.response_inclusion;
  setNote('responseInclusionNote', inclusion.status, inclusion.note);

  groupBadge('gbadgeModel', [data.controls.max_tokens.status]);
  groupBadge('gbadgeThinking', [thinkingControl.status,
    data.controls.budget_tokens.status, data.controls.effort.status]);
  groupBadge('gbadgeSearch', [typeControl.status, callerControl.status, inclusion.status]);
  groupBadge('gbadgeCache', [data.controls.cache_control.ttl.status]);
  groupBadge('gbadgeTransport', [streamControl.status]);

  showContext(data.context);
  // Kept so a brand-new conversation can put the meter back to "nothing
  // carries anything yet" instead of leaving the old conversation's total.
  state.formContext = data.context;
  refresh();
  schedulePreview();
}

/* --- panels, tabs, copy, keyboard ------------------------------------------ */

function toggleSidebar() {
  const collapsed = document.body.classList.toggle('sidebar-collapsed');
  localStorage.setItem('sdkview.sidebar', collapsed ? '0' : '1');
}

byId('sidebarToggle').addEventListener('click', toggleSidebar);

if (localStorage.getItem('sdkview.sidebar') === '0') {
  document.body.classList.add('sidebar-collapsed');
}

// The floating panels close on outside click and on Escape.
document.addEventListener('click', (event) => {
  const pop = byId('costPop');
  if (!pop.hidden
      && !pop.contains(event.target)
      && !byId('costToggle').contains(event.target)) {
    toggleCost();
  }
});

document.addEventListener('keydown', (event) => {
  if (event.key !== 'Escape') return;
  if (!byId('costPop').hidden) toggleCost();
});

byId('copyCode').addEventListener('click', async () => {
  const text = byId('codeContent').textContent;
  try {
    await navigator.clipboard.writeText(text);
  } catch (ex) {
    const range = document.createRange();
    range.selectNodeContents(byId('codeContent'));
    const selection = getSelection();
    selection.removeAllRanges();
    selection.addRange(range);
    document.execCommand('copy');
    selection.removeAllRanges();
  }
  const button = byId('copyCode');
  button.textContent = '✓ Copied';
  button.classList.add('done');
  setTimeout(() => {
    button.textContent = '⧉ Copy';
    button.classList.remove('done');
  }, 1200);
});

document.addEventListener('keydown', (event) => {
  if (!(event.metaKey || event.ctrlKey)) return;
  if (event.key === 'Enter') {
    event.preventDefault();
    byId('send').click();
  } else if (event.key === 'b') {
    event.preventDefault();
    toggleSidebar();
  } else if (event.key === '1') {
    setTab('request');
  } else if (event.key === '2') {
    setTab('response');
  }
});

byId('tabRequest').addEventListener('click', () => setTab('request'));
byId('tabResponse').addEventListener('click', () => setTab('response'));
byId('prompt').addEventListener('input', () => {
  // Typing starts a new request, so nothing in the conversation is selected
  // any more: the Request pane has to stop showing the turn that was. The
  // footer is unmoved by this - it totals the conversation, not the selection.
  state.selected = null;
  markSelected();
  renderPane();
});

/* --- conversations ----------------------------------------------------------- */
// Storage is llm's own SQLite, so anything saved here shows up in
// `llm logs` too, and anything saved there is reachable from this list.

/* The bar above the chat names the conversation the chat belongs to. An
   unsaved conversation has no name yet - it gets one from its first message
   - so the bar says what it is instead of inventing a title. */
function renderChatTitle() {
  const input = byId('chatTitleInput');
  if (!input.hidden) return; // an edit in progress owns the bar
  const name = state.conversationId ? state.conversationName : null;
  const title = byId('chatTitle');
  title.textContent = name || 'New conversation';
  title.classList.toggle('untitled', !name);
  title.title = name || 'saved under the first message you send';
  byId('renameConversation').hidden = !state.conversationId;
}

/* The pencil turns the bar into one field. Enter or leaving the field keeps
   the new name, Escape keeps the old one; an empty name is not a name, so it
   is refused here before it can reach the server. */
function startRename() {
  if (!state.conversationId) return;
  const input = byId('chatTitleInput');
  byId('chatTitle').hidden = true;
  byId('renameConversation').hidden = true;
  input.hidden = false;
  input.value = state.conversationName || '';
  input.focus();
  input.select();
}

function endRename() {
  byId('chatTitleInput').hidden = true;
  byId('chatTitle').hidden = false;
  renderChatTitle();
}

let renaming = false;

async function commitRename() {
  const input = byId('chatTitleInput');
  if (input.hidden || renaming) return;
  const name = input.value.trim();
  if (!name || name === state.conversationName) { endRename(); return; }
  renaming = true;
  try {
    const response = await fetch(
      '/api/conversations/' + state.conversationId + '/name', {
        method: 'POST', headers: { 'content-type': 'application/json' },
        body: JSON.stringify({ name })
      });
    const data = await response.json();
    if (data.error) {
      // The field stays open: the name was refused, not taken.
      setBlocked(data.error, true);
      return;
    }
    state.conversationName = data.name;
    endRename();
    loadConversations();
  } catch (ex) {
    setBlocked('rename failed: ' + ex, true);
  } finally {
    renaming = false;
  }
}

byId('renameConversation').addEventListener('click', startRename);
byId('chatTitleInput').addEventListener('keydown', (event) => {
  event.stopPropagation();
  if (event.key === 'Enter') { event.preventDefault(); commitRename(); }
  else if (event.key === 'Escape') { endRename(); }
});
byId('chatTitleInput').addEventListener('blur', commitRename);

/* --- per-conversation menu: export and delete ------------------------------
   Same rules as the pill menus: mounted in #pillLayer because the sidebar
   is a scroll container that would clip a nested menu, placed from the
   kebab's rect by the one placing function. */
let convMenu = null;

function closeConvMenu() {
  if (!convMenu) return;
  convMenu.el.remove();
  convMenu = null;
}

function openConvMenu(anchor, item) {
  if (convMenu && convMenu.id === item.id) { closeConvMenu(); return; }
  closeConvMenu();
  closePillMenu();
  const menu = document.createElement('div');
  menu.className = 'pill-menu conv-menu';
  menu.setAttribute('role', 'menu');
  menu.addEventListener('click', (event) => event.stopPropagation());
  byId('pillLayer').appendChild(menu);
  convMenu = { el: menu, id: item.id };

  menuItem(menu, {
    label: 'Export Markdown',
    sub: 'the whole conversation as one .md file',
    onPick: () => {
      closeConvMenu();
      // The server sets Content-Disposition, so a plain link is the
      // download: no fetch, no blob, no object URL to forget to revoke.
      const link = document.createElement('a');
      link.href = '/api/conversations/' + encodeURIComponent(item.id) + '/export.md';
      link.download = '';
      document.body.appendChild(link);
      link.click();
      link.remove();
    },
  });

  // Delete is two clicks on purpose: the first turns the item into the
  // confirmation, so a misclick can never cost a conversation.
  let confirming = false;
  const deleteItem = menuItem(menu, {
    label: 'Delete…',
    sub: 'remove this conversation from history',
    onPick: async () => {
      if (!confirming) {
        confirming = true;
        deleteItem.classList.add('danger');
        deleteItem.querySelector('.grow').innerHTML = esc('Really delete?')
          + '<span class="sub">' + esc('cannot be undone — click again') + '</span>';
        return;
      }
      deleteItem.classList.add('disabled');
      try {
        const res = await fetch(
          '/api/conversations/' + encodeURIComponent(item.id),
          { method: 'DELETE' });
        if (!res.ok) {
          const data = await res.json();
          throw new Error(data.error || ('HTTP ' + res.status));
        }
      } catch (ex) {
        closeConvMenu();
        byId('historyNote').textContent = 'delete failed: ' + ex.message;
        return;
      }
      closeConvMenu();
      // Deleting the conversation on screen is also leaving it.
      if (state.conversationId === item.id) startNewConversation();
      await loadConversations();
    },
  });

  anchorOverlay(menu, anchor, { align: 'right' });
}

async function loadConversations() {
  let data;
  try {
    data = await (await fetch('/api/conversations')).json();
  } catch (ex) {
    byId('historyNote').textContent = 'cannot read history: ' + ex;
    return;
  }
  if (data.error) {
    byId('historyNote').textContent = data.error;
    return;
  }
  const list = byId('conversationList');
  list.innerHTML = '';
  if (!data.items.length) {
    const empty = document.createElement('p');
    empty.className = 'notice';
    empty.textContent = 'No conversations yet.';
    list.appendChild(empty);
    return;
  }
  data.items.forEach((item) => {
    const row = document.createElement('div');
    row.className = 'conv-row';
    const button = document.createElement('button');
    button.className = 'conversation'
      + (item.id === state.conversationId ? ' current' : '');
    const label = document.createTextNode(item.name);
    const meta = document.createElement('small');
    // The stored stamp is UTC; the list shows it the way the bubbles do.
    const when = formatMoment(momentOf(item.last_turn));
    meta.textContent = item.turns + ' turn' + (item.turns === 1 ? '' : 's')
      + (when ? ' · ' + when : '');
    button.appendChild(label);
    button.appendChild(meta);
    button.addEventListener('click', () => openConversation(item.id));
    // The kebab is a sibling, never a child: a button inside a button is
    // not a thing a browser will click reliably.
    const kebab = document.createElement('button');
    kebab.className = 'conv-kebab';
    kebab.textContent = '⋯';
    kebab.title = 'Export or delete this conversation';
    kebab.setAttribute('aria-label', 'More actions for ' + item.name);
    kebab.setAttribute('aria-haspopup', 'menu');
    kebab.addEventListener('click', (event) => {
      event.stopPropagation();
      openConvMenu(kebab, item);
    });
    row.appendChild(button);
    row.appendChild(kebab);
    list.appendChild(row);
  });
  // The list is where a fresh conversation's name first exists (its first
  // turn was just saved), so the title bar reads it from here too.
  const current = data.items.find((item) => item.id === state.conversationId);
  if (current && current.name !== state.conversationName) {
    state.conversationName = current.name;
  }
  renderChatTitle();
}

/* --- what belongs to a conversation: its draft, and its settings ------------- */

/* Drafts are keyed by the conversation they were typed into. An unsaved
   conversation is not keyed `null`: that is also "no conversation at all",
   and the two must not share what the user wrote. */
const NEW_DRAFT_KEY = '__new__';

function draftKey() {
  return state.conversationId || NEW_DRAFT_KEY;
}

function saveDraft() {
  const text = byId('prompt').value;
  if (text) state.drafts[draftKey()] = text;
  else delete state.drafts[draftKey()];
}

function restoreDraft() {
  byId('prompt').value = state.drafts[draftKey()] || '';
}

function clearDraft() {
  delete state.drafts[draftKey()];
}

/* The settings a stored conversation ended with, written onto a form that
   apply() has already filled with the model's own defaults and
   capabilities: a stored value is only restorable while the model still
   offers it, and a control the runtime cannot honour keeps its status
   rather than being silently switched. */
function applyStoredOptions(options) {
  if (!options || !Object.keys(options).length) return;

  const restore = (id, value) => {
    if (value === null || value === undefined || value === '') return;
    const select = byId(id);
    // A value this model no longer offers stays at the model's default.
    // Writing it anyway would show a request the control cannot make.
    const known = Array.from(select.options).some((o) => o.value === String(value));
    if (known) select.value = String(value);
  };
  restore('thinking', options.thinking);
  restore('webSearchType', options.web_search_type);
  restore('allowedCallers', options.allowed_callers);
  restore('responseInclusion', options.response_inclusion);

  if (typeof options.system === 'string') byId('system').value = options.system;
  if (typeof options.max_tokens === 'number') {
    const ceiling = Number(byId('maxTokens').max);
    const value = ceiling ? Math.min(options.max_tokens, ceiling) : options.max_tokens;
    byId('maxTokens').value = value;
  }
  if (typeof options.max_uses === 'number') byId('maxUses').value = options.max_uses;
  if (typeof options.web_search === 'boolean') {
    byId('webSearch').value = options.web_search ? 'true' : 'false';
  }
  if (typeof options.cache_control === 'boolean') {
    byId('cacheControl').value = options.cache_control ? 'true' : 'false';
  }

  applyThinkingState();
  // Effort is not written above like the others: its option list is rebuilt
  // from the restored thinking, and a rebuild that ran after a plain write
  // would wipe it. That wipe left a reopened conversation on "default" and
  // then had the composer report the stored level as a change the user had
  // made. The stored level goes to the rebuild instead, which keeps it when
  // this model still offers it and falls back to the provider's own level
  // when it does not.
  applyEffortState(options.effort);
  refresh();
}

/* A stored conversation can point at a model the list no longer offers,
   because its own series moved on. It still has to be selectable: that turn
   was sent with it, and applying another model's capabilities to it would be
   a lie about what happens next. It is added marked, never silently
   equated with the model that superseded it. */
function ensureModelOption(id) {
  const select = byId('model');
  if (!id) return false;
  if (Array.from(select.options).some((option) => option.value === id)) return false;
  const option = document.createElement('option');
  option.value = id;
  option.textContent = shortModelName(id);
  option.dataset.superseded = 'true';
  option.title = 'used by this conversation, no longer the newest member of '
    + 'its series';
  select.appendChild(option);
  return true;
}

async function openConversation(id) {
  saveDraft();
  const data = await (await fetch('/api/conversations/' + id)).json();
  if (data.error) {
    setBlocked(data.error, true);
    return;
  }
  messages.innerHTML = '';
  state.conversationId = id;
  state.conversationName = data.name;
  renderChatTitle();
  state.turns = data.turns;
  // The cache anchor is the most recent stored turn that touched the cache.
  state.cache.anchor = null;
  state.cache.frozen = null;
  for (let i = data.turns.length - 1; i >= 0; i--) {
    const anchor = cacheAnchorOf(data.turns[i]);
    if (anchor) { state.cache.anchor = anchor; break; }
  }
  const lastTurn = data.turns[data.turns.length - 1];
  state.cache.lastInput = lastTurn && lastTurn.response
    && lastTurn.response.usage ? lastTurn.response.usage.input_tokens : null;
  // Settings belong to the conversation, not to the tab: a conversation is
  // reopened with the settings its own last turn used, so the next turn
  // continues it instead of quietly changing the request underneath it.
  // The model goes on first - it decides which defaults and capabilities
  // apply() writes, and the rest is restored over those.
  const lastOptions = (lastTurn && lastTurn.options) || {};
  const wantedModel = lastOptions.model;
  // The select may still be empty on the first conversation opened, so the
  // options have to exist before a stored id can be added to them. Model
  // first anyway: it decides which defaults and capabilities apply() writes.
  await loadForm();
  if (wantedModel && byId('model').value !== wantedModel) {
    ensureModelOption(wantedModel);
    byId('model').value = wantedModel;
    await loadForm();
  }
  applyStoredOptions(lastOptions);
  restoreDraft();
  data.turns.forEach((record, index) => {
    // Both panes of a stored turn come from what was recorded, never from
    // a second reconstruction of the provider response.
    if (index > 0) {
      const diff = diffBetweenTurns(data.turns[index - 1], record);
      if (diff.length) messages.appendChild(dividerFor(diff));
    }
    const user = addMessage('user', record.user_input, index + 1, record.timestamp);
    const assistant = addAssistantMessage(index + 1, record.timestamp);
    renderAssistantRecord(assistant, record.response);
    attachMeta(assistant, record.response);
    wireBubble(user.parentElement, index, 'request',
      'Show the request this turn sent');
    wireBubble(assistant.parentElement, index, 'response',
      'Show the response this turn came back with');
    wireSettingsCard(user.parentElement, index);
    wireTurnCostCard(assistant.parentElement, index);
  });
  // Opening a conversation lands at its end, where the next turn goes.
  pinnedToBottom = true;
  scrollChatToBottom(false);
  state.selected = data.turns.length - 1;
  showTurn(state.selected);
  refreshCost();
  updateCacheWarn();
  loadConversations();
  // The context meter counts the whole conversation, not the empty draft:
  // this fires the baseline preview that fills it in.
  schedulePreview();
}

function showTurn(index, tab) {
  const record = state.turns[index];
  if (!record) return;
  state.selected = index;
  state.request = {
    code: record.rendered_code,
    status: 'request · turn ' + (index + 1) + ' of this conversation, as it was sent'
  };
  // Which pane a click lands on is the click's to decide: the user's bubble
  // is the request that went out, the assistant's is the answer that came
  // back. Without a tab the pane keeps whatever it was showing.
  setTab(tab || state.tab);
  markSelected();
}

async function startNewConversation() {
  saveDraft();
  messages.innerHTML = '';
  state.conversationId = null;
  state.conversationName = null;
  renderChatTitle();
  state.turns = [];
  state.selected = null;
  state.request = null;
  state.cache = { anchor: null, lastInput: null, frozen: null };
  // A new conversation is the one case that starts from the model's
  // defaults: there is no last turn of its own to continue.
  await loadForm();
  restoreDraft();
  renderPane();
  refreshCost();
  renderCacheStatus();
  updateCacheWarn();
  loadConversations();
  // Back to the empty-draft baseline; the old conversation's total must
  // not keep hanging over a conversation it no longer describes.
  if (state.formContext) showContext(state.formContext);
  // Told explicitly on the next send, so the server cannot mistake an
  // empty id for the conversation this tab was using a moment ago.
  state.fresh = true;
}

byId('newConversation').addEventListener('click', startNewConversation);

async function loadForm() {
  const model = byId('model').value;
  const query = model ? '?model=' + encodeURIComponent(model) : '';
  const response = await fetch('/api/form' + query);
  const data = await response.json();
  if (data.error) {
    setBlocked(data.error, true);
    byId('send').disabled = true;
    return;
  }
  if (!byId('model').options.length) {
    setOptions(byId('model'), data.models.map((value) => ({value})), {});
    byId('model').value = data.model.id;
  }
  apply(data);
}

['webSearch', 'webSearchType', 'allowedCallers'].forEach(
  (id) => byId(id).addEventListener('change', refresh)
);
byId('thinking').addEventListener('change', () => {
  applyEffortState();
  applyThinkingState();
  refresh();
});
byId('model').addEventListener('change', loadForm);

// Any change to the form rebuilds the preview, so the right pane always
// answers "what would Send put on the wire" before it costs anything.
['model', 'thinking', 'effort', 'webSearch', 'webSearchType', 'allowedCallers',
 'responseInclusion', 'maxUses', 'cacheControl', 'system', 'maxTokens', 'prompt'
].forEach((id) => {
  byId(id).addEventListener('change', schedulePreview);
  byId(id).addEventListener('input', schedulePreview);
});
// The pills mirror the same controls, so they re-render on the same changes
// (apply()/refresh() cover the rest).
['effort', 'maxTokens', 'system', 'cacheControl', 'maxUses', 'responseInclusion'
].forEach((id) => {
  byId(id).addEventListener('change', renderPills);
  byId(id).addEventListener('change', updateCacheWarn);
});

wireSwitch('webSearch', 'webSearchSwitch');
wireSwitch('cacheControl', 'cacheControlSwitch');

/* --- send -------------------------------------------------------------------- */

byId('send').addEventListener('click', async () => {
  if (streaming || byId('send').disabled) return;
  const text = byId('prompt').value.trim();
  if (!text) return;
  streaming = true;
  byId('send').disabled = true;
  byId('send').textContent = 'Sending…';
  // Pressing send is one of only two things that stop the countdown (the
  // other is reaching zero): the reading freezes where it was, so the
  // margin this send had stays visible while the answer streams. The
  // record event starts a fresh countdown from the answer's end; an error
  // thaws this one, because an untouched cache keeps its old clock.
  if (state.cache.anchor && byId('cacheControl').value === 'true') {
    const left = state.cache.anchor.at + CACHE_TTL_MS - Date.now();
    if (left > 0) state.cache.frozen = { remaining: left };
  }
  renderCacheStatus();
  const turnNo = state.turns.length + 1;
  // Until the turn is recorded, the only clock it has is this page's own:
  // the moment the message was sent. The record's stamp replaces it below.
  const sentAt = new Date();
  const userBody = addMessage('user', text, turnNo, sentAt);
  byId('prompt').value = '';
  // Sent, so it is no longer a draft waiting in this conversation.
  clearDraft();
  const body = addAssistantMessage(turnNo, sentAt);
  // Named `bubble`, never `parts`: the SSE parser below splits its buffer
  // into a local `parts`, and a same-named binding here would be shadowed
  // inside the read loop - the first reasoning chunk would land on an
  // array and kill the stream.
  const bubble = assistantParts(body);
  // Sending is a decision to read what comes back: follow this stream.
  pinnedToBottom = true;
  let streamedText = '';
  let answerRenderQueued = false;
  // The thinking clock starts at the first reasoning chunk and stops when
  // the turn is done; its reading is what the finished strip reports.
  let thinkStarted = null;
  let thinkTimer = null;
  const sent = payload(text);
  state.fresh = false;

  /* Markdown re-parsed at most every 120ms while the answer streams, so a
     half-typed fence or table row never costs more than a frame of lag. */
  const queueAnswerRender = () => {
    if (answerRenderQueued) return;
    answerRenderQueued = true;
    setTimeout(() => {
      answerRenderQueued = false;
      renderMarkdown(bubble.answer, streamedText);
      followStream();
    }, 120);
  };

  const thinkTick = () => {
    bubble.thinkLabel.textContent = 'Thinking… ' + clock(performance.now() - thinkStarted);
  };

  const startThinking = () => {
    thinkStarted = performance.now();
    bubble.thinkStrip.hidden = false;
    bubble.thinkStrip.classList.add('open');
    bubble.thinkStrip.querySelector('.chev').textContent = '▾';
    bubble.thinkBody.hidden = false;
    bubble.thinkBody.classList.add('live');
    thinkTick();
    thinkTimer = setInterval(thinkTick, 250);
  };

  const stopThinkClock = () => {
    if (thinkTimer) {
      clearInterval(thinkTimer);
      thinkTimer = null;
    }
    bubble.thinkBody.classList.remove('live');
  };

  const finish = () => {
    streaming = false;
    stopThinkClock();
    bubble.answer.classList.remove('streaming');
    byId('send').innerHTML = 'Send <span class="key-hint">⌘⏎</span>';
    byId('send').disabled = state.blocked || state.fits === false;
  };

  let response;
  try {
    response = await fetch('/api/chat/stream', {
      method: 'POST', headers: {'content-type': 'application/json'}, body: JSON.stringify(sent)
    });
  } catch (ex) {
    body.classList.add('error');
    body.textContent = 'the request failed to reach the server: ' + ex;
    // Never reached the provider, so the cache was never touched: thaw.
    state.cache.frozen = null;
    renderCacheStatus();
    finish();
    return;
  }
  const reader = response.body.getReader();
  const decoder = new TextDecoder();
  let buffer = '';
  while (true) {
    const { done, value } = await reader.read();
    if (done) break;
    buffer += decoder.decode(value, { stream: true });
    const parts = buffer.split('\n\n');
    buffer = parts.pop();
    for (const part of parts) {
      if (!part.startsWith('data: ')) continue;
      const event = JSON.parse(part.slice(6));
      if (event.type === 'prepared') {
        state.request = { code: event.code, status: 'sent · the request that was executed' };
        if (state.tab === 'request') renderPane();
        showContext(event.context);
      } else if (event.type === 'text') {
        // The answer arriving collapses the thinking it came from.
        if (thinkStarted !== null && bubble.thinkStrip.classList.contains('open')) {
          bubble.thinkStrip.classList.remove('open');
          bubble.thinkStrip.querySelector('.chev').textContent = '▸';
          bubble.thinkBody.hidden = true;
        }
        bubble.answer.classList.add('streaming');
        streamedText += event.text;
        queueAnswerRender();
      } else if (event.type === 'reasoning') {
        if (thinkStarted === null) startThinking();
        bubble.thinkBody.textContent += event.text;
        bubble.thinkBody.scrollTop = bubble.thinkBody.scrollHeight;
        followStream();
      } else if (event.type === 'done') {
        // The finished Message outranks the stream's fragments: text and
        // thinking are both re-rendered from it, so display and storage
        // can never disagree about what the model said.
        stopThinkClock();
        streamedText = event.text;
        renderMarkdown(bubble.answer, streamedText);
        bubble.answer.classList.remove('streaming');
        const thinking = event.thinking
          || (thinkStarted !== null ? bubble.thinkBody.textContent : null);
        if (thinking) {
          showFinishedThinking(bubble, thinking, thinkStarted !== null
            ? (performance.now() - thinkStarted) / 1000 : null);
        }
        state.fits = event.context ? event.context.fits : null;
        showContext(event.context);
        followStream();
      } else if (event.type === 'record') {
        // One record drives every view of the turn: the bubble above was
        // already written from the same Message this came from.
        const previous = state.turns[state.turns.length - 1];
        state.turns.push(event.record);
        state.selected = state.turns.length - 1;
        // The stored stamp is the clock the turn is filed under, so it
        // takes over from the page's own send time the moment it exists.
        setMessageTime(userBody.parentElement, event.record.timestamp);
        setMessageTime(body.parentElement, event.record.timestamp);
        const diff = diffBetweenTurns(previous, event.record);
        if (diff.length) {
          messages.insertBefore(dividerFor(diff), userBody.parentElement);
        }
        wireBubble(userBody.parentElement, state.selected, 'request',
          'Show the request this turn sent');
        wireBubble(body.parentElement, state.selected, 'response',
          'Show the response this turn came back with');
        wireSettingsCard(userBody.parentElement, state.selected);
        wireTurnCostCard(body.parentElement, state.selected);
        showToolUse(bubble, event.record.response.server_tool_blocks);
        showSources(bubble, event.record.response.citations);
        attachMeta(body, event.record.response);
        // The answer has landed: the freeze lifts and a fresh window starts
        // from this moment (or keeps the old anchor if this turn never
        // touched the cache).
        state.cache.frozen = null;
        const anchor = cacheAnchorOf(event.record);
        if (anchor) state.cache.anchor = anchor;
        const usage = event.record.response.usage || {};
        if (typeof usage.input_tokens === 'number') state.cache.lastInput = usage.input_tokens;
        renderCacheStatus();
        refreshCost();
        updateCacheWarn();
        if (event.saved) {
          state.conversationId = event.record.conversation_id;
          setBlocked('');
        } else {
          // A completed, billed turn that is not in history is a loss the
          // page has to say out loud.
          setBlocked('turn sent but NOT saved to history: ' + (event.error || 'unknown error'), true);
        }
        // The new information is the response: show it.
        setTab('response');
        markSelected();
        loadConversations();
      } else if (event.type === 'error') {
        body.innerHTML = '';
        body.className = 'body error';
        body.textContent = event.error === 'missing_api_key'
          ? 'No Anthropic key found. Nothing was sent. Set one with: llm keys set anthropic'
          : event.error;
        // The turn never happened, so the cache was never touched: thaw.
        state.cache.frozen = null;
        renderCacheStatus();
      }
    }
  }
  finish();
});

loadForm();
loadConversations();
loadRates();
setTab('request');
renderCostBar();

/* --- which build this tab is, and whether the server has moved on -------------
   A tab keeps its js until it is reloaded, and this tool is rebuilt several
   times a day, so a tab can run a build the server has already replaced -
   and report bugs that no longer exist. The build the tab loaded is read off
   its own script URL, stamped in the sidebar, and whenever the tab regains
   focus the page asks the server which build it would serve now. A mismatch
   is announced on the page itself, never mistaken for a working app. */

const BUILD = (() => {
  const script = document.querySelector('script[src*="/static/app.js"]');
  if (!script) return null;
  return new URL(script.src, location.href).searchParams.get('v');
})();

byId('buildStamp').textContent = BUILD ? 'build ' + BUILD : '';

let staleCheckedAt = 0;

async function checkStaleBuild() {
  if (!BUILD) return;
  const now = Date.now();
  if (now - staleCheckedAt < 15000) return;
  staleCheckedAt = now;
  try {
    const response = await fetch('/api/version', { cache: 'no-store' });
    const data = await response.json();
    const stale = data.version && data.version !== BUILD;
    const banner = byId('staleBanner');
    banner.hidden = !stale;
    if (stale) {
      banner.textContent = 'This tab runs build ' + BUILD + ' but the server is on '
        + data.version + ' — click to reload';
    }
  } catch {
    // The server being unreachable is told by every other fetch already.
  }
}

byId('staleBanner').addEventListener('click', () => location.reload());
document.addEventListener('visibilitychange', () => {
  if (!document.hidden) checkStaleBuild();
});
window.addEventListener('focus', checkStaleBuild);
checkStaleBuild();
