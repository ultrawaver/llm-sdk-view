"""The reading experience: thinking, markdown, sources, and the scroll.

These assertions name stable element classes and function names rather than
visual details, so the next visual iteration does not touch them. What they
protect is behaviour: thinking must stream and stay reviewable, the answer
must render as markup without ever becoming HTML, and the stream must not
drag the reader's place away.
"""


def test_the_answer_renders_as_built_markup_never_as_html(static_page):
    """Model output is text; the page builds every element itself.

    A single ``innerHTML =`` of reply text would turn the chat into an
    injection surface, so the renderer is DOM construction end to end.
    """
    flat = " ".join(static_page.split())

    assert "function safeMarkdown(text)" in flat
    assert "container.replaceChildren(safeMarkdown(text))" in flat
    assert "document.createTextNode(text.slice(last, start))" in flat
    # Links are built only for http(s); anything else stays literal text.
    assert 'link.rel = \'noopener noreferrer\'' in flat
    assert "(https?:\\/\\/" in flat


def test_markdown_covers_the_blocks_a_reply_actually_uses(static_page):
    flat = " ".join(static_page.split())

    # Fenced code with the vendored highlighter and a per-block copy button.
    assert "function codeBlockElement(lang, codeText)" in flat
    assert "Prism.highlightElement(code)" in flat
    assert 'className = \'copycode\'' in flat
    # Tables are real tables, behind a scroller so they cannot deform the
    # bubble.
    assert "function tableElement(headerLine, rowLines)" in flat
    assert "function isTableDivider(line)" in flat
    assert 'className = \'tablewrap\'' in flat
    # Headings, lists, quotes, inline emphasis.
    assert "MD_HEADING" in flat
    assert "MD_UL" in flat
    assert "MD_OL" in flat
    assert "MD_QUOTE" in flat
    assert "document.createElement('strong')" in flat
    assert "document.createElement('em')" in flat


def test_thinking_streams_then_stays_reviewable(static_page):
    """The strip opens live with a clock, collapses when the answer starts,
    and finishes with what the record says - duration and size included."""
    flat = " ".join(static_page.split())

    assert 'class="think-strip"' in flat
    assert 'class="think-body"' in flat
    assert "const startThinking = () => {" in flat
    assert "'Thinking… ' + clock(performance.now() - thinkStarted)" in flat
    # The first answer chunk collapses the strip; the done event rewrites
    # both text and thinking from the finished Message.
    assert "event.type === 'reasoning'" in flat
    assert "function showFinishedThinking(parts, thinking, seconds)" in flat
    assert "'Thought for '" in flat


def test_tool_use_and_sources_are_strips_of_their_own(static_page):
    flat = " ".join(static_page.split())

    assert "function showToolUse(parts, serverToolBlocks)" in flat
    assert "'Searched the web: \"'" in flat
    assert "function showSources(parts, citations)" in flat
    assert "'Sources ('" in flat
    # A citation with no linkable address is not listed as a source.
    assert "!/^https?:\\/\\//.test(url)" in flat


def test_the_meta_row_carries_the_turns_own_figures(static_page):
    flat = " ".join(static_page.split())

    assert "function attachMeta(bodyEl, response)" in flat
    assert "'⧉ copy all'" in flat
    assert "parts.push(response.stop_reason)" in flat


def test_the_stream_never_steals_the_reading_place(static_page):
    """Pinned at the bottom the chat follows; scrolled up it offers the way
    back instead of dragging the reader along."""
    flat = " ".join(static_page.split())

    assert 'id="jumpPill"' in flat
    assert "↓ New messages" in flat
    assert "function followStream()" in flat
    # The pill lives outside the scroller: a scroller clips its descendants.
    chat_body = flat.split('id="chatBody"', 1)[1].split('id="paramsPanel"', 1)[0]
    assert 'id="jumpPill"' in chat_body
    assert '#jumpPill[hidden] { display: none; }' in flat


def test_an_empty_region_takes_no_room(static_page):
    """A bubble without thinking, tool use or citations must not carry blank
    lines in their place. `hidden` alone lost here: the strips are
    `display: flex`, an author `display` beats the browser's rule for
    `hidden`, and an empty strip still laid out - two blank rows above the
    answer, one below. Every region that can be empty needs its own
    `[hidden]` display override."""
    flat = " ".join(static_page.split())

    for region in ("think-strip", "tool-strip", "sources-head",
                   "think-body", "tool-list", "sources-list"):
        assert f".{region}[hidden]" in flat
    assert ".think-strip[hidden], .tool-strip[hidden], .sources-head[hidden]" in flat


def test_only_a_hand_moving_up_unpins(static_page):
    """A programmatic follow-scroll's event fires after the stream has grown
    past the position it scrolled to; reading that as "the user left the
    bottom" dropped the pin mid-stream, and the pill's own smooth ride
    unpinned itself on its first frame."""
    flat = " ".join(static_page.split())

    assert "log.scrollTop < lastScrollTop" in flat
    assert "ridingToBottom" in flat
    assert "if (atBottom) pinnedToBottom = true;" in flat


def test_the_bubble_regions_are_not_named_parts(static_page):
    """The SSE parser splits its buffer into a local `parts`; a same-named
    binding for the bubble's regions was shadowed inside the read loop and
    the first reasoning chunk landed on an array, killing the stream."""
    flat = " ".join(static_page.split())

    assert "const bubble = assistantParts(body);" in flat
    assert "bubble.thinkBody.textContent += event.text;" in flat


def test_reduced_motion_turns_off_what_moves(static_page):
    flat = " ".join(static_page.split())

    assert "prefers-reduced-motion: reduce" in flat
    assert "window.matchMedia('(prefers-reduced-motion: reduce)')" in flat
    # The ride back down is instant, not animated.
    assert "behavior: smooth && !reducedMotion ? 'smooth' : 'auto'" in flat
