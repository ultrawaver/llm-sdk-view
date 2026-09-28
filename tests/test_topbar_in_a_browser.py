"""The topbar, measured in a browser instead of read.

Every other test in this suite reads the page as text. That is enough for
what the page *says* and useless for what it *does*: "clicking a pill does
nothing" survived three repairs because each of them searched the
JavaScript, and the JavaScript was right. The handler ran, the menu was
built and inserted, and it reported ``display: block``, ``visibility:
visible``, ``z-index: 70``. It was inside ``#settingsPills``, which is a
scroller - and ``overflow-x: auto`` computes ``overflow-y`` to ``auto`` as
well, so the 28px-tall row clipped a menu that begins below it. The menu was
in the DOM and painted nowhere.

So the questions here are the ones only a browser can answer: is the element
the thing painted at its own rectangle, does any ancestor clip it, and does
the click the user aimed reach the handler at all.

No key and no network: the page runs against the same isolated application
the rest of the suite builds, and the browser only ever loads localhost.
"""

import os
import socket
import threading
import time

import pytest
import uvicorn

from llm_sdk_view.app import create_app

pytestmark = pytest.mark.browser


def _no_browser(reason: str):
    """Skip, unless the caller said a skip is not an acceptable answer.

    CI sets ``LLM_SDK_VIEW_REQUIRE_BROWSER``: a check that quietly skips is
    a check that is not running, and a suite that stays green while these
    never execute is precisely how the defect they exist for shipped.
    """
    if os.environ.get("LLM_SDK_VIEW_REQUIRE_BROWSER"):
        pytest.fail(f"LLM_SDK_VIEW_REQUIRE_BROWSER is set but {reason}")
    pytest.skip(reason)

# The pills that open a menu, and the heading each menu carries. Streaming
# and (on a model without the parameter) Effort open nothing by design.
MENU_PILLS = {
    "model": "Model",
    "maxTokens": "Max output tokens",
    "system": "System prompt",
    "webSearch": "Web search",
    "cacheControl": "Prompt caching",
}


def _free_port() -> int:
    with socket.socket() as sock:
        sock.bind(("127.0.0.1", 0))
        return sock.getsockname()[1]


def _launch(playwright):
    """Whatever chromium this machine has: the bundled one, or Chrome."""
    problems = []
    for options in ({}, {"channel": "chrome"}):
        try:
            return playwright.chromium.launch(headless=True, **options)
        except Exception as exc:  # noqa: BLE001 - the reason is reported, not handled
            name = options.get("channel", "bundled chromium")
            problems.append(f"{name}: {str(exc).splitlines()[0]}")
    _no_browser(
        "there is no browser to drive (" + "; ".join(problems) + "). Install "
        "one with `playwright install chromium`, or install Google Chrome."
    )


@pytest.fixture(scope="session")
def browser():
    try:
        from playwright.sync_api import sync_playwright
    except ImportError:
        _no_browser(
            "playwright is not installed. The browser checks need the "
            "optional extra: pip install -e '.[test,browser]'"
        )
    with sync_playwright() as playwright:
        launched = _launch(playwright)
        try:
            yield launched
        finally:
            launched.close()


@pytest.fixture
def live_app():
    """The real application on a real port.

    In-process and function-scoped, so the suite's isolation fixtures still
    apply: no API key, a throwaway cache directory, a throwaway database and
    a token counter that refuses. The page falls back to the versioned model
    profile, which is what a machine with no key sees anyway.
    """
    port = _free_port()
    server = uvicorn.Server(
        uvicorn.Config(create_app(), host="127.0.0.1", port=port, log_level="warning")
    )
    thread = threading.Thread(target=server.run, daemon=True)
    thread.start()
    deadline = time.monotonic() + 30
    while not server.started:
        if time.monotonic() > deadline or not thread.is_alive():
            raise RuntimeError("the application did not start")
        time.sleep(0.02)
    try:
        yield f"http://127.0.0.1:{port}"
    finally:
        server.should_exit = True
        thread.join(timeout=15)


@pytest.fixture
def page(browser, live_app):
    context = browser.new_context(viewport={"width": 1500, "height": 820})
    opened = context.new_page()
    opened.goto(live_app)
    # The pills render once the form and the capabilities have arrived.
    opened.wait_for_selector('#settingsPills [data-pill="model"]')
    try:
        yield opened
    finally:
        context.close()


def is_painted_at_its_own_rect(page, selector: str) -> bool:
    """Whether a hit test inside the element's box actually finds it.

    A clipped element keeps its box, its computed style and its place in the
    DOM. What it loses is this.
    """
    return page.evaluate(
        """(selector) => {
            const el = document.querySelector(selector);
            if (!el) return false;
            const rect = el.getBoundingClientRect();
            if (rect.width === 0 || rect.height === 0) return false;
            const hit = document.elementFromPoint(
                rect.x + rect.width / 2, rect.y + Math.min(20, rect.height / 2));
            return !!hit && el.contains(hit);
        }""",
        selector,
    )


def clipping_ancestors(page, selector: str) -> list:
    return page.evaluate(
        """(selector) => {
            const out = [];
            let node = document.querySelector(selector).parentElement;
            while (node && node !== document.documentElement) {
                const style = getComputedStyle(node);
                if (style.overflowX !== 'visible' || style.overflowY !== 'visible') {
                    out.push((node.id || node.tagName) + ' '
                             + style.overflowX + '/' + style.overflowY);
                }
                node = node.parentElement;
            }
            return out;
        }""",
        selector,
    )


# --- the hazard this layout is arranged around -------------------------------


def test_every_pill_can_be_reached_in_a_narrow_window(page):
    """A pill off the edge is another click that does nothing.

    Plain `justify-content: flex-end` pushes overflow past the start edge,
    and overflow in that direction is not scrollable. At 900px the row
    reported scrollWidth === clientWidth while three pills sat at negative
    x: not clipped, not scrollable to, just gone.
    """
    page.set_viewport_size({"width": 900, "height": 700})
    stranded = page.evaluate(
        """() => {
            const host = document.getElementById('settingsPills');
            host.scrollLeft = 0;
            const reach = host.scrollWidth - host.clientWidth;
            const left = host.getBoundingClientRect().left;
            return Array.from(host.children)
                .filter((pill) => pill.getBoundingClientRect().left < left - reach - 1)
                .map((pill) => pill.dataset.pill);
        }"""
    )
    assert stranded == []
    # And the click lands, rather than merely being geometrically possible.
    page.click('[data-pill="model"]')
    assert is_painted_at_its_own_rect(page, "#pillLayer .pill-menu")


def test_the_pill_row_clips_on_both_axes(page):
    """The fact that surprised three repairs, asserted so it stays known.

    The stylesheet only asks for `overflow-x: auto`. CSS then computes
    `overflow-y` from `visible` to `auto`, and the row clips vertically -
    which is why no overlay may be a descendant of a pill.
    """
    assert page.evaluate(
        """() => { const s = getComputedStyle(document.getElementById('settingsPills'));
                   return [s.overflowX, s.overflowY]; }"""
    ) == ["auto", "auto"]


# --- a menu has to be able to appear ------------------------------------------


def test_clicking_a_pill_puts_a_menu_on_the_screen(page):
    page.click('[data-pill="model"]')
    assert page.locator("#pillLayer .pill-menu").count() == 1
    assert is_painted_at_its_own_rect(page, "#pillLayer .pill-menu")


def test_nothing_between_an_open_menu_and_the_page_clips_it(page):
    page.click('[data-pill="model"]')
    assert clipping_ancestors(page, "#pillLayer .pill-menu") == []


@pytest.mark.parametrize(
    "width,height", [(1500, 820), (900, 700)], ids=["wide", "narrow"]
)
def test_every_pill_that_has_a_menu_opens_a_visible_one(page, width, height):
    """The narrow case matters: that is when the row really does scroll."""
    page.set_viewport_size({"width": width, "height": height})
    for pill, heading in MENU_PILLS.items():
        page.click(f'[data-pill="{pill}"]')
        menu = page.locator("#pillLayer .pill-menu")
        assert menu.count() == 1, f"{pill} opened no menu at {width}x{height}"
        assert page.text_content("#pillLayer .pill-menu .mhead") == heading
        assert is_painted_at_its_own_rect(page, "#pillLayer .pill-menu"), (
            f"{pill}'s menu is in the DOM but painted nowhere at {width}x{height}"
        )
        box = menu.bounding_box()
        assert box["x"] >= 0 and box["x"] + box["width"] <= width
        assert box["y"] >= 0
        page.keyboard.press("Escape")


def test_a_menu_is_anchored_under_the_pill_it_belongs_to(page):
    page.click('[data-pill="cacheControl"]')
    pill = page.locator('[data-pill="cacheControl"]').bounding_box()
    menu = page.locator("#pillLayer .pill-menu").bounding_box()
    assert menu["y"] >= pill["y"] + pill["height"]
    assert abs((menu["x"] + menu["width"]) - (pill["x"] + pill["width"])) < 2


# --- a press has to survive the render it sets off ----------------------------


def test_a_menu_field_opens_focused_so_typing_replaces_the_value(page):
    """focus() on a field in a detached node does nothing, and that is how
    typing 4096 into a 16384 ceiling produced 163844096."""
    assert page.input_value("#maxTokens") != "4096"
    page.click('[data-pill="maxTokens"]')
    assert page.evaluate(
        "() => document.activeElement === "
        "document.querySelector('#pillLayer .pill-menu input')"
    )
    page.keyboard.type("4096")
    page.keyboard.press("Enter")
    assert page.input_value("#maxTokens") == "4096"


def test_pressing_the_next_pill_commits_the_open_field_and_opens_that_pill(page):
    """The click the browser used to throw away.

    The press blurs the open field, the field commits, and the commit
    re-renders the row. While the row was rebuilt from scratch, the element
    the press landed on was detached before the button came back up - and a
    mousedown whose target has gone produces no click event at all. The pill
    aimed at did nothing, and the one already open simply closed.
    """
    page.click('[data-pill="maxTokens"]')
    page.keyboard.type("8192")
    page.click('[data-pill="webSearch"]')
    assert page.input_value("#maxTokens") == "8192"
    assert page.text_content("#pillLayer .pill-menu .mhead") == "Web search"


def test_editing_a_field_inside_a_menu_leaves_the_menu_standing(page):
    """Max uses is a control like any other, so writing it re-renders the
    row - and it lives inside the Web Search menu, which must therefore
    survive its own writes, still anchored and not rebuilt."""
    page.click('[data-pill="webSearch"]')
    page.fill("#menuMaxUses", "3")
    page.keyboard.press("Enter")
    assert page.input_value("#maxUses") == "3"
    assert page.locator("#pillLayer .pill-menu").count() == 1
    assert page.text_content("#pillLayer .pill-menu .mhead") == "Web search"
    assert is_painted_at_its_own_rect(page, "#pillLayer .pill-menu")


def test_a_pill_stays_the_same_element_across_a_render(page):
    """What makes the two tests above possible."""
    page.evaluate(
        """() => { window.__watched = document.querySelector('[data-pill="webSearch"]'); }"""
    )
    page.click('[data-pill="maxTokens"]')
    page.keyboard.type("2048")
    page.keyboard.press("Enter")
    assert page.input_value("#maxTokens") == "2048"
    assert page.evaluate(
        """() => window.__watched === document.querySelector('[data-pill="webSearch"]')
                 && window.__watched.isConnected"""
    )


# --- picking, dismissing, and the pills that open nothing ---------------------


def test_picking_writes_the_hidden_control_and_closes_the_menu(page):
    before = page.input_value("#model")
    page.click('[data-pill="model"]')
    page.locator("#pillLayer .mi").filter(has_not=page.locator(".selected")).first.click()
    assert page.input_value("#model") != before
    assert page.locator(".pill-menu").count() == 0


def test_an_outside_click_and_escape_both_dismiss(page):
    page.click('[data-pill="model"]')
    page.mouse.click(700, 500)
    assert page.locator(".pill-menu").count() == 0
    page.click('[data-pill="cacheControl"]')
    assert page.get_attribute('[data-pill="cacheControl"]', "aria-expanded") == "true"
    page.keyboard.press("Escape")
    assert page.locator(".pill-menu").count() == 0
    assert page.get_attribute('[data-pill="cacheControl"]', "aria-expanded") == "false"


def test_a_pill_that_opens_nothing_still_answers_its_click(page):
    """Streaming is runtime-fixed. It owes the click a reason, on screen."""
    page.click('[data-pill="streaming"]')
    card = page.locator("#whyTip")
    assert card.is_visible()
    assert "answered" in (page.get_attribute("#whyTip", "class") or "")
    box = card.bounding_box()
    assert box["y"] >= 0 and box["y"] + box["height"] <= 820
    assert "runtime fixed" in (card.text_content() or "")
    assert page.locator(".pill-menu").count() == 0
