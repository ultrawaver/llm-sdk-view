"""The per-conversation menu, measured in a real page.

Two of this project's rules are exercised here, not grepped for: the menu
must live in #pillLayer because the sidebar is a scroll container that
clips its descendants, and delete must take two clicks so a misclick can
never cost a conversation.
"""

import pytest

pytestmark = pytest.mark.browser


def send_one_turn(page, text):
    page.fill("#prompt", text)
    page.click("#send")
    page.wait_for_function("() => state.turns.length === 1")
    # The sidebar refreshes after the record lands.
    page.wait_for_selector(".conv-row .conversation")


def open_menu(page):
    page.hover(".conv-row")
    page.click(".conv-kebab")
    page.wait_for_selector(".pill-menu.conv-menu")


def menu_labels(page):
    return page.eval_on_selector_all(
        ".pill-menu.conv-menu .mi", "els => els.map(el => el.textContent)"
    )


def test_the_menu_lives_in_the_overlay_layer_not_the_scroller(
    page, key, fake_provider
):
    """The sidebar scrolls; anything mounted inside it is clipped by it."""
    send_one_turn(page, "hello menu")
    open_menu(page)

    placement = page.evaluate(
        """() => {
            const menu = document.querySelector('.pill-menu.conv-menu');
            return { parent: menu.parentElement.id,
                     insideSidebar: !!menu.closest('#sidebar'),
                     visible: menu.getBoundingClientRect().width > 0 };
        }"""
    )
    assert placement == {
        "parent": "pillLayer",
        "insideSidebar": False,
        "visible": True,
    }
    labels = " | ".join(menu_labels(page))
    assert "Export Markdown" in labels
    assert "Delete…" in labels


def test_the_kebab_appears_when_the_row_is_hovered(page, key, fake_provider):
    send_one_turn(page, "hover me")

    before = page.eval_on_selector(
        ".conv-kebab", "el => getComputedStyle(el).opacity"
    )
    page.hover(".conv-row")
    after = page.eval_on_selector(
        ".conv-kebab", "el => getComputedStyle(el).opacity"
    )

    assert before == "0"
    assert after == "1"


def test_export_downloads_the_markdown_endpoint(page, key, fake_provider):
    send_one_turn(page, "export me")
    open_menu(page)

    href = page.evaluate(
        """() => {
            let seen = null;
            const original = HTMLAnchorElement.prototype.click;
            HTMLAnchorElement.prototype.click = function () { seen = this.href; };
            const item = [...document.querySelectorAll('.pill-menu.conv-menu .mi')]
                .find(el => el.textContent.includes('Export Markdown'));
            item.click();
            HTMLAnchorElement.prototype.click = original;
            return seen;
        }"""
    )
    conversation_id = page.evaluate("() => state.conversationId")
    assert href is not None
    assert href.endswith(
        f"/api/conversations/{conversation_id}/export.md"
    )
    # The menu did its job and left.
    assert page.query_selector(".pill-menu.conv-menu") is None


def test_delete_takes_two_clicks_and_then_is_gone(page, key, fake_provider):
    send_one_turn(page, "delete me")
    open_menu(page)

    # The first click only arms the confirmation - no request leaves.
    page.evaluate(
        """() => {
            window.__deletes = [];
            const original = window.fetch;
            window.fetch = (url, options) => {
                if (options && options.method === 'DELETE') window.__deletes.push(url);
                return original(url, options);
            };
        }"""
    )
    delete_item = page.locator(".pill-menu.conv-menu .mi", has_text="Delete…")
    delete_item.click()

    armed = page.evaluate(
        """() => ({
            text: document.querySelector('.pill-menu.conv-menu .mi.danger')
                        ?.textContent || '',
            fired: window.__deletes.length,
        })"""
    )
    assert "Really delete?" in armed["text"]
    assert armed["fired"] == 0

    # The second click is the delete: the row goes, the chat resets.
    page.click(".pill-menu.conv-menu .mi.danger")
    page.wait_for_function("() => window.__deletes.length === 1")
    page.wait_for_selector(".conv-row", state="detached")
    assert page.evaluate("() => state.conversationId") is None
    assert "No conversations yet." in page.text_content("#conversationList")


def test_escape_closes_the_menu_without_arming_anything(page, key, fake_provider):
    send_one_turn(page, "escape me")
    open_menu(page)
    page.keyboard.press("Escape")

    assert page.query_selector(".pill-menu.conv-menu") is None
    assert page.query_selector(".conv-row") is not None
