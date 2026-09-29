"""The bubble's own geometry, measured rather than read.

A static grep of the stylesheet cannot see which rule wins the cascade:
``hidden`` is an attribute the browser honours with a rule the author's
``display: flex`` out-specifies, and the empty strips kept their rows -
two blank lines above the answer, one below - while every text-level test
stayed green. So this file does what ``test_topbar_in_a_browser.py`` does
for the pills, for the reading experience: build the bubble the page would
build, and measure it.

Reuses the conftest's browser fixtures on purpose: one harness, one
definition of "the real application on a real port".

No key and no network: the bubble is rendered from a record the test
supplies, never from a live turn.
"""

import pytest

pytestmark = pytest.mark.browser


def region_boxes(page):
    """Height and computed display of every collapsible bubble region."""
    return page.evaluate(
        """() => {
            const body = addAssistantMessage(1, null);
            document.getElementById('messages').appendChild(body.closest('.msg'));
            renderAssistantRecord(body, {
                text: 'A plain answer with nothing around it.',
            });
            const out = {};
            for (const name of ['think-strip', 'think-body', 'tool-strip',
                                'tool-list', 'sources-head', 'sources-list']) {
                const el = body.querySelector('.' + name);
                out[name] = {
                    height: el.getBoundingClientRect().height,
                    display: getComputedStyle(el).display,
                };
            }
            return out;
        }"""
    )


def test_a_bubble_with_nothing_to_report_shows_nothing(page):
    """No thinking, no tool use, no citations: no rows either. The blank
    lines a user reported were empty strips that `display: flex` kept
    laying out against the `hidden` attribute."""
    boxes = region_boxes(page)
    for name, box in boxes.items():
        assert box["height"] == 0, f"{name} takes {box['height']}px when empty"
        assert box["display"] == "none", f"{name} computes to {box['display']}"


def test_a_thinking_bubble_still_shows_its_strip(page):
    """The fix must not hide the regions that have something to say: with
    thinking on the record the strip is visible and carries its label."""
    heights = page.evaluate(
        """() => {
            const body = addAssistantMessage(2, null);
            document.getElementById('messages').appendChild(body.closest('.msg'));
            renderAssistantRecord(body, {
                text: 'The answer.',
                thinking: 'Two words of reasoning.',
            });
            const strip = body.querySelector('.think-strip');
            return {
                height: strip.getBoundingClientRect().height,
                label: strip.querySelector('.think-label').textContent,
                bodyHidden: body.querySelector('.think-body').hidden,
            };
        }"""
    )
    assert heights["height"] > 0
    assert "words" in heights["label"]
    # Collapsed until clicked, but only one click away.
    assert heights["bodyHidden"] is True
