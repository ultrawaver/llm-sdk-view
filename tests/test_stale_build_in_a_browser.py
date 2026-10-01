"""The build-freshness banner, measured in a real page.

Two mismatches share one symptom and need opposite answers. This tab's js can
be older than the build the server would serve - a reload fixes that. Or the
server *process* can be older than the code on disk, which a reload cannot
touch, and which nothing else on the page can reveal: the process serves the
current app.js off the disk, so every build string the page can see is the new
one.

Whether a banner is clickable is a behaviour, and this project's rule is that
behaviour is measured rather than read - the last control that was called
working from its source was being clipped away by an ancestor's overflow. So
the click in the first check is a real click on the element, and the thing it
would destroy is a value planted on the page immediately before it.
"""

import json

import pytest

pytestmark = pytest.mark.browser

BANNER = """() => {
  const b = byId('staleBanner');
  const style = getComputedStyle(b);
  return {
    hidden: b.hidden,
    text: b.textContent,
    inert: b.classList.contains('inert'),
    cursor: style.cursor,
    display: style.display,
    hasAction: !!staleBannerAction,
  };
}"""


def _version(page, *, version, server_stale):
    """Answer /api/version for this page, and nothing else."""
    page.route(
        "**/api/version",
        lambda route: route.fulfill(
            status=200,
            content_type="application/json",
            body=json.dumps(
                {
                    "version": version,
                    "server": {
                        "loaded_at": "2026-10-01T00:00:00+00:00",
                        "newest_source": "2026-10-01T09:00:00+00:00",
                        "stale": server_stale,
                    },
                }
            ),
        ),
    )


def _check(page):
    """Run the check now, past the fifteen-second throttle it keeps."""
    page.evaluate("() => { staleCheckedAt = 0; return checkStaleBuild(); }")


def test_a_server_running_old_code_is_told_apart_from_a_stale_tab(page):
    """The build matches, so the tab is current - and the server is not.

    This is the state that has no other symptom: the page is up to date, the
    build stamp is up to date, and the answers are wrong. It must say so, and
    it must not offer a reload it cannot perform.
    """
    _version(page, version=page.evaluate("() => BUILD"), server_stale=True)
    _check(page)

    banner = page.evaluate(BANNER)
    assert banner["hidden"] is False, "a stale server must be said out loud"
    assert banner["display"] != "none"
    assert "restart" in banner["text"]
    assert banner["text"].lower().count("reload") == 1, (
        "the thing that will not help is worth saying, exactly once"
    )
    assert banner["inert"] is True
    assert banner["cursor"] == "default", "it must not look clickable"
    assert banner["hasAction"] is False, "there is no action to take here"

    # A real click, and the reload it would cause would take this value with
    # it - so the value surviving is the measurement. The click is forced:
    # the banner carries aria-disabled while it has no action to offer, and
    # Playwright refuses to click a control that says it is disabled. A
    # pointer still reaches it, so the hit test above is what says it is not
    # being clipped or covered - the defect this project has shipped before.
    centre = page.evaluate(
        """() => {
          const b = byId('staleBanner');
          const r = b.getBoundingClientRect();
          const at = document.elementFromPoint(r.x + r.width / 2, r.y + r.height / 2);
          return at && (at.id || at.className);
        }"""
    )
    assert centre == "staleBanner", "the banner must be what is under the pointer"

    page.evaluate("() => { window.__bannerProbe = 'still here'; }")
    page.click("#staleBanner", force=True)
    page.wait_for_timeout(400)
    assert page.evaluate("() => window.__bannerProbe") == "still here"


def test_a_stale_tab_is_still_offered_the_reload_that_fixes_it(page):
    """The older state, kept working: this is the one reloading does fix."""
    _version(page, version=page.evaluate("() => BUILD") + "-older", server_stale=False)
    _check(page)

    banner = page.evaluate(BANNER)
    assert banner["hidden"] is False
    assert "click to reload" in banner["text"]
    assert banner["inert"] is False
    assert banner["cursor"] == "pointer"
    assert banner["hasAction"] is True


def test_a_current_server_and_tab_say_nothing(page):
    """And it goes away again, rather than sitting there once it is resolved."""
    _version(page, version=page.evaluate("() => BUILD"), server_stale=True)
    _check(page)
    assert page.evaluate(BANNER)["hidden"] is False

    page.unroute("**/api/version")
    _version(page, version=page.evaluate("() => BUILD"), server_stale=False)
    _check(page)

    banner = page.evaluate(BANNER)
    assert banner["hidden"] is True
    assert banner["inert"] is False, "a hidden banner must not stay inert"
    assert banner["hasAction"] is False
