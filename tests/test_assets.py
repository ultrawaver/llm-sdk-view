"""The page's own assets are versioned, so a fix cannot be lost to cache.

This project is edited several times a day and served from a local port. No
cache header alone settles whether a browser keeps its copy; the version in
the URL makes a replaced file unreachable instead of merely discouraged.
"""

import asyncio

import pytest
from starlette.requests import Request

from llm_sdk_view import app as app_module
from llm_sdk_view.app import asset_version


def _request(name: str | None = None) -> Request:
    """A request with only what these endpoints read - never a body."""
    scope = {
        "type": "http",
        "method": "GET",
        "path": "/static/" + name if name else "/",
        "query_string": b"",
        "headers": [],
    }
    if name:
        scope["path_params"] = {"name": name}
    return Request(scope)


def test_the_version_follows_the_file(tmp_path, monkeypatch):
    """A changed file is a different URL, or the browser answers it itself."""
    monkeypatch.setattr(app_module, "STATIC", tmp_path)
    (tmp_path / "app.js").write_text("one")

    first = asset_version("app.js")

    assert len(first) == 8
    (tmp_path / "app.js").write_text("one, changed")
    assert asset_version("app.js") != first


def test_the_page_points_at_the_version_it_serves():
    html = asyncio.run(app_module.index(_request())).body.decode()

    assert "/static/app.js?v=" in html
    assert "/static/app.css?v=" in html
    # The vendored highlighter is not versioned: it never changes.
    assert "/static/vendor/prism.min.js?v=" not in html


def test_the_page_is_never_reused_from_disk():
    response = asyncio.run(app_module.index(_request()))

    assert response.headers["cache-control"] == "no-store"


def test_a_versioned_asset_is_revalidated(monkeypatch, tmp_path):
    """The version decides what changed; the header decides it is asked."""
    monkeypatch.setattr(app_module, "STATIC", tmp_path)
    (tmp_path / "app.js").write_text("one")
    request = _request("app.js")

    response = asyncio.run(app_module.asset(request))

    assert response.headers["cache-control"] == "no-cache"
    assert response.headers["etag"] == f'"{asset_version("app.js")}"'


def test_the_asset_route_still_refuses_to_read_outside_static(tmp_path, monkeypatch):
    monkeypatch.setattr(app_module, "STATIC", tmp_path)

    assert asyncio.run(app_module.asset(_request("../app.py"))).status_code == 404


# --- a stale tab announces itself instead of impersonating the fixed app -------


def test_the_server_names_the_build_it_would_serve():
    response = asyncio.run(app_module.version(_request()))

    assert response.headers["cache-control"] == "no-store"


def test_the_version_answer_matches_the_asset_fingerprint():
    import json

    response = asyncio.run(app_module.version(_request()))

    assert json.loads(response.body)["version"] == asset_version("app.js")


def test_the_page_compares_its_build_against_the_servers(static_page):
    """A tab that kept yesterday's js must say so on itself - silently
    running an old build is how one click bug was reported three times."""
    assert 'id="staleBanner"' in static_page
    assert 'id="buildStamp"' in static_page
    assert "script[src*=\"/static/app.js\"]" in static_page
    assert "checkStaleBuild" in static_page
    assert "visibilitychange" in static_page
    assert "location.reload()" in static_page


def test_the_banner_sits_above_everything_and_is_the_reload_button(static_page):
    assert "#staleBanner" in static_page
    assert "z-index: 400" in static_page


def test_the_pill_flash_survives_the_scroller(static_page):
    """The pill row clips a box-shadow halo, and a background wash proved
    imperceptible, so the answer to a click is a shake - motion survives
    both the clipping and the human eye. An invisible flash is the same
    as no answer."""
    assert "@keyframes pill-shake" in static_page
    assert ".pill.hit { animation: pill-shake" in static_page


if __name__ == "__main__":
    pytest.main([__file__])
