"""The cache countdown's words and ladder, protected as text.

What the badge looks like and when its face changes are measured in
test_cache_countdown_in_a_browser.py - colour and animation cannot be
asserted here. This file protects the contract both rely on: the states
exist, the words say what they must say, and nothing is communicated by
motion alone.
"""


def test_the_countdown_slot_is_still_there(static_page):
    assert 'id="cacheState"' in static_page


def test_the_ladder_has_all_four_faces(static_page):
    """green while comfortable, amber in the last minute, red in the last
    thirty seconds, grey once the entry is gone."""
    for face in ('live', 'warn', 'urgent', 'expired'):
        assert f'[data-state="{face}"]' in static_page


def test_the_last_thirty_seconds_move(static_page):
    """The urgent face pulses - and the pulse is declared next to the
    reduced-motion rule that kills it, so nobody can delete one without
    seeing the other."""
    assert "TTL_URGENT_MS = 30 * 1000" in static_page
    assert "@keyframes ttl-pulse" in static_page
    assert "prefers-reduced-motion" in static_page


def test_the_urgent_state_says_what_to_do(static_page):
    """Colour says it is urgent; the words say what to do about it."""
    assert "send now" in static_page


def test_nothing_important_is_motion_only(static_page):
    """With animation off the state must still read: the badge keeps its
    colour when the pulse is killed."""
    assert "cache expired" in static_page
    assert "TTL " in static_page


def test_the_anchor_reads_stamps_the_way_everything_else_does(static_page):
    """Date.parse reads a zone-less stamp as local time, which would move
    the anchor by the whole timezone offset; momentOf exists precisely
    because llm's stored stamps can be that form."""
    assert "momentOf(record.timestamp)" in static_page
