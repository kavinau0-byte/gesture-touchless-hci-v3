import pytest
from collections import deque
from conftest import make_hand
from gesture_hci.geometry import classify_static, detect_swipe, palm_still


@pytest.mark.parametrize("g", ["OPEN_PALM", "FIST", "THUMB_UP", "THUMB_DOWN", "PEACE", "POINT"])
def test_classify_static(g):
    assert classify_static(make_hand(g)) == g


def test_classify_is_translation_invariant():
    assert classify_static(make_hand("PEACE", dx=-0.2, dy=-0.3)) == "PEACE"


def test_swipe_right_left_and_rejects_vertical_or_slow():
    right = deque([(0.00, 0.2, 0.5), (0.10, 0.3, 0.5), (0.20, 0.4, 0.5), (0.30, 0.55, 0.5)])
    left = deque((t, 1.0 - x, y) for t, x, y in right)
    vertical = deque([(0.00, 0.5, 0.2), (0.10, 0.5, 0.4), (0.20, 0.5, 0.6), (0.30, 0.5, 0.8)])
    assert detect_swipe(right, "OPEN_PALM") == "SWIPE_RIGHT"
    assert detect_swipe(left, "OPEN_PALM") == "SWIPE_LEFT"
    assert detect_swipe(vertical, "OPEN_PALM") is None
    assert detect_swipe(right, "FIST") is None                     # open palm only
    assert detect_swipe(deque(list(right)[:3]), "OPEN_PALM") is None


def test_palm_still():
    assert palm_still(deque([(0, 0.5, 0.5), (0.1, 0.51, 0.5), (0.2, 0.5, 0.52)]))
    assert not palm_still(deque([(0, 0.5, 0.5), (0.1, 0.6, 0.5), (0.2, 0.7, 0.5)]))
    assert not palm_still(deque([(0, 0.5, 0.5)]))
