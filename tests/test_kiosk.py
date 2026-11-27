from conftest import make_hand
from gesture_hci.kiosk import KioskController, OneEuro

SW, SH = 1920, 1080


def test_one_euro_smooths_jitter_but_tracks_motion():
    f = OneEuro(1.5, 8.0)
    xs = [f(i / 30, 0.5 + (0.01 if i % 2 else -0.01)) for i in range(30)]
    assert max(xs[10:]) - min(xs[10:]) < 0.01                       # jitter attenuated
    g = OneEuro(3.0, 15.0)
    out = [g(i / 30, i * 0.03) for i in range(30)]
    assert out[-1] > 0.7 * 29 * 0.03                                # follows a fast sweep


def test_open_palm_is_rest():
    k = KioskController(SW, SH)
    pos, click, state = k.update(0.0, make_hand("OPEN_PALM"))
    assert (pos, click, state) == (None, False, "REST")


def test_point_moves_cursor_across_screen():
    k = KioskController(SW, SH)
    left = k.update(0.0, make_hand("POINT", dx=-0.25))[0]
    k2 = KioskController(SW, SH)
    right = k2.update(0.0, make_hand("POINT", dx=0.25))[0]
    assert left[0] < right[0]
    assert 0 <= left[0] <= SW and 0 <= right[1] <= SH


def test_pinch_clicks_once_and_freezes_cursor():
    k = KioskController(SW, SH)
    t, clicks, frozen_positions = 0.0, 0, []
    for _ in range(15):                                              # approach, no pinch
        k.update(t, make_hand("POINT")); t += 1 / 30
    for _ in range(15):                                              # hold pinch 0.5 s
        pos, click, state = k.update(t, make_hand("POINT", pinch=True)); t += 1 / 30
        clicks += click
        frozen_positions.append(pos)
    assert clicks == 1 and state == "PINCH"
    assert len(set(frozen_positions)) == 1                           # cursor stayed put


def test_release_rearms_click_after_debounce():
    k = KioskController(SW, SH)
    t, clicks = 0.0, 0
    for phase in (True, False, True):
        for _ in range(20):
            clicks += k.update(t, make_hand("POINT", pinch=phase))[1]; t += 1 / 30
    assert clicks == 2


def test_click_threshold_configurable():
    strict = KioskController(SW, SH, click=0.001)
    assert not any(strict.update(i / 30, make_hand("POINT", pinch=True))[1] for i in range(10))


def test_ratio_is_exposed_and_reset():
    k = KioskController(SW, SH)
    assert k.ratio is None
    k.update(0.0, make_hand("POINT"))
    point_ratio = k.ratio
    assert point_ratio is not None
    k.update(1 / 30, make_hand("POINT", pinch=True))
    assert k.ratio < point_ratio
    k.lost()
    assert k.ratio is None
