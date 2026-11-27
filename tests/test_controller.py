from conftest import make_hand
from gesture_hci.controller import Controller

FPS = 30


def run(ctl, frames, t0=0.0):
    """frames: list of landmark sets at 30 fps. Returns list of (t, event)."""
    out = []
    for i, pts in enumerate(frames):
        t = t0 + i / FPS
        _, _, ev = ctl.update(t, pts)
        if ev:
            out.append((t, ev))
    return out


def test_open_palm_hold_fires_play_pause_once():
    ev = run(Controller("media"), [make_hand("OPEN_PALM")] * 60)   # 2 s hold
    assert [e[1][2] for e in ev] == ["playpause"]


def test_short_palm_does_not_fire():
    assert run(Controller("media"), [make_hand("OPEN_PALM")] * 12) == []   # 0.4 s


def test_rearm_after_gesture_change():
    frames = [make_hand("OPEN_PALM")] * 30 + [make_hand("FIST")] * 30 + [make_hand("OPEN_PALM")] * 30
    keys = [e[2] for _, e in run(Controller("media"), frames)]
    assert keys == ["playpause", "volumemute", "playpause"]


def test_thumb_up_repeats_while_held():
    ev = run(Controller("media"), [make_hand("THUMB_UP")] * 60)
    assert len(ev) >= 4 and all(e[1][2] == "volumeup" for e in ev)


def test_swipe_right_media_and_slides():
    frames = [make_hand("OPEN_PALM")] * 8 + [make_hand("OPEN_PALM", dx=0.04 * i) for i in range(1, 9)]
    assert [e[2] for _, e in run(Controller("media"), frames)] == ["nexttrack"]
    assert [e[2] for _, e in run(Controller("slides"), frames)] == ["right"]


def test_swipe_left():
    frames = [make_hand("OPEN_PALM", dx=0.2)] * 8 + [make_hand("OPEN_PALM", dx=0.2 - 0.04 * i) for i in range(1, 9)]
    assert [e[2] for _, e in run(Controller("media"), frames)] == ["prevtrack"]


def test_swipe_cooldown_blocks_return_stroke():
    out = [make_hand("OPEN_PALM", dx=0.04 * i) for i in range(0, 9)]
    back = [make_hand("OPEN_PALM", dx=0.32 - 0.04 * i) for i in range(1, 9)]
    ev = run(Controller("media"), [make_hand("OPEN_PALM")] * 8 + out + back)
    assert len(ev) == 1


def test_point_hold_switches_mode_and_is_one_shot():
    c = Controller("media")
    ev = run(c, [make_hand("POINT")] * 75)
    assert [e[1][0] for e in ev] == ["MODE"] and c.mode == "slides"


def test_peace_hold_differs_by_mode():
    assert run(Controller("media"), [make_hand("PEACE")] * 40)[0][1][2] == "f"
    assert run(Controller("slides"), [make_hand("PEACE")] * 40)[0][1][2] == "f5"


def test_lost_hand_resets_state():
    c = Controller("media")
    run(c, [make_hand("OPEN_PALM")] * 30)
    c.lost()
    assert c.stable is None and not c.votes
