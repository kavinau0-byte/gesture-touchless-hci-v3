import math
import os
import sys

from gesture_hci.cursor import CursorSmoother
from gesture_hci.output import CursorOutput

sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "tools"))
import cursor_bench as cb  # noqa: E402


def test_sample_before_feed_is_none():
    assert CursorSmoother().sample(0.0) is None


def test_stationary_target_stays_put():
    s = CursorSmoother()
    for i in range(30):
        s.feed(i / 30, 500.0, 300.0)
    assert all(s.sample(1.0 + k / 120) == (500.0, 300.0) for k in range(40))


def test_steady_move_is_continuous_between_camera_frames():
    s = CursorSmoother()
    xs = []
    for i in range(40):                     # 600 px/s, camera 30 Hz, sampled at 120 Hz
        s.feed(i / 30, 20.0 * i, 100.0)
        xs += [s.sample(i / 30 + k / 120)[0] for k in range(4)]
    steps = [b - a for a, b in zip(xs[40:], xs[41:])]
    assert all(st > 0 for st in steps)      # never stalls
    assert max(steps) < 3 * min(steps)      # roughly even steps (no stair-stepping)


def test_overshoot_after_stop_is_bounded():
    s = CursorSmoother()
    for i in range(10):
        s.feed(i / 30, 20.0 * i, 0.0)       # moving at 600 px/s
    last = 180.0
    for i in range(10, 20):
        s.feed(i / 30, last, 0.0)           # abrupt stop
    peak = max(s.sample(0.3 + k / 120)[0] for k in range(120))
    assert peak - last <= 600 * s.horizon + 1


def test_snap_lands_exactly():
    s = CursorSmoother()
    s.feed(0.0, 0.0, 0.0); s.feed(1 / 30, 50.0, 0.0)
    s.snap(123.4, 56.7, t=1 / 30)
    assert s.sample(1 / 30) == (123.4, 56.7)


def test_reset_starts_fresh():
    s = CursorSmoother()
    s.feed(0.0, 10.0, 10.0)
    s.reset()
    assert s.sample(0.1) is None


class Rec:
    def __init__(self): self.moves = []
    def __call__(self, x, y): self.moves.append((x, y))


def test_output_sends_more_moves_than_camera_frames_and_stops_on_rest():
    rec = Rec()
    out = CursorOutput(rec)
    for i in range(10):
        out.feed(i / 30, 20.0 * i, 0.0)
        for k in range(4):
            out.tick(i / 30 + k / 120)
    assert len(rec.moves) > 10 * 2
    assert all(isinstance(v, int) for m in rec.moves for v in m)
    out.rest()
    n = len(rec.moves)
    out.tick(1.0); out.tick(1.01)
    assert len(rec.moves) == n              # nothing sent while resting


def test_output_does_not_resend_same_pixel():
    rec = Rec()
    out = CursorOutput(rec)
    out.feed(0.0, 100.0, 100.0)
    for k in range(10):
        out.tick(k / 120)
    assert len(rec.moves) <= 1


def test_snap_now_moves_immediately():
    rec = Rec()
    out = CursorOutput(rec)
    out.feed(0.0, 10.0, 10.0)
    out.snap_now(400.4, 200.6, t=0.0)
    assert rec.moves[-1] == (400, 201)


def test_bench_smoother_beats_v3_baseline():
    """Regression guard for the numbers in tools/cursor_bench.py (simulated hand, noise 0.003)."""
    base = cb.metrics(cb.HoldStage, 3.0, 15.0, 0.003)
    new = cb.metrics(CursorSmoother, 3.0, 15.0, 0.003)
    assert base["stall_%"] > 50 and new["stall_%"] < 5       # no more stair-stepping
    assert new["lag_ms"] <= base["lag_ms"]                   # not slower
    assert new["rmse_px"] < base["rmse_px"]
    assert new["jitter_px"] < base["jitter_px"] + 1.0        # steadiness not meaningfully worse
    assert new["overshoot"] < 80                             # extreme 1100 px / 0.15 s flick


def test_bench_replay_runs(tmp_path, capsys):
    p = tmp_path / "t.csv"
    p.write_text("t,x,y\n" + "\n".join(f"{i/30:.4f},0.5,0.5" for i in range(30)) + "\n")
    class A: csv = str(p); cutoff = 3.0; beta = 15.0; stage = "smooth"
    cb.cmd_replay(A)
    assert "30 frames" in capsys.readouterr().out
