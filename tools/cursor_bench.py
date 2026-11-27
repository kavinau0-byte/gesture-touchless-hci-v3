"""Cursor smoothness benchmark (no camera needed for `sim`).

  sim     : synthetic hand + realistic landmark noise through KioskController (+ output stage)
            python tools/cursor_bench.py sim [--noise 0.003] [--plot out.png]
  replay  : re-run a trajectory recorded by the app (`--record traj.csv`) with chosen filter settings
            python tools/cursor_bench.py replay --csv traj.csv --cutoff 3 --beta 15

Metrics (lower is better unless noted):
  jitter_px   std-dev of cursor position while the hand is perfectly still
  lag_ms      time the cursor trails the true fingertip on a smooth sweep (added by OUR code only,
              not camera/MediaPipe latency)
  rmse_px     tracking error on that sweep
  overshoot   px past the target after a fast flick
  stall_%     share of 120 Hz output ticks where the cursor does not move during a steady move
              (high = stair-stepping / choppy)
  cv          irregularity of per-tick step size in a steady move (0 = perfectly even)
"""
import argparse
import csv
import math
import os
import random
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from gesture_hci.cursor import CursorSmoother  # noqa: E402
from gesture_hci.kiosk import KioskController  # noqa: E402

SW, SH, MARGIN = 1920, 1080, 0.15
CAM_HZ, OUT_HZ = 30, 120


def pointing_hand(x, y):
    """21 image landmarks of an index-pointing hand whose index tip is at (x, y)."""
    P = [(0.0, 0.0)] * 21
    P[0] = (0.5, 0.9)
    for k, up in enumerate((1, 0, 0, 0)):
        fx, base = 0.42 + 0.04 * k, 5 + 4 * k
        P[base], P[base + 1] = (fx, 0.65), (fx, 0.60)
        P[base + 2] = (fx, 0.52 if up else 0.68)
        P[base + 3] = (fx, 0.45 if up else 0.75)
    P[1], P[2], P[3], P[4] = (0.47, 0.85), (0.45, 0.80), (0.40, 0.75), (0.50, 0.70)
    dx, dy = x - 0.42, y - 0.45
    return [(px + dx, py + dy) for px, py in P]


class HoldStage:
    """Baseline (v3): cursor is set only when a camera frame arrives and then holds."""
    def reset(self): self.p = None
    def feed(self, t, x, y): self.p = (x, y)
    def sample(self, t): return self.p
    def snap(self, *a, **k): pass


def to_px(n):
    return (n - MARGIN) / (1 - 2 * MARGIN)


def simulate(stage, truth, seconds, noise, cutoff, beta, seed=1):
    """truth(t) -> normalised (x, y) fingertip. Returns list of (t, out_x, out_y, true_x_px, true_y_px)."""
    rng = random.Random(seed)
    kc = KioskController(SW, SH, cutoff=cutoff, beta=beta)
    stage.reset()
    out, dt_cam, dt_out = [], 1 / CAM_HZ, 1 / OUT_HZ
    next_cam, t, started = 0.0, 0.0, False
    while t <= seconds:
        if t >= next_cam - 1e-9:
            x, y = truth(next_cam)
            pos, _, _ = kc.update(next_cam, pointing_hand(x + rng.gauss(0, noise), y + rng.gauss(0, noise)))
            if pos:
                stage.feed(next_cam, *pos)
                started = True
            next_cam += dt_cam
        p = stage.sample(t) if started else None
        if p is not None:
            tx, ty = truth(t)
            out.append((t, round(p[0]), round(p[1]), to_px(tx) * SW, to_px(ty) * SH))
        t += dt_out
    return out


def std(v):
    m = sum(v) / len(v)
    return math.sqrt(sum((a - m) ** 2 for a in v) / len(v))


def metrics(stage_factory, cutoff, beta, noise):
    m = {}
    still = simulate(stage_factory(), lambda t: (0.5, 0.5), 10, noise, cutoff, beta)[CAM_HZ * 2:]
    m["jitter_px"] = (std([r[1] for r in still]) + std([r[2] for r in still])) / 2

    A, F = 0.2, 1.0
    sine = lambda t: (0.5 + A * math.sin(2 * math.pi * F * t), 0.5)
    run = simulate(stage_factory(), sine, 6, noise, cutoff, beta)[OUT_HZ:]
    best = None
    for shift_ms in range(0, 301, 2):
        s = shift_ms / 1000
        e = math.sqrt(sum((r[1] - to_px(sine(r[0] - s)[0]) * SW) ** 2 for r in run) / len(run))
        if best is None or e < best[1]:
            best = (shift_ms, e)
    m["lag_ms"] = best[0]
    m["rmse_px"] = math.sqrt(sum((r[1] - r[3]) ** 2 for r in run) / len(run))

    flick = lambda t: (0.3 + 0.4 * min(max(t - 1.0, 0) / 0.15, 1.0), 0.5)
    run = simulate(stage_factory(), flick, 3, 0.0, cutoff, beta)
    target = to_px(0.7) * SW
    m["overshoot"] = max(0.0, max(r[1] for r in run) - target)

    ramp = lambda t: (0.3 + 0.4 * min(max(t - 0.5, 0) / 1.0, 1.0), 0.5)
    run = simulate(stage_factory(), ramp, 2.0, 0.0, cutoff, beta)
    seg = [r[1] for r in run if 0.9 <= r[0] <= 1.4]
    steps = [b - a for a, b in zip(seg, seg[1:])]
    mean = sum(steps) / len(steps)
    m["stall_%"] = 100 * sum(1 for s in steps if s == 0) / len(steps)
    m["cv"] = std(steps) / mean if mean else float("inf")
    return m


PRESETS = {
    "v3 baseline (hold @30Hz)": (HoldStage, 3.0, 15.0),
    "smoother (default)":       (CursorSmoother, 3.0, 15.0),
}


def print_table(rows):
    keys = list(next(iter(rows.values())))
    w = max(len(n) for n in rows) + 2
    print(f"{'config':<{w}}" + "".join(f"{k:>11}" for k in keys))
    for name, m in rows.items():
        print(f"{name:<{w}}" + "".join(f"{m[k]:>11.1f}" for k in keys))


def cmd_sim(a):
    presets = dict(PRESETS)
    if a.extra:
        for spec in a.extra:   # name:tau:horizon:cutoff:beta
            n, tau, hor, co, be = spec.split(":")
            presets[n] = (lambda tau=float(tau), hor=float(hor): CursorSmoother(tau, hor), float(co), float(be))
    rows = {n: metrics(f, co, be, a.noise) for n, (f, co, be) in presets.items()}
    print(f"noise={a.noise} (normalised, ~{a.noise * 640:.1f}px on a 640px camera), camera {CAM_HZ} Hz, output {OUT_HZ} Hz\n")
    print_table(rows)
    if a.plot:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        sine = lambda t: (0.5 + 0.2 * math.sin(2 * math.pi * t), 0.5)
        fig, ax = plt.subplots(2, 1, figsize=(10, 7))
        for n, (f, co, be) in presets.items():
            r = simulate(f(), sine, 3, a.noise, co, be)
            ax[0].plot([x[0] for x in r], [x[1] for x in r], label=n, lw=1)
        ax[0].plot([x[0] for x in r], [x[3] for x in r], "k--", lw=1, label="true fingertip")
        ax[0].set_title("Sweep: cursor x (px) vs time"); ax[0].legend(); ax[0].set_xlim(1, 2.5)
        for n, (f, co, be) in presets.items():
            r = simulate(f(), lambda t: (0.5, 0.5), 5, a.noise, co, be)
            ax[1].plot([x[0] for x in r], [x[1] for x in r], label=n, lw=1)
        ax[1].set_title("Hand held still: cursor x (px) vs time (flatter = steadier)"); ax[1].legend()
        fig.tight_layout(); fig.savefig(a.plot, dpi=110)
        print(f"\nplot saved to {a.plot}")


def cmd_replay(a):
    """Replay a recorded trajectory (t, x, y = raw normalised index tip). Reports jitter assuming the hand was still."""
    rows = [(float(r["t"]), float(r["x"]), float(r["y"])) for r in csv.DictReader(open(a.csv))]
    kc = KioskController(SW, SH, cutoff=a.cutoff, beta=a.beta)
    st = CursorSmoother() if a.stage == "smooth" else HoldStage()
    st.reset()
    raw, out = [], []
    for t, x, y in rows:
        pos, _, _ = kc.update(t, pointing_hand(x, y))
        if pos:
            st.feed(t, *pos)
            raw.append(pos)
            out.append(st.sample(t))
    rx, ry = [p[0] for p in raw], [p[1] for p in raw]
    print(f"{len(rows)} frames. Position std-dev (px) after filter: x={std(rx):.2f} y={std(ry):.2f}")
    print("(only meaningful if you held your hand still while recording)")


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)
    s = sub.add_parser("sim"); s.add_argument("--noise", type=float, default=0.003)
    s.add_argument("--plot"); s.add_argument("--extra", nargs="*", help="name:tau:horizon:cutoff:beta")
    s.set_defaults(fn=cmd_sim)
    r = sub.add_parser("replay"); r.add_argument("--csv", required=True)
    r.add_argument("--cutoff", type=float, default=3.0); r.add_argument("--beta", type=float, default=15.0)
    r.add_argument("--stage", choices=["hold", "smooth"], default="smooth"); r.set_defaults(fn=cmd_replay)
    a = ap.parse_args(); a.fn(a)


if __name__ == "__main__":
    main()
