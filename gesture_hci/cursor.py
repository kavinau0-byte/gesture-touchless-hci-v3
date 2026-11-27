"""Cursor output stage: turns ~30 Hz target updates into a smooth high-rate path.

The camera only gives a new position about 30 times a second. If the OS cursor is moved only
then, it steps and feels choppy. `CursorSmoother` is sampled by a fast loop (e.g. 120 Hz) and
returns a continuous position between camera frames: it extrapolates a little along the current
velocity (capped), eases toward that point.

Pure Python (no cv2/mediapipe/pyautogui) so it is unit-testable and benchmarkable.
"""
import math


class CursorSmoother:
    def __init__(self, tau=0.01, horizon=0.02, vel_smooth=0.0):
        self.tau = tau              # ease time constant (s): lower = snappier, higher = smoother
        self.horizon = horizon      # max seconds to extrapolate past the last target
        self.vel_smooth = vel_smooth  # 0 = use newest velocity only (least overshoot after a stop)
        self.reset()

    def reset(self):
        self.last = None            # (t, x, y) of newest target
        self.vel = (0.0, 0.0)       # px/s
        self.pos = None             # (x, y) currently output
        self.pt = None              # time of last sample

    def feed(self, t, x, y):
        """New target from the controller (call once per camera frame)."""
        if self.last is not None and t > self.last[0]:
            dt = t - self.last[0]
            vx, vy = (x - self.last[1]) / dt, (y - self.last[2]) / dt
            a = self.vel_smooth
            self.vel = (a * self.vel[0] + (1 - a) * vx, a * self.vel[1] + (1 - a) * vy)
        self.last = (t, x, y)
        if self.pos is None:
            self.pos, self.pt = (x, y), t

    def snap(self, x, y, t=None):
        """Jump straight to a point (used right before a click so it lands exactly)."""
        self.pos = (x, y)
        self.last = (t if t is not None else (self.last[0] if self.last else 0.0), x, y)
        self.vel = (0.0, 0.0)
        if t is not None:
            self.pt = t

    def sample(self, t):
        """Position to show at time t (call at any rate, e.g. 120 Hz). None before first feed."""
        if self.last is None or self.pos is None:
            return None
        lt, lx, ly = self.last
        dt = min(max(t - lt, 0.0), self.horizon)
        tx, ty = lx + self.vel[0] * dt, ly + self.vel[1] * dt
        step = max(t - self.pt, 0.0) if self.pt is not None else 0.0
        self.pt = t
        k = 1.0 - math.exp(-step / self.tau) if self.tau > 0 else 1.0
        self.pos = (self.pos[0] + (tx - self.pos[0]) * k, self.pos[1] + (ty - self.pos[1]) * k)
        return self.pos
