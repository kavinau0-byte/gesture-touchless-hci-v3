"""Touchless kiosk: index-finger cursor + pinch-to-click, with One-Euro smoothing."""
import math
from collections import deque

from .geometry import INDEX_TIP, MIDDLE_MCP, THUMB_TIP, WRIST, dist, finger_states


class OneEuro:
    """Adaptive low-pass filter: smooth when slow (steady aim), responsive when fast (no lag)."""

    def __init__(self, min_cutoff=1.5, beta=8.0, d_cutoff=1.0):
        self.min_cutoff, self.beta, self.d_cutoff = min_cutoff, beta, d_cutoff
        self.t = self.x = None
        self.dx = 0.0

    @staticmethod
    def _alpha(te, cutoff):
        return 1.0 / (1.0 + (1.0 / (2 * math.pi * cutoff)) / te)

    def __call__(self, t, x):
        if self.t is None:
            self.t, self.x = t, x
            return x
        te = max(t - self.t, 1e-3)
        self.dx += self._alpha(te, self.d_cutoff) * ((x - self.x) / te - self.dx)
        self.x += self._alpha(te, self.min_cutoff + self.beta * abs(self.dx)) * (x - self.x)
        self.t = t
        return self.x


class KioskController:
    """Index-finger cursor + pinch-to-click. Open palm = rest."""

    FREEZE = 0.6   # pinch ratio below which the cursor stops following the fingertip

    def __init__(self, sw, sh, margin=0.15, cutoff=3.0, beta=15.0, click=0.25):
        self.sw, self.sh, self.m = sw, sh, margin
        self.cutoff, self.beta = cutoff, beta
        self.CLICK = click   # pinch ratio that counts as a click
        self.pos, self.pinched, self.last_click = None, False, -10.0
        self.ratio = None
        self.trail, self.frozen = deque(), None
        self.fx, self.fy = OneEuro(cutoff, beta), OneEuro(cutoff, beta)

    def lost(self):
        self.pinched, self.frozen = False, None
        self.ratio = None
        self.trail.clear()
        self.fx, self.fy = OneEuro(self.cutoff, self.beta), OneEuro(self.cutoff, self.beta)

    def update(self, now, pts, wpts=None):
        """pts: 2D image landmarks (cursor position). wpts: optional 3D world landmarks used for
        finger/pinch detection, so it works however the hand is oriented (e.g. arm straight,
        pointing at the camera). Returns (cursor_xy or None, click, state)."""
        fp = wpts if wpts is not None else pts
        up, _ = finger_states(fp)
        size = dist(fp[WRIST], fp[MIDDLE_MCP])
        ratio = dist(fp[THUMB_TIP], fp[INDEX_TIP]) / size
        self.ratio = ratio
        if all(up.values()) or not (up["index"] or ratio < self.FREEZE):
            self.pinched, self.frozen = False, None
            return None, False, "REST"
        span = 1 - 2 * self.m
        nx = min(max((pts[INDEX_TIP][0] - self.m) / span, 0), 1)
        ny = min(max((pts[INDEX_TIP][1] - self.m) / span, 0), 1)
        sx, sy = self.fx(now, nx) * self.sw, self.fy(now, ny) * self.sh
        self.trail.append((now, (sx, sy)))
        while self.trail and now - self.trail[0][0] > 0.3:
            self.trail.popleft()
        if ratio < self.FREEZE:
            if self.frozen is None:   # aim point = where the cursor was ~0.12 s before the pinch began
                old = [p for t, p in self.trail if now - t >= 0.12]
                self.frozen = old[-1] if old else (sx, sy)
            self.pos = self.frozen
        else:
            self.frozen, self.pos = None, (sx, sy)
        click = False
        if not self.pinched and ratio < self.CLICK and now - self.last_click > 0.5:
            self.pinched, self.last_click, click = True, now, True
        elif self.pinched and ratio > 0.4:
            self.pinched = False
        return self.pos, click, ("PINCH" if ratio < self.CLICK else "MOVE")
