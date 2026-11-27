"""Pure landmark geometry: finger state, static gesture classification, swipe and stillness.

No camera or OpenCV/MediaPipe imports here, so everything is unit-testable.
Landmarks are 21 (x, y) or (x, y, z) tuples in MediaPipe Hands order.
"""
import math

WRIST = 0
THUMB_MCP, THUMB_IP, THUMB_TIP = 2, 3, 4
INDEX_TIP = 8
PINKY_MCP = 17
MIDDLE_MCP = 9
FINGERS = {"index": (6, 8), "middle": (10, 12), "ring": (14, 16), "pinky": (18, 20)}

GESTURES = ("OPEN_PALM", "FIST", "THUMB_UP", "THUMB_DOWN", "PEACE", "POINT", "NONE")


def dist(a, b):
    return math.dist(a, b)   # works for 2D (x, y) and 3D (x, y, z) points


def finger_states(pts):
    wrist = pts[WRIST]
    up = {n: dist(wrist, pts[tip]) > dist(wrist, pts[pip]) for n, (pip, tip) in FINGERS.items()}
    thumb = dist(pts[THUMB_TIP], pts[PINKY_MCP]) > dist(pts[THUMB_IP], pts[PINKY_MCP])
    return up, thumb


def classify_static(pts):
    """OPEN_PALM, FIST, THUMB_UP, THUMB_DOWN, PEACE, POINT or NONE."""
    up, thumb = finger_states(pts)
    n_up = sum(up.values())
    if n_up == 4:
        return "OPEN_PALM"
    if n_up == 2 and up["index"] and up["middle"]:
        return "PEACE"
    if n_up == 1 and up["index"]:
        return "POINT"
    if n_up == 0:
        if thumb:
            size = dist(pts[WRIST], pts[MIDDLE_MCP])
            rise = pts[THUMB_MCP][1] - pts[THUMB_TIP][1]
            if rise > 0.5 * size:
                return "THUMB_UP"
            if rise < -0.5 * size:
                return "THUMB_DOWN"
            return "NONE"
        return "FIST"
    return "NONE"


def detect_swipe(hist, gesture, min_dx=0.25):
    """hist: deque of (t, x, y) palm positions. Returns SWIPE_RIGHT / SWIPE_LEFT / None."""
    if gesture != "OPEN_PALM" or len(hist) < 4:
        return None
    t0, x0, y0 = hist[0]
    t1, x1, y1 = hist[-1]
    dx, dy = x1 - x0, y1 - y0
    if t1 - t0 < 0.12:
        return None
    if abs(dx) >= min_dx and abs(dy) < 0.6 * abs(dx):
        return "SWIPE_RIGHT" if dx > 0 else "SWIPE_LEFT"
    return None


def palm_still(hist, tol=0.05):
    if len(hist) < 3:
        return False
    xs = [h[1] for h in hist]
    ys = [h[2] for h in hist]
    return (max(xs) - min(xs)) < tol and (max(ys) - min(ys)) < tol
