"""
Touchless gesture controller (MediaPipe Hands + OpenCV + PyAutoGUI)  --  v2

Two modes. Point with ONE index finger (hold ~1 s) to switch between them.

MEDIA mode (Spotify / YouTube / VLC -- media keys work even when the window is not focused)
  Open palm (hold) ...... Play / Pause
  Fist (hold) ........... Mute
  Thumb up / down ....... Volume + / -   (repeats while held)
  Peace sign (hold) ..... Fullscreen (f)
  Swipe right / left .... Next / Previous track

SLIDES mode (PowerPoint slideshow -- click the slideshow first so it has focus)
  Swipe right / left .... Next / Previous slide
  Peace sign (hold) ..... Start slideshow (F5)
  Open palm (hold) ...... Black screen (b)
  Thumb up / down ....... Volume + / -

One-shot gestures fire ONCE per hold: lower your hand or change gesture to re-arm.

KIOSK mode (--mode kiosk): touchless mouse for the ordering kiosk (kiosk.html)
  Index finger ......... move the cursor
  Pinch thumb+index .... click (cursor freezes while you pinch so the click lands)
  Open palm ............ rest (cursor stops following your hand)

Run:   python gesture_control.py --dry     (overlay only, sends nothing)
       python gesture_control.py           (real keystrokes)
       python gesture_control.py --mode slides
Quit:  press q in the video window.
"""
import argparse
import math
import time
from collections import Counter, deque

# ----------------------------------------------------------------- landmarks
WRIST = 0
THUMB_MCP, THUMB_IP, THUMB_TIP = 2, 3, 4
PINKY_MCP = 17
MIDDLE_MCP = 9
FINGERS = {"index": (6, 8), "middle": (10, 12), "ring": (14, 16), "pinky": (18, 20)}


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


# gesture -> (label, key, hold seconds, repeat interval or None)
MODES = {
    "media": {
        "OPEN_PALM": ("Play / Pause", "playpause", 0.6, None),
        "FIST": ("Mute", "volumemute", 0.6, None),
        "THUMB_UP": ("Volume +", "volumeup", 0.3, 0.25),
        "THUMB_DOWN": ("Volume -", "volumedown", 0.3, 0.25),
        "PEACE": ("Fullscreen", "f", 0.6, None),
        "SWIPE_RIGHT": ("Next track", "nexttrack", 0.0, None),
        "SWIPE_LEFT": ("Previous track", "prevtrack", 0.0, None),
    },
    "slides": {
        "OPEN_PALM": ("Black screen", "b", 0.6, None),
        "THUMB_UP": ("Volume +", "volumeup", 0.3, 0.25),
        "THUMB_DOWN": ("Volume -", "volumedown", 0.3, 0.25),
        "PEACE": ("Start slideshow", "f5", 0.6, None),
        "SWIPE_RIGHT": ("Next slide", "right", 0.0, None),
        "SWIPE_LEFT": ("Previous slide", "left", 0.0, None),
    },
}
SWIPE_COOLDOWN = 1.2   # long enough that bringing your hand back is not read as a swipe
MODE_HOLD = 1.0
VOTE_WINDOW, VOTE_MIN = 7, 4


class Controller:
    """Camera-independent logic: feed it landmarks, it returns events."""

    def __init__(self, mode="media"):
        self.mode = mode
        self.hist = deque()
        self.votes = deque(maxlen=VOTE_WINDOW)
        self.stable = None
        self.since = 0.0
        self.fired_hold = False
        self.last_fire = {}
        self.last_swipe = -10.0

    def lost(self):
        self.hist.clear()
        self.votes.clear()
        self.stable = None
        self.fired_hold = False

    def update(self, now, pts):
        """Returns (stable_gesture, progress 0-1, event or None).
        event = (name, label, key)  -- key is None for a mode switch."""
        raw = classify_static(pts)
        self.hist.append((now, pts[MIDDLE_MCP][0], pts[MIDDLE_MCP][1]))
        while self.hist and now - self.hist[0][0] > 0.5:
            self.hist.popleft()

        self.votes.append(raw)
        top, cnt = Counter(self.votes).most_common(1)[0]
        if cnt >= VOTE_MIN and top != self.stable:
            self.stable, self.since, self.fired_hold = top, now, False
        gesture = self.stable or "NONE"
        actions = MODES[self.mode]

        swipe = detect_swipe(self.hist, self.stable)
        if swipe and now - self.last_swipe > SWIPE_COOLDOWN:
            self.last_swipe = now
            self.hist.clear()
            self.fired_hold = True          # no play/pause right after a swipe
            if swipe in actions:
                label, key, _, _ = actions[swipe]
                return gesture, 0.0, (swipe, label, key)
            return gesture, 0.0, None

        if now - self.last_swipe < 0.6:
            return gesture, 0.0, None

        held = now - self.since
        if gesture == "POINT":
            progress = min(held / MODE_HOLD, 1.0)
            if held >= MODE_HOLD and not self.fired_hold and palm_still(self.hist):
                self.fired_hold = True
                self.mode = "slides" if self.mode == "media" else "media"
                return gesture, 1.0, ("MODE", f"Mode: {self.mode.upper()}", None)
            return gesture, progress, None

        if gesture in actions:
            label, key, hold, repeat = actions[gesture]
            progress = min(held / hold, 1.0) if hold else 0.0
            if held >= hold:
                if repeat:
                    if now - self.last_fire.get(gesture, -10.0) >= repeat:
                        self.last_fire[gesture] = now
                        return gesture, progress, (gesture, label, key)
                elif not self.fired_hold and palm_still(self.hist):
                    self.fired_hold = True
                    return gesture, progress, (gesture, label, key)
            return gesture, progress, None
        return gesture, 0.0, None


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
    CLICK = 0.25   # pinch ratio that counts as a click

    def __init__(self, sw, sh, margin=0.15, cutoff=3.0, beta=15.0):
        self.sw, self.sh, self.m = sw, sh, margin
        self.cutoff, self.beta = cutoff, beta
        self.pos, self.pinched, self.last_click = None, False, -10.0
        self.trail, self.frozen = deque(), None
        self.fx, self.fy = OneEuro(cutoff, beta), OneEuro(cutoff, beta)

    def lost(self):
        self.pinched, self.frozen = False, None
        self.trail.clear()
        self.fx, self.fy = OneEuro(self.cutoff, self.beta), OneEuro(self.cutoff, self.beta)

    def update(self, now, pts, wpts=None):
        """pts: 2D image landmarks (cursor position). wpts: optional 3D world landmarks used for
        finger/pinch detection, so it works however the hand is oriented (e.g. arm straight,
        pointing at the camera). Returns (cursor_xy or None, click, state)."""
        fp = wpts if wpts is not None else pts
        up, _ = finger_states(fp)
        size = dist(fp[WRIST], fp[MIDDLE_MCP])
        ratio = dist(fp[THUMB_TIP], fp[8]) / size
        if all(up.values()) or not (up["index"] or ratio < self.FREEZE):
            self.pinched, self.frozen = False, None
            return None, False, "REST"
        span = 1 - 2 * self.m
        nx = min(max((pts[8][0] - self.m) / span, 0), 1)
        ny = min(max((pts[8][1] - self.m) / span, 0), 1)
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


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dry", action="store_true", help="do not send keystrokes")
    ap.add_argument("--cam", type=int, default=0, help="camera index")
    ap.add_argument("--mode", choices=list(MODES) + ["kiosk"], default="media")
    ap.add_argument("--cutoff", type=float, default=3.0, help="kiosk: lower = steadier but laggier (try 1.5-6)")
    ap.add_argument("--beta", type=float, default=15.0, help="kiosk: higher = less lag on fast moves (try 8-30)")
    args = ap.parse_args()

    import cv2
    import mediapipe as mp

    kiosk = args.mode == "kiosk"
    press, pg = None, None
    if not args.dry:
        import pyautogui as pg

        pg.PAUSE = 0
        pg.FAILSAFE = not kiosk   # cursor must be able to reach screen corners in kiosk mode
        press = pg.press
    if kiosk:
        if pg is None:
            import pyautogui
            sw, sh = pyautogui.size()
        else:
            sw, sh = pg.size()
        kc = KioskController(sw, sh, cutoff=args.cutoff, beta=args.beta)

    import os
    cap = cv2.VideoCapture(args.cam, cv2.CAP_DSHOW) if os.name == "nt" else cv2.VideoCapture(args.cam)
    cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)   # always process the newest frame
    cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*"MJPG"))
    cap.set(cv2.CAP_PROP_FPS, 30)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
    print(f"Camera reports {cap.get(cv2.CAP_PROP_FPS):.0f} fps at "
          f"{cap.get(cv2.CAP_PROP_FRAME_WIDTH):.0f}x{cap.get(cv2.CAP_PROP_FRAME_HEIGHT):.0f}")
    hands = mp.solutions.hands.Hands(
        max_num_hands=1, model_complexity=0,
        min_detection_confidence=0.7, min_tracking_confidence=0.6,
    )
    draw = mp.solutions.drawing_utils

    ctl = Controller("media" if kiosk else args.mode)
    last_action, last_action_t = "", -10.0
    prev_t, fps = time.perf_counter(), 0.0

    while cap.isOpened():
        ok, frame = cap.read()
        if not ok:
            break
        t_start = time.perf_counter()
        frame = cv2.flip(frame, 1)
        res = hands.process(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
        now = time.perf_counter()
        gesture, progress = "NONE", 0.0

        if res.multi_hand_landmarks:
            lms = res.multi_hand_landmarks[0]
            draw.draw_landmarks(frame, lms, mp.solutions.hands.HAND_CONNECTIONS)
            pts = [(l.x, l.y) for l in lms.landmark]
            if kiosk:
                wl = res.multi_hand_world_landmarks
                wpts = [(l.x, l.y, l.z) for l in wl[0].landmark] if wl else None
                pos, click, gesture = kc.update(now, pts, wpts)
                if pos and pg:
                    pg.moveTo(pos[0], pos[1], _pause=False)
                if click:
                    if pg:
                        pg.click()
                    last_action, last_action_t = "Click", now
                    print(f"[{time.strftime('%H:%M:%S')}] CLICK at {pos}")
                tip = (int(pts[8][0] * frame.shape[1]), int(pts[8][1] * frame.shape[0]))
                cv2.circle(frame, tip, 14, (0, 0, 255) if gesture == "PINCH" else (0, 255, 0), 3)
                event = None
            else:
                gesture, progress, event = ctl.update(now, pts)
            if event:
                name, label, key = event
                if key and press:
                    press(key)
                last_action, last_action_t = label, now
                print(f"[{time.strftime('%H:%M:%S')}] {name} -> {label}")
        else:
            ctl.lost()
            if kiosk:
                kc.lost()

        latency_ms = (time.perf_counter() - t_start) * 1000
        fps = 0.9 * fps + 0.1 / max(now - prev_t, 1e-6)
        prev_t = now

        # ---- overlay
        w = frame.shape[1]
        cv2.rectangle(frame, (0, 0), (w, 92), (30, 30, 30), -1)
        cv2.putText(frame, f"Gesture: {gesture}", (10, 26), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
        if now - last_action_t < 1.5:
            cv2.putText(frame, f"Action: {last_action}", (10, 56), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
        cv2.putText(frame, f"MODE: {'KIOSK' if kiosk else ctl.mode.upper()} | {fps:3.0f} FPS | {latency_ms:4.1f} ms"
                    + (" | DRY RUN" if args.dry else ""),
                    (10, 82), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200, 200, 200), 1)
        cv2.rectangle(frame, (w - 130, 12), (w - 10, 30), (90, 90, 90), 1)
        cv2.rectangle(frame, (w - 130, 12), (w - 130 + int(120 * progress), 30), (0, 200, 255), -1)

        cv2.imshow("Gesture HCI  (q to quit)", frame)
        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
