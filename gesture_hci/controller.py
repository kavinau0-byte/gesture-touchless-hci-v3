"""Media / slides gesture controller (camera independent: feed landmarks, get events)."""
from collections import Counter, deque

from .geometry import MIDDLE_MCP, classify_static, detect_swipe, palm_still

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
