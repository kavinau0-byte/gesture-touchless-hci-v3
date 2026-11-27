"""High-rate cursor output thread. Hardware is injected as `move(x, y)`, so this stays testable."""
import threading
import time

from .cursor import CursorSmoother


class CursorOutput:
    """Calls move(x, y) ~120 times/s with a smooth path, while the camera loop only feeds ~30 Hz."""

    def __init__(self, move, smoother=None, hz=120, clock=time.perf_counter):
        self.move, self.hz, self.clock = move, hz, clock
        self.sm = smoother or CursorSmoother()
        self.lock = threading.Lock()
        self.stop_flag = threading.Event()
        self.thread = None
        self.last_sent = None

    # ---- called from the camera loop
    def feed(self, t, x, y):
        with self.lock:
            self.sm.feed(t, x, y)

    def rest(self):
        """Hand lost / open palm: stop moving the cursor; next feed starts fresh."""
        with self.lock:
            self.sm.reset()
            self.last_sent = None

    def snap_now(self, x, y, t=None):
        """Put the cursor exactly on (x, y) right now (call just before clicking)."""
        with self.lock:
            self.sm.snap(x, y, t)
            self.last_sent = (round(x), round(y))
        self.move(*self.last_sent)

    # ---- output side
    def tick(self, now=None):
        """One output step. Returns the point sent, or None."""
        with self.lock:
            p = self.sm.sample(self.clock() if now is None else now)
            if p is None:
                return None
            xy = (round(p[0]), round(p[1]))
            if xy == self.last_sent:
                return None
            self.last_sent = xy
        self.move(*xy)
        return xy

    def start(self):
        self.stop_flag.clear()
        self.thread = threading.Thread(target=self._run, daemon=True)
        self.thread.start()

    def _run(self):
        period = 1.0 / self.hz
        while not self.stop_flag.is_set():
            t0 = self.clock()
            try:
                self.tick(t0)
            except Exception as e:   # never let an output hiccup kill the thread silently
                print(f"cursor output error: {e}")
            time.sleep(max(0.0, period - (self.clock() - t0)))

    def stop(self):
        self.stop_flag.set()
        if self.thread:
            self.thread.join(timeout=1.0)
