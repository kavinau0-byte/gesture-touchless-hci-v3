"""Camera loop, MediaPipe tracking and on-screen overlay."""
import argparse
import os
import time
from collections import deque

from .controller import MODES, Controller
from .kiosk import KioskController
from .output import CursorOutput

WINDOW_NAME = "Gesture HCI  (q to quit)"


def parse_args(argv=None):
    ap = argparse.ArgumentParser(description="Touchless gesture controller")
    ap.add_argument("--dry", action="store_true", help="do not send keystrokes or mouse input")
    ap.add_argument("--cam", type=int, default=0, help="camera index")
    ap.add_argument("--mode", choices=list(MODES) + ["kiosk"], default="media")
    ap.add_argument("--cutoff", type=float, default=3.0, help="kiosk: lower = steadier but laggier (try 1.5-6)")
    ap.add_argument("--beta", type=float, default=15.0, help="kiosk: higher = less lag on fast moves (try 8-30)")
    ap.add_argument("--click", type=float, default=0.25, help="kiosk: pinch ratio that clicks; lower = tighter pinch")
    ap.add_argument("--fast-cursor", action="store_true",
                    help="kiosk: move the cursor from a 120 Hz thread with smoothing (smoother, experimental)")
    ap.add_argument("--record", metavar="CSV", help="kiosk: save raw fingertip positions (for tools/cursor_bench.py replay)")
    ap.add_argument("--fps", type=int, default=30, help="requested camera fps")
    ap.add_argument("--width", type=int, default=640, help="requested capture width")
    ap.add_argument("--height", type=int, default=480, help="requested capture height")
    return ap.parse_args(argv)


def make_mover(pg):
    """Fast cursor mover: Win32 SetCursorPos on Windows (pyautogui import already made the process DPI-aware)."""
    if os.name == "nt":
        import ctypes
        set_pos = ctypes.windll.user32.SetCursorPos
        return lambda x, y: set_pos(int(x), int(y))
    return lambda x, y: pg.moveTo(x, y, _pause=False)


def open_camera(cv2, index, fps=30, width=640, height=480):
    cap = cv2.VideoCapture(index, cv2.CAP_DSHOW) if os.name == "nt" else cv2.VideoCapture(index)
    cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)   # always process the newest frame
    cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*"MJPG"))
    cap.set(cv2.CAP_PROP_FPS, fps)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, width)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, height)
    return cap


def main(argv=None):
    args = parse_args(argv)

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
        kc = KioskController(sw, sh, cutoff=args.cutoff, beta=args.beta, click=args.click)
        out = None
        if args.fast_cursor and pg is not None:
            out = CursorOutput(make_mover(pg))
            out.start()
        rec = open(args.record, "w", newline="") if args.record else None
        if rec:
            rec.write("t,x,y\n")

    cap = open_camera(cv2, args.cam, args.fps, args.width, args.height)
    if not cap.isOpened():
        raise SystemExit(f"Could not open camera {args.cam}. Try --cam 1.")
    print(f"Camera request {args.fps} fps {args.width}x{args.height}; reports "
          f"{cap.get(cv2.CAP_PROP_FPS):.1f} fps at "
          f"{cap.get(cv2.CAP_PROP_FRAME_WIDTH):.0f}x{cap.get(cv2.CAP_PROP_FRAME_HEIGHT):.0f}")
    hands = mp.solutions.hands.Hands(
        max_num_hands=1, model_complexity=0,
        min_detection_confidence=0.7, min_tracking_confidence=0.6,
    )
    draw = mp.solutions.drawing_utils

    ctl = Controller("media" if kiosk else args.mode)
    last_action, last_action_t = "", -10.0
    prev_t, fps = time.perf_counter(), 0.0
    frame_times = deque()
    cv2.namedWindow(WINDOW_NAME, cv2.WINDOW_NORMAL)

    while cap.isOpened():
        ok, frame = cap.read()
        if not ok:
            break
        frame_t = time.perf_counter()
        frame_times.append(frame_t)
        while frame_times and frame_t - frame_times[0] > 1.0:
            frame_times.popleft()
        cam_fps = ((len(frame_times) - 1) / (frame_t - frame_times[0])
                   if len(frame_times) > 1 else 0.0)
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
                if rec:
                    rec.write(f"{now:.4f},{pts[8][0]:.5f},{pts[8][1]:.5f}\n")
                pos, click, gesture = kc.update(now, pts, wpts)
                if out:
                    if pos:
                        out.feed(now, *pos)
                    else:
                        out.rest()
                elif pos and pg:
                    pg.moveTo(pos[0], pos[1], _pause=False)
                if click:
                    if out:
                        out.snap_now(*pos, t=now)   # land exactly on the aim point
                    if pg:
                        pg.click()
                    last_action, last_action_t = "Click", now
                    print(f"[{time.strftime('%H:%M:%S')}] CLICK at {pos} "
                          f"(pinch {kc.ratio:.3f})")
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
                if out:
                    out.rest()

        latency_ms = (time.perf_counter() - t_start) * 1000
        fps = 0.9 * fps + 0.1 / max(now - prev_t, 1e-6)
        prev_t = now

        # ---- overlay
        w = frame.shape[1]
        cv2.rectangle(frame, (0, 0), (w, 120 if kiosk else 92), (30, 30, 30), -1)
        cv2.putText(frame, f"Gesture: {gesture}", (10, 26), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
        if now - last_action_t < 1.5:
            cv2.putText(frame, f"Action: {last_action}", (10, 56), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
        cv2.putText(frame, f"MODE: {'KIOSK' if kiosk else ctl.mode.upper()} | {fps:3.0f} FPS | {latency_ms:4.1f} ms"
                    + (" | DRY RUN" if args.dry else ""),
                    (10, 82), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200, 200, 200), 1)
        if kiosk:
            ratio_text = "--" if kc.ratio is None else f"{kc.ratio:.2f}"
            cv2.putText(frame, f"FAST-CURSOR {'ON' if out else 'OFF'} | pinch {ratio_text} "
                        f"(click<{kc.CLICK:.2f}) | cam {cam_fps:4.1f} fps",
                        (10, 108), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200, 200, 200), 1)
        cv2.rectangle(frame, (w - 130, 12), (w - 10, 30), (90, 90, 90), 1)
        cv2.rectangle(frame, (w - 130, 12), (w - 130 + int(120 * progress), 30), (0, 200, 255), -1)

        cv2.imshow(WINDOW_NAME, frame)
        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

    cap.release()
    cv2.destroyAllWindows()
    if kiosk:
        if out:
            out.stop()
        if rec:
            rec.close()
