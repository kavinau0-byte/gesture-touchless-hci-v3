# Gesture-Based Interface for Touchless Human-Computer Interaction

Control a computer with hand gestures and no physical contact, using only a standard webcam.
The camera feed goes to [MediaPipe Hands](https://developers.google.com/mediapipe), which predicts 21 hand landmarks per frame.
Our code turns those landmarks into gestures and sends the matching mouse or keyboard action with PyAutoGUI.

The system is general purpose. The demo use case is a **touchless self-order kiosk**: shared touchscreens are a hygiene
concern, so you point to move the cursor and pinch to click instead of touching the screen. Media-player and
slideshow control are also built in.

## Features

- Real-time single-hand tracking, with an on-screen overlay (gesture, action, FPS, per-frame processing time)
- Rule-based gesture recognition on landmark geometry (finger extension, pinch distance, palm motion)
- Three modes: **media**, **slides**, **kiosk**
- Touchless kiosk: index-finger cursor, pinch to click, adaptive smoothing, aim assist on the demo page
- Dry-run mode that shows the overlay without sending any input

## Gestures

| Gesture | Media mode | Slides mode | Kiosk mode |
|---|---|---|---|
| Open palm (hold) | Play / Pause | Black screen | Rest (cursor stops) |
| Fist (hold) | Mute | - | - |
| Thumb up / down | Volume + / - | Volume + / - | - |
| Peace sign (hold) | Fullscreen | Start slideshow (F5) | - |
| Swipe right / left (open palm) | Next / previous track | Next / previous slide | - |
| Index finger point (hold ~1 s) | Switch mode | Switch mode | - |
| Index finger point | - | - | Move cursor |
| Pinch thumb + index | - | - | Click |

One-shot gestures fire once per hold. Lower your hand or change gesture to re-arm.

## Setup

Python 3.10 or 3.11 is required (the pinned MediaPipe version does not install on 3.13).

```bash
python -m venv venv
venv\Scripts\activate          # Windows
pip install -r requirements.txt
```

## Run

```bash
python gesture_control.py --dry                # overlay only, sends nothing (use this to test)
python gesture_control.py                      # media mode, real keystrokes
python gesture_control.py --mode slides        # PowerPoint slideshow control
python gesture_control.py --mode kiosk         # touchless mouse for the kiosk page
python -m gesture_hci --dry                    # same thing; gesture_control.py is a thin wrapper
```

Press `q` in the camera window to quit. Keystrokes go to whichever window has focus, so click your
slideshow or video player first (media keys work regardless of focus).

### Kiosk demo

1. Open `kiosk.html` in a browser (F11 for fullscreen).
2. Run `python gesture_control.py --mode kiosk`.
3. Point to move the cursor, pinch to click, open palm to rest. In kiosk mode the screen-corner failsafe is off,
   so take your hand out of the camera view (or press `Ctrl+C`) to stop.

### Tuning (kiosk mode)

| Option | Effect |
|---|---|
| `--cutoff` (default 3.0) | Lower = steadier cursor but more lag |
| `--beta` (default 15) | Higher = less lag on fast movements |
| `--click` (default 0.25) | Lower = pinch must be tighter to click |

Good lighting on your hand, a charger plugged in, and a steady forearm help more than any setting.

## Project layout

| Path | Purpose |
|---|---|
| `gesture_hci/geometry.py` | Pure landmark geometry: finger state, gesture classification, swipe, stillness |
| `gesture_hci/controller.py` | Media / slides modes: vote smoothing, hold timing, swipe cooldown, mode switch |
| `gesture_hci/kiosk.py` | Kiosk cursor: One-Euro smoothing, pinch-to-click, aim freeze |
| `gesture_hci/app.py` | Webcam loop, MediaPipe tracking, overlay, CLI |
| `gesture_control.py` | Backwards-compatible entry point (`python gesture_control.py ...`) |
| `kiosk.html` | Touchless ordering kiosk demo page |
| `tests/` | Unit tests on synthetic hand landmarks (no camera needed) |
| `tools/evaluate_accuracy.py` | Record labelled gestures from your webcam and report accuracy |
| `.github/workflows/tests.yml` | Runs the tests on every push |

## Testing

```bash
pip install pytest
python -m pytest -q
```

The core logic never imports OpenCV or MediaPipe, so the tests run anywhere.

## Measuring accuracy

```bash
python tools/evaluate_accuracy.py collect --out data.csv --seconds 8   # hold each gesture when prompted
python tools/evaluate_accuracy.py report --csv data.csv                # confusion matrix, precision, recall
```

Collect data from several people and lighting conditions before quoting a number.

## Limitations

- One hand at a time
- Tracking gets worse in poor light, with cluttered backgrounds, or when a finger points straight at the camera
- Gesture recognition is rule-based; a trained classifier (e.g. SVM) is a possible extension
- Accuracy has not been formally measured yet; `tools/evaluate_accuracy.py` is provided to do it

## Tech

Python, OpenCV, MediaPipe Hands, PyAutoGUI, HTML/JavaScript (demo page)
