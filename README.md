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
| `--click` (default 0.20) | Lower = pinch must be tighter to click |

Good lighting on your hand, a charger plugged in, and a steady forearm help more than any setting.

## Files

| File | Purpose |
|---|---|
| `gesture_control.py` | Camera capture, MediaPipe tracking, gesture logic, modes, overlay |
| `kiosk.html` | Touchless ordering kiosk demo page |
| `requirements.txt` | Python dependencies |

## Limitations

- One hand at a time
- Tracking gets worse in poor light, with cluttered backgrounds, or when a finger points straight at the camera
- Gesture recognition is rule-based; a trained classifier (e.g. SVM) is a possible extension
- Accuracy has not been formally measured yet

## Tech

Python, OpenCV, MediaPipe Hands, PyAutoGUI, HTML/JavaScript (demo page)
