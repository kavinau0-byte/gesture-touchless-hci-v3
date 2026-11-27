import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


def make_hand(gesture, dx=0.0, dy=0.0, pinch=False):
    """Synthetic 21-landmark hand (image coords, y grows downward) for a given gesture."""
    P = [(0.0, 0.0)] * 21
    P[0] = (0.5, 0.9)                                    # wrist
    fingers_up = {"OPEN_PALM": (1, 1, 1, 1), "FIST": (0, 0, 0, 0), "THUMB_UP": (0, 0, 0, 0),
                  "THUMB_DOWN": (0, 0, 0, 0), "PEACE": (1, 1, 0, 0), "POINT": (1, 0, 0, 0)}[gesture]
    for k, up in enumerate(fingers_up):                 # index, middle, ring, pinky
        x = 0.42 + 0.04 * k
        base = 5 + 4 * k
        P[base] = (x, 0.65)                              # MCP
        P[base + 1] = (x, 0.60)                          # PIP
        P[base + 2] = (x, 0.52 if up else 0.68)
        P[base + 3] = (x, 0.45 if up else 0.75)          # TIP
    P[1], P[2] = (0.47, 0.85), (0.45, 0.80)
    if gesture == "THUMB_UP":
        P[3], P[4] = (0.40, 0.65), (0.38, 0.50)
    elif gesture == "THUMB_DOWN":
        P[3], P[4] = (0.40, 0.95), (0.38, 1.10)
    elif gesture in ("OPEN_PALM", "PEACE"):
        P[3], P[4] = (0.40, 0.75), (0.33, 0.70)          # thumb out to the side
    elif gesture == "POINT":
        P[3], P[4] = (0.40, 0.75), (0.50, 0.70)          # thumb tucked
    if gesture == "FIST":
        P[3], P[4] = (0.47, 0.74), (0.50, 0.70)          # thumb folded across palm
    if pinch:
        P[4] = (P[8][0] + 0.005, P[8][1] + 0.005)        # thumb tip meets index tip
    return [(x + dx, y + dy) for x, y in P]
