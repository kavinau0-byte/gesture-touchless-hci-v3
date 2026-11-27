"""Measure static-gesture accuracy.

  collect : records labelled landmarks from your webcam to a CSV
            python tools/evaluate_accuracy.py collect --out data.csv --seconds 8
  report  : per-gesture precision / recall and a confusion matrix from a CSV
            python tools/evaluate_accuracy.py report --csv data.csv

CSV columns: label, x0,y0, x1,y1, ... x20,y20  (image-normalised MediaPipe landmarks).
"""
import argparse
import csv
import os
import sys
import time
from collections import Counter

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from gesture_hci.geometry import classify_static  # noqa: E402

LABELS = ["OPEN_PALM", "FIST", "THUMB_UP", "THUMB_DOWN", "PEACE", "POINT"]


def collect(args):
    import cv2
    import mediapipe as mp

    cap = cv2.VideoCapture(args.cam)
    hands = mp.solutions.hands.Hands(max_num_hands=1, model_complexity=0,
                                     min_detection_confidence=0.7, min_tracking_confidence=0.6)
    rows = []
    for label in LABELS:
        end_prep = time.time() + 4
        while time.time() < end_prep:   # get ready countdown
            ok, f = cap.read()
            f = cv2.flip(f, 1)
            cv2.putText(f, f"Get ready: {label}  ({end_prep - time.time():.0f}s)", (10, 40),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 255), 2)
            cv2.imshow("collect (q to abort)", f)
            if cv2.waitKey(1) & 0xFF == ord("q"):
                return
        end = time.time() + args.seconds
        n = 0
        while time.time() < end:
            ok, f = cap.read()
            f = cv2.flip(f, 1)
            res = hands.process(cv2.cvtColor(f, cv2.COLOR_BGR2RGB))
            if res.multi_hand_landmarks:
                lm = res.multi_hand_landmarks[0].landmark
                rows.append([label] + [v for l in lm for v in (l.x, l.y)])
                n += 1
            cv2.putText(f, f"Recording {label}: {n} frames", (10, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)
            cv2.imshow("collect (q to abort)", f)
            if cv2.waitKey(1) & 0xFF == ord("q"):
                return
    cap.release()
    cv2.destroyAllWindows()
    with open(args.out, "w", newline="") as fh:
        csv.writer(fh).writerows(rows)
    print(f"Saved {len(rows)} labelled frames to {args.out}")


def load(path):
    with open(path) as fh:
        for row in csv.reader(fh):
            v = list(map(float, row[1:]))
            yield row[0], list(zip(v[0::2], v[1::2]))


def report(args):
    labels = LABELS + ["NONE"]
    conf = {l: Counter() for l in LABELS}
    for label, pts in load(args.csv):
        conf[label][classify_static(pts)] += 1
    total = sum(sum(c.values()) for c in conf.values())
    if not total:
        sys.exit("CSV has no rows")
    print(f"{'':12}" + "".join(f"{l[:9]:>10}" for l in labels))
    for l in LABELS:
        print(f"{l:12}" + "".join(f"{conf[l][p]:>10}" for p in labels))
    print(f"\n{'gesture':12}{'frames':>8}{'recall':>9}{'precision':>11}")
    correct = 0
    for l in LABELS:
        n = sum(conf[l].values())
        tp = conf[l][l]
        correct += tp
        predicted = sum(conf[t][l] for t in LABELS)
        rec = tp / n if n else float("nan")
        prec = tp / predicted if predicted else float("nan")
        print(f"{l:12}{n:>8}{rec:>9.1%}{prec:>11.1%}")
    print(f"\nOverall per-frame accuracy: {correct / total:.1%} over {total} frames")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)
    c = sub.add_parser("collect")
    c.add_argument("--out", default="gesture_data.csv")
    c.add_argument("--cam", type=int, default=0)
    c.add_argument("--seconds", type=float, default=8)
    r = sub.add_parser("report")
    r.add_argument("--csv", required=True)
    a = ap.parse_args()
    {"collect": collect, "report": report}[a.cmd](a)
