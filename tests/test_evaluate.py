import csv
import subprocess
import sys

from conftest import make_hand


def test_report_runs_on_csv(tmp_path):
    p = tmp_path / "d.csv"
    with open(p, "w", newline="") as fh:
        w = csv.writer(fh)
        for g in ["OPEN_PALM", "FIST", "THUMB_UP", "THUMB_DOWN", "PEACE", "POINT"]:
            for _ in range(3):
                w.writerow([g] + [v for pt in make_hand(g) for v in pt])
    out = subprocess.run([sys.executable, "tools/evaluate_accuracy.py", "report", "--csv", str(p)],
                         capture_output=True, text=True, check=True).stdout
    assert "Overall per-frame accuracy: 100.0%" in out
