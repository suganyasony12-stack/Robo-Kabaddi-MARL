"""
check_positions.py -- check the camera calibration at known places on the court
===============================================================================

Place one robot (its marker on top, as in the trials) centred over each spot
in turn, measured with a tape; press SPACE and the camera reading (median of
the last 15 frames) is compared with the spot. Uses rig/camera_calib.json.

Court coordinates (metres), as camera_model.py:
  x across the court, 0 = the side line on the raider's left, 2.6 = raider's right
  y along the court,  0 = the raider's end line, 2.0 = midline, 4.0 = defenders' end line

Keys (camera window): SPACE record this spot | s skip this spot | u redo the last spot | q quit
Every spot is written at once to physical_experiment/rig/position_check_<time>.csv.

    python3 physical_experiment/rig/check_positions.py            # marker 3 (raider)
    python3 physical_experiment/rig/check_positions.py --marker 0  # defender 1
"""
from __future__ import annotations

import argparse
import csv
import math
import sys
import time
from collections import deque
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import camera_model as CM                                             # noqa: E402

CALIB = HERE / "camera_calib.json"
W, L, MID = CM.COURT_W, CM.COURT_L, CM.MID
# (name, x, y): middle, defenders' half (where the raids are), near every line
SPOTS = [
    ("midline centre", W / 2, MID),
    ("defenders' half centre", W / 2, 3.00),
    ("near defenders' end line, centre", W / 2, 3.70),
    ("near defenders' end line, raider's left", 0.30, 3.70),
    ("near defenders' end line, raider's right", W - 0.30, 3.70),
    ("defenders' half, near left side line", 0.30, 2.50),
    ("defenders' half, near right side line", W - 0.30, 2.50),
    ("raider's half centre", W / 2, 1.00),
]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--camera", type=int, default=0)
    ap.add_argument("--marker", type=int, default=3, help="ArUco id: 3 raider, 0/1/2 defenders 1/2/3")
    a = ap.parse_args()
    import cv2
    import cv2.aruco as aruco
    model = CM.CameraModel.load(CALIB)
    print("[camera] " + model.describe())
    cap = cv2.VideoCapture(a.camera)
    if not cap.isOpened():
        sys.exit(f"camera {a.camera} will not open")
    det = aruco.ArucoDetector(aruco.getPredefinedDictionary(aruco.DICT_4X4_50),
                              aruco.DetectorParameters())
    scale = 150                                                       # px per metre, top-down
    tw, th = int(scale * W), int(scale * L)
    maps = model.topdown_maps(scale, tw, th)
    out = HERE / f"position_check_{time.strftime('%Y%m%d_%H%M%S')}.csv"
    rows, recent, i = [], deque(maxlen=15), 0
    WIN = "Position check"
    cv2.namedWindow(WIN, cv2.WINDOW_NORMAL)

    def write():
        with open(out, "w", newline="") as f:
            w = csv.writer(f)
            w.writerow(["spot", "x_true_m", "y_true_m", "x_cam_m", "y_cam_m", "error_cm", "marker"])
            w.writerows(rows)

    print(f"\nmarker {a.marker}. Place the robot centred over each spot, then SPACE.")
    print(f"  spot 1: {SPOTS[0][0]}  (x {SPOTS[0][1]:.2f}, y {SPOTS[0][2]:.2f})")
    while True:
        ok, frame = cap.read()
        if not ok:
            continue
        cs, ids, _ = det.detectMarkers(cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY))
        now = None
        if ids is not None:
            for c, m in zip(cs, ids.flatten()):
                if int(m) == a.marker:
                    now = model.pix_to_court(*c[0].mean(0))
        if now is not None:
            recent.append(now)
        else:
            recent.clear()
        td = cv2.remap(frame, *maps, cv2.INTER_LINEAR)
        cv2.line(td, (0, th // 2), (tw, th // 2), (0, 255, 0), 1)
        if i < len(SPOTS):
            _, sx, sy = SPOTS[i]
            cv2.drawMarker(td, (int(sx * scale), int(th - sy * scale)), (0, 255, 255),
                           cv2.MARKER_CROSS, 24, 2)
        if now is not None:
            cv2.circle(td, (int(now[0] * scale), int(th - now[1] * scale)), 10, (0, 0, 255), 2)
        msg = (f"spot {i + 1}/{len(SPOTS)}: {SPOTS[i][0]}" if i < len(SPOTS) else "done - q to quit")
        rd = (f"camera x {now[0]:.3f} y {now[1]:.3f}" if now is not None else "marker not seen")
        cv2.rectangle(td, (0, 0), (tw, 44), (0, 0, 0), -1)
        cv2.putText(td, msg[:48], (6, 18), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
        cv2.putText(td, rd, (6, 38), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
        cv2.imshow(WIN, td)
        key = cv2.waitKey(30) & 0xFF
        if key == ord("q"):
            break
        if key == ord("u") and rows:
            rows.pop(); i -= 1; write()
            print(f"  redo spot {i + 1}: {SPOTS[i][0]}")
        if i >= len(SPOTS):
            continue
        if key == ord("s"):
            i += 1
            if i < len(SPOTS):
                print(f"  skipped; spot {i + 1}: {SPOTS[i][0]}  (x {SPOTS[i][1]:.2f}, y {SPOTS[i][2]:.2f})")
        if key == 32:
            if len(recent) < 10:
                print("  marker not seen steadily - check the robot is in view"); continue
            name, sx, sy = SPOTS[i]
            cx, cy = np.median(np.array(recent), axis=0)
            err = 100 * math.hypot(cx - sx, cy - sy)
            rows.append([name, f"{sx:.3f}", f"{sy:.3f}", f"{cx:.3f}", f"{cy:.3f}", f"{err:.1f}", a.marker])
            write()
            print(f"  spot {i + 1} {name}: camera ({cx:.3f}, {cy:.3f})  off by {err:.1f} cm "
                  f"(x {100 * (cx - sx):+.1f}, y {100 * (cy - sy):+.1f})")
            i += 1
            if i < len(SPOTS):
                print(f"  spot {i + 1}: {SPOTS[i][0]}  (x {SPOTS[i][1]:.2f}, y {SPOTS[i][2]:.2f})")
            else:
                e = [float(r[5]) for r in rows]
                print(f"\n  {len(e)} spots: mean {np.mean(e):.1f} cm, max {max(e):.1f} cm  -> {out}")
    cap.release()
    cv2.destroyAllWindows()
    if rows:
        print(f"saved {len(rows)} spots -> {out}")


if __name__ == "__main__":
    main()
