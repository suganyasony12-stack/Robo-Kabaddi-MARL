#!/usr/bin/env python3
"""
Copied 4 Nov 2026 (author) from the Paper 2 rig (hardware/calibrate_p2.py) for the
Robo-Kabaddi robot runs; the original is unchanged. Changed here: the drawn court
lines and the top-down check follow the court size in camera_model.py (2.6 x 4.0 m).

calibrate_rig.py -- court calibration WITH fisheye/lens correction (camera_model.py).
Saves physical_experiment/rig/camera_calib.json (physical_experiment/camera_calib.json,
the old 4-corner calibration, is never touched).

Why: the overhead camera has a fisheye lens, so the court lines are curved in the
image. A 4-corner calibration is exact at the corners only (2026-10-26: ~0.2 m off
at the middle of the long sides). Clicking points along the court lines lets the
lens bending be measured and removed.

Steps (click on the camera window; points need not be exact, a few pixels is fine):
  1. the 4 COURT CORNERS, in order: top-left = defender end, raider's left;
     top-right = defender end, raider's right; bottom-right = raider end, raider's
     right; bottom-left = raider end, raider's left (same as before)
  2. then 5-8 points ALONG each line, spread over its whole length, including the
     middle (where the bending is largest) -- the SAME lines that meet at the corners:
       defender end line, raider end line, left side line (raider's left),
       right side line, mid-line
Keys: SPACE/ENTER next step (and accept at the end) | u undo last point | r start over | q quit

  python3 calibrate_rig.py --camera-height 2.60 --marker-height 0.21
  python3 calibrate_rig.py --set-heights --camera-height 2.60 --marker-height 0.21   # no re-clicking

--camera-height: camera lens above the floor (m); --marker-height: top of the
markers above the floor (m). Without them positions are not height-corrected.
After ANY camera move: recalibrate, then re-measure the direction tables.
"""
import argparse
import json
import sys
from pathlib import Path

import numpy as np
import cv2
import cv2.aruco as aruco

import camera_model as CM

HERE = Path(__file__).resolve().parent
OUT = HERE / "camera_calib.json"
STEPS = ["corners"] + list(CM.LINES)
CORNER_NAMES = ["TOP-LEFT (defender end, raider's left)", "TOP-RIGHT (defender end, raider's right)",
                "BOTTOM-RIGHT (raider end, raider's right)", "BOTTOM-LEFT (raider end, raider's left)"]
COLS = {"corners": (0, 255, 255), "defender end line": (255, 120, 0), "raider end line": (0, 140, 255),
        "left side line": (255, 0, 255), "right side line": (0, 255, 0), "mid-line": (255, 255, 255)}
ARUCO_ID_MAP = {0: "defender_1", 1: "defender_2", 2: "defender_3", 3: "raider_1"}
DISP_W, DISP_H = 1600, 900


def set_heights(cam_h, mk_h):
    d = json.load(open(OUT))
    d["heights"] = {"camera_m": cam_h, "marker_m": mk_h}
    json.dump(d, open(OUT, "w"), indent=1)
    print(f"[calibrate_rig] heights set in {OUT}: camera {cam_h} m, marker {mk_h} m")
    print("  " + CM.CameraModel.load(OUT).describe())


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--camera", type=int, default=0)
    ap.add_argument("--camera-height", type=float, default=None)
    ap.add_argument("--marker-height", type=float, default=None)
    ap.add_argument("--set-heights", action="store_true", help="only update the heights in the saved file")
    a = ap.parse_args()
    if a.set_heights:
        return set_heights(a.camera_height, a.marker_height)

    cap = cv2.VideoCapture(a.camera)
    if not cap.isOpened():
        sys.exit(f"camera {a.camera} will not open")
    ok, frame = cap.read()
    for _ in range(10):
        ok, frame = cap.read()
    h, w = frame.shape[:2]
    scale = min(DISP_W / w, DISP_H / h, 1.0)
    clicks = {s: [] for s in STEPS}
    state = {"i": 0}

    def on_mouse(ev, x, y, *_):
        if ev == cv2.EVENT_LBUTTONDOWN and state["i"] < len(STEPS):
            s = STEPS[state["i"]]
            if s == "corners" and len(clicks[s]) >= 4:
                return
            clicks[s].append((x / scale, y / scale))
            lab = CORNER_NAMES[len(clicks[s]) - 1] if s == "corners" else s
            print(f"  {s}: point {len(clicks[s])} ({x / scale:.0f}, {y / scale:.0f})  {lab if s == 'corners' else ''}")

    WIN = "Calibration - click on the camera image"
    cv2.namedWindow(WIN, cv2.WINDOW_NORMAL)
    cv2.setMouseCallback(WIN, on_mouse)
    print(__doc__.split("Keys:")[0])
    print(f"  STEP 1: click the 4 corners. {CORNER_NAMES[0]} first.")
    model = info = None
    while True:
        ok, frame = cap.read()
        if not ok:
            continue
        disp = cv2.resize(frame, None, fx=scale, fy=scale)
        for s, pts in clicks.items():
            for j, (x, y) in enumerate(pts):
                cv2.circle(disp, (int(x * scale), int(y * scale)), 6, COLS[s], 2)
                if s == "corners":
                    cv2.putText(disp, str(j + 1), (int(x * scale) + 8, int(y * scale) - 8),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.8, COLS[s], 2)
        if state["i"] < len(STEPS):
            s = STEPS[state["i"]]
            msg = (f"Click corner {len(clicks[s]) + 1}: {CORNER_NAMES[len(clicks[s])]}" if s == "corners" and len(clicks[s]) < 4
                   else "4 corners done - SPACE for next step" if s == "corners"
                   else f"Click 5-8 points along the {s.upper()} (whole length) - {len(clicks[s])} so far - SPACE when done")
        else:
            msg = "Fitted: green = calibrated court lines. SPACE = save, r = start over, q = quit"
            W, L, M = CM.COURT_W, CM.COURT_L, CM.MID
            for (p, q) in [((0, 0), (W, 0)), ((W, 0), (W, L)), ((W, L), (0, L)), ((0, L), (0, 0)), ((0, M), (W, M))]:
                pts = (model.court_polyline(p, q) * scale).astype(np.int32)
                cv2.polylines(disp, [pts], False, (0, 255, 0), 2)
        cv2.rectangle(disp, (0, 0), (disp.shape[1], 40), (0, 0, 0), -1)
        cv2.putText(disp, msg, (10, 28), cv2.FONT_HERSHEY_SIMPLEX, 0.75, (255, 255, 255), 2)
        cv2.imshow(WIN, disp)
        key = cv2.waitKey(30) & 0xFF
        if key == ord("q"):
            print("  quit - nothing saved"); break
        if key == ord("r"):
            clicks = {s: [] for s in STEPS}; state["i"] = 0; model = None
            try:
                cv2.destroyWindow("Top-down check")
            except cv2.error:
                pass
            print("\n  starting over: click the 4 corners"); continue
        if key == ord("u") and state["i"] < len(STEPS):
            s = STEPS[state["i"]]
            if clicks[s]:
                clicks[s].pop(); print(f"  {s}: removed last point")
        if key in (32, 13, 10):
            if state["i"] < len(STEPS):
                s = STEPS[state["i"]]
                need = 4 if s == "corners" else 3
                if len(clicks[s]) < need:
                    print(f"  {s}: need at least {need} points"); continue
                state["i"] += 1
                if state["i"] < len(STEPS):
                    print(f"\n  NEXT: click 5-8 points along the {STEPS[state['i']].upper()}, then SPACE")
                else:
                    corners = np.array(clicks["corners"])
                    lines = {k: np.array(v) for k, v in clicks.items() if k != "corners"}
                    model, info = CM.fit(corners, lines, (w, h))
                    model = CM.CameraModel(model.H, model.k, model.cx, model.cy, a.camera_height, a.marker_height)
                    print(f"\n  FIT: lens k = {info['k']:.3e}, centre ({info['cx']:.0f}, {info['cy']:.0f})")
                    print(f"  points off their court line: without lens correction rms {100 * info['rms_before_m']:.1f} cm "
                          f"(max {100 * info['max_before_m']:.1f}) -> with correction rms {100 * info['rms_after_m']:.1f} cm "
                          f"(max {100 * info['max_after_m']:.1f})")
                    if info["max_after_m"] > 0.03:
                        print("  ! some points are >3 cm off their line: a mis-click or the wrong line? (r = start over)")
                    tw, th = int(200 * CM.COURT_W), int(200 * CM.COURT_L)       # 200 px per metre
                    mx, my = model.topdown_maps(200, tw, th)
                    td = cv2.remap(frame, mx, my, cv2.INTER_LINEAR)
                    cv2.line(td, (0, th // 2), (tw, th // 2), (0, 255, 0), 1)   # mid-line
                    cv2.imshow("Top-down check", td)
                    g = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
                    det = aruco.ArucoDetector(aruco.getPredefinedDictionary(aruco.DICT_4X4_50), aruco.DetectorParameters())
                    cs, ids, _ = det.detectMarkers(g)
                    if ids is not None:
                        print("  robots now:  " + "  ".join(
                            f"{ARUCO_ID_MAP.get(int(m), m)} x {model.pix_to_court(*c[0].mean(0))[0]:.2f} "
                            f"y {model.pix_to_court(*c[0].mean(0))[1]:.2f}" for c, m in zip(cs, ids.flatten())))
                    print("  Top-down check: the real court lines should be STRAIGHT and on the window edges,"
                          " mid-line on the green line.  SPACE = save, r = start over.")
            elif model is not None:
                CM.save(OUT, model, info, np.array(clicks["corners"]),
                        {k: np.array(v) for k, v in clicks.items() if k != "corners"},
                        a.camera, a.camera_height, a.marker_height)
                print(f"\n[SAVED] -> {OUT}\n  {model.describe()}")
                break
    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
