"""
find_camera.py -- which camera number sees the court markers (author, 13 Nov)
==============================================================================
After unplugging the camera (or another app using it) the court camera can get
a different number. For cameras 0-3: picture size, brightness, whether frames
change, and which ArUco markers (DICT_4X4_50: 3 raider, 0-2 defenders) it sees.
Then use --camera N with raid_runner.py / check_positions.py.

    python3 physical_experiment/rig/find_camera.py
"""
import time

import cv2
import cv2.aruco as aruco
import numpy as np

det = aruco.ArucoDetector(aruco.getPredefinedDictionary(aruco.DICT_4X4_50), aruco.DetectorParameters())
for i in range(4):
    cap = cv2.VideoCapture(i)
    if not cap.isOpened():
        print(f"camera {i}: not available")
        continue
    t0, frames, seen, last = time.time(), 0, set(), None
    changing = False
    while time.time() - t0 < 3.0:
        ok, f = cap.read()
        if not ok:
            continue
        frames += 1
        if last is not None and np.abs(f.astype(int) - last.astype(int)).mean() > 0.5:
            changing = True
        last = f
        _, ids, _ = det.detectMarkers(cv2.cvtColor(f, cv2.COLOR_BGR2GRAY))
        if ids is not None:
            seen |= set(int(v) for v in ids.ravel())
    cap.release()
    if last is None:
        print(f"camera {i}: opens but gives no pictures (frozen or busy)")
    else:
        print(f"camera {i}: {last.shape[1]}x{last.shape[0]}, {frames} frames in 3 s, "
              f"brightness {last.mean():.0f}, picture {'changing' if changing else 'NOT changing'}, "
              f"markers seen {sorted(seen) or 'none'}")
