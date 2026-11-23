#!/usr/bin/env python3
"""
b3_drive.py -- B2/B3 speed trials on the 2.6 x 4.0 m court (Bluetooth robots,
fisheye camera). CORRECTIONS_TRACKER.md B3; protocol in ../phase_b/PROTOCOLS.md.

One key press = one trial: the robot under test drives at 100% with one command
until it has gone MAX_DIST_M, comes within EDGE_STOP_M of a court line, MAX_TRIAL_S
pass, or you press SPACE; then it stops. Every camera frame of the trial is logged.

  python3 physical_experiment/rig/b3_drive.py --robot raider_1
  python3 physical_experiment/rig/b3_drive.py --robot defender_2

Keys (in the camera window):
  f  forward 100%            b  backward 100%
  r  diagonal forward-right  l  diagonal forward-left
  SPACE  stop now (the trial is kept, marked "stopped by hand")
  q  quit

Before each trial put the robot so it has about 2 m free in the direction of travel
(the trial stops itself near a line). Run forward trials too: B3 is diagonal speed
divided by forward speed of the same robot.

Writes (same columns as log_fsr_proximity.py, so analyse_motion.py reads them):
  ../phase_b/data/session_<time>/trials.csv     one row per frame of each trial
  ../phase_b/trials_sheet.csv                   one row per trial (appended)
Then:
  python3 physical_experiment/phase_b/analyse_motion.py \\
      --log physical_experiment/phase_b/data/session_*/trials.csv \\
      --sheet physical_experiment/phase_b/trials_sheet.csv

--fake: no camera and no robots; a made-up robot (forward 0.30 m/s, diagonal
0.21 m/s) runs 2 trials of each command, to test the logging and the analysis.
It needs --outdir (never the real data folder).
"""
from __future__ import annotations

import argparse
import csv
import datetime
import math
import sys
import time
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import camera_model as CM                                         # noqa: E402

PHASE_B = HERE.parent / "phase_b"
CALIB = HERE / "camera_calib.json"

MAX_DIST_M = 2.0       # trial length (PROTOCOLS.md: about 2 m)
EDGE_STOP_M = 0.30     # stop when the robot centre is this close to a court line
MAX_TRIAL_S = 10.0
SETTLE_S = 1.0         # frames still logged after the stop
SEND_EVERY_S = 0.10    # command repeated while driving (BLE keeps only the newest)
LOST_STOP_S = 1.0      # marker not seen this long while driving -> stop

COMMANDS = {           # key -> (command name, 100% character, test)
    "f": ("forward", "[", "b2_speed"),
    "b": ("backward", "_", "b2_speed"),
    "r": ("diag_fwd_right", "o", "b3_diag"),
    "l": ("diag_fwd_left", "k", "b3_diag"),
}
STOP = "}"
MARKER_OF = {"raider_1": 3, "defender_1": 0, "defender_2": 1, "defender_3": 2}
CSV_COLUMNS = ["t_mono", "raider_x", "raider_y", "raider_yaw",
               "defender_x", "defender_y", "recording", "trial_id"]
SHEET_COLUMNS = ["session", "trial_id", "test", "robot", "tracked", "command",
                 "level_pct", "angle_deg", "notes"]


def edge_distance(x, y):
    return min(x, CM.COURT_W - x, y, CM.COURT_L - y)


# ------------------------------------------------------------------ sources ---
class CameraTracker:
    """Marker position (court metres, floor under the marker) and yaw (degrees)."""

    def __init__(self, camera, marker):
        import cv2
        import cv2.aruco as aruco
        self.cv2, self.marker = cv2, marker
        self.model = CM.CameraModel.load(CALIB)
        print("[camera] " + self.model.describe())
        self.cap = cv2.VideoCapture(camera)
        if not self.cap.isOpened():
            sys.exit(f"camera {camera} will not open")
        self.det = aruco.ArucoDetector(aruco.getPredefinedDictionary(aruco.DICT_4X4_50),
                                       aruco.DetectorParameters())
        self.scale = 150                                              # px per metre, top-down view
        self.tw, self.th = int(self.scale * CM.COURT_W), int(self.scale * CM.COURT_L)
        self.maps = self.model.topdown_maps(self.scale, self.tw, self.th)
        self.frame = None

    def read(self):
        ok, frame = self.cap.read()
        if not ok:
            return None
        self.frame = frame
        cs, ids, _ = self.det.detectMarkers(self.cv2.cvtColor(frame, self.cv2.COLOR_BGR2GRAY))
        if ids is None:
            return None
        for c, m in zip(cs, ids.flatten()):
            if int(m) == self.marker:
                p = c[0]
                x, y = self.model.pix_to_court(*p.mean(0))
                a = self.model.pix_to_court(*p[0]); b = self.model.pix_to_court(*p[1])
                return x, y, math.degrees(math.atan2(b[1] - a[1], b[0] - a[0]))
        return None

    def show(self, pose, lines):
        cv2 = self.cv2
        if self.frame is None:
            return -1
        td = cv2.remap(self.frame, *self.maps, cv2.INTER_LINEAR)
        cv2.line(td, (0, self.th // 2), (self.tw, self.th // 2), (0, 255, 0), 1)
        if pose is not None:
            px, py = int(pose[0] * self.scale), int(self.th - pose[1] * self.scale)
            cv2.circle(td, (px, py), 10, (0, 0, 255), 2)
        panel = np.zeros((self.th, 420, 3), np.uint8)
        for i, s in enumerate(lines):
            cv2.putText(panel, s, (8, 26 + 24 * i), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 255), 1)
        cv2.imshow("B2/B3 speed trials", np.hstack([panel, td]))
        return cv2.waitKey(1) & 0xFF

    def close(self):
        self.cap.release()
        self.cv2.destroyAllWindows()


class BleRobot:
    def __init__(self, rid):
        import ble_link
        self.L, self.rid = ble_link, rid
        ble_link.start_listeners([rid])
        if not ble_link.wait_for_all_connections([rid], timeout=60.0):
            sys.exit(f"{rid} did not connect over Bluetooth")

    def send(self, ch):
        self.L.send_command(self.rid, ch)

    def close(self):
        self.L.shutdown()


class FakeRobot:
    """Made-up robot for --fake: start (1.3, 0.5) facing +y, 0.30 m/s straight,
    0.21 m/s diagonal (ratio 0.70), 0.3 s to reach speed, 3 mm camera noise."""
    SPEED = {"[": (0.0, 0.30), "_": (0.0, -0.30), "o": (0.1485, 0.1485), "k": (-0.1485, 0.1485)}

    def __init__(self, seed=0):
        self.rng = np.random.default_rng(seed)
        self.p, self.v, self.cmd, self.t = np.array([1.3, 0.5]), np.zeros(2), STOP, None

    def place(self, cmd):
        self.p = np.array([1.3, 3.5]) if cmd == "_" else np.array([0.6 if cmd == "o" else 2.0 if cmd == "k" else 1.3, 0.5])

    def send(self, ch):
        self.cmd = ch

    def read(self, now):
        if self.t is not None:
            dt = now - self.t
            target = np.array(self.SPEED.get(self.cmd, (0.0, 0.0)))
            self.v += (target - self.v) * min(1.0, dt / 0.3)
            self.p = self.p + self.v * dt
        self.t = now
        n = self.rng.normal(0, 0.003, 2)
        return float(self.p[0] + n[0]), float(self.p[1] + n[1]), 90.0 + float(self.rng.normal(0, 0.5))

    def close(self):
        pass


# -------------------------------------------------------------------- trials ---
def run_trial(key, tid, robot, read, writer, clock, show=None, sleep=None):
    """Drive one trial; returns the stop reason. read() -> (x, y, yaw) or None."""
    name, ch, _ = COMMANDS[key]
    t0 = clock(); last_send = -1.0; start = None; last_seen = t0; reason = None; stop_t = None
    while True:
        now = clock()
        pose = read()
        if pose is not None:
            last_seen = now
            start = start or pose[:2]
            writer.writerow({"t_mono": f"{now:.4f}", "raider_x": f"{pose[0]:.4f}",
                             "raider_y": f"{pose[1]:.4f}", "raider_yaw": f"{pose[2]:.2f}",
                             "defender_x": "", "defender_y": "", "recording": 1, "trial_id": tid})
        if stop_t is None:
            if pose is not None and math.hypot(pose[0] - start[0], pose[1] - start[1]) >= MAX_DIST_M:
                reason = f"{MAX_DIST_M} m done"
            elif pose is not None and now - t0 > 0.5 and edge_distance(*pose[:2]) < EDGE_STOP_M:
                reason = f"within {EDGE_STOP_M} m of a court line"
            elif now - t0 > MAX_TRIAL_S:
                reason = f"{MAX_TRIAL_S} s"
            elif now - last_seen > LOST_STOP_S:
                reason = "marker lost"
            if reason is None and now - last_send >= SEND_EVERY_S:
                robot.send(ch); last_send = now
            if show is not None:
                k = show(pose, [f"trial {tid}: {name} 100%", "SPACE = stop"])
                if k == ord(" "):
                    reason = "stopped by hand"
            if reason is not None:
                robot.send(STOP); stop_t = now
        else:
            if now - last_send >= SEND_EVERY_S:
                robot.send(STOP); last_send = now
            if show is not None:
                show(pose, [f"trial {tid}: stopping ({reason})"])
            if now - stop_t >= SETTLE_S:
                return reason
        if sleep is not None:
            sleep()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--robot", default="raider_1", choices=list(MARKER_OF))
    ap.add_argument("--marker", type=int, default=None, help="ArUco id (default: the robot's own)")
    ap.add_argument("--camera", type=int, default=0)
    ap.add_argument("--fake", action="store_true", help="no camera, no robots (test)")
    ap.add_argument("--outdir", default=None, help="data folder (default ../phase_b; needed with --fake)")
    a = ap.parse_args()
    if a.fake and a.outdir is None:
        sys.exit("--fake needs --outdir (fake trials never go into phase_b/)")
    base = Path(a.outdir) if a.outdir else PHASE_B
    session = "session_" + datetime.datetime.now().strftime("%Y-%m-%d_%H%M%S")
    sdir = base / "data" / session
    sdir.mkdir(parents=True, exist_ok=True)
    sheet = base / "trials_sheet.csv"
    new_sheet = not sheet.exists()
    fh = open(sdir / "trials.csv", "w", newline="")
    writer = csv.DictWriter(fh, fieldnames=CSV_COLUMNS); writer.writeheader()
    sh = open(sheet, "a", newline="")
    sw = csv.DictWriter(sh, fieldnames=SHEET_COLUMNS)
    if new_sheet:
        sw.writeheader()

    def log_sheet(tid, key, reason):
        name, _, test = COMMANDS[key]
        sw.writerow({"session": session, "trial_id": tid, "test": test, "robot": a.robot,
                     "tracked": "raider", "command": name, "level_pct": 100, "angle_deg": "",
                     "notes": f"b3_drive.py{' --fake' if a.fake else ''}; stop: {reason}"})
        sh.flush(); fh.flush()
        print(f"  trial {tid} {name}: {reason}")

    tid = 0
    if a.fake:
        sim = {"t": 0.0}
        clock = lambda: sim["t"]                                         # noqa: E731

        def tick():
            sim["t"] += 1 / 30
        robot = FakeRobot()
        for key in "ffrrll":
            tid += 1
            robot.place(COMMANDS[key][1])
            reason = run_trial(key, tid, robot, lambda: robot.read(sim["t"]), writer, clock, sleep=tick)
            log_sheet(tid, key, reason)
            sim["t"] += 2.0; robot.t = None
    else:
        marker = a.marker if a.marker is not None else MARKER_OF[a.robot]
        cam = CameraTracker(a.camera, marker)
        robot = BleRobot(a.robot)
        print(__doc__.split("Before each trial")[0].split("Keys")[1])
        try:
            while True:
                pose = cam.read()
                where = "marker not seen" if pose is None else \
                    f"x {pose[0]:.2f}  y {pose[1]:.2f}  yaw {pose[2]:.0f}"
                k = cam.show(pose, [f"{a.robot} (marker {marker})", where,
                                    "f fwd  b back  r diag-R  l diag-L", "q quit",
                                    f"trials so far: {tid}"])
                if k == ord("q"):
                    break
                if k in (ord(c) for c in COMMANDS):
                    key = chr(k)
                    if pose is None:
                        print("  marker not seen - trial not started"); continue
                    tid += 1
                    reason = run_trial(key, tid, robot, cam.read, writer, time.monotonic, cam.show)
                    log_sheet(tid, key, reason)
        finally:
            robot.send(STOP)
            robot.close(); cam.close()
    fh.close(); sh.close()
    print(f"\n{tid} trials -> {sdir / 'trials.csv'}\nsheet -> {sheet}")


if __name__ == "__main__":
    main()
