#!/usr/bin/env python3
"""
raid_runner.py -- raids on the real robots with the trained (or scripted) policies
=================================================================================

Author, 11 Nov 2026: a few real raids to see how the accepted pairs behave on
the robots (Bluetooth firmware, fisheye camera, 2.6 x 4.0 m court), before
deciding on retraining with the measured speeds (B3).

How it works: a "shadow" copy of the simulator (RoboKabaddiEnv, the same
CORRECTED_CONFIG + C3 rules as match.py) runs in real time. The robots are
not simulated: at every physics sub-step each robot is put where the camera
sees it, and the velocity the simulator would drive is sent to that robot
as its nearest command (8 directions, 25/50/75/100%), turned into the
robot's own frame with the yaw the camera measures. So the robots' inputs,
the contact rule (0.25 m, counted 0.20 m past the midline, 2 steps), held
(4 s still with a defender touching), out of court, defender fouls, Empty on
return and the 30 s timer are the simulator's own code; nothing in the
simulator is changed. Raid kinds as marl/match.py.

Raiders / defences: the accepted pairs (marl/match.py PAIRS) chosen by the
number of defenders, or the scripted raider / gate defenders. Raid i uses
seed SEED0 + i (default 50000, as the simulator tests), so each real raid
starts from the same places as simulated raid i and can be compared one to one.

Before each raid the runner drives each robot to its start place (circles in
the camera window, R = raider, 1-3 = defenders; within 10 cm, author 11 Nov),
then waits for SPACE to start (--manual-place: put them there by hand).
Raid time (author, 13 Nov): reported from the moment the raider crosses the
midline (the simulator's 30 s raid clock starts there too): raids.csv raid_t_s
and midline_t_s, steps.csv raid_t_s, the window and the printed result; t_s
stays the time from the start of the raid, as before.
--media (author, 13 Nov): every raid's camera video, tracked top-down video, snapshots
(start, touch, end) and a path picture (rig/raid_media.py); after each raid the window
shows the paths and asks y = save / n = discard (session/media/raid<NN>_seed<S>/).
--return-margin 0.03 (author, 13 Nov): Empty on return only 3 cm behind the midline.
--pairs 13nov: the 13 Nov pairs (match.py PAIRS_13NOV). --raider-rule deep: the learned
raider goes deep until the touch (defender_env.deep_rule). --def-start LO,HI (author, 13 Nov): defenders start LO-HI metres past the
midline (1.0,1.5 = 3.0-3.5 m from the raider's end line); default the back
row, as before. During the raid SPACE stops everything (raid kept as
"stopped"). A raid also stops if a marker is not seen for 1 s. q quits.

    python3 physical_experiment/rig/raid_runner.py --check                 # direction check, one robot at a time
    python3 physical_experiment/rig/raid_runner.py --raider learned --defence learned --n-def 3 --raids 3
    python3 physical_experiment/rig/raid_runner.py --raider learned --defence scripted --gate v8 --n-def 3 --raids 3
    python3 physical_experiment/rig/raid_runner.py ... --fake --outdir <scratch>   # no camera, no robots

Writes physical_experiment/match_data/session_<time>/: raids.csv (one row per
raid) and steps.csv (every 0.05 s: positions, commands sent, flags).
--fake: no camera and no robots; made-up robots move at the measured speeds
(straight 0.35 m/s, diagonal 0.58 of it) and pass through each other. Needs
--outdir (never the real data folder).
"""
from __future__ import annotations

import argparse
import csv
import math
import sys
import threading
import time
from collections import deque
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
sys.path[:0] = [str(HERE), str(ROOT / "marl"), str(ROOT)]
import camera_model as CM                                             # noqa: E402

CALIB = HERE / "camera_calib.json"
DATA = HERE.parent / "match_data"
MARKER = {"raider_1": 3, "defender_1": 0, "defender_2": 1, "defender_3": 2}
LOST_STOP_S = 1.0
OFFSET = np.array([CM.COURT_W / 2, CM.COURT_L / 2])   # court frame (camera) = sim frame + OFFSET

# Bluetooth firmware commands (smorphi_controller.py), speed index 0-3 = 25/50/75/100 %
CMD = {0: ["^", "]", "A", "["],          # forward
       45: ["n", "m", "l", "k"],         # diagonal forward-left
       90: ["f", "e", "d", "c"],         # left
       135: ["v", "u", "t", "s"],        # diagonal back-left
       180: ["b", "a", "`", "_"],        # backward
       -135: ["z", "y", "x", "w"],       # diagonal back-right
       -90: ["j", "i", "h", "g"],        # right
       -45: ["r", "q", "p", "o"]}        # diagonal forward-right
STOP = "}"
CHAR_DIR = {c: (a, i) for a, cs in CMD.items() for i, c in enumerate(cs)}


def robot_command(vx, vy, yaw_deg, v_full, diag_ratio):
    """Court-frame velocity (sim frame, m/s) -> firmware character for a robot
    whose camera yaw is yaw_deg. Forward of a robot at yaw 0 is +y (toward the
    defenders' end), its left is -x (B3 runs, 11 Nov)."""
    sp = math.hypot(vx, vy)
    if sp < 1e-6:
        return STOP
    diag = abs(vx) > 1e-6 and abs(vy) > 1e-6
    frac = sp / (v_full * (diag_ratio if diag else 1.0))
    level = int(np.clip(round(frac * 4) - 1, 0, 3))
    rel = math.degrees(math.atan2(vy, vx)) - (90.0 + yaw_deg)       # robot frame, left +
    a = int(round(((rel + 180) % 360 - 180) / 45.0)) * 45
    a = 180 if a == -180 else a                                      # straight back
    return CMD[a][level]


def char_velocity(c, yaw_deg, v_straight=0.35, ratio=0.58):
    """--fake: court velocity of a robot driven by character c (measured speeds)."""
    if c == STOP or c not in CHAR_DIR:
        return 0.0, 0.0
    a, level = CHAR_DIR[c]
    v = v_straight * (level + 1) / 4 * (ratio if a % 90 else 1.0)
    th = math.radians(90.0 + yaw_deg + a)
    return v * math.cos(th), v * math.sin(th)


# ------------------------------------------------------------------ sources ---
class Camera:
    """Camera thread: court position (metres, court frame), yaw and velocity of
    every marker, from the latest frames."""

    def __init__(self, index=0):
        import cv2
        import cv2.aruco as aruco
        self.cv2 = cv2
        self.model = CM.CameraModel.load(CALIB)
        print("[camera] " + self.model.describe())
        self.cap = cv2.VideoCapture(index)
        if not self.cap.isOpened():
            sys.exit(f"camera {index} will not open")
        self.det = aruco.ArucoDetector(aruco.getPredefinedDictionary(aruco.DICT_4X4_50),
                                       aruco.DetectorParameters())
        self.scale = 150
        self.tw, self.th = int(self.scale * CM.COURT_W), int(self.scale * CM.COURT_L)
        self.maps = self.model.topdown_maps(self.scale, self.tw, self.th)
        self.hist = {m: deque(maxlen=12) for m in MARKER.values()}
        self.frame, self.lock, self.run = None, threading.Lock(), True
        self.rec = None                       # --media: called with every frame
        threading.Thread(target=self._loop, daemon=True).start()

    def _loop(self):
        while self.run:
            ok, frame = self.cap.read()
            if not ok:
                continue
            t = time.monotonic()
            rec = self.rec
            if rec is not None:
                rec(frame, t)
            cs, ids, _ = self.det.detectMarkers(self.cv2.cvtColor(frame, self.cv2.COLOR_BGR2GRAY))
            with self.lock:
                self.frame = frame
                if ids is None:
                    continue
                for c, m in zip(cs, ids.flatten()):
                    if int(m) in self.hist:
                        q = c[0]
                        x, y = self.model.pix_to_court(*q.mean(0))
                        a = self.model.pix_to_court(*q[0]); b = self.model.pix_to_court(*q[1])
                        yaw = math.degrees(math.atan2(b[1] - a[1], b[0] - a[0]))
                        self.hist[int(m)].append((t, x, y, yaw))

    def get(self, marker):
        """(x, y, yaw_deg, vx, vy, age_s) in the court frame, or None."""
        with self.lock:
            h = list(self.hist[marker])
        if not h:
            return None
        t, x, y, yaw = h[-1]
        old = [s for s in h if t - s[0] <= 0.25]
        vx = vy = 0.0
        if len(old) >= 3 and old[-1][0] - old[0][0] > 0.05:
            ts = np.array([s[0] for s in old]); ts -= ts.mean()
            vx = float(np.polyfit(ts, [s[1] for s in old], 1)[0])
            vy = float(np.polyfit(ts, [s[2] for s in old], 1)[0])
        return x, y, yaw, vx, vy, time.monotonic() - t

    def topdown(self):
        with self.lock:
            f = None if self.frame is None else self.frame.copy()
        return None if f is None else self.cv2.remap(f, *self.maps, self.cv2.INTER_LINEAR)

    def close(self):
        self.run = False
        time.sleep(0.1)
        self.cap.release()


class Robots:
    """Bluetooth robots (rig/ble_link.py)."""

    def __init__(self, names):
        import ble_link
        self.ble, self.names = ble_link, names
        ble_link.start_listeners(names)
        if not ble_link.wait_for_all_connections(names, timeout=90.0):
            missing = [n for n in names if not ble_link.is_connected(n)]
            sys.exit(f"robots not connected: {missing}")
        self.last = {}

    def send(self, name, c):
        self.last[name] = c
        self.ble.send_command(name, c)

    def stop_all(self):
        for n in self.names:
            self.send(n, STOP)
        self.ble.stop_all()

    def connected(self):
        return all(self.ble.is_connected(n) for n in self.names)

    def close(self):
        self.ble.shutdown()


class FakeWorld:
    """--fake: camera and robots at once. Robots follow their last command at
    the measured speeds (with a little noise) and pass through each other."""

    def __init__(self, names, seed=0, speeds=(0.35, 0.58)):
        self.rng = np.random.default_rng(seed)
        self.v, self.ratio = speeds
        self.names = names
        self.pose = {MARKER[n]: [1.3, 1.0, 0.0] for n in names}
        self.cmd = {n: STOP for n in names}
        self.last = self.cmd
        self.t = 0.0

    def place(self, targets):                        # court frame
        for n, (x, y) in targets.items():
            self.pose[MARKER[n]] = [x + self.rng.normal(0, 0.02), y + self.rng.normal(0, 0.02),
                                    self.rng.normal(0, 5.0)]

    def advance(self, dt):
        self.t += dt
        for n in self.names:
            p = self.pose[MARKER[n]]
            vx, vy = char_velocity(self.cmd[n], p[2], self.v, self.ratio)
            p[0] += vx * dt; p[1] += vy * dt
            if vx or vy:
                p[2] += self.rng.normal(0, 0.3)       # the robots turn a little (B3)

    def get(self, marker):
        x, y, yaw = self.pose[marker]
        n = [k for k, v in MARKER.items() if v == marker][0]
        vx, vy = char_velocity(self.cmd.get(n, STOP), yaw, self.v, self.ratio)
        return (x + self.rng.normal(0, 0.005), y + self.rng.normal(0, 0.005), yaw, vx, vy, 0.0)

    def send(self, name, c):
        self.cmd[name] = c

    def stop_all(self):
        for n in self.names:
            self.cmd[n] = STOP

    def connected(self):
        return True

    def close(self):
        pass


class Stop(Exception):
    pass


# --------------------------------------------------------------- shadow sim ---
def make_court(cam, robots, k, def_kind, gate, fake, log, ui, def_start=None, return_margin=None, hold=None):
    """RoboKabaddiEnv whose robots are the real ones (see module doc)."""
    from envs.kabaddi_env import RoboKabaddiEnv, CORRECTED_CONFIG
    import pybullet as p
    from match import GATES

    class RealCourt(RoboKabaddiEnv):
        live = False                                   # off while the simulator sets up

        def reset(self, seed=None, options=None):
            self.live = False
            super().reset(seed=seed, options=options)
            self.names = {self.raider_id: "raider_1"}
            self.names.update({d: f"defender_{i + 1}" for i, d in enumerate(self.defender_ids)})
            targets = {n: tuple(self._get_pos(b)[:2] + OFFSET) for b, n in self.names.items()}
            ui.place(targets)                          # waits for the operator (or --fake)
            self.place_err = {n: math.hypot(*(np.array(cam.get(MARKER[n])[:2]) - t))
                              for n, t in targets.items() if cam.get(MARKER[n])}
            self.live = True
            for b in self.names:
                self._enforce_planar(b)
            self._prev_pos = np.array(self._get_pos(self.raider_id)[:2], dtype=np.float32)
            self._wall0 = cam.t if fake else time.monotonic()
            return self._obs(), {}

        def _step_once(self, action):
            due = self._wall0 + (self._step + 1) * self.dt
            if fake:
                cam.advance(self.dt)
            else:
                lag = due - time.monotonic()
                if lag > 0:
                    time.sleep(lag)
            out = super()._step_once(action)
            log(self)
            return out

        def _drive(self, bid, vx, vy, omega):
            """Send the robot the command for (vx, vy), instead of driving a body."""
            if not self.live:
                return super()._drive(bid, vx, vy, omega)
            n = self.names[bid]
            pose = cam.get(MARKER[n])
            yaw = pose[2] if pose else 0.0
            c = robot_command(vx, vy, yaw, self.MAX_LIN_VEL, self.diagonal_speed_ratio)
            if robots.last.get(n) != c:
                robots.send(n, c)

        def _enforce_planar(self, bid):
            """Put the robot where the camera sees it."""
            if not self.live:
                return super()._enforce_planar(bid)
            n = self.names[bid]
            pose = cam.get(MARKER[n])
            if pose is None or pose[5] > LOST_STOP_S:
                raise Stop(f"{n}: marker not seen")
            x, y = pose[0] - OFFSET[0], pose[1] - OFFSET[1]
            p.resetBasePositionAndOrientation(bid, [x, y, self._spawn_z], [0, 0, 0, 1],
                                              physicsClientId=self._client)
            p.resetBaseVelocity(bid, [pose[3], pose[4], 0], [0, 0, 0],
                                physicsClientId=self._client)

    kw = dict(CORRECTED_CONFIG, n_defenders=k, defender_fouls=True, held_needs_touch=True,
              raider_edge_escape=False,
              defender_type="learned" if def_kind == "learned" else "gate")
    if def_start:                                      # author, 13 Nov
        kw["defender_start_depth"] = def_start
    if return_margin:                                  # author, 13 Nov
        kw["return_margin"] = return_margin
    if hold:                                           # author, 14 Nov
        from defender_env import hold_kwargs
        kw.update(hold_kwargs(hold))
    if def_kind == "scripted":
        kw["gate_style"] = {"v6": "keepout"}.get(gate, gate)          # as eval_gates.py
        assert kw["gate_style"] in GATES, f"gate must be v6..v11, not {gate}"
    env = RealCourt(**kw)
    from envs.kabaddi_env import R_HEIGHT
    env._spawn_z = R_HEIGHT / 2
    return env


# ------------------------------------------------------------------ the UI ---
PLACE_TOL = 0.10        # m: close enough to a start place (author, 11 Nov: no high accuracy)
PLACE_TIMEOUT_S = 45.0  # 25 s was too short for 2.9 m across the court (11 Nov, raid 2)
START_TOL = 0.25        # m: SPACE starts a raid only if every robot is this close (f = start anyway)


class UI:
    def __init__(self, cam, robots, fake, auto_place=True):
        self.cam, self.robots, self.fake, self.auto = cam, robots, fake, auto_place
        self.targets, self.board = {}, ""
        self.media = None                     # --media (raid_media.Media)
        if not fake:
            import cv2
            self.cv2 = cv2
            cv2.namedWindow("Raid runner", cv2.WINDOW_NORMAL)

    def draw(self, extra=""):
        if self.fake:
            return -1
        cv2, cam = self.cv2, self.cam
        td = cam.topdown()
        if td is None:
            return cv2.waitKey(1) & 0xFF
        s, th = cam.scale, cam.th
        cv2.line(td, (0, th // 2), (cam.tw, th // 2), (0, 255, 0), 1)
        for n, (x, y) in self.targets.items():
            lab = "R" if n.startswith("raider") else n[-1]
            col = (0, 0, 255) if lab == "R" else (255, 128, 0)
            cv2.circle(td, (int(x * s), int(th - y * s)), 14, col, 2)
            cv2.putText(td, lab, (int(x * s) + 16, int(th - y * s)), cv2.FONT_HERSHEY_SIMPLEX, 0.6, col, 2)
        for n, m in MARKER.items():
            g = cam.get(m)
            if g and g[5] < 0.5:
                cv2.circle(td, (int(g[0] * s), int(th - g[1] * s)), 5, (0, 255, 255), -1)
        cv2.rectangle(td, (0, 0), (cam.tw, 44), (0, 0, 0), -1)
        cv2.putText(td, self.board[:52], (6, 18), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
        cv2.putText(td, extra[:52], (6, 38), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
        if self.media is not None and self.media.recording:
            td = self.media.tracked(td, s, th)
        cv2.imshow("Raid runner", td)
        return cv2.waitKey(1) & 0xFF

    def now(self):
        return self.cam.t if self.fake else time.monotonic()

    def wait(self, s, extra=""):
        if self.fake:
            self.cam.advance(s)
            return -1
        t0, key = time.monotonic(), -1
        while time.monotonic() - t0 < s:
            k = self.draw(extra)
            key = k if k not in (-1, 255) else key
        return key

    def _drive_all(self, targets):
        try:
            missed = self.drive_to(targets)
        except Stop:
            missed = ["placing stopped by hand"]
        if missed:
            print("  not placed: " + "; ".join(missed))

    def drive_to(self, targets):
        """Drive the robots to their start places, one at a time (defenders,
        deepest first, then the raider), each to within PLACE_TOL: 75 % speed,
        50 % in the last 0.30 m. SPACE or q stops."""
        V = 0.30
        order = sorted((n for n in targets if n != "raider_1"), key=lambda n: -targets[n][1])
        order += [n for n in targets if n == "raider_1"]
        missed = []
        for n in order:
            tx, ty = targets[n]
            t0, sent = self.now(), None
            while True:
                g = self.cam.get(MARKER[n])
                if g is None or g[5] > 0.5:
                    missed.append(f"{n} not seen"); break
                dx, dy = tx - g[0], ty - g[1]
                d = math.hypot(dx, dy)
                if d < PLACE_TOL:
                    break
                if self.now() - t0 > PLACE_TIMEOUT_S:
                    missed.append(f"{n} {100 * d:.0f} cm off"); break
                f = 0.75 if d > 0.30 else 0.5
                c = robot_command(dx / d * f * V, dy / d * f * V, g[2], V, 1.0)
                if c != sent:
                    self.robots.send(n, c); sent = c
                key = self.wait(0.1, f"driving {n} to its place ({100 * d:.0f} cm) - SPACE stops")
                if key in (32, ord("q")):
                    self.robots.stop_all()
                    raise KeyboardInterrupt if key == ord("q") else Stop("placing stopped by hand")
            self.robots.send(n, STOP)
            self.wait(0.3)
        return missed

    def place(self, targets):
        self.targets = targets
        if self.auto:
            self._drive_all(targets)
        elif self.fake:
            self.cam.place(targets)
        if self.fake:
            return
        print(("  check the robots are near the circles" if self.auto else "  place the robots on the circles") + " (" + ", ".join(
            f"{n} x {x:.2f} y {y:.2f}" for n, (x, y) in targets.items()) + "), then SPACE")
        while True:
            far = []
            for n, (x, y) in targets.items():
                g = self.cam.get(MARKER[n])
                if g is None or g[5] > 0.5:
                    far.append(f"{n} not seen")
                elif math.hypot(g[0] - x, g[1] - y) > 0.10:
                    far.append(f"{n} {100 * math.hypot(g[0] - x, g[1] - y):.0f} cm off")
            worst = max((math.hypot(g[0] - x, g[1] - y) if (g := self.cam.get(MARKER[n])) and g[5] <= 0.5
                         else 9.0) for n, (x, y) in targets.items())
            key = self.draw("SPACE start | d drive again | f start anyway | "
                            + ("; ".join(far) if far else "all on place"))
            if key == ord("q"):
                raise KeyboardInterrupt
            if key == ord("d"):
                self._drive_all(targets); continue
            if key in (32, ord("f")):
                if any("not seen" in f for f in far):
                    print("  " + "; ".join(far) + " - put it back on the court, then d"); continue
                if key == 32 and worst > START_TOL:
                    print(f"  a robot is {100 * worst:.0f} cm from its place: d = drive again, "
                          f"f = start anyway"); continue
                if far:
                    print("  starting with: " + "; ".join(far))
                return

    def ask_media(self, img, auto=None):
        """--media after a raid: show the path picture, y = save / n = discard."""
        if auto is not None or self.fake:
            return auto != "n"
        cv2 = self.cv2
        if img is None:
            img = np.zeros((120, 390, 3), np.uint8)
        img = img.copy()
        cv2.rectangle(img, (0, 0), (img.shape[1], 44), (0, 0, 0), -1)
        cv2.putText(img, "save video + snapshots?", (6, 18), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 1)
        cv2.putText(img, "y = save    n = discard", (6, 38), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 1)
        while True:
            cv2.imshow("Raid runner", img)
            k = cv2.waitKey(50) & 0xFF
            if k in (ord("y"), ord("n")):
                return k == ord("y")

    def during(self, env):
        if self.media is not None:
            self.media.step(env, OFFSET)
        rt = raid_time(env)
        key = self.draw((f"raid {rt:4.1f} s" if rt is not None else f"approach {env._step * env.dt:4.1f} s")
                        + f"  contact {'yes' if env._flags['contact'] else 'no'}"
                        "  SPACE = stop")
        if key in (32, ord("q")):
            raise Stop("stopped by hand")
        if not self.robots.connected():
            raise Stop("a robot lost Bluetooth")


def raid_time(env):
    """Seconds since the raider crossed the midline (the raid clock), or None
    before it crosses (author, 13 Nov)."""
    s = getattr(env, "_raid_start_step", None)
    return None if s is None else (env._step - s) * env.dt


def midline_time(env):
    """Seconds from the start to the raider crossing the midline, or None."""
    s = getattr(env, "_raid_start_step", None)
    return None if s is None else s * env.dt


# ----------------------------------------------------------------- checks ---
CHECK_S = 1.0           # s per direction (0.6 s on 11 Nov: too short, 0-36 cm moves)


def check_directions(cam, robots, names, ui):
    """Drive each robot CHECK_S per command; compare the camera's direction with
    the expected one (robot frame from the marker yaw). A test waits for the
    robot's Bluetooth link (11 Nov: a drop during the raider's 'right' test)."""
    tests = [("forward", 0), ("right", -90), ("diag fwd-right", -45), ("diag fwd-left", 45), ("left", 90)]
    for n in names:
        print(f"\n{n}: put it in the middle of the court with room around it, then SPACE (s = skip)")
        while True:
            key = ui.draw(f"direction check {n}: SPACE start, s skip")
            if key == ord("s") or key == 32:
                break
        if key == ord("s"):
            continue
        for label, a in tests:
            t_w = time.monotonic()
            while not robots.ble.is_connected(n) and time.monotonic() - t_w < 15:
                ui.draw(f"{n}: waiting for Bluetooth")
            if not robots.ble.is_connected(n):
                print(f"  {label:<15} skipped: no Bluetooth"); continue
            g0 = cam.get(MARKER[n])
            if g0 is None:
                print("  marker not seen"); break
            robots.send(n, CMD[a][3])
            t0 = time.monotonic()
            while time.monotonic() - t0 < CHECK_S:
                ui.draw(f"{n}: {label}")
            robots.send(n, STOP)
            t1 = time.monotonic()
            while time.monotonic() - t1 < 0.6:
                ui.draw(f"{n}: {label} - stopping")
            g1 = cam.get(MARKER[n])
            dx, dy = g1[0] - g0[0], g1[1] - g0[1]
            got = math.degrees(math.atan2(dy, dx))
            want = 90.0 + g0[2] + a
            err = (got - want + 180) % 360 - 180
            ok = "OK" if abs(err) < 30 else "WRONG WAY?"
            print(f"  {label:<15} moved {100 * math.hypot(dx, dy):4.0f} cm, {err:+5.0f} deg off expected  {ok}")
            # come back
            robots.send(n, CMD[(a + 180 + 180) % 360 - 180 if a != 0 else 180][3])
            t2 = time.monotonic()
            while time.monotonic() - t2 < CHECK_S:
                ui.draw(f"{n}: back")
            robots.send(n, STOP)
            time.sleep(0.4)


# ------------------------------------------------------------------- main ---
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--raider", choices=("learned", "scripted"), default="learned")
    ap.add_argument("--defence", choices=("learned", "scripted"), default="learned")
    ap.add_argument("--n-def", type=int, choices=(1, 2, 3), default=3)
    ap.add_argument("--gate", default="v8", help="scripted defence style v6..v11")
    ap.add_argument("--raids", type=int, default=3)
    ap.add_argument("--seed0", type=int, default=50_000, help="raid i starts as simulated raid seed0 + i")
    ap.add_argument("--pairs", choices=("accepted", "13nov"), default="accepted",
                    help="learned pairs: accepted (match.py PAIRS) or 13nov (diagvar defences; "
                         "1 defender: touch-input raider)")
    ap.add_argument("--media", action="store_true",
                    help="record every raid's videos and snapshots; after each raid y = save / n = discard")
    ap.add_argument("--media-answer", choices=("y", "n"), default=None,
                    help="with --media: answer every save question (tests); default ask")
    ap.add_argument("--raider-rule", choices=("deep",), default=None,
                    help="rule on top of the learned raider (author, 13 Nov): deep = go deep until "
                         "the touch (defender_env.deep_rule); default the policy alone")
    ap.add_argument("--hold", default=None,
                    help="held rule T,D,C: T s within D m with a defender centre within C m "
                         "(author, 14 Nov: 2,0.25,0.25); default 4 s, 5 cm, body contact")
    ap.add_argument("--return-margin", type=float, default=None,
                    help="Empty on return only when the raider's centre is this many m behind the "
                         "midline (author, 13 Nov: 0.03); default 0, as before")
    ap.add_argument("--def-start", default=None,
                    help="defenders start LO,HI metres past the midline (1.0,1.5 = 3.0-3.5 m from the "
                         "raider's end line); default the back row")
    ap.add_argument("--check", action="store_true", help="direction check of every robot, no raids")
    ap.add_argument("--midline-foul", action="store_true",
                    help="old rule: a defender over the midline is always a foul (author's midline touch off)")
    ap.add_argument("--manual-place", action="store_true",
                    help="put the robots on their start places by hand (default: the runner drives them)")
    ap.add_argument("--camera", type=int, default=0)
    ap.add_argument("--fake", action="store_true")
    ap.add_argument("--outdir")
    ap.add_argument("--fake-speeds", default="0.35,0.58",
                    help="--fake: straight speed m/s, diagonal ratio (the sim's own: 0.30,1.0)")
    a = ap.parse_args()
    if a.fake and not a.outdir:
        sys.exit("--fake needs --outdir (never the real data folder)")
    k = a.n_def
    names = ["raider_1"] + [f"defender_{i + 1}" for i in range(k)]
    if a.fake:
        cam = robots = FakeWorld(names, speeds=tuple(float(v) for v in a.fake_speeds.split(",")))
    else:
        cam = Camera(a.camera)
        robots = Robots(names if not a.check else list(MARKER))
    ui = UI(cam, robots, a.fake, auto_place=not a.manual_place)
    out = Path(a.outdir) if a.outdir else DATA / f"session_{time.strftime('%Y-%m-%d_%H%M%S')}"
    try:
        if a.check:
            return check_directions(cam, robots, list(MARKER), ui)
        from match import Player, play_raid, PAIR_SETS
        kinds = {a.raider, a.defence}
        players = {kd: Player(kd, PAIR_SETS[a.pairs], a.raider_rule) for kd in kinds}
        att, dfn = players[a.raider], players[a.defence]
        out.mkdir(parents=True, exist_ok=True)
        steps_f = open(out / "steps.csv", "w", newline="")
        sw = csv.writer(steps_f)
        sw.writerow(["raid", "seed", "t_s", "raider_x", "raider_y"]
                    + [f"d{i + 1}_{c}" for i in range(k) for c in "xy"]
                    + ["cmd_raider"] + [f"cmd_d{i + 1}" for i in range(k)]
                    + ["midline", "contact", "contact_count", "raid_t_s"])
        raid_no = {"i": 0, "seed": 0}

        def log(env):
            P = [env._get_pos(b)[:2] + OFFSET for b in [env.raider_id] + env.defender_ids]
            sw.writerow([raid_no["i"], raid_no["seed"], f"{env._step * env.dt:.2f}"]
                        + [f"{v:.3f}" for q in P for v in q]
                        + [robots.last.get(n, "") for n in names]
                        + [int(env._flags["midline"]), int(env._flags["contact"]), env._contact_count]
                        + ["" if raid_time(env) is None else f"{raid_time(env):.2f}"])
        rows = []
        def_start = tuple(float(v) for v in a.def_start.split(",")) if a.def_start else None
        env = make_court(cam, robots, k, a.defence, a.gate, a.fake, log, ui, def_start, a.return_margin, a.hold)
        court = type("C", (), {"get": lambda self, *_: env})()
        media = None
        if a.media:
            import cv2
            from raid_media import Media
            media = Media(cv2, out, names, fake=a.fake,
                          **({} if a.fake else dict(scale=cam.scale, tw=cam.tw, th=cam.th)))
            ui.media = media
            if not a.fake:
                cam.rec = media.camera_frame
        for i in range(a.raids):
            seed = a.seed0 + i
            raid_no.update(i=i + 1, seed=seed)
            ui.board = (f"raid {i + 1}/{a.raids} seed {seed}: {a.raider} raider vs "
                        f"{k} {a.defence} defenders" + (f" ({a.gate})" if a.defence == "scripted" else ""))
            print(f"\n{ui.board}")
            note = ""
            if media is not None:
                media.start(i + 1, seed)
            try:
                kind, idx, info = play_raid(court, att, dfn, k, seed, a.gate if a.defence == "scripted" else None,
                                            on_step=ui.during, midline_rule=not a.midline_foul)
            except Stop as e:
                kind, idx, info, note = "stopped", [], dict(t=env._step * env.dt), str(e)
            finally:
                robots.stop_all()
            rows.append(dict(raid=i + 1, seed=seed, raider=a.raider, defence=a.defence, n_def=k,
                             gate=a.gate if a.defence == "scripted" else "", kind=kind,
                             out_defenders=" ".join(str(j + 1) for j in idx), t_s=round(info["t"], 1),
                             raid_t_s="" if midline_time(env) is None else round(info["t"] - midline_time(env), 1),
                             midline_t_s="" if midline_time(env) is None else round(midline_time(env), 1),
                             contact=int(env._flags["contact"]),
                             note=note or ("touch through a defender over the midline" if info.get("via") else ""),
                             fake=int(a.fake), def_start=a.def_start or "back_row", pairs=a.pairs,
                             return_margin=a.return_margin or 0, hold=a.hold or "4,0.05,body",
                             raider_rule=(a.raider_rule or "") if a.raider == "learned" else "",
                             start_off_cm_max=round(100 * max(getattr(env, "place_err", {0: 0}).values()), 1)))
            note = rows[-1]["note"]
            if media is not None:
                rts = rows[-1]["raid_t_s"]
                pic = media.finish(f"{kind}" + (f", raid {rts} s from the midline" if rts != "" else
                                                f" after {info['t']:.1f} s (midline not crossed)"))
                saved = media.keep(ui.ask_media(pic, a.media_answer))
                rows[-1]["media"] = saved
                print(f"  media: {saved}")
            rts = rows[-1]["raid_t_s"]
            when = (f"after {rts} s from the midline ({info['t']:.1f} s from the start)" if rts != ""
                    else f"after {info['t']:.1f} s, midline not crossed")
            print(f"  -> {kind}{' (' + note + ')' if note else ''} {when}"
                  f"  (start places within {rows[-1]['start_off_cm_max']:.0f} cm)")
            with open(out / "raids.csv", "w", newline="") as f:
                w = csv.DictWriter(f, fieldnames=list(rows[0]))
                w.writeheader(); w.writerows(rows)
            steps_f.flush()
        steps_f.close()
        print(f"\n{len(rows)} raids -> {out}")
    except KeyboardInterrupt:
        print("quit")
    finally:
        robots.stop_all()
        robots.close()
        if not a.fake:
            cam.close()


if __name__ == "__main__":
    main()
