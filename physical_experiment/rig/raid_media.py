"""
raid_media.py -- videos and snapshots of every robot raid (author, 13 Nov)
==========================================================================

Used by raid_runner.py --media. For each raid, from the first step to the end:

  camera.mp4         the camera picture (fisheye), half size (960 x 540) to save
                     disk space (13 Nov: 6.7 GB free); snapshots are full size
  tracked.mp4        the runner's top-down view with each robot's path so far
  frame_times.csv    the time of every picture taken (s from the raid start); the
                     videos run at 25 (camera) and 20 (tracked) frames per second
  start_tracked.png / start_camera.jpg   first frame of the raid
  touch_tracked.png / touch_camera.jpg   first frame with the touch counted (if any)
  end_tracked.png   / end_camera.jpg     last frame (camera: full size, JPG quality 95)
  paths.png          the top-down view at the end, the whole path of every
                     robot, the start places and the touch point, the result

Recorded into media/.tmp_<raid>; after the raid the runner shows paths.png and
asks y = save / n = discard. Saved: media/raid<NN>_seed<S>/ (a suffix if that
folder exists). Discarded: the temporary folder is removed.
"""
from __future__ import annotations

import csv
import shutil
import threading
import time
from pathlib import Path

import numpy as np

COLOURS = {"raider_1": (0, 0, 255), "defender_1": (255, 128, 0), "defender_2": (0, 200, 0),
           "defender_3": (200, 0, 200)}
CAM_FPS = 25.0        # the court camera gives 19-28 frames per second (13 Nov)
TRACK_FPS = 20.0      # tracked video frames per second
CAM_VIDEO_SCALE = 0.5  # camera video size (snapshots full size)
# Both videos follow the raid clock: a frame is repeated or skipped so that
# frame k is the picture at k / fps seconds after the raid's first step.


class Media:
    def __init__(self, cv2, session: Path, names, fake=False, scale=150, tw=390, th=600):
        self.cv2, self.root, self.names, self.fake = cv2, Path(session) / "media", names, fake
        self.scale, self.tw, self.th = scale, tw, th
        self.recording, self.dir = False, None
        self.lock = threading.Lock()
        self.jpg = [cv2.IMWRITE_JPEG_QUALITY, 95]

    def _snap_camera(self, name, frame):
        self.cv2.imwrite(str(self.dir / f"{name}_camera.jpg"), frame, self.jpg)

    # ------------------------------------------------------------ recording
    def start(self, raid, seed):
        """Before play_raid: a new temporary folder; recording begins at the
        raid's first step (placing the robots is not recorded)."""
        self.name = f"raid{raid:02d}_seed{seed}"
        self.dir = self.root / f".tmp_{self.name}"
        if self.dir.exists():
            shutil.rmtree(self.dir)
        self.dir.mkdir(parents=True)
        self.cam_w = self.trk_w = None
        self.times = []                          # (video, frame, t)
        self.paths = {n: [] for n in self.names}
        self.starts = None
        self.touch_at = None
        self.last_cam = self.last_trk = None
        self.saved_touch = False
        self.recording = False
        self.t0 = None
        self.clock = 0.0                         # raid time (env), s
        self.n_cam = self.n_trk = 0

    def camera_frame(self, frame, t):
        """Camera thread: every camera frame while recording."""
        with self.lock:
            if not self.recording:
                return
            h, w = frame.shape[:2]
            size = (int(w * CAM_VIDEO_SCALE), int(h * CAM_VIDEO_SCALE))
            if self.cam_w is None:
                self.cam_w = self.cv2.VideoWriter(str(self.dir / "camera.mp4"),
                                                  self.cv2.VideoWriter_fourcc(*"mp4v"), CAM_FPS, size)
                self._snap_camera("start", frame)
            due = int((t - self.t0) * CAM_FPS) + 1 - self.n_cam
            if due > 0:
                small = self.cv2.resize(frame, size, interpolation=self.cv2.INTER_AREA)
                for _ in range(due):
                    self.cam_w.write(small)
                    self.n_cam += 1
            self.times.append(("camera", 0, t - self.t0))
            self.last_cam = frame

    def step(self, env, offset):
        """Runner step (UI.during): robot places, the touch."""
        if not self.recording:
            self.recording, self.t0 = True, time.monotonic()
        self.clock = env._step * env.dt
        P = [env._get_pos(b)[:2] + offset for b in [env.raider_id] + env.defender_ids]
        if self.starts is None:
            self.starts = [tuple(p) for p in P]
        for n, p in zip(self.names, P):
            self.paths[n].append((float(p[0]), float(p[1])))
        if env._flags["contact"] and self.touch_at is None:
            self.touch_at = self.paths["raider_1"][-1]
        if self.fake:                            # no camera: a plain court picture
            self.tracked(self.blank(), self.scale, self.th)

    def blank(self):
        img = np.full((self.th, self.tw, 3), 200, np.uint8)
        for y in (0.0, 2.0, 4.0):
            self.cv2.line(img, (0, int(self.th - y * self.scale)), (self.tw, int(self.th - y * self.scale)),
                          (60, 60, 60), 1)
        return img

    def _draw_paths(self, img, s, th, labels=False):
        cv2 = self.cv2
        px = lambda q: (int(q[0] * s), int(th - q[1] * s))
        for n, pts in self.paths.items():
            if len(pts) > 1:
                cv2.polylines(img, [np.array([px(q) for q in pts], np.int32)], False, COLOURS[n], 2)
        if labels and self.starts:
            for n, q in zip(self.names, self.starts):
                cv2.circle(img, px(q), 7, COLOURS[n], 1)
                cv2.putText(img, "R" if n.startswith("raider") else n[-1], (px(q)[0] + 9, px(q)[1]),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, COLOURS[n], 1)
        if self.touch_at is not None:
            cv2.drawMarker(img, px(self.touch_at), (0, 255, 255), cv2.MARKER_STAR, 18, 2)
        return img

    def tracked(self, td, s, th):
        """UI.draw while recording: the top-down view with the paths; written
        to tracked.mp4. Returns the picture to show."""
        img = self._draw_paths(td.copy(), s, th)
        h, w = img.shape[:2]
        if self.trk_w is None:
            self.trk_w = self.cv2.VideoWriter(str(self.dir / "tracked.mp4"),
                                              self.cv2.VideoWriter_fourcc(*"mp4v"), TRACK_FPS, (w, h))
            self.cv2.imwrite(str(self.dir / "start_tracked.png"), img)
        t = self.clock if self.fake else time.monotonic() - self.t0
        for _ in range(max(0, int(t * TRACK_FPS) + 1 - self.n_trk)):
            self.trk_w.write(img)
            self.n_trk += 1
        self.times.append(("tracked", 0, t))
        self.last_trk = img
        if self.touch_at is not None and not self.saved_touch:
            self.saved_touch = True
            self.cv2.imwrite(str(self.dir / "touch_tracked.png"), img)
            with self.lock:
                if self.last_cam is not None:
                    self._snap_camera("touch", self.last_cam)
        return img

    # ------------------------------------------------------------ after the raid
    def finish(self, result):
        """Stop recording, close the videos, write the end snapshots, the
        frame times and paths.png. Returns paths.png (or None)."""
        with self.lock:
            self.recording = False
            if self.cam_w is not None:
                self.cam_w.release()
            if self.last_cam is not None:
                self._snap_camera("end", self.last_cam)
        if self.trk_w is not None:
            self.trk_w.release()
        with open(self.dir / "frame_times.csv", "w", newline="") as f:
            w = csv.writer(f)
            w.writerow(["video", "frame", "t_s"])
            n = {"camera": 0, "tracked": 0}
            for v, _, t in self.times:
                w.writerow([v, n[v], f"{t:.3f}"])
                n[v] += 1
        if self.last_trk is None:
            return None
        self.cv2.imwrite(str(self.dir / "end_tracked.png"), self.last_trk)
        img = self._draw_paths(self.last_trk.copy(), self.scale, self.th, labels=True)
        self.cv2.rectangle(img, (0, img.shape[0] - 26), (img.shape[1], img.shape[0]), (0, 0, 0), -1)
        self.cv2.putText(img, f"{self.name}: {result}"[:52], (6, img.shape[0] - 8),
                         self.cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
        self.cv2.imwrite(str(self.dir / "paths.png"), img)
        return img

    def keep(self, yes: bool):
        """Save (rename the temporary folder) or discard it. Returns the saved
        folder, or 'discarded'."""
        if not yes:
            shutil.rmtree(self.dir, ignore_errors=True)
            return "discarded"
        dest, k = self.root / self.name, 2
        while dest.exists():
            dest, k = self.root / f"{self.name}_{k}", k + 1
        self.dir.rename(dest)
        return str(dest)
