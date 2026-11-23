"""
Copied 4 Nov 2026 (author) from the Paper 2 rig (hardware/camera_model.py) for the
Robo-Kabaddi robot runs; the original is unchanged. Changed here: court 2.6 x 4.0 m
(was 2.0 x 3.0), and load() warns when a file was made for another court size.

camera_model.py -- pixel <-> court mapping for the overhead camera, with
  (1) fisheye/barrel lens correction (one-parameter DIVISION model about a centre c:
        p_undist = c + (p_dist - c) / (1 + k r^2),  r = |p_dist - c|  in pixels),
  (2) the floor homography H (undistorted pixel -> court metres, from the 4 corners),
  (3) marker-height correction: a marker h metres above the floor at camera height
      Hc appears pushed AWAY from the point under the camera (the nadir) by the
      factor Hc / (Hc - h); positions are pulled back toward the nadir by (1 - h/Hc).
      The nadir is taken as the court point under the image centre c (camera looking
      down; a small tilt only shifts the nadir a little).

Calibration files without the 'distortion' / 'heights' entries (Paper 1, the first
Paper 2 calibration) load with k = 0 and no height correction, i.e. exactly the old
4-corner homography.

Court frame: x 0..2.6 m across, y 0..4.0 m along the raid axis (raider end y = 0).
"""
from __future__ import annotations
import json
import math
import numpy as np

COURT_W, COURT_L = 2.6, 4.0
MID = COURT_L / 2


class CameraModel:
    def __init__(self, H, k=0.0, cx=None, cy=None, camera_height=None, marker_height=None, nadir=None):
        self.H = np.asarray(H, float)
        self.Hinv = np.linalg.inv(self.H)
        self.k = float(k)
        self.cx, self.cy = cx, cy
        self.camera_height, self.marker_height = camera_height, marker_height
        self.nadir = None
        if cx is not None and cy is not None:
            self.nadir = self._floor(np.array([[cx, cy]], float))[0]
        if nadir is not None:                       # measured with check_points.py
            self.nadir = np.asarray(nadir, float)

    # ---- loading -------------------------------------------------------
    @classmethod
    def load(cls, path):
        d = json.load(open(path))
        if (d.get("court_width_m"), d.get("court_length_m")) != (COURT_W, COURT_L):
            print(f"[camera_model] WARNING: {path} was made for a "
                  f"{d.get('court_width_m')} x {d.get('court_length_m')} m court, not "
                  f"{COURT_W} x {COURT_L}: recalibrate with calibrate_rig.py")
        dist = d.get("distortion") or {}
        hts = d.get("heights") or {}
        return cls(d["homography_matrix"], k=dist.get("k", 0.0), cx=dist.get("cx"), cy=dist.get("cy"),
                   camera_height=hts.get("camera_m"), marker_height=hts.get("marker_m"), nadir=hts.get("nadir"))

    @property
    def height_factor(self):
        if self.camera_height and self.marker_height and self.nadir is not None:
            return 1.0 - self.marker_height / self.camera_height
        return 1.0

    def describe(self):
        s = f"lens k={self.k:.3e}" + (f" centre ({self.cx:.0f},{self.cy:.0f})" if self.cx is not None else "")
        if self.height_factor != 1.0:
            s += (f"; marker height {self.marker_height:.3f} m / camera {self.camera_height:.2f} m"
                  f" (positions x{self.height_factor:.3f} about nadir {self.nadir[0]:.2f},{self.nadir[1]:.2f})")
        else:
            s += "; no marker-height correction"
        return s

    # ---- lens ----------------------------------------------------------
    def undistort(self, pts):
        p = np.atleast_2d(np.asarray(pts, float))
        if self.k == 0.0 or self.cx is None:
            return p
        c = np.array([self.cx, self.cy])
        d = p - c
        r2 = (d ** 2).sum(1, keepdims=True)
        return c + d / (1.0 + self.k * r2)

    def distort(self, pts):
        p = np.atleast_2d(np.asarray(pts, float))
        if self.k == 0.0 or self.cx is None:
            return p
        c = np.array([self.cx, self.cy])
        d = p - c
        ru = np.sqrt((d ** 2).sum(1, keepdims=True))
        with np.errstate(invalid="ignore", divide="ignore"):
            rd = (1.0 - np.sqrt(np.maximum(1.0 - 4.0 * self.k * ru ** 2, 0.0))) / (2.0 * self.k * ru)
            scale = np.where(ru > 1e-9, rd / np.maximum(ru, 1e-12), 1.0)
        return c + d * scale

    # ---- floor plane ---------------------------------------------------
    def _floor(self, pts_px):
        """distorted pixels -> court metres ON THE FLOOR (no height correction)"""
        u = self.undistort(pts_px)
        q = np.c_[u, np.ones(len(u))] @ self.H.T
        return q[:, :2] / q[:, 2:3]

    def floor_to_pix(self, pts_court):
        """court metres on the floor -> distorted pixels"""
        p = np.atleast_2d(np.asarray(pts_court, float))
        q = np.c_[p, np.ones(len(p))] @ self.Hinv.T
        return self.distort(q[:, :2] / q[:, 2:3])

    # ---- markers (on top of the robots) --------------------------------
    def pix_to_court(self, px, py):
        """marker pixel -> court position of the robot (floor point under the marker)"""
        f = self._floor(np.array([[px, py]], float))[0]
        if self.height_factor != 1.0:
            f = self.nadir + (f - self.nadir) * self.height_factor
        return float(f[0]), float(f[1])

    # ---- display -------------------------------------------------------
    def topdown_maps(self, scale, out_w, out_h, margin=0.0):
        """cv2.remap maps: top-down court image (scale px/m, raider end at the bottom)
        sampled from the distorted camera image. Floor plane (court lines).
        margin: m of floor shown beyond each court line."""
        xs = (np.arange(out_w) + 0.5) / scale - margin
        ys = COURT_L + margin - (np.arange(out_h) + 0.5) / scale
        X, Y = np.meshgrid(xs, ys)
        pix = self.floor_to_pix(np.c_[X.ravel(), Y.ravel()])
        return (pix[:, 0].reshape(out_h, out_w).astype(np.float32),
                pix[:, 1].reshape(out_h, out_w).astype(np.float32))

    def court_polyline(self, a, b, n=40):
        """pixels of the straight court segment a->b (curved in the camera image)"""
        t = np.linspace(0, 1, n)[:, None]
        return self.floor_to_pix(np.asarray(a, float) * (1 - t) + np.asarray(b, float) * t)


# ------------------------------------------------------------------- fitting ---
COURT_CORNERS = np.array([[0.0, COURT_L], [COURT_W, COURT_L], [COURT_W, 0.0], [0.0, 0.0]])   # TL TR BR BL
# line name -> (axis index fixed, value): points on it must map to court[axis] == value
LINES = {"defender end line": (1, COURT_L), "raider end line": (1, 0.0),
         "left side line": (0, 0.0), "right side line": (0, COURT_W), "mid-line": (1, MID)}


def _homography(corners_u):
    """Exact 4-point homography (undistorted pixel -> court), in float64 (OpenCV's
    float32 version is too coarse for the numerical derivatives of the fit)."""
    src = np.asarray(corners_u, float); dst = COURT_CORNERS
    A, b = [], []
    for (x, y), (u, v) in zip(src, dst):
        A.append([x, y, 1, 0, 0, 0, -u * x, -u * y]); b.append(u)
        A.append([0, 0, 0, x, y, 1, -v * x, -v * y]); b.append(v)
    h = np.linalg.solve(np.array(A), np.array(b))
    return np.append(h, 1.0).reshape(3, 3)


def residuals(params, corners_px, line_pts, k_scale=1e-7):
    """metres off their line, for every clicked line point; corners are exact by construction."""
    k = params[0] * k_scale
    m = CameraModel(np.eye(3), k=k, cx=params[1], cy=params[2])
    Hm = _homography(m.undistort(corners_px))
    m = CameraModel(Hm, k=k, cx=params[1], cy=params[2])
    res = []
    for name, pts in line_pts.items():
        if len(pts) == 0:
            continue
        ax, val = LINES[name]
        f = m._floor(np.asarray(pts, float))
        res.append(f[:, ax] - val)
    return np.concatenate(res) if res else np.zeros(1)


def fit(corners_px, line_pts, image_size, k_scale=1e-7):
    """Fit lens k and centre (and the corner homography). Returns (CameraModel, info)."""
    from scipy.optimize import least_squares
    w, h = image_size
    x0 = np.array([0.0, w / 2.0, h / 2.0])
    r0 = residuals(x0, corners_px, line_pts, k_scale)
    sol = least_squares(residuals, x0, args=(corners_px, line_pts, k_scale),
                        bounds=([-20.0, 0.3 * w, 0.3 * h], [20.0, 0.7 * w, 0.7 * h]),
                        x_scale=[1.0, 50.0, 50.0], diff_step=1e-3)
    r1 = residuals(sol.x, corners_px, line_pts, k_scale)
    k = sol.x[0] * k_scale
    m0 = CameraModel(np.eye(3), k=k, cx=sol.x[1], cy=sol.x[2])
    model = CameraModel(_homography(m0.undistort(corners_px)), k=k, cx=sol.x[1], cy=sol.x[2])
    info = dict(rms_before_m=float(np.sqrt(np.mean(r0 ** 2))), max_before_m=float(np.abs(r0).max()),
                rms_after_m=float(np.sqrt(np.mean(r1 ** 2))), max_after_m=float(np.abs(r1).max()),
                n_points=int(len(r1)), k=float(k), cx=float(sol.x[1]), cy=float(sol.x[2]))
    return model, info


def save(path, model, info, corners_px, line_pts, camera_index, camera_height=None, marker_height=None):
    d = {"homography_matrix": model.H.tolist(), "court_width_m": COURT_W, "court_length_m": COURT_L,
         "camera_index": camera_index, "calibration_method": "corners + court lines, division lens model",
         "fisheye_corrected": True,
         "distortion": {"model": "division", "k": model.k, "cx": model.cx, "cy": model.cy},
         "heights": {"camera_m": camera_height, "marker_m": marker_height},
         "fit": info, "clicks": {"corners": np.asarray(corners_px).tolist(),
                                 "lines": {k: np.asarray(v).tolist() for k, v in line_pts.items()}}}
    json.dump(d, open(path, "w"), indent=1)
