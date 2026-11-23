"""
show_starts.py -- where the robots start in the simulator tests (author, 13 Nov)
==============================================================================
Raid i of the robot runner (raid_runner.py, --seed0 50000) starts like
simulated test raid seed 50000 + i, so these are also the robots' start places.
Court frame as the runner's circles: x across (0-2.6 m), y from the raider's
end line (0-4.0 m), midline y = 2.0. Writes start_places.csv and .png here.
--def-start LO,HI (metres past the midline; author, 13 Nov: 1.0,1.5 = 3.0-3.5 m
on the court) uses the simulator's defender_start_depth and writes
start_places_<LO>_<HI>.csv / .png.

    python3 physical_experiment/rig/start_places/show_starts.py --raids 15
    python3 physical_experiment/rig/start_places/show_starts.py --raids 15 --def-start 1.0,1.5
"""
import argparse, csv, sys
from pathlib import Path
import numpy as np
ROOT = Path(__file__).resolve().parents[3]
sys.path[:0] = [str(ROOT), str(ROOT / "marl")]
from envs.kabaddi_env import RoboKabaddiEnv, CORRECTED_CONFIG, BACK_ROW_FROM_WALL, NEAR_MIDLINE_X, NEAR_MIDLINE_Y

OFF = np.array([1.3, 2.0])
ap = argparse.ArgumentParser(); ap.add_argument("--raids", type=int, default=15)
ap.add_argument("--def-start", default=None)
a = ap.parse_args()
DS = tuple(float(v) for v in a.def_start.split(",")) if a.def_start else None
KW = {"defender_start_depth": DS} if DS else {}
TAG = f"_{DS[0]:g}_{DS[1]:g}" if DS else ""
OUT = Path(__file__).resolve().parent
rows = []
for k in (1, 2, 3):
    for i in range(a.raids):
        e = RoboKabaddiEnv(**dict(CORRECTED_CONFIG, n_defenders=k, **KW)); e.reset(seed=50000 + i)
        P = [e._get_pos(b)[:2] + OFF for b in [e.raider_id] + e.defender_ids]; e.close()
        names = ["raider_1"] + [f"defender_{j + 1}" for j in range(k)]
        for n, (x, y) in zip(names, P):
            rows.append(dict(n_def=k, raid=i + 1, seed=50000 + i, robot=n, x_m=round(float(x), 2), y_m=round(float(y), 2)))
with open(OUT / f"start_places{TAG}.csv", "w", newline="") as f:
    w = csv.DictWriter(f, fieldnames=list(rows[0])); w.writeheader(); w.writerows(rows)

import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
XR = {1: [(-0.7, 0.7)], 2: [(-1.0, -0.2), (0.2, 1.0)], 3: [(-1.0, -0.3), (-0.2, 0.2), (0.3, 1.0)]}
C = ["#d62728", "#1f77b4", "#2ca02c"]
fig, ax = plt.subplots(1, 3, figsize=(12, 6.4))
for k, axk in zip((1, 2, 3), ax):
    axk.add_patch(Rectangle((0, 0), 2.6, 4.0, fill=False, lw=1.5))
    for y, lab in [(2.0, "midline"), (2.5, "baulk"), (3.5, "bonus")]:
        axk.axhline(y, color="k", lw=1.2 if y == 2.0 else 0.6, ls="-" if y == 2.0 else "--")
        axk.text(2.62, y, lab, va="center", fontsize=8)
    y0, y1 = ((2.0 + DS[0], 2.0 + DS[1]) if DS else (4.0 - BACK_ROW_FROM_WALL[1], 4.0 - BACK_ROW_FROM_WALL[0]))
    for j, (x0, x1) in enumerate(XR[k]):
        axk.add_patch(Rectangle((x0 + 1.3, y0), x1 - x0, y1 - y0, color=C[j], alpha=0.18))
    axk.add_patch(Rectangle((NEAR_MIDLINE_X[0] + 1.3, 2.0 + NEAR_MIDLINE_Y[0]), np.ptp(NEAR_MIDLINE_X), np.ptp(NEAR_MIDLINE_Y), color="k", alpha=0.12))
    for r in rows:
        if r["n_def"] != k: continue
        c = "k" if r["robot"] == "raider_1" else C[int(r["robot"][-1]) - 1]
        axk.plot(r["x_m"], r["y_m"], "o", ms=5, color=c)
    axk.set(xlim=(-0.1, 3.0), ylim=(-0.1, 4.1), aspect="equal", title=f"{k} defender{'s' if k > 1 else ''}: first {a.raids} raids",
            xlabel="across the court (m)", ylabel="from the raider's end line (m)")
handles = [plt.Line2D([], [], marker="o", ls="", color=c, label=l) for c, l in
           [("k", "raider"), (C[0], "defender 1"), (C[1], "defender 2"), (C[2], "defender 3")]]
fig.legend(handles=handles, loc="lower center", ncol=4); fig.tight_layout(rect=(0, 0.05, 1, 1))
fig.savefig(OUT / f"start_places{TAG}.png", dpi=130)
print("written", OUT / f"start_places{TAG}.csv", OUT / f"start_places{TAG}.png")
