"""
view_defenders.py — watch learned defenders against a frozen raider (C3)
========================================================================

Opens the PyBullet window and plays seeded raids in real time, the defenders
taking their most likely action, as in eval_defenders.py (same seeds give the
same raids). Defenders are tagged 1-3; the status line shows time, counted
contact and the raider's distance to the nearest line.

    python3 marl/view_defenders.py --ckpt results/marl/run/checkpoints/mappo_N_steps.pt \\
        --raider main_1v3_seed42_ext --seed 50000 --episodes 3
"""
from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
from defender_env import DefenderEnv                                  # noqa: E402
from eval_defenders import load                                       # noqa: E402
from envs.kabaddi_env import COURT_W, COURT_L                         # noqa: E402
import pybullet as p                                                  # noqa: E402

COLOUR = [0.1, 0.1, 0.1]


def text(t, pos, uid=None, size=1.3, parent=None):
    kw = dict(textColorRGB=COLOUR, textSize=size)
    if uid is not None:
        kw["replaceItemUniqueId"] = uid
    if parent is not None:
        kw["parentObjectUniqueId"] = parent
    return p.addUserDebugText(t, pos, **kw)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--raider", default="main_1v3_seed42_ext")
    ap.add_argument("--seed", type=int, default=50000)
    ap.add_argument("--episodes", type=int, default=3)
    ap.add_argument("--speed", type=float, default=1.0, help="1 = real time")
    ap.add_argument("--no-raider-escape", action="store_true")
    ap.add_argument("--no-held-touch", action="store_true")
    args = ap.parse_args()
    act = load(args.ckpt)
    denv = DefenderEnv(args.raider, render=True,
                       raider_edge_escape=not args.no_raider_escape,
                       held_needs_touch=not args.no_held_touch)
    env = denv.env
    try:
        p.configureDebugVisualizer(p.COV_ENABLE_GUI, 0)
    except Exception:
        pass
    title = f"learned defenders ({Path(args.ckpt).stem})  vs  raider {args.raider}"
    try:
        for ep in range(args.episodes):
            seed = args.seed + ep
            obs, _ = denv.reset(seed=seed)
            text(title, [-1.3, 2.25, 0.6])
            stat = text("", [-1.3, 2.25, 0.42], size=1.2)
            for k, did in enumerate(env.defender_ids):
                text(str(k + 1), [0, 0, 0.25], size=2.0, parent=did)
            t0, info = time.time(), {}
            while True:
                obs, _, _, done, info = denv.step(act(obs))
                r = env._get_pos(env.raider_id)
                to_line = min(COURT_W / 2 - abs(r[0]), COURT_L / 2 - abs(r[1]))
                stat = text(f"raid seed {seed}   t = {env._step * env.dt:4.1f} s   "
                            f"contact = {'yes' if env._flags['contact'] else 'no'}   "
                            f"raider to nearest line {to_line:+.2f} m",
                            [-1.3, 2.25, 0.42], uid=stat, size=1.2)
                lag = t0 + env._step * env.dt / args.speed - time.time()
                if lag > 0:
                    time.sleep(lag)
                if done:
                    break
            out = info.get("outcome", "empty")
            r = env._get_pos(env.raider_id)
            how = ""
            if out == "tackle" and (abs(r[0]) > COURT_W / 2 or abs(r[1]) > COURT_L / 2):
                how = " (raider over the line)"
            elif out == "foul":
                how = f" ({info.get('fouls', 1)} defender(s) over a line)"
            text(f"{out.upper()}{how} after {env._step * env.dt:.1f} s", [-1.3, 2.25, 0.24],
                 size=1.6)
            print(f"raid seed {seed}: {out}{how} after {env._step * env.dt:4.1f} s", flush=True)
            time.sleep(3.0 / args.speed)
    except KeyboardInterrupt:
        pass
    except p.error:
        if p.isConnected(env._client):
            raise
        print("window closed")
    finally:
        try:
            denv.close()
        except Exception:
            pass


if __name__ == "__main__":
    main()
