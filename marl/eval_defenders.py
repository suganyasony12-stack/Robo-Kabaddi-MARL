"""
eval_defenders.py — play learned defenders against frozen raiders (C3)
======================================================================

Seeded raids (starts 50000 + i, as exp_C1_evaluate.py), the defenders taking
their most likely action (deterministic). One job per (checkpoint, raider).
Reports touch / tackle / foul / empty with Wilson 95% intervals, how tackles
ended (held, 30 s timer, out of court), contact depth and raid length.

    python3 marl/eval_defenders.py --ckpt results/marl/run/checkpoints/mappo_N_steps.pt \\
        --raiders main_1v3_seed42_ext,scripted --episodes 200
    --ckpt random    plays uniformly random defenders (the starting point)
"""
from __future__ import annotations

import argparse
import json
import os
import sys
from collections import Counter
from multiprocessing import get_context
from pathlib import Path

os.environ.setdefault("OMP_NUM_THREADS", "1")
import numpy as np                                                    # noqa: E402

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
from defender_env import DefenderEnv, N_ACTIONS, ROOT, diag_kwargs, start_kwargs, margin_kwargs, hold_kwargs                # noqa: E402
sys.path.insert(0, str(ROOT / "revision_experiments"))
from common import wilson_ci                                          # noqa: E402
from envs.kabaddi_env import COURT_W, COURT_L, RoboKabaddiEnv         # noqa: E402

DEF_OBS_FIXED = RoboKabaddiEnv.DEF_OBS_FIXED

EVAL_SEED0 = 50_000


def load(ckpt):
    import torch
    from mappo import Agent
    torch.set_num_threads(1)
    c = torch.load(ckpt, map_location="cpu", weights_only=False)
    agent = Agent(c["obs_dim"], c["state_dim"], c["algo"])
    agent.load_state_dict(c["agent"])
    m, v = np.array(c["onorm"]["mean"]), np.array(c["onorm"]["var"])

    def act(obs):
        o = np.clip((obs - m) / np.sqrt(v + 1e-8), -10, 10).astype(np.float32)
        with torch.no_grad():
            return agent.actor(torch.as_tensor(o)).argmax(-1).numpy()
    act.n_in = (c["obs_dim"] - DEF_OBS_FIXED) // 2 + 1    # defenders in its input
    return act


def run(job):
    ckpt, raider, n, escape, held_touch, n_def, *rest = job
    phantom = rest[0] if rest else None
    diag = rest[1] if len(rest) > 1 else None          # measured robots: B3, 11 Nov
    dkw = rest[2] if len(rest) > 2 else {}             # per-robot diagonal speed
    rule = rest[3] if len(rest) > 3 else None          # author, 13 Nov
    if ckpt == "random":
        rng = np.random.default_rng(0)
        act = lambda obs: rng.integers(N_ACTIONS, size=len(obs))
    else:
        act = load(ckpt)
    env = DefenderEnv(raider, n_def, pad_to=getattr(act, "n_in", None), raider_phantom=phantom, raider_rule=rule,
                      raider_edge_escape=escape, held_needs_touch=held_touch,
                      **({"diagonal_speed_ratio": diag} if diag else {}), **dkw)
    outs, why, depth, dur = [], Counter(), [], []
    for ep in range(n):
        obs, _ = env.reset(seed=EVAL_SEED0 + ep)
        e, counted, info = env.env, False, {}
        while True:
            obs, _, _, done, info = env.step(act(obs))
            if e._flags["contact"] and not counted:
                counted = True
                depth.append(float(e._get_pos(e.raider_id)[1]))
            if done:
                break
        out = info.get("outcome", "empty")
        if out == "tackle":
            pos = e._get_pos(e.raider_id)
            outside = abs(pos[0]) > COURT_W / 2 or abs(pos[1]) > COURT_L / 2
            start = e._raid_start_step
            why["out_of_court" if outside else "held" if (
                start is not None and e._step - start < e._raid_steps) else "timer"] += 1
        outs.append(out)
        dur.append(e._step * e.dt)
    env.close()
    c = Counter(outs)
    res = {"ckpt": str(ckpt), "raider": raider, "n": n, "n_def": n_def,
           "raider_phantom": getattr(env.raiders[0], "phantom", None),
           "raider_edge_escape": env.escape[0],
           "held_needs_touch": held_touch,
           "diagonal_speed_ratio": env.env.diagonal_speed_ratio,
           "diag_ratio_range": env.env.diag_ratio_range, "diag_ratio_period": dkw.get("diag_ratio_period"),
           "defender_start_depth": env.env.defender_start_depth, "return_margin": env.env.return_margin,
           "hold": [env.env.static_time, env.env.hold_dist, env.env.hold_touch_dist],
           "raider_rule": getattr(env.raiders[0], "rule", None),
           "rule_changed_decisions": getattr(env.raiders[0], "rule_changed", 0),
           "tackle_how": dict(why),
           "contact_depth_m": float(np.mean(depth)) if depth else None,
           "median_raid_s": float(np.median(dur)), "outcomes": outs}
    for k in ("touch", "tackle", "foul", "empty"):
        lo, hi = wilson_ci(c[k], n)
        res[k] = {"rate": c[k] / n, "ci": [lo, hi], "count": c[k]}
    return res


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", nargs="+", required=True)
    ap.add_argument("--raiders", default="main_1v3_seed42_ext")
    ap.add_argument("--episodes", type=int, default=1000)
    ap.add_argument("--procs", type=int, default=6)
    ap.add_argument("--no-raider-escape", action="store_true",
                    help="raiders without the robots' boundary escape")
    ap.add_argument("--no-held-touch", action="store_true",
                    help="old held rule, no defender touch needed (27-28 Oct night)")
    ap.add_argument("--n-def", type=int, default=3, help="defenders on the court")
    ap.add_argument("--raider-phantom", choices=("deep", "copy"), default=None,
                    help="where frozen raiders' phantom defenders are; default: each raider's own")
    ap.add_argument("--diag-ratio", type=float, default=None,
                    help="diagonal speed / straight speed for every robot (B3 measured 0.58); default 1.0")
    ap.add_argument("--diag-range", default=None, help="every robot gets its own diagonal speed ratio per raid, uniform in LO,HI (B3: 0.35,0.85); default off")
    ap.add_argument("--diag-period", type=float, default=None, help="with --diag-range: new ratios every this many seconds; default once per raid")
    ap.add_argument("--raider-rule", choices=("deep",), default=None, help="rule on top of the trained raider (author, 13 Nov; defender_env.deep_rule): deep = go deep until the touch; default the policy alone")
    ap.add_argument("--return-margin", type=float, default=None, help="Empty on return only this many m behind the midline (author, 13 Nov: 0.03); default 0")
    ap.add_argument("--hold", default=None, help="held rule T,D,C: T s within D m with a defender centre within C m (author, 14 Nov: 2,0.25,0.25); default 4 s, 5 cm, body contact")
    ap.add_argument("--def-start", default=None, help="defenders start LO,HI metres past the midline (author, 13 Nov: 1.0,1.5 = 3.0-3.5 m from the raider's end line); default the back row")
    ap.add_argument("--out", help="JSON file to write (optional)")
    args = ap.parse_args()
    assert args.procs <= 7, "at most 7 jobs: keep one core free"
    jobs = [(c, r, args.episodes, not args.no_raider_escape, not args.no_held_touch, args.n_def,
             args.raider_phantom, args.diag_ratio,
             {**diag_kwargs(args.diag_range, args.diag_period), **start_kwargs(args.def_start), **margin_kwargs(args.return_margin), **hold_kwargs(args.hold)}, args.raider_rule)
            for c in args.ckpt for r in args.raiders.split(",")]
    with get_context("spawn").Pool(min(args.procs, len(jobs))) as pool:
        res = pool.map(run, jobs)
    print(f"\n  {'defence':<34}{'raider':<24}{'touch':>14}{'tackle':>14}{'foul':>12}"
          f"{'empty':>10}  held/timer/out  depth")
    for r in res:
        f = lambda k: f"{100 * r[k]['rate']:.1f} [{100 * r[k]['ci'][0]:.0f},{100 * r[k]['ci'][1]:.0f}]"
        h = r["tackle_how"]
        name = Path(r["ckpt"]).stem if r["ckpt"] != "random" else "random"
        d = f"{r['contact_depth_m']:.2f}" if r["contact_depth_m"] is not None else "-"
        print(f"  {name:<34}{r['raider']:<24}{f('touch'):>14}{f('tackle'):>14}"
              f"{100 * r['foul']['rate']:>11.1f}%{100 * r['empty']['rate']:>9.1f}%  "
              f"{h.get('held', 0)}/{h.get('timer', 0)}/{h.get('out_of_court', 0)}  {d}")
    if args.out:
        Path(args.out).write_text(json.dumps(res, indent=2, default=float))
        print(f"\nwritten to {args.out}")


if __name__ == "__main__":
    main()
