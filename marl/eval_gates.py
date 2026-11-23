"""
eval_gates.py — the learned raiders (C3) against the frozen gate set v6-v11
===========================================================================

exp_C1_evaluate.py is tied to results/c1; this plays any trained raider
(a C1 run or an alternating-training raider, results/marl/alt/rN_raider_sK)
against the scripted gate defenders, every style of the frozen set (v6 =
gate_style 'keepout', training gate; v7-v11 held-out), side by side.

Every raider is played under the same C3 rules (author, 28 Oct), so the C1
PPO starting point and the learned raiders are compared like for like:
held needs a defender touch, no boundary steering, defenders' fouls on
(the gates keep off the lines, so they hardly matter here). Best checkpoint
with its normalisation statistics, deterministic, seeded starts 50000 + i,
as exp_C1_evaluate.py. Writes one JSON list (one entry per raider x gate).

    python3 marl/eval_gates.py --raiders main_1v3_seed42_ext,results/marl/alt/r6_raider_s42 \\
        --episodes 1000 --out results/marl/alt/eval_gates_r6.json
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
from defender_env import FrozenRaider, ROOT, diag_kwargs, start_kwargs, margin_kwargs, hold_kwargs  # noqa: E402
sys.path.insert(0, str(ROOT / "revision_experiments"))
from common import wilson_ci                                          # noqa: E402
from envs.kabaddi_env import RoboKabaddiEnv, COURT_W, COURT_L         # noqa: E402

EVAL_SEED0 = 50_000
GATES = {"v6": "keepout", "v7": "v7", "v8": "v8", "v9": "v9", "v10": "v10", "v11": "v11"}
C3_RULES = dict(held_needs_touch=True, raider_edge_escape=False, defender_fouls=True)


def evaluate(job):
    raider, gate, n, n_def, *rest = job
    r = FrozenRaider(raider, rest[0] if rest else None, rest[3] if len(rest) > 3 else None)
    diag = rest[1] if len(rest) > 1 else None          # measured robots: B3, 11 Nov
    dkw = rest[2] if len(rest) > 2 else {}             # per-robot diagonal speed
    env = RoboKabaddiEnv(**dict(r.env_kwargs, defender_type="gate",
                                gate_style=GATES[gate], **C3_RULES,
                                **({"n_defenders": n_def} if n_def else {}),
                                **({"diagonal_speed_ratio": diag} if diag else {}), **dkw))
    outs, why, dur, depth = [], Counter(), [], []
    for ep in range(n):
        obs, _ = env.reset(seed=EVAL_SEED0 + ep)
        counted, info = False, {}
        while True:
            obs, _, term, trunc, info = env.step(r.act(env, obs))
            if env._flags["contact"] and not counted:
                counted = True
                depth.append(float(env._get_pos(env.raider_id)[1]))
            if term or trunc:
                break
        out = info.get("outcome", "empty")
        pos = env._get_pos(env.raider_id)
        outside = abs(pos[0]) > COURT_W / 2 or abs(pos[1]) > COURT_L / 2
        start = env._raid_start_step
        if out == "tackle":
            why["out" if outside else
                "held" if start is not None and env._step - start < env._raid_steps
                else "timer"] += 1
        outs.append(out)
        dur.append(env._step * env.dt)
    env.close()
    c = Counter(outs)
    res = {"raider": raider, "gate": gate, "gate_style": GATES[gate], "n": n,
           "n_def": env.n_def, "raider_phantom": r.phantom, **C3_RULES,
           "diagonal_speed_ratio": env.diagonal_speed_ratio,
           "diag_ratio_range": env.diag_ratio_range, "diag_ratio_period": dkw.get("diag_ratio_period"),
           "defender_start_depth": env.defender_start_depth, "return_margin": env.return_margin,
           "hold": [env.static_time, env.hold_dist, env.hold_touch_dist],
           "raider_rule": r.rule, "rule_changed_decisions": r.rule_changed,
           "tackle_how": dict(why),
           "contact_depth_m": float(np.mean(depth)) if depth else None,
           "median_raid_s": float(np.median(dur)), "outcomes": outs}
    for k in ("touch", "tackle", "foul", "empty"):
        lo, hi = wilson_ci(c[k], n)
        res[k] = {"rate": c[k] / n, "ci": [lo, hi], "count": c[k]}
    return res


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--raiders", required=True, help="comma list: C1 run names or paths")
    ap.add_argument("--gates", default=",".join(GATES), help="comma list of v6..v11")
    ap.add_argument("--episodes", type=int, default=1000)
    ap.add_argument("--procs", type=int, default=7)
    ap.add_argument("--n-def", type=int, default=None,
                    help="gate defenders on the court; default: as the raider was trained")
    ap.add_argument("--raider-phantom", choices=("deep", "copy"), default=None,
                    help="where the raiders' phantom defenders are; default: each raider's own")
    ap.add_argument("--diag-ratio", type=float, default=None,
                    help="diagonal speed / straight speed for every robot (B3 measured 0.58); "
                         "default: as each raider was trained (1.0)")
    ap.add_argument("--diag-range", default=None, help="every robot gets its own diagonal speed ratio per raid, uniform in LO,HI (B3: 0.35,0.85); default off")
    ap.add_argument("--diag-period", type=float, default=None, help="with --diag-range: new ratios every this many seconds; default once per raid")
    ap.add_argument("--raider-rule", choices=("deep",), default=None, help="rule on top of the trained raider (author, 13 Nov; defender_env.deep_rule): deep = go deep until the touch; default the policy alone")
    ap.add_argument("--return-margin", type=float, default=None, help="Empty on return only this many m behind the midline (author, 13 Nov: 0.03); default 0")
    ap.add_argument("--hold", default=None, help="held rule T,D,C: T s within D m with a defender centre within C m (author, 14 Nov: 2,0.25,0.25); default 4 s, 5 cm, body contact")
    ap.add_argument("--def-start", default=None, help="defenders start LO,HI metres past the midline (author, 13 Nov: 1.0,1.5 = 3.0-3.5 m from the raider's end line); default the back row")
    ap.add_argument("--out", required=True)
    args = ap.parse_args()
    assert args.procs <= 7, "at most 7 jobs: keep one core free"
    gates = args.gates.split(",")
    assert all(g in GATES for g in gates), f"gates must be among {list(GATES)}"
    jobs = [(r, g, args.episodes, args.n_def, args.raider_phantom, args.diag_ratio,
             {**diag_kwargs(args.diag_range, args.diag_period), **start_kwargs(args.def_start), **margin_kwargs(args.return_margin), **hold_kwargs(args.hold)}, args.raider_rule) for r in args.raiders.split(",") for g in gates]
    print(f"{len(jobs)} evaluations x {args.episodes} raids", flush=True)
    with get_context("spawn").Pool(min(args.procs, len(jobs))) as pool:
        results = pool.map(evaluate, jobs, chunksize=1)
    print(f"\n  {'raider':<34}{'gate':<5}{'touch':>15}{'tackle':>15}{'foul':>7}{'empty':>7}"
          f"  held/timer/out  depth")
    for x in results:
        f = lambda k: f"{100*x[k]['rate']:.1f} [{100*x[k]['ci'][0]:.0f},{100*x[k]['ci'][1]:.0f}]"
        h = x["tackle_how"]
        d = f"{x['contact_depth_m']:.2f}" if x["contact_depth_m"] is not None else "-"
        print(f"  {x['raider'][-34:]:<34}{x['gate']:<5}{f('touch'):>15}{f('tackle'):>15}"
              f"{100*x['foul']['rate']:6.1f}%{100*x['empty']['rate']:6.1f}%"
              f"  {h.get('held', 0)}/{h.get('timer', 0)}/{h.get('out', 0)}  {d}")
    Path(args.out).write_text(json.dumps(results, indent=1, default=float))
    print(f"\nwritten to {args.out}")


if __name__ == "__main__":
    main()
