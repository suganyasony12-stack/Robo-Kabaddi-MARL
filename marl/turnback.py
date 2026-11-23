"""
turnback.py — how deep the raider goes and when it turns home (author, 12 Nov)
==============================================================================

For each raider x gate: per raid the deepest point past the midline, the time
it was reached, the nearest defender then, and the raid length, with medians
by result. Same raids as eval_gates.py (seeds 50000 + i, deterministic).

    python3 marl/turnback.py --raiders A,B --gates v8,v10 --episodes 100 \\
        --n-def 3 --diag-range 0.35,0.85 --diag-period 3 --out x.json
"""
from __future__ import annotations

import argparse
import json
import os
import sys
from multiprocessing import get_context
from pathlib import Path

os.environ.setdefault("OMP_NUM_THREADS", "1")
import numpy as np                                                    # noqa: E402

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
from defender_env import FrozenRaider, diag_kwargs, start_kwargs, margin_kwargs, hold_kwargs  # noqa: E402
from eval_gates import C3_RULES, GATES, EVAL_SEED0                    # noqa: E402
from envs.kabaddi_env import RoboKabaddiEnv                           # noqa: E402


def run(job):
    raider, gate, n, n_def, phantom, dkw, rule = job
    r = FrozenRaider(raider, phantom, rule)
    env = RoboKabaddiEnv(**dict(r.env_kwargs, defender_type="gate", gate_style=GATES[gate],
                                **C3_RULES, **({"n_defenders": n_def} if n_def else {}), **dkw))
    rows = []
    for ep in range(n):
        obs, _ = env.reset(seed=EVAL_SEED0 + ep)
        best, t_best, near, info = -9.0, 0.0, 0.0, {}
        while True:
            obs, _, term, trunc, info = env.step(r.act(env, obs))
            p = env._get_pos(env.raider_id)
            if p[1] > best:
                best, t_best = float(p[1]), env._step * env.dt
                near = float(min(np.hypot(*(env._get_pos(d)[:2] - p[:2])) for d in env.defender_ids))
            if term or trunc:
                break
        rows.append({"outcome": info.get("outcome", "empty"), "deepest_m": best,
                     "deepest_at_s": t_best, "nearest_def_m": near, "raid_s": env._step * env.dt})
    env.close()
    by = {}
    for k in ("touch", "tackle", "foul", "empty"):
        sel = [x for x in rows if x["outcome"] == k]
        if sel:
            by[k] = {"n": len(sel), **{f: float(np.median([x[f] for x in sel]))
                                        for f in ("deepest_m", "deepest_at_s", "nearest_def_m", "raid_s")}}
    return {"raider": raider, "gate": gate, "n": n, "touch_obs": r.touch, "raider_rule": rule,
            "diag": dkw, "by_result": by, "raids": rows}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--raiders", required=True)
    ap.add_argument("--gates", default="v8,v10")
    ap.add_argument("--episodes", type=int, default=100)
    ap.add_argument("--n-def", type=int, default=None)
    ap.add_argument("--raider-phantom", choices=("deep", "copy"), default=None)
    ap.add_argument("--diag-range", default=None)
    ap.add_argument("--diag-period", type=float, default=None)
    ap.add_argument("--raider-rule", choices=("deep",), default=None, help="rule on top of the trained raider (author, 13 Nov; defender_env.deep_rule): deep = go deep until the touch; default the policy alone")
    ap.add_argument("--return-margin", type=float, default=None, help="Empty on return only this many m behind the midline (author, 13 Nov: 0.03); default 0")
    ap.add_argument("--hold", default=None, help="held rule T,D,C: T s within D m with a defender centre within C m (author, 14 Nov: 2,0.25,0.25); default 4 s, 5 cm, body contact")
    ap.add_argument("--def-start", default=None, help="defenders start LO,HI m past the midline; default back row")
    ap.add_argument("--procs", type=int, default=7)
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    assert a.procs <= 7, "at most 7 jobs: keep one core free"
    dkw = {**diag_kwargs(a.diag_range, a.diag_period), **start_kwargs(a.def_start), **margin_kwargs(a.return_margin), **hold_kwargs(a.hold)}
    jobs = [(r, g, a.episodes, a.n_def, a.raider_phantom, dkw, a.raider_rule)
            for r in a.raiders.split(",") for g in a.gates.split(",")]
    with get_context("spawn").Pool(min(a.procs, len(jobs))) as pool:
        res = pool.map(run, jobs, chunksize=1)
    print(f"  {'raider':<34}{'gate':<5}{'result':<8}{'n':>4}  deepest  at s  nearest def  raid s")
    for x in res:
        for k, v in x["by_result"].items():
            print(f"  {x['raider'][-34:]:<34}{x['gate']:<5}{k:<8}{v['n']:>4}  {v['deepest_m']:6.2f}"
                  f"  {v['deepest_at_s']:4.1f}  {v['nearest_def_m']:10.2f}  {v['raid_s']:6.1f}")
    Path(a.out).write_text(json.dumps(res, indent=1, default=float))
    print(f"\nwritten to {a.out}")


if __name__ == "__main__":
    main()
