"""
exp_C1_evaluate.py — evaluate the C1 policies (CORRECTIONS_TRACKER.md C1)
=========================================================================

For every finished run in results/c1/ (training/train_corrected.py):

  policy      the best checkpoint, loaded WITH the normalisation statistics
              saved at the same moment (best_vecnorm.pkl) — the T4 lesson.
              The PPO comparison point for the SAC/TD3 baselines is the
              ppo250k runs (same budget and the same two stages)
  episodes    1000, deterministic actions, seeded starts 50000 + i, so every
              number reproduces exactly
  conditions  own   — the environment it was trained in
              cross — main runs against the robots' pursuit defenders,
                      pursuit runs against the v6 gate
  modes (A3)  policy  — the policy drives the whole raid
              retreat — once contact is counted, the robots' scripted
                        retreat takes over (send_raider_retreat law, through
                        the hardware command mapping)

Reports touch / tackle / empty with Wilson 95% intervals, how each tackle
ended (held, 30 s timer, out of court) and each empty (never crossed, came
back, left before crossing), raid length and contact depth; pooled per study
and per seed. Writes results/c1/eval_results.json and registry entries.

Usage (from the robo_kabaddi/ folder)
-------------------------------------
    python3 revision_experiments/exp_C1_evaluate.py               # all finished runs
    python3 revision_experiments/exp_C1_evaluate.py --runs main_1v3_seed42_pilot --episodes 200
    python3 revision_experiments/exp_C1_evaluate.py --gate-style v7 --out eval_results_v7.json
        (every gate the policies meet, own or cross, is played as gate_style v7;
         no registry entries)
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from collections import Counter, defaultdict
from multiprocessing import Pool
from pathlib import Path

os.environ.setdefault("OMP_NUM_THREADS", "1")

import numpy as np                                              # noqa: E402

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT.parent))
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT.parent / "training"))

from common import wilson_ci, Registry                          # noqa: E402
from hardware_raider import retreat_intent                      # noqa: E402
from envs.kabaddi_env import RoboKabaddiEnv, COURT_W, COURT_L   # noqa: E402

C1 = ROOT.parent / "results" / "c1"
EVAL_SEED0 = 50_000


def load_policy(run_dir: Path, which: str):
    import torch
    from stable_baselines3 import PPO, SAC, TD3
    from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize
    torch.set_num_threads(1)
    info = json.loads((run_dir / "run_info.json").read_text())
    algo = info["spec"]["algo"]
    cls = {"ppo": PPO, "sac": SAC, "td3": TD3}[algo]
    if which == "best":
        model_p, vn_p = run_dir / "best_model" / "best_model.zip", run_dir / "best_vecnorm.pkl"
    else:                                   # "ckpt<N>": the checkpoint at N decisions
        n = which[4:]
        model_p = run_dir / "checkpoints" / f"{algo}_{n}_steps.zip"
        vn_p = run_dir / "checkpoints" / f"vecnorm_{n}_steps.pkl"
    env_kw = info["env_kwargs"]
    dummy = DummyVecEnv([lambda: RoboKabaddiEnv(**env_kw)])
    vn = VecNormalize.load(str(vn_p), dummy)
    vn.training, vn.norm_reward = False, False
    return cls.load(str(model_p), device="cpu"), vn, info


def evaluate(job):
    run, which, cond, mode, n, gate_style = job
    run_dir = C1 / run
    model, vn, info = load_policy(run_dir, which)
    env_kw = dict(info["env_kwargs"])
    if cond == "cross":
        env_kw["defender_type"] = ("physical" if env_kw["defender_type"] == "gate"
                                   else "gate")
    if gate_style:
        env_kw["gate_style"] = gate_style
    env = RoboKabaddiEnv(**env_kw)
    for k, v in info["spec"].get("reward", {}).items():
        setattr(env, k, v)
    outs, why, dur, depth = [], Counter(), [], []
    for ep in range(n):
        obs, _ = env.reset(seed=EVAL_SEED0 + ep)
        counted = False
        info_s = {}
        while True:
            if mode == "retreat" and env._flags["contact"]:
                x = float(env._get_pos(env.raider_id)[0])
                a = np.array([*retreat_intent(x), 0.0], dtype=np.float32)
            else:
                a, _ = model.predict(vn.normalize_obs(obs[None]).astype(np.float32)[0],
                                     deterministic=True)
            obs, _, term, trunc, info_s = env.step(a)
            if env._flags["contact"] and not counted:
                counted = True
                depth.append(float(env._get_pos(env.raider_id)[1]))
            if term or trunc:
                break
        out = info_s.get("outcome", "empty")
        pos = env._get_pos(env.raider_id)
        outside = abs(pos[0]) > COURT_W / 2 or abs(pos[1]) > COURT_L / 2
        start = env._raid_start_step
        if out == "tackle":
            why["tackle_out_of_court" if outside else
                "tackle_held" if (start is not None
                                  and env._step - start < env._raid_steps)
                else "tackle_timer"] += 1
        elif out == "empty":
            why["empty_left_before_crossing" if outside else
                "empty_never_crossed" if start is None else "empty_came_back"] += 1
        outs.append(out)
        dur.append(env._step * env.dt)
    env.close()
    c, d, o = Counter(outs), np.array(dur), np.array(outs)
    res = {"run": run, "policy": which, "condition": cond, "mode": mode, "n": n,
           "defender_type": env_kw["defender_type"],
           "gate_style": env_kw.get("gate_style", "keepout"), **dict(why),
           "median_touch_s": float(np.median(d[o == "touch"])) if c["touch"] else None,
           "median_tackle_s": float(np.median(d[o == "tackle"])) if c["tackle"] else None,
           "contact_depth_m": float(np.mean(depth)) if depth else None,
           "outcomes": outs}
    for k in ("touch", "tackle", "empty"):
        lo, hi = wilson_ci(c[k], n)
        res[k] = {"rate": c[k] / n, "ci": [lo, hi], "count": c[k]}
    return res


def jobs_for(run: str, n: int, which: str = "best", gate_style=None):
    name = run.split("_seed")[0]
    js = [(run, which, "own", m, n, gate_style) for m in ("policy", "retreat")]
    # main's cross is against pursuit (no gate): nothing to replay with a style
    if name.startswith("pursuit") or (name.startswith("main") and not gate_style):
        js.append((run, "best", "cross", "policy", n, gate_style))
    return js


def pooled(rows):
    n = sum(r["n"] for r in rows)
    out = {"n": n, "seeds": len(rows)}
    for k in ("touch", "tackle", "empty"):
        c = sum(r[k]["count"] for r in rows)
        lo, hi = wilson_ci(c, n)
        out[k] = {"rate": c / n, "ci": [lo, hi], "count": c}
    for k in ("tackle_held", "tackle_timer", "tackle_out_of_court",
              "empty_never_crossed", "empty_came_back", "empty_left_before_crossing"):
        out[k] = sum(r.get(k, 0) for r in rows)
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--runs", nargs="*")
    ap.add_argument("--episodes", type=int, default=1000)
    ap.add_argument("--procs", type=int, default=7)
    ap.add_argument("--no-registry", action="store_true")
    ap.add_argument("--ckpt", type=int,
                    help="evaluate the checkpoint at this many decisions, not the best")
    ap.add_argument("--gate-style", choices=("keepout", "v5", "v7", "v8", "v9", "v10", "v11"),
                    help="play every gate as this style (default: as trained)")
    ap.add_argument("--out", default="eval_results.json",
                    help="file name inside results/c1/")
    args = ap.parse_args()
    assert args.procs <= 7, "at most 7 jobs: keep one core free"
    runs = args.runs or sorted(p.name for p in C1.iterdir()
                               if (p / "final_model.zip").exists()
                               and not p.name.endswith("pilot"))
    which = f"ckpt{args.ckpt}" if args.ckpt else "best"
    jobs = [j for r in runs for j in jobs_for(r, args.episodes, which, args.gate_style)]
    print(f"{len(runs)} runs, {len(jobs)} evaluations x {args.episodes} raids", flush=True)
    with Pool(min(args.procs, len(jobs))) as pool:
        results = pool.map(evaluate, jobs)

    groups = defaultdict(list)
    for r in results:
        groups[(r["run"].split("_seed")[0], r["policy"], r["condition"], r["mode"])].append(r)
    summary = {}
    print(f"\n  {'study':<26}{'policy':<9}{'vs':<9}{'mode':<8}{'seeds':>5}"
          f"{'touch':>16}{'tackle':>16}{'empty':>16}  held/timer/out")
    for key in sorted(groups):
        p = pooled(groups[key])
        summary["|".join(key)] = p
        dt = groups[key][0]["defender_type"]
        f = lambda k: f"{100*p[k]['rate']:.1f} [{100*p[k]['ci'][0]:.0f},{100*p[k]['ci'][1]:.0f}]"
        print(f"  {key[0]:<26}{key[1]:<9}{dt:<9}{key[3]:<8}{p['seeds']:>5}"
              f"{f('touch'):>16}{f('tackle'):>16}{f('empty'):>16}  "
              f"{p['tackle_held']}/{p['tackle_timer']}/{p['tackle_out_of_court']}")
    dest = C1 / args.out
    dest.write_text(json.dumps({"episodes": args.episodes, "eval_seed0": EVAL_SEED0,
                                "per_run": results, "pooled": summary},
                               indent=2, default=float))
    print(f"\nwritten to {dest}")
    if not args.no_registry and not args.gate_style:
        reg = Registry()
        for r in results:
            if r["policy"] == "best" and r["condition"] == "own" and r["mode"] == "policy":
                info = json.loads((C1 / r["run"] / "run_info.json").read_text())
                flat = {"n_episodes": r["n"]}
                for k in ("touch", "tackle", "empty"):
                    flat[f"{k}_rate"] = r[k]["rate"]
                    flat[f"{k}_ci95"] = r[k]["ci"]
                    flat[f"{k}_count"] = r[k]["count"]
                reg.record(f"c1_{r['run']}", info, flat,
                           notes="C1 on CORRECTED_CONFIG; best checkpoint with its "
                                 "vecnorm; seeded raids")


if __name__ == "__main__":
    main()
