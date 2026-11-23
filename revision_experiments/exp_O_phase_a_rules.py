"""
Experiment O — Phase A rules: static-raider Tackle and contact counting
=======================================================================

Adds the physical system's rules to the simulation one at a time and
measures each with real PyBullet (CORRECTIONS_TRACKER.md, A1, A2, A4):

    base       outcome_rules='section_2_3', force contact (Experiment N)
    +A1        outcome_rules='physical': raider < 5 cm in 7 s in the
               defenders' half = Tackle
    +A2 0.27   contact counts only with the raider > 0.27 m past the midline,
               after 2 control steps (cumulative) — the code's 0.27 sim units
    +A2 0.20   the same at 0.20 m — the code's value on the real court
    +A4 0.25   camera contact, centres within 0.25 m (paper value), with A2 0.27
    +A4 0.35   camera contact, centres within 0.35 m (code value), with A2 0.27

Both values of each open parameter are shown; neither is chosen here.
Everything else is fixed: planar locomotion, 3 'physical' defenders in the
back row, the same seeds.

The raider is the Experiment M scripted raider (straight at the nearest
defender, straight home), with one change: it turns for home when the
environment has COUNTED the contact, as the physical raider's retreat was
triggered by confirmed contact (run_experiment.py ~687). Otherwise it would
turn back on contacts the rules do not count.

Usage (from the robo_kabaddi/ folder)
-------------------------------------
    python3 revision_experiments/exp_O_phase_a_rules.py
    python3 revision_experiments/exp_O_phase_a_rules.py --episodes 300 --procs 6
Writes revision_experiments/results/expO_phase_a_rules.json
"""

from __future__ import annotations

import argparse
import json
import sys
from collections import Counter
from multiprocessing import Pool
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent
PROJECT = ROOT.parent
sys.path.insert(0, str(PROJECT))
sys.path.insert(0, str(ROOT))

from common import wilson_ci                       # noqa: E402
from exp_B_rule_based import RuleBasedRaider        # noqa: E402
from envs.kabaddi_env import RoboKabaddiEnv, MIDLINE_Y   # noqa: E402

N_DEF = 3
I_FSR = 6 + 2 * N_DEF
A2_27 = dict(contact_min_depth=0.27, contact_confirm_steps=2)
CONFIGS = {
    "base":     dict(outcome_rules="section_2_3"),
    "+A1":      dict(outcome_rules="physical"),
    "+A2 0.27": dict(outcome_rules="physical", **A2_27),
    "+A2 0.20": dict(outcome_rules="physical", contact_min_depth=0.20,
                     contact_confirm_steps=2),
    "+A4 0.25": dict(outcome_rules="physical", contact_model="camera",
                     contact_dist=0.25, **A2_27),
    "+A4 0.35": dict(outcome_rules="physical", contact_model="camera",
                     contact_dist=0.35, **A2_27),
}
COMMON = dict(n_defenders=N_DEF, defender_type="physical",
              start_layout="back_row", locomotion="planar")


class CountedContactRaider(RuleBasedRaider):
    """Experiment M raider that turns home on the environment's counted
    contact instead of its own contact reading."""

    def act_env(self, env, obs):
        o = obs.copy()
        o[I_FSR:I_FSR + 2] = 0.0
        self.contacted = bool(env._flags["contact"])
        return self.act(o)


def rates(outs):
    n, c = len(outs), Counter(outs)
    res = {}
    for k in ("touch", "tackle", "empty"):
        lo, hi = wilson_ci(c[k], n)
        res[k] = {"rate": c[k] / n, "ci": [lo, hi], "count": c[k]}
    return res


def run(args):
    name, n = args
    env = RoboKabaddiEnv(**COMMON, **CONFIGS[name])
    raider = CountedContactRaider(N_DEF)
    outs, dur = [], []
    tk_static = tk_timer = em_never = em_home = 0
    raw_eps = counted_eps = 0
    t_count, depth_count = [], []
    for ep in range(n):
        obs, _ = env.reset(seed=1000 + ep)
        raider.reset()
        raw = False
        counted_at = None
        info = {}
        while True:
            obs, _, term, trunc, info = env.step(raider.act_env(env, obs))
            raw |= bool(max(obs[I_FSR], obs[I_FSR + 1]) > env.FSR_THRESH)
            if env._flags["contact"] and counted_at is None:
                counted_at = env._step
                depth_count.append(float(env._get_pos(env.raider_id)[1]))
            if term or trunc:
                break
        out = info.get("outcome", "empty")
        outs.append(out)
        dur.append(env._step * env.dt)
        raw_eps += raw
        counted_eps += counted_at is not None
        if counted_at is not None:
            t_count.append(counted_at * env.dt)
        start = env._raid_start_step
        if out == "tackle":
            if start is not None and env._step - start < env._raid_steps:
                tk_static += 1
            else:
                tk_timer += 1
        elif out == "empty":
            if start is None:
                em_never += 1
            else:
                em_home += 1
    env.close()
    return name, {"n": n, "kwargs": CONFIGS[name], **rates(outs),
                  "tackle_static": tk_static, "tackle_timer": tk_timer,
                  "empty_never_crossed": em_never, "empty_came_home": em_home,
                  "raw_contact_episodes": raw_eps,
                  "counted_contact_episodes": counted_eps,
                  "time_to_counted_contact_s":
                      float(np.mean(t_count)) if t_count else None,
                  "raider_y_at_counted_contact_m":
                      float(np.mean(depth_count)) if depth_count else None,
                  "raid_duration_s": float(np.mean(dur)),
                  "outcomes": outs}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--episodes", type=int, default=300)
    ap.add_argument("--procs", type=int, default=6,
                    help="parallel worker processes (leave a core free)")
    args = ap.parse_args()

    with Pool(min(args.procs, len(CONFIGS))) as pool:
        res = dict(pool.map(run, [(k, args.episodes) for k in CONFIGS]))
    out = {"episodes": args.episodes, "common": COMMON,
           "raider": "CountedContactRaider (Exp M raider, turns on counted "
                     "contact)", "seeds": "1000 + episode",
           "configs": {k: res[k] for k in CONFIGS}}

    R, names, w = out["configs"], list(CONFIGS), 15

    def row(label, f):
        print(f"  {label:<30}" + "".join(f"{f(R[k]):>{w}}" for k in names))

    def pct(e, k):
        return f"{100*e[k]['rate']:.0f}% [{100*e[k]['ci'][0]:.0f},{100*e[k]['ci'][1]:.0f}]"

    print("\n" + "=" * (30 + w * len(names)))
    print(f"scripted raider vs 3 physical back-row defenders, planar, "
          f"{args.episodes} episodes per column")
    print("=" * (30 + w * len(names)))
    print(f"  {'':<30}" + "".join(f"{k:>{w}}" for k in names))
    for k in ("touch", "tackle", "empty"):
        row(f"{k} [95% CI]", lambda e, k=k: pct(e, k))
    row("  tackle: static 7 s", lambda e: str(e["tackle_static"]))
    row("  tackle: 30 s timer", lambda e: str(e["tackle_timer"]))
    row("  empty: never crossed", lambda e: str(e["empty_never_crossed"]))
    row("  empty: came home", lambda e: str(e["empty_came_home"]))
    row("raw contact episodes", lambda e: str(e["raw_contact_episodes"]))
    row("counted contact episodes", lambda e: str(e["counted_contact_episodes"]))
    row("time to counted contact", lambda e: "-" if e["time_to_counted_contact_s"]
        is None else f"{e['time_to_counted_contact_s']:.1f} s")
    row("raider depth at counted", lambda e: "-" if e["raider_y_at_counted_contact_m"]
        is None else f"{e['raider_y_at_counted_contact_m']:.2f} m")
    row("raid duration (mean)", lambda e: f"{e['raid_duration_s']:.1f} s")

    dest = ROOT / "results" / "expO_phase_a_rules.json"
    dest.parent.mkdir(exist_ok=True)
    dest.write_text(json.dumps(
        out, indent=2,
        default=lambda o: o.item() if hasattr(o, 'item') else str(o)))
    print(f"\nwritten to {dest}")


if __name__ == "__main__":
    main()
