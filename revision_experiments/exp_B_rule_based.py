"""
Experiment B — Rule-based raider reference
===========================================

Addresses Reviewer 2, comment 1a:
    "Adding representative rule-based methods, additional RL algorithms such
     as SAC or TD3, and possibly multi-agent or self-play baselines would
     make the benchmark significantly more convincing."

Purpose
-------
Establishes a non-learned reference point. Without one, a reader has no basis
for judging whether the learned raider's touch rate is good: 43% could be
excellent or trivial depending on what a hand-coded controller achieves. This
experiment answers that.

Two scripted controllers are provided, both operating on the same observation
vector available to the trained policy, so the comparison is like for like:

  direct     Move at full speed toward the nearest defender until contact is
             registered, then move straight back toward the raider's half.
             The simplest sensible strategy; the intended floor.

  cautious   As direct, but abandons the approach and returns early if the
             remaining raid time falls below a threshold. Tests whether the
             commitment decision, which the reward function shapes explicitly,
             matters at all.

Neither controller learns, has parameters fitted to the task, or uses any
information the policy does not also receive.

Because these are deterministic given the environment, the only source of
variation is the randomised start position. Each controller is evaluated over
100 episodes per configuration, matching the protocol used for the learned
policies.

Usage
-----
    # the paper's headline configuration
    python3 revision_experiments/exp_B_rule_based.py --runs 3v3_reactive

    # the full 3x3 defender grid, both controllers
    python3 revision_experiments/exp_B_rule_based.py --all

    # quick check
    python3 revision_experiments/exp_B_rule_based.py --smoke

Runtime is a few minutes for the whole grid; there is no training.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from collections import Counter
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent
PROJECT = ROOT.parent
sys.path.insert(0, str(PROJECT))
sys.path.insert(0, str(ROOT))

from common import Registry, wilson_ci  # noqa: E402

N_EPISODES = 100
SEEDS = [42, 456, 123]
DEFENDER_COUNTS = [1, 2, 3]
DEFENDER_TYPES = ["static", "reactive", "wall"]

# Matches envs/kabaddi_env.py
MIDLINE_Y = 0.0
FSR_THRESH = 0.30
COURT_W = 2.6
COURT_L = 4.0

# cautious controller: abort the approach when this fraction of raid time
# remains. 0.45 leaves a little under half the raid for the return, which is
# roughly the time needed to cross the defender half at full speed.
ABORT_TIME_FRACTION = 0.45


def parse_obs(obs: np.ndarray, n_def: int) -> dict:
    """
    Unpack the observation vector.

    Layout (envs/kabaddi_env.py _obs):
        [0:3]                 raider x, y, theta
        [3:6]                 raider vx, vy, omega
        [6 : 6+2N]            defender positions relative to raider (dx, dy)
        [6+2N : 8+2N]         fsr left, right
        [8+2N]                normalised time remaining (1 -> 0)
        [9+2N], [10+2N]       midline crossed, baulk crossed
    """
    i = 6 + 2 * n_def
    return {
        "pos": obs[0:2],
        "theta": float(obs[2]),
        "vel": obs[3:5],
        "def_rel": obs[6:i].reshape(n_def, 2),
        "fsr": obs[i:i + 2],
        "t_remaining": float(obs[i + 2]),
        "midline": bool(obs[i + 3] > 0.5),
        "baulk": bool(obs[i + 4] > 0.5),
    }


class RuleBasedRaider:
    """
    Scripted raider. Holonomic world-frame velocity commands in [-1, 1].

    State machine:
        APPROACH  -> head for the nearest defender
        RETURN    -> head for the raider's own half

    The transition is triggered by simulated FSR contact, matching the
    environment's own touch condition. `cautious` adds a second trigger on
    remaining time.
    """

    def __init__(self, n_def: int, mode: str = "direct"):
        assert mode in ("direct", "cautious")
        self.n_def = n_def
        self.mode = mode
        self.reset()

    def reset(self):
        self.contacted = False
        self.returning = False

    def act(self, obs: np.ndarray) -> np.ndarray:
        s = parse_obs(obs, self.n_def)

        # Latch contact: the environment's touch flag is set on the first
        # frame the force threshold is exceeded, so the controller must
        # remember it rather than reading an instantaneous value.
        if float(np.max(s["fsr"])) > FSR_THRESH:
            self.contacted = True

        if self.contacted:
            self.returning = True
        elif self.mode == "cautious" and s["t_remaining"] < ABORT_TIME_FRACTION:
            self.returning = True

        if self.returning:
            direction = np.array([0.0, -1.0])          # straight home
        else:
            # nearest defender by relative position
            d = s["def_rel"]
            nearest = d[int(np.argmin(np.linalg.norm(d, axis=1)))]
            norm = float(np.linalg.norm(nearest))
            direction = nearest / norm if norm > 1e-6 else np.array([0.0, 1.0])

        # keep clear of the side walls
        x = float(s["pos"][0])
        margin = COURT_W / 2 - 0.15
        if x > margin and direction[0] > 0:
            direction[0] = -0.3
        elif x < -margin and direction[0] < 0:
            direction[0] = 0.3

        n = float(np.linalg.norm(direction))
        if n > 1e-6:
            direction = direction / n

        return np.array([direction[0], direction[1], 0.0], dtype=np.float32)


def evaluate(n_def: int, def_type: str, mode: str, seed: int,
             n_episodes: int) -> dict:
    from envs.kabaddi_env import RoboKabaddiEnv

    env = RoboKabaddiEnv(n_defenders=n_def, defender_type=def_type,
                         render=False)
    raider = RuleBasedRaider(n_def, mode)

    outcomes, durations = [], []
    for ep in range(n_episodes):
        obs, _ = env.reset(seed=seed * 10_000 + ep)
        raider.reset()
        steps = 0
        while True:
            action = raider.act(obs)
            obs, _, terminated, truncated, info = env.step(action)
            steps += 1
            if terminated or truncated:
                break
        outcomes.append(info.get("outcome", "empty"))
        durations.append(steps * env.dt)
    env.close()

    n = len(outcomes)
    counts = Counter(outcomes)
    res = {"n_episodes": n, "outcomes": outcomes,
           "mean_duration": float(np.mean(durations))}
    for k in ("touch", "tackle", "empty"):
        c = counts.get(k, 0)
        lo, hi = wilson_ci(c, n)
        res[f"{k}_count"] = c
        res[f"{k}_rate"] = c / n
        res[f"{k}_ci95"] = [lo, hi]
    return res


def main() -> None:
    ap = argparse.ArgumentParser(description="Experiment B — rule-based raider")
    ap.add_argument("--runs", default="3v3_reactive",
                    help="'3v3_reactive', 'all', or a substring filter")
    ap.add_argument("--all", action="store_true")
    ap.add_argument("--modes", default="direct,cautious")
    ap.add_argument("--episodes", type=int, default=N_EPISODES)
    ap.add_argument("--seeds", default="42,456,123")
    ap.add_argument("--smoke", action="store_true")
    args = ap.parse_args()

    modes = [m.strip() for m in args.modes.split(",")]
    seeds = [int(s) for s in args.seeds.split(",")]

    configs = [(n, dt) for n in DEFENDER_COUNTS for dt in DEFENDER_TYPES]
    if not args.all:
        configs = [(n, dt) for n, dt in configs
                   if args.runs in f"{n}v{n}_{dt}" or args.runs == "all"]
    if args.smoke:
        configs, modes, seeds, args.episodes = configs[:1], modes[:1], seeds[:1], 5

    if not configs:
        print(f"No configuration matched '{args.runs}'.")
        return

    print("Experiment B — rule-based raider reference")
    print(f"{len(configs)} config(s) x {len(modes)} controller(s) x "
          f"{len(seeds)} seed(s) x {args.episodes} episodes")
    print("=" * 84)
    print(f"{'config':<20}{'mode':<11}{'seed':>6}{'touch':>16}{'tackle':>15}{'empty':>15}")
    print("-" * 84)

    t0 = time.time()
    records = []
    for n_def, def_type in configs:
        for mode in modes:
            for seed in seeds:
                r = evaluate(n_def, def_type, mode, seed, args.episodes)
                r.update({"n_defenders": n_def, "defender_type": def_type,
                          "mode": mode, "seed": seed})
                records.append(r)

                def cell(k):
                    lo, hi = r[f"{k}_ci95"]
                    return f"{r[f'{k}_rate']*100:.1f} [{lo*100:.0f},{hi*100:.0f}]"

                print(f"{f'{n_def}v{n_def}_{def_type}':<20}{mode:<11}{seed:>6}"
                      f"{cell('touch'):>16}{cell('tackle'):>15}{cell('empty'):>15}")

    if args.smoke:
        print("\n[smoke] plumbing verified; nothing written.")
        return

    # ── Aggregate per config x mode, pooled across seeds ──────────────
    print("\n" + "=" * 84)
    print("POOLED ACROSS SEEDS — touch rate")
    print("=" * 84)
    print(f"{'config':<20}{'mode':<11}{'n':>7}{'touch':>18}")
    print("-" * 84)
    pooled = {}
    for n_def, def_type in configs:
        for mode in modes:
            sel = [r for r in records
                   if r["n_defenders"] == n_def and r["defender_type"] == def_type
                   and r["mode"] == mode]
            if not sel:
                continue
            t = sum(r["touch_count"] for r in sel)
            N = sum(r["n_episodes"] for r in sel)
            lo, hi = wilson_ci(t, N)
            key = f"{n_def}v{n_def}_{def_type}_{mode}"
            pooled[key] = {"touch": t, "n": N, "touch_rate": t / N,
                           "ci95": [lo, hi],
                           "tackle_rate": sum(r["tackle_count"] for r in sel) / N,
                           "empty_rate": sum(r["empty_count"] for r in sel) / N}
            print(f"{f'{n_def}v{n_def}_{def_type}':<20}{mode:<11}{N:>7}"
                  f"{f'{t/N*100:.1f} [{lo*100:.1f}, {hi*100:.1f}]':>18}")

    outpath = ROOT / "results" / "expB_rule_based.json"
    outpath.parent.mkdir(parents=True, exist_ok=True)
    outpath.write_text(json.dumps(
        {"episodes_per_run": args.episodes, "abort_time_fraction": ABORT_TIME_FRACTION,
         "pooled": pooled, "records": records}, indent=2, default=float))
    print(f"\nWritten to {outpath}")
    print(f"Elapsed {(time.time()-t0)/60:.1f} min")

    Registry().record(
        "expB_rule_based",
        config={"experiment": "B_rule_based", "modes": modes, "seeds": seeds,
                "episodes_per_run": args.episodes,
                "configs": [f"{n}v{n}_{dt}" for n, dt in configs]},
        results=pooled,
        notes="Non-learned scripted raider. Operates on the same observation "
              "vector as the trained policy. No training, no fitted "
              "parameters. Provides the reference floor for the learned "
              "policy's touch rate.",
    )
    print("Recorded as expB_rule_based")


if __name__ == "__main__":
    main()
