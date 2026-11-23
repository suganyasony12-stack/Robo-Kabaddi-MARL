"""
raider_vs_learned.py — the raider learns against frozen learned defenders (C3, stage 3)
=======================================================================================

Continues a trained C1 raider (its best model with the normalisation statistics
saved with it, the raider the defenders were trained against) against one or
more frozen learned defender teams: a pool, one team drawn per raid. Same PPO
settings (kept in the saved model), the same 4 copies of the game and the
raider's own reward, unchanged. The author's rules: defender fouls on, the
out rule for the raider, and NO boundary escape: a learning raider learns the
lines itself (author, 27 Oct). The defenders take their most likely action.
--ent sets the exploration bonus (default: the saved model's, 0.02 from C1);
the author set 0 from round 4 on (28 Oct): with actions clipped to +-1 the
bonus made the action spread grow every round.
--empty sets the raider's reward for an empty raid (default: the env's R_EMPTY,
-2.0); the author set -6 on 29 Oct: with +1 (midline) and +1.5 (baulk) the
raider netted +0.5 by crossing the baulk line and walking home, so it
declined raids; with -6 an empty raid is worse than a tackle (-5).
--walkout sets the raider's reward for a walk-out: over a line (scored as a
tackle) with no defender touching it in the last WALKOUT_STEPS decisions
(default: scored as any tackle, R_TACKLE -5). The author closed this loophole
on 29 Oct: with the empty penalty, walking out (-5) had become cheaper than
going home empty, so the raider walked out of hopeless raids. A tackle with
contact (pushed out, held) keeps R_TACKLE. The outcome is unchanged (tackle);
only the raider's training reward differs.
--touch-obs gives the raider one more input, the touch-made flag (author,
12 Nov; defender_env.raider_input): the raiders turned home by the clock
(7-8 s), not after the touch. Starting from a raider without it, the network
is widened: the new input gets weight 0 in the first layer of the action
and value networks, and normalisation mean 0, variance 1, so the widened
raider plays exactly as the old one until it learns to use the flag (Adam
starts fresh). Saved as run_info "raider_touch_obs"; default off.
--fresh (author, 13 Nov): a new raider network instead of continuing --raider
(the widened raiders had not learned to use the touch flag after 2M
decisions): PPO with the C1 settings (training/train_corrected.PPO_KWARGS,
net 256 x 256), new normalisation as C1 (observations and rewards, clip 10);
--raider gives only its phantom setting and seed. With --touch-obs the flag
is an input from the first decision. Default off.
--def-start LO,HI (author, 13 Nov): defenders start LO-HI metres past the
midline (RoboKabaddiEnv defender_start_depth; 1.0,1.87 = from 3.0 m to the
back row); default the back row.

    python3 marl/raider_vs_learned.py --raider main_1v3_seed42_ext \\
        --defence results/marl/a/checkpoints/x.pt,results/marl/b/checkpoints/y.pt \\
        --steps 1000000 --out results/marl/raider_ppo42_vs_mappo

Writes as training/train_corrected.py does: best_model/ with best_vecnorm.pkl
(best by 20 evaluation raids against the same defence pool), checkpoints/
with their vecnorm, final_model.zip, vecnorm.pkl, metrics.csv (touch / tackle /
foul / empty rates every 10k decisions) and run_info.json with env_kwargs, so
exp_C1_evaluate.load_policy and defender_env.FrozenRaider can load the result.
"""
from __future__ import annotations

import argparse
import csv
import json
import os
import sys
import time
from pathlib import Path

os.environ.setdefault("OMP_NUM_THREADS", "1")
import gymnasium as gym                                               # noqa: E402
import numpy as np                                                    # noqa: E402

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
from defender_env import (C1, ROOT, raider_input, pad_defender_obs,  # noqa: E402
                          touch_space, start_kwargs)
from eval_defenders import load                                       # noqa: E402
sys.path.insert(0, str(ROOT / "training"))
from envs.kabaddi_env import (RoboKabaddiEnv, CORRECTED_CONFIG,      # noqa: E402
                               COURT_W, COURT_L)

N_ENVS = 4                    # as the C1 PPO runs (N_ENVS_PPO)
WALKOUT_STEPS = 5             # decisions (1 s at 5 Hz) with no defender touching


def env_kwargs(n_def=3):
    return dict(CORRECTED_CONFIG, n_defenders=n_def, defender_type="learned",
                defender_fouls=True, raider_edge_escape=False,
                held_needs_touch=True)       # author, 28 Oct


class LearnedDefenceEnv(gym.Env):
    """The raider's game against frozen learned defenders (a pool: one team
    per raid, drawn from a generator seeded at the first reset)."""

    def __init__(self, defences, n_def=3, r_empty=None, r_walkout=None, pad_to=None,
                 phantom="deep", diag=None, touch=False):
        import torch
        torch.set_num_threads(1)
        # diag: per-robot diagonal speed (diag_kwargs, author 11 Nov); kept out of
        # env_kwargs / run_info["env_kwargs"], so tests choose it themselves
        self.env = RoboKabaddiEnv(**env_kwargs(n_def), **(diag or {}))
        if r_empty is not None:
            self.env.R_EMPTY = r_empty
        self.r_walkout = r_walkout
        self._touch = []
        # raider input for pad_to defenders, phantoms for the missing ones (author, 4 Nov)
        self.n_in = max(n_def, pad_to or n_def)
        self.phantom = phantom
        self.observation_space = gym.spaces.Box(-np.inf, np.inf, (11 + 2 * self.n_in,),
                                                np.float32) if self.n_in > n_def \
            else self.env.observation_space
        self.touch = touch                   # touch-made flag as one more input (12 Nov)
        if touch:
            self.observation_space = touch_space(self.observation_space)
        self.action_space = self.env.action_space
        self.defences = [load(c) for c in defences]
        self._rng = np.random.default_rng(0)
        self._act = self.defences[0]

    def reset(self, seed=None, options=None):
        if seed is not None:
            self._rng = np.random.default_rng(seed)
        self._act = self.defences[int(self._rng.integers(len(self.defences)))]
        self._touch = []
        obs, info = self.env.reset(seed=seed)
        return raider_input(self.env, obs, self.n_in, self.phantom, self.touch), info

    def step(self, action):
        e = self.env
        e.set_defender_actions(self._act(pad_defender_obs(e, e.defender_obs(),
                                                          getattr(self._act, "n_in", 0))))
        obs, r, term, trunc, info = e.step(action)
        obs = raider_input(e, obs, self.n_in, self.phantom, self.touch)
        self._touch.append(e._defender_touching())
        if term or trunc:
            pos = e._get_pos(e.raider_id)
            info["walkout"] = (info.get("outcome") == "tackle"
                               and (abs(pos[0]) > COURT_W / 2 or abs(pos[1]) > COURT_L / 2)
                               and not any(self._touch[-WALKOUT_STEPS:]))
            if info["walkout"] and self.r_walkout is not None:
                r += self.r_walkout - e.R_TACKLE
        return obs, r, term, trunc, info

    def close(self):
        self.env.close()


def make_fn(defences, seed, r_empty=None, r_walkout=None, n_def=3, pad_to=None, phantom="deep",
            diag=None, touch=False):
    from stable_baselines3.common.monitor import Monitor

    def _init():
        env = Monitor(LearnedDefenceEnv(defences, n_def, r_empty=r_empty, r_walkout=r_walkout,
                                        pad_to=pad_to, phantom=phantom, diag=diag, touch=touch),
                      info_keywords=("outcome", "walkout"))
        env.reset(seed=seed)
        return env
    return _init


WIDEN_KEYS = ("mlp_extractor.policy_net.0.weight", "mlp_extractor.value_net.0.weight")


def widened_raider(src, venv):
    """--touch-obs from a raider without the flag (author, 12 Nov): its best
    model and normalisation with one more input (the last), weight 0 and
    mean 0 / variance 1, so it plays exactly as before. venv: the training
    games, already with the flag. Returns (VecNormalize, PPO)."""
    import pickle
    import torch
    from stable_baselines3 import PPO
    with open(src / "best_vecnorm.pkl", "rb") as fh:
        vn = pickle.load(fh)
    n_old = vn.obs_rms.mean.shape[0]
    assert venv.observation_space.shape == (n_old + 1,), (venv.observation_space, n_old)
    vn.obs_rms.mean = np.append(vn.obs_rms.mean, 0.0)
    vn.obs_rms.var = np.append(vn.obs_rms.var, 1.0)
    vn.observation_space = venv.observation_space
    vn.set_venv(venv)
    model = PPO.load(str(src / "best_model" / "best_model.zip"), device="cpu")
    sd = {k: v.clone() for k, v in model.policy.state_dict().items()}
    for k in WIDEN_KEYS:
        w = sd[k]
        assert w.shape[1] == n_old, (k, w.shape)
        sd[k] = torch.cat([w, torch.zeros(w.shape[0], 1, dtype=w.dtype)], dim=1)
    model.observation_space = venv.observation_space
    model._setup_model()                 # new network and buffers for the larger input
    model.policy.load_state_dict(sd)     # strict: every other weight as saved
    model.set_env(vn)
    return vn, model


def main():
    from stable_baselines3 import PPO
    from stable_baselines3.common.callbacks import (BaseCallback, CallbackList,
                                                    EvalCallback)
    from stable_baselines3.common.env_util import make_vec_env
    from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize
    from train_corrected import CheckpointWithVecNorm, SaveVecNorm

    ap = argparse.ArgumentParser()
    ap.add_argument("--raider", required=True,
                    help="a C1 PPO run in results/c1/, or a run folder path")
    ap.add_argument("--defence", required=True,
                    help="learned defender checkpoint(s), separated by commas")
    ap.add_argument("--steps", type=int, default=1_000_000)
    ap.add_argument("--seed", type=int, default=None, help="default: the raider's seed")
    ap.add_argument("--ent", type=float, default=None,
                    help="exploration bonus (ent_coef); default: the saved model's")
    ap.add_argument("--empty", type=float, default=None,
                    help="raider's reward for an empty raid; default: the env's R_EMPTY")
    ap.add_argument("--walkout", type=float, default=None,
                    help="raider's reward for walking out (no defender touching); "
                         "default: R_TACKLE, as any tackle")
    ap.add_argument("--n-def", type=int, default=3, help="defenders on the court")
    ap.add_argument("--pad-to", type=int, default=None,
                    help="raider input for this many defenders (phantoms for the missing "
                         "ones, defender_env.PHANTOM_POS); default: --n-def")
    ap.add_argument("--raider-phantom", choices=("deep", "copy"), default=None,
                    help="where the raider's phantom defenders are (defender_env."
                         "pad_raider_obs); default: the starting raider's setting, else deep")
    ap.add_argument("--diag-range", default=None, help="every robot gets its own diagonal speed ratio per raid, uniform in LO,HI (B3: 0.35,0.85); default off")
    ap.add_argument("--diag-period", type=float, default=None, help="with --diag-range: new ratios every this many seconds; default once per raid")
    ap.add_argument("--touch-obs", action="store_true",
                    help="touch-made flag as one more raider input (author, 12 Nov); a raider "
                         "without it is widened first; default: the starting raider's setting")
    ap.add_argument("--fresh", action="store_true",
                    help="new raider network (C1 PPO settings) instead of continuing --raider; "
                         "author, 13 Nov; default off")
    ap.add_argument("--def-start", default=None,
                    help="defenders start LO,HI metres past the midline (author, 13 Nov); "
                         "default the back row")
    ap.add_argument("--out", required=True)
    args = ap.parse_args()
    from defender_env import diag_kwargs
    diag = {**diag_kwargs(args.diag_range, args.diag_period), **start_kwargs(args.def_start)}
    src = Path(args.raider) if "/" in args.raider else C1 / args.raider
    info = json.loads((src / "run_info.json").read_text())
    phantom = args.raider_phantom or info.get("raider_phantom") or "deep"
    assert info["spec"]["algo"] == "ppo", "PPO raiders only"
    seed = args.seed if args.seed is not None else info["spec"]["seed"]
    defences = args.defence.split(",")
    out = Path(args.out)
    (out / "checkpoints").mkdir(parents=True, exist_ok=True)

    class Outcomes(BaseCallback):
        """Training raids' touch / tackle / foul / empty rates every 10k decisions."""

        def __init__(self, path, start):
            super().__init__()
            self.fh = open(path, "w", newline="")
            self.w = csv.writer(self.fh)
            self.w.writerow(["timestep", "touch_rate", "tackle_rate", "foul_rate",
                             "empty_rate", "episodes", "elapsed_s", "walkout_rate"])
            self.buf, self.walk, self.next, self.t0 = [], [], start + 10_000, time.time()

        def _on_step(self):
            for inf, done in zip(self.locals["infos"], self.locals["dones"]):
                if done and "episode" in inf:
                    self.buf.append(inf.get("outcome", "empty"))
                    self.walk.append(bool(inf.get("walkout")))
            if self.num_timesteps >= self.next and self.buf:
                n = len(self.buf)
                self.w.writerow([self.num_timesteps,
                                 *(round(self.buf.count(k) / n, 4)
                                   for k in ("touch", "tackle", "foul", "empty")),
                                 n, round(time.time() - self.t0, 1),
                                 round(sum(self.walk) / n, 4)])
                self.fh.flush()
                self.buf, self.walk, self.next = [], [], self.next + 10_000
            return True

        def _on_training_end(self):
            self.fh.close()

    src_touch = bool(info.get("raider_touch_obs", False)) and not args.fresh
    touch = args.touch_obs or src_touch
    widen = touch and not src_touch and not args.fresh
    venv = make_vec_env(make_fn(defences, seed + 11, args.empty, args.walkout, args.n_def, args.pad_to, phantom, diag, touch), n_envs=N_ENVS, seed=seed + 11,
                        vec_env_cls=DummyVecEnv)
    if args.fresh:                                   # author, 13 Nov
        from train_corrected import PPO_KWARGS
        train_env = VecNormalize(venv, norm_obs=True, norm_reward=True, clip_obs=10.0)
        model = PPO("MlpPolicy", train_env, seed=seed, verbose=0, device="cpu", **PPO_KWARGS)
    elif widen:
        train_env, model = widened_raider(src, venv)
    else:
        train_env = VecNormalize.load(str(src / "best_vecnorm.pkl"), venv)
        model = PPO.load(str(src / "best_model" / "best_model.zip"), env=train_env,
                         device="cpu")
    train_env.training = True
    if args.seed is not None:
        # PPO.load re-seeds everything with the saved model's seed; use --seed instead
        model.seed = seed
        model.set_random_seed(seed)
    if args.ent is not None:
        model.ent_coef = args.ent
    eval_env = make_vec_env(make_fn(defences, seed + 999, args.empty, args.walkout, args.n_def, args.pad_to, phantom, diag, touch), n_envs=1)
    eval_env = VecNormalize(eval_env, norm_obs=True, norm_reward=False, training=False)
    start = int(model.num_timesteps)
    run_info = {"run": out.name, "spec": {**info["spec"], "env": {}},
                "continued_from": None if args.fresh else args.raider,
                "from_model": "new network" if args.fresh else "best",
                "fresh": bool(args.fresh), "settings_from": args.raider,
                "start_decisions": start, "extra_decisions": args.steps,
                "env_kwargs": env_kwargs(args.n_def), "defences": defences, "n_envs": N_ENVS,
                "pad_to": args.pad_to if args.pad_to and args.pad_to > args.n_def else None,
                "raider_phantom": phantom,
                "raider_touch_obs": touch, "widened_from": args.raider if widen else None,
                "diag_ratio_range": diag.get("diag_ratio_range"),
                "diag_ratio_period": diag.get("diag_ratio_period"),
                "defender_start_depth": diag.get("defender_start_depth"),
                "seed": seed, "ent_coef": model.ent_coef,
                "r_empty": args.empty if args.empty is not None else RoboKabaddiEnv.R_EMPTY,
                "r_walkout": args.walkout if args.walkout is not None else RoboKabaddiEnv.R_TACKLE,
                "started": time.strftime("%Y-%m-%d %H:%M:%S")}
    (out / "run_info.json").write_text(json.dumps(run_info, indent=2, default=str))
    t0 = time.time()
    model.learn(total_timesteps=args.steps, progress_bar=False, reset_num_timesteps=False,
                callback=CallbackList([
                    EvalCallback(eval_env, best_model_save_path=str(out / "best_model"),
                                 log_path=str(out / "eval_logs"),
                                 eval_freq=max(1, 25_000 // N_ENVS),
                                 n_eval_episodes=20, deterministic=True, verbose=0,
                                 callback_on_new_best=SaveVecNorm(
                                     train_env, out / "best_vecnorm.pkl")),
                    CheckpointWithVecNorm(save_freq=max(1, 50_000 // N_ENVS),
                                          save_path=str(out / "checkpoints"),
                                          name_prefix="ppo", verbose=0),
                    Outcomes(out / "metrics.csv", start)]))
    run_info["total_actual_decisions"] = int(model.num_timesteps)
    run_info["elapsed_min"] = round((time.time() - t0) / 60, 2)
    run_info["finished"] = time.strftime("%Y-%m-%d %H:%M:%S")
    model.save(str(out / "final_model"))
    train_env.save(str(out / "vecnorm.pkl"))
    (out / "run_info.json").write_text(json.dumps(run_info, indent=2, default=str))
    print(f"{out.name}: {start:,} -> {model.num_timesteps:,} decisions in "
          f"{run_info['elapsed_min']} min", flush=True)


if __name__ == "__main__":
    main()
