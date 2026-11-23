"""
train_corrected.py — C1 training on the corrected environment
==============================================================

Trains one run of the C1 set (CORRECTIONS_TRACKER.md) on
envs.kabaddi_env.CORRECTED_CONFIG. Each run is one process limited to one
CPU thread, so the launcher can run up to 7 at once on the 8-core Mac.

The runs (see RUNS below):
    main_1v3_seed{S}            PPO, 1 vs 3, v6 gate defenders
    count_1v1_seed{S}           PPO, 1 vs 1   (also the ablation's full reward)
    count_1v2_seed{S}           PPO, 1 vs 2
    abl_{variant}_1v1_seed{S}   PPO, 1 vs 1, reward variant (paper Sec 6.4)
    sac_1v3_seed{S}, td3_1v3_seed{S}   baselines, library-default settings
                                as in Experiment A
    ppo250k_1v3_seed{S}         PPO at the baselines' budget, for comparison
    pursuit_1v3_seed{S}         PPO, 1 vs 3, the robots' pursuit defenders
                                ('physical'), for comparison with the 210 raids
Seeds 42, 456, 123.

Two stages (author's decision after the pilot, 25 Oct): every run against
the v6 gate trains its first third against the robots' pursuit defenders,
then the remaining two thirds against the gate, in one run with the same seed
and the normalisation statistics carried over. Trained against the gate from
the start, the policy never found a touch and learned only to cross and come
straight back (100% Empty in the 100k pilot); against pursuit it learned to
touch (98%). The best model is chosen only during the gate stage, by
evaluation against the gate. The pursuit runs train against pursuit
throughout.

Budget is in decisions (one env.step = one 5 Hz decision = 0.2 s): PPO 750k
(the same simulated play as the paper's 3M steps at 0.05 s), baselines 250k
(the paper's 1M). PPO hyperparameters are the paper's (training/train_raider.py
DEFAULT_CONFIG); gamma 0.99 now applies per 0.2 s decision (author's choice).

Each run writes results/c1/<run>/: best_model/ (with the normalisation
statistics saved at the same moment, best_vecnorm.pkl), checkpoints/ (each
with its vecnorm), final_model.zip, vecnorm.pkl, metrics.csv (outcome rates
during training), eval_logs/, run_info.json. Registry entries are written by
the evaluation, not here, so parallel runs never write the same file.

Usage (from the robo_kabaddi/ folder)
-------------------------------------
    python3 training/train_corrected.py --run main_1v3_seed42
    python3 training/train_corrected.py --run main_1v3_seed42 --steps 100000 --tag pilot
    python3 training/train_corrected.py --list
"""

from __future__ import annotations

import argparse
import copy
import csv
import json
import os
import sys
import time
from pathlib import Path

os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")

import numpy as np                                              # noqa: E402
import torch                                                    # noqa: E402

torch.set_num_threads(1)

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "training"))

from envs.kabaddi_env import RoboKabaddiEnv, CORRECTED_CONFIG    # noqa: E402
from train_raider import DEFAULT_CONFIG, ABLATION_CONFIGS        # noqa: E402
from stable_baselines3 import PPO, SAC, TD3                      # noqa: E402
from stable_baselines3.common.callbacks import (                 # noqa: E402
    BaseCallback, CallbackList, CheckpointCallback, EvalCallback)
from stable_baselines3.common.env_util import make_vec_env       # noqa: E402
from stable_baselines3.common.monitor import Monitor             # noqa: E402
from stable_baselines3.common.vec_env import (                   # noqa: E402
    DummyVecEnv, SubprocVecEnv, VecNormalize)

SEEDS = (42, 456, 123)
PPO_STEPS, BASELINE_STEPS = 750_000, 250_000
PRETRAIN_FRACTION = 1 / 3        # first stage against pursuit (gate runs)
N_ENVS_PPO = 4
OUT = ROOT / "results" / "c1"
# --parallel-envs: step the training copies of the game in separate processes
# (one core each) instead of one after another. Same algorithm, settings and
# seeds; only the work is spread across cores. Default off.
VEC_CLS = DummyVecEnv

PPO_KWARGS = {k: DEFAULT_CONFIG[k] for k in (
    "learning_rate", "n_steps", "batch_size", "n_epochs", "gamma", "gae_lambda",
    "clip_range", "ent_coef", "vf_coef", "max_grad_norm")}
PPO_KWARGS["policy_kwargs"] = dict(net_arch=DEFAULT_CONFIG["net_arch"])
# Experiment A's baseline settings (revision_experiments/exp_A_rl_baselines.py)
SAC_KWARGS = dict(learning_rate=3e-4, buffer_size=300_000, batch_size=256,
                  tau=0.005, gamma=0.99, train_freq=1, gradient_steps=1,
                  learning_starts=10_000, policy_kwargs=dict(net_arch=[256, 256]))
TD3_KWARGS = dict(learning_rate=1e-3, buffer_size=300_000, batch_size=256,
                  tau=0.005, gamma=0.99, train_freq=(1, "episode"),
                  gradient_steps=-1, learning_starts=10_000,
                  policy_kwargs=dict(net_arch=[256, 256]))


def _runs() -> dict:
    """run name -> dict(algo, n_def, env overrides, reward overrides, steps)."""
    r = {}
    for s in SEEDS:
        r[f"main_1v3_seed{s}"] = dict(algo="ppo", n_def=3, seed=s)
        r[f"count_1v1_seed{s}"] = dict(algo="ppo", n_def=1, seed=s)
        r[f"count_1v2_seed{s}"] = dict(algo="ppo", n_def=2, seed=s)
        for v, over in ABLATION_CONFIGS.items():
            if v != "full_reward":                    # = count_1v1
                r[f"abl_{v}_1v1_seed{s}"] = dict(algo="ppo", n_def=1, seed=s,
                                                 reward=over)
        r[f"sac_1v3_seed{s}"] = dict(algo="sac", n_def=3, seed=s)
        r[f"td3_1v3_seed{s}"] = dict(algo="td3", n_def=3, seed=s)
        r[f"ppo250k_1v3_seed{s}"] = dict(algo="ppo", n_def=3, seed=s,
                                         budget=BASELINE_STEPS)
        r[f"pursuit_1v3_seed{s}"] = dict(algo="ppo", n_def=3, seed=s,
                                         env=dict(defender_type="physical"))
    for spec in r.values():
        spec.setdefault("env", {})
        spec.setdefault("reward", {})
        spec["steps"] = spec.pop("budget", PPO_STEPS if spec["algo"] == "ppo"
                                 else BASELINE_STEPS)
        spec["pretrain"] = spec["env"].get("defender_type", "gate") == "gate"
    return r


RUNS = _runs()


def env_kwargs(spec) -> dict:
    return {"n_defenders": spec["n_def"], **CORRECTED_CONFIG, **spec["env"]}


def make_env_fn(spec, seed, defenders=None):
    """defenders='physical' gives the pursuit stage of a gate run."""
    def _init():
        kw = env_kwargs(spec)
        if defenders:
            kw["defender_type"] = defenders
        env = RoboKabaddiEnv(**kw)
        for k, v in spec["reward"].items():
            setattr(env, k, v)
        env = Monitor(env, info_keywords=("outcome",))
        env.reset(seed=seed)
        return env
    return _init


class OutcomeLog(BaseCallback):
    """Touch / tackle / empty rates of training episodes, every 10k steps."""

    def __init__(self, path: Path, stage: str, start: int = 0,
                 every: int = 10_000, t0: float | None = None):
        super().__init__()
        self.path, self.every, self.buf = path, every, []
        self.stage, self.next = stage, start + every
        new = start == 0 or not path.exists()
        self.fh = open(path, "w" if new else "a", newline="")
        self.w = csv.writer(self.fh)
        if new:
            self.w.writerow(["timestep", "stage", "touch_rate", "tackle_rate",
                             "empty_rate", "episodes", "mean_ep_decisions",
                             "elapsed_s"])
        self.t0 = t0 or time.time()

    def _on_step(self) -> bool:
        for info, done in zip(self.locals["infos"], self.locals["dones"]):
            if done and "episode" in info:
                self.buf.append((info.get("outcome", "empty"), info["episode"]["l"]))
        if self.num_timesteps >= self.next and self.buf:
            o = [b[0] for b in self.buf]
            n = len(o)
            self.w.writerow([self.num_timesteps, self.stage, o.count("touch") / n,
                             o.count("tackle") / n, o.count("empty") / n, n,
                             float(np.mean([b[1] for b in self.buf])),
                             round(time.time() - self.t0, 1)])
            self.fh.flush()
            self.buf, self.next = [], self.next + self.every
        return True

    def _on_training_end(self) -> None:
        self.fh.close()


class SaveVecNorm(BaseCallback):
    """Save the training normalisation statistics next to a saved model."""

    def __init__(self, vec_env, path: Path):
        super().__init__()
        self.vec_env, self.path = vec_env, path

    def _on_step(self) -> bool:
        self.vec_env.save(str(self.path))
        return True


class CheckpointWithVecNorm(CheckpointCallback):
    def _on_step(self) -> bool:
        if self.n_calls % self.save_freq == 0:
            self.training_env.save(
                str(Path(self.save_path) / f"vecnorm_{self.num_timesteps}_steps.pkl"))
        return super()._on_step()


def info_stage(spec) -> str:
    return "gate" if spec["env"].get("defender_type", "gate") == "gate" else "pursuit"


def train(name: str, steps: int | None = None, tag: str = "",
          stage1_steps: int | None = None) -> Path:
    spec = RUNS[name]
    steps = steps or spec["steps"]
    run = f"{name}_{tag}" if tag else name
    out = OUT / run
    out.mkdir(parents=True, exist_ok=True)
    seed, algo = spec["seed"], spec["algo"]
    n_envs = N_ENVS_PPO if algo == "ppo" else 1

    stage1 = 0
    if spec["pretrain"]:
        stage1 = (stage1_steps if stage1_steps is not None
                  else int(round(steps * PRETRAIN_FRACTION)))

    def vec(defenders=None):
        e = make_vec_env(make_env_fn(spec, seed, defenders), n_envs=n_envs, seed=seed,
                         vec_env_cls=VEC_CLS)
        return VecNormalize(e, norm_obs=True, norm_reward=(algo == "ppo"),
                            clip_obs=10.0)

    train_env = vec("physical" if stage1 else None)
    eval_env = make_vec_env(make_env_fn(spec, seed + 999), n_envs=1)
    eval_env = VecNormalize(eval_env, norm_obs=True, norm_reward=False,
                            training=False)

    if algo == "ppo":
        model = PPO("MlpPolicy", train_env, seed=seed, verbose=0, device="cpu",
                    **PPO_KWARGS)
    else:
        cls, kw = (SAC, SAC_KWARGS) if algo == "sac" else (TD3, TD3_KWARGS)
        model = cls("MlpPolicy", train_env, seed=seed, verbose=0, device="cpu", **kw)

    every = max(1, 25_000 // n_envs)

    def ckpt():
        return CheckpointWithVecNorm(save_freq=max(1, 50_000 // n_envs),
                                     save_path=str(out / "checkpoints"),
                                     name_prefix=algo, verbose=0)
    info = {"run": run, "spec": spec, "steps": steps, "n_envs": n_envs,
            "env_kwargs": env_kwargs(spec),
            "hyperparameters": {"ppo": PPO_KWARGS, "sac": SAC_KWARGS,
                                "td3": TD3_KWARGS}[algo],
            "stages": ({"pursuit": stage1, "gate": steps - stage1} if stage1
                       else {info_stage(spec): steps}),
            "started": time.strftime("%Y-%m-%d %H:%M:%S")}
    (out / "run_info.json").write_text(json.dumps(info, indent=2, default=str))
    t0 = time.time()
    if stage1:                                   # stage 1: against pursuit
        model.learn(total_timesteps=stage1, progress_bar=False,
                    callback=CallbackList([ckpt(), OutcomeLog(
                        out / "metrics.csv", "pursuit", 0, t0=t0)]))
        info["stage1_actual_decisions"] = int(model.num_timesteps)
        gate_env = vec(None)                     # stage 2: against the gate
        gate_env.obs_rms = copy.deepcopy(train_env.obs_rms)
        gate_env.ret_rms = copy.deepcopy(train_env.ret_rms)
        train_env.close()
        train_env = gate_env
        model.set_env(train_env)
    final_stage = "gate" if spec["pretrain"] else info_stage(spec)
    model.learn(total_timesteps=steps - stage1, progress_bar=False,
                reset_num_timesteps=not stage1,
                callback=CallbackList([
                    EvalCallback(eval_env, best_model_save_path=str(out / "best_model"),
                                 log_path=str(out / "eval_logs"), eval_freq=every,
                                 n_eval_episodes=20, deterministic=True, verbose=0,
                                 callback_on_new_best=SaveVecNorm(
                                     train_env, out / "best_vecnorm.pkl")),
                    ckpt(),
                    OutcomeLog(out / "metrics.csv", final_stage, stage1, t0=t0)]))
    info["total_actual_decisions"] = int(model.num_timesteps)   # whole rollouts
    info["elapsed_min"] = round((time.time() - t0) / 60, 2)
    info["finished"] = time.strftime("%Y-%m-%d %H:%M:%S")
    model.save(str(out / "final_model"))
    train_env.save(str(out / "vecnorm.pkl"))
    (out / "run_info.json").write_text(json.dumps(info, indent=2, default=str))
    print(f"{run}: {steps:,} decisions in {info['elapsed_min']} min -> {out}")
    return out


def continue_run(name: str, extra: int, tag: str = "ext") -> Path:
    """Continue a finished run from its final model and normalisation
    statistics, in its final environment (the gate for gate runs), for
    `extra` more decisions. Writes to results/c1/<run>_<tag>/; the original
    run is left untouched. Its best model is chosen within the extension."""
    spec = RUNS[name]
    src, out = OUT / name, OUT / f"{name}_{tag}"
    out.mkdir(parents=True, exist_ok=True)
    seed, algo = spec["seed"], spec["algo"]
    n_envs = N_ENVS_PPO if algo == "ppo" else 1
    venv = make_vec_env(make_env_fn(spec, seed + 7), n_envs=n_envs, seed=seed + 7,
                        vec_env_cls=VEC_CLS)
    train_env = VecNormalize.load(str(src / "vecnorm.pkl"), venv)
    train_env.training = True
    cls = {"ppo": PPO, "sac": SAC, "td3": TD3}[algo]
    model = cls.load(str(src / "final_model.zip"), env=train_env, device="cpu")
    eval_env = make_vec_env(make_env_fn(spec, seed + 999), n_envs=1)
    eval_env = VecNormalize(eval_env, norm_obs=True, norm_reward=False,
                            training=False)
    start = int(model.num_timesteps)
    info = json.loads((src / "run_info.json").read_text())
    info.update({"run": out.name, "continued_from": name,
                 "start_decisions": start, "extra_decisions": extra,
                 "started": time.strftime("%Y-%m-%d %H:%M:%S")})
    info.pop("finished", None)
    (out / "run_info.json").write_text(json.dumps(info, indent=2, default=str))
    stage = "gate" if spec["pretrain"] else info_stage(spec)
    t0 = time.time()
    model.learn(total_timesteps=extra, progress_bar=False, reset_num_timesteps=False,
                callback=CallbackList([
                    EvalCallback(eval_env, best_model_save_path=str(out / "best_model"),
                                 log_path=str(out / "eval_logs"),
                                 eval_freq=max(1, 25_000 // n_envs),
                                 n_eval_episodes=20, deterministic=True, verbose=0,
                                 callback_on_new_best=SaveVecNorm(
                                     train_env, out / "best_vecnorm.pkl")),
                    CheckpointWithVecNorm(save_freq=max(1, 50_000 // n_envs),
                                          save_path=str(out / "checkpoints"),
                                          name_prefix=algo, verbose=0),
                    OutcomeLog(out / "metrics.csv", stage, start, t0=t0)]))
    info["total_actual_decisions"] = int(model.num_timesteps)
    info["elapsed_min"] = round((time.time() - t0) / 60, 2)
    info["finished"] = time.strftime("%Y-%m-%d %H:%M:%S")
    model.save(str(out / "final_model"))
    train_env.save(str(out / "vecnorm.pkl"))
    (out / "run_info.json").write_text(json.dumps(info, indent=2, default=str))
    print(f"{out.name}: {start:,} -> {model.num_timesteps:,} decisions in "
          f"{info['elapsed_min']} min")
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", choices=sorted(RUNS))
    ap.add_argument("--continue", dest="cont", choices=sorted(RUNS),
                    help="continue this finished run (with --extra)")
    ap.add_argument("--extra", type=int, default=1_500_000,
                    help="decisions to add with --continue")
    ap.add_argument("--stage1", type=int,
                    help="pursuit-stage decisions for a gate run (default: a third)")
    ap.add_argument("--parallel-envs", action="store_true",
                    help="run the training envs in separate processes")
    ap.add_argument("--steps", type=int, help="override the budget (pilot)")
    ap.add_argument("--tag", default="", help="suffix for the output folder")
    ap.add_argument("--list", action="store_true")
    args = ap.parse_args()
    global VEC_CLS
    if args.parallel_envs:
        VEC_CLS = SubprocVecEnv
    if args.cont:
        continue_run(args.cont, args.extra, args.tag or "ext")
        return
    if args.list or not args.run:
        for k, v in RUNS.items():
            st = (f"{round(v['steps'] * PRETRAIN_FRACTION):,} pursuit + "
                  f"{v['steps'] - round(v['steps'] * PRETRAIN_FRACTION):,} gate"
                  if v["pretrain"] else f"{v['steps']:,} pursuit")
            print(f"{k:<34} {v['algo']}  1v{v['n_def']}  {st}")
        print(f"{len(RUNS)} runs")
        return
    train(args.run, args.steps, args.tag, args.stage1)


if __name__ == "__main__":
    main()
