"""
Robo-Kabaddi — PPO Raider Training
====================================
Train a PPO raider policy against rule-based defenders.

Usage examples
--------------
# Basic training (1v1, reactive defenders, 3M steps):
  python train_raider.py

# 1v2 wall defenders, custom steps:
  python train_raider.py --n_def 2 --def_type wall --steps 3000000

# All configurations sequentially (generates full paper results):
  python train_raider.py --all

# With Weights & Biases logging:
  python train_raider.py --wandb

# Quick test run (fast, low steps):
  python train_raider.py --test
"""

import sys
import os
import argparse
import time
import numpy as np
from pathlib import Path

# ── Add project root to path ──────────────────────────────────────────────────
ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT))

from envs.kabaddi_env import RoboKabaddiEnv

from stable_baselines3 import PPO
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.vec_env import VecNormalize
from stable_baselines3.common.callbacks import (
    EvalCallback,
    CheckpointCallback,
    BaseCallback,
)
from stable_baselines3.common.monitor import Monitor


# ══════════════════════════════════════════════════════════════════════════════
#  Default configuration
# ══════════════════════════════════════════════════════════════════════════════

DEFAULT_CONFIG = {
    # Environment
    "n_defenders"    : 1,
    "defender_type"  : "reactive",
    "dt"             : 0.05,

    # Training
    "total_timesteps": 3_000_000,
    "n_envs"         : 4,           # reduce to 2 if Mac runs slow
    "n_seeds"        : 3,
    "seed"           : 42,

    # PPO hyperparameters (tuned for this environment)
    "learning_rate"  : 1e-4,
    "n_steps"        : 2048,
    "batch_size"     : 256,
    "n_epochs"       : 10,
    "gamma"          : 0.99,
    "gae_lambda"     : 0.95,
    "clip_range"     : 0.15,
    "ent_coef"       : 0.02,
    "vf_coef"        : 0.5,
    "max_grad_norm"  : 0.5,

    # Network architecture
    "net_arch"       : [256, 256],

    # Logging
    "eval_freq"      : 50_000,
    "checkpoint_freq": 100_000,
    "n_eval_episodes": 20,
}

# All experiment configurations for the paper
ALL_CONFIGS = [
    {"n_defenders": n, "defender_type": dt}
    for n  in [1, 2, 3]
    for dt in ["static", "reactive", "wall"]
]

# Ablation configurations (1v1 reactive, vary reward)
ABLATION_CONFIGS = {
    "full_reward"   : {},   # default
    "no_progress"   : {"R_PROGRESS": 0.0},
    "no_time_penalty": {"R_TIME": 0.0},
    "no_milestones" : {"R_MIDLINE": 0.0, "R_BAULK": 0.0, "R_BONUS": 0.0},
    "sparse_only"   : {
        "R_PROGRESS": 0.0, "R_TIME": 0.0,
        "R_MIDLINE" : 0.0, "R_BAULK": 0.0, "R_BONUS": 0.0,
        "R_BOUNDARY": 0.0,
    },
}


# ══════════════════════════════════════════════════════════════════════════════
#  Custom metrics callback
# ══════════════════════════════════════════════════════════════════════════════

class KabaddiCallback(BaseCallback):
    """
    Logs game-specific metrics every LOG_FREQ steps.
    Metrics: touch_rate, tackle_rate, empty_rate, mean_raid_duration
    """
    LOG_FREQ = 10_000

    def __init__(self, log_path, use_wandb=False, verbose=0):
        super().__init__(verbose)
        self.log_path   = Path(log_path)
        self.use_wandb  = use_wandb
        self.log_path.mkdir(parents=True, exist_ok=True)

        self._outcomes  = []
        self._durations = []
        self._rewards   = []
        self._log_file  = open(self.log_path / "metrics.csv", "w")
        self._log_file.write(
            "timestep,touch_rate,tackle_rate,empty_rate,"
            "mean_duration,mean_ep_reward\n"
        )

    def _on_step(self):
        for info in self.locals.get("infos", []):
            if "outcome" in info:
                self._outcomes.append(info["outcome"])
            if "episode" in info:
                self._durations.append(info["episode"]["l"] * 0.05)
                self._rewards.append(info["episode"]["r"])

        if self.n_calls % self.LOG_FREQ == 0 and self._outcomes:
            n   = len(self._outcomes[-200:])
            recent = self._outcomes[-200:]
            touch  = recent.count("touch")  / n
            tackle = recent.count("tackle") / n
            empty  = recent.count("empty")  / n
            dur    = np.mean(self._durations[-200:]) if self._durations else 0
            rew    = np.mean(self._rewards[-200:])   if self._rewards   else 0

            self._log_file.write(
                f"{self.num_timesteps},{touch:.4f},{tackle:.4f},"
                f"{empty:.4f},{dur:.2f},{rew:.2f}\n"
            )
            self._log_file.flush()

            if self.verbose:
                print(f"  [{self.num_timesteps:>8d}] "
                      f"touch={touch:.2f} tackle={tackle:.2f} "
                      f"empty={empty:.2f} dur={dur:.1f}s")

            if self.use_wandb:
                import wandb
                wandb.log({
                    "touch_rate" : touch,
                    "tackle_rate": tackle,
                    "empty_rate" : empty,
                    "mean_dur"   : dur,
                    "mean_reward": rew,
                    "timesteps"  : self.num_timesteps,
                })
        return True

    def _on_training_end(self):
        self._log_file.close()


# ══════════════════════════════════════════════════════════════════════════════
#  Training function
# ══════════════════════════════════════════════════════════════════════════════

def make_env_fn(n_defenders, defender_type, reward_overrides=None, seed=0):
    """Factory to create a monitored environment."""
    def _init():
        env = RoboKabaddiEnv(
            n_defenders   = n_defenders,
            defender_type = defender_type,
            render        = False,
        )
        # Apply reward overrides for ablation
        if reward_overrides:
            for k, v in reward_overrides.items():
                setattr(env, k, v)

        env = Monitor(env)
        env.reset(seed=seed)
        return env
    return _init


def train_single(config, run_name, use_wandb=False, verbose=True):
    """
    Train a single PPO raider policy.

    Returns the trained model and final evaluation results.
    """
    outdir = ROOT / "results" / run_name
    outdir.mkdir(parents=True, exist_ok=True)

    n_def    = config["n_defenders"]
    def_type = config["defender_type"]
    seed     = config.get("seed", 42)
    n_envs   = config.get("n_envs", 4)

    print(f"\n{'='*60}")
    print(f"  Training: {run_name}")
    print(f"  Defenders: {n_def} × '{def_type}'")
    print(f"  Steps: {config['total_timesteps']:,}  |  Envs: {n_envs}")
    print(f"  Output: {outdir}")
    print(f"{'='*60}")

    # ── Environments ──────────────────────────────────────────────────
    reward_overrides = config.get("reward_overrides", None)

    train_env = make_vec_env(
        make_env_fn(n_def, def_type, reward_overrides, seed),
        n_envs = n_envs,
        seed   = seed,
    )
    train_env = VecNormalize(
        train_env, norm_obs=True, norm_reward=True, clip_obs=10.0
    )

    eval_env = make_vec_env(
        make_env_fn(n_def, def_type, reward_overrides, seed=seed+999),
        n_envs = 1,
    )
    eval_env = VecNormalize(
        eval_env, norm_obs=True, norm_reward=False, training=False
    )

    # ── Model ─────────────────────────────────────────────────────────
    model = PPO(
        policy        = "MlpPolicy",
        env           = train_env,
        learning_rate = config["learning_rate"],
        n_steps       = config["n_steps"],
        batch_size    = config["batch_size"],
        n_epochs      = config["n_epochs"],
        gamma         = config["gamma"],
        gae_lambda    = config["gae_lambda"],
        clip_range    = config["clip_range"],
        ent_coef      = config["ent_coef"],
        vf_coef       = config["vf_coef"],
        max_grad_norm = config["max_grad_norm"],
        policy_kwargs = dict(net_arch=config["net_arch"]),
        verbose       = 0,
        seed          = seed,
        tensorboard_log = str(ROOT / "tb_logs"),
        device        = "auto",   # uses MPS on Apple Silicon if available
    )

    # ── Callbacks ─────────────────────────────────────────────────────
    eval_cb = EvalCallback(
        eval_env,
        best_model_save_path = str(outdir / "best_model"),
        log_path             = str(outdir / "eval_logs"),
        eval_freq            = config["eval_freq"] // n_envs,
        n_eval_episodes      = config["n_eval_episodes"],
        deterministic        = True,
        verbose              = 0,
    )

    ckpt_cb = CheckpointCallback(
        save_freq   = config["checkpoint_freq"] // n_envs,
        save_path   = str(outdir / "checkpoints"),
        name_prefix = "ppo_kabaddi",
        verbose     = 0,
    )

    metrics_cb = KabaddiCallback(
        log_path  = str(outdir),
        use_wandb = use_wandb,
        verbose   = 1 if verbose else 0,
    )

    # ── Train ─────────────────────────────────────────────────────────
    t0 = time.time()
    model.learn(
        total_timesteps = config["total_timesteps"],
        callback        = [eval_cb, ckpt_cb, metrics_cb],
        progress_bar    = verbose,
    )
    elapsed = time.time() - t0

    # ── Save ──────────────────────────────────────────────────────────
    model.save(str(outdir / "final_model"))
    train_env.save(str(outdir / "vecnorm.pkl"))

    print(f"\n  Training complete in {elapsed/60:.1f} min")
    print(f"  Model saved to {outdir}")

    # ── Final evaluation ───────────────────────────────────────────────
    results = evaluate_policy(
        model        = model,
        env_fn       = make_env_fn(n_def, def_type, reward_overrides,
                                   seed=seed+9999),
        n_episodes   = config["n_eval_episodes"],
        deterministic= True,
    )

    print(f"\n  Final evaluation ({config['n_eval_episodes']} episodes):")
    print(f"    Touch  : {results['touch_rate']:.2%}")
    print(f"    Tackle : {results['tackle_rate']:.2%}")
    print(f"    Empty  : {results['empty_rate']:.2%}")
    print(f"    Mean raid duration: {results['mean_duration']:.1f}s")

    # Save results
    import json
    with open(outdir / "final_results.json", "w") as f:
        json.dump({**results, "elapsed_min": elapsed/60}, f, indent=2)

    train_env.close()
    eval_env.close()

    return model, results


def evaluate_policy(model, env_fn, n_episodes=50, deterministic=True):
    """
    Evaluate a trained policy and return game outcome statistics.
    """
    env = make_vec_env(env_fn, n_envs=1)

    outcomes  = []
    durations = []
    rewards   = []

    obs = env.reset()
    ep_reward = 0.0
    ep_steps  = 0
    ep_count  = 0

    while ep_count < n_episodes:
        action, _ = model.predict(obs, deterministic=deterministic)
        obs, reward, done, info = env.step(action)
        ep_reward += reward[0]
        ep_steps  += 1

        if done[0]:
            ep_count += 1
            outcome   = info[0].get("outcome", "empty")
            outcomes.append(outcome)
            durations.append(ep_steps * 0.05)
            rewards.append(ep_reward)
            ep_reward = 0.0
            ep_steps  = 0

    env.close()
    n = len(outcomes)
    touch_outcomes  = [o for o in outcomes if o == "touch"]
    tackle_outcomes = [o for o in outcomes if o == "tackle"]
    empty_outcomes  = [o for o in outcomes if o == "empty"]

    return {
        "n_episodes"    : n,
        "touch_rate"    : len(touch_outcomes)  / n,
        "tackle_rate"   : len(tackle_outcomes) / n,
        "empty_rate"    : len(empty_outcomes)  / n,
        "mean_duration" : float(np.mean(durations)),
        "std_duration"  : float(np.std(durations)),
        "mean_reward"   : float(np.mean(rewards)),
    }


# ══════════════════════════════════════════════════════════════════════════════
#  CLI entry point
# ══════════════════════════════════════════════════════════════════════════════

def main():
    parser = argparse.ArgumentParser(description="Train Robo-Kabaddi PPO raider")

    parser.add_argument("--n_def",    type=int, default=1, choices=[1,2,3])
    parser.add_argument("--def_type", type=str, default="reactive",
                        choices=["static","reactive","wall"])
    parser.add_argument("--steps",    type=int, default=3_000_000)
    parser.add_argument("--seed",     type=int, default=42)
    parser.add_argument("--n_envs",   type=int, default=4)

    parser.add_argument("--all",      action="store_true",
                        help="Run all 9 training configurations sequentially")
    parser.add_argument("--ablation", action="store_true",
                        help="Run reward ablation study (1v1 reactive)")
    parser.add_argument("--test",     action="store_true",
                        help="Quick test run (50k steps, 1 env)")
    parser.add_argument("--wandb",    action="store_true",
                        help="Enable Weights & Biases logging")

    args = parser.parse_args()

    # ── Quick test ────────────────────────────────────────────────────
    if args.test:
        print("Running QUICK TEST (50k steps)...")
        cfg = {**DEFAULT_CONFIG, "total_timesteps": 50_000, "n_envs": 1}
        train_single(cfg, run_name="test_run", use_wandb=False, verbose=True)
        return

    # ── Full ablation study ───────────────────────────────────────────
    if args.ablation:
        print("Running ABLATION STUDY...")
        for abl_name, overrides in ABLATION_CONFIGS.items():
            for seed in [42, 123, 456]:
                cfg = {
                    **DEFAULT_CONFIG,
                    "n_defenders"    : 1,
                    "defender_type"  : "reactive",
                    "seed"           : seed,
                    "reward_overrides": overrides,
                    "total_timesteps": args.steps,
                    "n_envs"         : args.n_envs,
                }
                run_name = f"ablation_{abl_name}_seed{seed}"
                train_single(cfg, run_name, use_wandb=args.wandb)
        return

    # ── All 9 configurations ──────────────────────────────────────────
    if args.all:
        print("Running ALL CONFIGURATIONS (9 configs × 3 seeds)...")
        for cfg_override in ALL_CONFIGS:
            for seed in [42, 123, 456]:
                cfg = {
                    **DEFAULT_CONFIG,
                    **cfg_override,
                    "seed"           : seed,
                    "total_timesteps": args.steps,
                    "n_envs"         : args.n_envs,
                }
                n   = cfg["n_defenders"]
                dt  = cfg["defender_type"]
                run_name = f"{n}v{n}_{dt}_seed{seed}"
                train_single(cfg, run_name, use_wandb=args.wandb)
        return

    # ── Single run ────────────────────────────────────────────────────
    cfg = {
        **DEFAULT_CONFIG,
        "n_defenders"    : args.n_def,
        "defender_type"  : args.def_type,
        "seed"           : args.seed,
        "total_timesteps": args.steps,
        "n_envs"         : args.n_envs,
    }
    run_name = f"{args.n_def}v{args.n_def}_{args.def_type}_seed{args.seed}"
    train_single(cfg, run_name, use_wandb=args.wandb, verbose=True)


if __name__ == "__main__":
    main()
