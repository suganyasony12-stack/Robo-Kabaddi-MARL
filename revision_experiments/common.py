"""
common.py — shared utilities for the RAS revision experiments
==============================================================

Every revision experiment writes its results through this module so that:

  * results accumulate in one registry rather than scattered folders
  * every run records the exact configuration that produced it
  * finished runs are never silently overwritten or re-run
  * paper tables are regenerated from the registry, never hand-typed

Registry location: revision_experiments/registry.json

Usage from an experiment script:

    from common import Registry, evaluate_policy, wilson_ci

    reg = Registry()
    if reg.has(run_id):            # skip work already done
        print("already complete")
    else:
        ...train...
        reg.record(run_id, config=cfg, results=res)
"""

from __future__ import annotations

import json
import math
import platform
import subprocess
from datetime import datetime
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent
REGISTRY_PATH = ROOT / "registry.json"
RESULTS_DIR = ROOT / "results"
RESULTS_DIR.mkdir(parents=True, exist_ok=True)

# Standard evaluation size for all revision experiments.
# 100 episodes gives a 95% Wilson half-width of roughly +/-10 points at p=0.5,
# versus roughly +/-22 points at the 20 episodes used in the original
# submission. Reviewer 1 (comment 2) asked specifically about the reliability
# of rate estimates from small numbers of rollouts.
N_EVAL_EPISODES = 100


# ──────────────────────────────────────────────────────────────────────────
# Statistics
# ──────────────────────────────────────────────────────────────────────────
def wilson_ci(k: int, n: int, z: float = 1.959964) -> tuple[float, float]:
    """95% Wilson score interval for a binomial proportion, as fractions."""
    if n == 0:
        return 0.0, 0.0
    p = k / n
    den = 1 + z * z / n
    centre = (p + z * z / (2 * n)) / den
    half = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / den
    return max(0.0, centre - half), min(1.0, centre + half)


def summarise_outcomes(outcomes: list[str]) -> dict:
    """Turn a list of raid outcomes into rates with confidence intervals."""
    n = len(outcomes)
    out = {"n_episodes": n}
    for name in ("touch", "tackle", "empty"):
        k = sum(1 for o in outcomes if o == name)
        lo, hi = wilson_ci(k, n)
        out[f"{name}_count"] = k
        out[f"{name}_rate"] = k / n if n else 0.0
        out[f"{name}_ci95"] = [lo, hi]
    return out


# ──────────────────────────────────────────────────────────────────────────
# Evaluation
# ──────────────────────────────────────────────────────────────────────────
def evaluate_policy(model, env_fn, n_episodes: int = N_EVAL_EPISODES,
                    deterministic: bool = True) -> dict:
    """
    Evaluate any SB3 model and return outcome statistics with CIs.

    Matches the evaluation procedure of training/train_raider.py so that
    revision results are directly comparable with the original submission,
    but records per-episode outcomes and confidence intervals.
    """
    from stable_baselines3.common.env_util import make_vec_env

    env = make_vec_env(env_fn, n_envs=1)

    outcomes, durations, rewards = [], [], []
    obs = env.reset()
    ep_reward, ep_steps = 0.0, 0

    while len(outcomes) < n_episodes:
        action, _ = model.predict(obs, deterministic=deterministic)
        obs, reward, done, info = env.step(action)
        ep_reward += float(reward[0])
        ep_steps += 1
        if done[0]:
            outcomes.append(info[0].get("outcome", "empty"))
            durations.append(ep_steps * 0.05)
            rewards.append(ep_reward)
            ep_reward, ep_steps = 0.0, 0

    env.close()

    res = summarise_outcomes(outcomes)
    res.update({
        "mean_duration": float(np.mean(durations)),
        "std_duration": float(np.std(durations)),
        "mean_reward": float(np.mean(rewards)),
        "outcomes": outcomes,          # kept for exact reproducibility
    })
    return res


# ──────────────────────────────────────────────────────────────────────────
# Provenance
# ──────────────────────────────────────────────────────────────────────────
def _git_commit() -> str | None:
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "--short", "HEAD"],
            cwd=ROOT, stderr=subprocess.DEVNULL, text=True,
        ).strip()
    except Exception:
        return None


def _environment() -> dict:
    info = {"python": platform.python_version(), "platform": platform.platform()}
    for mod in ("stable_baselines3", "torch", "gymnasium", "pybullet", "numpy"):
        try:
            m = __import__(mod)
            info[mod] = getattr(m, "__version__", "unknown")
        except Exception:
            info[mod] = "not installed"
    commit = _git_commit()
    if commit:
        info["git_commit"] = commit
    return info


# ──────────────────────────────────────────────────────────────────────────
# Registry
# ──────────────────────────────────────────────────────────────────────────
class Registry:
    """Append-only record of every completed revision experiment."""

    def __init__(self, path: Path = REGISTRY_PATH):
        self.path = path
        self.data = json.loads(path.read_text()) if path.exists() else {"runs": {}}

    # -- queries ----------------------------------------------------------
    def has(self, run_id: str) -> bool:
        return run_id in self.data["runs"]

    def get(self, run_id: str) -> dict | None:
        return self.data["runs"].get(run_id)

    def find(self, **criteria) -> list[dict]:
        """Return runs whose config matches all given key=value pairs."""
        hits = []
        for rid, rec in self.data["runs"].items():
            cfg = rec.get("config", {})
            if all(cfg.get(k) == v for k, v in criteria.items()):
                hits.append({"run_id": rid, **rec})
        return hits

    # -- mutation ---------------------------------------------------------
    def record(self, run_id: str, config: dict, results: dict,
               notes: str = "") -> None:
        self.data["runs"][run_id] = {
            "config": config,
            "results": results,
            "notes": notes,
            "recorded_at": datetime.now().isoformat(timespec="seconds"),
            "environment": _environment(),
        }
        self.save()

    def save(self) -> None:
        self.path.write_text(json.dumps(self.data, indent=2))

    # -- reporting --------------------------------------------------------
    def summary(self) -> str:
        if not self.data["runs"]:
            return "Registry is empty."
        lines = [f"{'run_id':<40}{'touch':>16}{'tackle':>16}{'empty':>16}",
                 "-" * 88]
        for rid, rec in sorted(self.data["runs"].items()):
            r = rec["results"]
            def cell(name):
                rate = r.get(f"{name}_rate")
                if rate is None:
                    return "-"
                lo, hi = r.get(f"{name}_ci95", (0, 0))
                return f"{rate*100:.1f} [{lo*100:.0f},{hi*100:.0f}]"
            lines.append(f"{rid:<40}{cell('touch'):>16}"
                         f"{cell('tackle'):>16}{cell('empty'):>16}")
        return "\n".join(lines)


if __name__ == "__main__":
    print(Registry().summary())
