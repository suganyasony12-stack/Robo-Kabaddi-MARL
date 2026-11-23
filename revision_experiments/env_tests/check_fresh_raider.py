"""
check_fresh_raider.py -- raider_vs_learned.py --fresh and --def-start (author, 13 Nov)
=====================================================================================

    python3 revision_experiments/env_tests/check_fresh_raider.py

A short --fresh --touch-obs --def-start 1.0,1.87 run in a temporary folder:
finishes, run_info right, a new network (not the --raider's weights), PPO
settings as C1 (except --ent), 18 inputs, the training games start the
defenders 1.0-1.87 m past the midline, the result loads in FrozenRaider and
plays the gate test. Without the new options the training games start in the
back row, as before.
"""
import json
import subprocess
import sys
import tempfile
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path[:0] = [str(ROOT / "marl"), str(ROOT), str(ROOT / "training")]
from defender_env import FrozenRaider, diag_kwargs, start_kwargs     # noqa: E402
from eval_gates import evaluate                                       # noqa: E402
from raider_vs_learned import LearnedDefenceEnv                       # noqa: E402
import envs.kabaddi_env as E                                          # noqa: E402

passed = failed = 0
R = ROOT / "results/marl"
F = "checkpoints/mappo_3999744_steps.pt"
SRC = str(R / "diagvar/d2/raider2_s42")
DEF = [str(R / "diagvar/d2/defence_s0" / F)]


def check(name, cond):
    global passed, failed
    print(("  ok   " if cond else "  FAIL ") + name, flush=True)
    passed += bool(cond)
    failed += not cond


# training games: start depth with and without the option
def depths(kw):
    env = LearnedDefenceEnv(DEF, 2, diag=kw, touch=True)
    ys = []
    for i in range(40):
        env.reset(seed=1000 + i)
        e = env.env
        ys += [float(e._get_pos(d)[1]) for d in e.defender_ids]
    env.close()
    return np.array(ys)


ys = depths({**diag_kwargs("0.35,0.85", 3), **start_kwargs("1.0,1.87")})
check(f"training games with --def-start 1.0,1.87: defenders {ys.min():.2f}-{ys.max():.2f} m past the midline",
      ys.min() >= 1.0 - 1e-6 and ys.max() <= 1.87 + 1e-6)
ys = depths(diag_kwargs("0.35,0.85", 3))
lo, hi = E.COURT_L / 2 - E.BACK_ROW_FROM_WALL[1], E.COURT_L / 2 - E.BACK_ROW_FROM_WALL[0]
check("training games without it: back row as before", ys.min() >= lo - 1e-6 and ys.max() <= hi + 1e-6)

with tempfile.TemporaryDirectory() as tmp:
    out = Path(tmp) / "fresh"
    cmd = [sys.executable, str(ROOT / "marl/raider_vs_learned.py"), "--raider", SRC, "--defence", ",".join(DEF),
           "--n-def", "2", "--pad-to", "3", "--fresh", "--touch-obs", "--def-start", "1.0,1.87", "--steps", "26000", "--seed", "7",
           "--ent", "0", "--empty", "-6", "--walkout", "-7", "--diag-range", "0.35,0.85", "--diag-period", "3",
           "--out", str(out)]
    p = subprocess.run(cmd, cwd=ROOT, capture_output=True, text=True)
    check("short --fresh --touch-obs --def-start run finishes", p.returncode == 0)
    if p.returncode:
        print(p.stderr[-2000:])
    else:
        info = json.loads((out / "run_info.json").read_text())
        check("run_info: fresh, not continued, touch input, start 1.0-1.87, from 0 decisions",
              info["fresh"] is True and info["continued_from"] is None and info["raider_touch_obs"] is True
              and info["defender_start_depth"] == [1.0, 1.87] and info["start_decisions"] == 0
              and info["widened_from"] is None)
        from stable_baselines3 import PPO
        from train_corrected import PPO_KWARGS
        m = PPO.load(str(out / "final_model.zip"), device="cpu")
        old = PPO.load(str(Path(SRC) / "best_model/best_model.zip"), device="cpu")
        w_new = m.policy.state_dict()["mlp_extractor.policy_net.2.weight"]
        w_old = old.policy.state_dict()["mlp_extractor.policy_net.2.weight"]
        check("a new network, not the old raider's weights", float((w_new - w_old).abs().mean()) > 0.01)
        same = all(getattr(m, k) == v if not callable(getattr(m, k)) else True
                   for k, v in PPO_KWARGS.items() if k not in ("policy_kwargs", "ent_coef", "learning_rate", "clip_range"))
        check("PPO settings as C1 (n_steps, batch, epochs, gamma, ...); ent 0 as set",
              same and m.ent_coef == 0.0 and m.policy.net_arch == PPO_KWARGS["policy_kwargs"]["net_arch"])
        rt = FrozenRaider(str(out))
        check("loads with 18 inputs (touch flag)", rt.touch and rt.vn.obs_rms.mean.shape == (18,))
        res = evaluate((str(out), "v8", 3, 2, None, None, {**diag_kwargs("0.35,0.85", 3),
                                                            **start_kwargs("1.0,1.5")}))
        check("plays the gate test from the new start",
              sum(res[k]["count"] for k in ("touch", "tackle", "foul", "empty")) == 3
              and res["defender_start_depth"] == (1.0, 1.5))

print(f"\n{passed} passed, {failed} failed")
sys.exit(1 if failed else 0)
