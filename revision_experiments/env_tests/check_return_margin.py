"""
check_return_margin.py -- Empty on return only 3 cm behind the midline (author, 13 Nov)
=======================================================================================

    python3 revision_experiments/env_tests/check_return_margin.py

RoboKabaddiEnv return_margin / --return-margin. Off (default 0): saved gate
results raid for raid. On (0.03): a raider over the midline without a touch
that comes back 2 cm plays on, 4 cm back ends Empty; with no margin 2 cm back
ends Empty, as before. A raider that has touched and comes back 1 cm is a
touch with or without the margin. The tools record the setting.
"""
import json
import subprocess
import sys
import tempfile
from pathlib import Path

import numpy as np
import pybullet as p

ROOT = Path(__file__).resolve().parents[2]
sys.path[:0] = [str(ROOT / "marl"), str(ROOT)]
from defender_env import diag_kwargs, margin_kwargs                  # noqa: E402
from eval_gates import evaluate                                       # noqa: E402
import envs.kabaddi_env as E                                          # noqa: E402

passed = failed = 0
R = ROOT / "results/marl"
PY = sys.executable


def check(name, cond):
    global passed, failed
    print(("  ok   " if cond else "  FAIL ") + name, flush=True)
    passed += bool(cond)
    failed += not cond


# 1. off = as before
check("default return_margin 0", E.RoboKabaddiEnv(**E.CORRECTED_CONFIG).return_margin == 0.0)
check("margin_kwargs off gives no setting", margin_kwargs(None) == {} and margin_kwargs(0) == {})
saved = json.loads((R / "diagvar/d3/eval_gates_fixed.json").read_text())
s = [x for x in saved if x["raider"].endswith("alt_wo_r2/r2_raider_s42") and x["gate"] == "v8"][0]
now = evaluate((str(R / "alt_wo_r2/r2_raider_s42"), "v8", 30, None, None, None, {}))
check("off: accepted 3-def raider, v8, 1.0: 30 raids as saved", now["outcomes"] == s["outcomes"][:30])
try:
    E.RoboKabaddiEnv(**dict(E.CORRECTED_CONFIG, return_margin=-0.01)).close()
    check("negative margin refused", False)
except AssertionError:
    check("negative margin refused", True)


# 2. the rule itself: put the raider back behind the midline after crossing
def back_by(dist, margin, touched=False):
    env = E.RoboKabaddiEnv(**dict(E.CORRECTED_CONFIG, n_defenders=1, return_margin=margin))
    env.reset(seed=50000)
    env._flags["midline"] = True
    env._raid_start_step = env._step
    if touched:
        env._flags["contact"] = True
    for _ in range(2):                         # hold the raider there for two steps
        p.resetBasePositionAndOrientation(env.raider_id, [0.0, E.MIDLINE_Y - dist, E.R_HEIGHT / 2],
                                          [0, 0, 0, 1], physicsClientId=env._client)
        p.resetBaseVelocity(env.raider_id, [0, 0, 0], [0, 0, 0], physicsClientId=env._client)
        _, _, term, trunc, info = env.step(np.zeros(env.action_space.shape, np.float32))
        if term or trunc:
            break
    env.close()
    return (info.get("outcome") if term or trunc else "plays on")


check("no margin: 2 cm back without a touch = Empty (as before)", back_by(0.02, 0.0) == "empty")
check("margin 3 cm: 2 cm back without a touch plays on", back_by(0.02, 0.03) == "plays on")
check("margin 3 cm: 4 cm back without a touch = Empty", back_by(0.04, 0.03) == "empty")
check("touched, 1 cm back: the same result with and without the margin",
      back_by(0.01, 0.0, True) == back_by(0.01, 0.03, True))

# 3. tools record it, robot runner runs with it
with tempfile.TemporaryDirectory() as tmp:
    out = Path(tmp) / "g.json"
    r = subprocess.run([PY, "marl/eval_gates.py", "--raiders", str(R / "one_def_copy/raider_s456"), "--gates", "v8",
                        "--n-def", "1", "--episodes", "3", "--procs", "1", "--return-margin", "0.03", "--out", str(out)],
                       cwd=ROOT, capture_output=True, text=True)
    check("eval_gates.py --return-margin runs and records it",
          r.returncode == 0 and json.loads(out.read_text())[0]["return_margin"] == 0.03)
    out = Path(tmp) / "d.json"
    r = subprocess.run([PY, "marl/eval_defenders.py", "--ckpt",
                        str(R / "one_def_copy/defence_s0/checkpoints/mappo_3999744_steps.pt"),
                        "--raiders", str(R / "one_def_copy/raider_s456"), "--n-def", "1", "--episodes", "3",
                        "--procs", "1", "--return-margin", "0.03", "--out", str(out)],
                       cwd=ROOT, capture_output=True, text=True)
    check("eval_defenders.py --return-margin runs and records it",
          r.returncode == 0 and json.loads(out.read_text())[0]["return_margin"] == 0.03)
    r = subprocess.run([PY, "physical_experiment/rig/raid_runner.py", "--raider", "learned", "--defence", "scripted",
                        "--gate", "v8", "--n-def", "2", "--raids", "1", "--raider-rule", "deep",
                        "--def-start", "1.0,1.5", "--return-margin", "0.03", "--fake", "--outdir", tmp + "/rr"],
                       cwd=ROOT, capture_output=True, text=True)
    rows = list(Path(tmp, "rr").rglob("raids.csv"))
    check("raid_runner.py --return-margin 0.03 --fake runs and records it",
          r.returncode == 0 and bool(rows) and ",0.03," in rows[0].read_text())
    if r.returncode:
        print(r.stderr[-1500:])

print(f"\n{passed} passed, {failed} failed")
sys.exit(1 if failed else 0)
