"""
check_raider_rule.py -- 'go deep until touch' on top of the trained raider (author, 13 Nov)
==========================================================================================

    python3 revision_experiments/env_tests/check_raider_rule.py

defender_env.deep_rule / FrozenRaider(rule="deep") / --raider-rule deep.
Off: the policy alone, saved gate results raid for raid. On, every decision
of 3- and 1-defender gate raids (varying speed, defenders from 3.0-3.5 m):
before the touch with more than 8 s left, never home and never stopped, and
an changed command goes at 100 % toward the nearest defender; after the touch
and in the last 8 s, exactly the policy's command. The tools record the rule;
the robot runner runs it (--fake) for 1, 2 and 3 defenders.
"""
import json
import subprocess
import sys
import tempfile
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path[:0] = [str(ROOT / "marl"), str(ROOT)]
from defender_env import FrozenRaider, diag_kwargs, start_kwargs, deep_rule, RULE_HOME_S   # noqa: E402
from eval_gates import evaluate, C3_RULES, GATES                                            # noqa: E402
from envs.kabaddi_env import RoboKabaddiEnv, hardware_command, CMD_DEADZONE                 # noqa: E402

passed = failed = 0
R = ROOT / "results/marl"
PY = sys.executable
R3 = str(R / "alt_wo_r2/r2_raider_s42")
R1 = str(R / "one_def_copy/raider_s456")
KW = {**diag_kwargs("0.35,0.85", 3), **start_kwargs("1.0,1.5")}


def check(name, cond):
    global passed, failed
    print(("  ok   " if cond else "  FAIL ") + name, flush=True)
    passed += bool(cond)
    failed += not cond


# 1. off = the policy alone
check("default: no rule", FrozenRaider(R3).rule is None)
saved = json.loads((R / "diagvar/d3/eval_gates_fixed.json").read_text())
s = [x for x in saved if x["raider"].endswith("alt_wo_r2/r2_raider_s42") and x["gate"] == "v8"][0]
now = evaluate((R3, "v8", 30, None, None, None, {}))
check("off: accepted 3-def raider, v8, 1.0: 30 raids as saved", now["outcomes"] == s["outcomes"][:30])
try:
    FrozenRaider(R3, rule="other")
    check("unknown rule refused", False)
except AssertionError:
    check("unknown rule refused", True)

# 2. on: every decision against the policy's own command
for run, k, gate in ((R3, 3, "v8"), (R3, 3, "v10"), (R1, 1, "v8")):
    pol, rul = FrozenRaider(run), FrozenRaider(run, rule="deep")
    env = RoboKabaddiEnv(**dict(pol.env_kwargs, defender_type="gate", gate_style=GATES[gate], **C3_RULES,
                                n_defenders=k, **KW))
    bad_before, bad_after, bad_dir, n_changed, n_before, n_after = 0, 0, 0, 0, 0, 0
    for ep in range(15):
        obs, _ = env.reset(seed=50000 + ep)
        while True:
            a_pol = pol.act(env, obs)
            a = rul.act(env, obs)
            touched = bool(env._flags["contact"])
            left = (None if env._raid_start_step is None
                    else (env._raid_steps - (env._step - env._raid_start_step)) * env.dt)
            active = not touched and (left is None or left > RULE_HOME_S)
            if active:
                n_before += 1
                sx, sy, f = hardware_command(float(a[0]), float(a[1]))
                bad_before += sy < 0 or (sx == 0 and sy == 0)
                if not np.allclose(a, a_pol):
                    n_changed += 1
                    r = env._get_pos(env.raider_id)[:2]
                    d = min((env._get_pos(i)[:2] for i in env.defender_ids),
                            key=lambda q: float(np.hypot(*(q - r))))
                    v = d - r
                    ok_dir = (f == 1.0 and (sx == 0 or sx == np.sign(v[0]))
                              and (sy == 0 or sy == np.sign(v[1])))
                    bad_dir += not ok_dir
            else:
                n_after += 1
                bad_after += not np.allclose(a, a_pol)
            obs, _, term, trunc, info = env.step(a)
            if term or trunc:
                break
    env.close()
    tag = f"{k}-def {gate}"
    check(f"{tag}: before the touch (>8 s left) never home or stopped ({n_before} decisions)", bad_before == 0)
    check(f"{tag}: changed commands go 100 % toward the nearest defender ({n_changed} changed)",
          bad_dir == 0 and n_changed == rul.rule_changed)
    check(f"{tag}: after the touch / last 8 s exactly the policy ({n_after} decisions)", bad_after == 0)

# 3. the rule function alone: no change after a touch
env = RoboKabaddiEnv(**dict(FrozenRaider(R3).env_kwargs, defender_type="gate", gate_style="v8", **C3_RULES))
env.reset(seed=50000)
env._flags["contact"] = True
a = np.array([0.0, -1.0, 0.0], dtype=np.float32)
check("deep_rule: after a touch a command home stays home", np.allclose(deep_rule(env, a)[0], a))
env._flags["contact"] = False
check("deep_rule: before the touch a command home is changed", deep_rule(env, a)[1])
env.close()

# 4. tools and robot runner
with tempfile.TemporaryDirectory() as tmp:
    out = Path(tmp) / "g.json"
    p = subprocess.run([PY, "marl/eval_gates.py", "--raiders", R3, "--gates", "v8", "--episodes", "3", "--procs", "1",
                        "--raider-rule", "deep", "--def-start", "1.0,1.5", "--out", str(out)],
                       cwd=ROOT, capture_output=True, text=True)
    ok = p.returncode == 0 and json.loads(out.read_text())[0]["raider_rule"] == "deep"
    check("eval_gates.py --raider-rule deep runs and records it", ok)
    out = Path(tmp) / "d.json"
    p = subprocess.run([PY, "marl/eval_defenders.py", "--ckpt",
                        str(R / "alt_wo_init/defence_s0/checkpoints/mappo_3999744_steps.pt"),
                        "--raiders", R3, "--episodes", "3", "--procs", "1", "--raider-rule", "deep",
                        "--out", str(out)], cwd=ROOT, capture_output=True, text=True)
    ok = p.returncode == 0 and json.loads(out.read_text())[0]["raider_rule"] == "deep"
    check("eval_defenders.py --raider-rule deep runs and records it", ok)
    if p.returncode:
        print(p.stderr[-1500:])
    out = Path(tmp) / "t.json"
    p = subprocess.run([PY, "marl/turnback.py", "--raiders", R3, "--gates", "v8", "--episodes", "3", "--procs", "1",
                        "--raider-rule", "deep", "--out", str(out)], cwd=ROOT, capture_output=True, text=True)
    ok = p.returncode == 0 and json.loads(out.read_text())[0]["raider_rule"] == "deep"
    check("turnback.py --raider-rule deep runs and records it", ok)
    ok = True
    for k in (1, 2, 3):
        p = subprocess.run([PY, "physical_experiment/rig/raid_runner.py", "--raider", "learned", "--defence", "learned",
                            "--n-def", str(k), "--raids", "1", "--raider-rule", "deep", "--def-start", "1.0,1.5",
                            "--fake", "--outdir", f"{tmp}/k{k}"], cwd=ROOT, capture_output=True, text=True)
        rows = list(Path(f"{tmp}/k{k}").rglob("raids.csv"))
        ok &= p.returncode == 0 and bool(rows) and ",deep" in rows[0].read_text()
        if p.returncode:
            print(p.stderr[-1500:])
    check("raid_runner.py --raider-rule deep --fake: 1, 2 and 3 defenders, recorded", ok)

print(f"\n{passed} passed, {failed} failed")
sys.exit(1 if failed else 0)
