"""
check_hold_rule.py -- the held rule with --hold T,D,C (author, 14 Nov)
======================================================================

    python3 revision_experiments/env_tests/check_hold_rule.py

RoboKabaddiEnv hold_dist / hold_touch_dist (with static_time) and
defender_env.hold_kwargs. Off: saved gate results raid for raid, the old rule
(4 s, 5 cm, body contact). On (2 s, 0.25 m, 0.25 m): a raider kept in place with
a defender 0.24 m away (not in body contact) is held after 2 s; moving 20 cm
inside the area is still held; a defender 0.27 m away does not hold; the old
rule does not hold in the same 2.5 s. The tools record the setting.
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
from defender_env import hold_kwargs                                 # noqa: E402
from eval_gates import evaluate                                       # noqa: E402
import envs.kabaddi_env as E                                          # noqa: E402

passed = failed = 0
R = ROOT / "results/marl"
PY = sys.executable
NEW = hold_kwargs("2,0.25,0.25")


def check(name, cond):
    global passed, failed
    print(("  ok   " if cond else "  FAIL ") + name, flush=True)
    passed += bool(cond)
    failed += not cond


# 1. off = as before
env = E.RoboKabaddiEnv(**E.CORRECTED_CONFIG)
check("default: 5 cm, body contact, 4 s (CORRECTED_CONFIG)",
      env.hold_dist == E.STATIC_DIST and env.hold_touch_dist is None and env.static_time == 4.0)
env.close()
check("hold_kwargs off gives no setting; 2,0.25,0.25 parsed",
      hold_kwargs(None) == {} and NEW == {"static_time": 2.0, "hold_dist": 0.25, "hold_touch_dist": 0.25})
saved = json.loads((R / "diagvar/d3/eval_gates_fixed.json").read_text())
s = [x for x in saved if x["raider"].endswith("alt_wo_r2/r2_raider_s42") and x["gate"] == "v8"][0]
now = evaluate((str(R / "alt_wo_r2/r2_raider_s42"), "v8", 30, None, None, None, {}))
check("off: accepted 3-def raider, v8, 1.0: 30 raids as saved", now["outcomes"] == s["outcomes"][:30])


# 2. the rule: raider placed 1.0 m past the midline, one defender at a set distance
def hold_case(kw, gap, wander=0.0, secs=2.5):
    env = E.RoboKabaddiEnv(**dict(E.CORRECTED_CONFIG, n_defenders=1, defender_type="static",
                                  held_needs_touch=True, **kw))          # as the robots (C3 rules)
    env.reset(seed=50000)
    env._flags["midline"] = True
    env._raid_start_step = env._step
    n = int(round(secs / env.dt))
    for i in range(n):
        dx = wander * np.sin(2 * np.pi * i / 20)                 # back and forth inside the area
        p.resetBasePositionAndOrientation(env.raider_id, [dx, 1.0, E.R_HEIGHT / 2], [0, 0, 0, 1],
                                          physicsClientId=env._client)
        p.resetBasePositionAndOrientation(env.defender_ids[0], [dx + gap, 1.0, E.R_HEIGHT / 2], [0, 0, 0, 1],
                                          physicsClientId=env._client)
        for b in (env.raider_id, env.defender_ids[0]):
            p.resetBaseVelocity(b, [0, 0, 0], [0, 0, 0], physicsClientId=env._client)
        out = env._static_tackle(env._get_pos(env.raider_id))
        env._step += 1
        if out:
            env.close()
            return i * env.dt
    env.close()
    return None


t = hold_case(NEW, 0.24)
check(f"new rule: defender 0.24 m away (no body contact) holds the raider after 2 s (held at {t})",
      t is not None and 1.95 <= t <= 2.2)
t = hold_case(NEW, 0.24, wander=0.10)
check(f"new rule: raider moving +-10 cm inside the 25 cm area is still held ({t})", t is not None)
check("new rule: defender 0.27 m away does not hold", hold_case(NEW, 0.27) is None)
check("old rule (4 s, 5 cm, body contact): no hold in the same 2.5 s", hold_case({}, 0.24) is None)
check("old rule: still no hold at 0.24 m after 5 s (no body contact)", hold_case({}, 0.24, secs=5.0) is None)

# 3. tools
with tempfile.TemporaryDirectory() as tmp:
    out = Path(tmp) / "g.json"
    r = subprocess.run([PY, "marl/eval_gates.py", "--raiders", str(R / "one_def_copy/raider_s456"), "--gates", "v8",
                        "--n-def", "1", "--episodes", "3", "--procs", "1", "--hold", "2,0.25,0.25", "--out", str(out)],
                       cwd=ROOT, capture_output=True, text=True)
    check("eval_gates.py --hold runs and records it",
          r.returncode == 0 and json.loads(out.read_text())[0]["hold"] == [2.0, 0.25, 0.25])
    out = Path(tmp) / "d.json"
    r = subprocess.run([PY, "marl/eval_defenders.py", "--ckpt",
                        str(R / "one_def_copy/defence_s0/checkpoints/mappo_3999744_steps.pt"),
                        "--raiders", "scripted", "--n-def", "1", "--episodes", "3", "--procs", "1",
                        "--hold", "2,0.25,0.25", "--out", str(out)], cwd=ROOT, capture_output=True, text=True)
    check("eval_defenders.py --hold runs and records it",
          r.returncode == 0 and json.loads(out.read_text())[0]["hold"] == [2.0, 0.25, 0.25])
    r = subprocess.run([PY, "physical_experiment/rig/raid_runner.py", "--raider", "learned", "--defence", "learned",
                        "--n-def", "2", "--raids", "1", "--hold", "2,0.25,0.25", "--fake", "--outdir", tmp + "/rr"],
                       cwd=ROOT, capture_output=True, text=True)
    rows = list(Path(tmp, "rr").rglob("raids.csv"))
    check("raid_runner.py --hold --fake runs and records it",
          r.returncode == 0 and bool(rows) and "2,0.25,0.25" in rows[0].read_text())
    if r.returncode:
        print(r.stderr[-1500:])

print(f"\n{passed} passed, {failed} failed")
sys.exit(1 if failed else 0)
