"""
check_touch_obs.py — the raider's touch-made input (author, 12 Nov)
===================================================================

    python3 revision_experiments/env_tests/check_touch_obs.py

Old raiders: same inputs, saved gate results raid for raid. The flag: 0 before
the touch, 1 from the touch to the end, 0 again next raid. A widened raider
(raider_vs_learned.widened_raider) plays exactly as the old one. A short
training run with --touch-obs loads and plays in the gate test. Temporary
files in a temporary folder, removed at the end.
"""
import json
import subprocess
import sys
import tempfile
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path[:0] = [str(ROOT / "marl"), str(ROOT)]
from defender_env import FrozenRaider, raider_input, pad_raider_obs, diag_kwargs  # noqa: E402
from eval_gates import evaluate, C3_RULES, GATES                                  # noqa: E402
from raider_vs_learned import LearnedDefenceEnv, widened_raider                   # noqa: E402
from envs.kabaddi_env import RoboKabaddiEnv                                        # noqa: E402

passed = failed = 0
R = ROOT / "results/marl"
F = "checkpoints/mappo_3999744_steps.pt"
OLD3 = str(R / "alt_wo_r2/r2_raider_s42")
OLD1 = str(R / "diagvar/d1/raider2_s123")
VARY = diag_kwargs("0.35,0.85", 3)


def check(name, cond):
    global passed, failed
    print(("  ok   " if cond else "  FAIL ") + name, flush=True)
    passed += bool(cond)
    failed += not cond


def gate_env(r, gate="v8", n_def=None, dkw=None):
    return RoboKabaddiEnv(**dict(r.env_kwargs, defender_type="gate", gate_style=GATES[gate],
                                 **C3_RULES, **({"n_defenders": n_def} if n_def else {}),
                                 **(dkw or {})))


# 1. old raiders: no flag, same inputs, saved results raid for raid
r3 = FrozenRaider(OLD3)
check("old raider: no touch input", r3.touch is False)
env = gate_env(r3)
obs, _ = env.reset(seed=50000)
check("old raider input = before (17 numbers, same values)",
      np.array_equal(raider_input(env, obs, r3.n_in, r3.phantom, False),
                     pad_raider_obs(env, obs, r3.n_in, r3.phantom)) and len(obs) == 17)
env.close()
N = 30
saved = json.loads((R / "diagvar/d3/eval_gates_fixed.json").read_text())
s = [x for x in saved if x["raider"].endswith("alt_wo_r2/r2_raider_s42") and x["gate"] == "v8"][0]
now = evaluate((OLD3, "v8", N, None, None, None, {}))
check(f"accepted 3-def raider, v8, 1.0: {N} raids as saved", now["outcomes"] == s["outcomes"][:N])
saved = json.loads((R / "diagvar/d1/eval2_gates_vary.json").read_text())
s = [x for x in saved if x["raider"].endswith("d1/raider2_s123") and x["gate"] == "v10"][0]
now = evaluate((OLD1, "v10", N, 1, "copy", None, VARY))
check(f"1-def raider (copy phantoms), v10, varying speed: {N} raids as saved",
      now["outcomes"] == s["outcomes"][:N])

# 2. the flag through a raid
env = gate_env(r3)
ok_flag, ok_rest, ok_len, seen_touch = True, True, True, 0
for ep in range(6):
    obs, _ = env.reset(seed=50000 + ep)
    x = raider_input(env, obs, r3.n_in, r3.phantom, True)
    ok_flag &= x[-1] == 0.0
    was = 0.0
    while True:
        obs, _, term, trunc, info = env.step(r3.act(env, obs))
        x = raider_input(env, obs, r3.n_in, r3.phantom, True)
        ok_len &= len(x) == 18
        ok_rest &= np.array_equal(x[:-1], pad_raider_obs(env, obs, r3.n_in, r3.phantom))
        ok_flag &= x[-1] == float(env._flags["contact"]) and x[-1] >= was
        was = x[-1]
        if term or trunc:
            break
    seen_touch += was == 1.0
env.close()
check("flag is the 18th input, the first 17 unchanged", ok_len and ok_rest)
check("flag 0 at the start, = the touch-made flag, never back to 0 in a raid", ok_flag)
check(f"flag reaches 1 in touched raids ({seen_touch} of 6)", seen_touch >= 3)

# 3. widened raider = old raider (1-def, copy phantoms, varying speed)
from stable_baselines3.common.vec_env import DummyVecEnv  # noqa: E402
r1 = FrozenRaider(OLD1)
dfn = [str(R / "diagvar/d1/defence_s1" / F)]
venv = DummyVecEnv([lambda: LearnedDefenceEnv(dfn, 1, pad_to=3, phantom="copy",
                                              diag=VARY, touch=True)] * 4)
vn, model = widened_raider(Path(OLD1), venv)
vn.training = False
check("widened network: 18 inputs", model.policy.observation_space.shape == (18,))
check("widened: new input weight 0",
      all(float(abs(model.policy.state_dict()[k][:, -1]).max()) == 0.0
          for k in ("mlp_extractor.policy_net.0.weight", "mlp_extractor.value_net.0.weight")))
ea, eb = gate_env(r1, "v10", 1, VARY), gate_env(r1, "v10", 1, VARY)
max_diff, same = 0.0, True
for ep in range(15):
    oa, _ = ea.reset(seed=50000 + ep)
    ob, _ = eb.reset(seed=50000 + ep)
    while True:
        a = r1.act(ea, oa)
        x = raider_input(eb, ob, 3, "copy", True)
        b, _ = model.predict(vn.normalize_obs(x[None]).astype(np.float32)[0], deterministic=True)
        max_diff = max(max_diff, float(np.abs(a - b).max()))
        oa, _, ta, tra, ia = ea.step(a)
        ob, _, tb, trb, ib = eb.step(b)
        if ta or tra or tb or trb:
            same &= (ta or tra) == (tb or trb) and ia.get("outcome") == ib.get("outcome")
            break
ea.close(); eb.close(); venv.close()
check(f"widened raider plays as the old one: 15 raids, same results (largest action difference {max_diff:.1e})",
      same and max_diff < 1e-5)

# 4. a short training run with the flag, then the gate test
with tempfile.TemporaryDirectory() as tmp:
    out = Path(tmp) / "raider_touch"
    cmd = [sys.executable, str(ROOT / "marl/raider_vs_learned.py"), "--raider", OLD1,
           "--defence", ",".join(dfn), "--n-def", "1", "--pad-to", "3", "--raider-phantom", "copy",
           "--steps", "26000", "--seed", "42", "--ent", "0", "--empty", "-6", "--walkout", "-7",
           "--diag-range", "0.35,0.85", "--diag-period", "3", "--touch-obs", "--out", str(out)]
    p = subprocess.run(cmd, cwd=ROOT, capture_output=True, text=True)
    check("short training run with --touch-obs finishes", p.returncode == 0)
    if p.returncode:
        print(p.stderr[-2000:])
    else:
        info = json.loads((out / "run_info.json").read_text())
        check("run_info: raider_touch_obs, widened from the old raider",
              info["raider_touch_obs"] is True and info["widened_from"] == OLD1)
        rt = FrozenRaider(str(out))
        check("trained raider loads with the touch input", rt.touch and rt.vn.obs_rms.mean.shape == (18,))
        res = evaluate((str(out), "v8", 3, 1, "copy", None, VARY))
        check("trained raider plays the gate test", sum(res[k]["count"] for k in
                                                         ("touch", "tackle", "foul", "empty")) == 3)

print(f"\n{passed} passed, {failed} failed")
sys.exit(1 if failed else 0)
