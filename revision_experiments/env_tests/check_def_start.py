"""
check_def_start.py -- defenders' start depth (author, 13 Nov)
=============================================================

    python3 revision_experiments/env_tests/check_def_start.py

defender_start_depth (RoboKabaddiEnv) / --def-start (eval_gates.py,
eval_defenders.py, raid_runner.py, show_starts.py). Off: start places and
saved gate results as before, raid for raid. On (1.0, 1.5): every defender
1.0-1.5 m past the midline, across-court places and the raider's start the
same as the back row, robots apart as far as the back row allows; wrong
values refused; the test tools record the setting.
"""
import json
import subprocess
import sys
import tempfile
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path[:0] = [str(ROOT / "marl"), str(ROOT)]
from defender_env import diag_kwargs, start_kwargs                  # noqa: E402
from eval_gates import evaluate                                      # noqa: E402
import envs.kabaddi_env as E                                         # noqa: E402

passed = failed = 0
R = ROOT / "results/marl"
PY = sys.executable
DS = (1.0, 1.5)
VARY = diag_kwargs("0.35,0.85", 3)
N = 200


def check(name, cond):
    global passed, failed
    print(("  ok   " if cond else "  FAIL ") + name, flush=True)
    passed += bool(cond)
    failed += not cond


def starts(k, seed, **kw):
    env = E.RoboKabaddiEnv(**dict(E.CORRECTED_CONFIG, n_defenders=k, **kw))
    env.reset(seed=seed)
    P = np.array([env._get_pos(b)[:2] for b in [env.raider_id] + env.defender_ids])
    env.close()
    return P


# 1. off = as before
check("default: defender_start_depth is None", E.RoboKabaddiEnv(**E.CORRECTED_CONFIG).defender_start_depth is None)
check("start_kwargs off gives no setting", start_kwargs(None) == {} and start_kwargs("") == {})
y_lo, y_hi = E.COURT_L / 2 - E.BACK_ROW_FROM_WALL[1], E.COURT_L / 2 - E.BACK_ROW_FROM_WALL[0]
ok = all(np.all((P[1:, 1] >= y_lo) & (P[1:, 1] <= y_hi))
         for k in (1, 2, 3) for P in [starts(k, 50000 + i) for i in range(30)])
check("off: defenders in the back row (0.13-0.31 m from the end line)", ok)
saved = json.loads((R / "diagvar/d3/eval_gates_fixed.json").read_text())
s = [x for x in saved if x["raider"].endswith("alt_wo_r2/r2_raider_s42") and x["gate"] == "v8"][0]
now = evaluate((str(R / "alt_wo_r2/r2_raider_s42"), "v8", 30, None, None, None, {}))
check("off: accepted 3-def raider, v8, 1.0: 30 raids as saved", now["outcomes"] == s["outcomes"][:30])
saved = json.loads((R / "diagvar/d1/eval2_gates_vary.json").read_text())
s = [x for x in saved if x["raider"].endswith("d1/raider2_s123") and x["gate"] == "v10"][0]
now = evaluate((str(R / "diagvar/d1/raider2_s123"), "v10", 30, 1, "copy", None, VARY))
check("off: 1-def raider, v10, varying speed: 30 raids as saved", now["outcomes"] == s["outcomes"][:30])

# 2. on: depth, same across places and raider, spacing
# The drawn places are compared (before the 1 s settling): two defenders drawn
# closer than a robot's width are pushed apart by the settling, in the back
# row as well (a few raids in 600); those raids are counted.
_orig = E.RoboKabaddiEnv._defender_starts_back_row
_drawn = {}


def _spy(self):
    r = _orig(self)
    _drawn["v"] = np.array(r)
    return r


E.RoboKabaddiEnv._defender_starts_back_row = _spy
in_range, same_x, same_r = True, True, True
min_on, min_off = 9.0, 9.0
pushed = {"off": 0, "on": 0}
for k in (1, 2, 3):
    for i in range(N):
        a = starts(k, 50000 + i)
        da = _drawn["v"]
        b = starts(k, 50000 + i, defender_start_depth=DS)
        db = _drawn["v"]
        in_range &= bool(np.all((b[1:, 1] >= DS[0] - 1e-6) & (b[1:, 1] <= DS[1] + 1e-6)))
        same_x &= bool(np.allclose(da[:, 0], db[:, 0]))
        same_r &= bool(np.allclose(a[0], b[0]))
        pushed["off"] += not np.allclose(a[1:], da, atol=1e-3)
        pushed["on"] += not np.allclose(b[1:], db, atol=1e-3)
        for P, which in ((a, "off"), (b, "on")):
            d = min(np.hypot(*(P[i1] - P[i2])) for i1 in range(len(P)) for i2 in range(i1))
            if which == "on":
                min_on = min(min_on, d)
            else:
                min_off = min(min_off, d)
check(f"on: every defender 1.0-1.5 m past the midline (1, 2, 3 defenders, {N} raids each)", in_range)
E.RoboKabaddiEnv._defender_starts_back_row = _orig
check("on: drawn across-court places the same as the back row", same_x)
check(f"defenders drawn touching and pushed apart at the start: back row {pushed['off']}, "
      f"new start {pushed['on']} of {3 * N} raids (no more than before)", pushed["on"] <= pushed["off"])
check("on: raider's start the same", same_r)
check(f"on: closest two robots {min_on:.2f} m apart (back row {min_off:.2f} m; touching below "
      f"{2 * E.R_RADIUS:.2f} m)", min_on >= min_off - 1e-6)
check("on with varying speed: same start places", np.allclose(
    starts(3, 50007, defender_start_depth=DS), starts(3, 50007, defender_start_depth=DS, **VARY)))


def refused(**kw):
    try:
        E.RoboKabaddiEnv(**dict(E.CORRECTED_CONFIG, **kw)).close()
        return False
    except AssertionError:
        return True


check("court metres (3.0, 3.5) refused: the setting is metres past the midline",
      refused(defender_start_depth=(3.0, 3.5)))
check("low above high refused", refused(defender_start_depth=(1.5, 1.0)))
check("needs the back-row layout", refused(start_layout="legacy", defender_start_depth=DS))

# 3. test tools record the setting
with tempfile.TemporaryDirectory() as tmp:
    out = Path(tmp) / "g.json"
    p = subprocess.run([PY, "marl/eval_gates.py", "--raiders", str(R / "touchobs/d1/raider_s123"), "--gates", "v8",
                        "--n-def", "1", "--raider-phantom", "copy", "--episodes", "3", "--procs", "1",
                        "--diag-range", "0.35,0.85", "--diag-period", "3", "--def-start", "1.0,1.5",
                        "--out", str(out)], cwd=ROOT, capture_output=True, text=True)
    ok = p.returncode == 0 and json.loads(out.read_text())[0]["defender_start_depth"] == [1.0, 1.5]
    check("eval_gates.py --def-start runs and records it", ok)
    if p.returncode:
        print(p.stderr[-1500:])
    out = Path(tmp) / "d.json"
    p = subprocess.run([PY, "marl/eval_defenders.py", "--ckpt", str(R / "diagvar/d1/defence_s1/checkpoints/mappo_3999744_steps.pt"),
                        "--raiders", str(R / "touchobs/d1/raider_s123"), "--n-def", "1", "--raider-phantom", "copy",
                        "--episodes", "3", "--procs", "1", "--def-start", "1.0,1.5", "--out", str(out)],
                       cwd=ROOT, capture_output=True, text=True)
    ok = p.returncode == 0 and json.loads(out.read_text())[0]["defender_start_depth"] == [1.0, 1.5]
    check("eval_defenders.py --def-start runs and records it", ok)
    if p.returncode:
        print(p.stderr[-1500:])
    p = subprocess.run([PY, "physical_experiment/rig/raid_runner.py", "--raider", "learned", "--defence", "learned",
                        "--n-def", "3", "--raids", "1", "--def-start", "1.0,1.5", "--fake", "--outdir", tmp],
                       cwd=ROOT, capture_output=True, text=True)
    rows = list(Path(tmp).rglob("raids.csv"))
    steps = list(Path(tmp).rglob("steps.csv"))
    ok = p.returncode == 0 and rows and '"1.0,1.5"' in rows[0].read_text()
    if ok:
        first = steps[0].read_text().splitlines()[1].split(",")
        ys = [float(first[i]) for i in (6, 8, 10)]          # d1_y, d2_y, d3_y (court frame)
        ok = all(2.0 + DS[0] - 0.12 <= y <= 2.0 + DS[1] + 0.12 for y in ys)   # fake placing: within 10 cm
    check("raid_runner.py --def-start --fake: one raid, recorded, defenders start 3.0-3.5 m", bool(ok))
    if p.returncode:
        print(p.stderr[-1500:])

# 4. robot runner pairs: default the accepted pairs, --pairs 13nov the new ones
import match                                                        # noqa: E402
check("match.py: default pairs unchanged (accepted, 4 Nov)", match.PAIR_SETS["accepted"] is match.PAIRS
      and match.PAIRS[3][1].endswith("alt_wo_r2/r2_raider_s42")
      and match.PAIRS[1][1].endswith("one_def_copy/raider_s456"))
pl = match.Player("learned", match.PAIR_SETS["13nov"])
check("13nov pairs load; 1-def raider has the touch input and copy phantoms",
      pl.raider[1].touch and pl.raider[1].phantom == "copy" and not pl.raider[3].touch)
with tempfile.TemporaryDirectory() as tmp:
    ok = True
    for k in (1, 2, 3):
        p = subprocess.run([PY, "physical_experiment/rig/raid_runner.py", "--raider", "learned", "--defence", "learned",
                            "--pairs", "13nov", "--n-def", str(k), "--raids", "1", "--def-start", "1.0,1.5",
                            "--fake", "--outdir", f"{tmp}/k{k}"], cwd=ROOT, capture_output=True, text=True)
        rows = list(Path(f"{tmp}/k{k}").rglob("raids.csv"))
        ok &= p.returncode == 0 and bool(rows) and ",13nov" in rows[0].read_text()
        if p.returncode:
            print(p.stderr[-1500:])
    check("raid_runner.py --pairs 13nov --def-start 1.0,1.5 --fake: 1, 2 and 3 defenders, one raid each", ok)

print(f"\n{passed} passed, {failed} failed")
sys.exit(1 if failed else 0)
