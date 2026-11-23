"""
check_diag_range.py — per-robot diagonal speed (diag_ratio_range, author 11 Nov)
================================================================================

    python3 revision_experiments/env_tests/check_diag_range.py
"""
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from envs.kabaddi_env import (RoboKabaddiEnv, CORRECTED_CONFIG, DEF_ACTIONS,  # noqa: E402
                              hardware_command)

passed = failed = 0


def check(name, cond):
    global passed, failed
    print(("  ok   " if cond else "  FAIL ") + name)
    passed += bool(cond)
    failed += not cond


def learned(**kw):
    return RoboKabaddiEnv(**dict(CORRECTED_CONFIG, n_defenders=3, defender_type="learned",
                                 defender_fouls=True, held_needs_touch=True, **kw))


RANGE = (0.35, 0.85)
DIAG = [i for i, (sx, sy) in enumerate(DEF_ACTIONS) if sx and sy][0]
STRAIGHT = [i for i, (sx, sy) in enumerate(DEF_ACTIONS) if sy and not sx][0]

off, on = learned(), learned(diag_ratio_range=RANGE)
off.reset(seed=123)
on.reset(seed=123)
check("default: no per-robot ratios", off._diag is None and off._ratio(0) == 1.0)
r = on._diag
check("one ratio per robot (raider + 3 defenders)", r is not None and len(r) == 4)
check("ratios within the range", all(RANGE[0] <= v <= RANGE[1] for v in r))
check("robots get different ratios", len(set(np.round(r, 6))) == 4)
same = all(np.allclose(off._get_pos(a)[:2], on._get_pos(b)[:2])
           for a, b in zip([off.raider_id] + off.defender_ids, [on.raider_id] + on.defender_ids))
check("start positions as without the setting (same seed)", same)
on2 = learned(diag_ratio_range=RANGE)
on2.reset(seed=123)
check("same seed, same ratios", np.allclose(on2._diag, r))
on2.reset(seed=124)
check("another seed, other ratios", not np.allclose(on2._diag, r))

# defenders: each diagonal at its own ratio, straight unchanged
on.set_defender_actions([DIAG] * 3)
v = on._defender_commands()
V = on.DEF_MAX_LIN_VEL
check("defender i diagonal speed = V x its ratio",
      all(abs(np.hypot(*v[i]) - V * r[i + 1]) < 1e-9 for i in range(3)))
on.set_defender_actions([STRAIGHT] * 3)
check("defender straight speed unchanged", all(abs(np.hypot(*c) - V) < 1e-9 for c in on._defender_commands()))
check("no robot left selected after the commands", on._cur_robot is None)

# raider: hardware diagonal at its own ratio
a = np.array([1.0, 1.0, 0.0])
sx, sy, frac = hardware_command(1.0, 1.0)
vx, vy, _ = on._raider_velocity(a)
check("raider diagonal speed = level x V x its ratio",
      abs(np.hypot(vx, vy) - frac * on.MAX_LIN_VEL * r[0]) < 1e-9)
vx, vy, _ = on._raider_velocity(np.array([0.0, 1.0, 0.0]))
check("raider straight speed unchanged", abs(np.hypot(vx, vy) - on.MAX_LIN_VEL) < 1e-9)

# default env: diagonal speed as before (diagonal_speed_ratio)
off.set_defender_actions([DIAG] * 3)
check("default: defender diagonal at diagonal_speed_ratio",
      all(abs(np.hypot(*c) - V) < 1e-9 for c in off._defender_commands()))

# period: new ratios every 3 s
per = learned(diag_ratio_range=RANGE, diag_ratio_period=3.0)
per.reset(seed=5)
first = per._diag.copy()
for _ in range(14):                       # 14 x 0.2 s = 2.8 s
    per.step(np.zeros(3))
check("same ratios before 3 s", np.allclose(per._diag, first))
for _ in range(2):                        # past 3 s
    per.step(np.zeros(3))
check("new ratios after 3 s", not np.allclose(per._diag, first))
once = learned(diag_ratio_range=RANGE)
once.reset(seed=5)
f2 = once._diag.copy()
for _ in range(30):
    once.step(np.zeros(3))
check("no period: same ratios for the whole raid", np.allclose(once._diag, f2))

for e in (off, on, on2, per, once):
    e.close()
print(f"\n{passed} passed, {failed} failed")
sys.exit(1 if failed else 0)
