"""Checks of court_walls and the out-of-court tackle (CORRECTIONS_TRACKER.md
A5), and of the author's choices in CORRECTED_CONFIG, against the recording
mock. The mock has no collisions, so the walls' physical effect is measured
with real PyBullet (exp_T_raider_command.py, Part 3)."""
import importlib.util
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent / 'mock'))
import pybullet as p

spec = importlib.util.spec_from_file_location(
    'env_new', str(Path(__file__).resolve().parents[2] / 'envs' / 'kabaddi_env.py'))
E = importlib.util.module_from_spec(spec)
spec.loader.exec_module(E)

passed = failed = 0
STOP = np.zeros(3, dtype=np.float32)


def check(name, cond, detail=''):
    global passed, failed
    if cond:
        passed += 1
        print(f'  pass  {name}')
    else:
        failed += 1
        print(f'  FAIL  {name}   {detail}')


def walls_built(**kw):
    env = E.RoboKabaddiEnv(n_defenders=3, **kw)
    p.LOG.clear()
    env.reset(seed=0)
    n = sum(1 for c in p.LOG if c[0] == 'colshape' and c[1][0] == p.GEOM_BOX
            and dict(c[2]).get('halfExtents', (0, 0, 0))[2] == 0.075)
    env.close()
    return n


check('default: four 0.15 m walls', walls_built() == 4, walls_built())
check('court_walls=False: no walls', walls_built(court_walls=False) == 0)
env = E.RoboKabaddiEnv(n_defenders=3)
check('court_walls defaults to True', env.court_walls is True)
env.close()


def one_step_at(x, y, rules='physical', walls=False):
    env = E.RoboKabaddiEnv(n_defenders=3, defender_type='static',
                           start_layout='back_row', outcome_rules=rules,
                           court_walls=walls)
    env.reset(seed=0)
    p.resetBasePositionAndOrientation(env.raider_id, [x, y, E.R_HEIGHT / 2],
                                      (0.0, 0.0, 0.0, 1.0))
    env._prev_pos = np.array([x, y], dtype=np.float32)
    _, r, term, _, info = env.step(STOP)
    env.close()
    return info.get('outcome'), term, r


W, L = E.COURT_W / 2, E.COURT_L / 2
for label, xy in (('over the left line, defenders\' half', (-W - 0.01, 0.8)),
                  ('over the right line, defenders\' half', (W + 0.01, 0.3)),
                  ('over the far end line', (0.0, L + 0.01))):
    o, term, r = one_step_at(*xy)
    check(f'raid started, centre {label}: Tackle at once', o == 'tackle' and term,
          (o, term))
o, term, r_out = one_step_at(W + 0.01, 0.8)
_, _, r_in = one_step_at(W - 0.01, 0.8)
check('...the Tackle penalty: reward differs from just inside by exactly -5',
      abs((r_out - r_in) - E.RoboKabaddiEnv.R_TACKLE) < 1e-6, (r_out, r_in))
for label, xy in (('over the right line', (W + 0.01, -0.8)),
                  ('over its own back line', (0.0, -L - 0.01))):
    o, term, r = one_step_at(*xy)
    check(f'before crossing, centre {label}: Empty (no raid)',
          o == 'empty' and term and r < E.RoboKabaddiEnv.R_EMPTY + 0.5, (o, term, r))
env = E.RoboKabaddiEnv(n_defenders=3, defender_type='static', start_layout='back_row',
                       outcome_rules='physical', court_walls=False)
env.reset(seed=0)
p.resetBasePositionAndOrientation(env.raider_id, [0.0, 0.5, E.R_HEIGHT / 2],
                                  (0.0, 0.0, 0.0, 1.0))
env.step(STOP)                                   # the raid starts
p.resetBasePositionAndOrientation(env.raider_id, [W + 0.01, -0.5, E.R_HEIGHT / 2],
                                  (0.0, 0.0, 0.0, 1.0))
_, _, term, _, info = env.step(STOP)
check('raid started, then out over a line in its own half: Tackle',
      term and info.get('outcome') == 'tackle', info)
env.close()
o, term, _ = one_step_at(W - 0.01, 0.8)
check('centre just inside the line: no Tackle', o is None and not term, (o, term))
o, term, _ = one_step_at(W + 0.01, 0.8, rules='section_2_3')
check('section_2_3 rules: no out-of-court Tackle (only "physical")',
      o is None and not term, (o, term))
o, term, _ = one_step_at(W + 0.01, 0.8, rules='legacy')
check('legacy rules: no out-of-court Tackle', o is None and not term, (o, term))

# defenders stay inside without walls (their 0.30 m margins) -----------------------
out = -np.inf
for dtype in ('physical', 'gate'):
    for seed in range(4):
        env = E.RoboKabaddiEnv(n_defenders=3, defender_type=dtype, court_walls=False,
                               start_layout='back_row')
        env.reset(seed=seed)
        for t in range(300):
            x = env._get_pos(env.raider_id)[0]
            env.step(np.array([1.0 if x < 1.0 else 0.0, 0.6, 0.0], dtype=np.float32))
            for d in env.defender_ids:
                q = env._get_pos(d)
                out = max(out, abs(q[0]) - W, abs(q[1]) - L)
        env.close()
check('no walls: defenders stay inside the lines (physical and gate, driven raids)',
      out < 0.0, out)

# CORRECTED_CONFIG holds the author's choices -----------------------------------------
C = E.CORRECTED_CONFIG
check('CORRECTED_CONFIG: camera contact at 0.25 m (A4)',
      C['contact_model'] == 'camera' and C['contact_dist'] == 0.25)
check('CORRECTED_CONFIG: counted 0.20 m past the midline after 2 steps (A2)',
      C['contact_min_depth'] == 0.20 and C['contact_confirm_steps'] == 2)
check('CORRECTED_CONFIG: no walls, physical rules with the out-of-court Tackle (A5)',
      C['court_walls'] is False and C['outcome_rules'] == 'physical')
check('CORRECTED_CONFIG: Empty as soon as the raider crosses back without contact',
      C['empty_on_return'] is True)
env = E.RoboKabaddiEnv(n_defenders=3, **C)
env.reset(seed=0)
for _ in range(10):
    env.step(np.array([0.0, 1.0, 0.0], dtype=np.float32))
check('CORRECTED_CONFIG builds and runs', env._step == 40)
env.close()

print(f'\n{passed} passed, {failed} failed')
