"""Checks of raider_command, action_repeat, raider_start and CORRECTED_CONFIG
(CORRECTIONS_TRACKER.md A6, P8) against the recording mock. The command
mapping is compared with the robots' own smorphi_controller.action_to_cmd,
loaded with its network module stubbed."""
import importlib.util
import sys
import types
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE / 'mock'))
import pybullet as p

REPO = HERE.parents[1]
spec = importlib.util.spec_from_file_location('env_new', str(REPO / 'envs' / 'kabaddi_env.py'))
E = importlib.util.module_from_spec(spec)
spec.loader.exec_module(E)

passed = failed = 0


def check(name, cond, detail=''):
    global passed, failed
    if cond:
        passed += 1
        print(f'  pass  {name}')
    else:
        failed += 1
        print(f'  FAIL  {name}   {detail}')


# 1. defaults and validation ----------------------------------------------------
env = E.RoboKabaddiEnv(n_defenders=3)
check('defaults: continuous commands, action_repeat 1, legacy start',
      (env.raider_command, env.action_repeat, env.raider_start)
      == ('continuous', 1, 'legacy'))
env.close()
for kw in (dict(raider_command='joystick'), dict(action_repeat=0),
           dict(action_repeat=2.5), dict(raider_start='anywhere')):
    try:
        E.RoboKabaddiEnv(n_defenders=1, **kw)
        check(f'{kw} rejected', False)
    except AssertionError:
        check(f'{kw} rejected', True)

# 2. mapping == the robots' action_to_cmd ------------------------------------------
sys.modules['fsr_listener'] = types.ModuleType('fsr_listener')
sspec = importlib.util.spec_from_file_location(
    'smorphi_controller', str(REPO / 'physical_experiment' / 'smorphi_controller.py'))
SC = importlib.util.module_from_spec(sspec)
sspec.loader.exec_module(SC)
FRAC = (0.25, 0.5, 0.75, 1.0)
DECODE = {SC.CMD_STOP: (0, 0, 0.0)}
for table, sx, sy in ((SC.CMD_FORWARD, 0, 1), (SC.CMD_BACKWARD, 0, -1),
                      (SC.CMD_RIGHT, -1, 0), (SC.CMD_LEFT, 1, 0),
                      (SC.CMD_DIAG_FWD_R, -1, 1), (SC.CMD_DIAG_FWD_L, 1, 1),
                      (SC.CMD_DIAG_BCK_R, -1, -1), (SC.CMD_DIAG_BCK_L, 1, -1)):
    for i, ch in enumerate(table):
        DECODE[ch] = (sx, sy, FRAC[i])   # raider: FLIP_VX, "right" = court -x
grid = np.round(np.linspace(-1, 1, 81), 6)
bad = [(a, b) for a in grid for b in grid
       if E.hardware_command(a, b) != DECODE[SC.action_to_cmd([a, b, 0.7])]]
check(f'hardware_command == action_to_cmd on {len(grid) ** 2} actions (turn ignored)',
      not bad, bad[:4])


# 3. the raider moves at the mapped velocity ----------------------------------------
def raider_vel(action, **kw):
    env = E.RoboKabaddiEnv(n_defenders=3, defender_type='static',
                           raider_command='hardware', **kw)
    env.reset(seed=0)
    p.LOG.clear()
    env.step(np.array(action, dtype=np.float32))
    v = [c[1][1] for c in p.LOG if c[0] == 'resetBaseVelocity'
         and c[1][0] == env.raider_id and c[1][2] is not None][0]
    w = [c[1][2] for c in p.LOG if c[0] == 'resetBaseVelocity'
         and c[1][0] == env.raider_id and c[1][2] is not None][0]
    env.close()
    return np.array(v[:2]), w[2]


v, _ = raider_vel([1.0, 1.0, 0.0])
check('diagonal (1, 1): 0.30 m/s, not 0.42', abs(np.hypot(*v) - 0.30) < 1e-9,
      np.hypot(*v))
v, _ = raider_vel([1.0, 1.0, 0.0], diagonal_speed_ratio=0.71)
check('diagonal with ratio 0.71: 0.213 m/s', abs(np.hypot(*v) - 0.213) < 1e-9,
      np.hypot(*v))
v, _ = raider_vel([0.5, 0.0, 0.0])
check('(0.5, 0): straight at the 75% level, 0.225 m/s',
      np.allclose(v, [0.225, 0.0]), v)
v, _ = raider_vel([0.13, 0.9, 0.0])
check('(0.13, 0.9): a 0.13 side component makes it a full-speed diagonal',
      np.allclose(np.sign(v), [1, 1]) and abs(np.hypot(*v) - 0.30) < 1e-9, v)
v, _ = raider_vel([0.1, 0.11, 0.0])
check('inside the 0.12 dead zone: stop', np.allclose(v, [0, 0]), v)
_, w = raider_vel([0.0, 1.0, 1.0])
check('turn command ignored (pivot disabled, as on the robots)', w == 0.0, w)
env = E.RoboKabaddiEnv(n_defenders=3, defender_type='static')
env.reset(seed=0)
p.LOG.clear()
env.step(np.array([1.0, 1.0, 0.5], dtype=np.float32))
v = [c[1][1] for c in p.LOG if c[0] == 'resetBaseVelocity' and c[1][0] == env.raider_id][0]
check('continuous (default) unchanged: diagonal 0.42 m/s',
      abs(np.hypot(*v[:2]) - 0.3 * np.sqrt(2)) < 1e-9, np.hypot(*v[:2]))
env.close()

# 4. action_repeat ----------------------------------------------------------------
A = np.array([0.4, 0.8, 0.0], dtype=np.float32)
env = E.RoboKabaddiEnv(n_defenders=3, defender_type='physical', action_repeat=4)
env.reset(seed=4)
p.LOG.clear()
o4, r4, *_ = env.step(A)
n_phys = sum(1 for c in p.LOG if c[0] == 'stepSimulation')
check('action_repeat 4: one step = 4 control steps = 48 physics sub-steps',
      env._step == 4 and n_phys == 48, (env._step, n_phys))
env.close()
env = E.RoboKabaddiEnv(n_defenders=3, defender_type='physical')
env.reset(seed=4)
rs = [env.step(A) for _ in range(4)]
check('...same observation and summed reward as 4 single steps',
      np.array_equal(rs[-1][0], o4) and abs(sum(r[1] for r in rs) - r4) < 1e-9,
      (sum(r[1] for r in rs), r4))
env.close()
env = E.RoboKabaddiEnv(n_defenders=3, defender_type='static', action_repeat=4)
env.reset(seed=0)
p.resetBasePositionAndOrientation(env.raider_id, [0.0, 0.01, E.R_HEIGHT / 2],
                                  (0.0, 0.0, 0.0, 1.0))
env._flags.update(midline=True, contact=True)
_, _, term, _, info = env.step(np.array([0.0, -1.0, 0.0], dtype=np.float32))
check('raid ends mid-decision: step returns at once (Touch after 1 control step)',
      term and info.get('outcome') == 'touch' and env._step == 1,
      (term, info, env._step))
env.close()

# 5. raider_start ------------------------------------------------------------------
xs, ys, same_def = [], [], True
for seed in range(50):
    e1 = E.RoboKabaddiEnv(n_defenders=3, raider_start='near_midline')
    e1.reset(seed=seed)
    x, y = e1._get_pos(e1.raider_id)[:2]
    d1 = [tuple(e1._get_pos(d)[:2]) for d in e1.defender_ids]
    e1.close()
    e2 = E.RoboKabaddiEnv(n_defenders=3)
    e2.reset(seed=seed)
    d2 = [tuple(e2._get_pos(d)[:2]) for d in e2.defender_ids]
    e2.close()
    xs.append(x); ys.append(y); same_def &= d1 == d2
check('near_midline start: 0.15-0.41 m before the midline, x -0.63 to +0.36',
      min(ys) >= -0.41 and max(ys) <= -0.15 and min(xs) >= -0.63 and max(xs) <= 0.36,
      (min(ys), max(ys), min(xs), max(xs)))
check('defender starts unchanged by the raider start (same seeds)', same_def)

# 6. CORRECTED_CONFIG ----------------------------------------------------------------
env = E.RoboKabaddiEnv(n_defenders=3, **E.CORRECTED_CONFIG)
o, _ = env.reset(seed=0)
for _ in range(10):
    o, r, term, trunc, info = env.step(np.array([0.0, 1.0, 0.0], dtype=np.float32))
check('CORRECTED_CONFIG builds and runs (planar_force, box, gate v6, hardware, 5 Hz)',
      (env.locomotion, env.robot_shape, env.def_type, env.gate_style,
       env.raider_command, env.action_repeat) ==
      ('planar_force', 'box', 'gate', 'keepout', 'hardware', 4)
      and env._step == 40 and o.shape == (11 + 2 * 3,))
env.close()

print(f'\n{passed} passed, {failed} failed')
