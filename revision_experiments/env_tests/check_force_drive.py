"""Control-flow checks of locomotion='planar_force' against the recording mock:
a capped drive force on every robot at every sub-step, free motion reaching
the commanded speed, and the gain set from push_yield. The mock has no
contacts, so pushing between robots is measured with real PyBullet
(exp_Q_blocking.py)."""
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
FWD = np.array([0.0, 1.0, 0.0], dtype=np.float32)


def check(name, cond, detail=''):
    global passed, failed
    if cond:
        passed += 1
        print(f'  pass  {name}')
    else:
        failed += 1
        print(f'  FAIL  {name}   {detail}')


def make(**kw):
    env = E.RoboKabaddiEnv(n_defenders=3, defender_type='static',
                           locomotion='planar_force', **kw)
    env.reset(seed=0)
    return env


# 1. construction ------------------------------------------------------------
env = make()
check('planar_force: 12 sub-steps of 1/240 s, 1.8 kg like planar',
      env.n_substeps == 12 and env.physics_dt == 1 / 240)
check('gain K = 1.7 / (0.11 x 0.30) = 51.5 N per m/s',
      abs(env._drive_gain - 1.7 / (0.11 * 0.30)) < 1e-9, env._drive_gain)
check('defaults: push_yield 0.11, drive_force 1.7 N',
      env.push_yield == 0.11 and env.drive_force == 1.7)
env.close()
env = E.RoboKabaddiEnv(n_defenders=1)
check('default locomotion is still planar', env.locomotion == 'planar')
env.close()
for kw in (dict(push_yield=0.0), dict(push_yield=1.0), dict(drive_force=0.0)):
    try:
        E.RoboKabaddiEnv(n_defenders=1, locomotion='planar_force', **kw)
        check(f'{kw} rejected', False)
    except AssertionError:
        check(f'{kw} rejected', True)

# 2. one control step: drive forces, not velocity resets -----------------------
env = make()
p.LOG.clear()
env.step(FWD)
forces = [c for c in p.LOG if c[0] == 'applyExternalForce']
check('one drive force per robot per sub-step (4 x 12)', len(forces) == 48,
      len(forces))
lin_sets = [c for c in p.LOG if c[0] == 'resetBaseVelocity'
            and c[1][1] is not None and c[1][0] == env.raider_id]
holds = sum(1 for c in p.LOG if c[0] == 'resetPose' and c[1][0] == env.raider_id)
check('raider velocity is never set to the command (only the upright hold)',
      len(lin_sets) == holds == 12, (len(lin_sets), holds))
r_f = [np.hypot(*c[1][2][:2]) for c in forces if c[1][0] == env.raider_id]
check('raider starts at the force cap, 1.7 N, and never exceeds it',
      abs(r_f[0] - 1.7) < 1e-9 and max(r_f) <= 1.7 + 1e-9, (r_f[0], max(r_f)))
check('forces are horizontal, in the world frame',
      all(c[1][2][2] == 0.0 and c[1][4] == p.WORLD_FRAME for c in forces))
d_f = [np.hypot(*c[1][2][:2]) for c in forces if c[1][0] != env.raider_id]
check('stopped defenders get no force when not pushed', max(d_f) == 0.0,
      max(d_f))
env.close()

# 2b. resisting: a pushed robot holds with the higher limit ----------------------
env = make()
did = env.defender_ids[0]
p.resetBaseVelocity(did, linearVelocity=[0.0, 0.2, 0.0])   # being pushed at 0.2 m/s
p.LOG.clear()
env.step(FWD)
f0 = [np.hypot(*c[1][2][:2]) for c in p.LOG
      if c[0] == 'applyExternalForce' and c[1][0] == did][0]
check('a stopped robot moving at 0.2 m/s resists with hold_force 5.1 N',
      abs(f0 - 5.1) < 1e-9, f0)
check('hold_force default 5.1 N, above drive_force',
      env.hold_force == 5.1 > env.drive_force)
env.close()
try:
    E.RoboKabaddiEnv(n_defenders=1, locomotion='planar_force', hold_force=1.7)
    check('hold_force not above drive_force rejected', False)
except AssertionError:
    check('hold_force not above drive_force rejected', True)

# 3. free motion -----------------------------------------------------------------
env = make()
y0 = env._get_pos(env.raider_id)[1]
v, t90 = [], None
for k in range(40):                                   # 2 s
    env.step(FWD)
    vy = float(env._get_vel(env.raider_id)[1])
    v.append(vy)
    if t90 is None and vy >= 0.9 * 0.30:
        t90 = (k + 1) * env.dt
check('free motion reaches the commanded 0.30 m/s', abs(v[-1] - 0.30) < 1e-3,
      v[-1])
check('time to 90% speed 0.25-0.35 s (logged robot: median 0.27 s)',
      t90 is not None and 0.25 <= t90 <= 0.35, t90)
moved = float(env._get_pos(env.raider_id)[1] - y0)
check('2 s at full command: 0.55-0.60 m (start-up costs a few cm)',
      0.55 <= moved <= 0.60, moved)
env.close()

# 4. gain follows push_yield -----------------------------------------------------
g = {y: E.RoboKabaddiEnv(n_defenders=1, locomotion='planar_force',
                         push_yield=y)._drive_gain for y in (0.06, 0.11, 0.19)}
check('lower push_yield = stiffer drive (0.06 > 0.11 > 0.19)',
      g[0.06] > g[0.11] > g[0.19], g)

# 5. other modes untouched -------------------------------------------------------
for loco in ('planar', 'legacy'):
    env = E.RoboKabaddiEnv(n_defenders=3, locomotion=loco)
    env.reset(seed=0)
    p.LOG.clear()
    env.step(FWD)
    check(f'{loco}: no drive forces applied',
          not any(c[0] == 'applyExternalForce' for c in p.LOG))
    env.close()

print(f'\n{passed} passed, {failed} failed')
