"""Control-flow checks of the planar model against the recording mock.
The mock integrator is frictionless and has no contacts, so these check
timing, sub-stepping, parameters and data flow — not the physics itself."""
import importlib.util
import os
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


def check(name, cond, detail=''):
    global passed, failed
    if cond:
        passed += 1
        print(f'  pass  {name}')
    else:
        failed += 1
        print(f'  FAIL  {name}   {detail}')


# 1. default mode, sub-step count -------------------------------------------
env = E.RoboKabaddiEnv(n_defenders=3, defender_type='reactive')
check('default locomotion is planar', env.locomotion == 'planar', env.locomotion)
check('12 sub-steps per 0.05 s control step', env.n_substeps == 12, env.n_substeps)
check('env.dt still means the control period', env.dt == 0.05, env.dt)

# 2. physics timestep and robot parameters ----------------------------------
p.LOG.clear()
obs, _ = env.reset(seed=3)
ts = [c for c in p.LOG if c[0] == 'setTimeStep']
check('physics timestep is 1/240 s',
      ts and abs(ts[0][1][0] - 1 / 240) < 1e-12, ts)
robots = [c for c in p.LOG if c[0] == 'createMultiBody'
          and dict(c[2]).get('baseMass', 0) > 0]
masses = {dict(c[2])['baseMass'] for c in robots}
check('robot mass is 1.8 kg', masses == {1.8}, masses)
dyn = [dict(c[2]) for c in p.LOG if c[0] == 'changeDynamics']
check('floor friction zero on every robot',
      len(dyn) == 4 and all(d.get('lateralFriction') == 0.0 for d in dyn), dyn)
check('no damping on every robot',
      all(d.get('linearDamping') == 0.0 and d.get('angularDamping') == 0.0
          for d in dyn), dyn)
settle = sum(1 for c in p.LOG if c[0] == 'stepSimulation')
check('reset settles for 1 s of simulated time (240 sub-steps)',
      settle == 240, settle)

# 3. sub-stepping inside one control step -----------------------------------
p.LOG.clear()
env.step(np.array([0.0, 1.0, 0.0], dtype=np.float32))
n_phys = sum(1 for c in p.LOG if c[0] == 'stepSimulation')
check('one control step runs 12 physics steps', n_phys == 12, n_phys)
# each sub-step: one command reset, then one upright-hold reset per robot
raider_resets = sum(1 for c in p.LOG if c[0] == 'resetBaseVelocity'
                    and c[1][0] == env.raider_id)
raider_holds = sum(1 for c in p.LOG if c[0] == 'resetPose'
                   and c[1][0] == env.raider_id)
check('raider command re-applied at every sub-step',
      raider_resets - raider_holds == 12, (raider_resets, raider_holds))
uprights = sum(1 for c in p.LOG if c[0] == 'resetPose')
check('every robot held upright at every sub-step (4 x 12)',
      uprights == 48, uprights)

# 4. displacement equals command x time (mock is frictionless) ---------------
for cmd in (0.2, 0.4, 0.67, 1.0):
    env.reset(seed=5)
    start = env._get_pos(env.raider_id)[:2].copy()
    for _ in range(50):
        env.step(np.array([cmd, 0.0, 0.0], dtype=np.float32))
    moved = float(np.linalg.norm(env._get_pos(env.raider_id)[:2] - start))
    expect = cmd * env.MAX_LIN_VEL * env.dt * 50
    check(f'cmd {cmd:4.2f}: travels v*t = {expect:.3f} m',
          abs(moved - expect) < 1e-6, f'moved {moved:.6f}')

# 5. contact peak over sub-steps reaches the step and the observation -------
env.reset(seed=1)
seq = iter([(0.0, 0.0)] * 5 + [(0.9, 0.1)] + [(0.0, 0.0)] * 6)
env._fsr = lambda: next(seq)          # contact only in sub-step 6 of 12
obs, r, term, trunc, info = env.step(np.zeros(3, dtype=np.float32))
i = 6 + 2 * env.n_def
check('a contact in one sub-step sets the contact flag',
      env._flags['contact'] is True, env._flags['contact'])
check('observation carries the sub-step peak, not the last sample',
      abs(obs[i] - 0.9) < 1e-6 and abs(obs[i + 1] - 0.1) < 1e-6, obs[i:i+2])
del env._fsr
env.close()

# 6. patched defender controller still drives the planar model --------------
env = E.RoboKabaddiEnv(n_defenders=3, defender_type='reactive')
env.reset(seed=2)
env._reactive = lambda did, r: (0.123, -0.045)
p.LOG.clear()
env.step(np.zeros(3, dtype=np.float32))
d_cmds = [c for c in p.LOG if c[0] == 'resetBaseVelocity'
          and c[1][0] in env.defender_ids]
d_holds = sum(1 for c in p.LOG if c[0] == 'resetPose'
              and c[1][0] in env.defender_ids)
check('patched _reactive reaches every defender at every sub-step',
      len(d_cmds) - d_holds == 36
      and all(abs(c[1][1][0] - 0.123) < 1e-12 for c in d_cmds),
      (len(d_cmds), d_holds))
env.close()

# 7. environment-variable override ------------------------------------------
os.environ['ROBO_KABADDI_LOCOMOTION'] = 'legacy'
env = E.RoboKabaddiEnv(n_defenders=1)
check('ROBO_KABADDI_LOCOMOTION=legacy selects legacy',
      env.locomotion == 'legacy' and env.n_substeps == 1, env.locomotion)
env.close()
env = E.RoboKabaddiEnv(n_defenders=1, locomotion='planar')
check('explicit argument overrides the environment variable',
      env.locomotion == 'planar', env.locomotion)
env.close()
del os.environ['ROBO_KABADDI_LOCOMOTION']

try:
    E.RoboKabaddiEnv(n_defenders=1, locomotion='nonsense')
    check('an invalid mode is rejected', False)
except AssertionError:
    check('an invalid mode is rejected', True)

# 8. observation shape unchanged (trained policies' input width) -------------
for nd in (1, 3, 7):
    env = E.RoboKabaddiEnv(n_defenders=nd)
    o, _ = env.reset(seed=0)
    check(f'observation width unchanged for {nd} defender(s)',
          o.shape == (11 + 2 * nd,), o.shape)
    env.close()

print(f'\n{passed} passed, {failed} failed')
