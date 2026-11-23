"""Control-flow checks of the contact rules (contact_model, contact_dist,
contact_min_depth, contact_confirm_steps) against the recording mock. The
mock has no contact forces, so force contact is injected by replacing _fsr;
camera contact comes from placed positions."""
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
BACK = np.array([0.0, -1.0, 0.0], dtype=np.float32)
N = 3
I_FSR = 6 + 2 * N          # observation index of the left contact channel


def check(name, cond, detail=''):
    global passed, failed
    if cond:
        passed += 1
        print(f'  pass  {name}')
    else:
        failed += 1
        print(f'  FAIL  {name}   {detail}')


def make(**kw):
    env = E.RoboKabaddiEnv(n_defenders=N, defender_type='static',
                           start_layout='back_row', **kw)
    env.reset(seed=0)
    return env


def place(env, bid, x, y):
    p.resetBasePositionAndOrientation(
        bid, [x, y, E.R_HEIGHT / 2], (0.0, 0.0, 0.0, 1.0))


def raider_at(env, y, x=0.0):
    place(env, env.raider_id, x, y)
    env._prev_pos = np.array([x, y], dtype=np.float32)


def press(env, on):
    env._fsr = (lambda: (0.9, 0.0)) if on else (lambda: (0.0, 0.0))


def step(env, a=STOP):
    return env.step(a)


def rejected(**kw):
    try:
        E.RoboKabaddiEnv(n_defenders=1, **kw)
        return False
    except AssertionError:
        return True


# 1. defaults and validation ---------------------------------------------------
env = make()
check('defaults: force model, any depth, 1 step',
      (env.contact_model, env.contact_min_depth, env.contact_confirm_steps)
      == ('force', None, 1))
env.close()
check('camera without contact_dist rejected', rejected(contact_model='camera'))
check('camera with contact_dist 0 rejected',
      rejected(contact_model='camera', contact_dist=0.0))
check('unknown contact model rejected', rejected(contact_model='sonar'))
check('contact_confirm_steps 0 rejected', rejected(contact_confirm_steps=0))
check('contact_confirm_steps 1.5 rejected', rejected(contact_confirm_steps=1.5))
check('negative contact_min_depth rejected', rejected(contact_min_depth=-0.1))

# 2. defaults keep the old behaviour --------------------------------------------
env = make()
raider_at(env, -0.5)                       # in its own half
press(env, True)
o, *_ = step(env)
check('default: any force contact counts at once, anywhere',
      env._flags['contact'] is True)
check('default: observation carries the force (0.9 N), not 1.0',
      abs(o[I_FSR] - 0.9) < 1e-6, o[I_FSR])
env.close()

# 3. depth ----------------------------------------------------------------------
env = make(contact_min_depth=0.27)
raider_at(env, 0.20)
press(env, True)
step(env)
check('depth 0.27: contact 0.20 m past the midline does not count',
      env._flags['contact'] is False and env._contact_count == 0)
raider_at(env, 0.30)
step(env)
check('depth 0.27: contact 0.30 m past the midline counts',
      env._flags['contact'] is True)
env.close()

# 4. confirmation, cumulative ----------------------------------------------------
env = make(contact_confirm_steps=2)
raider_at(env, 0.5)
press(env, True)
step(env)
check('confirm 2: one contact step is not enough', env._flags['contact'] is False)
press(env, False)
step(env)
step(env)
check('confirm 2: steps without contact do not reset the count',
      env._contact_count == 1)
press(env, True)
step(env)
check('confirm 2: a second, separate contact step counts',
      env._flags['contact'] is True)
press(env, False)
for _ in range(5):
    step(env)
check('counted contact stays counted', env._flags['contact'] is True)
env.reset(seed=0)
check('reset clears the count', env._contact_count == 0
      and env._flags['contact'] is False)
env.close()

env = make(contact_confirm_steps=2, contact_min_depth=0.27)
raider_at(env, 0.10)
press(env, True)
step(env)
step(env)
check('shallow contact steps do not add to the count',
      env._contact_count == 0 and env._flags['contact'] is False)
env.close()

# 5. camera distance --------------------------------------------------------------
for dist, expect in ((0.35, True), (0.25, False)):
    env = make(contact_model='camera', contact_dist=dist)
    raider_at(env, 0.5)
    place(env, env.defender_ids[0], 0.0, 0.80)   # centres 0.30 m apart
    o, *_ = step(env)
    check(f'camera {dist} m, centres 0.30 m apart: contact {expect}',
          env._flags['contact'] is expect)
    check(f'camera {dist} m: observation left = {float(expect)}, right = 0.0',
          o[I_FSR] == float(expect) and o[I_FSR + 1] == 0.0, o[I_FSR:I_FSR + 2])
    env.close()

env = make(contact_model='camera', contact_dist=0.35)
o0 = env._obs()
check('camera: observation 0.0 at reset with defenders far away',
      o0[I_FSR] == 0.0 and o0.shape == (11 + 2 * N,))
press(env, True)                                  # a force alone is ignored
raider_at(env, 0.5)
step(env)
check('camera: force without proximity does not count',
      env._flags['contact'] is False)
env.close()

# 6. observation is raw proximity, game contact follows the rules ----------------
env = make(contact_model='camera', contact_dist=0.35, contact_min_depth=0.27,
           contact_confirm_steps=2)
raider_at(env, 0.10)
place(env, env.defender_ids[0], 0.0, 0.35)       # 0.25 m apart, shallow
o, *_ = step(env)
check('camera: shallow proximity shows in the observation (1.0)...',
      o[I_FSR] == 1.0)
check('...but does not count for the game', env._flags['contact'] is False)
env._flags['midline'] = True
info = {}
for _ in range(30):
    _, _, term, _, info = step(env, BACK)
    if term:
        break
check('raider returns home after an uncounted contact: no Touch',
      info.get('outcome') != 'touch' and env._flags['contact'] is False,
      info)
env.close()

print(f'\n{passed} passed, {failed} failed')
