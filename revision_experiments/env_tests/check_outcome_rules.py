"""Control-flow checks of outcome_rules='section_2_3' against the recording
mock. Scripted raider paths decide each outcome; the mock has no contacts, so
contact is set by hand. Defenders are static, so only the raider moves."""
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
BACK = np.array([0.0, -1.0, 0.0], dtype=np.float32)
STOP = np.zeros(3, dtype=np.float32)


def check(name, cond, detail=''):
    global passed, failed
    if cond:
        passed += 1
        print(f'  pass  {name}')
    else:
        failed += 1
        print(f'  FAIL  {name}   {detail}')


def run(rules, policy, start_y=-0.5):
    """Run one raid. policy(env, k) -> action, may also set flags.
    Returns dict(steps, outcome, last_r, term, trunc, cross)."""
    env = E.RoboKabaddiEnv(n_defenders=3, defender_type='static',
                           start_layout='back_row', outcome_rules=rules)
    env.reset(seed=0)
    p.resetBasePositionAndOrientation(
        env.raider_id, [0.0, start_y, E.R_HEIGHT / 2], (0.0, 0.0, 0.0, 1.0))
    env._prev_pos = np.array([0.0, start_y], dtype=np.float32)
    cross = None
    k = 0
    while True:
        k += 1
        _, r, term, trunc, info = env.step(policy(env, k))
        if cross is None and env._flags['midline']:
            cross = k
        if term or trunc:
            break
    out = dict(steps=k, outcome=info.get('outcome'), last_r=r, term=term,
               trunc=trunc, cross=cross, max_steps=env.max_steps)
    env.close()
    return out


def park_past(env, k):
    """Drive forward until 0.3 m past the midline, then stand still."""
    return FWD if env._get_pos(env.raider_id)[1] < 0.3 else STOP


def cross_and_return(env, k):
    y = env._get_pos(env.raider_id)[1]
    if not getattr(env, '_went', False):
        if y < 0.3:
            return FWD
        env._went = True
    return BACK if y > -0.3 else STOP


def touch_and_return(env, k):
    y = env._get_pos(env.raider_id)[1]
    if y >= 0.3:
        env._flags['contact'] = True       # mock has no contacts
    return BACK if env._flags['contact'] else FWD


def contact_then_park(env, k):
    y = env._get_pos(env.raider_id)[1]
    if y >= 0.3:
        env._flags['contact'] = True
        return STOP
    return FWD


never_cross = lambda env, k: STOP
R_TIME, R_TACKLE, R_EMPTY = (E.RoboKabaddiEnv.R_TIME, E.RoboKabaddiEnv.R_TACKLE,
                             E.RoboKabaddiEnv.R_EMPTY)

# 1. construction -------------------------------------------------------------
env = E.RoboKabaddiEnv(n_defenders=3)
check('outcome rules default to legacy', env.outcome_rules == 'legacy')
check('legacy episode limit unchanged: 600 steps', env.max_steps == 600,
      env.max_steps)
env.close()
env = E.RoboKabaddiEnv(n_defenders=3, outcome_rules='section_2_3')
check('section_2_3 limit: 20 s approach + 30 s raid = 1000 steps',
      env.max_steps == 1000 and env._approach_steps == 400
      and env._raid_steps == 600, (env.max_steps,))
env.close()
try:
    E.RoboKabaddiEnv(n_defenders=3, outcome_rules='nonsense')
    check('invalid outcome rules rejected', False)
except AssertionError:
    check('invalid outcome rules rejected', True)

# 2. stranded past the midline, no contact: the fix ---------------------------
L, S = run('legacy', park_past), run('section_2_3', park_past)
check('stranded, no contact, legacy: Empty at 30 s from reset',
      L['outcome'] == 'empty' and L['steps'] == 600, (L['outcome'], L['steps']))
check('stranded, no contact, section_2_3: Tackle',
      S['outcome'] == 'tackle', S['outcome'])
check('section_2_3 raid timer starts at the crossing: ends 600 steps later',
      S['steps'] == S['cross'] + 600, (S['cross'], S['steps']))
check('legacy reward on the last step is the Empty penalty (-2)',
      abs(L['last_r'] - (R_TIME + R_EMPTY)) < 1e-6, L['last_r'])
check('section_2_3 reward on the last step is the Tackle penalty (-5)',
      abs(S['last_r'] - (R_TIME + R_TACKLE)) < 1e-6, S['last_r'])
check('ended by termination, not truncation', S['term'] and not S['trunc'])

# 3. stranded after contact ---------------------------------------------------
L, S = run('legacy', contact_then_park), run('section_2_3', contact_then_park)
check('stranded after contact: Tackle under both',
      L['outcome'] == 'tackle' and S['outcome'] == 'tackle',
      (L['outcome'], S['outcome']))

# 4. back home without contact ------------------------------------------------
L, S = run('legacy', cross_and_return), run('section_2_3', cross_and_return)
check('crossed and came back without contact: Empty under both',
      L['outcome'] == 'empty' and S['outcome'] == 'empty',
      (L['outcome'], S['outcome']))
check('...section_2_3 ends 30 s after the crossing',
      S['steps'] == S['cross'] + 600, (S['cross'], S['steps']))

# 5. touch and return ---------------------------------------------------------
L, S = run('legacy', touch_and_return), run('section_2_3', touch_and_return)
check('touch and return: Touch under both, on the same step',
      L['outcome'] == S['outcome'] == 'touch' and L['steps'] == S['steps'],
      (L, S))

# 6. never crosses ------------------------------------------------------------
L, S = run('legacy', never_cross), run('section_2_3', never_cross)
check('never crosses, legacy: Empty at 30 s', L['outcome'] == 'empty'
      and L['steps'] == 600, (L['outcome'], L['steps']))
check('never crosses, section_2_3: Empty at the 20 s approach limit',
      S['outcome'] == 'empty' and S['steps'] == 400 and S['term'],
      (S['outcome'], S['steps']))
check('...with the Empty penalty', abs(S['last_r'] - (R_TIME + R_EMPTY)) < 1e-6,
      S['last_r'])


# 7. crossing on the last approach step counts --------------------------------
def late_cross(env, k):
    if k == 400:            # put the raider across just before step 400
        p.resetBasePositionAndOrientation(
            env.raider_id, [0.0, 0.1, E.R_HEIGHT / 2], (0.0, 0.0, 0.0, 1.0))
    return STOP


S = run('section_2_3', late_cross)
check('crossing on step 400 starts the raid instead of ending it Empty',
      S['cross'] == 400 and S['outcome'] == 'tackle' and S['steps'] == 1000,
      (S['cross'], S['outcome'], S['steps']))
check('...the longest raid ends at the episode limit (1000 steps)',
      S['steps'] == S['max_steps'])

# 8. time-left observation unchanged (counts from reset) ----------------------
obs = {}
for rules in ('legacy', 'section_2_3'):
    env = E.RoboKabaddiEnv(n_defenders=3, outcome_rules=rules)
    env.reset(seed=3)
    for _ in range(100):
        o, *_ = env.step(FWD)
    obs[rules] = o
    env.close()
check('observation identical under both rules (incl. time left)',
      np.array_equal(obs['legacy'], obs['section_2_3']))
check('...and the same width', obs['section_2_3'].shape == (11 + 2 * 3,))

# 8b. empty_on_return -----------------------------------------------------------
def run_eor(policy):
    env = E.RoboKabaddiEnv(n_defenders=3, defender_type='static',
                           start_layout='back_row', outcome_rules='section_2_3',
                           empty_on_return=True)
    env.reset(seed=0)
    p.resetBasePositionAndOrientation(
        env.raider_id, [0.0, -0.5, E.R_HEIGHT / 2], (0.0, 0.0, 0.0, 1.0))
    env._prev_pos = np.array([0.0, -0.5], dtype=np.float32)
    k, back = 0, None
    while True:
        k += 1
        _, r, term, trunc, info = env.step(policy(env, k))
        if back is None and env._flags['midline'] and env._get_pos(env.raider_id)[1] < 0:
            back = k
        if term or trunc:
            break
    env.close()
    return k, back, info.get('outcome'), r


env = E.RoboKabaddiEnv(n_defenders=3)
check('empty_on_return defaults to False', env.empty_on_return is False)
env.close()
k, back, o, r = run_eor(cross_and_return)
check('empty_on_return: back across without contact = Empty on that very step',
      o == 'empty' and k == back, (o, k, back))
check('...with the Empty penalty', abs(r - (R_TIME + R_EMPTY)) < 0.05, r)
k, back, o, r = run_eor(touch_and_return)
check('empty_on_return: touch and return is still a Touch', o == 'touch', o)
k, back, o, r = run_eor(park_past)
check('empty_on_return: stranded raider still a Tackle at the timer', o == 'tackle', o)

# 9. reset clears the raid timer ----------------------------------------------
env = E.RoboKabaddiEnv(n_defenders=3, outcome_rules='section_2_3')
env.reset(seed=0)
env._raid_start_step = 5
env.reset(seed=0)
check('reset clears the raid start', env._raid_start_step is None)
env.close()

print(f'\n{passed} passed, {failed} failed')
