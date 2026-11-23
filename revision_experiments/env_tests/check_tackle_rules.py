"""Control-flow checks of outcome_rules='physical' (the static-raider Tackle,
run_experiment.py check 3c) against the recording mock. Scripted raider
paths decide each outcome; contact is set by hand where needed."""
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
V = E.RoboKabaddiEnv.MAX_LIN_VEL
STOP = np.zeros(3, dtype=np.float32)


def act(vy):
    """Action for a forward speed of vy m/s."""
    return np.array([0.0, vy / V, 0.0], dtype=np.float32)


def check(name, cond, detail=''):
    global passed, failed
    if cond:
        passed += 1
        print(f'  pass  {name}')
    else:
        failed += 1
        print(f'  FAIL  {name}   {detail}')


def run(rules, policy, start_y=-0.5):
    env = E.RoboKabaddiEnv(n_defenders=3, defender_type='static',
                           start_layout='back_row', outcome_rules=rules)
    env.reset(seed=0)
    p.resetBasePositionAndOrientation(
        env.raider_id, [0.0, start_y, E.R_HEIGHT / 2], (0.0, 0.0, 0.0, 1.0))
    env._prev_pos = np.array([0.0, start_y], dtype=np.float32)
    state = {}
    cross = None
    k = 0
    while True:
        k += 1
        _, r, term, trunc, info = env.step(policy(env, k, state))
        if cross is None and env._flags['midline']:
            cross = k
        if term or trunc:
            break
    out = dict(steps=k, outcome=info.get('outcome'), last_r=r, cross=cross,
               since=env._static_since, state=state)
    env.close()
    return out


def y_of(env):
    return float(env._get_pos(env.raider_id)[1])


def park_past(env, k, s):
    if y_of(env) < 0.3:
        return act(V)
    s.setdefault('park', k)
    return STOP


def creep(speed):
    def pol(env, k, s):
        return act(V) if y_of(env) < 0.3 and 'creep' not in s else \
            (s.setdefault('creep', k), act(speed))[1]
    return pol


def contact_then_park(env, k, s):
    if y_of(env) >= 0.3:
        env._flags['contact'] = True     # mock has no contacts
        s.setdefault('park', k)
        return STOP
    return act(V)


def cross_home_park(env, k, s):
    if not s.get('went') and y_of(env) < 0.3:
        return act(V)
    s['went'] = True
    return act(-V) if y_of(env) > -0.3 else STOP


def out_and_back(env, k, s):
    """Cross to 0.1 m, park 5 s, go back to -0.1 m, return to 0.1 m, park."""
    y = y_of(env)
    ph = s.setdefault('phase', 1)
    if ph == 1:
        if y < 0.1:
            return act(V)
        s['park1'] = k
        s['phase'] = 2
    if ph == 2:
        if k - s['park1'] < 100:
            return STOP
        s['phase'] = 3
    if s['phase'] == 3:
        if y > -0.1:
            return act(-V)
        s['phase'] = 4
    if s['phase'] == 4:
        if y < 0.1:
            return act(V)
        s['phase'] = 5
        s['park2'] = k
    return STOP


T = E.RoboKabaddiEnv

# 1. construction -------------------------------------------------------------
env = E.RoboKabaddiEnv(n_defenders=3, outcome_rules='physical')
check("'physical' accepted; same limits as section_2_3 (1000 steps)",
      env.max_steps == 1000 and env._raid_steps == 600)
check('static window is 7 s = 140 control steps, 5 cm',
      env._static_steps == 140 and E.STATIC_DIST == 0.05, env._static_steps)
env.close()

env = E.RoboKabaddiEnv(n_defenders=3, outcome_rules='physical', static_time=4.0)
check('static_time=4.0 gives an 80-step window (default 7 s stays 140)',
      env._static_steps == 80 and env.static_time == 4.0)
env.close()
try:
    E.RoboKabaddiEnv(n_defenders=3, static_time=0.0)
    check('static_time 0 rejected', False)
except AssertionError:
    check('static_time 0 rejected', True)

# 2. parked past the midline ---------------------------------------------------
P = run('physical', park_past)
check('parked past the midline: Tackle', P['outcome'] == 'tackle', P['outcome'])
check('...7 s after the window last started (strictly more than 140 steps)',
      P['steps'] - P['since'] == 141, (P['since'], P['steps']))
check('...well before the 30 s raid timer',
      P['steps'] < P['cross'] + 600, (P['cross'], P['steps']))
check('...with the Tackle penalty',
      abs(P['last_r'] - (T.R_TIME + T.R_TACKLE)) < 1e-6, P['last_r'])
S = run('section_2_3', park_past)
check('section_2_3 unchanged: same path ends at the 30 s timer',
      S['outcome'] == 'tackle' and S['steps'] == S['cross'] + 600,
      (S['cross'], S['steps']))
L = run('legacy', park_past)
check('legacy unchanged: Empty at 30 s', L['outcome'] == 'empty'
      and L['steps'] == 600, (L['outcome'], L['steps']))

env = E.RoboKabaddiEnv(n_defenders=3, defender_type='static', start_layout='back_row',
                       outcome_rules='physical', static_time=4.0)
env.reset(seed=0)
p.resetBasePositionAndOrientation(env.raider_id, [0.0, -0.5, E.R_HEIGHT / 2],
                                  (0.0, 0.0, 0.0, 1.0))
env._prev_pos = np.array([0.0, -0.5], dtype=np.float32)
st, k = {}, 0
while True:
    k += 1
    _, _, term, trunc, info = env.step(park_past(env, k, st))
    if term or trunc:
        break
check('static_time 4 s: parked raider tackled 81 steps after the window starts',
      info.get('outcome') == 'tackle' and k - env._static_since == 81,
      (info.get('outcome'), k, env._static_since))
env.close()

# 3. the 5 cm threshold --------------------------------------------------------
F = run('physical', creep(0.010))   # 7 cm per 7 s: window keeps restarting
check('creeping 1 cm/s (> 5 cm per 7 s): no static Tackle, ends at the timer',
      F['outcome'] == 'tackle' and F['steps'] == F['cross'] + 600,
      (F['cross'], F['steps']))
Sl = run('physical', creep(0.006))  # 4.2 cm per 7 s: inside the window
check('creeping 0.6 cm/s (< 5 cm per 7 s): static Tackle',
      Sl['outcome'] == 'tackle' and Sl['steps'] - Sl['since'] == 141,
      (Sl['since'], Sl['steps']))

# 4. after contact -------------------------------------------------------------
C = run('physical', contact_then_park)
check('held after contact: static Tackle 7 s after stopping',
      C['outcome'] == 'tackle' and C['steps'] - C['since'] == 141
      and C['steps'] < C['cross'] + 600, (C['since'], C['steps']))

# 5. only in the defenders' half -----------------------------------------------
H = run('physical', cross_home_park)
check('parked in its own half: no static Tackle; Empty at the timer',
      H['outcome'] == 'empty' and H['steps'] == H['cross'] + 600,
      (H['outcome'], H['steps']))
B = run('physical', out_and_back)
check('window restarts after the raider leaves the defenders\' half',
      B['outcome'] == 'tackle' and B['since'] > B['state']['park1'] + 100
      and B['steps'] - B['since'] == 141,
      (B['state'], B['since'], B['steps']))

# 6. reset ---------------------------------------------------------------------
env = E.RoboKabaddiEnv(n_defenders=3, outcome_rules='physical')
env.reset(seed=0)
env._static_ref, env._static_since = np.zeros(2), 5
env.reset(seed=0)
check('reset clears the static window',
      env._static_ref is None and env._static_since is None)
env.close()

print(f'\n{passed} passed, {failed} failed')
