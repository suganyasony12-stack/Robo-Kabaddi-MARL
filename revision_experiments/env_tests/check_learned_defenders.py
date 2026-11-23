"""Checks of defender_type="learned", defender_fouls and the team rewards
(CORRECTIONS_TRACKER.md C3, author's rules of 27 Oct) against the recording
mock. The mock has no collisions; the end-to-end replay against the gate with
real PyBullet is ../check_learned_equivalence.py."""
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
W, L = E.COURT_W / 2, E.COURT_L / 2


def check(name, cond, detail=''):
    global passed, failed
    if cond:
        passed += 1
        print(f'  pass  {name}')
    else:
        failed += 1
        print(f'  FAIL  {name}   {detail}')


def learned(n=3, fouls=True, **kw):
    cfg = dict(E.CORRECTED_CONFIG, n_defenders=n, defender_type='learned',
               defender_fouls=fouls)
    cfg.update(kw)
    env = E.RoboKabaddiEnv(**cfg)
    env.reset(seed=0)
    return env


def place(env, bid, x, y):
    p.resetBasePositionAndOrientation(bid, [x, y, E.R_HEIGHT / 2], (0.0, 0.0, 0.0, 1.0))
    if bid == env.raider_id:
        env._prev_pos = np.array([x, y], dtype=np.float32)


print('1. defaults and options')
env = E.RoboKabaddiEnv(n_defenders=3)
check('defender_fouls defaults to False', env.defender_fouls is False)
check('default defender_type is unchanged (reactive)', env.def_type == 'reactive')
env.close()
check('CORRECTED_CONFIG unchanged: gate v6, no fouls',
      E.CORRECTED_CONFIG['defender_type'] == 'gate'
      and E.CORRECTED_CONFIG['gate_style'] == 'keepout'
      and 'defender_fouls' not in E.CORRECTED_CONFIG)

print('2. the robots\' commands')
check('9 actions: stop and the 8 directions, each once',
      len(E.DEF_ACTIONS) == 9 and len(set(E.DEF_ACTIONS)) == 9
      and set(E.DEF_ACTIONS) == {(sx, sy) for sx in (-1, 0, 1) for sy in (-1, 0, 1)})
env = learned()
ok = True
for k, (sx, sy) in enumerate(E.DEF_ACTIONS):
    env.set_defender_actions([k] * 3)
    c = env._defender_command(0, env.defender_ids[0], env._get_pos(env.raider_id))
    ok &= np.allclose(c, env._phys_velocity(sx, sy))
    ok &= np.isclose(np.hypot(*c), 0.0 if k == 0 else env.DEF_MAX_LIN_VEL)
check('each action = the physical defenders\' command at the 100% level, 0.30 m/s', ok)
env.close()
env = learned(diagonal_speed_ratio=0.71)
env.set_defender_actions([2, 3, 0])
c = env._defender_commands()
check('diagonal_speed_ratio applies to diagonal actions only',
      np.isclose(np.hypot(*c[0]), 0.71 * 0.30) and np.isclose(np.hypot(*c[1]), 0.30)
      and np.allclose(c[2], 0.0), c)
env.close()
env = learned()
bad = 0
for a in ([0, 0], [0, 0, 9], [-1, 0, 0]):
    try:
        env.set_defender_actions(a)
    except AssertionError:
        bad += 1
check('wrong number of actions or index out of range is refused', bad == 3)
env.set_defender_actions([1, 5, 7])
seen = []
orig = env._defender_commands
env._defender_commands = lambda: (seen.append(orig()), seen[-1])[1]
env.step(STOP)
check('held for the action_repeat control steps of one env.step (4)',
      len(seen) == 4 and all(s == seen[0] for s in seen), len(seen))
env.set_defender_actions([0, 0, 0])
seen.clear()
env.step(STOP)
check('a new action takes effect at the next env.step', all(np.allclose(s, 0) for s in seen))
env._defender_commands = orig
env.close()

env = learned(fouls=False)
place(env, env.defender_ids[0], 0.0, 0.05)
place(env, env.defender_ids[1], W - 0.05, 1.0)
env.set_defender_actions([0, 0, 0])
c = env._defender_commands()
check('no boundary escape: stop near the midline and at a wall stays stop',
      np.allclose(c[0], 0) and np.allclose(c[1], 0), c)
env.close()

print('3. fouls (author: defender over the midline, a side line or the end line is out)')


def foul_step(xy_list, fouls=True, raider=(0.0, -0.3)):
    env = learned(fouls=fouls)
    place(env, env.raider_id, *raider)
    for did, xy in zip(env.defender_ids, xy_list):
        place(env, did, *xy)
    env.set_defender_actions([0, 0, 0])
    _, r, term, _, info = env.step(STOP)
    env.close()
    return info.get('outcome'), term, r, info


inside = [(-0.5, 1.6), (0.0, 1.6), (0.5, 1.6)]
o, term, r_in, info = foul_step(inside)
check('all defenders inside: no result, team reward 0',
      o is None and not term and info['team_reward'] == (0.0, 0.0), (o, info))
for label, xy in (('over the midline', (0.0, -0.01)),
                  ('over the left line', (-W - 0.01, 1.0)),
                  ('over the right line', (W + 0.01, 1.0)),
                  ('over the end line', (0.0, L + 0.01))):
    o, term, r, info = foul_step([xy] + inside[1:])
    check(f'defender {label}: foul at once, 1 point to the raider',
          o == 'foul' and term and info['fouls'] == 1
          and info['team_reward'] == (1.0, -1.0), (o, info))
R = E.RoboKabaddiEnv
check('...the raider\'s reward is R_TOUCH for it (plus one step\'s time cost; '
      'the raid ends on the first control step)',
      abs(r - (R.R_TOUCH + R.R_TIME)) < 1e-6, r)
for label, xy in (('on the midline side, just in', (0.0, 0.01)),
                  ('just inside the right line', (W - 0.01, 1.0)),
                  ('just inside the end line', (0.0, L - 0.01))):
    o, term, _, _ = foul_step([xy] + inside[1:])
    check(f'defender {label}: no foul', o is None and not term, o)
o, term, r2, info = foul_step([(0.0, -0.01), (W + 0.01, 1.0), (0.5, 1.6)])
check('two defenders over: 2 points (team 2 / -2, raider 2 x R_TOUCH)',
      o == 'foul' and info['fouls'] == 2 and info['team_reward'] == (2.0, -2.0)
      and abs(r2 - (2 * R.R_TOUCH + R.R_TIME)) < 1e-6, (info, r2))
o, term, _, _ = foul_step([(0.0, -0.01)] + inside[1:], raider=(0.0, -0.3))
check('a foul counts before the raider has crossed (any time in the raid)', o == 'foul')
o, term, _, _ = foul_step([(0.0, -0.01)] + inside[1:], fouls=False)
check('defender_fouls=False: no foul', o is None and not term, o)
env = E.RoboKabaddiEnv(**dict(E.CORRECTED_CONFIG, n_defenders=3))
env.reset(seed=0)
place(env, env.defender_ids[0], 0.0, -0.01)
_, _, term, _, info = env.step(STOP)
check('gate defenders: no foul rule and no team_reward key',
      info.get('outcome') != 'foul' and 'team_reward' not in info, info)
env.close()

print('4. team rewards for the other results')
env = learned()
place(env, env.raider_id, 0.0, 0.5)
env.step(STOP)                                         # raid starts
place(env, env.raider_id, W + 0.01, 0.8)
_, _, term, _, info = env.step(STOP)
env.close()
check('raider over a line after crossing (its own foul, A5 tackle): team -1 / +1',
      info.get('outcome') == 'tackle' and info['team_reward'] == (-1.0, 1.0), info)
env = learned()
place(env, env.raider_id, 0.0, -L - 0.01)
_, _, term, _, info = env.step(STOP)
env.close()
check('Empty raid: both teams -0.3',
      info.get('outcome') == 'empty' and info['team_reward'] == (-0.3, -0.3), info)
env = learned()
place(env, env.raider_id, 0.0, 0.5)
env.step(STOP)
env._flags['contact'] = True
place(env, env.raider_id, 0.0, -0.05)
_, _, term, _, info = env.step(STOP)
env.close()
check('Touch: team +1 / -1', info.get('outcome') == 'touch'
      and info['team_reward'] == (1.0, -1.0), info)
check('TEAM_REWARD mirrors for touch, foul and tackle; Empty costs both',
      all(E.TEAM_REWARD[k][0] == -E.TEAM_REWARD[k][1] for k in ('touch', 'foul', 'tackle'))
      and E.TEAM_REWARD['empty'] == (-0.3, -0.3))

print('5. defender observations and the critic state')
for n in (1, 2, 3, 7):
    env = learned(n=n)
    o = env.defender_obs()
    check(f'{n} defenders: obs {o.shape}, state {env.global_state().shape}',
          o.shape == (n, 15 + 2 * (n - 1)) and env.global_state().shape == (4 + 4 * n + 3,)
          and o.dtype == np.float32)
    env.close()
env = learned()
place(env, env.raider_id, 0.2, 0.4)
for did, xy in zip(env.defender_ids, [(0.0, 1.0), (1.0, 1.2), (-0.3, 1.1)]):
    place(env, did, *xy)
o = env.defender_obs()
row = o[0]
check('own position, raider relative', np.allclose(row[:2], [0.0, 1.0])
      and np.allclose(row[4:6], [0.2, -0.6]), row[:8])
check('teammates relative, nearest first',
      np.allclose(row[8:10], [-0.3, 0.1]) and np.allclose(row[10:12], [1.0, 0.2]), row[8:12])
check('distances to the midline, left, right and end lines',
      np.allclose(row[12:16], [1.0, W, W, L - 1.0]), row[12:16])
check('tail: contact 0, raid not started, time left 1 at reset',
      np.allclose(row[16:], [0.0, 0.0, 1.0]), row[16:])
check('rows are egocentric: defender 2 sees defender 1 at (-1.0, -0.2)',
      np.allclose(o[1][:2], [1.0, 1.2])
      and any(np.allclose(o[1][8 + 2 * k:10 + 2 * k], [-1.0, -0.2]) for k in range(2)))
env.step(STOP)
check('raid started once the raider is past the midline; raid timer runs',
      env.defender_obs()[0][17] == 1.0 and env.defender_obs()[0][18] < 1.0)
env.close()
env = learned()
before = env._obs().shape
env.close()
check('raider observation width unchanged (17 with 3 defenders)', before == (17,), before)

print('6. replay of the gate through the learned interface (mock)')


def run_gate(seed):
    env = E.RoboKabaddiEnv(**dict(E.CORRECTED_CONFIG, n_defenders=3, action_repeat=1))
    env.reset(seed=seed)
    acts, cmds, traj = [], [], []
    orig = env._defender_commands
    env._defender_commands = lambda: (cmds.append(orig()), cmds[-1])[1]
    for t in range(600):
        a = np.array([0.0, 1.0 if not env._flags['contact'] else -1.0, 0.0], np.float32)
        _, _, term, trunc, info = env.step(a)
        acts.append(a)
        traj.append([tuple(env._get_pos(b)[:2]) for b in [env.raider_id] + env.defender_ids])
        if term or trunc:
            break
    env.close()
    return acts, cmds, traj, info.get('outcome')


def run_learned(seed, acts, cmds):
    env = learned(fouls=False, action_repeat=1)
    env.reset(seed=seed)
    lut = {tuple(np.round(env._phys_velocity(*d), 9)): k for k, d in enumerate(E.DEF_ACTIONS)}
    traj = []
    for a, c in zip(acts, cmds):
        env.set_defender_actions([lut[tuple(np.round(ci, 9))] for ci in c])
        _, _, term, trunc, info = env.step(a)
        traj.append([tuple(env._get_pos(b)[:2]) for b in [env.raider_id] + env.defender_ids])
        if term or trunc:
            break
    env.close()
    return traj, info.get('outcome')


same = True
for seed in range(4):
    acts, cmds, tg, og = run_gate(seed)
    tl, ol = run_learned(seed, acts, cmds)
    same &= (og == ol) and len(tg) == len(tl) and np.allclose(tg, tl)
check('every gate command is one of the 9 actions, and replaying them gives the '
      'same raids (4 seeds)', same)

print('7. raider_edge_escape (the robots\' 0.30 m boundary escape for the raider)')
env = E.RoboKabaddiEnv(n_defenders=3)
check('raider_edge_escape defaults to False', env.raider_edge_escape is False)
env.close()
check('RAIDER_EDGE is the robots\' BOUNDARY_MARGIN, 0.30 m', E.RAIDER_EDGE == 0.30)


def raider_cmd(x, y, action=(0.0, 1.0, 0.0), escape=True, ratio=1.0):
    env = learned(raider_edge_escape=escape, diagonal_speed_ratio=ratio)
    place(env, env.raider_id, x, y)
    seen = []
    orig = env._drive
    env._drive = lambda bid, vx, vy, om: (seen.append((vx, vy)) if bid == env.raider_id
                                          else None, orig(bid, vx, vy, om))[1]
    env.set_defender_actions([0, 0, 0])
    env._step_once(np.array(action, dtype=np.float32))
    env.close()
    return seen[0]


V = E.RoboKabaddiEnv.MAX_LIN_VEL
for label, xy, want in (('left line', (-W + 0.29, 0.8), (V, 0)),
                        ('right line', (W - 0.29, 0.8), (-V, 0)),
                        ('far end line', (0.0, L - 0.29), (0, -V)),
                        ('own end line', (0.0, -L + 0.29), (0, V))):
    c = raider_cmd(*xy, action=(1.0 if 'left' not in label else -1.0, 1.0, 0.0))
    check(f'within 0.30 m of the {label}: command replaced by the escape at 0.30 m/s',
          np.allclose(c, want), c)
c = raider_cmd(-W + 0.2, L - 0.2, ratio=0.71)
check('in a corner: diagonal escape, with diagonal_speed_ratio',
      np.allclose(c, (0.71 * V / np.sqrt(2), -0.71 * V / np.sqrt(2))), c)
c = raider_cmd(-W + 0.31, 0.8, action=(-1.0, 0.0, 0.0))
check('0.31 m from the line: the raider\'s own command', np.allclose(c, (-V, 0)), c)
c = raider_cmd(-W + 0.2, 0.8, action=(-1.0, 0.0, 0.0), escape=False)
check('escape off: the raider\'s own command at the line', np.allclose(c, (-V, 0)), c)
env = learned(raider_edge_escape=True)
place(env, env.raider_id, 0.0, 0.5)
env.step(STOP)
place(env, env.raider_id, W + 0.01, 0.8)
_, _, term, _, info = env.step(STOP)
env.close()
check('not a wall: a raider pushed over a line is still tackled (A5)',
      info.get('outcome') == 'tackle' and term, info)

print('8. held_needs_touch (author, 28 Oct: held only while a defender touches)')
env = E.RoboKabaddiEnv(n_defenders=3)
check('held_needs_touch defaults to False', env.held_needs_touch is False)
env.close()


def still_raid(needs_touch, touching=lambda t: False):
    """Raider still at (0, 0.8) in the defenders' half; returns (outcome, end time,
    time since the raid started)."""
    env = learned(held_needs_touch=needs_touch)
    env._defender_touching = lambda: touching(env._step * env.dt)
    place(env, env.raider_id, 0.0, 0.8)
    info, term = {}, False
    while not term:
        place(env, env.raider_id, 0.0, 0.8)
        env.set_defender_actions([0, 0, 0])
        _, _, term, trunc, info = env.step(STOP)
        term = term or trunc
    t, t0 = env._step * env.dt, env._raid_start_step * env.dt
    env.close()
    return info.get('outcome'), t, t - t0


out, t, raid_t = still_raid(False)
check('flag off: a still raider is held after 4 s, no touch needed (as before)',
      out == 'tackle' and 4.0 < raid_t < 4.5, (out, raid_t))
out, t, raid_t = still_raid(True)
check('flag on, no defender touching: no hold; tackled at the 30 s timeout',
      out == 'tackle' and raid_t >= 30.0 - 1e-9, (out, raid_t))
out, t, raid_t = still_raid(True, touching=lambda t: True)
check('flag on, a defender touching all along: held after 4 s',
      out == 'tackle' and 4.0 < raid_t < 4.5, (out, raid_t))
out, t, raid_t = still_raid(True, touching=lambda t: not (2.0 <= t < 2.1))
check('flag on, the touch lost at 2 s: the 4 s clock restarts',
      out == 'tackle' and 6.0 < raid_t < 6.6, (out, raid_t))

print(f'\n{passed} passed, {failed} failed')
