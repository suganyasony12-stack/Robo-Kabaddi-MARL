"""Control-flow checks of defender_type='physical' against the recording mock.
Each rule of the physical defenders' controller (smorphi_controller.py,
send_defender_action; run_experiment.py, boundary override) is checked from
placed positions. The mock has no contacts, so contact is set by hand."""
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
V = 0.30


def check(name, cond, detail=''):
    global passed, failed
    if cond:
        passed += 1
        print(f'  pass  {name}')
    else:
        failed += 1
        print(f'  FAIL  {name}   {detail}')


def place(env, bid, x, y):
    p.resetBasePositionAndOrientation(
        bid, [x, y, E.R_HEIGHT / 2], (0.0, 0.0, 0.0, 1.0))


def cmd(env, raider, defender, contact=False, i=0):
    """Command for defender i on a pursuit tick, from placed positions."""
    place(env, env.raider_id, *raider)
    place(env, env.defender_ids[i], *defender)
    env._flags['contact'] = contact
    env._step = 1
    return env._defender_command(i, env.defender_ids[i],
                                 env._get_pos(env.raider_id))


def close(a, b, tol=1e-9):
    return all(abs(u - v) < tol for u, v in zip(a, b))


# 1. construction ------------------------------------------------------------
env = E.RoboKabaddiEnv(n_defenders=3, defender_type='physical')
env.reset(seed=0)
check("'physical' is accepted", env.def_type == 'physical')
check('defender speed equals the raider maximum (0.30 m/s)',
      env.DEF_MAX_LIN_VEL == env.MAX_LIN_VEL == V, env.DEF_MAX_LIN_VEL)
check('pursuit every 4 control steps (5 Hz at dt = 0.05 s)',
      env._def_period == 4, env._def_period)
check('diagonal speed ratio defaults to 1.0', env.diagonal_speed_ratio == 1.0)

# 2. stationary while the raider is in its own half --------------------------
check('still while raider is in its own half',
      close(cmd(env, (0.0, -0.5), (0.0, 1.0)), (0, 0)))
check('still while raider is on the midline',
      close(cmd(env, (0.0, 0.0), (0.0, 1.0)), (0, 0)))
env._flags['midline'] = True          # the raider has crossed before
check('still again once the raider goes back (current position, not a flag)',
      close(cmd(env, (0.0, -0.2), (0.0, 1.0)), (0, 0)))

# 3. pursuit direction and speed ---------------------------------------------
c = cmd(env, (0.0, 0.5), (0.0, 1.0))
check('straight pursuit at 0.30 m/s toward the raider', close(c, (0, -V)), c)
c = cmd(env, (0.0, 0.5), (0.5, 1.0))
check('diagonal pursuit, equal speed by default',
      close(c, (-V / np.sqrt(2), -V / np.sqrt(2))) and abs(np.hypot(*c) - V) < 1e-9, c)
c = cmd(env, (1.0, 0.75), (0.0, 1.0))
check('small component (< 0.3 of the direction) is dropped: pure +x',
      close(c, (V, 0)), c)
check('stop within 0.05 m of the raider',
      close(cmd(env, (0.0, 0.70), (0.03, 0.70)), (0, 0)))
for name, (dx, dy) in {'+x': (1, 0), '-x': (-1, 0), '+y': (0, 1), '-y': (0, -1),
                       '+x+y': (1, 1), '+x-y': (1, -1),
                       '-x+y': (-1, 1), '-x-y': (-1, -1)}.items():
    c = cmd(env, (0.0 + 0.4 * dx, 1.0 + 0.4 * dy), (0.0, 1.0))
    check(f'direction {name:<5} is one of the eight',
          close(np.sign(c), (dx, dy)), c)
env.close()

env = E.RoboKabaddiEnv(n_defenders=3, defender_type='physical',
                       diagonal_speed_ratio=0.71)
env.reset(seed=0)
c = cmd(env, (0.0, 0.5), (0.5, 1.0))
check('diagonal_speed_ratio=0.71 scales a diagonal to 0.213 m/s',
      abs(np.hypot(*c) - 0.71 * V) < 1e-9, np.hypot(*c))
check('diagonal_speed_ratio leaves straight moves at 0.30 m/s',
      close(cmd(env, (0.0, 0.5), (0.0, 1.0)), (0, -V)))
env.close()

# 4. soft boundary before contact, released after -----------------------------
env = E.RoboKabaddiEnv(n_defenders=3, defender_type='physical')
env.reset(seed=0)
check('before contact: stop inside the soft boundary (0.30 m)',
      close(cmd(env, (0.0, 0.10), (0.0, 0.25)), (0, 0)))
check('before contact: no step toward the midline within 0.10 m of soft',
      close(cmd(env, (0.0, 0.10), (0.0, 0.35)), (0, 0)))
check('before contact: sideways move near soft still allowed',
      close(cmd(env, (-0.8, 0.40), (0.0, 0.35)), (-V, 0)))
check('before contact: toward the midline allowed beyond soft + 0.10',
      close(cmd(env, (0.0, 0.10), (0.0, 0.45)), (0, -V)))
check('after contact: soft boundary released',
      close(cmd(env, (0.0, 0.05), (0.0, 0.25), contact=True), (0, -V)))
check('after contact: still stop once the raider is home',
      close(cmd(env, (0.0, -0.05), (0.0, 0.5), contact=True), (0, 0)))

# 5. boundary override, before and after contact -----------------------------
for contact in (False, True):
    tag = 'after contact' if contact else 'before contact'
    check(f'{tag}: inside 0.15 m of the midline, driven straight back',
          close(cmd(env, (0.0, 0.05), (0.0, 0.10), contact), (0, V)))
    check(f'{tag}: in the raider half, driven straight back',
          close(cmd(env, (0.0, -0.3), (0.0, -0.05), contact), (0, V)))
check('left wall (0.30 m): driven +x',
      close(cmd(env, (0.0, 0.5), (-1.15, 1.0)), (V, 0)))
check('right wall (0.30 m): driven -x',
      close(cmd(env, (0.0, 0.5), (1.15, 1.0)), (-V, 0)))
check('rear wall (0.30 m): driven -y',
      close(cmd(env, (0.0, 0.5), (0.0, 1.80)), (0, -V)))
c = cmd(env, (0.0, 0.5), (-1.15, 1.80))
check('rear-left corner: diagonal +x -y',
      close(np.sign(c), (1, -1)), c)
check('override applies even while the raider is home',
      close(cmd(env, (0.0, -1.0), (1.15, 1.0)), (-V, 0)))

# 6. 5 Hz hold, with the override pre-empting it ------------------------------
env.reset(seed=4)
place(env, env.raider_id, 0.0, 0.5)
for k, did in enumerate(env.defender_ids):
    place(env, did, -0.6 + 0.6 * k, 1.2)
seen = []
for step in range(9):
    if step == 1:                       # raider jumps to the far right
        place(env, env.raider_id, 1.0, 1.2)
    p.LOG.clear()
    env.step(np.zeros(3, dtype=np.float32))
    seen.append(env._def_cmd[1])
check('command held for steps 1-4 although the raider moved at step 2',
      all(close(seen[j], seen[0]) for j in range(4)), seen[:4])
check('new command at step 5 (the next 5 Hz tick)',
      not close(seen[4], seen[0]) and all(close(seen[j], seen[4]) for j in range(4, 8)),
      seen[4:8])
# each sub-step sets the command first, then the upright hold re-sets velocity
sent = [c[1][1][:2] for c in p.LOG if c[0] == 'resetBaseVelocity'
        and c[1][0] == env.defender_ids[1]][::2]
check('held command is what PyBullet receives at every sub-step',
      len(sent) == 12 and all(close(s, env._def_cmd[1]) for s in sent), sent)

env.reset(seed=4)
place(env, env.raider_id, 0.0, 0.5)
place(env, env.defender_ids[0], 0.0, 1.0)
env.step(np.zeros(3, dtype=np.float32))            # tick: pursue -y
place(env, env.defender_ids[0], 0.0, 0.10)         # now inside the guard
env.step(np.zeros(3, dtype=np.float32))            # not a tick
check('override takes effect between ticks', close(env._def_cmd[0], (0, V)),
      env._def_cmd[0])
env.reset(seed=4)
check('reset clears held commands',
      all(close(c, (0, 0)) for c in env._def_cmd), env._def_cmd)
env.close()

# 7. whole raids: defenders never enter the raider half, never exceed 0.30 ----
min_y, max_v = np.inf, 0.0
for seed in range(6):
    env = E.RoboKabaddiEnv(n_defenders=3, defender_type='physical')
    env.reset(seed=seed)
    for t in range(600):
        if t == 150:
            env._flags['contact'] = True   # mock has no contacts; force one
        a = np.array([0.0, 1.0 if t < 150 else -1.0, 0.0], dtype=np.float32)
        _, _, term, trunc, _ = env.step(a)
        for did in env.defender_ids:
            min_y = min(min_y, env._get_pos(did)[1])
            max_v = max(max_v, float(np.hypot(*env._get_vel(did)[:2])))
        if term or trunc:
            break
    env.close()
check('defenders never enter the raider half (6 driven raids)',
      min_y > 0.0, f'min y {min_y:.3f}')
check('defenders held near the 0.15 m guard (min y > 0.13 m)',
      min_y > 0.13, f'min y {min_y:.3f}')
check('defender speed never exceeds 0.30 m/s (float32 read-back)',
      max_v <= V + 1e-6, max_v)

# 8. other modes and shapes --------------------------------------------------
env = E.RoboKabaddiEnv(n_defenders=3, defender_type='physical', locomotion='legacy')
env.reset(seed=0)
env.step(np.zeros(3, dtype=np.float32))
check('physical defenders also run under legacy locomotion', True)
env.close()
for nd in (1, 3, 7):
    env = E.RoboKabaddiEnv(n_defenders=nd, defender_type='physical')
    o, _ = env.reset(seed=0)
    check(f'observation width unchanged for {nd} defender(s)',
          o.shape == (11 + 2 * nd,), o.shape)
    env.close()
try:
    E.RoboKabaddiEnv(n_defenders=1, diagonal_speed_ratio=0.0)
    check('a non-positive diagonal ratio is rejected', False)
except AssertionError:
    check('a non-positive diagonal ratio is rejected', True)

# 9. start layout -------------------------------------------------------------
spec_b = importlib.util.spec_from_file_location(
    'env_backup', str(Path(__file__).resolve().parents[2] / 'envs'
                      / 'kabaddi_env_legacy_backup.py'))
B = importlib.util.module_from_spec(spec_b)
spec_b.loader.exec_module(B)


def def_starts(env, seed):
    env.reset(seed=seed)
    return [tuple(env._get_pos(d)[:2]) for d in env.defender_ids]


env = E.RoboKabaddiEnv(n_defenders=3)
check('start layout defaults to legacy', env.start_layout == 'legacy')
env.close()
try:
    E.RoboKabaddiEnv(n_defenders=3, start_layout='nonsense')
    check('an invalid start layout is rejected', False)
except AssertionError:
    check('an invalid start layout is rejected', True)

same = True
for nd in range(1, 8):
    new = E.RoboKabaddiEnv(n_defenders=nd, locomotion='legacy')
    old = B.RoboKabaddiEnv(n_defenders=nd)
    for s in range(10):
        same &= def_starts(new, s) == def_starts(old, s)
    new.close(); old.close()
check('legacy starts identical to the submitted code (1-7 defenders, 10 seeds)',
      same)

wall = E.COURT_L / 2
y_lo, y_hi = wall - E.BACK_ROW_FROM_WALL[1], wall - E.BACK_ROW_FROM_WALL[0]
for nd in range(1, 8):
    env = E.RoboKabaddiEnv(n_defenders=nd, defender_type='physical',
                           start_layout='back_row')
    depth_ok = sep_ok = x_ok = True
    min_sep = np.inf
    for s in range(50):
        pts = np.array(def_starts(env, s))
        depth_ok &= bool(np.all((pts[:, 1] >= y_lo - 1e-6)
                                & (pts[:, 1] <= y_hi + 1e-6)))
        x_ok &= bool(np.all(np.diff(pts[:, 0]) > 0))      # stays in its slot
        if nd > 1:
            d = np.linalg.norm(pts[:, None] - pts[None], axis=-1)
            min_sep = min(min_sep, d[np.triu_indices(nd, 1)].min())
    sep_ok = nd == 1 or min_sep > 2 * E.R_RADIUS
    check(f'back_row, {nd} defender(s): 0.13-0.31 m from the back wall, '
          f'in order, no overlap', depth_ok and x_ok and sep_ok,
          (depth_ok, x_ok, round(float(min_sep), 3)))
    o, _ = env.reset(seed=0)
    if nd in (1, 3, 7):
        check(f'back_row: observation width unchanged for {nd} defender(s)',
              o.shape == (11 + 2 * nd,), o.shape)
    env.close()

print(f'\n{passed} passed, {failed} failed')
