"""Control-flow checks of defender_type='gate', gate_style='v5' against the recording mock:
from the moment the raider crosses, every defender steps out of the raider's
reach, passes it in an outer lane, then closes diagonally into a wall behind
it. Also the rule limits and the command layer shared with 'physical'."""
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


def decide(env, raider, defs, contact=False):
    """All defenders' commands on one decision step, from placed positions."""
    place(env, env.raider_id, *raider)
    for did, xy in zip(env.defender_ids, defs):
        place(env, did, *xy)
    env._flags['contact'] = contact
    env._step = 1
    r = env._get_pos(env.raider_id)
    return [env._defender_command(i, d, r) for i, d in enumerate(env.defender_ids)]


def sgn(c):
    return tuple(int(np.sign(round(v, 9))) for v in c)


ROW = [(-0.6, 1.60), (0.0, 1.60), (0.6, 1.60)]   # back margin starts at 1.70

env = E.RoboKabaddiEnv(n_defenders=3, defender_type='gate', gate_style='v5')
env.reset(seed=0)
check("'gate' accepted", env.def_type == 'gate')

# 1. raider at home ---------------------------------------------------------------
c = decide(env, (0.2, -0.3), ROW)
check('raider in its own half: everyone stands still, gate not started',
      all(v == (0, 0) for v in c) and env._gate_closed is False, c)

# 2. raider crosses: slots, step out, pass -----------------------------------------
c = decide(env, (0.2, 0.30), ROW)
check('gate starts as soon as the raider crosses the midline', env._gate_closed)
check('slots by left-to-right order: -0.19, 0, +0.19 m from the lane',
      np.allclose(env._gate_role, [-0.19, 0.0, 0.19]), env._gate_role)
check('left defender (0.8 m clear) passes straight toward the midline',
      sgn(c[0]) == (0, -1), c[0])
check('middle defender (0.2 m from the lane) steps sideways out first (-x)',
      sgn(c[1]) == (-1, 0), c[1])
check('right defender (0.4 m clear) steps out to 0.50 m first (+x)',
      sgn(c[2]) == (1, 0), c[2])

# 3. closing diagonally once level with the raider ----------------------------------
c = decide(env, (0.2, 0.90), [(-0.35, 0.85), (0.75, 1.30), (0.75, 0.85)])
check('level with the raider: left defender closes diagonally (+x, -y)',
      sgn(c[0]) == (1, -1), c[0])
check('level with the raider: right defender closes diagonally (-x, -y)',
      sgn(c[2]) == (-1, -1), c[2])
check('not yet level: the other keeps passing in its lane (-y)',
      sgn(c[1]) == (0, -1), c[1])
t = env._gate_target(0.19, (0.75, 0.85), (0.2, 0.9))
check('slot is 0.25 m behind the raider, 0.19 m beside the lane',
      abs(t[0] - 0.39) < 1e-6 and abs(t[1] - 0.65) < 1e-6, t)

# 3a. forced diagonal and early start ------------------------------------------------
c = decide(env, (0.0, 0.90), [(-0.6, 1.60), (0.3, 1.60), (1.0, 0.90)])
check('level and 1 m out: a full diagonal (-x, -y), not a sideways slide',
      sgn(c[2]) == (-1, -1), c[2])
c = decide(env, (-0.2, 0.60), [(-1.0, 1.60), (-0.6, 1.60), (0.8, 1.00)])
check('1 m out and 0.4 m above the raider: no early start, keeps passing (-x, -y moving in)',
      not env._gate_closing(0.19, (0.8, 1.0), (-0.2, 0.6)) and sgn(c[2])[1] == -1, c[2])
check('the diagonal starts exactly when level with the raider',
      env._gate_closing(0.19, (0.8, 0.6), (-0.2, 0.6))
      and not env._gate_closing(0.19, (0.8, 0.61), (-0.2, 0.6)))
env.close()
env1 = E.RoboKabaddiEnv(n_defenders=1, defender_type='gate', gate_style='v5')
env1.reset(seed=0)
place(env1, env1.raider_id, 0.0, 0.90)
place(env1, env1.defender_ids[0], 0.50, 1.00)
env1._prev_pos = np.array([0.0, 0.9], dtype=np.float32)
dmin, dall, diag = np.inf, np.inf, 0
for k in range(60):
    env1.step(np.zeros(3, dtype=np.float32))
    q, r = env1._get_pos(env1.defender_ids[0]), env1._get_pos(env1.raider_id)
    d = float(np.hypot(q[0] - r[0], q[1] - r[1]))
    dall = min(dall, d)
    if env1._def_cmd[0][0] != 0 and env1._def_cmd[0][1] != 0:
        diag += 1
        dmin = min(dmin, d)
check('from 0.5 m out, 0.1 m above: passes, then the diagonal stays > 0.24 m away',
      dmin > 0.24 and diag > 0, (dmin, diag))
check('...and it settles 0.20-0.30 m behind the raider (face-on contact is 0.17 m)',
      0.17 < dall, dall)
env1.close()
env = E.RoboKabaddiEnv(n_defenders=3, defender_type='gate', gate_style='v5')
env.reset(seed=0)

# 3b. stay within 1 m; never stuck at a wall -----------------------------------------
c = decide(env, (-0.5, 0.30), [(-0.9, 1.60), (-0.4, 1.60), (0.95, 1.60)])
t = env._gate_target(0.19, (0.95, 1.60), (-0.5, 0.30))
check('a defender 1.45 m to the side passes 1.00 m out (moves in)',
      abs(t[0] - 0.50) < 1e-6, t)
check('...diagonally inward and toward the midline (-x, -y)',
      sgn(c[2]) == (-1, -1), c[2])
c = decide(env, (0.90, 0.60), [(-0.5, 1.60), (0.3, 1.60), (0.95, 1.60)])
check('raider by the right wall: the right defender passes on the left instead',
      env._gate_target(0.19, (0.95, 1.60), (0.90, 0.60))[0] < 0.90 and c[2][0] < 0,
      (env._gate_target(0.19, (0.95, 1.60), (0.90, 0.60)), c[2]))

# 4. rule limits before contact -------------------------------------------------------
t = env._gate_target(0.0, (0.6, 0.30), (0.2, 0.40))
check('shallow raider: slot held at the 0.40 m limit', abs(t[1] - 0.40) < 1e-6, t)
bad = []
rng = np.random.default_rng(0)
for _ in range(300):
    defs = [(rng.uniform(-1.0, 1.0), rng.uniform(0.32, 1.65)) for _ in range(3)]
    r = (rng.uniform(-1.0, 1.0), rng.uniform(0.05, 1.6))
    c = decide(env, r, defs)
    for (x, y), v in zip(defs, c):
        if v[1] < 0 and 0.15 <= y < 0.40:
            bad.append((r, (x, y), v))
check('random layouts before contact: no step toward the midline inside 0.40 m',
      not bad, bad[:2])
c = decide(env, (0.6, 0.60), [(0.0, 0.39), (0.8, 1.60), (1.0, 1.60)])
check('at the 0.40 m limit a defender slides sideways instead of stopping',
      sgn(c[0]) == (1, 0), c[0])

# 5. after contact -------------------------------------------------------------------
c = decide(env, (0.2, 0.90), [(-0.2, 0.66), (0.2, 1.30), (0.6, 0.66)],
           contact=True)
check('after contact, defenders behind the raider drive at it (left: +x +y)',
      sgn(c[0]) == (1, 1), c[0])
check('...(right: -x +y), so they keep pressing instead of stopping',
      sgn(c[2]) == (-1, 1), c[2])
check('after contact, a defender not yet level steps out / passes, never at the raider',
      sgn(c[1]) in ((-1, 0), (1, 0), (0, -1)), c[1])

# 6. command layer shared with 'physical' --------------------------------------------
c = decide(env, (0.2, 0.9), [(-1.2, 1.0), (0.0, 1.60), (0.6, 1.60)])
check('wall margin still applies (defender at the left wall driven +x)',
      sgn(c[0]) == (1, 0), c[0])
c = decide(env, (0.2, 0.9), [(-0.3, 0.10), (0.0, 1.60), (0.6, 1.60)])
check('midline guard still applies (driven straight back)', sgn(c[0]) == (0, 1), c[0])
env.close()

# 7. driven raids --------------------------------------------------------------------
min_y, max_v, holds, min_pass = np.inf, 0.0, True, np.inf
for seed in range(6):
    env = E.RoboKabaddiEnv(n_defenders=3, defender_type='gate', gate_style='v5',
                           start_layout='back_row')
    env.reset(seed=seed)
    last = None
    for t in range(500):
        if t == 200:
            env._flags['contact'] = True
        ry = env._get_pos(env.raider_id)[1]
        a = np.array([0.0, (1.0 if ry < 0.9 else 0.0) if t < 200 else -1.0, 0.0],
                     dtype=np.float32)
        esc = any(env._phys_escape(env._get_pos(d)) is not None
                  for d in env.defender_ids)
        _, _, term, trunc, _ = env.step(a)
        if (env._step - 1) % env._def_period and last is not None:
            holds &= esc or env._def_cmd == last
        last = list(env._def_cmd)
        r = env._get_pos(env.raider_id)
        for d in env.defender_ids:
            q = env._get_pos(d)
            min_y = min(min_y, q[1])
            max_v = max(max_v, float(np.hypot(*env._get_vel(d)[:2])))
            if t < 200 and q[1] > r[1] and r[1] > 0.05:
                min_pass = min(min_pass, float(np.hypot(q[0] - r[0], q[1] - r[1])))
        if term or trunc:
            break
    env.close()
check('defenders never enter the raider half (6 driven raids)', min_y > 0.0, min_y)
check('defender speed never exceeds 0.30 m/s', max_v <= V + 1e-6, max_v)
check('commands held between 5 Hz decisions (unless escaping)', holds)
check('while passing, defenders stay out of touching range (> 0.24 m)',
      min_pass > 0.24, min_pass)

# 8. other defender counts ------------------------------------------------------------
ok = True
for nd in (1, 2, 5, 7):
    env = E.RoboKabaddiEnv(n_defenders=nd, defender_type='gate', gate_style='v5',
                           start_layout='back_row')
    env.reset(seed=1)
    for t in range(80):
        env.step(np.array([0.0, 1.0, 0.0], dtype=np.float32))
    roles = sorted(env._gate_role or [])
    want = list((np.arange(nd) - (nd - 1) / 2) * 0.19)
    ok &= len(roles) == nd and np.allclose(roles, want)
    env.close()
check('1, 2, 5 and 7 defenders: one slot each, 0.19 m apart, centred', ok)

# 9. gate_style='keepout' (default) -----------------------------------------------------
env = E.RoboKabaddiEnv(n_defenders=3, defender_type='gate')
env.reset(seed=0)
check("gate_style defaults to 'keepout'", env.gate_style == 'keepout')
env.close()
try:
    E.RoboKabaddiEnv(n_defenders=3, defender_type='gate', gate_style='zigzag')
    check('unknown gate_style rejected', False)
except AssertionError:
    check('unknown gate_style rejected', True)

env = E.RoboKabaddiEnv(n_defenders=3, defender_type='gate')
env.reset(seed=0)
c = decide(env, (-0.2, 0.60), [(-1.0, 1.60), (-0.6, 1.60), (0.6, 1.00)])
check('keep-out: 0.8 m out, 0.4 m above: early diagonal (-x, -y)', sgn(c[2]) == (-1, -1), c[2])
c = decide(env, (0.2, 0.60), [(-0.8, 1.60), (0.25, 1.00), (1.0, 1.60)])
check('keep-out: defender already 0.4 m from the raider backs away, never toward it',
      c[1][1] > 0 and np.hypot(0.05 + c[1][0] * 0.2, 0.4 + c[1][1] * 0.2) > 0.4, c[1])
# raider driving at a defender: the defender backs away
place(env, env.raider_id, 0.0, 0.8)
p.resetBaseVelocity(env.raider_id, linearVelocity=[0.3, 0.0, 0.0])
for did, xy in zip(env.defender_ids, [(-0.8, 1.6), (-0.4, 1.6), (0.45, 0.8)]):
    place(env, did, *xy)
env._flags['contact'] = False
env._step = 1
r = env._get_pos(env.raider_id)
c = [env._defender_command(i, d, r) for i, d in enumerate(env.defender_ids)]
check('keep-out: raider driving at it from 0.45 m: defender moves away (+x)',
      c[2][0] > 0, c[2])
p.resetBaseVelocity(env.raider_id, linearVelocity=[0.0, 0.0, 0.0])
# every chosen move is safe when a safe move exists
viol = []
rng = np.random.default_rng(1)
for _ in range(300):
    defs = [(rng.uniform(-1.0, 1.0), rng.uniform(0.45, 1.65)) for _ in range(3)]
    rp = (rng.uniform(-1.0, 1.0), rng.uniform(0.05, 1.6))
    c = decide(env, rp, defs)
    for (x, y), v in zip(defs, c):
        q = np.array([x, y]) + np.array(v) * 0.2
        now = np.hypot(x - rp[0], y - rp[1])
        nxt = np.hypot(*(q - np.array(rp)))
        if now >= 0.56 and nxt < 0.5 - 1e-9 and env._phys_escape((x, y, 0)) is None:
            viol.append((rp, (x, y), v))
check('keep-out: from >= 0.56 m, no chosen move ends within 0.5 m (300 layouts)',
      not viol, viol[:2])
c = decide(env, (0.2, 0.90), [(-0.2, 0.70), (0.2, 1.40), (0.6, 0.70)], contact=True)
check('keep-out: after contact, defenders level or past press on the raider',
      sgn(c[0]) == (1, 1) and sgn(c[2]) == (-1, 1), c)
env.close()

dmin = np.inf
for seed in range(6):
    env = E.RoboKabaddiEnv(n_defenders=3, defender_type='gate', start_layout='back_row')
    env.reset(seed=seed)
    for t in range(240):
        ry = env._get_pos(env.raider_id)[1]
        env.step(np.array([0.0, 1.0 if ry < 0.9 else 0.0, 0.0], dtype=np.float32))
        r = env._get_pos(env.raider_id)
        if env._gate_closed:
            for d in env.defender_ids:
                q = env._get_pos(d)
                dmin = min(dmin, float(np.hypot(q[0] - r[0], q[1] - r[1])))
    env.close()
check('keep-out: 6 driven raids before contact, closest defender >= 0.45 m',
      dmin >= 0.45, dmin)

# 10. gate_style='v7': never freeze, go back near the midline ----------------------------
env = E.RoboKabaddiEnv(n_defenders=3, defender_type='gate', gate_style='v7')
env.reset(seed=0)
check("'v7' accepted", env.gate_style == 'v7')
OTHERS = [(-1.0, 1.60), (-0.6, 1.60)]


def decide7(raider, d2, contact=False):
    env._gate_back = [False] * 3
    return decide(env, raider, OTHERS + [d2], contact)


c = decide7((0.4, 0.85), (0.9, 0.45))
check('v7: 0.05 m past the 0.40 m line, raider 0.64 m away and deeper: goes back (0, +y)',
      sgn(c[2]) == (0, 1) and env._gate_back[2], (c[2], env._gate_back))
c = decide7((0.4, 1.25), (0.9, 0.85))
check('v7: same layout 0.40 m deeper (outside the band): not going back',
      not env._gate_back[2], env._gate_back)
c = decide7((0.4, 0.75), (0.9, 0.45))
env._step = 1
c2 = env._defender_command(2, env.defender_ids[2], env._get_pos(env.raider_id))
place(env, env.defender_ids[2], 0.9, 0.80)
c2 = env._defender_command(2, env.defender_ids[2], env._get_pos(env.raider_id))
check('v7: level with the raider: still going back (wall-edge fix)',
      env._gate_back[2], env._gate_back)
place(env, env.defender_ids[2], 0.9, 1.28)
c2 = env._defender_command(2, env.defender_ids[2], env._get_pos(env.raider_id))
check('v7: going back ends 0.55 m deeper than the raider',
      not env._gate_back[2], env._gate_back)
env._gate_back = [True] * 3
place(env, env.raider_id, 0.4, 1.40)
place(env, env.defender_ids[2], 0.9, 1.67)
c2 = env._defender_command(2, env.defender_ids[2], env._get_pos(env.raider_id))
check('v7: raider deep: going back ends at the rear margin',
      not env._gate_back[2], env._gate_back)
c = decide7((0.4, 0.85), (0.9, 0.45), contact=True)
check('v7: no going back after contact', not env._gate_back[2], env._gate_back)
# trapped in the corner of the 0.40 m line and the side margin: still moves
c = decide7((0.75, 0.80), (0.98, 0.41))
check('v7: cornered at the line and the side margin: moves, never stops',
      sgn(c[2]) != (0, 0), c[2])
env.gate_style = 'keepout'
c = decide7((0.75, 0.80), (0.98, 0.41))
check('keepout (v6) unchanged: same corner, the defender stops', sgn(c[2]) == (0, 0), c[2])
env.gate_style = 'v7'
# at its slot: keeps moving (keepout stops there)
c = decide7((0.0, 1.20), (0.19, 0.65))
check('v7: at its slot: still moving', sgn(c[2]) != (0, 0), c[2])
# property checks over random layouts, raider in the defenders' half, before contact
viol, stops = [], []
rng = np.random.default_rng(2)
for _ in range(300):
    defs = [(rng.uniform(-1.0, 1.0), rng.uniform(0.45, 1.65)) for _ in range(3)]
    rp = (rng.uniform(-1.0, 1.0), rng.uniform(0.05, 1.6))
    env._gate_back = [False] * 3
    c = decide(env, rp, defs)
    for (x, y), v in zip(defs, c):
        if env._phys_escape((x, y, 0)) is not None:
            continue
        stops += [(rp, (x, y))] if sgn(v) == (0, 0) else []
        q = np.array([x, y]) + np.array(v) * 0.2
        if (np.hypot(x - rp[0], y - rp[1]) >= 0.56
                and np.hypot(*(q - np.array(rp))) < 0.5 - 1e-9):
            viol.append((rp, (x, y), v))
check('v7: no defender ever stops (300 layouts)', not stops, stops[:2])
check('v7: from >= 0.56 m, no chosen move ends within 0.5 m (300 layouts)',
      not viol, viol[:2])
c = decide(env, (0.2, -0.3), ROW)
check('v7: raider at home: defenders stand still (hardware rule kept)',
      all(sgn(v) == (0, 0) for v in c), c)
env.close()
dmin = np.inf
for seed in range(6):
    env = E.RoboKabaddiEnv(n_defenders=3, defender_type='gate', gate_style='v7',
                           start_layout='back_row')
    env.reset(seed=seed)
    for t in range(240):
        ry = env._get_pos(env.raider_id)[1]
        env.step(np.array([0.0, 1.0 if ry < 0.9 else 0.0, 0.0], dtype=np.float32))
        r = env._get_pos(env.raider_id)
        if env._gate_closed:
            for d in env.defender_ids:
                q = env._get_pos(d)
                dmin = min(dmin, float(np.hypot(q[0] - r[0], q[1] - r[1])))
    env.close()
check('v7: 6 driven raids before contact, closest defender >= 0.45 m', dmin >= 0.45, dmin)

# 11. gate_style='v8': hold deep and shadow, close once the raider is 1.0 m in --------------
env = E.RoboKabaddiEnv(n_defenders=3, defender_type='gate', gate_style='v8')
env.reset(seed=0)
check("'v8' accepted", env.gate_style == 'v8')


def decide8(raider, defs, contact=False, keep_in=False):
    env._gate_back = [False] * 3
    if not keep_in:
        env._gate_in = False
    return decide(env, raider, defs, contact)


c = decide8((0.0, 0.50), ROW)
check('v8 holding (raider 0.5 m in): no defender moves toward the midline',
      all(sgn(v)[1] >= 0 for v in c) and not env._gate_in, (c, env._gate_in))
check('v8 holding: no defender stops', all(sgn(v) != (0, 0) for v in c), c)
c = decide8((0.5, 0.50), ROW)
check('v8 holding: shadows the raider sideways (left defender moves +x)',
      sgn(c[0])[0] > 0 and sgn(c[0])[1] >= 0, c[0])
env.gate_style = 'v7'
c = decide8((0.0, 0.50), ROW)
check('v7 in the same layout already closes (some defender moves toward the midline)',
      any(sgn(v)[1] < 0 for v in c), c)
env.gate_style = 'v8'
c = decide8((0.0, 1.05), ROW)
check('v8: raider 1.05 m in: closing starts', env._gate_in)
c = decide8((0.0, 0.70), ROW, keep_in=True)
check('v8: closing stays on when the raider pulls back to 0.70 m', env._gate_in)
c = decide8((0.0, 1.20), [(-1.0, 1.60), (-0.6, 1.60), (0.9, 0.58)], keep_in=True)
check('v8 before contact: no step toward the midline inside 0.60 m', sgn(c[2])[1] >= 0, c[2])
c = decide8((0.2, 0.90), [(-0.2, 0.70), (0.2, 1.40), (0.6, 0.70)], contact=True)
check('v8 after contact: defenders level or past press on the raider',
      sgn(c[0]) == (1, 1) and sgn(c[2]) == (-1, 1), c)
viol, stops = [], []
rng = np.random.default_rng(3)
for _ in range(300):
    defs = [(rng.uniform(-1.0, 1.0), rng.uniform(0.62, 1.65)) for _ in range(3)]
    rp = (rng.uniform(-1.0, 1.0), rng.uniform(0.05, 1.6))
    c = decide8(rp, defs)
    for (x, y), v in zip(defs, c):
        if env._phys_escape((x, y, 0)) is not None:
            continue
        stops += [(rp, (x, y))] if sgn(v) == (0, 0) else []
        q = np.array([x, y]) + np.array(v) * 0.2
        if (np.hypot(x - rp[0], y - rp[1]) >= 0.56
                and np.hypot(*(q - np.array(rp))) < 0.5 - 1e-9):
            viol.append((rp, (x, y), v))
check('v8: no defender ever stops (300 layouts)', not stops, stops[:2])
check('v8: from >= 0.56 m, no chosen move ends within 0.5 m (300 layouts)',
      not viol, viol[:2])
env.close()
def driven_v8(stop_at):
    dmin, ymin = np.inf, np.inf
    for seed in range(6):
        env = E.RoboKabaddiEnv(n_defenders=3, defender_type='gate', gate_style='v8',
                               start_layout='back_row')
        env.reset(seed=seed)
        for t in range(240):
            ry = env._get_pos(env.raider_id)[1]
            env.step(np.array([0.0, 1.0 if ry < stop_at else 0.0, 0.0], dtype=np.float32))
            r = env._get_pos(env.raider_id)
            if env._gate_closed and not env._flags['contact']:
                for d in env.defender_ids:
                    q = env._get_pos(d)
                    dmin = min(dmin, float(np.hypot(q[0] - r[0], q[1] - r[1])))
                    ymin = min(ymin, float(q[1]))
        env.close()
    return dmin, ymin


dmin, ymin = driven_v8(0.9)
check('v8: 6 raids driven to 0.9 m (holding): closest defender >= 0.45 m', dmin >= 0.45, dmin)
dmin, ymin = driven_v8(1.2)
check('v8: 6 raids driven to 1.2 m: closest >= 0.40 m (a defender holding the back row '
      'has 5 cm before the rear margin)', dmin >= 0.40, dmin)
check('v8: driven raids before contact, no defender nearer the midline than 0.55 m',
      ymin >= 0.55, ymin)

# 12. gate_style='v9': side defenders wait 1.0 m deep and wide, then close the way home ------
env = E.RoboKabaddiEnv(n_defenders=3, defender_type='gate', gate_style='v9')
env.reset(seed=0)
check("'v9' accepted", env.gate_style == 'v9')
decide9 = decide8
c = decide9((0.0, 0.50), ROW)
check('v9 holding (raider 0.5 m in): side defenders come forward to 1.0 m (-y), centre does not',
      sgn(c[0])[1] < 0 and sgn(c[2])[1] < 0 and sgn(c[1])[1] >= 0 and not env._gate_in, c)
HOLD9 = [(-0.8, 1.00), (0.0, 1.65), (0.8, 1.00)]
c = decide9((0.0, 0.50), HOLD9)
check('v9 holding at the hold points: no defender moves toward the midline, none stops',
      all(sgn(v)[1] >= 0 and sgn(v) != (0, 0) for v in c), c)
c = decide9((0.4, 0.50), HOLD9)
check('v9 holding: shadows the raider sideways (side defenders move +x)',
      sgn(c[0])[0] > 0 and sgn(c[2])[0] > 0, c)
c = decide9((0.0, 1.05), HOLD9)
check('v9: raider 1.05 m in (level with the side defenders): closing starts', env._gate_in)
check('v9 closing: side defenders head for the midline side of the raider (-y)',
      sgn(c[0])[1] < 0 and sgn(c[2])[1] < 0, c)
check('v9 closing: the centre defender stays deep until contact (no -y)',
      sgn(c[1])[1] >= 0, c[1])
c = decide9((0.0, 1.20), [(-0.8, 1.00), (0.0, 1.65), (0.9, 0.58)], keep_in=True)
check('v9 before contact: no step toward the midline inside 0.60 m', sgn(c[2])[1] >= 0, c[2])
viol, stops = [], []
rng = np.random.default_rng(4)
for _ in range(300):
    defs = [(rng.uniform(-1.0, 1.0), rng.uniform(0.62, 1.65)) for _ in range(3)]
    rp = (rng.uniform(-1.0, 1.0), rng.uniform(0.05, 1.6))
    c = decide9(rp, defs, keep_in=bool(rng.integers(2)))
    for (x, y), v in zip(defs, c):
        if env._phys_escape((x, y, 0)) is not None:
            continue
        stops += [(rp, (x, y))] if sgn(v) == (0, 0) else []
        q = np.array([x, y]) + np.array(v) * 0.2
        if (np.hypot(x - rp[0], y - rp[1]) >= 0.56
                and np.hypot(*(q - np.array(rp))) < 0.5 - 1e-9):
            viol.append((rp, (x, y), v))
check('v9: no defender ever stops (300 layouts)', not stops, stops[:2])
check('v9: from >= 0.56 m, no chosen move ends within 0.5 m (300 layouts)',
      not viol, viol[:2])
env.close()


def driven_v9(n_def=3, x0=0.0):
    """Raider driven straight in to 1.45 m, held 2 s, then straight home."""
    dmin, ymin, ahead = np.inf, np.inf, []   # dmin: side defenders only
    for seed in range(6):
        env = E.RoboKabaddiEnv(n_defenders=n_def, defender_type='gate', gate_style='v9',
                               start_layout='back_row')
        env.reset(seed=seed)
        place(env, env.raider_id, x0, -0.3)
        deep_at, cut = None, False
        for t in range(300):
            r = env._get_pos(env.raider_id)
            if deep_at is None and r[1] >= 1.45:
                deep_at = t
            vy = 1.0 if deep_at is None else (0.0 if t < deep_at + 10 else -1.0)
            env.step(np.array([0.0, vy, 0.0], dtype=np.float32))
            r = env._get_pos(env.raider_id)
            D = [env._get_pos(d) for d in env.defender_ids]
            if env._gate_closed and not env._flags['contact']:
                for k, q in enumerate(D):
                    if env._gate_role[k] != 0:
                        dmin = min(dmin, float(np.hypot(q[0] - r[0], q[1] - r[1])))
                    ymin = min(ymin, float(q[1]))
            if deep_at is not None and r[1] > 0.9:
                cut |= any(q[1] < r[1] - 0.3 and abs(q[0] - r[0]) < 0.6 for q in D)
            if deep_at is not None and r[1] < 0:
                break                                 # home
        ahead.append(cut)
        env.close()
    return dmin, ymin, ahead


dmin, ymin, ahead = driven_v9()
check('v9: 6 raids driven in to 1.45 m and home: before the raider is back to 0.9 m, '
      'a defender is on its midline side in every raid', all(ahead), ahead)
check('v9: driven raids before contact, closest side defender >= 0.40 m (raider driving '
      'home straight at a defender on the 0.60 m line; the centre one on the back row is '
      'the one driven at: v8 limit)', dmin >= 0.40, dmin)
check('v9: driven raids before contact, no defender nearer the midline than 0.55 m',
      ymin >= 0.55, ymin)
dmin, ymin, ahead = driven_v9(n_def=2)
check('v9, 1 vs 2: both are side defenders; a defender cuts the way home in every raid',
      all(ahead), ahead)

# 13. gate_style='v10': two blockers, each choosing the move that best covers the way home ----
env = E.RoboKabaddiEnv(n_defenders=3, defender_type='gate', gate_style='v10')
env.reset(seed=0)
check("'v10' accepted", env.gate_style == 'v10')
c = decide9((0.0, 0.50), ROW)
check('v10 holding: as v9 (no blockers yet; side defenders forward, centre not)',
      env._gate_blockers == [] and sgn(c[0])[1] < 0 and sgn(c[2])[1] < 0
      and sgn(c[1])[1] >= 0, (env._gate_blockers, c))
c = decide9((-0.95, 1.15), [(-0.98, 1.49), (-0.91, 1.66), (-0.14, 0.91)])
check('v10 closing: two blockers, the best placed (raider at the side line: the open-side '
      'defender is one)', len(env._gate_blockers) == 2 and 2 in env._gate_blockers,
      env._gate_blockers)
c = decide9((-0.96, 1.0), [(-0.98, 1.5), (0.5, 1.66), (-0.4, 0.2)], contact=True, keep_in=True)
check('v10 after contact: a blocker already at the nearest depth, raider level with the '
      'side line, moves straight sideways into its way home (-x, 0)', sgn(c[2]) == (-1, 0), c[2])
c = decide9((-0.96, 1.4), [(-0.98, 1.6), (0.5, 1.66), (-0.2, 1.0)], contact=True, keep_in=True)
check('v10 after contact: a blocker beside the raider and deeper than the nearest depth '
      'moves diagonally (-x, -y)', sgn(c[2]) == (-1, -1), c[2])
c = decide9((-0.9, 0.5), [(-0.98, 1.6), (0.5, 1.66), (-0.95, 0.25)], contact=True, keep_in=True)
check('v10 after contact: a blocker in front of the raider within 0.35 m drives at it',
      2 in env._gate_blockers and sgn(c[2])[1] > 0, (env._gate_blockers, c[2]))
viol, stops = [], []
rng = np.random.default_rng(5)
for _ in range(300):
    defs = [(rng.uniform(-1.0, 1.0), rng.uniform(0.62, 1.65)) for _ in range(3)]
    rp = (rng.uniform(-1.0, 1.0), rng.uniform(0.05, 1.6))
    c = decide9(rp, defs, keep_in=True)
    for (x, y), v in zip(defs, c):
        if env._phys_escape((x, y, 0)) is not None:
            continue
        stops += [(rp, (x, y))] if sgn(v) == (0, 0) else []
        q = np.array([x, y]) + np.array(v) * 0.2
        if (np.hypot(x - rp[0], y - rp[1]) >= 0.56
                and np.hypot(*(q - np.array(rp))) < 0.5 - 1e-9):
            viol.append((rp, (x, y), v))
        if y < 0.60 and v[1] < 0:
            viol.append(('0.60 m line', rp, (x, y), v))
check('v10 closing: no defender ever stops (300 layouts)', not stops, stops[:2])
check('v10 closing: from >= 0.56 m no move ends within 0.5 m, none toward the midline '
      'inside 0.60 m (300 layouts)', not viol, viol[:2])
env.close()

# 14. gate_style='v11': v10 with defenders allowed up to the boundary lines -----------------
w10 = E.RoboKabaddiEnv(n_defenders=3, defender_type='gate', gate_style='v10')._def_wall
env = E.RoboKabaddiEnv(n_defenders=3, defender_type='gate', gate_style='v11')
env.reset(seed=0)
check("'v11' accepted; wall margin 0.12 m (v10 and every other style 0.30 m)",
      env._def_wall == 0.12 and w10 == 0.30, (env._def_wall, w10))
check('v11: a defender 1.15 m out (inside the old 0.30 m margin) is not driven back',
      env._phys_escape((1.15, 1.0, 0)) is None and env._phys_escape((-1.15, 1.0, 0)) is None)
check('v11: a defender 1.20 m out is driven back in, and one 1.90 m deep is driven forward',
      env._phys_escape((1.20, 1.0, 0)) == (-1, 0) and env._phys_escape((0.0, 1.90, 0)) == (0, -1))
c = decide9((-1.15, 0.6), [(-0.98, 1.6), (0.5, 1.66), (-0.9, 0.25)], contact=True, keep_in=True)
check('v11 after contact: raider in the side strip, the blocker follows it into the strip (-x)',
      2 in env._gate_blockers and sgn(c[2])[0] < 0, (env._gate_blockers, c[2]))
env.gate_style, env._def_wall = 'v10', E.PHYS_DEF_WALL
c = decide9((-1.15, 0.6), [(-0.98, 1.6), (0.5, 1.66), (-0.99, 0.25)], contact=True, keep_in=True)
check('v10 in the same layout cannot follow (no -x at the 0.30 m margin)', sgn(c[2])[0] >= 0, c[2])
env.close()

env = E.RoboKabaddiEnv(n_defenders=3)
check('default defender type unchanged (reactive)', env.def_type == 'reactive')
env.close()

print(f'\n{passed} passed, {failed} failed')
