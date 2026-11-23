"""Checks of revision_experiments/hardware_raider.py against the recording
mock, and of its command mapping against the physical code itself:
smorphi_controller.action_to_cmd is loaded (with its network module stubbed)
and compared over a grid of actions."""
import importlib.util
import sys
import types
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE / 'mock'))
import pybullet as p                                       # the mock

REPO = HERE.parents[1]
sys.path.insert(0, str(REPO / 'revision_experiments'))
sys.path.insert(0, str(REPO))
import hardware_raider as H                                # noqa: E402
from exp_O_phase_a_rules import CountedContactRaider       # noqa: E402
from envs.kabaddi_env import RoboKabaddiEnv, COURT_W, R_HEIGHT  # noqa: E402

passed = failed = 0


def check(name, cond, detail=''):
    global passed, failed
    if cond:
        passed += 1
        print(f'  pass  {name}')
    else:
        failed += 1
        print(f'  FAIL  {name}   {detail}')


# 1. quantise == the robot's action_to_cmd ------------------------------------
sys.modules['fsr_listener'] = types.ModuleType('fsr_listener')   # no network
spec = importlib.util.spec_from_file_location(
    'smorphi_controller', str(REPO / 'physical_experiment' / 'smorphi_controller.py'))
SC = importlib.util.module_from_spec(spec)
spec.loader.exec_module(SC)
FRAC = (0.25, 0.5, 0.75, 1.0)
# Raider chars in court terms. FLIP_VX is set for the raider: a "right" char
# moves it to court -x (smorphi_controller.py, trial_run validation).
DECODE = {SC.CMD_STOP: (0, 0, 0.0)}
for table, sx, sy in ((SC.CMD_FORWARD, 0, 1), (SC.CMD_BACKWARD, 0, -1),
                      (SC.CMD_RIGHT, -1, 0), (SC.CMD_LEFT, 1, 0),
                      (SC.CMD_DIAG_FWD_R, -1, 1), (SC.CMD_DIAG_FWD_L, 1, 1),
                      (SC.CMD_DIAG_BCK_R, -1, -1), (SC.CMD_DIAG_BCK_L, 1, -1)):
    for i, ch in enumerate(table):
        DECODE[ch] = (sx, sy, FRAC[i])
grid = np.round(np.linspace(-1, 1, 81), 6)
mismatch = [(vx, vy) for vx in grid for vy in grid
            if H.quantise(vx, vy) != DECODE[SC.action_to_cmd([vx, vy, 0.0])]]
check(f'quantise matches action_to_cmd on all {len(grid)**2} grid actions',
      not mismatch, mismatch[:5])
r = H.retreat_intent(COURT_W / 2 - 0.3)
check('retreat near the right line = diagonal back-left at 100% (as hardware)',
      H.quantise(*r) == DECODE[SC.action_to_cmd([r[0], r[1], 0.0])] == (-1, -1, 1.0),
      H.quantise(*r))
check('retreat in mid-court = straight back at 100%',
      H.quantise(*H.retreat_intent(0.0)) == (0, -1, 1.0))


def make(**kw):
    env = RoboKabaddiEnv(n_defenders=3, defender_type='static',
                         start_layout='back_row', **kw)
    obs, _ = env.reset(seed=3)
    return env, obs


def place(env, x, y, yaw=0.0):
    p.resetBasePositionAndOrientation(
        env.raider_id, [x, y, R_HEIGHT / 2], p.getQuaternionFromEuler([0, 0, yaw]))


# 2. all switches off == CountedContactRaider ---------------------------------
def run_actions(raider, hw):
    """Actions over 120 steps; the mock holds one simulation at a time."""
    env, o = make()
    if hw:
        o = raider.reset(env, o)
    else:
        raider.reset()
    out = []
    for k in range(120):
        a = raider.act(env, o) if hw else raider.act_env(env, o)
        out.append(a)
        if k == 60:
            env._flags['contact'] = True
        o, *_ = env.step(a)
    env.close()
    return out


A1 = run_actions(H.HardwareRaider(3), True)
A2 = run_actions(CountedContactRaider(3), False)
check('every switch off: identical actions to CountedContactRaider (120 steps)',
      all(np.array_equal(a, b) for a, b in zip(A1, A2)))

# 3. start band ------------------------------------------------------------------
xs, ys = [], []
for seed in range(50):
    env = RoboKabaddiEnv(n_defenders=3)
    o, _ = env.reset(seed=seed)
    o = H.HardwareRaider(3, start=True).reset(env, o)
    x, y = env._get_pos(env.raider_id)[:2]
    xs.append(x); ys.append(y)
    ok_obs = abs(o[0] - x) < 1e-6 and abs(o[1] - y) < 1e-6
    env.close()
check('start: 0.15-0.41 m before the midline, x -0.63 to +0.36 (50 seeds)',
      min(ys) >= -0.41 and max(ys) <= -0.15 and min(xs) >= -0.63
      and max(xs) <= 0.36, (min(ys), max(ys), min(xs), max(xs)))
check('start: returned observation shows the new position', ok_obs)

# 4. 5 Hz hold ---------------------------------------------------------------------
env, o = make()
hr = H.HardwareRaider(3, hz5=True)
o = hr.reset(env, o)
acts = []
for k in range(24):
    a = hr.act(env, o)
    acts.append(a)
    p.resetBasePositionAndOrientation(            # move the raider each step
        env.raider_id, [0.3 * np.sin(k), -0.5, R_HEIGHT / 2], [0, 0, 0, 1])
    o, *_ = env.step(a)
held = all(np.array_equal(acts[i], acts[i - i % 4]) for i in range(24))
changes = sum(not np.array_equal(acts[i], acts[i - 1]) for i in range(1, 24))
check('5 Hz: command held for 4 steps, new decisions only on steps 0, 4, 8 ...',
      held and changes >= 1, (held, changes))
env.close()

# 5. retreat waits for the next decision -------------------------------------------
env, o = make()
hr = H.HardwareRaider(3, hz5=True, discrete=True, hw_retreat=True)
o = hr.reset(env, o)
place(env, 0.0, 0.8)
seq = []
for k in range(12):
    if k == 5:
        env._flags['contact'] = True               # counted mid-block
    seq.append(hr.act(env, o).copy())
    o, *_ = env.step(np.zeros(3, dtype=np.float32))
back = np.array([0.0, -1.0, 0.0], dtype=np.float32)
check('retreat starts at the next decision (step 8), not when contact counts (5)',
      not np.array_equal(seq[5], back) and not np.array_equal(seq[7], back)
      and np.array_equal(seq[8], back), [s.tolist() for s in seq[4:9]])
env.close()

# 6. edge escape -----------------------------------------------------------------
env, o = make()
hr = H.HardwareRaider(3, hz5=True, discrete=True, edge_escape=True)
o = hr.reset(env, o)
hr.act(env, o)                                     # step 0 decides
place(env, -COURT_W / 2 + 0.2, -0.8)
a = hr.act(env, o)                                 # step 1: mid-block
check('near the left line: +x at full speed at once, mid-block',
      np.allclose(a, [1.0, 0.0, 0.0]), a)
place(env, COURT_W / 2 - 0.2, 1.85)                # right line and far line
a = hr.act(env, o)
check('in the far-right corner: diagonal back-left (equal-speed diagonal)',
      np.allclose(a, [-1 / np.sqrt(2), -1 / np.sqrt(2), 0.0]), a)
env._flags['contact'] = True
hr.retreating = True
place(env, -COURT_W / 2 + 0.2, 0.8)
a = hr.act(env, o)
check('while retreating the escape does not override', not np.allclose(a, [1, 0, 0]),
      a)
env.close()

# 7. diagonal speed ratio ------------------------------------------------------------
for ratio in (1.0, 0.71):
    env = RoboKabaddiEnv(n_defenders=3, diagonal_speed_ratio=ratio)
    env.reset(seed=0)
    a = H.to_action(1, 1, 1.0, env.diagonal_speed_ratio)
    check(f'diagonal at 100% with ratio {ratio}: speed {ratio} of straight',
          abs(np.hypot(*a) - ratio) < 1e-9, np.hypot(*a))
    env.close()

# 8. robot frame -----------------------------------------------------------------------
env, o = make()
hr = H.HardwareRaider(3, robot_frame=True)
o = hr.reset(env, o)
hr.base.act_env = lambda e, ob: np.array([0.0, 1.0, 0.0], dtype=np.float32)
place(env, 0.0, -0.5, yaw=np.pi / 2)               # turned 90 deg left
a = hr.act(env, o)
check('robot turned 90 deg: "forward" now drives along -x',
      np.allclose(a, [-1.0, 0.0, 0.0], atol=1e-6), a)
env.close()

try:
    H.HardwareRaider(3, wings=True)
    check('unknown switch rejected', False)
except AssertionError:
    check('unknown switch rejected', True)

print(f'\n{passed} passed, {failed} failed')
