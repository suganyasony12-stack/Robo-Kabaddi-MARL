"""
hardware_raider.py — a scripted raider that behaves like the physical one
=========================================================================

The approach is the Experiment M scripted raider (straight at the nearest
defender, straight home after the environment counts the contact): the
trained policy's own path cannot be scripted. Around it, each switch adds one
behaviour of the physical raider pipeline (physical_experiment/
run_experiment.py and smorphi_controller.py):

  start        R1  placed 0.15-0.41 m before the midline, x from -0.63 to
                   +0.36 m about the centre line, as in all 24 start
                   snapshots (physical y = 1.09-1.35 m, x = 0.37-1.36 m)
  hz5          R2  a new decision 5 times a second (POLICY_HZ = 5), held
  discrete     R3  action_to_cmd: 8 directions + stop, speed level 25/50/
                   75/100% from the larger component (dead zone 0.12; levels
                   above 0.15/0.35/0.60), no turning. Levels are taken as
                   that fraction of full speed; a diagonal uses the
                   environment's diagonal_speed_ratio
  hw_retreat   R4  after counted contact, at the next decision: the
                   R5  send_raider_retreat law, (vx, vy) = (nudge, -0.8), the
                   nudge 0.5 toward the centre within 0.45 m of a side line
  edge_escape  R6  within 0.30 m of a side or end line, the boundary escape
                   command at full speed, every step, unless retreating
                   (_near_edge / _escape_cmd)
  robot_frame  R7  commands are in the robot's own frame: rotated by how far
                   the raider has turned since the start of the raid

Not ported: the x-part of _boundary_safe_cmd (a 0.20 m side-line stop that
applies the defenders' left/right sense to the raider, whose x is flipped;
the 0.30 m escape and 0.45 m nudge act before it), and the blind backward
drive used when the camera lost the raider after contact.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pybullet as p

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT.parent))
sys.path.insert(0, str(ROOT))

from exp_O_phase_a_rules import CountedContactRaider      # noqa: E402
from envs.kabaddi_env import (COURT_W, COURT_L, MIDLINE_Y, R_HEIGHT,  # noqa: E402
                              hardware_command)

START_Y = (-0.41, -0.15)          # m from the midline (24 snapshots)
START_X = (-0.63, 0.36)           # m from the centre line (24 snapshots)
DECIDE_EVERY = 4                  # control steps per decision at 5 Hz, dt 0.05
EDGE = 0.30                       # BOUNDARY_MARGIN (run_experiment.py)
NUDGE_MARGIN, NUDGE, RETREAT_VY = 0.45, 0.5, -0.8   # send_raider_retreat
SWITCHES = ("start", "hz5", "discrete", "hw_retreat", "edge_escape",
            "robot_frame")


# action_to_cmd in court terms: (sx, sy, speed fraction). One definition,
# in the environment (raider_command="hardware" uses it too).
quantise = hardware_command


def to_action(sx, sy, frac, diag_ratio):
    """Court direction and speed fraction -> environment action."""
    if sx and sy:
        frac *= diag_ratio / np.sqrt(2.0)
    return np.array([frac * sx, frac * sy])


def escape_dir(x, y):
    """_escape_cmd for the raider, in court terms; None when clear."""
    left, right = x < -COURT_W / 2 + EDGE, x > COURT_W / 2 - EDGE
    home, far = y < -COURT_L / 2 + EDGE, y > COURT_L / 2 - EDGE
    if not (left or right or home or far):
        return None
    return (1 if left else -1 if right else 0), (1 if home else -1 if far else 0)


def retreat_intent(x):
    """send_raider_retreat: drive home, nudged toward the centre."""
    vx = (NUDGE if x < -COURT_W / 2 + NUDGE_MARGIN
          else -NUDGE if x > COURT_W / 2 - NUDGE_MARGIN else 0.0)
    return np.array([vx, RETREAT_VY])


class HardwareRaider:
    """Scripted raider with switchable hardware behaviours (see module doc).
    With every switch off it acts exactly as CountedContactRaider."""

    def __init__(self, n_def, **switches):
        bad = set(switches) - set(SWITCHES)
        assert not bad, f"unknown switches {bad}"
        self.sw = {k: bool(switches.get(k, False)) for k in SWITCHES}
        self.base = CountedContactRaider(n_def)

    def reset(self, env, obs):
        """Call right after env.reset(); returns the observation to use."""
        self.base.reset()
        self.k = 0
        self.cmd = np.zeros(2)
        self.retreating = False
        if self.sw["start"]:
            x = env.np_random.uniform(*START_X)
            y = MIDLINE_Y + env.np_random.uniform(*START_Y)
            p.resetBasePositionAndOrientation(
                env.raider_id, [x, y, R_HEIGHT / 2], [0, 0, 0, 1],
                physicsClientId=env._client)
            env._prev_pos = np.array([x, y], dtype=np.float32)
            obs = env._obs()
        self.yaw0 = float(env._get_pos(env.raider_id)[2])
        return obs

    def act(self, env, obs):
        sw = self.sw
        pos = env._get_pos(env.raider_id)
        tick = (self.k % DECIDE_EVERY == 0) if sw["hz5"] else True
        self.k += 1
        esc = escape_dir(pos[0], pos[1]) if sw["edge_escape"] else None

        if esc is not None and not self.retreating:
            self.cmd = to_action(esc[0], esc[1], 1.0, env.diagonal_speed_ratio)
        elif tick:
            if env._flags["contact"]:
                self.retreating = True
            if self.retreating and sw["hw_retreat"]:
                intent = retreat_intent(pos[0])
            else:
                intent = self.base.act_env(env, obs)[:2]
            if sw["discrete"]:
                self.cmd = to_action(*quantise(*intent), env.diagonal_speed_ratio)
            else:
                self.cmd = np.asarray(intent, dtype=float)

        a = self.cmd
        if sw["robot_frame"]:
            th = float(pos[2]) - self.yaw0
            c, s = np.cos(th), np.sin(th)
            a = np.array([c * a[0] - s * a[1], s * a[0] + c * a[1]])
        return np.array([a[0], a[1], 0.0], dtype=np.float32)
