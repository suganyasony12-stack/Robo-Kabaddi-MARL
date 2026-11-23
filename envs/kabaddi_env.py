"""
Robo-Kabaddi Simulation Environment
====================================
PyBullet-based multi-agent environment for Robo-Kabaddi.

Coordinate system:
  - Court: 2.6m wide (x), 4.0m long (y)
  - Raider half:   y in [-2.0,  0.0]
  - Defender half: y in [ 0.0,  2.0]
  - Midline at y = 0.0
  - Baulk line at y = 0.5  (raider must cross to validate raid)
  - Bonus line at y = 1.5  (crossing earns bonus point)

Usage:
  python kabaddi_env.py          # runs a quick demo with random actions
  python kabaddi_env.py --render # runs with PyBullet GUI (visual)
"""

import argparse
import os
import time
import numpy as np
import pybullet as p
import pybullet_data
import gymnasium as gym
from gymnasium import spaces


# ══════════════════════════════════════════════════════════════════════════════
#  Court constants
# ══════════════════════════════════════════════════════════════════════════════

COURT_W   = 2.6   # metres
COURT_L   = 4.0
MIDLINE_Y = 0.0
BAULK_Y   = 0.5
BONUS_Y   = 1.5
R_RADIUS  = 0.085   # robot radius  (Smorphi ~170 mm footprint)
R_HEIGHT  = 0.315   # robot height

# Robot shape. "cylinder" (default) is the shape behind the submitted results.
# "box" is the real Smorphi footprint, a 170 mm square (half-side R_RADIUS),
# spawned with its faces along the court axes as the robots were placed.
# Two boxes touch at 0.17 m centre to centre face on and 0.24 m corner to
# corner; two cylinders always touch at 0.17 m.
ROBOT_SHAPES = ("cylinder", "box")


# ══════════════════════════════════════════════════════════════════════════════
#  Locomotion model
# ══════════════════════════════════════════════════════════════════════════════
#
# Two locomotion models are provided. Select with the `locomotion` argument, or
# set the environment variable ROBO_KABADDI_LOCOMOTION to override the default
# without editing any calling script.
#
# "planar"  (default)  — the corrected model.
#     The robots are holonomic, velocity-commanded mecanum platforms, so they
#     are modelled as a velocity servo on a flat floor:
#       * physics runs at PHYSICS_DT (1/240 s, PyBullet's recommended step) and
#         each 0.05 s control step is split into sub-steps;
#       * the commanded velocity is re-applied at every sub-step (zero-order
#         hold), as a motor controller would track it;
#       * the base is frictionless against the floor: a wheeled robot rolls
#         rather than sliding its chassis, so floor friction must not oppose
#         the commanded motion;
#       * roll and pitch are held at zero after every sub-step, since the real
#         platform does not tip over during play;
#       * mass is the measured 1.8 kg;
#       * contact force is sampled at every sub-step and the maximum over the
#         control step is reported, as a 100 Hz FSR would see it.
#
# "legacy"  — the model used for the results submitted in ROBOT-D-26-01371.
#     One 0.05 s physics step per control step, velocity reset once before it,
#     floor friction 0.8, damping 0.9, mass 1.5 kg. Under this model ground
#     friction removes up to mu*g*dt = 0.39 m/s within each step, more than the
#     0.30 m/s maximum command, so a robot does not translate unless the
#     command is near maximum, and the friction torque at the base tips it.
#     Kept unchanged, line for line, so that the submitted results remain
#     reproducible. Do not use it for new experiments.
#
# "planar_force" — "planar", but each robot's drive has a force limit, so a
#     robot can be held or pushed back. Instead of setting the velocity, every
#     sub-step applies F = K (v_cmd - v), capped at drive_force in magnitude.
#     A stopped robot resists being pushed with the same drive. A full-speed
#     robot pushing a stopped one settles at drive_force / K, so
#         K = drive_force / (push_yield * MAX_LIN_VEL).
#     Values from hardware logs (revision_experiments/FINDINGS_P.md, FINDINGS_Q.md):
#       push_yield  0.11 — a stopped defender moved at 0.11 of the raider's
#                   approach speed while in contact (FSR-validation runs,
#                   n = 60, IQR 0.06-0.19). In "planar" it is 0.50.
#       drive_force 1.7 N — the raider reaches 90% of 0.30 m/s in a median
#                   0.27 s from rest (about 0.93 m/s^2 on 1.8 kg). Provisional
#                   until a force-gauge measurement (protocol B1b).
#       hold_force  5.1 N — the limit when the drive is resisting motion (its
#                   force opposes the robot's current velocity): the robot's
#                   grip and gearbox, which resist being pushed harder than the
#                   motors can drive. It must exceed drive_force; with equal
#                   limits a pushed pair coasts at the bump speed and the
#                   yield is about 0.46 whatever push_yield is (FINDINGS_Q.md).
#                   Provisional, 3 x drive_force; the measured yield changes
#                   by under 0.01 for 3.4-8.5 N. Protocol B1b
#                   (resist_stopped) measures it directly.
#     Turning is still set directly, as in "planar". Pushes are isotropic: the
#     logged approaches were all face on, so sideways yielding is unmeasured.
#
LOCOMOTION_MODES   = ("planar", "legacy", "planar_force")
SUBSTEPPED_MODES   = ("planar", "planar_force")
DEFAULT_LOCOMOTION = "planar"
PHYSICS_DT         = 1.0 / 240.0   # sub-step length for the planar model
PUSH_YIELD         = 0.11          # planar_force: see above
DRIVE_FORCE        = 1.7           # N, planar_force: see above
HOLD_FORCE         = 5.1           # N, planar_force: see above
ROBOT_MASS_PLANAR  = 1.8           # kg, measured on the Smorphi platform
ROBOT_MASS_LEGACY  = 1.5           # kg, value used in the submitted results


# ══════════════════════════════════════════════════════════════════════════════
#  Physical defender controller  (defender_type="physical")
# ══════════════════════════════════════════════════════════════════════════════
#
# A port of the controller the defenders ran in the physical experiment:
# physical_experiment/smorphi_controller.py (send_defender_action,
# defender_move_toward) and the boundary override in
# physical_experiment/run_experiment.py (_near_edge, _escape_cmd). Distances
# are measured from the court lines, so they carry over between the physical
# court (2.0 x 3.0 m, midline at y = 1.5 m) and this one.
#
#   * a new pursuit command 5 times a second (DEFENDER_HZ), held in between;
#   * eight directions plus stop, chosen by the 0.3 component test of
#     defender_move_toward, always at the 100% level (DEFENDER_SPEED = 100),
#     which is the raider's full speed: it is the same robot;
#   * stationary while the raider is in its own half (its current position,
#     not whether it has crossed before);
#   * before contact, stop at the soft boundary 0.30 m past the midline
#     (DEFENDER_SOFT_Y = 1.80) and never step toward the midline within
#     0.10 m of it;
#   * after contact, the soft boundary is released and defenders pursue until
#     the raider is back in its own half;
#   * always: a defender within 0.15 m of the midline is driven straight back
#     (DEFENDER_MIN_Y = 1.65), and one within 0.30 m of a side or the rear
#     wall is driven away from it (BOUNDARY_MARGIN, DEFENDER_TOP_MARGIN). This
#     override is checked at every control step, as the hardware loop checked
#     it at every camera frame, and it pre-empts pursuit. The manuscript says
#     pursuit after contact runs "up to the hard boundary"; the code held
#     defenders 0.15 m short of the midline, and this port follows the code.
#
# Not ported, because the override above makes them unreachable for a
# defender: the 1.50 m hard-boundary reversal and _boundary_safe_cmd (20 cm wall
# margin) inside send_defender_action.
#
PHYS_DEF_HZ          = 5      # pursuit command rate (DEFENDER_HZ)
PHYS_DEF_GUARD       = 0.15   # always driven back inside this, from the midline
PHYS_DEF_SOFT        = 0.30   # soft boundary before contact, from the midline
PHYS_DEF_CREEP       = 0.10   # no step toward the midline this close to soft
PHYS_DEF_WALL        = 0.30   # driven away from a side or the rear wall
PHYS_DEF_STOP_DIST   = 0.05   # stop when this close to the raider's centre
PHYS_DEF_AXIS_THRESH = 0.3    # direction component needed to move along an axis

# Defender start for start_layout="back_row": one row at the back of the
# defenders' half. Distance from the back wall, as measured in the 24 start
# snapshots of the physical experiment (physical_experiment/episode_snapshots,
# defender y = 2.69-2.87 m on the 3.0 m court).
BACK_ROW_FROM_WALL = (0.13, 0.31)   # metres, min and max


# ══════════════════════════════════════════════════════════════════════════════
#  Gate defence  (defender_type="gate")
# ══════════════════════════════════════════════════════════════════════════════
#
# A coordinated defence designed by the author, NOT the one the physical
# defenders ran (they used "physical" pursuit). Same robot and rules as
# "physical": 5 Hz decisions, eight directions at full speed, the 0.15 m
# midline guard and 0.30 m wall margins at every step, the 0.30 m line before
# contact (no step toward the midline inside 0.40 m), and standing still while
# the raider is in its own half. Only the targets differ.
#
# As soon as the raider crosses the midline, every defender works to close
# the raider's way home:
#   slots: one per defender in a wall GATE_BEHIND behind the raider (on the
#     midline side, or as close as the rules allow), GATE_SLOT apart, centred
#     on the raider's lane. Defenders take the slots in their left-to-right
#     order, so their paths do not cross.
#   passing: a defender not yet level with the raider first steps sideways
#     out of the raider's lane until it is GATE_CLEAR to the side (out of its
#     reach), then moves toward the midline in a lane between GATE_CLEAR and
#     GATE_FAR to the side (one further out moves in as it passes), on the
#     side of its slot (the centre defender on the side it is already on).
#     If that side has no room inside the wall margin, it passes on the
#     other side, so it is never held against a wall.
#   closing: it drives a 45-degree diagonal, in toward its slot and toward
#     the midline together, at full speed, until it reaches the slot's
#     sideways position or depth, then finishes straight. It starts as soon
#     as it is level with the raider.
#   after contact: a defender level with or past the raider drives at it and
#     keeps pressing, to catch and hold it; the others keep passing.
#   a defender at a rule limit (no step toward the midline inside 0.40 m
#     before contact) slides sideways toward its target instead of stopping.
#
# The distances are design values from the robot's size (0.17 m box, contact
# at 0.17-0.24 m); they are not tuned to outcomes.
#
GATE_SLOT        = 0.19   # m between neighbouring slots (0.17 m box + 0.02 m)
GATE_BEHIND      = 0.25   # m, the wall behind the raider, toward the midline
GATE_CLEAR       = 0.50   # m, sideways clearance while passing the raider
GATE_FAR         = 1.00   # m, furthest to the side while passing (author, 25 Oct)
GATE_ARRIVE      = 0.05   # m, 'at slot depth' within this
GATE_FLOOR_AFTER = 0.20   # m past the midline: nearest slot depth after contact
#
# gate_style="keepout" (default since 25 Oct, author's design): before contact
# no defender comes within GATE_KEEPOUT of the raider. The wall stands
# GATE_BEHIND_KO behind it. At every decision each defender picks, from the
# eight directions and stop, the move that brings it closest to its slot
# while its position GATE_LOOKAHEAD ahead stays GATE_KEEPOUT from the raider's
# predicted position (current velocity). This gives an early diagonal
# whenever it is safe, and a step out or a pass round the raider when it is
# not. If no move is safe, it takes the one that ends furthest from the
# raider. After contact, defenders level with or past the raider press on
# it; the others keep the rule until they are level.
# gate_style="v5" keeps the previous version (step out, pass, forced
# diagonal from level; wall 0.25 m behind).
# gate_style="v7" (author, 26 Oct) is 'keepout' with two changes, so that no
# defender waits near the midline to be touched (FINDINGS_U.md):
#   never freeze: while the raider is in the defenders' half a defender
#     always moves (stop is not a choice, also at its slot). If no move keeps
#     GATE_KEEPOUT, it takes the move ending furthest from the raider,
#     preferring one away from the midline.
#   go back: before contact, a defender less than GATE_BACK_BAND past the
#     0.40 m line with the raider closer than GATE_KEEPOUT + GATE_BACK_WARN
#     heads for a point GATE_BEHIND_KO deeper than the raider and GATE_CLEAR
#     to its own side, going round the raider under the keep-out rule, until
#     it reaches that depth (GATE_BEHIND_KO deeper than the raider, or the
#     rear margin); then back to its slot. (Stopping when only level left it
#     beside the raider against a side wall, to be caught there: wall-edge
#     fix, 26 Oct, FINDINGS_U.md.)
#   FROZEN 26 Oct (author): v7 is the held-out evaluation gate; v6 stays the
#     training gate. Do not change v7; add a new style instead.
# gate_style="v8" (author, 27 Oct; FINDINGS_V.md) is v7 played the way real
# kabaddi defenders play: hold back and deep, let the raider come in, then
# surround it and close its way home.
#   holding: from the raider's crossing until it is GATE_IN_DEPTH deep, the
#     defenders keep the back row (GATE_HOLD_Y) and shadow the raider
#     sideways, GATE_HOLD_SPREAD apart and centred on its lane, under the
#     keep-out rule (never within GATE_KEEPOUT; never stopping).
#   closing: once the raider is GATE_IN_DEPTH deep (or contact), the v7
#     gate for the rest of the raid, with the nearest approach to the
#     midline before contact GATE_V8_LIMIT instead of 0.40 m (the go-back
#     zone moves with it). After contact unchanged (pressing; 0.20 m).
# gate_style="v9" (author, 27 Oct; FINDINGS_W.md) is v8 with a line that can
# close the way home: in v8 every defender waits deeper than the raider, so
# after a deep touch none can get to its midline side at equal speed.
#   holding: the side defenders (every one but the centre; with 1 vs 2 both)
#     wait GATE_SIDE_Y deep, GATE_SIDE_X to each side of the raider, and
#     shadow it sideways; the centre one keeps the back row (GATE_HOLD_Y).
#     The raider walks in between the side defenders.
#   closing: once the raider is GATE_IN_DEPTH deep (level with the side
#     defenders) or contact, the side defenders go round it to their slots
#     on its midline side (v8's closing); the centre one stays on the back
#     row until contact. Every other v8 rule unchanged.
# gate_style="v10" (author, 27 Oct; FINDINGS_X.md) is v9 with at least two
# defenders closing the way home, each choosing its move to give the best
# chance of a catch (in v9 one closer was often pinned at a side wall, and
# the other drove one fixed diagonal):
#   blockers: once closing starts, the GATE_CATCH_N defenders best placed to
#     reach the raider's way home first (largest catch margin, below; all
#     of them with fewer defenders); the rest keep the v9 rules.
#   ways home: the raider's three fastest lines to the midline, straight and
#     45 degrees to either side (straight down along a side line once it is
#     there). Catch margin of a defender on a line = the most, over points
#     on it, of (raider's distance to the point - defender's distance), at
#     equal speed: > 0 means the defender gets there first.
#   choosing a move: at every decision a blocker takes, from the eight
#     directions, the move that makes the worst-covered way home (each way
#     covered by the blocker with the best margin on it) as well covered as
#     possible, given where the other blocker is. So it moves sideways when
#     sideways gains most, diagonally when that does.
#   before contact: points on the ways home no nearer the midline than
#     GATE_BEHIND_KO in front of the raider or GATE_V8_LIMIT; keep-out, the
#     0.60 m limit and going back as in v9. After contact: points down to
#     GATE_FLOOR_AFTER; a blocker on the raider's midline side within
#     GATE_CATCH_PRESS drives at it (catch and hold).
# gate_style="v11" (author, 27 Oct; FINDINGS_X.md) is v10 with the defenders
# allowed up to the boundary lines: the 0.30 m wall margin (PHYS_DEF_WALL,
# the robots' controller rule) becomes GATE_LINE_MARGIN, enough to keep the
# whole robot inside the line. In v10 a raider stopped in front of a blocker
# stepped into the side strip the defenders could not enter and got home.
GATE_STYLES      = ("keepout", "v5", "v7", "v8", "v9", "v10", "v11")
GATE_KEEPOUT     = 0.50   # m, the author's safe distance before contact
GATE_BEHIND_KO   = 0.55   # m, keep-out wall behind the raider (0.50 + 0.05)
GATE_LOOKAHEAD   = 0.20   # s, one 5 Hz decision
GATE_BACK_BAND   = 0.20   # m past the 0.40 m line: 'too close to the midline'
GATE_BACK_WARN   = 0.20   # m beyond GATE_KEEPOUT: the raider is coming
GATE_IN_DEPTH    = 1.00   # m past the midline: v8 starts closing (author)
GATE_V8_LIMIT    = 0.60   # m, v8: nearest to the midline before contact (author)
GATE_HOLD_SPREAD = 0.60   # m between neighbouring defenders while holding
GATE_HOLD_Y      = 1.65   # m, the back row while holding (rear margin - 0.05)
GATE_SIDE_Y      = 1.00   # m, v9: side defenders' depth while holding (= GATE_IN_DEPTH)
GATE_SIDE_X      = 0.80   # m, v9: side defenders beside the raider while holding (author)
GATE_CATCH_N     = 2      # v10: defenders closing the way home (author: at least two)
GATE_CATCH_STEP  = 0.05   # m, v10: spacing of points on a way home
GATE_CATCH_PRESS = 0.35   # m, v10: after contact, drive at the raider this close
GATE_RAIDER_X    = COURT_W / 2 - 0.10   # m, v10: raider's ways home stay inside this
GATE_LINE_MARGIN = 0.12   # m, v11: defender centre to a boundary line (half box 0.085 + 0.035)


# ══════════════════════════════════════════════════════════════════════════════
#  Learned defenders and the defenders' fouls (C3, author's design, 27 Oct)
# ══════════════════════════════════════════════════════════════════════════════
#
# defender_type="learned": each defender is an agent. Before every env.step it
# is given one of the robots' own defender commands, DEF_ACTIONS: stop or one
# of the eight directions at the 100% level, as the physical defenders were
# driven (_phys_velocity, so diagonal_speed_ratio applies). The command is
# held for the action_repeat control steps of that env.step (5 Hz with
# CORRECTED_CONFIG). There is no boundary escape: the fouls below take its
# place, so the defenders must learn the lines.
#
# defender_fouls=True (author's Kabaddi rules): a defender whose centre
# crosses the midline into the raider's half, or a side line or the end line,
# at any time during the raid, is out and the raiding team scores one point
# per defender who went over. One raid = one result and no defender is
# removed, so the raid ends at once as "foul": the raider's reward is R_TOUCH
# per defender over. The robot centre is judged, as the camera judges the
# raider. The raider's own foul, its centre over a line, is the A5 tackle.
#
# Team rewards (TEAM_REWARD, info["team_reward"] = (raider, defenders), on the
# last step of the raid, with defender_type="learned"): results only, the same
# value for every defender, mirror values for the two teams. A foul counts
# once per defender over, for both the defenders' and the raider's foul. An
# Empty raid costs both teams 0.3 (author: push for a result).
DEF_ACTIONS = ((0, 0), (0, 1), (1, 1), (1, 0), (1, -1),
               (0, -1), (-1, -1), (-1, 0), (-1, 1))   # (sx, sy); +y away from the midline
TEAM_REWARD = {"touch": (1.0, -1.0), "foul": (1.0, -1.0),
               "tackle": (-1.0, 1.0), "empty": (-0.3, -0.3)}

# raider_edge_escape=True: the robots' boundary escape for the raider
# (run_experiment.py, _near_edge / _escape_cmd with BOUNDARY_MARGIN; R6 of
# hardware_raider.py). Within RAIDER_EDGE of a side line or either end line
# the raider's own command is replaced, every control step, by the move away
# from that line (diagonal in a corner) at the 100% level. It overrides the
# raider's steering only: a defender can still push it over a line, which is
# still the A5 tackle. Author (27 Oct): used for frozen raiders while the
# defenders learn, since a frozen raider cannot learn the lines; a learning
# raider plays without it, under the out rule alone.
RAIDER_EDGE = 0.30   # m, BOUNDARY_MARGIN


# ══════════════════════════════════════════════════════════════════════════════
#  Outcome rules
# ══════════════════════════════════════════════════════════════════════════════
#
# "legacy"      — the classification behind the submitted results. The raid
#     timer runs from reset; at timeout the raid is a Tackle if contact was
#     made and an Empty otherwise, wherever the raider is.
# "section_2_3" — the rules of manuscript Section 2.3, as the physical system
#     applied them (physical_experiment/run_experiment.py):
#       * the raid timer (raid_time, 30 s) starts when the raider crosses the
#         midline (RAID_TIME_S, counted from raid_start_t);
#       * a raider that has not crossed within APPROACH_TIME is an Empty raid
#         (MAX_APPROACH_S);
#       * at timeout the raid is a Tackle if the raider is still past the
#         midline, with or without contact, and an Empty otherwise. A raider
#         back home after contact has already ended the raid as a Touch.
#     The reward follows the label (R_TACKLE or R_EMPTY). The time-left
#     observation still counts from reset, as the physical observation did
#     (obs_builder.py). The 7 s "static" Tackle is not included.
# "physical"    — everything in "section_2_3", plus the static-raider Tackle
#     of the physical system (run_experiment.py, RAIDER_STATIC_DIST_M and
#     RAIDER_STATIC_TIME_S, check 3c): once the raid has started, a raider in
#     the defenders' half that stays within 5 cm of a reference point for
#     more than 7 s is tackled. The reference point moves to the raider each
#     time it gets more than 5 cm away, and is cleared whenever the raider is
#     not in the defenders' half. Applies before and after contact.
#     Also (author's rule, A5; the robots' checks 3a/3b): a raider whose
#     centre crosses a court line is tackled once the raid has started, and
#     the raid ends Empty if it leaves before crossing the midline. With the
#     default court_walls=True the walls keep it inside, so this never happens.
#   held_needs_touch=True (author, 28 Oct): the static-raider Tackle counts
#     only while a defender's body touches the raider (PyBullet contact
#     points, as the force model reads them); the still clock restarts when
#     no defender touches. Real Kabaddi: the defenders must hold the raider.
#     The 30 s timeout is unchanged: a raider still past the midline is
#     tackled with or without contact (author, 28 Oct). Default False.
#
OUTCOME_RULES = ("legacy", "section_2_3", "physical")
APPROACH_TIME = 20.0   # seconds to reach the midline (MAX_APPROACH_S)
STATIC_DIST   = 0.05   # metres (RAIDER_STATIC_DIST_M)
STATIC_TIME   = 7.0    # seconds (RAIDER_STATIC_TIME_S); the static_time default


# ══════════════════════════════════════════════════════════════════════════════
#  Contact rules
# ══════════════════════════════════════════════════════════════════════════════
#
# What counts as a contact for the game (touch, tackle-after-contact, the
# reward terms and the 'physical' defenders' pursuit mode):
#
# contact_model "force"  (default) — a defender presses on the raider with
#     more than FSR_THRESH (0.30 N), as a force sensor would read it.
# contact_model "camera" — a defender's centre is within contact_dist of the
#     raider's centre, as the physical system's overhead camera judged it
#     (run_experiment.py, camera_contact). contact_dist has no default: the
#     code used 0.35 m and the paper states 0.25 m. The observation's contact
#     input is then 1.0 on the left channel while any defender is that close
#     and 0.0 otherwise, right channel 0.0, as on the robots (obs_builder.py).
#
# contact_min_depth  — count a contact only while the raider is more than this
#     far past the midline. Default None (anywhere). The physical system used
#     0.27 in its simulation units (deep = raider_sim[1] > 0.27), which is
#     0.20 m on the 3.0 m physical court.
# contact_confirm_steps — the contact counts once this many control steps have
#     qualified. Default 1. The physical system used CONTACT_CONFIRM_N = 2
#     camera frames at 30 Hz, counted cumulatively: its counter is never reset
#     between frames, so the two need not be consecutive. Here the count is in
#     0.05 s control steps.
#
# Once counted, contact stays counted for the rest of the raid. The defaults
# reproduce the submitted results exactly.
#
CONTACT_MODELS = ("force", "camera")


# ══════════════════════════════════════════════════════════════════════════════
#  Raider commands and start  (raider_command, action_repeat, raider_start)
# ══════════════════════════════════════════════════════════════════════════════
#
# raider_command "continuous" (default) — the action is (vx, vy, turn), each
#     scaled to its maximum, so a diagonal reaches 0.42 m/s.
# raider_command "hardware" — the action goes through the robots' own mapping
#     (physical_experiment/smorphi_controller.py, action_to_cmd): eight
#     directions + stop, speed level 25/50/75/100% from the larger component
#     (dead zone 0.12; levels above 0.15/0.35/0.60), turning disabled. Speed is
#     the level times MAX_LIN_VEL; a diagonal uses diagonal_speed_ratio.
# action_repeat — control steps (0.05 s) per env.step. 4 = one 5 Hz decision
#     (POLICY_HZ = 5) held for 0.2 s; physics, defenders, contact and the
#     outcome rules still run every 0.05 s inside; rewards are summed; the
#     step returns as soon as the raid ends. Default 1.
# raider_start "near_midline" — 0.15-0.41 m before the midline, x -0.63 to
#     +0.36 m about the centre line, as in all 24 physical start snapshots
#     (CORRECTIONS_TRACKER.md P8). Default "legacy".
#
RAIDER_COMMANDS   = ("continuous", "hardware")
RAIDER_STARTS     = ("legacy", "near_midline")
CMD_DEADZONE      = 0.12
CMD_LEVELS        = ((0.60, 1.0), (0.35, 0.75), (0.15, 0.5))   # else 0.25
NEAR_MIDLINE_X    = (-0.63, 0.36)
NEAR_MIDLINE_Y    = (-0.41, -0.15)


def hardware_command(vx, vy):
    """action_to_cmd in court terms: (sx, sy, speed fraction)."""
    if abs(vx) < CMD_DEADZONE and abs(vy) < CMD_DEADZONE:
        return 0, 0, 0.0
    sx = int(vx > CMD_DEADZONE) - int(vx < -CMD_DEADZONE)
    sy = int(vy > CMD_DEADZONE) - int(vy < -CMD_DEADZONE)
    m = max(abs(vx), abs(vy))
    frac = next((f for t, f in CMD_LEVELS if m > t), 0.25)
    return sx, sy, frac


# Every correction made in the rebuild, for training (C1) to read from one
# place, with the author's choices of 25 Oct: camera contact at 0.25 m (A4)
# counted 0.20 m past the midline after 2 steps (A2), no walls and the
# out-of-court rule (A5), and Empty as soon as the raider crosses back without
# contact (Section 2.3). The held-raider time of 4 s and the gate defence
# are the author's design (G1), not the robot code.
CORRECTED_CONFIG = dict(
    locomotion="planar_force", robot_shape="box",
    defender_type="gate", gate_style="keepout", start_layout="back_row",
    raider_start="near_midline", raider_command="hardware", action_repeat=4,
    outcome_rules="physical", static_time=4.0,
    contact_model="camera", contact_dist=0.25,
    contact_min_depth=0.20, contact_confirm_steps=2,
    court_walls=False, empty_on_return=True,
)


# ══════════════════════════════════════════════════════════════════════════════
#  Environment
# ══════════════════════════════════════════════════════════════════════════════

class RoboKabaddiEnv(gym.Env):
    """
    Single-raider Robo-Kabaddi environment.

    Parameters
    ----------
    n_defenders   : int   — number of defenders (1 to 7)

                            Kabaddi permits exactly one raider per raid, so
                            the scaling axis toward the full 7-a-side format
                            is the defender count. Layouts for 1, 2 and 3 are
                            the original hand-placed ones and are unchanged;
                            4 and above use a generated arc formation.
    defender_type : str   — 'static' | 'reactive' | 'wall' | 'physical' | 'gate'
                            | 'learned'
                            'reactive' is the controller behind the submitted
                            results (0.20 m/s cap). 'physical' is the
                            controller the physical defenders ran, at the
                            raider's speed (see the block above the class).
                            'gate' is a coordinated defence on the same
                            robot (block above the class); not the hardware's.
                            'learned': each defender takes an action from
                            set_defender_actions (block above the class).
    render        : bool  — open PyBullet GUI
    dt            : float — control period (seconds). The policy acts once
                            per dt. Under the legacy model this is also the
                            physics step; under the planar model physics runs
                            at PHYSICS_DT and each control step is sub-stepped.
    raid_time     : float — seconds per raid. Under outcome_rules='legacy'
                            counted from reset; under 'section_2_3' from the
                            raider's midline crossing.
    locomotion    : str   — 'planar' (default, corrected) | 'legacy'
                            (reproduces the submitted results) |
                            'planar_force' (planar with force-limited drives,
                            so robots can block each other). If omitted,
                            the ROBO_KABADDI_LOCOMOTION environment variable
                            is used, then DEFAULT_LOCOMOTION.
    diagonal_speed_ratio : float — speed of a diagonal command as a fraction
                            of a straight one. Used by 'physical' defenders.
                            Default 1.0 (equal speed): not yet measured on the
                            hardware. Ideal mecanum kinematics gives about
                            0.71 on a pure diagonal.
    diag_ratio_range : (lo, hi) or None — author, 11 Nov: the robots' diagonal
                            speed varies by robot, direction, trial and place
                            on the court (B3, 4 robots: 0.34-0.83 of straight).
                            Set: every robot (raider and each defender) gets
                            its own diagonal ratio, uniform in [lo, hi], at
                            every reset, from a generator seeded by the reset
                            seed (start positions unchanged). None (default):
                            diagonal_speed_ratio for every robot, as before.
    diag_ratio_period : float or None — with diag_ratio_range, draw new
                            ratios every this many seconds of the raid
                            (places on the court). None: once per raid.
    start_layout  : str   — 'legacy' (default; the layout behind the submitted
                            results) | 'back_row' (defenders in one row at the
                            back wall, as placed in the physical experiment).
    defender_start_depth : (lo, hi) or None — author, 13 Nov: with
                            start_layout 'back_row', the defenders' start depth
                            in metres past the midline, uniform in [lo, hi]
                            (e.g. (1.0, 1.5): between the baulk and bonus
                            lines, 3.0-3.5 m from the raider's end line on the
                            2.6 x 4.0 m court). Same random draws as the back
                            row, so a seed gives the same across-court places.
                            None (default): the back row (BACK_ROW_FROM_WALL).
    outcome_rules : str   — 'legacy' (default; the classification behind the
                            submitted results) | 'section_2_3' (Section 2.3
                            and the physical system) | 'physical'
                            ('section_2_3' plus the static-raider Tackle);
                            see the block above the class.
    contact_model : str   — 'force' (default) | 'camera'
    contact_dist  : float — camera contact distance, metres; required with
                            contact_model='camera', no default
    contact_min_depth : float or None — count contact only this far past the
                            midline (default None: anywhere)
    contact_confirm_steps : int — control steps of contact needed, counted
                            cumulatively (default 1)
    raider_command : str  — 'continuous' (default) | 'hardware' (the robots'
                            action_to_cmd mapping; block above the class)
    action_repeat : int   — control steps per env.step (default 1; 4 = 5 Hz)
    raider_start  : str   — 'legacy' (default) | 'near_midline' (snapshots)
    court_walls   : bool  — True (default): 0.15 m walls at the lines. False:
                            painted lines only, as on the physical court, so
                            a robot can cross them
    empty_on_return : bool — False (default): a raider back in its own half
                            without contact plays on until the timer. True:
                            crossing back without a counted contact ends the
                            raid at once as Empty, as Section 2.3 defines it
                            (the robot code waited for the timer)
    return_margin : float — author, 13 Nov: with empty_on_return, the raider
                            counts as back only when its centre is this many
                            metres behind the midline (0.03: a few mm of camera
                            wobble at the line is not a return). Default 0.0,
                            as before. The touch and the midline crossing are
                            judged as before.
    gate_style    : str   — 'keepout' (default) | 'v5' | 'v7' | 'v8'; used by
                            defender_type='gate' (block above the class)
    static_time   : float — seconds a raider may stay within 5 cm in the
                            defenders' half before outcome_rules='physical'
                            tackles it. Default 7.0, the robots' value
                            (RAIDER_STATIC_TIME_S); other values depart from
                            the robot code.
    hold_dist     : float or None — author, 14 Nov: the distance (m) the raider
                            may move and still count as held (the 'within 5 cm'
                            above). None (default): STATIC_DIST, 0.05 m.
    hold_touch_dist : float or None — author, 14 Nov: with held_needs_touch, a
                            defender counts as touching the raider when the
                            centres are at most this far apart (0.25 m: the
                            touch rule's distance; on the robots the raider
                            slips and turns while held, so body contact comes
                            and goes). None (default): body contact, as before.
    robot_shape   : str   — 'cylinder' (default) | 'box' (170 mm square, the
                            real footprint)
    push_yield, drive_force, hold_force : used only by locomotion='planar_force' (see the
                            block above the class)
    defender_fouls : bool — False (default). True: a defender's centre over the
                            midline, a side line or the end line ends the raid
                            as 'foul', the raider's point (block above the class)
    raider_edge_escape : bool — False (default). True: the robots' 0.30 m
                            boundary escape overrides the raider's command
                            near a line (block above the class)
    held_needs_touch : bool — False (default). True: the static-raider
                            Tackle needs a defender touching the raider
                            (Outcome rules block)
    """

    metadata = {"render_modes": ["human", "rgb_array"]}

    # ── Reward weights ────────────────────────────────────────────────
    R_TOUCH    =  10.0
    R_TACKLE   =  -5.0
    R_EMPTY    =  -2.0
    R_MIDLINE  =   1.0
    R_BAULK    =   1.5
    R_BONUS    =   2.0
    R_PROGRESS =   0.3
    R_TIME     =  -0.05
    R_BOUNDARY =  -1.0

    # ── Physics ───────────────────────────────────────────────────────
    MAX_LIN_VEL = 0.30   # m/s
    MAX_ANG_VEL = 1.50   # rad/s
    FSR_THRESH  = 0.30   # Newtons  (~30 g)
    DEF_MAX_LIN_VEL = MAX_LIN_VEL   # 'physical' defenders: same robot, same speed

    def __init__(
        self,
        n_defenders   = 1,
        defender_type = "reactive",
        render        = False,
        dt            = 0.05,
        raid_time     = 30.0,
        locomotion    = None,
        diagonal_speed_ratio = 1.0,
        diag_ratio_range = None,
        diag_ratio_period = None,
        start_layout  = "legacy",
        defender_start_depth = None,
        outcome_rules = "legacy",
        contact_model = "force",
        contact_dist  = None,
        contact_min_depth = None,
        contact_confirm_steps = 1,
        robot_shape   = "cylinder",
        static_time   = STATIC_TIME,
        hold_dist     = None,
        hold_touch_dist = None,
        gate_style    = "keepout",
        raider_command = "continuous",
        action_repeat = 1,
        raider_start  = "legacy",
        court_walls   = True,
        empty_on_return = False,
        return_margin = 0.0,
        push_yield    = PUSH_YIELD,
        drive_force   = DRIVE_FORCE,
        hold_force    = HOLD_FORCE,
        defender_fouls = False,
        raider_edge_escape = False,
        held_needs_touch = False,
    ):
        super().__init__()
        assert robot_shape in ROBOT_SHAPES, \
            f"robot_shape must be one of {ROBOT_SHAPES}, got {robot_shape!r}"
        self.robot_shape = robot_shape
        assert static_time > 0, "static_time must be positive"
        assert gate_style in GATE_STYLES, f"gate_style must be one of {GATE_STYLES}"
        self.gate_style = gate_style
        # defenders' margin to the side and rear walls (v11: up to the line)
        self._def_wall = GATE_LINE_MARGIN if gate_style == "v11" else PHYS_DEF_WALL
        assert raider_command in RAIDER_COMMANDS, \
            f"raider_command must be one of {RAIDER_COMMANDS}"
        assert int(action_repeat) == action_repeat >= 1
        assert raider_start in RAIDER_STARTS, \
            f"raider_start must be one of {RAIDER_STARTS}"
        self.raider_command = raider_command
        self.action_repeat = int(action_repeat)
        self.raider_start = raider_start
        self.court_walls = bool(court_walls)
        self.empty_on_return = bool(empty_on_return)
        assert return_margin >= 0, "return_margin is metres behind the midline (>= 0)"
        self.return_margin = float(return_margin)
        self.static_time = static_time
        assert hold_dist is None or hold_dist > 0, "hold_dist must be positive"
        assert hold_touch_dist is None or hold_touch_dist > 0, "hold_touch_dist must be positive"
        self.hold_dist = STATIC_DIST if hold_dist is None else float(hold_dist)
        self.hold_touch_dist = None if hold_touch_dist is None else float(hold_touch_dist)

        assert 1 <= n_defenders <= 7, "n_defenders must be between 1 and 7"
        assert defender_type in ("static", "reactive", "wall", "physical", "gate",
                                 "learned")
        self.defender_fouls = bool(defender_fouls)
        self.raider_edge_escape = bool(raider_edge_escape)
        self.held_needs_touch = bool(held_needs_touch)
        assert diagonal_speed_ratio > 0, "diagonal_speed_ratio must be positive"
        assert start_layout in ("legacy", "back_row")
        if defender_start_depth is not None:
            lo, hi = (float(v) for v in defender_start_depth)
            assert start_layout == "back_row", "defender_start_depth needs start_layout='back_row'"
            assert 0 < lo <= hi <= COURT_L / 2 - R_RADIUS, \
                f"defender_start_depth is metres past the midline, 0-{COURT_L / 2 - R_RADIUS:.2f}"
            defender_start_depth = (lo, hi)
        self.defender_start_depth = defender_start_depth
        assert outcome_rules in OUTCOME_RULES, \
            f"outcome_rules must be one of {OUTCOME_RULES}, got {outcome_rules!r}"
        assert contact_model in CONTACT_MODELS, \
            f"contact_model must be one of {CONTACT_MODELS}, got {contact_model!r}"
        assert contact_model != "camera" or (contact_dist is not None
                                             and contact_dist > 0), \
            "contact_model='camera' needs a positive contact_dist"
        assert contact_min_depth is None or contact_min_depth >= 0
        assert int(contact_confirm_steps) == contact_confirm_steps >= 1

        if locomotion is None:
            locomotion = os.environ.get("ROBO_KABADDI_LOCOMOTION",
                                        DEFAULT_LOCOMOTION)
        assert locomotion in LOCOMOTION_MODES, \
            f"locomotion must be one of {LOCOMOTION_MODES}, got {locomotion!r}"

        self.n_def   = n_defenders
        self.def_type = defender_type
        self.start_layout = start_layout
        self.render_mode = "human" if render else None
        self.dt      = dt                 # control period
        self.raid_time = raid_time
        self.max_steps = int(raid_time / dt)
        self.outcome_rules = outcome_rules
        if outcome_rules in ("section_2_3", "physical"):
            # Integer step counts, so the timers do not depend on float sums.
            self._approach_steps = int(round(APPROACH_TIME / dt))
            self._raid_steps     = int(round(raid_time / dt))
            self._static_steps   = int(round(static_time / dt))
            self.max_steps = self._approach_steps + self._raid_steps
        self._raid_start_step = None      # step at which the raider crossed
        self._static_ref   = None         # static-raider window: reference xy
        self._static_since = None         # ...and the step it was set

        self.contact_model = contact_model
        self.contact_dist  = contact_dist
        self.contact_min_depth = contact_min_depth
        self.contact_confirm_steps = int(contact_confirm_steps)
        self._contact_count = 0           # qualifying contact steps so far

        self.locomotion = locomotion
        assert 0 < push_yield < 1 and 0 < drive_force < hold_force
        self.push_yield  = push_yield
        self.drive_force = drive_force
        self.hold_force  = hold_force
        self._drive_gain = drive_force / (push_yield * self.MAX_LIN_VEL)
        if locomotion in SUBSTEPPED_MODES:
            self.physics_dt = PHYSICS_DT
            self.n_substeps = max(1, int(round(dt / PHYSICS_DT)))
        else:
            self.physics_dt = dt          # one physics step per control step
            self.n_substeps = 1
        self._last_fsr = None             # max FSR over the last control step

        # 'physical' defenders: pursuit runs every _def_period control steps
        # and each defender holds its last command in between.
        self.diagonal_speed_ratio = diagonal_speed_ratio
        self.diag_ratio_range = (None if diag_ratio_range is None
                                 else (float(diag_ratio_range[0]), float(diag_ratio_range[1])))
        if self.diag_ratio_range is not None:
            assert 0 < self.diag_ratio_range[0] <= self.diag_ratio_range[1], diag_ratio_range
        self._diag_period_steps = (None if diag_ratio_range is None or not diag_ratio_period
                                   else max(1, int(round(diag_ratio_period / dt))))
        self._diag_rng = np.random.default_rng(0)
        self._diag = None             # per-robot ratios [raider, defender 1, ...]
        self._cur_robot = None        # robot whose command is being worked out
        self._def_period = max(1, int(round(1.0 / (PHYS_DEF_HZ * dt))))
        self._def_cmd    = [(0.0, 0.0)] * n_defenders
        self._gate_role, self._gate_closed = None, False
        self._gate_back = [False] * n_defenders   # v7: going back
        self._gate_in = False                     # v8: raider is in, closing
        self._gate_blockers = []                  # v10: defenders closing the way
        self._def_act = [0] * n_defenders         # learned: DEF_ACTIONS indices

        # Connect PyBullet
        if render:
            self._client = p.connect(p.GUI)
            p.resetDebugVisualizerCamera(
                cameraDistance    = 5.0,
                cameraYaw         = 0,
                cameraPitch       = -60,
                cameraTargetPosition = [0, 0, 0]
            )
        else:
            self._client = p.connect(p.DIRECT)

        p.setAdditionalSearchPath(pybullet_data.getDataPath(),
                                  physicsClientId=self._client)

        # ── Spaces ────────────────────────────────────────────────────
        # obs: pose(3) + vel(3) + def_rel(2*N) + fsr(2) + time(1) + flags(2)
        obs_dim = 11 + 2 * n_defenders
        self.observation_space = spaces.Box(
            low=-np.inf, high=np.inf, shape=(obs_dim,), dtype=np.float32
        )
        # action: [vx, vy, omega] normalised to [-1, 1]
        self.action_space = spaces.Box(
            low=-1.0, high=1.0, shape=(3,), dtype=np.float32
        )

        # Body IDs (filled in reset)
        self.raider_id    = None
        self.defender_ids = []
        self._plane_id    = None

        # Episode state
        self._step       = 0
        self._t          = 0.0
        self._prev_pos   = None
        self._flags      = {}   # midline, baulk, bonus, contact, rewards_given

    # ──────────────────────────────────────────────────────────────────
    # Reset
    # ──────────────────────────────────────────────────────────────────

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)

        p.resetSimulation(physicsClientId=self._client)
        p.setGravity(0, 0, -9.81, physicsClientId=self._client)
        p.setTimeStep(self.physics_dt, physicsClientId=self._client)
        self._last_fsr = None

        self._build_court()

        # Episode counters
        self._step = 0
        self._t    = 0.0
        self._raid_start_step = None
        self._static_ref   = None
        self._static_since = None
        self._contact_count = 0
        self._flags = {
            "midline" : False,
            "baulk"   : False,
            "bonus"   : False,
            "contact" : False,
            "r_mid"   : False,   # reward already given?
            "r_baulk" : False,
            "r_bonus" : False,
        }

        # Spawn raider (raider half: y in [-1.8, -0.3])
        if self.raider_start == "near_midline":
            rx = self.np_random.uniform(*NEAR_MIDLINE_X)
            ry = MIDLINE_Y + self.np_random.uniform(*NEAR_MIDLINE_Y)
        else:
            rx = self.np_random.uniform(-0.9, 0.9)
            ry = self.np_random.uniform(-1.8, -0.3)
        self.raider_id = self._spawn_robot(
            [rx, ry, R_HEIGHT / 2], color=[1, 1, 1, 1]
        )
        self._prev_pos = np.array([rx, ry], dtype=np.float32)

        # Spawn defenders (defender half: y in [0.3, 1.8])
        self.defender_ids = []
        starts = (self._defender_starts() if self.start_layout == "legacy"
                  else self._defender_starts_back_row())
        for pos in starts:
            did = self._spawn_robot(
                [pos[0], pos[1], R_HEIGHT / 2], color=[1, 0.1, 0.1, 1]
            )
            self.defender_ids.append(did)
        self._def_cmd = [(0.0, 0.0)] * self.n_def
        self._gate_role, self._gate_closed = None, False
        self._gate_back = [False] * self.n_def
        self._gate_in = False
        self._gate_blockers = []
        self._def_act = [0] * self.n_def
        if self.diag_ratio_range is not None:
            if seed is not None:
                self._diag_rng = np.random.default_rng([int(seed), 9973])
            self._draw_diag()

        # Settle physics. Both models settle for the same simulated time:
        # legacy 20 x 0.05 s, planar 20 x n_substeps x (1/240) s = 1 s.
        for _ in range(20 * self.n_substeps):
            p.stepSimulation(physicsClientId=self._client)
            if self.locomotion in SUBSTEPPED_MODES:
                for bid in [self.raider_id] + self.defender_ids:
                    self._enforce_planar(bid)

        return self._obs(), {}

    # ──────────────────────────────────────────────────────────────────
    # Step
    # ──────────────────────────────────────────────────────────────────

    def step(self, action):
        if self.action_repeat == 1:
            obs, total, term, trunc, info = self._step_once(action)
        else:
            total = 0.0
            for _ in range(self.action_repeat):
                obs, r, term, trunc, info = self._step_once(action)
                total += r
                if term or trunc:
                    break
        if self.def_type == "learned":
            info["team_reward"] = self._team_reward(info, term or trunc)
        return obs, total, term, trunc, info

    def _team_reward(self, info, done):
        """(raider, defenders) team rewards for this env.step (block above
        the class): zero until the raid ends; a raid cut off by max_steps
        without a result counts as Empty."""
        if not done:
            return 0.0, 0.0
        r, d = TEAM_REWARD[info.get("outcome", "empty")]
        n = info.get("fouls", 1)
        return r * n, d * n

    def set_defender_actions(self, actions):
        """defender_type='learned': one DEF_ACTIONS index per defender, used
        from the next step on and held until changed."""
        acts = [int(a) for a in actions]
        assert len(acts) == self.n_def, f"need {self.n_def} defender actions"
        assert all(0 <= a < len(DEF_ACTIONS) for a in acts), acts
        self._def_act = acts

    def _raider_velocity(self, action):
        """Raider (vx, vy, omega) for an action (block above the class)."""
        if self.raider_command == "hardware":
            sx, sy, frac = hardware_command(float(action[0]), float(action[1]))
            v = frac * self.MAX_LIN_VEL
            if sx and sy:
                v *= self._ratio(0) / np.sqrt(2.0)
            return v * sx, v * sy, 0.0
        return (float(action[0]) * self.MAX_LIN_VEL,
                float(action[1]) * self.MAX_LIN_VEL,
                float(action[2]) * self.MAX_ANG_VEL)

    def _raider_escape(self, pos):
        """raider_edge_escape: the (vx, vy, omega) away from a line within
        RAIDER_EDGE, at the 100% level, or None when clear."""
        x, y = pos[0], pos[1]
        left, right = x < -COURT_W / 2 + RAIDER_EDGE, x > COURT_W / 2 - RAIDER_EDGE
        home, far = y < -COURT_L / 2 + RAIDER_EDGE, y > COURT_L / 2 - RAIDER_EDGE
        if not (left or right or home or far):
            return None
        sx = 1 if left else -1 if right else 0
        sy = 1 if home else -1 if far else 0
        v = self.MAX_LIN_VEL
        if sx and sy:
            v *= self._ratio(0) / np.sqrt(2.0)
        return v * sx, v * sy, 0.0

    def _step_once(self, action):
        self._step += 1
        self._t    += self.dt
        if self._diag_period_steps and self._step % self._diag_period_steps == 0:
            self._draw_diag()

        # Raider command for this control step
        vx, vy, omega = self._raider_velocity(action)
        if self.raider_edge_escape:
            esc = self._raider_escape(self._get_pos(self.raider_id))
            if esc is not None:
                vx, vy, omega = esc

        if self.locomotion == "legacy":
            # ── Legacy model: unchanged order of operations ──────────────
            p.resetBaseVelocity(
                self.raider_id,
                linearVelocity=[vx, vy, 0],
                angularVelocity=[0, 0, omega],
                physicsClientId=self._client,
            )

            # Apply defender velocities (rule-based)
            self._step_defenders()

            # Physics step
            p.stepSimulation(physicsClientId=self._client)

            pos    = self._get_pos(self.raider_id)
            fsr_l, fsr_r = self._fsr()
        else:
            # ── Planar model: velocity held across sub-steps ─────────────
            # Defender commands are computed once per control step, from
            # positions at its start, as their controller would be.
            def_cmds = self._defender_commands()
            fsr_l = fsr_r = 0.0
            for _ in range(self.n_substeps):
                if self.locomotion == "planar_force":
                    self._drive(self.raider_id, vx, vy, omega)
                    for did, (dvx, dvy) in zip(self.defender_ids, def_cmds):
                        self._drive(did, dvx, dvy, 0.0)
                    p.stepSimulation(physicsClientId=self._client)
                    for bid in [self.raider_id] + self.defender_ids:
                        self._enforce_planar(bid)
                    sl, sr = self._fsr()
                    fsr_l = max(fsr_l, sl)
                    fsr_r = max(fsr_r, sr)
                    continue
                p.resetBaseVelocity(
                    self.raider_id,
                    linearVelocity=[vx, vy, 0],
                    angularVelocity=[0, 0, omega],
                    physicsClientId=self._client,
                )
                for did, (dvx, dvy) in zip(self.defender_ids, def_cmds):
                    p.resetBaseVelocity(
                        did,
                        linearVelocity=[dvx, dvy, 0],
                        angularVelocity=[0, 0, 0],
                        physicsClientId=self._client,
                    )
                p.stepSimulation(physicsClientId=self._client)
                for bid in [self.raider_id] + self.defender_ids:
                    self._enforce_planar(bid)
                # Sample contact at every sub-step and keep the peak, so a
                # brief contact inside the control step is not missed.
                sl, sr = self._fsr()
                fsr_l = max(fsr_l, sl)
                fsr_r = max(fsr_r, sr)
            self._last_fsr = (fsr_l, fsr_r)
            pos = self._get_pos(self.raider_id)

        contact = self._contact_event(pos, fsr_l, fsr_r)

        # Update flags
        if pos[1] > MIDLINE_Y:
            self._flags["midline"] = True
        if pos[1] > BAULK_Y:
            self._flags["baulk"] = True
        if pos[1] > BONUS_Y:
            self._flags["bonus"] = True
        if contact:
            self._flags["contact"] = True

        # Compute reward & termination
        reward, terminated, info = self._reward(pos, contact)

        # Back across the midline without a counted contact (Section 2.3):
        # Empty at once, instead of waiting for the timer.
        if (self.empty_on_return and not terminated and self._flags["midline"]
                and not self._flags["contact"] and pos[1] < MIDLINE_Y - self.return_margin):
            reward += self.R_EMPTY
            info["outcome"] = "empty"
            terminated = True

        # A defender over the midline or a court line (block above the class)
        if self.defender_fouls and not terminated:
            over = self._defender_fouls()
            if over:
                reward += self.R_TOUCH * len(over)
                info["outcome"], info["fouls"] = "foul", len(over)
                terminated = True

        # Timeout
        if self.outcome_rules == "legacy":
            if not terminated and self._t >= self.raid_time:
                if self._flags["contact"]:
                    reward += self.R_TACKLE
                    info["outcome"] = "tackle"
                else:
                    reward += self.R_EMPTY
                    info["outcome"] = "empty"
                terminated = True
        elif not terminated:
            outcome = self._timeout_section_2_3(pos)
            if outcome is None and self.outcome_rules == "physical":
                outcome = self._out_of_court(pos) or self._static_tackle(pos)
            if outcome is not None:
                reward += self.R_TACKLE if outcome == "tackle" else self.R_EMPTY
                info["outcome"] = outcome
                terminated = True

        truncated = self._step >= self.max_steps
        self._prev_pos = pos[:2].copy()

        return self._obs(), reward, terminated, truncated, info

    def _timeout_section_2_3(self, pos):
        """'tackle' or 'empty' if a Section 2.3 timer has run out this step,
        else None. The raid timer starts at the step the raider first crosses
        the midline; until then the approach timer runs."""
        if self._raid_start_step is None and self._flags["midline"]:
            self._raid_start_step = self._step
        if self._raid_start_step is None:
            return "empty" if self._step >= self._approach_steps else None
        if self._step - self._raid_start_step >= self._raid_steps:
            return "tackle" if pos[1] > MIDLINE_Y else "empty"
        return None

    def _defender_fouls(self):
        """Indices of the defenders whose centre is over the midline, a side
        line or the end line (block above the class)."""
        over = []
        for i, did in enumerate(self.defender_ids):
            x, y = self._get_pos(did)[:2]
            if y < MIDLINE_Y or abs(x) > COURT_W / 2 or y > COURT_L / 2:
                over.append(i)
        return over

    def _out_of_court(self, pos):
        """Raider's centre outside the court lines (the author's rule, A5;
        the robots judged by the marker centre, checks 3a/3b): 'tackle' once
        the raid has started (the raider has crossed the midline), 'empty'
        before it (no raid). Only possible with court_walls=False."""
        if abs(pos[0]) > COURT_W / 2 or abs(pos[1]) > COURT_L / 2:
            return "tackle" if self._raid_start_step is not None else "empty"
        return None

    def _static_tackle(self, pos):
        """'tackle' if the raider has stayed within STATIC_DIST in the
        defenders' half for more than static_time, else None. Mirrors check
        3c of run_experiment.py, sampled once per control step. With
        held_needs_touch, only while a defender touches the raider."""
        if (self._raid_start_step is None or pos[1] <= MIDLINE_Y
                or (self.held_needs_touch and not self._defender_holding())):
            self._static_ref = None
            return None
        xy = np.array(pos[:2], dtype=np.float64)
        if (self._static_ref is None
                or np.hypot(*(xy - self._static_ref)) > self.hold_dist):
            self._static_ref, self._static_since = xy, self._step
            return None
        if self._step - self._static_since > self._static_steps:
            return "tackle"
        return None

    def _defender_holding(self):
        """Touching for the hold rule: body contact (default) or, with
        hold_touch_dist, a defender centre within that distance (author, 14 Nov)."""
        if self.hold_touch_dist is None:
            return self._defender_touching()
        r = self._get_pos(self.raider_id)[:2]
        return any(np.hypot(*(self._get_pos(d)[:2] - r)) <= self.hold_touch_dist
                   for d in self.defender_ids)

    def _defender_touching(self):
        """Whether any defender's body is in contact with the raider."""
        return any(p.getContactPoints(self.raider_id, did, physicsClientId=self._client)
                   for did in self.defender_ids)

    def _contact_event(self, pos, fsr_l, fsr_r):
        """Whether contact counts for the game this step (block above the
        class): the raw contact of the chosen model, then the depth and
        confirmation rules. With the defaults this is the raw force test."""
        if self.contact_model == "camera":
            raw = self._camera_contact(pos)
        else:
            raw = fsr_l > self.FSR_THRESH or fsr_r > self.FSR_THRESH
        if not raw:
            return False
        if (self.contact_min_depth is not None
                and pos[1] <= MIDLINE_Y + self.contact_min_depth):
            return False
        self._contact_count += 1
        return self._contact_count >= self.contact_confirm_steps

    def _camera_contact(self, r_pos):
        """Any defender centre within contact_dist of the raider's centre."""
        for did in self.defender_ids:
            d = self._get_pos(did)
            if np.hypot(d[0] - r_pos[0], d[1] - r_pos[1]) <= self.contact_dist:
                return True
        return False

    # ──────────────────────────────────────────────────────────────────
    # Reward
    # ──────────────────────────────────────────────────────────────────

    def _reward(self, pos, contact):
        r          = 0.0
        terminated = False
        info       = {}

        # One-time milestone rewards
        if self._flags["midline"] and not self._flags["r_mid"]:
            r += self.R_MIDLINE
            self._flags["r_mid"] = True

        if self._flags["baulk"] and not self._flags["r_baulk"]:
            r += self.R_BAULK
            self._flags["r_baulk"] = True

        if self._flags["bonus"] and not self._flags["r_bonus"]:
            r += self.R_BONUS
            self._flags["r_bonus"] = True

        # Progress: reward closing distance to nearest defender
        #           but only before first contact
        if not self._flags["contact"]:
            if self._flags["midline"]:
                cur_d  = self._nearest_def_dist(pos[:2])
                prev_d = self._nearest_def_dist(self._prev_pos)
                r += self.R_PROGRESS * (prev_d - cur_d)
        else:
            # reward moving toward midline
            r += 0.2 * (pos[1] - self._prev_pos[1])

        # NEW: after contact, reward moving back toward home
        # after contact, reward moving back toward home
        if self._flags["contact"] and self._flags["midline"]:
            dy = self._prev_pos[1] - pos[1]
            r += 5.0 * dy    # increased from 1.0

        # One-time bonus for crossing back over midline after contact
        if (self._flags["contact"] and self._flags["midline"]
                and pos[1] < MIDLINE_Y and self._prev_pos[1] >= 0):
            r += 5.0

        # Terminal: touch + return
        if (self._flags["contact"]
                and self._flags["midline"]
                and pos[1] < MIDLINE_Y):
            r += self.R_TOUCH
            info["outcome"] = "touch"
            terminated = True

        # Per-step time penalty
        r += self.R_TIME

        # Boundary penalty
        hw = COURT_W / 2 - 0.05
        hl = COURT_L / 2 - 0.05
        if abs(pos[0]) > hw or abs(pos[1]) > hl:
            r += self.R_BOUNDARY

        return r, terminated, info

    # ──────────────────────────────────────────────────────────────────
    # Observation
    # ──────────────────────────────────────────────────────────────────

    def _obs(self):
        r_pos = self._get_pos(self.raider_id)   # (x, y, theta)
        r_vel = self._get_vel(self.raider_id)   # (vx, vy, omega)
        # Under the planar model the step sampled contact at every sub-step;
        # report that peak so the observation agrees with the contact flag.
        if self.contact_model == "camera":
            # As the physical observation: current proximity, left channel.
            fsr_l, fsr_r = float(self._camera_contact(r_pos)), 0.0
        elif self.locomotion in SUBSTEPPED_MODES and self._last_fsr is not None:
            fsr_l, fsr_r = self._last_fsr
        else:
            fsr_l, fsr_r = self._fsr()

        def_rel = []
        for did in self.defender_ids:
            d_pos = self._get_pos(did)
            def_rel += [d_pos[0] - r_pos[0], d_pos[1] - r_pos[1]]

        t_norm = max(0.0, (self.raid_time - self._t) / self.raid_time)

        obs = np.concatenate([
            r_pos,                                           # 3
            r_vel,                                           # 3
            np.array(def_rel, dtype=np.float32),             # 2*N
            np.array([fsr_l, fsr_r], dtype=np.float32),      # 2
            np.array([t_norm], dtype=np.float32),             # 1
            np.array([
                float(self._flags["midline"]),
                float(self._flags["baulk"]),
            ], dtype=np.float32),                             # 2
        ]).astype(np.float32)

        return obs

    # Learned defenders' inputs (block above the class). Each row is
    # egocentric so one shared policy can drive any defender:
    #   own x, y, vx, vy (4) | raider relative x, y and its vx, vy (4)
    #   | teammates relative x, y, nearest first (2*(N-1))
    #   | distance to the midline, the left line, the right line, the end line (4)
    #   | contact counted, raid started, time left (3)
    # Time left: the raid timer once the raider has crossed, the approach
    # timer before (outcome_rules 'section_2_3' / 'physical').
    DEF_OBS_FIXED = 15

    def _time_left(self):
        if getattr(self, "_raid_steps", None) is None:
            return max(0.0, 1.0 - self._step / self.max_steps)
        if self._raid_start_step is None:
            return max(0.0, 1.0 - self._step / self._approach_steps)
        return max(0.0, 1.0 - (self._step - self._raid_start_step) / self._raid_steps)

    def defender_obs(self):
        """(N, 15 + 2*(N-1)) array, one egocentric row per defender."""
        r, rv = self._get_pos(self.raider_id), self._get_vel(self.raider_id)
        P = [self._get_pos(d) for d in self.defender_ids]
        V = [self._get_vel(d) for d in self.defender_ids]
        tail = [float(self._flags["contact"]), float(self._raid_start_step is not None),
                self._time_left()]
        rows = []
        for i in range(self.n_def):
            x, y = float(P[i][0]), float(P[i][1])
            mates = sorted(((P[j][0] - x, P[j][1] - y) for j in range(self.n_def)
                            if j != i), key=lambda q: np.hypot(*q))
            rows.append([x, y, V[i][0], V[i][1],
                         r[0] - x, r[1] - y, rv[0], rv[1],
                         *[c for q in mates for c in q],
                         y - MIDLINE_Y, COURT_W / 2 + x, COURT_W / 2 - x,
                         COURT_L / 2 - y, *tail])
        return np.array(rows, dtype=np.float32)

    def global_state(self):
        """Whole court for a training critic: raider x, y, vx, vy, each
        defender's x, y, vx, vy in index order, contact, raid started, time left."""
        s = []
        for bid in [self.raider_id] + self.defender_ids:
            q, v = self._get_pos(bid), self._get_vel(bid)
            s += [q[0], q[1], v[0], v[1]]
        s += [float(self._flags["contact"]), float(self._raid_start_step is not None),
              self._time_left()]
        return np.array(s, dtype=np.float32)

    # ──────────────────────────────────────────────────────────────────
    # FSR simulation
    # ──────────────────────────────────────────────────────────────────

    def _fsr(self):
        """
        Simulate dual FSR via PyBullet contact forces.
        Returns (left_force, right_force) in Newtons.
        """
        left = right = 0.0
        r_pos = self._get_pos(self.raider_id)
        yaw   = r_pos[2]

        # Front-face normal
        front  = np.array([np.cos(yaw), np.sin(yaw)])
        r_normal = np.array([front[1], -front[0]])  # right-hand normal

        for did in self.defender_ids:
            contacts = p.getContactPoints(
                self.raider_id, did, physicsClientId=self._client
            )
            if not contacts:
                continue
            for c in contacts:
                force = c[9]   # normal force (N)
                if force < 0.05:
                    continue
                cp = np.array(c[5][:2])               # contact point XY
                cv = cp - np.array(r_pos[:2])         # vector to contact
                if np.dot(cv, r_normal) >= 0:
                    right = max(right, force)
                else:
                    left  = max(left, force)

        return left, right

    # ──────────────────────────────────────────────────────────────────
    # Defender rule-based controllers
    # ──────────────────────────────────────────────────────────────────

    def _defender_command(self, i, did, r_pos):
        """Commanded (vx, vy) for one defender.

        Calls self._reactive / self._wall, so experiments that patch those
        methods at runtime (e.g. exp_G_defender_speed.py) still take effect.
        Sets the robot whose diagonal ratio _phys_velocity uses
        (diag_ratio_range).
        """
        self._cur_robot = i + 1
        try:
            return self._defender_command_of(i, did, r_pos)
        finally:
            self._cur_robot = None

    def _defender_command_of(self, i, did, r_pos):
        if self.def_type == "static":
            return 0.0, 0.0
        if self.def_type == "reactive":
            return self._reactive(did, r_pos)
        if self.def_type == "gate":
            return self._gate(i, did, r_pos)
        if self.def_type == "physical":
            return self._physical(i, did, r_pos)
        if self.def_type == "learned":
            return self._phys_velocity(*DEF_ACTIONS[self._def_act[i]])
        return self._wall(did, i, r_pos)

    def _defender_commands(self):
        """Planar path: every defender's command, computed once per control
        step and then held across the sub-steps."""
        r_pos = self._get_pos(self.raider_id)
        return [self._defender_command(i, did, r_pos)
                for i, did in enumerate(self.defender_ids)]

    def _step_defenders(self):
        """Legacy path: compute and apply each defender's command in turn,
        before the single physics step. The per-defender interleaving is the
        original order of operations and is kept so legacy reproduces the
        submitted results call for call."""
        r_pos = self._get_pos(self.raider_id)

        for i, did in enumerate(self.defender_ids):
            vx, vy = self._defender_command(i, did, r_pos)

            p.resetBaseVelocity(
                did,
                linearVelocity=[vx, vy, 0],
                angularVelocity=[0, 0, 0],
                physicsClientId=self._client,
            )

    def _reactive(self, did, r_pos):
        """Move toward raider when raider enters defender half."""
        if not self._flags["midline"]:
            return 0.0, 0.0
        d_pos = self._get_pos(did)
        dx = r_pos[0] - d_pos[0]
        dy = r_pos[1] - d_pos[1]
        dist = np.hypot(dx, dy) + 1e-6
        speed = min(0.20, dist * 0.8)
        return speed * dx / dist, speed * dy / dist

    def _wall(self, did, idx, r_pos):
        """
        Maintain an evenly-spaced horizontal wall.
        Compress toward raider depth when raider crosses midline.
        """
        xs = np.linspace(
            -COURT_W / 2 + 0.3,
             COURT_W / 2 - 0.3,
             self.n_def
        )
        tx = xs[idx]
        ty = (min(1.2, r_pos[1] - 0.30)
              if self._flags["midline"] else 0.80)

        d_pos = self._get_pos(did)
        dx, dy = tx - d_pos[0], ty - d_pos[1]
        dist = np.hypot(dx, dy) + 1e-6
        speed = min(0.15, dist)
        return speed * dx / dist, speed * dy / dist

    def _physical(self, i, did, r_pos):
        """The physical defenders' controller (block above the class).

        The boundary override is checked every control step; pursuit is
        recomputed every _def_period steps, and the last command is held in
        between, as the robots held the last command they were sent.
        """
        d_pos = self._get_pos(did)
        escape = self._phys_escape(d_pos)
        if escape is not None:
            self._def_cmd[i] = self._phys_velocity(*escape)
        elif (self._step - 1) % self._def_period == 0:
            self._def_cmd[i] = self._phys_velocity(*self._phys_pursuit(d_pos, r_pos))
        return self._def_cmd[i]

    def _phys_escape(self, d_pos):
        """Direction (sx, sy) away from the midline or a wall, or None.
        Mirrors _near_edge / _escape_cmd in run_experiment.py."""
        x, y = d_pos[0], d_pos[1]
        if y < MIDLINE_Y + PHYS_DEF_GUARD:
            return 0, 1
        left  = x < -COURT_W / 2 + self._def_wall
        right = x >  COURT_W / 2 - self._def_wall
        rear  = y >  COURT_L / 2 - self._def_wall
        if not (left or right or rear):
            return None
        return (1 if left else -1 if right else 0), (-1 if rear else 0)

    def _phys_pursuit(self, d_pos, r_pos):
        """Direction (sx, sy) from send_defender_action."""
        if r_pos[1] <= MIDLINE_Y:
            return 0, 0
        if self._flags["contact"]:
            return self._phys_toward(d_pos, r_pos)
        if d_pos[1] < MIDLINE_Y + PHYS_DEF_SOFT:
            return 0, 0
        sx, sy = self._phys_toward(d_pos, r_pos)
        if sy < 0 and d_pos[1] < MIDLINE_Y + PHYS_DEF_SOFT + PHYS_DEF_CREEP:
            return 0, 0
        return sx, sy

    @staticmethod
    def _phys_toward(d_pos, r_pos):
        """One of eight directions or stop, as defender_move_toward."""
        dx, dy = r_pos[0] - d_pos[0], r_pos[1] - d_pos[1]
        dist = np.hypot(dx, dy)
        if dist < PHYS_DEF_STOP_DIST:
            return 0, 0
        nx, ny = dx / dist, dy / dist
        sx = int(nx > PHYS_DEF_AXIS_THRESH) - int(nx < -PHYS_DEF_AXIS_THRESH)
        sy = int(ny > PHYS_DEF_AXIS_THRESH) - int(ny < -PHYS_DEF_AXIS_THRESH)
        return sx, sy

    def _phys_velocity(self, sx, sy):
        """Court-frame velocity for a direction at the 100% speed level."""
        v = self.DEF_MAX_LIN_VEL
        if sx and sy:
            v *= self._ratio(self._cur_robot) / np.sqrt(2.0)
        return v * sx, v * sy

    def _draw_diag(self):
        """diag_ratio_range: a new diagonal ratio for every robot."""
        lo, hi = self.diag_ratio_range
        self._diag = self._diag_rng.uniform(lo, hi, 1 + self.n_def)

    def _ratio(self, k):
        """Diagonal ratio of robot k (0 = raider, i + 1 = defender i); None or
        no diag_ratio_range: diagonal_speed_ratio."""
        if self._diag is None or k is None:
            return self.diagonal_speed_ratio
        return float(self._diag[k])

    def _gate(self, i, did, r_pos):
        """The gate defence (block above the class). Same command layer as
        _physical: boundary escape every step, a new target every
        _def_period steps, the last command held in between."""
        d_pos = self._get_pos(did)
        tick = (self._step - 1) % self._def_period == 0
        if tick and i == 0:
            self._gate_assign(r_pos)          # slots for every defender
        escape = self._phys_escape(d_pos)
        if escape is not None:
            self._def_cmd[i] = self._phys_velocity(*escape)
        elif tick:
            self._def_cmd[i] = self._phys_velocity(
                *self._gate_dir(self._gate_role[i], d_pos, r_pos, i))
        return self._def_cmd[i]

    def _gate_floor(self):
        """Nearest a slot may be to the midline: the rules' limit."""
        return MIDLINE_Y + (GATE_FLOOR_AFTER if self._flags["contact"]
                            else PHYS_DEF_SOFT + PHYS_DEF_CREEP)

    def _gate_assign(self, r_pos):
        """Slot offsets (from the raider's lane), by left-to-right order."""
        if r_pos[1] > MIDLINE_Y:
            self._gate_closed = True
        if self._flags.get("contact") or r_pos[1] >= MIDLINE_Y + GATE_IN_DEPTH:
            self._gate_in = True                      # v8: stays in all raid
        if self.gate_style in ("v10", "v11"):
            self._gate_blockers = []
            if self._gate_in and r_pos[1] > MIDLINE_Y:
                lanes = self._gate_lanes(r_pos)
                marg = [max((self._gate_margin(self._get_pos(d)[:2], ln)
                             for ln in lanes if ln), default=-np.inf)
                        for d in self.defender_ids]
                self._gate_blockers = sorted(
                    np.argsort(marg, kind="stable")[::-1][:GATE_CATCH_N].tolist())
        order = np.argsort([self._get_pos(d)[0] for d in self.defender_ids],
                           kind="stable")
        offsets = (np.arange(self.n_def) - (self.n_def - 1) / 2) * GATE_SLOT
        roles = [0.0] * self.n_def
        for rank, k in enumerate(order):
            roles[int(k)] = float(offsets[rank])
        self._gate_role = roles

    def _gate_side(self, off, d_pos, r_pos):
        """Side (-1 left, +1 right) on which a defender passes the raider."""
        side = np.sign(off) if off else (1.0 if d_pos[0] >= r_pos[0] else -1.0)
        if abs(r_pos[0] + side * GATE_CLEAR) > COURT_W / 2 - self._def_wall:
            side = -side                              # no room: other side
        return side

    def _gate_closing(self, off, d_pos, r_pos):
        """True once a defender starts its diagonal: level with or past the
        raider. (Starting earlier, while still above it, roughly halved the
        tackles against the hardware-like raider; FINDINGS_S.md, v4.)"""
        return d_pos[1] <= r_pos[1]

    def _gate_target(self, off, d_pos, r_pos):
        """Target point for a slot offset, before the rule limits."""
        slot_x = r_pos[0] + off
        slot_y = max(r_pos[1] - GATE_BEHIND, self._gate_floor())
        if not self._gate_closing(off, d_pos, r_pos):    # pass
            side = self._gate_side(off, d_pos, r_pos)
            out = side * (d_pos[0] - r_pos[0])
            if out < GATE_CLEAR - GATE_ARRIVE:
                return r_pos[0] + side * GATE_CLEAR, d_pos[1]   # step out first
            return r_pos[0] + side * min(out, GATE_FAR), slot_y  # then pass
        return slot_x, slot_y                         # close: diagonal

    def _gate_dir(self, off, d_pos, r_pos, i=None):
        """Direction (sx, sy) for a slot, within the physical rules."""
        if r_pos[1] <= MIDLINE_Y or not self._gate_closed:
            return 0, 0
        contact = self._flags["contact"]
        if i is not None and i in self._gate_blockers:
            return self._gate_catch_dir(i, d_pos, r_pos)   # v10
        if contact and d_pos[1] <= r_pos[1]:
            return self._phys_toward(d_pos, r_pos)    # press: catch and hold
        if self.gate_style in ("keepout", "v7", "v8", "v9", "v10", "v11"):
            return self._gate_keepout_dir(off, d_pos, r_pos, i)
        tx, ty = self._gate_target(off, d_pos, r_pos)
        ty = max(ty, self._gate_floor())
        tx = float(np.clip(tx, -COURT_W / 2 + self._def_wall,
                           COURT_W / 2 - self._def_wall))
        if self._gate_closing(off, d_pos, r_pos):
            # Forced diagonal: in and back together until one is done.
            sx = (int(np.sign(tx - d_pos[0]))
                  if abs(tx - d_pos[0]) > GATE_ARRIVE else 0)
            sy = -1 if d_pos[1] > ty + GATE_ARRIVE else 0
        else:
            sx, sy = self._phys_toward(d_pos, (tx, ty))
        if (not contact and sy < 0
                and d_pos[1] < MIDLINE_Y + PHYS_DEF_SOFT + PHYS_DEF_CREEP):
            sy = 0                                    # at the limit: sideways
        return sx, sy

    def _gate_keepout_dir(self, off, d_pos, r_pos, i=None):
        """gate_style='keepout' / 'v7' / 'v8' / 'v9': the safe move that gets
        closest to the slot (block above the class), within the rule limits."""
        contact = self._flags["contact"]
        v9 = self.gate_style in ("v9", "v10", "v11")  # v10, v11 keep every v9 rule
        v8 = self.gate_style == "v8" or v9          # v9 keeps every v8 rule
        v7 = self.gate_style == "v7" or v8          # v8 keeps every v7 rule
        lo = MIDLINE_Y + (GATE_FLOOR_AFTER if contact
                          else GATE_V8_LIMIT if v8
                          else PHYS_DEF_SOFT + PHYS_DEF_CREEP)
        wall = lambda x: float(np.clip(x, -COURT_W / 2 + self._def_wall,
                                       COURT_W / 2 - self._def_wall))
        slot = np.array([
            float(np.clip(r_pos[0] + off, -COURT_W / 2 + self._def_wall,
                          COURT_W / 2 - self._def_wall)),
            max(r_pos[1] - GATE_BEHIND_KO, lo)])
        if v9 and not contact and (off == 0 or not self._gate_in):
            slot = (np.array([wall(r_pos[0]), MIDLINE_Y + GATE_HOLD_Y])
                    if off == 0 else                  # centre: back row
                    np.array([wall(r_pos[0] + np.sign(off) * GATE_SIDE_X),
                              MIDLINE_Y + GATE_SIDE_Y]))  # side: wide, 1 m
        elif v8 and not contact and not self._gate_in:  # holding: back row
            slot = np.array([
                float(np.clip(r_pos[0] + off / GATE_SLOT * GATE_HOLD_SPREAD,
                              -COURT_W / 2 + self._def_wall,
                              COURT_W / 2 - self._def_wall)),
                MIDLINE_Y + GATE_HOLD_Y])
        d = np.array(d_pos[:2], dtype=float)
        if v7 and i is not None:
            gap_now = float(np.hypot(d[0] - r_pos[0], d[1] - r_pos[1]))
            if (not contact and d[1] < lo + GATE_BACK_BAND
                    and gap_now < GATE_KEEPOUT + GATE_BACK_WARN):
                self._gate_back[i] = True             # too close: go back
            back_y = min(r_pos[1] + GATE_BEHIND_KO, COURT_L / 2 - self._def_wall)
            if contact or d[1] >= back_y - GATE_ARRIVE:
                self._gate_back[i] = False            # reached the back point
            if self._gate_back[i]:
                own = 1.0 if d[0] >= r_pos[0] else -1.0
                slot = np.array([
                    float(np.clip(r_pos[0] + own * GATE_CLEAR,
                                  -COURT_W / 2 + self._def_wall,
                                  COURT_W / 2 - self._def_wall)),
                    back_y])
        if not v7 and np.hypot(*(d - slot)) < GATE_ARRIVE:
            return 0, 0
        r_next = (np.array(r_pos[:2], dtype=float)
                  + np.array(self._get_vel(self.raider_id)[:2]) * GATE_LOOKAHEAD)
        side = self._gate_side(off, d_pos, r_pos)
        safe, unsafe = [], []
        for sx in (-1, 0, 1):
            for sy in (-1, 0, 1):
                if v7 and sx == 0 and sy == 0:
                    continue                          # never freeze
                if not contact and sy < 0 and d[1] < lo:
                    continue                          # the 0.40 m line
                q = d + np.array(self._phys_velocity(sx, sy)) * GATE_LOOKAHEAD
                if abs(q[0]) > COURT_W / 2 - self._def_wall and sx * q[0] > 0:
                    continue                          # into a wall margin
                gap = float(np.hypot(*(q - r_next)))
                cost = float(np.hypot(*(q - slot)))
                pref = 0 if sx == side else 1         # tie-break: own side
                (safe if gap >= GATE_KEEPOUT else unsafe).append(
                    (cost, pref, -gap, sx, sy))
        if safe:
            return min(safe)[3:]
        if unsafe and v7:                             # furthest, then back
            return max(unsafe, key=lambda c: (round(-c[2], 2), c[4]))[3:]
        if unsafe:
            return max(unsafe, key=lambda c: -c[2])[3:]   # furthest away
        return 0, 0

    def _gate_lanes(self, r_pos):
        """v10: the raider's three fastest ways home (straight, 45 degrees to
        either side; straight down once at a side line), as points
        (x, y, raider's path length) every GATE_CATCH_STEP m, limited to
        where a blocker may stand now (block above the class)."""
        contact = self._flags["contact"]
        floor = MIDLINE_Y + (GATE_FLOOR_AFTER if contact else GATE_V8_LIMIT)
        if not contact:
            floor = max(floor, r_pos[1] - GATE_BEHIND_KO)
        lanes = []
        for lean in (-1.0, 0.0, 1.0):
            x, y, s, pts = float(r_pos[0]), float(r_pos[1]), 0.0, []
            while True:
                sx = lean if abs(x + lean * GATE_CATCH_STEP) <= GATE_RAIDER_X else 0.0
                n = np.hypot(sx, 1.0)
                x, y = x + sx / n * GATE_CATCH_STEP, y - GATE_CATCH_STEP / n
                s += GATE_CATCH_STEP
                if y < floor:
                    break
                pts.append((x, y, s))
            lanes.append(pts)
        return lanes

    def _gate_margin(self, q, lane):
        """v10: catch margin of a defender at q on one way home (m, equal speed):
        the most, over its points, of raider's distance - defender's distance.
        Defenders cannot stand in the wall margins, so points are clipped."""
        best = -np.inf
        lim = COURT_W / 2 - self._def_wall
        for x, y, s in lane:
            best = max(best, s - float(np.hypot(q[0] - np.clip(x, -lim, lim), q[1] - y)))
        return best

    def _gate_catch_dir(self, i, d_pos, r_pos):
        """v10: a blocker's move (block above the class)."""
        contact = self._flags["contact"]
        d = np.array(d_pos[:2], dtype=float)
        r = np.array(r_pos[:2], dtype=float)
        gap = float(np.hypot(*(d - r)))
        if contact and d[1] < r[1] and gap < GATE_CATCH_PRESS:
            return self._phys_toward(d_pos, r_pos)    # catch and hold
        lo = MIDLINE_Y + GATE_V8_LIMIT
        if not contact and (self._gate_back[i] or (
                d[1] < lo + GATE_BACK_BAND and gap < GATE_KEEPOUT + GATE_BACK_WARN)):
            return self._gate_keepout_dir(self._gate_role[i], d_pos, r_pos, i)  # go back
        lanes = [ln for ln in self._gate_lanes(r_pos) if ln]
        if not lanes:
            return self._gate_keepout_dir(self._gate_role[i], d_pos, r_pos, i)
        others = [np.array(self._get_pos(self.defender_ids[k])[:2], dtype=float)
                  for k in self._gate_blockers if k != i]
        r_next = r + np.array(self._get_vel(self.raider_id)[:2]) * GATE_LOOKAHEAD
        best_key, best_dir = None, None
        for sx in (-1, 0, 1):
            for sy in (-1, 0, 1):
                if sx == 0 and sy == 0:
                    continue                          # never freeze
                if not contact and sy < 0 and d[1] < lo:
                    continue                          # the 0.60 m line
                q = d + np.array(self._phys_velocity(sx, sy)) * GATE_LOOKAHEAD
                if abs(q[0]) > COURT_W / 2 - self._def_wall and sx * q[0] > 0:
                    continue                          # into a side margin
                if q[1] > COURT_L / 2 - self._def_wall and sy > 0:
                    continue                          # into the rear margin
                safe = contact or float(np.hypot(*(q - r_next))) >= GATE_KEEPOUT
                cover = [max(self._gate_margin(b, ln) for b in [q] + others)
                         for ln in lanes]
                key = (safe, round(min(cover), 4), round(sum(cover), 4))
                if best_key is None or key > best_key:
                    best_key, best_dir = key, (sx, sy)
        if best_key is None or not best_key[0]:       # no safe move: as v7
            return self._gate_keepout_dir(self._gate_role[i], d_pos, r_pos, i)
        return best_dir

    # ──────────────────────────────────────────────────────────────────
    # PyBullet helpers
    # ──────────────────────────────────────────────────────────────────

    def _build_court(self):
        """Create floor + boundary walls (or, with court_walls=False, only
        painted boundary lines, as on the physical court)."""
        p.loadURDF("plane.urdf", physicsClientId=self._client)
        if not self.court_walls:
            hw, hl = COURT_W / 2, COURT_L / 2
            for pos, ext in (([0, hl, 0.002], [hw, 0.01, 0.002]),
                             ([0, -hl, 0.002], [hw, 0.01, 0.002]),
                             ([hw, 0, 0.002], [0.01, hl, 0.002]),
                             ([-hw, 0, 0.002], [0.01, hl, 0.002])):
                vid = p.createVisualShape(
                    p.GEOM_BOX, halfExtents=ext, rgbaColor=[0.9, 0.9, 0.9, 1],
                    physicsClientId=self._client)
                p.createMultiBody(baseMass=0, baseVisualShapeIndex=vid,
                                  basePosition=pos, physicsClientId=self._client)
            self._add_line(y=MIDLINE_Y, color=[0.2, 0.6, 1.0, 0.8])
            self._add_line(y=BAULK_Y,   color=[1.0, 0.8, 0.0, 0.8])
            self._add_line(y=BONUS_Y,   color=[0.2, 1.0, 0.2, 0.8])
            return

        wt = 0.02   # wall thickness
        wh = 0.15   # wall height
        hw = COURT_W / 2
        hl = COURT_L / 2

        walls = [
            # pos                  half-extents
            ([0,  hl, wh/2],   [hw, wt/2, wh/2]),   # far
            ([0, -hl, wh/2],   [hw, wt/2, wh/2]),   # near
            ([ hw, 0, wh/2],   [wt/2, hl, wh/2]),   # right
            ([-hw, 0, wh/2],   [wt/2, hl, wh/2]),   # left
        ]
        for pos, ext in walls:
            cid = p.createCollisionShape(
                p.GEOM_BOX, halfExtents=ext, physicsClientId=self._client
            )
            vid = p.createVisualShape(
                p.GEOM_CYLINDER, radius=R_RADIUS, length=R_HEIGHT,
                rgbaColor=[0.6, 0.6, 0.6, 1], physicsClientId=self._client,
            )
            p.createMultiBody(
                baseMass=0,
                baseCollisionShapeIndex=cid,
                baseVisualShapeIndex=vid,
                basePosition=pos,
                physicsClientId=self._client,
            )

        # Midline visual (thin flat box)
        self._add_line(y=MIDLINE_Y, color=[0.2, 0.6, 1.0, 0.8])
        self._add_line(y=BAULK_Y,   color=[1.0, 0.8, 0.0, 0.8])
        self._add_line(y=BONUS_Y,   color=[0.2, 1.0, 0.2, 0.8])

    def _add_line(self, y, color):
        ext = [COURT_W / 2, 0.01, 0.002]
        vid = p.createVisualShape(
            p.GEOM_BOX, halfExtents=ext,
            rgbaColor=color, physicsClientId=self._client
        )
        p.createMultiBody(
            baseMass=0, baseVisualShapeIndex=vid,
            basePosition=[0, y, 0.002],
            physicsClientId=self._client,
        )

    def _spawn_robot(self, position, color):
        if self.robot_shape == "box":
            ext = [R_RADIUS, R_RADIUS, R_HEIGHT / 2]
            cid = p.createCollisionShape(
                p.GEOM_BOX, halfExtents=ext, physicsClientId=self._client)
            vid = p.createVisualShape(
                p.GEOM_BOX, halfExtents=ext, rgbaColor=color,
                physicsClientId=self._client)
        else:
            cid = p.createCollisionShape(
                p.GEOM_CYLINDER, radius=R_RADIUS, height=R_HEIGHT,
            physicsClientId=self._client,
            )
            vid = p.createVisualShape(
                p.GEOM_CYLINDER, radius=R_RADIUS, length=R_HEIGHT,
                rgbaColor=color, physicsClientId=self._client,
            )
        planar = self.locomotion in SUBSTEPPED_MODES
        bid = p.createMultiBody(
            baseMass=ROBOT_MASS_PLANAR if planar else ROBOT_MASS_LEGACY,
            baseCollisionShapeIndex=cid,
            baseVisualShapeIndex=vid,
            basePosition=position,
            physicsClientId=self._client,
        )
        if planar:
            # Velocity servo on a frictionless base: the commanded velocity is
            # re-applied every sub-step, so no damping or floor friction is
            # needed, and neither may oppose the commanded motion.
            p.changeDynamics(
                bid, -1,
                lateralFriction=0.0,
                rollingFriction=0.0,
                spinningFriction=0.0,
                linearDamping=0.0,
                angularDamping=0.0,
                physicsClientId=self._client,
            )
        else:
            p.changeDynamics(
                bid, -1,
                lateralFriction=0.8,
                linearDamping=0.9,
                angularDamping=0.9,
                physicsClientId=self._client,
            )
        return bid

    def _drive(self, bid, vx, vy, omega):
        """planar_force: push the robot toward (vx, vy) with a force-limited
        drive for the next sub-step (block above the class): capped at
        drive_force when driving, hold_force when resisting motion. Turning
        is set directly."""
        lin, _ = p.getBaseVelocity(bid, physicsClientId=self._client)
        fx = self._drive_gain * (vx - lin[0])
        fy = self._drive_gain * (vy - lin[1])
        cap = (self.hold_force if fx * lin[0] + fy * lin[1] < 0
               else self.drive_force)
        f = np.hypot(fx, fy)
        if f > cap:
            fx, fy = fx * cap / f, fy * cap / f
        pos, _ = p.getBasePositionAndOrientation(bid, physicsClientId=self._client)
        p.applyExternalForce(bid, -1, [fx, fy, 0.0], pos, p.WORLD_FRAME,
                             physicsClientId=self._client)
        p.resetBaseVelocity(bid, angularVelocity=[0.0, 0.0, omega],
                            physicsClientId=self._client)

    def _enforce_planar(self, bid):
        """Hold a robot upright on the floor plane.

        Keeps x, y, z and yaw from the simulation; forces roll and pitch to
        zero and removes vertical and tipping velocity. The real platform
        does not tip during play, so this removes a failure mode the robot
        does not have, without constraining any motion it does have.
        """
        pos, orn = p.getBasePositionAndOrientation(
            bid, physicsClientId=self._client)
        yaw = p.getEulerFromQuaternion(orn)[2]
        lin, ang = p.getBaseVelocity(bid, physicsClientId=self._client)
        p.resetBasePositionAndOrientation(
            bid, pos, p.getQuaternionFromEuler([0.0, 0.0, yaw]),
            physicsClientId=self._client)
        p.resetBaseVelocity(
            bid, linearVelocity=[lin[0], lin[1], 0.0],
            angularVelocity=[0.0, 0.0, ang[2]],
            physicsClientId=self._client)

    def _get_pos(self, bid):
        """Returns (x, y, theta)."""
        pos, orn = p.getBasePositionAndOrientation(
            bid, physicsClientId=self._client
        )
        yaw = p.getEulerFromQuaternion(orn)[2]
        return np.array([pos[0], pos[1], yaw], dtype=np.float32)

    def _get_vel(self, bid):
        """Returns (vx, vy, omega)."""
        lin, ang = p.getBaseVelocity(bid, physicsClientId=self._client)
        return np.array([lin[0], lin[1], ang[2]], dtype=np.float32)

    def _nearest_def_dist(self, pos2d):
        dists = []
        for did in self.defender_ids:
            dp = self._get_pos(did)[:2]
            dists.append(np.linalg.norm(pos2d - dp))
        return min(dists) if dists else 0.0

    def _defender_starts(self):
        """Randomised defender starting positions."""
        rng = self.np_random
        if self.n_def == 1:
            return [[rng.uniform(-0.7, 0.7), rng.uniform(0.5, 1.5)]]
        elif self.n_def == 2:
            return [
                [rng.uniform(-1.0, -0.2), rng.uniform(0.4, 1.2)],
                [rng.uniform( 0.2,  1.0), rng.uniform(0.4, 1.2)],
            ]
        elif self.n_def == 3:
            return [
                [rng.uniform(-1.0, -0.3), rng.uniform(0.3, 0.9)],
                [rng.uniform(-0.2,  0.2), rng.uniform(0.9, 1.5)],
                [rng.uniform( 0.3,  1.0), rng.uniform(0.3, 0.9)],
            ]

        # 4 or more: generated formation. Defenders are spread evenly across
        # the court width and alternate between a front and a back rank, which
        # generalises the staggered 3-defender layout above. Layouts for 1-3
        # are left exactly as they were so existing results remain
        # reproducible.
        starts = []
        xs = np.linspace(-1.0, 1.0, self.n_def)
        for i, x in enumerate(xs):
            front = (i % 2 == 0)
            y_lo, y_hi = (0.3, 0.9) if front else (0.9, 1.5)
            starts.append([
                float(rng.uniform(x - 0.12, x + 0.12)),
                float(rng.uniform(y_lo, y_hi)),
            ])
        return starts

    def _defender_starts_back_row(self):
        """Defenders in one row at the back wall, as in the physical
        experiment. Each defender keeps its legacy across-court range; only
        the depth changes. Hardware only ran 1-vs-3; other counts use the same
        rule, with jitter narrowed so neighbours cannot overlap."""
        rng = self.np_random
        if self.n_def == 1:
            xr = [(-0.7, 0.7)]
        elif self.n_def == 2:
            xr = [(-1.0, -0.2), (0.2, 1.0)]
        elif self.n_def == 3:
            xr = [(-1.0, -0.3), (-0.2, 0.2), (0.3, 1.0)]
        else:
            xs = np.linspace(-1.0, 1.0, self.n_def)
            j = min(0.12, (xs[1] - xs[0] - 2 * R_RADIUS - 0.02) / 2)
            xr = [(x - j, x + j) for x in xs]
        if self.defender_start_depth is not None:        # author, 13 Nov
            y_lo, y_hi = self.defender_start_depth
        else:
            y_lo = COURT_L / 2 - BACK_ROW_FROM_WALL[1]
            y_hi = COURT_L / 2 - BACK_ROW_FROM_WALL[0]
        return [[float(rng.uniform(*r)), float(rng.uniform(y_lo, y_hi))]
                for r in xr]

    def close(self):
        p.disconnect(physicsClientId=self._client)


# ══════════════════════════════════════════════════════════════════════════════
#  Quick demo
# ══════════════════════════════════════════════════════════════════════════════

def run_demo(n_defenders=1, defender_type="reactive", render=True, n_episodes=3):
    """
    Run a quick demo with random actions to verify the environment works.
    """
    print(f"\nDemo: {n_defenders} defender(s), type='{defender_type}', "
          f"render={render}")
    print("─" * 50)

    env = RoboKabaddiEnv(
        n_defenders   = n_defenders,
        defender_type = defender_type,
        render        = render,
    )

    outcomes = {"touch": 0, "tackle": 0, "empty": 0}

    for ep in range(n_episodes):
        obs, _ = env.reset()
        total_r = 0.0
        steps   = 0
        outcome = "empty"

        while True:
            action = env.action_space.sample()   # random policy
            obs, reward, terminated, truncated, info = env.step(action)
            total_r += reward
            steps   += 1

            if render:
                time.sleep(env.dt)   # real-time pacing

            if terminated or truncated:
                outcome = info.get("outcome", "empty")
                outcomes[outcome] += 1
                break

        print(f"  Episode {ep+1}: outcome={outcome:7s}  "
              f"steps={steps:4d}  reward={total_r:7.2f}")

    print(f"\nSummary over {n_episodes} episodes:")
    print(f"  Touch  : {outcomes['touch']}")
    print(f"  Tackle : {outcomes['tackle']}")
    print(f"  Empty  : {outcomes['empty']}")

    env.close()
    print("\nEnvironment test PASSED ✓")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Robo-Kabaddi demo")
    parser.add_argument("--render",   action="store_true",
                        help="Open PyBullet GUI")
    parser.add_argument("--n_def",    type=int, default=1,
                        choices=[1, 2, 3], help="Number of defenders")
    parser.add_argument("--def_type", type=str, default="reactive",
                        choices=["static", "reactive", "wall", "physical", "gate"])
    parser.add_argument("--episodes", type=int, default=3)
    args = parser.parse_args()

    run_demo(
        n_defenders   = args.n_def,
        defender_type = args.def_type,
        render        = args.render,
        n_episodes    = args.episodes,
    )
