"""
smorphi_controller.py  —  Send movement commands to Smorphi robots via TCP
===========================================================================
Translates PPO continuous actions [vx, vy, omega] → ASCII movement characters,
and reactive defender actions (move-toward target) → ASCII commands.

All commands use the EXACT same ASCII character map as the BLE firmware.

ASCII Command Map (from handleBLEData in firmware):
  Forward:      [ (100%)  A (75%)  ] (50%)  ^ (25%)
  Backward:     _ (100%)  ` (75%)  a (50%)  b (25%)
  Left:         c (100%)  d (75%)  e (50%)  f (25%)
  Right:        g (100%)  h (75%)  i (50%)  j (25%)
  Diag Fwd-L:   k (100%)  l (75%)  m (50%)  n (25%)
  Diag Fwd-R:   o (100%)  p (75%)  q (50%)  r (25%)
  Diag Back-L:  s (100%)  t (75%)  u (50%)  v (25%)
  Diag Back-R:  w (100%)  x (75%)  y (50%)  z (25%)
  Pivot Left:   {
  Pivot Right:  |
  Stop:         }

Speed thresholds for PPO action mapping:
  |component| > HIGH_THRESH  → 100%
  |component| > MID_THRESH   → 75%
  |component| > LOW_THRESH   → 50%
  else                       → 25%
"""

import numpy as np
import fsr_listener   # uses send_command() from fsr_listener

# ========================== COURT BOUNDARIES ==================================

COURT_W       = 2.0    # physical court width  (x: 0.0 → 2.0 m)
COURT_L       = 3.0    # physical court length (y: 0.0 → 3.0 m)
BOUND_MARGIN  = 0.20   # stop within 20cm of any boundary

# ========================== SPEED THRESHOLDS =================================

HIGH_THRESH = 0.60   # 100% speed
MID_THRESH  = 0.35   # 75%  speed
LOW_THRESH  = 0.15   # 50%  speed (below this = 25%, above deadzone)
DEADZONE    = 0.12   # below this in ALL axes → stop

# ========================== DEFENDER EVASION ==================================

EVASION_DIST_M = 0.50   # defender evades when raider centre within 50 cm

# ========================== COMMAND TABLES ===================================

# Index: 0=25%  1=50%  2=75%  3=100%
CMD_FORWARD     = ["^", "]", "A", "["]
CMD_BACKWARD    = ["b", "a", "`", "_"]
CMD_LEFT        = ["f", "e", "d", "c"]
CMD_RIGHT       = ["j", "i", "h", "g"]
CMD_DIAG_FWD_L  = ["n", "m", "l", "k"]
CMD_DIAG_FWD_R  = ["r", "q", "p", "o"]
CMD_DIAG_BCK_L  = ["v", "u", "t", "s"]
CMD_DIAG_BCK_R  = ["z", "y", "x", "w"]
CMD_PIVOT_LEFT  = "{"
CMD_PIVOT_RIGHT = "|"
CMD_STOP        = "}"

# ========================== BOUNDARY ENFORCEMENT ==============================

def _near_boundary(pos):
    """Return True if robot is within BOUND_MARGIN of any court edge."""
    if pos is None:
        return False
    x, y = pos
    return (x < BOUND_MARGIN or x > COURT_W - BOUND_MARGIN or
            y < BOUND_MARGIN or y > COURT_L - BOUND_MARGIN)

def _boundary_safe_cmd(pos, cmd):
    """
    Override cmd with STOP if robot is at a boundary AND the command
    would push it further out.  Returns the (possibly overridden) command.
    """
    if pos is None:
        return cmd
    x, y = pos

    MOVES_LEFT  = set(CMD_LEFT  + CMD_DIAG_FWD_L + CMD_DIAG_BCK_L)
    MOVES_RIGHT = set(CMD_RIGHT + CMD_DIAG_FWD_R + CMD_DIAG_BCK_R)
    MOVES_FWD   = set(CMD_FORWARD  + CMD_DIAG_FWD_L + CMD_DIAG_FWD_R)
    MOVES_BACK  = set(CMD_BACKWARD + CMD_DIAG_BCK_L + CMD_DIAG_BCK_R)

    if x < BOUND_MARGIN            and cmd in MOVES_LEFT:  return CMD_STOP
    if x > COURT_W - BOUND_MARGIN  and cmd in MOVES_RIGHT: return CMD_STOP
    if y < BOUND_MARGIN            and cmd in MOVES_BACK:  return CMD_STOP
    if y > COURT_L - BOUND_MARGIN  and cmd in MOVES_FWD:   return CMD_STOP

    return cmd

# ========================== HELPERS ==========================================

def _speed_idx(magnitude):
    """Map |velocity| to speed index 0-3."""
    if magnitude > HIGH_THRESH: return 3
    if magnitude > MID_THRESH:  return 2
    if magnitude > LOW_THRESH:  return 1
    return 0   # 25% — only used if above DEADZONE


def action_to_cmd(action):
    """
    Convert PPO action [vx, vy, omega] (each in [-1, 1]) to an ASCII command char.

    Coordinate convention (matches simulation):
      vx > 0 = move RIGHT (in robot's local frame → physical right)
      vy > 0 = move FORWARD (toward defender side)
      omega > 0 = pivot LEFT (counter-clockwise)

    NOTE: If your physical robot's forward direction is flipped relative to
    the simulation, set FLIP_VY = True and/or FLIP_VX = True below.
    """
    # ── Orientation correction ──────────────────────────────────────────────
    # Set these True if the robot moves opposite to expected direction:
    FLIP_VX = True    # Confirmed by trial_run validation: raider right→went left, left→went right
    FLIP_VY = False   # True if robot goes backward when it should go forward
    # ────────────────────────────────────────────────────────────────────────

    vx, vy, omega = float(action[0]), float(action[1]), float(action[2])

    if FLIP_VX: vx = -vx
    if FLIP_VY: vy = -vy

    # Pivot disabled — keeps robot orientation fixed throughout the episode
    # so court-frame and robot-frame always match.
    omega = 0.0

    abs_vx    = abs(vx)
    abs_vy    = abs(vy)
    abs_omega = abs(omega)

    # All below deadzone → stop
    if abs_vx < DEADZONE and abs_vy < DEADZONE and abs_omega < DEADZONE:
        return CMD_STOP

    # Rotation dominant → pivot
    if abs_omega > abs_vx and abs_omega > abs_vy and abs_omega > 0.25:
        return CMD_PIVOT_LEFT if omega > 0 else CMD_PIVOT_RIGHT

    # Determine direction components
    fwd   = vy >  DEADZONE
    back  = vy < -DEADZONE
    right = vx >  DEADZONE
    left  = vx < -DEADZONE

    # Speed from the dominant axis
    dominant = max(abs_vx, abs_vy)
    spd = _speed_idx(dominant)

    # Diagonal movements
    if fwd  and right: return CMD_DIAG_FWD_R[spd]
    if fwd  and left:  return CMD_DIAG_FWD_L[spd]
    if back and right: return CMD_DIAG_BCK_R[spd]
    if back and left:  return CMD_DIAG_BCK_L[spd]

    # Cardinal movements
    if fwd:   return CMD_FORWARD[spd]
    if back:  return CMD_BACKWARD[spd]
    if right: return CMD_RIGHT[spd]
    if left:  return CMD_LEFT[spd]

    return CMD_STOP


def defender_move_toward(defender_pos, target_pos, speed_pct=75):
    """
    Generate command for a defender to move toward target_pos.

    ORIENTATION NOTE:
      Defenders face toward the raider side (-y direction in court).
      So their robot-frame FORWARD = court -y = toward raider.
      ny < 0  (target below defender) → use CMD_FORWARD  (move toward raider)
      ny > 0  (target above defender) → use CMD_BACKWARD (move away — shouldn't happen)
      nx > 0  (target to the right)   → use CMD_RIGHT
      nx < 0  (target to the left)    → use CMD_LEFT

    If your defenders face the opposite direction, set FLIP_DEFENDER_Y = True.
    """
    # ── Orientation flags — adjust if defenders move in wrong direction ────
    FLIP_DEFENDER_Y = False   # True if defenders move away instead of toward raider
    FLIP_DEFENDER_X = False   # True if defenders move left instead of right
    # ────────────────────────────────────────────────────────────────────────

    speed_map = {25: 0, 50: 1, 75: 2, 100: 3}
    spd = speed_map.get(speed_pct, 2)

    if defender_pos is None or target_pos is None:
        return CMD_STOP

    dx = target_pos[0] - defender_pos[0]
    dy = target_pos[1] - defender_pos[1]

    dist = np.hypot(dx, dy)
    if dist < 0.05:
        return CMD_STOP

    # Normalise
    nx = dx / dist
    ny = dy / dist

    if FLIP_DEFENDER_Y: ny = -ny
    if FLIP_DEFENDER_X: nx = -nx

    # Defenders face -y (toward raider side):
    # ny < 0 means target is at lower y → move FORWARD to approach
    # ny > 0 means target is at higher y → move BACKWARD
    fwd   = ny < -0.3    # target below defender → forward = toward raider
    back  = ny >  0.3    # target above defender → backward
    right = nx >  0.3
    left  = nx < -0.3

    if fwd  and right: return CMD_DIAG_FWD_R[spd]
    if fwd  and left:  return CMD_DIAG_FWD_L[spd]
    if back and right: return CMD_DIAG_BCK_R[spd]
    if back and left:  return CMD_DIAG_BCK_L[spd]
    if fwd:            return CMD_FORWARD[spd]
    if back:           return CMD_BACKWARD[spd]
    if right:          return CMD_RIGHT[spd]
    if left:           return CMD_LEFT[spd]

    return CMD_STOP


def defender_move_away(defender_pos, raider_pos, speed_pct=75):
    """
    Move defender AWAY from raider (evasion when raider is within EVASION_DIST_M).

    Direction is the mirror of defender_move_toward: the displacement vector
    points FROM the raider TO the defender, so the robot retreats.
    Same orientation convention as defender_move_toward (faces -y).
    """
    speed_map = {25: 0, 50: 1, 75: 2, 100: 3}
    spd = speed_map.get(speed_pct, 2)

    if defender_pos is None or raider_pos is None:
        return CMD_STOP

    # Vector pointing AWAY from raider
    dx = defender_pos[0] - raider_pos[0]
    dy = defender_pos[1] - raider_pos[1]

    dist = np.hypot(dx, dy)
    if dist < 0.05:
        return CMD_STOP

    nx = dx / dist
    ny = dy / dist

    # Defenders face -y (same convention as move_toward):
    # ny < 0 → moving -y → FORWARD  (toward raider side — only if raider is behind)
    # ny > 0 → moving +y → BACKWARD (away toward back wall)
    fwd   = ny < -0.3
    back  = ny >  0.3
    right = nx >  0.3
    left  = nx < -0.3

    if fwd  and right: return CMD_DIAG_FWD_R[spd]
    if fwd  and left:  return CMD_DIAG_FWD_L[spd]
    if back and right: return CMD_DIAG_BCK_R[spd]
    if back and left:  return CMD_DIAG_BCK_L[spd]
    if fwd:            return CMD_FORWARD[spd]
    if back:           return CMD_BACKWARD[spd]
    if right:          return CMD_RIGHT[spd]
    if left:           return CMD_LEFT[spd]

    return CMD_STOP


# ========================== HIGH-LEVEL CONTROL ===============================

def send_raider_action(action, raider_pos=None):
    """
    Convert PPO action and send to raider_1 via TCP.
    Applies boundary enforcement if raider_pos is provided.
    Returns (success, cmd).
    """
    cmd = action_to_cmd(action)
    cmd = _boundary_safe_cmd(raider_pos, cmd)
    ok  = fsr_listener.send_command("raider_1", cmd)
    return ok, cmd


def send_raider_retreat(raider_pos, speed_pct=75):
    """
    Position-based retreat controller for the raider.

    Goal: return to own half (court y < 1.5 m) without hitting side boundaries.

    Strategy
    --------
    * Primary drive : vy = -0.80  →  court -y  →  toward own half.
    * Lateral nudge : if the raider is within SIDE_MARGIN of a side wall,
                      add an x-component toward court centre.
    * No pivot      : omega is always 0, so orientation never changes.

    Uses action_to_cmd() so FLIP_VX is applied automatically — no need to
    manually account for the raider's physical left/right swap.

    Returns
    -------
    (success: bool, action_vec: [vx, vy, omega]) — action_vec is useful for
    the live sidebar display in run_experiment.py.
    """
    SIDE_MARGIN = 0.45   # start nudging at 45 cm from each side wall

    if raider_pos is None:
        fsr_listener.send_command("raider_1", CMD_STOP)
        return False, [0.0, 0.0, 0.0]

    x, y = raider_pos

    vy = -0.80   # primary: drive toward own half (-y direction)

    # Lateral nudge — steer robot toward centre before reaching the wall
    if x < SIDE_MARGIN:
        vx = +0.50          # court +x (centre direction)
    elif x > COURT_W - SIDE_MARGIN:
        vx = -0.50          # court -x (centre direction)
    else:
        vx = 0.0

    action = np.array([vx, vy, 0.0], dtype=np.float32)
    cmd    = action_to_cmd(action)            # FLIP_VX applied automatically
    cmd    = _boundary_safe_cmd(raider_pos, cmd)

    ok = fsr_listener.send_command("raider_1", cmd, retries=2, retry_delay=0.02)
    return ok, [vx, vy, 0.0]


def send_defender_action(defender_id, defender_pos, raider_pos, speed_pct=75,
                         contact_confirmed=False):
    """
    Reactive defender controller — two modes depending on contact state.

    NORMAL MODE  (contact_confirmed=False)
      • Evade when raider is within EVASION_DIST_M  (avoid easy tag)
      • Chase / block when raider is farther away
      • Soft boundary at DEFENDER_SOFT_Y stops defenders well above midline

    TACKLE MODE  (contact_confirmed=True)
      • Full-speed (100%) aggressive pursuit — no evasion at all
      • Soft boundary skipped so defenders can chase raider right up to midline
      • Hard boundary (1.50 m) still enforced — defenders never cross midline

    Parameters
    ----------
    defender_id       : "defender_1" | "defender_2" | "defender_3"
    defender_pos      : (x_phys, y_phys) of this defender
    raider_pos        : (x_phys, y_phys) of raider
    speed_pct         : movement speed used in normal mode  (25/50/75/100)
    contact_confirmed : True once raider has made confirmed contact with any defender
    """
    MIDLINE_Y       = 1.5
    DEFENDER_HARD_Y = 1.5    # hard boundary — always enforced
    DEFENDER_SOFT_Y = 1.80   # soft boundary — normal mode only

    if raider_pos is None or defender_pos is None:
        fsr_listener.send_command(defender_id, CMD_STOP)
        return False

    # ── Hard boundary: crossed midline → stop + push back (always) ────────────
    if defender_pos[1] < DEFENDER_HARD_Y:
        fsr_listener.send_command(defender_id, CMD_STOP, retries=2, retry_delay=0.02)
        fsr_listener.send_command(defender_id, CMD_BACKWARD[2], retries=2, retry_delay=0.02)
        return True

    MOVES_TOWARD_MIDLINE = set(CMD_FORWARD + CMD_DIAG_FWD_L + CMD_DIAG_FWD_R)

    if contact_confirmed:
        # ── TACKLE MODE: full-speed pursuit, no evasion ───────────────────────
        # Stop only if raider has already escaped back to its own half
        if raider_pos[1] <= MIDLINE_Y:
            fsr_listener.send_command(defender_id, CMD_STOP)
            return True

        cmd = defender_move_toward(defender_pos, raider_pos, 100)
        cmd = _boundary_safe_cmd(defender_pos, cmd)

        # Last-line safety: never step over the hard boundary
        if cmd in MOVES_TOWARD_MIDLINE and defender_pos[1] < DEFENDER_HARD_Y + 0.10:
            cmd = CMD_STOP

        ok = fsr_listener.send_command(defender_id, cmd)
        return ok

    else:
        # ── NORMAL MODE: evade when close, chase when far ─────────────────────
        # Soft boundary: absorb momentum before reaching midline
        if defender_pos[1] < DEFENDER_SOFT_Y:
            fsr_listener.send_command(defender_id, CMD_STOP, retries=2, retry_delay=0.02)
            return True

        # Only engage once raider is in defender half
        if raider_pos[1] <= MIDLINE_Y:
            fsr_listener.send_command(defender_id, CMD_STOP)
            return True

        cmd = defender_move_toward(defender_pos, raider_pos, speed_pct)

        cmd = _boundary_safe_cmd(defender_pos, cmd)

        # Don't creep toward midline when already near soft boundary
        if cmd in MOVES_TOWARD_MIDLINE and defender_pos[1] < DEFENDER_SOFT_Y + 0.10:
            cmd = CMD_STOP

        ok = fsr_listener.send_command(defender_id, cmd)
        return ok


def stop_all_robots(robot_ids):
    """Send stop command to all robots with retries to handle TCP drops."""
    for rid in robot_ids:
        fsr_listener.send_command(rid, CMD_STOP, retries=5, retry_delay=0.03)


# ========================== STANDALONE TEST ==================================

if __name__ == "__main__":
    """Quick test of action_to_cmd mapping."""
    test_cases = [
        ([0.0,  0.8,  0.0], "Forward 100%"),
        ([0.0, -0.8,  0.0], "Backward 100%"),
        ([0.8,  0.0,  0.0], "Right 100%"),
        ([-0.8, 0.0,  0.0], "Left 100%"),
        ([0.5,  0.5,  0.0], "Diag Fwd-Right 75%"),
        ([-0.5, 0.5,  0.0], "Diag Fwd-Left 75%"),
        ([0.0,  0.0,  0.7], "Pivot Left"),
        ([0.0,  0.0, -0.7], "Pivot Right"),
        ([0.0,  0.0,  0.0], "Stop"),
        ([0.05, 0.05, 0.05], "All below deadzone → Stop"),
    ]

    print("=== Action → Command Mapping Test ===\n")
    print(f"{'Action':35s}  {'Expected':25s}  {'CMD':5s}")
    print("-" * 70)
    for action, label in test_cases:
        cmd = action_to_cmd(action)
        print(f"  {str(action):33s}  {label:25s}  '{cmd}'")
