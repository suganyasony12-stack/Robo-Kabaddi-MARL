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
    defender_type : str   — 'static' | 'reactive' | 'wall'
    render        : bool  — open PyBullet GUI
    dt            : float — physics timestep (seconds)
    raid_time     : float — seconds per raid
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

    def __init__(
        self,
        n_defenders   = 1,
        defender_type = "reactive",
        render        = False,
        dt            = 0.05,
        raid_time     = 30.0,
    ):
        super().__init__()

        assert 1 <= n_defenders <= 7, "n_defenders must be between 1 and 7"
        assert defender_type in ("static", "reactive", "wall")

        self.n_def   = n_defenders
        self.def_type = defender_type
        self.render_mode = "human" if render else None
        self.dt      = dt
        self.raid_time = raid_time
        self.max_steps = int(raid_time / dt)

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
        p.setTimeStep(self.dt, physicsClientId=self._client)

        self._build_court()

        # Episode counters
        self._step = 0
        self._t    = 0.0
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
        rx = self.np_random.uniform(-0.9, 0.9)
        ry = self.np_random.uniform(-1.8, -0.3)
        self.raider_id = self._spawn_robot(
            [rx, ry, R_HEIGHT / 2], color=[1, 1, 1, 1]
        )
        self._prev_pos = np.array([rx, ry], dtype=np.float32)

        # Spawn defenders (defender half: y in [0.3, 1.8])
        self.defender_ids = []
        for pos in self._defender_starts():
            did = self._spawn_robot(
                [pos[0], pos[1], R_HEIGHT / 2], color=[1, 0.1, 0.1, 1]
            )
            self.defender_ids.append(did)

        # Settle physics
        for _ in range(20):
            p.stepSimulation(physicsClientId=self._client)

        return self._obs(), {}

    # ──────────────────────────────────────────────────────────────────
    # Step
    # ──────────────────────────────────────────────────────────────────

    def step(self, action):
        self._step += 1
        self._t    += self.dt

        # Apply raider velocity
        vx    = float(action[0]) * self.MAX_LIN_VEL
        vy    = float(action[1]) * self.MAX_LIN_VEL
        omega = float(action[2]) * self.MAX_ANG_VEL
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

        # Current raider pose
        pos    = self._get_pos(self.raider_id)
        fsr_l, fsr_r = self._fsr()
        contact = fsr_l > self.FSR_THRESH or fsr_r > self.FSR_THRESH

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

        # Timeout
        if not terminated and self._t >= self.raid_time:
            if self._flags["contact"]:
                reward += self.R_TACKLE
                info["outcome"] = "tackle"
            else:
                reward += self.R_EMPTY
                info["outcome"] = "empty"
            terminated = True

        truncated = self._step >= self.max_steps
        self._prev_pos = pos[:2].copy()

        return self._obs(), reward, terminated, truncated, info

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

    def _step_defenders(self):
        r_pos = self._get_pos(self.raider_id)

        for i, did in enumerate(self.defender_ids):
            if self.def_type == "static":
                vx, vy = 0.0, 0.0

            elif self.def_type == "reactive":
                vx, vy = self._reactive(did, r_pos)

            elif self.def_type == "wall":
                vx, vy = self._wall(did, i, r_pos)

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

    # ──────────────────────────────────────────────────────────────────
    # PyBullet helpers
    # ──────────────────────────────────────────────────────────────────

    def _build_court(self):
        """Create floor + boundary walls."""
        p.loadURDF("plane.urdf", physicsClientId=self._client)

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
        cid = p.createCollisionShape(
            p.GEOM_CYLINDER, radius=R_RADIUS, height=R_HEIGHT,
        physicsClientId=self._client,
        )
        vid = p.createVisualShape(
            p.GEOM_CYLINDER, radius=R_RADIUS, length=R_HEIGHT,
            rgbaColor=color, physicsClientId=self._client,
        )
        bid = p.createMultiBody(
            baseMass=1.5,
            baseCollisionShapeIndex=cid,
            baseVisualShapeIndex=vid,
            basePosition=position,
            physicsClientId=self._client,
        )
        p.changeDynamics(
            bid, -1,
            lateralFriction=0.8,
            linearDamping=0.9,
            angularDamping=0.9,
            physicsClientId=self._client,
        )
        return bid

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
                        choices=["static", "reactive", "wall"])
    parser.add_argument("--episodes", type=int, default=3)
    args = parser.parse_args()

    run_demo(
        n_defenders   = args.n_def,
        defender_type = args.def_type,
        render        = args.render,
        n_episodes    = args.episodes,
    )
