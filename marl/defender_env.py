"""
defender_env.py — the defenders learn, the raider is frozen (C3, stage 2)
=========================================================================

One raid of RoboKabaddiEnv with defender_type="learned" and the author's foul
rules (CORRECTED_CONFIG otherwise). The raider is a trained C1 run, loaded with
the normalisation statistics saved with it and played deterministically, as
in exp_C1_evaluate.py, with the robots' boundary escape (raider_edge_escape)
because a frozen raider cannot learn the lines; or the scripted hardware-like raider; or a pool of
them, one drawn per raid. A raider trained against learned defenders under the
out rule (marl/raider_vs_learned.py) has learned the lines and plays without
the steering (author, 28 Oct).

    env = DefenderEnv(raider="main_1v3_seed42_ext")
    obs, state = env.reset(seed)          # obs (N, d): one row per defender
    obs, state, r, done, info = env.step(actions)   # N indices into DEF_ACTIONS

r is the defenders' team reward (TEAM_REWARD), the same for every defender.
VecDefenderEnv runs several in worker processes and resets them itself.

r_walkout sets the defenders' reward for a walk-out: the raider over a line
(scored as a tackle) with no defender touching it in the last WALKOUT_STEPS
decisions, the same rule as raider_vs_learned.py --walkout (default None: +1,
as any tackle). The author set 0 on 30 Oct: trained against the seed 42 raider
of the walk-out round and the scripted raider, the defenders won every raid
against seed 42 by stepping aside while it ran over the end line (0 holds) and
never learned to hold the scripted raider. A tackle with contact (held, pushed
out) keeps +1. The outcome is unchanged (tackle); info["walkout"] marks it.

pad_to plays policies built for more defenders than are on the court (author,
4 Nov: train the final pair for 2 defenders, starting from the 3-defender
policies): each missing defender is a phantom standing still at PHANTOM_POS,
the centre of the defenders' end line, in the inputs only (not on the court).
Default None: inputs for the defenders on the court, as before. A trained
raider gets the same phantoms when the court has fewer defenders than its
input (FrozenRaider, pad_raider_obs).

raider_phantom sets where the raider's phantoms are (author, 4 Nov): "deep"
(default, as above) or "copy", on top of the nearest real defender. With 1
defender the two deep phantoms drew the raider to the end line, where it
walked out; copy keeps its input on the real defender. Defenders' own
phantom teammates stay deep. A raider trained with copy keeps it
(run_info "raider_phantom"); the option overrides a frozen raider's setting.

Touch input (author, 12 Nov): the raider's inputs have the touch sensors now
but no record of a touch already made, and its network has no memory, so the
raiders learned to turn home by the clock (about 7-8 s, when they had always
touched at full speed). A raider trained with raider_vs_learned.py
--touch-obs (run_info "raider_touch_obs") gets one more input after the
others: the simulator's own touch-made flag (_flags["contact"], 0 before the
touch, 1 after, as the raid is judged). Raiders without it get exactly the
same inputs as before (raider_input, FrozenRaider).
"""
from __future__ import annotations

import json
import os
import sys
from multiprocessing import get_context
from pathlib import Path

os.environ.setdefault("OMP_NUM_THREADS", "1")
import numpy as np                                                    # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT), str(ROOT / "revision_experiments"), str(ROOT / "training")]
from envs.kabaddi_env import (RoboKabaddiEnv, CORRECTED_CONFIG, DEF_ACTIONS,   # noqa: E402
                              COURT_W, COURT_L)

C1 = ROOT / "results" / "c1"
N_ACTIONS = len(DEF_ACTIONS)
WALKOUT_STEPS = 5             # decisions (1 s at 5 Hz) with no defender touching
PHANTOM_POS = (0.0, 1.9)      # a missing defender (pad_to), inputs only
PHANTOM_MODES = ("deep", "copy")


def diag_kwargs(diag_range=None, diag_period=None):
    """RoboKabaddiEnv settings for --diag-range LO,HI / --diag-period S
    (author, 11 Nov: each robot its own diagonal speed); {} when off."""
    if not diag_range:
        return {}
    lo, hi = (float(v) for v in str(diag_range).split(","))
    return {"diag_ratio_range": (lo, hi), "diag_ratio_period": diag_period}


def hold_kwargs(hold=None):
    """RoboKabaddiEnv settings for --hold T,D,C (author, 14 Nov: 2,0.25,0.25):
    held after T s within D m with a defender centre within C m; {} when off."""
    if not hold:
        return {}
    t, d, c = (float(v) for v in str(hold).split(","))
    return {"static_time": t, "hold_dist": d, "hold_touch_dist": c}


def margin_kwargs(margin=None):
    """RoboKabaddiEnv setting for --return-margin M (author, 13 Nov: 0.03):
    back over the midline counts as Empty only M metres behind it; {} when off."""
    return {} if not margin else {"return_margin": float(margin)}


def start_kwargs(def_start=None):
    """RoboKabaddiEnv setting for --def-start LO,HI: the defenders' start depth
    in metres past the midline (author, 13 Nov: 1.0,1.5 = 3.0-3.5 m from the
    raider's end line); {} when off (back row, as before)."""
    if not def_start:
        return {}
    lo, hi = (float(v) for v in str(def_start).split(","))
    return {"defender_start_depth": (lo, hi)}


def pad_raider_obs(env, obs, n_in, mode="deep"):
    """Raider input for n_in defenders on a court with fewer: phantoms after
    the real defenders' relative positions, at PHANTOM_POS (deep) or on top
    of the nearest real defender (copy)."""
    k = len(env.defender_ids)
    if k >= n_in:
        return obs
    assert mode in PHANTOM_MODES, mode
    i = 6 + 2 * k
    if mode == "copy":
        real = obs[6:i].reshape(k, 2)
        near = real[int(np.argmin(np.hypot(real[:, 0], real[:, 1])))]
        ph = list(near) * (n_in - k)
    else:
        r = env._get_pos(env.raider_id)
        ph = [PHANTOM_POS[0] - r[0], PHANTOM_POS[1] - r[1]] * (n_in - k)
    return np.concatenate([obs[:i], ph, obs[i:]]).astype(np.float32)


def raider_input(env, obs, n_in, mode="deep", touch=False):
    """The raider's whole input: pad_raider_obs, then (touch=True, author
    12 Nov) the touch-made flag as one more number at the end."""
    obs = pad_raider_obs(env, obs, n_in, mode)
    if not touch:
        return obs
    return np.append(obs, np.float32(env._flags["contact"])).astype(np.float32)


def touch_space(space):
    """An input space one number larger (the touch-made flag)."""
    import gymnasium as gym
    n = space.shape[0] + 1
    return gym.spaces.Box(-np.inf, np.inf, (n,), np.float32)


def pad_defender_obs(env, rows, n_in):
    """defender_obs() rows for n_in defenders: phantom teammates at PHANTOM_POS
    join the real ones, all sorted by distance as in defender_obs()."""
    k = env.n_def
    if k >= n_in:
        return rows
    P = [env._get_pos(d) for d in env.defender_ids]
    out = []
    for i in range(k):
        x, y = float(P[i][0]), float(P[i][1])
        mates = [(P[j][0] - x, P[j][1] - y) for j in range(k) if j != i]
        mates += [(PHANTOM_POS[0] - x, PHANTOM_POS[1] - y)] * (n_in - k)
        mates.sort(key=lambda q: np.hypot(*q))
        out.append(np.concatenate([rows[i, :8], [c for q in mates for c in q],
                                   rows[i, 8 + 2 * (k - 1):]]))
    return np.array(out, dtype=np.float32)


def pad_global_state(env, s, n_in):
    """global_state() for n_in defenders: phantoms (x, y, 0, 0) after the real ones."""
    k = env.n_def
    if k >= n_in:
        return s
    ph = [PHANTOM_POS[0], PHANTOM_POS[1], 0.0, 0.0] * (n_in - k)
    return np.concatenate([s[:4 + 4 * k], ph, s[4 + 4 * k:]]).astype(np.float32)


RAIDER_RULES = ("deep",)
RULE_HOME_S = 8.0       # author, 13 Nov: the rule stops this many seconds before the raid ends


def deep_rule(env, a, home_s=RULE_HOME_S):
    """'Go deep until touch' (author, 13 Nov), a rule on top of the trained
    raider: before a counted touch, with more than home_s seconds of the raid
    left, a command home (vy below the dead zone) or a stop (both components
    in the dead zone) becomes a full-speed move toward the nearest defender
    on the court. Every other command, every command after the touch and in
    the last home_s seconds are the policy's own. Returns (action, changed)."""
    from envs.kabaddi_env import CMD_DEADZONE
    if env._flags["contact"]:
        return a, False
    if env._raid_start_step is not None:
        left = (env._raid_steps - (env._step - env._raid_start_step)) * env.dt
        if left <= home_s:
            return a, False
    vx, vy = float(a[0]), float(a[1])
    home = vy < -CMD_DEADZONE
    stop = abs(vx) < CMD_DEADZONE and abs(vy) < CMD_DEADZONE
    if not (home or stop):
        return a, False
    r = env._get_pos(env.raider_id)[:2]
    d = min((env._get_pos(i)[:2] for i in env.defender_ids), key=lambda q: float(np.hypot(*(q - r))))
    v = (d - r) / (float(np.hypot(*(d - r))) + 1e-9)
    b = np.array(a, dtype=np.float32).copy()
    b[0], b[1] = v[0], v[1]                    # larger component >= 0.71: the 100 % level
    if len(b) > 2:
        b[2] = 0.0
    return b, True


class FrozenRaider:
    """A trained C1 raider: best checkpoint with its VecNormalize statistics.
    rule='deep': deep_rule on top (author, 13 Nov; default None, the policy alone)."""

    def __init__(self, run: str, phantom=None, rule=None, rule_home_s=RULE_HOME_S):
        assert rule is None or rule in RAIDER_RULES, f"rule must be None or one of {RAIDER_RULES}"
        self.rule, self.rule_home_s = rule, float(rule_home_s)
        self.rule_changed = 0                  # decisions changed by the rule (counter)
        import torch
        from stable_baselines3 import PPO, SAC, TD3
        from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize
        torch.set_num_threads(1)
        run_dir = Path(run) if "/" in run else C1 / run     # a C1 run or a path
        info = json.loads((run_dir / "run_info.json").read_text())
        self.env_kwargs = info["env_kwargs"]
        # defenders in its input (more than on the court when trained with pad_to)
        self.n_in = info.get("pad_to") or self.env_kwargs.get("n_defenders", 3)
        self.phantom = phantom or info.get("raider_phantom") or "deep"
        # trained against learned defenders under the out rule, no steering:
        # it has learned the lines (author, 28 Oct: plays without the steering)
        self.learned_lines = (self.env_kwargs.get("defender_type") == "learned"
                              and not self.env_kwargs.get("raider_edge_escape", False))
        cls = {"ppo": PPO, "sac": SAC, "td3": TD3}[info["spec"]["algo"]]
        kw = dict(self.env_kwargs, **({"n_defenders": info["pad_to"]} if info.get("pad_to") else {}))
        self.touch = bool(info.get("raider_touch_obs", False))     # author, 12 Nov
        dummy = DummyVecEnv([lambda: RoboKabaddiEnv(**kw)])
        if self.touch:                       # the saved statistics have one more input
            dummy.observation_space = touch_space(dummy.observation_space)
        self.vn = VecNormalize.load(str(run_dir / "best_vecnorm.pkl"), dummy)
        self.vn.training, self.vn.norm_reward = False, False
        self.model = cls.load(str(run_dir / "best_model" / "best_model.zip"), device="cpu")
        dummy.close()

    def reset(self, env, obs):
        return obs

    def act(self, env, obs):
        obs = raider_input(env, obs, self.n_in, self.phantom, self.touch)
        a, _ = self.model.predict(self.vn.normalize_obs(obs[None]).astype(np.float32)[0],
                                  deterministic=True)
        if self.rule == "deep":
            a, changed = deep_rule(env, a, self.rule_home_s)
            self.rule_changed += changed
        return a


def make_raider(raider: str, n_def=3, phantom=None, rule=None):
    if raider == "scripted":
        from hardware_raider import HardwareRaider
        return HardwareRaider(n_def)
    return FrozenRaider(raider, phantom, rule)


class DefenderEnv:
    """raider: one name, or several separated by commas (a pool): each raid
    draws one of them from its own seed, so a raid is reproducible."""

    def __init__(self, raider="main_1v3_seed42_ext", n_def=3, r_walkout=None, pad_to=None,
                 raider_phantom=None, raider_rule=None, **env_overrides):
        self.r_walkout = r_walkout
        self.n_in = max(n_def, pad_to or n_def)     # defenders in the policy's input
        self._touch = []
        self.raider_names = raider.split(",")
        self.raiders = [make_raider(r, n_def, raider_phantom, raider_rule) for r in self.raider_names]
        self.raider, self.raider_name = self.raiders[0], self.raider_names[0]
        # frozen raiders get the robots' boundary escape (author, 27 Oct);
        # held needs a defender touching the raider (author, 28 Oct; the runs
        # of 27-28 Oct night used held_needs_touch=False)
        kw = dict(CORRECTED_CONFIG, n_defenders=n_def, defender_type="learned",
                  defender_fouls=True, raider_edge_escape=True, held_needs_touch=True)
        kw.update(env_overrides)
        self.env = RoboKabaddiEnv(**kw)
        # the steering only for raiders that have not learned the lines
        self.escape = [kw["raider_edge_escape"] and not getattr(r, "learned_lines", False)
                       for r in self.raiders]
        self.env.raider_edge_escape = self.escape[0]
        self.n_def = n_def
        self.obs_dim = RoboKabaddiEnv.DEF_OBS_FIXED + 2 * (self.n_in - 1)
        self.state_dim = 4 + 4 * self.n_in + 3

    def _inputs(self):
        e = self.env
        return (pad_defender_obs(e, e.defender_obs(), self.n_in),
                pad_global_state(e, e.global_state(), self.n_in))

    def reset(self, seed=None):
        if len(self.raiders) > 1:
            k = int(np.random.default_rng(seed).integers(len(self.raiders)))
            self.raider, self.raider_name = self.raiders[k], self.raider_names[k]
            self.env.raider_edge_escape = self.escape[k]
        r_obs, _ = self.env.reset(seed=seed)
        self._touch = []
        self._r_obs = self.raider.reset(self.env, r_obs)
        return self._inputs()

    def step(self, actions):
        self.env.set_defender_actions(actions)
        e = self.env
        self._r_obs, _, term, trunc, info = e.step(self.raider.act(e, self._r_obs))
        r = info["team_reward"][1]
        self._touch.append(e._defender_touching())
        if term or trunc:
            pos = e._get_pos(e.raider_id)
            info["walkout"] = (info.get("outcome") == "tackle"
                               and (abs(pos[0]) > COURT_W / 2 or abs(pos[1]) > COURT_L / 2)
                               and not any(self._touch[-WALKOUT_STEPS:]))
            if info["walkout"] and self.r_walkout is not None:
                r = self.r_walkout
        return (*self._inputs(), r, term or trunc, info)

    def close(self):
        self.env.close()


def _worker(remote, raider, n_def, seeds, stride, overrides, r_walkout=None, pad_to=None,
            raider_phantom=None):
    envs = [DefenderEnv(raider, n_def, r_walkout, pad_to, raider_phantom, **overrides)
            for _ in seeds]
    next_seed = list(seeds)

    def reset(k):
        o, s = envs[k].reset(seed=next_seed[k])
        next_seed[k] += stride
        return o, s
    while True:
        cmd, data = remote.recv()
        if cmd == "reset":
            out = [reset(k) for k in range(len(envs))]
            remote.send(([o for o, _ in out], [s for _, s in out]))
        elif cmd == "step":
            res = []
            for k, a in enumerate(data):
                o, s, r, d, info = envs[k].step(a)
                outcome = info.get("outcome", "empty") if d else None
                walkout = bool(info.get("walkout")) if d else False
                if d:
                    o, s = reset(k)
                res.append((o, s, r, d, outcome, walkout))
            remote.send(res)
        elif cmd == "close":
            for e in envs:
                e.close()
            remote.close()
            return


class VecDefenderEnv:
    """n_workers processes x envs_per_worker raids. Env k starts with seed
    seed0 + k and each finished raid is reset at once with that env's next
    seed (+ n_envs), so no two raids share a seed and a run is reproducible
    for a fixed layout. After each step, last_walkouts marks the raids that
    ended in a walk-out (see r_walkout)."""

    def __init__(self, raider, n_def=3, n_workers=6, envs_per_worker=2, seed0=0,
                 r_walkout=None, pad_to=None, raider_phantom=None, **overrides):
        ctx = get_context("spawn")
        n = n_workers * envs_per_worker
        self.n_envs, self.n_def = n, n_def
        self.remotes, self.procs = [], []
        for w in range(n_workers):
            seeds = [seed0 + w * envs_per_worker + j for j in range(envs_per_worker)]
            parent, child = ctx.Pipe()
            pr = ctx.Process(target=_worker,
                             args=(child, raider, n_def, seeds, n, overrides, r_walkout,
                                   pad_to, raider_phantom),
                             daemon=True)
            pr.start()
            child.close()
            self.remotes.append(parent)
            self.procs.append(pr)
        self.epw = envs_per_worker
        n_in = max(n_def, pad_to or n_def)
        self.obs_dim = RoboKabaddiEnv.DEF_OBS_FIXED + 2 * (n_in - 1)
        self.state_dim = 4 + 4 * n_in + 3

    def reset(self):
        for r in self.remotes:
            r.send(("reset", None))
        O, S = [], []
        for r in self.remotes:
            o, s = r.recv()
            O += o
            S += s
        return np.array(O, np.float32), np.array(S, np.float32)

    def step(self, actions):
        """actions: (n_envs, N) ints. Returns obs (n_envs, N, d), state
        (n_envs, s), team reward (n_envs,), done (n_envs,), outcomes list
        (None where the raid goes on)."""
        for w, r in enumerate(self.remotes):
            r.send(("step", actions[w * self.epw:(w + 1) * self.epw].tolist()))
        res = [x for r in self.remotes for x in r.recv()]
        O, S, R, D, OUT, WO = zip(*res)
        self.last_walkouts = list(WO)
        return (np.array(O, np.float32), np.array(S, np.float32),
                np.array(R, np.float32), np.array(D, bool), list(OUT))

    def close(self):
        for r in self.remotes:
            r.send(("close", None))
        for pr in self.procs:
            pr.join(timeout=10)
