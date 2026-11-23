"""
mappo.py — train the defender team with MAPPO or IPPO (C3, stage 2)
====================================================================

One policy is shared by all defenders (each acts on its own egocentric row,
defender_obs) and picks one of the robots' 9 commands. Training only:

  mappo  a critic V(state) that sees the whole court (global_state). The team
         reward is shared, so the advantage of a step is the same for every
         defender.
  ippo   the same, but each defender's critic V(o_i) sees only its own row.

PPO with clipping, GAE, running normalisation of the inputs (saved with each
checkpoint, as VecNormalize is for the raiders). The raider is frozen
(defender_env.py). Rewards: TEAM_REWARD only, nothing else; --walkout sets
the reward for a walk-out tackle (defender_env.py, r_walkout; author, 30 Oct).
--init starts from a saved defence (policy, critic and input normalisation;
a fresh optimiser) instead of from scratch (author, 30 Oct: start from run D,
which holds the scripted raider).

    python3 marl/mappo.py --algo mappo --raider main_1v3_seed42_ext --seed 0 \\
        --steps 2000000 --out results/marl/mappo_vs_ppo42_s0

Output: config.json, metrics.csv (one line per update), checkpoints/.
"""
from __future__ import annotations

import argparse
import csv
import json
import os
import sys
import time
from collections import Counter
from pathlib import Path

os.environ.setdefault("OMP_NUM_THREADS", "1")
import numpy as np                                                    # noqa: E402
import torch                                                          # noqa: E402
import torch.nn as nn                                                 # noqa: E402

sys.path.insert(0, str(Path(__file__).resolve().parent))
from defender_env import VecDefenderEnv, N_ACTIONS, diag_kwargs                    # noqa: E402

TRAIN_SEED0 = 10_000_000      # raid seeds for training, far from evaluation (50000+)


class RunningNorm:
    """Running mean / variance of an input (Welford over batches)."""

    def __init__(self, dim):
        self.mean, self.var, self.count = np.zeros(dim), np.ones(dim), 1e-4

    def update(self, x):
        x = x.reshape(-1, x.shape[-1])
        bm, bv, bc = x.mean(0), x.var(0), x.shape[0]
        d, tot = bm - self.mean, self.count + bc
        self.mean = self.mean + d * bc / tot
        self.var = (self.var * self.count + bv * bc + d ** 2 * self.count * bc / tot) / tot
        self.count = tot

    def __call__(self, x):
        return np.clip((x - self.mean) / np.sqrt(self.var + 1e-8), -10, 10).astype(np.float32)

    def state_dict(self):
        return {"mean": self.mean.tolist(), "var": self.var.tolist(), "count": self.count}

    def load_state_dict(self, d):
        self.mean, self.var, self.count = np.array(d["mean"]), np.array(d["var"]), d["count"]


def mlp(i, o, h=128, gain=1.0):
    net = nn.Sequential(nn.Linear(i, h), nn.Tanh(), nn.Linear(h, h), nn.Tanh(), nn.Linear(h, o))
    for k, m in enumerate(net):
        if isinstance(m, nn.Linear):
            nn.init.orthogonal_(m.weight, gain if k == len(net) - 1 else np.sqrt(2))
            nn.init.zeros_(m.bias)
    return net


class Agent(nn.Module):
    def __init__(self, obs_dim, state_dim, algo):
        super().__init__()
        self.algo = algo
        self.actor = mlp(obs_dim, N_ACTIONS, gain=0.01)
        self.critic = mlp(state_dim if algo == "mappo" else obs_dim, 1, gain=1.0)

    def dist(self, obs):
        return torch.distributions.Categorical(logits=self.actor(obs))

    def value(self, state, obs):
        """mappo: (B,) from the state; ippo: (B, N) from each row."""
        if self.algo == "mappo":
            return self.critic(state).squeeze(-1)
        return self.critic(obs).squeeze(-1)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--algo", choices=("mappo", "ippo"), default="mappo")
    ap.add_argument("--raider", default="main_1v3_seed42_ext",
                    help="a C1 run in results/c1/, 'scripted', or several separated "
                         "by commas (one drawn per raid)")
    ap.add_argument("--no-raider-escape", action="store_true",
                    help="frozen raiders without the robots' boundary escape")
    ap.add_argument("--no-held-touch", action="store_true",
                    help="old held rule: a still raider is held without a defender "
                         "touching it (the runs of 27-28 Oct night)")
    ap.add_argument("--walkout", type=float, default=None,
                    help="defenders' reward for a walk-out (raider over a line, no "
                         "defender touching in the last 1 s); default: +1, as any tackle")
    ap.add_argument("--init", default=None,
                    help="start from this saved defence checkpoint (.pt); default: from scratch")
    ap.add_argument("--n-def", type=int, default=3)
    ap.add_argument("--pad-to", type=int, default=None,
                    help="policy input for this many defenders; the missing ones are "
                         "phantoms (defender_env.PHANTOM_POS); default: --n-def")
    ap.add_argument("--raider-phantom", choices=("deep", "copy"), default=None,
                    help="where a frozen raider's phantom defenders are when the court has "
                         "fewer defenders than its input; default: the raider's own setting")
    ap.add_argument("--diag-range", default=None, help="every robot gets its own diagonal speed ratio per raid, uniform in LO,HI (B3: 0.35,0.85); default off")
    ap.add_argument("--diag-period", type=float, default=None, help="with --diag-range: new ratios every this many seconds; default once per raid")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--steps", type=int, default=2_000_000, help="env steps (5 Hz decisions)")
    ap.add_argument("--workers", type=int, default=6)
    ap.add_argument("--envs-per-worker", type=int, default=2)
    ap.add_argument("--rollout", type=int, default=256, help="steps per env per update")
    ap.add_argument("--epochs", type=int, default=5)
    ap.add_argument("--minibatches", type=int, default=4)
    ap.add_argument("--lr", type=float, default=3e-4)
    ap.add_argument("--gamma", type=float, default=0.99)
    ap.add_argument("--lam", type=float, default=0.95)
    ap.add_argument("--clip", type=float, default=0.2)
    ap.add_argument("--ent", type=float, default=0.01)
    ap.add_argument("--vf", type=float, default=0.5)
    ap.add_argument("--save-every", type=int, default=20, help="updates")
    ap.add_argument("--out", required=True)
    args = ap.parse_args()
    assert args.workers + 1 <= 7, "at most 7 jobs: keep one core free"

    out = Path(args.out)
    (out / "checkpoints").mkdir(parents=True, exist_ok=True)
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    torch.set_num_threads(1)

    venv = VecDefenderEnv(args.raider, args.n_def, args.workers, args.envs_per_worker,
                          seed0=TRAIN_SEED0 * (args.seed + 1), r_walkout=args.walkout,
                          pad_to=args.pad_to, raider_phantom=args.raider_phantom,
                          raider_edge_escape=not args.no_raider_escape,
                          held_needs_touch=not args.no_held_touch,
                          **diag_kwargs(args.diag_range, args.diag_period))
    E, N, T = venv.n_envs, args.n_def, args.rollout
    agent = Agent(venv.obs_dim, venv.state_dim, args.algo)
    opt = torch.optim.Adam(agent.parameters(), lr=args.lr, eps=1e-5)
    onorm, snorm = RunningNorm(venv.obs_dim), RunningNorm(venv.state_dim)
    if args.init:
        c = torch.load(args.init, map_location="cpu", weights_only=False)
        assert (c["algo"], c["obs_dim"], c["state_dim"]) == (args.algo, venv.obs_dim,
                                                            venv.state_dim), "--init does not match"
        agent.load_state_dict(c["agent"])
        onorm.load_state_dict(c["onorm"])
        snorm.load_state_dict(c["snorm"])
    (out / "config.json").write_text(json.dumps({**vars(args), "n_envs": E,
                                                 "obs_dim": venv.obs_dim,
                                                 "state_dim": venv.state_dim,
                                                 "train_seed0": TRAIN_SEED0 * (args.seed + 1)},
                                                indent=2))
    log = open(out / "metrics.csv", "w", newline="")
    w = csv.writer(log)
    w.writerow(["update", "env_steps", "raids", "touch", "tackle", "foul", "empty",
                "mean_team_return", "entropy", "value_loss", "fps", "walkout"])

    obs, state = venv.reset()
    n_updates = args.steps // (E * T)
    steps, t0 = 0, time.time()
    for upd in range(1, n_updates + 1):
        buf_o = np.zeros((T, E, N, venv.obs_dim), np.float32)
        buf_s = np.zeros((T, E, venv.state_dim), np.float32)
        buf_a = np.zeros((T, E, N), np.int64)
        buf_lp = np.zeros((T, E, N), np.float32)
        buf_r = np.zeros((T, E), np.float32)
        buf_d = np.zeros((T, E), np.float32)
        vshape = (T, E) if args.algo == "mappo" else (T, E, N)
        buf_v = np.zeros(vshape, np.float32)
        outs, rets, ep_ret = Counter(), [], np.zeros(E)
        for t in range(T):
            onorm.update(obs)
            snorm.update(state)
            o, s = onorm(obs), snorm(state)
            with torch.no_grad():
                ot, st = torch.as_tensor(o), torch.as_tensor(s)
                d = agent.dist(ot)
                a = d.sample()
                buf_lp[t] = d.log_prob(a).numpy()
                buf_v[t] = agent.value(st, ot).numpy()
            buf_o[t], buf_s[t], buf_a[t] = o, s, a.numpy()
            obs, state, r, done, outcome = venv.step(a.numpy())
            buf_r[t], buf_d[t] = r, done
            ep_ret += r
            for k in np.flatnonzero(done):
                outs[outcome[k]] += 1
                outs["walkout"] += venv.last_walkouts[k]
                rets.append(ep_ret[k])
                ep_ret[k] = 0.0
        steps += E * T
        with torch.no_grad():
            last_v = agent.value(torch.as_tensor(snorm(state)), torch.as_tensor(onorm(obs))).numpy()
        # GAE; the next value is zero after the last step of a raid
        adv = np.zeros_like(buf_v)
        gae = np.zeros_like(last_v)
        for t in reversed(range(T)):
            nv = last_v if t == T - 1 else buf_v[t + 1]
            nd = 1.0 - buf_d[t]
            if args.algo == "ippo":
                nd, r_t = nd[:, None], buf_r[t][:, None]
            else:
                r_t = buf_r[t]
            delta = r_t + args.gamma * nv * nd - buf_v[t]
            gae = delta + args.gamma * args.lam * nd * gae
            adv[t] = gae
        ret = adv + buf_v
        if args.algo == "mappo":            # shared reward: one advantage for every defender
            adv_n = np.repeat(adv[:, :, None], N, axis=2)
        else:
            adv_n = adv

        b_o = torch.as_tensor(buf_o.reshape(T * E, N, -1))
        b_s = torch.as_tensor(buf_s.reshape(T * E, -1))
        b_a = torch.as_tensor(buf_a.reshape(T * E, N))
        b_lp = torch.as_tensor(buf_lp.reshape(T * E, N))
        b_adv = torch.as_tensor(adv_n.reshape(T * E, N))
        b_ret = torch.as_tensor(ret.reshape((T * E,) + ret.shape[2:]))
        idx = np.arange(T * E)
        mb = T * E // args.minibatches
        for _ in range(args.epochs):
            np.random.shuffle(idx)
            for k in range(args.minibatches):
                j = torch.as_tensor(idx[k * mb:(k + 1) * mb])
                d = agent.dist(b_o[j])
                lp = d.log_prob(b_a[j])
                ratio = torch.exp(lp - b_lp[j])
                a_ = b_adv[j]
                a_ = (a_ - a_.mean()) / (a_.std() + 1e-8)
                pg = -torch.min(ratio * a_, torch.clamp(ratio, 1 - args.clip, 1 + args.clip) * a_).mean()
                v = agent.value(b_s[j], b_o[j])
                vl = 0.5 * ((v - b_ret[j]) ** 2).mean()
                ent = d.entropy().mean()
                loss = pg + args.vf * vl - args.ent * ent
                opt.zero_grad()
                loss.backward()
                nn.utils.clip_grad_norm_(agent.parameters(), 0.5)
                opt.step()
        fps = steps / (time.time() - t0)
        n_r = sum(outs[k] for k in ("touch", "tackle", "foul", "empty"))
        w.writerow([upd, steps, n_r, *(outs[k] for k in ("touch", "tackle", "foul", "empty")),
                    f"{np.mean(rets):.3f}" if rets else "", f"{ent.item():.3f}",
                    f"{vl.item():.4f}", f"{fps:.0f}", outs["walkout"]])
        log.flush()
        if upd % args.save_every == 0 or upd == n_updates:
            torch.save({"agent": agent.state_dict(), "onorm": onorm.state_dict(),
                        "snorm": snorm.state_dict(), "algo": args.algo,
                        "obs_dim": venv.obs_dim, "state_dim": venv.state_dim,
                        "env_steps": steps},
                       out / "checkpoints" / f"{args.algo}_{steps}_steps.pt")
        if n_r:
            print(f"upd {upd}/{n_updates} steps {steps} raids {n_r} "
                  f"touch {outs['touch']} tackle {outs['tackle']} foul {outs['foul']} "
                  f"empty {outs['empty']} (walk-out {outs['walkout']}) "
                  f"ret {np.mean(rets):+.2f} fps {fps:.0f}", flush=True)
    venv.close()
    log.close()


if __name__ == "__main__":
    main()
