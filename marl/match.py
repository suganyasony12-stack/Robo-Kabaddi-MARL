"""
match.py — C4 matches in the simulator: two teams of 3, taking turns to raid
============================================================================

The author's match rules (5 Nov 2026, revision_experiments/C4_MATCH_RULES.md,
scoring in match_rules.py). Each raid is one raid of RoboKabaddiEnv under the
C3 rules (CORRECTED_CONFIG, defenders' fouls, held needs a defender touch),
with as many defenders as the defending team has on the mat. Nothing in the
simulator or the trained policies is changed.

Teams
  learned   the accepted pairs: for k defenders on the mat the defence
            trained for k, and against k defenders the raider trained for k
            (each raider with its own phantom setting, run_info)
  scripted  the scripted raider (HardwareRaider, as in C3) and the scripted
            gate defenders; one gate style per match, v6..v11 in turn

Raid result -> rules kind (match_rules.KINDS):
  touch       -> touch; the defender nearest the raider when contact was
                 first counted is out
  foul        -> foul; the defenders over a line (_defender_fouls) are out
  tackle      -> raider outside the court: walked_out if no defender touched
                 it in the last WALKOUT_STEPS decisions (as DefenderEnv),
                 else pushed_out; inside the court: held if the raid time
                 had not run out (as eval_defenders), else timer
  empty       -> empty

Seeded: raid j of match m starts from seed SEED0 + 100 m + j. Team A raids
first in even matches, team B in odd ones.

    python3 marl/match.py --setup learned-learned --matches 200 --out results/marl/match/ll.json
    python3 marl/match.py --setup learned-scripted --matches 4 --verbose
"""
from __future__ import annotations

import argparse
import json
import os
import sys
from collections import Counter
from multiprocessing import get_context
from pathlib import Path

os.environ.setdefault("OMP_NUM_THREADS", "1")
import numpy as np                                                    # noqa: E402

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
from match_rules import Match, midline_touch                          # noqa: E402

SEED0 = 80_000
GATES = ("keepout", "v7", "v8", "v9", "v10", "v11")    # v6..v11
GATE_NAME = dict(zip(GATES, ("v6", "v7", "v8", "v9", "v10", "v11")))
R = "results/marl"
F = "checkpoints/mappo_3999744_steps.pt"
# accepted pairs (author, 4 Nov): defenders on the mat -> (defence, raider)
PAIRS = {3: (f"{R}/alt_wo_init/defence_s0/{F}", f"{R}/alt_wo_r2/r2_raider_s42"),
         2: (f"{R}/two_def/defence_s1/{F}", f"{R}/two_def_r2/raider_s123"),
         1: (f"{R}/one_def_copy/defence_s0/{F}", f"{R}/one_def_copy/raider_s456")}
# pairs for the robot runs of 13 Nov (author): trained with varying diagonal
# speed (diagvar, 11-12 Nov); 1 defender's raider with the touch input
PAIRS_13NOV = {3: (f"{R}/diagvar/d3/defence_s1/{F}", f"{R}/alt_wo_r2/r2_raider_s42"),
               2: (f"{R}/diagvar/d2/defence_s0/{F}", f"{R}/diagvar/d2/raider2_s42"),
               1: (f"{R}/diagvar/d1/defence_s1/{F}", f"{R}/touchobs/d1/raider_s123")}
PAIR_SETS = {"accepted": PAIRS, "13nov": PAIRS_13NOV}
SETUPS = {"learned-learned": ("learned", "learned"),
          "learned-scripted": ("learned", "scripted"),
          "scripted-scripted": ("scripted", "scripted")}


class Player:
    """Policies for one team type, loaded once per worker."""

    def __init__(self, kind, pairs=None, rule=None):
        from defender_env import FrozenRaider, make_raider, ROOT
        self.kind = kind                             # rule: defender_env.deep_rule (author, 13 Nov)
        pairs = pairs or PAIRS                       # default: the accepted pairs
        if kind == "learned":
            from eval_defenders import load
            self.defence = {k: load(str(ROOT / d)) for k, (d, _) in pairs.items()}
            self.raider = {k: FrozenRaider(str(ROOT / r), rule=rule) for k, (_, r) in pairs.items()}
        else:
            self.defence = None
            self.raider = {k: make_raider("scripted", k) for k in pairs}


class Court:
    """One RoboKabaddiEnv per (defence type, defenders, gate style).
    render=True (view_match.py): one PyBullet window at a time, so the
    court is rebuilt when the defenders on the mat change."""

    def __init__(self, render=False, diag=None, dkw=None):
        self.envs, self.render, self.diag, self.dkw = {}, render, diag, dkw or {}

    def get(self, def_kind, k, gate):
        from envs.kabaddi_env import RoboKabaddiEnv, CORRECTED_CONFIG
        key = (def_kind, k, gate if def_kind == "scripted" else None)
        if self.render and key not in self.envs:
            self.close()
            self.envs = {}
        if key not in self.envs:
            kw = dict(CORRECTED_CONFIG, n_defenders=k, defender_fouls=True,
                      held_needs_touch=True, raider_edge_escape=False,
                      defender_type="learned" if def_kind == "learned" else "gate")
            if def_kind == "scripted":
                kw["gate_style"] = gate
            if self.diag:
                kw["diagonal_speed_ratio"] = self.diag
            kw.update(self.dkw)
            self.envs[key] = RoboKabaddiEnv(**kw, render=self.render)
        return self.envs[key]

    def close(self):
        for e in self.envs.values():
            e.close()


def play_raid(court, att, dfn, k, seed, gate, on_step=None, midline_rule=True):
    """One raid: att's raider against k of dfn's defenders. Returns
    (kind, index of the defender out for a touch / those over for a foul, info).
    on_step(env) is called after every decision (view_match.py), or None.
    midline_rule: the author's midline touch (match_rules.midline_touch, 11 Nov);
    False = every defender over the midline is a foul (results before 11 Nov)."""
    from defender_env import pad_defender_obs, WALKOUT_STEPS
    from envs.kabaddi_env import COURT_W, COURT_L, MIDLINE_Y
    e = court.get(dfn.kind, k, gate)
    raider = att.raider[k]
    # the robots' boundary steering only for raiders that have not learned
    # the lines (as DefenderEnv): the scripted raider
    e.raider_edge_escape = not getattr(raider, "learned_lines", False)
    obs, _ = e.reset(seed=seed)
    obs = raider.reset(e, obs)
    act = dfn.defence[k] if dfn.defence else None
    first, touching, info = None, [], {}
    while True:
        if act is not None:
            e.set_defender_actions(act(pad_defender_obs(e, e.defender_obs(), act.n_in)))
        obs, _, term, trunc, info = e.step(raider.act(e, obs))
        if e._flags["contact"] and first is None:
            r = e._get_pos(e.raider_id)
            d = [np.hypot(*(e._get_pos(i)[:2] - r[:2])) for i in e.defender_ids]
            first = int(np.argmin(d))
        touching.append(e._defender_touching())
        if on_step is not None:
            on_step(e)
        if term or trunc:
            break
    out = info.get("outcome", "empty")
    pos = e._get_pos(e.raider_id)
    if out == "touch":
        return "touch", [first], dict(t=e._step * e.dt)
    if out == "foul":
        over = e._defender_fouls()
        if midline_rule:
            j = midline_touch(pos[:2], [e._get_pos(d)[:2] for d in e.defender_ids], over,
                              MIDLINE_Y, e.contact_dist)
            if j is not None:
                return "touch", [j], dict(t=e._step * e.dt, via="midline")
        return "foul", over, dict(t=e._step * e.dt)
    if out == "tackle":
        if abs(pos[0]) > COURT_W / 2 or abs(pos[1]) > COURT_L / 2:
            kind = "pushed_out" if any(touching[-WALKOUT_STEPS:]) else "walked_out"
        else:
            start = e._raid_start_step
            kind = ("held" if start is not None and e._step - start < e._raid_steps
                    else "timer")
        return kind, [], dict(t=e._step * e.dt)
    return "empty", [], dict(t=e._step * e.dt)


def play_matches(job):
    setup, matches, verbose, *rest = job
    diag = rest[0] if rest else None
    midline_rule = rest[1] if len(rest) > 1 else True
    dkw = rest[2] if len(rest) > 2 else {}
    ka, kb = SETUPS[setup]
    kinds = {"learned": Player("learned")} if "learned" in (ka, kb) else {}
    if "scripted" in (ka, kb):
        kinds["scripted"] = Player("scripted")
    players = {"A": kinds[ka], "B": kinds[kb]}
    court, out = Court(diag=diag, dkw=dkw), []
    for m in matches:
        gate = GATES[m % len(GATES)]
        game = Match("A", "B", a_first=(m % 2 == 0))
        while not game.done():
            att, dfn, raider, defenders = game.raid_turn()
            k = len(defenders)
            j = len(game.log)
            kind, idx, info = play_raid(court, players[att.name], players[dfn.name], k,
                                        SEED0 + 100 * m + j, gate, midline_rule=midline_rule)
            ev = game.record(raider, kind, [defenders[i] for i in idx])
            ev["t"] = round(info["t"], 1)
            if info.get("via"):
                ev["via"] = info["via"]
            if verbose:
                print(f"  match {m} raid {ev['raid']:>2} {ev['raiding']} vs {k}: "
                      f"{kind:<10} score {ev['score']} on mat {ev['on_mat']}"
                      + (f" ALL OUT {ev['all_out']}" if ev["all_out"] else ""), flush=True)
        out.append(dict(match=m, setup=setup, team_A=ka, team_B=kb,
                        gate=GATE_NAME[gate] if "scripted" in (ka, kb) else None,
                        a_first=m % 2 == 0, diagonal_speed_ratio=diag or 1.0,
                        **game.result(), raids=game.log))
    court.close()
    return out


def summary(res):
    ka, kb = res[0]["team_A"], res[0]["team_B"]
    n = len(res)
    w = Counter(r["winner"] for r in res)
    pts = {t: [r[t] for r in res] for t in "AB"}
    print(f"\n{res[0]['setup']}: team A = {ka}, team B = {kb}, {n} matches")
    print(f"  wins A {w['A']}, B {w['B']}, draws {w['draw']}")
    for t in "AB":
        print(f"  team {t}: points mean {np.mean(pts[t]):.1f} (median {np.median(pts[t]):.0f})")
    raids = [e for r in res for e in r["raids"]]
    for t in "AB":
        mine = [e for e in raids if e["raiding"] == t]
        c = Counter(e["kind"] for e in mine)
        by_k = Counter(e["n_def"] for e in mine)
        print(f"  team {t} raids ({len(mine)}): " + ", ".join(
            f"{k} {100 * c[k] / len(mine):.1f}%" for k in
            ("touch", "foul", "held", "pushed_out", "walked_out", "timer", "empty"))
              + "  | against 3/2/1 defenders: "
              + "/".join(str(by_k[k]) for k in (3, 2, 1)))
    ao = Counter(e["all_out"] for e in raids if e["all_out"])
    rv = {t: sum(len(e["revived"]) for e in raids if e["scorer"] == t) for t in "AB"}
    print(f"  all outs: A {ao['A']}, B {ao['B']}; players revived: A {rv['A']}, B {rv['B']}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--setup", choices=list(SETUPS), required=True)
    ap.add_argument("--matches", type=int, default=200)
    ap.add_argument("--first", type=int, default=0, help="first match number")
    ap.add_argument("--procs", type=int, default=7)
    ap.add_argument("--verbose", action="store_true", help="print every raid")
    ap.add_argument("--midline-foul", action="store_true",
                    help="old rule: a defender over the midline is always a foul (results before 11 Nov)")
    ap.add_argument("--diag-ratio", type=float, default=None,
                    help="diagonal speed / straight speed for every robot (B3 measured 0.58); default 1.0")
    ap.add_argument("--diag-range", default=None, help="every robot gets its own diagonal speed ratio per raid, uniform in LO,HI (B3: 0.35,0.85); default off")
    ap.add_argument("--diag-period", type=float, default=None, help="with --diag-range: new ratios every this many seconds; default once per raid")
    ap.add_argument("--out", help="JSON file to write (optional)")
    args = ap.parse_args()
    assert args.procs <= 7, "at most 7 jobs: keep one core free"
    ms = list(range(args.first, args.first + args.matches))
    procs = min(args.procs, len(ms))
    from defender_env import diag_kwargs
    dkw = diag_kwargs(args.diag_range, args.diag_period)
    jobs = [(args.setup, ms[i::procs], args.verbose, args.diag_ratio, not args.midline_foul, dkw)
            for i in range(procs)]
    with get_context("spawn").Pool(procs) as pool:
        res = sorted((r for part in pool.map(play_matches, jobs) for r in part),
                     key=lambda r: r["match"])
    summary(res)
    if args.out:
        Path(args.out).parent.mkdir(parents=True, exist_ok=True)
        Path(args.out).write_text(json.dumps(dict(rules="revision_experiments/C4_MATCH_RULES.md",
                                                  pairs=PAIRS, seed0=SEED0,
                                                  diagonal_speed_ratio=args.diag_ratio or 1.0,
                                                  midline_touch=not args.midline_foul,
                                                  diag_range=args.diag_range, diag_period=args.diag_period,
                                                  matches=res),
                                              indent=1, default=float))
        print(f"\nwritten to {args.out}")


if __name__ == "__main__":
    main()
