"""
view_match.py — watch one full C4 match in the PyBullet window
==============================================================

Plays one match of marl/match.py (same setup, match number and seeds give
the same match as in the saved results) in real time, with the score board:
raid number, who raids, score, players on the mat. The defenders are tagged
with their player numbers (1-3) and the raider with R and its number. The
court has as many defenders as the defending team has on the mat, so the
window is rebuilt when that number changes. Close the window or Ctrl+C to stop.

    python3 marl/view_match.py --setup learned-learned --match 0
    python3 marl/view_match.py --setup learned-scripted --match 4 --speed 2
"""
from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
from match import SETUPS, GATES, GATE_NAME, SEED0, Player, Court, play_raid  # noqa: E402
from match_rules import Match, RAIDS_PER_TEAM                         # noqa: E402
import pybullet as p                                                  # noqa: E402

DARK = [0.1, 0.1, 0.1]
RED = [0.75, 0.1, 0.1]
X0, Y0 = -1.3, 2.25          # board above the defenders' end line
CLIENT = {"id": 0}           # the window's PyBullet client (changes when rebuilt)


def text(t, pos, size=1.3, colour=DARK, parent=None):
    kw = dict(textColorRGB=colour, textSize=size, physicsClientId=CLIENT["id"])
    if parent is not None:
        kw["parentObjectUniqueId"] = parent
    return p.addUserDebugText(t, pos, **kw)


def replace(uid, t, pos, size=1.3, colour=DARK):
    return p.addUserDebugText(t, pos, textColorRGB=colour, textSize=size,
                              replaceItemUniqueId=uid, physicsClientId=CLIENT["id"])


NICE = {"touch": "TOUCH", "foul": "DEFENDER FOUL", "held": "SUPER TACKLE (held)",
        "pushed_out": "SUPER TACKLE (pushed out)", "walked_out": "RAIDER WALKED OUT",
        "timer": "RAIDER OUT (30 s)", "empty": "EMPTY RAID"}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--setup", choices=list(SETUPS), default="learned-learned")
    ap.add_argument("--match", type=int, default=0, help="match number (as in the results)")
    ap.add_argument("--speed", type=float, default=1.0, help="1 = real time")
    args = ap.parse_args()
    ka, kb = SETUPS[args.setup]
    kinds = {k: Player(k) for k in {ka, kb}}
    players = {"A": kinds[ka], "B": kinds[kb]}
    m = args.match
    gate = GATES[m % len(GATES)]
    names = {"A": f"A ({ka})", "B": f"B ({kb}"
             + (f", gates {GATE_NAME[gate]})" if kb == "scripted" else ")")}
    court, game = Court(render=True), Match("A", "B", a_first=(m % 2 == 0))
    print(f"match {m}: team {names['A']} vs team {names['B']}, "
          f"{RAIDS_PER_TEAM} raids each", flush=True)
    try:
        while not game.done():
            att, dfn, raider, defenders = game.raid_turn()
            k, j = len(defenders), len(game.log)
            sc = {t.name: t.points for t in game.teams}
            mat = {t.name: len(t.on_mat) for t in game.teams}
            board = (f"Raid {j + 1}/{2 * RAIDS_PER_TEAM}   team {names[att.name]} raids   "
                     f"score A {sc['A']} - B {sc['B']}   on mat A {mat['A']}, B {mat['B']}")
            state = {"ids": None, "stat": None, "t0": None}

            def on_step(e):
                if state["ids"] is None:              # first decision of the raid
                    CLIENT["id"] = e._client
                    p.removeAllUserDebugItems(physicsClientId=e._client)
                    try:
                        p.configureDebugVisualizer(p.COV_ENABLE_GUI, 0, physicsClientId=e._client)
                    except Exception:
                        pass
                    text(board, [X0, Y0, 0.6])
                    state["stat"] = text("", [X0, Y0, 0.42], size=1.2)
                    for i, did in enumerate(e.defender_ids):
                        text(str(defenders[i] + 1), [0, 0, 0.25], size=2.0, parent=did)
                    text(f"R{raider + 1}", [0, 0, 0.25], size=2.0, colour=RED,
                         parent=e.raider_id)
                    state["ids"], state["t0"] = True, time.time() - e._step * e.dt / args.speed
                state["stat"] = replace(
                    state["stat"], f"t = {e._step * e.dt:4.1f} s   contact = "
                    f"{'yes' if e._flags['contact'] else 'no'}", [X0, Y0, 0.42], size=1.2)
                lag = state["t0"] + e._step * e.dt / args.speed - time.time()
                if lag > 0:
                    time.sleep(lag)

            kind, idx, info = play_raid(court, players[att.name], players[dfn.name], k,
                                        SEED0 + 100 * m + j, gate, on_step)
            ev = game.record(raider, kind, [defenders[i] for i in idx])
            res = NICE[kind]
            if ev["scorer"]:
                res += f":  team {ev['scorer']} +{ev['points']}"
            if ev["out"]:
                res += "   out: " + ", ".join(f"{t}{q + 1}" for t, q in ev["out"])
            if ev["revived"]:
                res += f"   back: {len(ev['revived'])}"
            if ev["all_out"]:
                res += f"   ALL OUT team {ev['all_out']} (+{ev['bonus']})"
            text(res, [X0, Y0, 0.24], size=1.5, colour=RED)
            print(f"raid {ev['raid']:>2} team {ev['raiding']} vs {k}: {res}   "
                  f"score A {ev['score']['A']} - B {ev['score']['B']}", flush=True)
            time.sleep(3.0 / args.speed)
        r = game.result()
        final = (f"FINAL  A {r['A']} - B {r['B']}   "
                 + ("draw" if r["winner"] == "draw" else f"team {r['winner']} wins"))
        text(final, [X0, Y0, 0.06], size=1.8, colour=RED)
        print(final, flush=True)
        time.sleep(8.0)
    except KeyboardInterrupt:
        pass
    except p.error:
        print("window closed")
    finally:
        try:
            court.close()
        except Exception:
            pass


if __name__ == "__main__":
    main()
