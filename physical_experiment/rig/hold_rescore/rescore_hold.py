"""
rescore_hold.py -- the 13 Nov robot raids re-scored with a shorter hold (author, 14 Nov)
======================================================================================

For comparison only (author: whether it goes in the paper is decided later).
On the robots the raider was often held and then slipped away while the
defenders pushed; the 4 s hold may be too long for real robots. This script
re-scores the logged raids (steps.csv, every 0.05 s) with another hold time.

Hold rule, as RoboKabaddiEnv._static_tackle (outcome_rules 'physical',
held_needs_touch): after the raider has crossed the midline and while it is in
the defenders' half with a defender touching it, a raid ends 'held' (a tackle)
once the raider has stayed within STATIC_DIST (5 cm) of a reference point for
more than HOLD seconds; the reference resets when it moves further or no
defender touches. 'Touching' on the robots came from the shadow simulator's box
contact with the camera yaw, which is not logged; here a defender touches when
the centre distance is at most TOUCH_M. Default 0.19 m: re-scoring with the 4 s
hold then reproduces all 180 logged results (0.18-0.19 m do; 0.20-0.24 m add
7-12 'held' raids the runner did not see, 14 Nov). The script first re-scores
with 4 s and reports how well that reproduces the logged results.

A raid that would have been held earlier than it really ended becomes 'held'
at that moment; otherwise its logged result stands. Stopped raids are left
out; for every case and seed the last finished raid is used (the reruns).

    python3 physical_experiment/rig/hold_rescore/rescore_hold.py --hold 2.0
"""
import argparse
import csv
import glob
import json
import math
from collections import Counter, defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
DATA = ROOT / "physical_experiment/match_data"
STATIC_DIST, DT, MID = 0.05, 0.05, 2.0
SUPERSEDED = {("session_2026-11-13_171426", "50005")}   # Empty from a 4 mm wobble, run again with the 3 cm margin


def case_key(r):
    return (int(r["n_def"]), r["raider"], r["defence"], r["gate"] if r["defence"] == "scripted" else "",
            r.get("raider_rule", "") if r["raider"] == "learned" else "")


def load():
    """case -> seed -> (session, raids.csv row, steps of that raid); last finished raid per seed."""
    cases = defaultdict(dict)
    for d in sorted(glob.glob(str(DATA / "session_2026-11-13_*"))):
        try:
            R = list(csv.DictReader(open(d + "/raids.csv")))
            S = list(csv.DictReader(open(d + "/steps.csv")))
        except FileNotFoundError:
            continue
        if not R or R[0].get("fake") == "1":
            continue
        name = Path(d).name
        for r in R:
            if r["kind"] == "stopped" or (name, r["seed"]) in SUPERSEDED:
                continue
            cases[case_key(r)][int(r["seed"])] = (name, r, [x for x in S if x["raid"] == r["raid"]])
    return cases


def held_at(steps, k, hold, touch_m, still=STATIC_DIST):
    """Time (s from the start) at which the hold rule fires, or None."""
    started, ref, since = False, None, None
    need = int(round(hold / DT))
    for i, x in enumerate(steps):
        started |= x["midline"] == "1"
        rx, ry = float(x["raider_x"]), float(x["raider_y"])
        touching = any(math.hypot(rx - float(x[f"d{j}_x"]), ry - float(x[f"d{j}_y"])) <= touch_m
                       for j in range(1, k + 1))
        if not started or ry <= MID or not touching:
            ref = None
            continue
        if ref is None or math.hypot(rx - ref[0], ry - ref[1]) > still:
            ref, since = (rx, ry), i
            continue
        if i - since > need:
            return float(x["t_s"])
    return None


def rescore(cases, hold, touch_m, still=STATIC_DIST):
    out = {}
    for key, seeds in cases.items():
        rows = []
        for seed, (sess, r, st) in sorted(seeds.items()):
            t = held_at(st, key[0], hold, touch_m, still)
            end = float(r["t_s"])
            kind = "held" if t is not None and t < end - 1e-6 else r["kind"]
            rows.append(dict(seed=seed, session=sess, logged=r["kind"], logged_t=end,
                             rescored=kind, held_t=t))
        out[key] = rows
    return out


SIDE = {"touch": "raider", "foul": "raider", "empty": "empty"}


def points(rows, field):
    return sum(SIDE.get(x[field], "def") == "raider" for x in rows)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--hold", type=float, default=2.0)
    ap.add_argument("--touch-m", type=float, default=0.19)
    ap.add_argument("--still", type=float, default=STATIC_DIST,
                    help="m the raider may move and still count as held (author, 14 Nov: 0.10, 0.15); default 0.05")
    a = ap.parse_args()
    cases = load()
    # 1. check: 4 s reproduces the logged results?
    chk = rescore(cases, 4.0, a.touch_m)
    n = same = 0; diffs = Counter()
    for rows in chk.values():
        for x in rows:
            n += 1
            ok = x["rescored"] == x["logged"]
            same += ok
            if not ok:
                diffs[(x["logged"], x["rescored"])] += 1
    print(f"check, 4 s hold re-scored: {same} of {n} raids as logged; differences {dict(diffs)}")
    logged_held = sum(x["logged"] == "held" for rows in chk.values() for x in rows)
    print(f"  logged 'held' raids: {logged_held}")
    # 2. the new hold time
    res = rescore(cases, a.hold, a.touch_m, a.still)
    print(f"\nhold {a.hold:g} s within {100 * a.still:g} cm (touching = centre distance <= {a.touch_m} m): raider points of the raids per case")
    print(f"  {'case':<48}{'n':>3}{'logged':>8}{'rescored':>10}  changed")
    table = []
    for key in sorted(res):
        rows = res[key]
        ch = [f"{x['seed']} {x['logged']}->held@{x['held_t']:.1f}s" for x in rows if x["rescored"] != x["logged"]]
        lab = f"{key[0]} def, {key[1]} raider{' + rule' if key[4] else ''} vs {key[2]}{' ' + key[3] if key[3] else ''}"
        print(f"  {lab:<48}{len(rows):>3}{points(rows, 'logged'):>8}{points(rows, 'rescored'):>10}  {len(ch)}")
        table.append(dict(case=lab, n=len(rows), points_logged=points(rows, "logged"),
                          points_rescored=points(rows, "rescored"),
                          held_logged=sum(x["logged"] == "held" for x in rows),
                          held_rescored=sum(x["rescored"] == "held" for x in rows), changed=ch, raids=rows))
    out = Path(__file__).with_name(f"rescore_hold_{a.hold:g}s" + ("" if a.still == STATIC_DIST else f"_{100 * a.still:g}cm") + ".json")
    out.write_text(json.dumps(dict(hold_s=a.hold, touch_m=a.touch_m, still_m=a.still, check_4s=dict(same=same, n=n,
                                   differences={f"{k[0]}->{k[1]}": v for k, v in diffs.items()}),
                                   cases=table), indent=1, default=str))
    print(f"\nwritten {out}")


if __name__ == "__main__":
    main()
