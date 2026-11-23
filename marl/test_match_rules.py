"""Checks of marl/match_rules.py against the author's examples (5 Nov).
    python3 marl/test_match_rules.py"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from match_rules import Match  # noqa: E402

ok = 0


def check(cond, msg):
    global ok
    assert cond, msg
    ok += 1


# Author's example: A touches B's defender #1 and gets back
m = Match()
att, dfn, r, d = m.raid_turn()
check((att.name, len(d)) == ("A", 3), "A raids against 3")
e = m.record(r, "touch", [d[0]])
check(e["score"] == {"A": 1, "B": 0}, "A gets 1 point")
check(e["on_mat"] == {"A": 3, "B": 2}, "A 3 on the mat, B 2")
# Scenario 1: B raids next and touches; its out player comes back
att, dfn, r, d = m.raid_turn()
check((att.name, len(d)) == ("B", 3), "B's turn, against 3")
e = m.record(r, "touch", [d[1]])
check(e["score"] == {"A": 1, "B": 1} and e["on_mat"] == {"A": 2, "B": 3}, "B revives")
check(e["revived"] == [0], "first out first back")

# Scenario 2: B empty raid, then super tackle with 2 on the mat
m = Match()
_, _, r, d = m.raid_turn(); m.record(r, "touch", [d[0]])          # A touch
_, _, r, d = m.raid_turn(); e = m.record(r, "empty")               # B empty
check(e["score"] == {"A": 1, "B": 0} and e["on_mat"] == {"A": 3, "B": 2}, "empty: nothing")
att, dfn, r, d = m.raid_turn()
check(len(d) == 2, "A raids against 2")
e = m.record(r, "held")
check(e["score"] == {"A": 1, "B": 2}, "super tackle 2 points")
check(e["on_mat"] == {"A": 2, "B": 3}, "only 1 on the bench, so 1 revived; A raider out")

# walk-out / timer: 1 point, raider out
m = Match()
_, _, r, _ = m.raid_turn(); e = m.record(r, "walked_out")
check(e["score"] == {"A": 0, "B": 1} and e["on_mat"] == {"A": 2, "B": 3}, "walk-out 1 point")

# foul: 1 point per defender over, each out
m = Match()
_, _, r, d = m.raid_turn(); e = m.record(r, "foul", d[:2])
check(e["score"] == {"A": 2, "B": 0} and e["on_mat"] == {"A": 3, "B": 1}, "2 fouls")

# all out: B loses its last player -> A +2, B all back
m = Match()
_, _, r, d = m.raid_turn(); m.record(r, "foul", d[:2])             # A 2, B on 1
_, _, r, d = m.raid_turn(); m.record(r, "empty")                   # B empty
_, _, r, d = m.raid_turn(); e = m.record(r, "touch", d)            # A touches the last
check(e["all_out"] == "B" and e["bonus"] == 2, "all out")
check(e["score"] == {"A": 5, "B": 0} and e["on_mat"] == {"A": 3, "B": 3}, "+2 and B all back")

# all out of the raiding team: its last raider tackled
m = Match()
_, _, r, d = m.raid_turn(); m.record(r, "empty")                   # A
_, _, r, d = m.raid_turn(); m.record(r, "foul", d)                 # B fouls 3 -> A all out
check(m.log[-1]["all_out"] == "A" and m.log[-1]["score"] == {"A": 0, "B": 5}, "3 fouls = all out")

# raiders take turns; match length
m = Match()
seen = []
while not m.done():
    att, _, r, _ = m.raid_turn()
    if att.name == "A":
        seen.append(r)
    m.record(r, "empty")
check(len(m.log) == 20 and seen[:4] == [0, 1, 2, 0], "20 raids, raiders take turns")
check(m.result()["winner"] == "draw", "0-0 draw")
# midline touch (author, 11 Nov): defender over the midline in contact with the raider
from match_rules import midline_touch  # noqa: E402
R = (0.10, 0.05)                                     # raider just past the midline (y 0)
D = [(0.12, -0.10), (0.9, 1.5), (-0.8, 1.6)]         # defender 1 over the midline, 0.15 m away
check(midline_touch(R, D, [0], 0.0, 0.25) == 0, "over the midline, touching: touch")
check(midline_touch(R, [(0.12, -0.30)] + D[1:], [0], 0.0, 0.25) is None, "over the midline, 0.35 m away: foul")
check(midline_touch(R, D[:1] + [(1.4, 1.5)] + D[2:], [0, 1], 0.0, 0.25) is None,
      "another defender over the side line too: foul")
check(midline_touch(R, [(0.9, 2.1)] + D[1:], [0], 0.0, 0.25) is None, "over the end line: foul")
check(midline_touch(R, [(0.12, -0.10), (0.0, -0.05)] + D[2:], [0, 1], 0.0, 0.25) == 1,
      "two over the midline, both touching: touch, the nearest out")
print(f"{ok} checks passed")
