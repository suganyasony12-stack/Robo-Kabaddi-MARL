"""
match_rules.py — the author's C4 match rules (5 Nov 2026), no simulator
======================================================================

Scoring only, so the same rules serve the simulator (marl/match.py) and,
later, the robot runner. Rules in revision_experiments/C4_MATCH_RULES.md:

- two teams of 3 players (robots) on the mat (court); teams take turns to
  send one raider; 10 raids per team
- touch (contact, then back over the midline): raiding team 1 point and the
  touched defender is out; two defenders touched still count as 1 point,
  one out
- defender foul (over the midline or a court line): raiding team 1 point per
  defender over, each of them out (earlier rule, defender_fouls)
- tackle with contact (held, pushed out): super tackle, defending team
  2 points (3 or fewer on the mat, always true with 3 robots); raider out
- raider out with no tackle (walked out over a line, or the 30 s ran out):
  defending team 1 point (proposed, see C4_MATCH_RULES.md); raider out
- empty: no points, nobody out
- midline touch (author, 11 Nov): a defender who crosses the midline while
  in contact with the raider takes the raider home with him: the raider has
  touched him and reached the midline through him, so the raid is a touch
  (1 point, that defender out), not a defender foul. Contact = centres within
  the touch distance (0.25 m), as for any touch. If another defender is over
  a line too, the raid stays a foul (1 point per defender over). Old rule:
  midline_touch=False.
- revival: every point brings back one out player of the scoring team,
  first out first back, up to the players on the bench
- all out: the other team gets 2 bonus points (which revive as any points)
  and the whole team comes back on the mat
"""
from __future__ import annotations

from dataclasses import dataclass, field

TEAM_SIZE = 3
RAIDS_PER_TEAM = 10
ALL_OUT_BONUS = 2

# raid kind -> (points, who scores, raider out?)
#   who scores: "raid" = raiding team, "def" = defending team, None = nobody
KINDS = {
    "touch":      (1, "raid", False),
    "foul":       (None, "raid", False),   # 1 per defender over
    "held":       (2, "def", True),        # super tackle
    "pushed_out": (2, "def", True),        # super tackle
    "walked_out": (1, "def", True),        # raider out, no tackle
    "timer":      (1, "def", True),        # raider out, no tackle
    "empty":      (0, None, False),
}


def midline_touch(raider_xy, defenders_xy, over, mid_y, contact_dist):
    """Author's midline-touch rule (module doc). over: indices of the
    defenders over a line. Returns the index of the defender who takes the
    raider home (nearest to the raider), or None (the raid stays a foul)."""
    rx, ry = raider_xy
    home = [i for i in over if defenders_xy[i][1] < mid_y
            and ((defenders_xy[i][0] - rx) ** 2 + (defenders_xy[i][1] - ry) ** 2) ** 0.5 <= contact_dist]
    if not home or len(home) != len(over):
        return None
    return min(home, key=lambda i: (defenders_xy[i][0] - rx) ** 2 + (defenders_xy[i][1] - ry) ** 2)


@dataclass
class Team:
    name: str
    on_mat: list = field(default_factory=lambda: list(range(TEAM_SIZE)))
    bench: list = field(default_factory=list)      # first out first back
    points: int = 0
    next_raider: int = 0                           # turn order on the mat

    def raider(self):
        """The player who raids next: players on the mat take turns."""
        p = self.on_mat[self.next_raider % len(self.on_mat)]
        self.next_raider += 1
        return p

    def put_out(self, players):
        for p in players:
            self.on_mat.remove(p)
            self.bench.append(p)

    def revive(self, n):
        back = self.bench[:n]
        self.bench = self.bench[n:]
        self.on_mat += back
        return back


class Match:
    """Keeps the score. Call raid_turn() for who raids against how many
    defenders, play the raid, then record() its result."""

    def __init__(self, name_a="A", name_b="B", a_first=True):
        self.teams = [Team(name_a), Team(name_b)]
        self.turn = 0 if a_first else 1
        self.log = []

    def done(self):
        return len(self.log) >= 2 * RAIDS_PER_TEAM

    def raid_turn(self):
        """(raiding team, defending team, raider player, defenders on the mat)"""
        att, dfn = self.teams[self.turn], self.teams[1 - self.turn]
        return att, dfn, att.raider(), list(dfn.on_mat)

    def record(self, raider, kind, out_defenders=()):
        """kind: a KINDS key. out_defenders: the defending players who are
        out (the touched one for a touch; those over a line for a foul)."""
        assert kind in KINDS, kind
        att, dfn = self.teams[self.turn], self.teams[1 - self.turn]
        n_att, n_def = len(att.on_mat), len(dfn.on_mat)
        pts, scorer, raider_out = KINDS[kind]
        out_defenders = list(out_defenders)
        if kind == "touch":
            assert len(out_defenders) == 1, "a touch puts out one defender"
        elif kind == "foul":
            assert out_defenders, "a foul needs the defenders over a line"
            pts = len(out_defenders)
        else:
            assert not out_defenders, f"{kind}: no defender is out"
        # outs first, then the scoring team's revivals, then the all-out check
        dfn.put_out(out_defenders)
        if raider_out:
            att.put_out([raider])
        revived, bonus, all_out = [], 0, None
        if scorer:
            team = att if scorer == "raid" else dfn
            team.points += pts
            revived = team.revive(pts)
        for loser, winner in ((dfn, att), (att, dfn)):
            if not loser.on_mat:
                all_out, bonus = loser.name, ALL_OUT_BONUS
                winner.points += bonus
                revived += winner.revive(bonus)
                loser.on_mat, loser.bench = sorted(loser.bench), []
        self.log.append(dict(
            raid=len(self.log) + 1, raiding=att.name, raider=raider, kind=kind,
            n_att=n_att, n_def=n_def, points=pts if scorer else 0, scorer=(
                att.name if scorer == "raid" else dfn.name if scorer else None),
            out=[(dfn.name, p) for p in out_defenders] + ([(att.name, raider)] if raider_out else []),
            revived=revived, all_out=all_out, bonus=bonus,
            score={t.name: t.points for t in self.teams},
            on_mat={t.name: len(t.on_mat) for t in self.teams}))
        self.turn = 1 - self.turn
        return self.log[-1]

    def result(self):
        a, b = self.teams
        return {a.name: a.points, b.name: b.points,
                "winner": a.name if a.points > b.points else b.name if b.points > a.points
                else "draw"}
