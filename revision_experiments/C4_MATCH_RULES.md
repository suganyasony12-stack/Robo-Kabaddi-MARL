# C4 match rules (author, 5 Nov 2026)

Players = robots, mat = court (2.6 x 4.0 m), the chant = the 30 s raid time.

## Author's rules

- Two teams, 3 robots each on the mat. Teams take turns to send one raider.
- **Touch:** the raider touches a defender and crosses back over the midline.
  The raiding team gets **1 point**; the touched defender is out (bench).
  Two defenders touched in one raid still count as **1 point, one out**.
- **Tackle:** the defenders stop the raider. With 3 robots per team the
  defending team always has 3 or fewer on the mat, so every tackle is a
  **super tackle: 2 points**. The raider is out.
- **Revival:** every point a team scores brings back one of its out players
  (first out, first back), up to the players on the bench.
- **All out:** if every player of a team is out, the other team gets
  **2 bonus points** and the whole team comes back on the mat.
- **Match length:** **10 raids per team** (20 raids), teams take turns.
- **Midline touch (author, 11 Nov):** a defender who crosses the midline while
  in contact with the raider takes the raider home with him: the raider has
  touched him and reached the midline through him. The raid is a **touch**
  (1 point, that defender out), not a defender foul. Contact = centres within
  0.25 m, as for any touch. (If another defender is over a line at the same
  time, the raid stays a foul. Old rule: `--midline-foul`.)

## How the simulator results map to the rules

| Raid result in the simulator | Points | Who is out |
|---|---|---|
| touch (contact, then back over the midline) | raiding team 1 | the defender nearest the raider when contact was counted |
| defender over the midline while in contact with the raider | raiding team 1 (touch) | that defender |
| defender foul (any other defender over a line) | raiding team 1 per defender over (earlier rule) | each defender over |
| tackle: held or pushed out with contact | defending team 2 (super tackle) | raider |
| raider out with no tackle: walked out, or 30 s ran out | defending team 1 (proposed, see below) | raider |
| empty (back over the midline with no touch) | none | none |

Proposed, to be confirmed by the author: a raider who walks out over a line
on its own, or runs out of the 30 s, is out and the defending team gets
1 point (not a super tackle, as no tackle was made).

## Policies used (accepted pairs)

| Defenders on the mat | Defence | Raider against them |
|---|---|---|
| 3 | `alt_wo_init/defence_s0` | `alt_wo_r2/r2_raider_s42` |
| 2 | `two_def/defence_s1` | `two_def_r2/raider_s123` |
| 1 | `one_def_copy/defence_s0` | `one_def_copy/raider_s456` (copy input) |

The raider is chosen by how many defenders the other team has on the mat.
