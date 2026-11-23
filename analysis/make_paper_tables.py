"""
make_paper_tables.py -- every table and figure for the resubmission from saved results (author, 19 Nov)
======================================================================================================

    python3 analysis/make_paper_tables.py            # writes paper_tables/

Reviewer 2 asked that figures and tables come from one source; nothing here is
typed in by hand. Inputs (all saved runs, no training, no simulation):

  results/c1/eval_results*.json        single-agent raiders on the corrected simulator
  results/c1/<run>/metrics.csv          training curves
  results/marl/match/*.json             matches between teams of 3 (C4)
  results/marl/rule_margin/*.json       simulator, robot-run settings (held 4 s, 5 cm)
  results/marl/hold2/*.json             simulator, held 2 s / 25 cm / 0.25 m (extra)
  physical_experiment/match_data/       the 13 Nov robot runs (raids.csv, steps.csv)

Outputs (paper_tables/): T1..T6 as .md, .csv and .tex (booktabs), F1..F2 as
.pdf and .png. Rates carry 95 % Wilson intervals. The old figures/ and
paper_figures/ are not touched.
"""
from __future__ import annotations

import csv
import glob
import json
import math
import sys
from collections import Counter, defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "paper_tables"
sys.path.insert(0, str(ROOT / "physical_experiment/rig/hold_rescore"))
import rescore_hold as RH                                             # noqa: E402

GATES = ["v6", "v7", "v8", "v9", "v10", "v11"]
SEEDS = ("seed42", "seed123", "seed456")


# ------------------------------------------------------------------ helpers
def wilson(k, n, z=1.96):
    if n == 0:
        return (0.0, 0.0)
    p = k / n
    d = 1 + z * z / n
    c = (p + z * z / (2 * n)) / d
    h = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / d
    return (max(0.0, c - h), min(1.0, c + h))


def pct(k, n, ci=True):
    if n == 0:
        return "—"
    lo, hi = wilson(k, n)
    return f"{100 * k / n:.1f}" + (f" [{100 * lo:.0f}, {100 * hi:.0f}]" if ci else "")


def write_table(name, title, header, rows, note=""):
    OUT.mkdir(exist_ok=True)
    with open(OUT / f"{name}.csv", "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(header)
        w.writerows(rows)
    md = [f"### {name}: {title}", "", "| " + " | ".join(header) + " |", "|" + "---|" * len(header)]
    md += ["| " + " | ".join(str(c) for c in r) + " |" for r in rows]
    if note:
        md += ["", note]
    (OUT / f"{name}.md").write_text("\n".join(md) + "\n")
    esc = lambda s: str(s).replace("%", r"\%").replace("_", r"\_").replace("—", "--")
    tex = [r"\begin{table}[t]", r"\centering", rf"\caption{{{esc(title)}}}", rf"\label{{tab:{name}}}",
           r"\begin{tabular}{" + "l" + "c" * (len(header) - 1) + "}", r"\toprule",
           " & ".join(esc(h) for h in header) + r" \\", r"\midrule"]
    tex += [" & ".join(esc(c) for c in r) + r" \\" for r in rows]
    tex += [r"\bottomrule", r"\end{tabular}"]
    if note:
        tex += [rf"\par\smallskip\footnotesize {esc(note)}"]
    tex += [r"\end{table}"]
    (OUT / f"{name}.tex").write_text("\n".join(tex) + "\n")
    return md


# ------------------------------------------------- single-agent (C1) results
def c1_runs():
    """(label, gate, mode, condition) -> list of per-seed entries (one per seed, no duplicates)."""
    pool = defaultdict(dict)
    for f in sorted(glob.glob(str(ROOT / "results/c1/eval_results*.json"))):
        for r in json.load(open(f))["per_run"]:
            run, pol = r["run"], r.get("policy", "best")
            gate = r.get("gate_style") or "keepout"
            if r.get("defender_type") == "physical":
                gate = "pursuit"
            gate = "v6" if gate in ("keepout", "gate") else gate
            seed = next((s for s in SEEDS if s in run), None)
            base = run.replace(f"_{seed}", "") if seed else run
            if base == "main_1v3_ext":
                label = "PPO 2.27M" if pol == "best" else "PPO 1.01M"
            elif base == "main_1v3":
                label = "PPO 0.76M"
            elif base.startswith(("sac_1v3", "td3_1v3")):
                algo = base[:3].upper()
                budget = base.split("_")[-1] if base.endswith(("_1M", "_2.27M")) else "250k"
                label = f"{algo} {budget}"
            elif base == "ppo250k_1v3":
                label = "PPO 250k"
            elif base in ("count_1v2", "count_1v2_ext"):
                label = "1 vs 2"
            elif base == "count_1v1":
                label = "1 vs 1"
            elif base.startswith("abl_"):
                label = base[4:].replace("_1v1", "")
            elif base == "pursuit_1v3":
                label = "PPO pursuit-trained"
            else:
                label = base
            key = (label, gate, r.get("mode", "policy"), r.get("condition", "own"))
            prio = 1 if base == "count_1v2_ext" else 0       # seed 123 of 1 vs 2 was trained longer
            old = pool[key].get(seed)
            if old is None or prio >= old[0]:
                pool[key][seed] = (prio, r)
    return {k: [v[1] for v in d.values()] for k, d in pool.items()}


def pooled(entries, kind):
    k = sum(e[kind]["count"] for e in entries)
    n = sum(e["n"] for e in entries)
    return k, n


def t1_t2_t3(C):
    out = []
    # T1: algorithms, 1 vs 3, touch % over the gates (policy mode)
    rows = []
    for lab in ("PPO 2.27M", "PPO 1.01M", "SAC 2.27M", "SAC 1M", "TD3 1M"):
        r = [lab]
        for g in GATES:
            e = C.get((lab, g, "policy", "own"), [])
            r.append(pct(*pooled(e, "touch")) + ("" if len(e) == 3 else f" ({len(e)} seed{'s' if len(e) != 1 else ''})")
                     if e else "—")
        rows.append(r)
    out += write_table("T1_single_agent_algorithms", "Single-agent raiders, 1 vs 3, touch % against the gate "
                       "defences (v6 training, v7-v11 held out), 3 seeds x 1000 raids",
                       ["Raider", *GATES], rows, "95 % Wilson intervals. TD3 was evaluated on v6 and v7 only.")
    # T2: number of defenders (PPO)
    rows = []
    for lab, nm in (("1 vs 1", "1 defender"), ("1 vs 2", "2 defenders"), ("PPO 2.27M", "3 defenders")):
        rows.append([nm] + [pct(*pooled(C.get((lab, g, "policy", "own"), []), "touch")) for g in GATES])
    out += write_table("T2_defender_count", "PPO raider against 1, 2 and 3 gate defenders, touch %",
                       ["Defenders", *GATES], rows, "3 seeds x 1000 raids per cell; 1 vs 2 seed 123 trained longer; v10 was evaluated for 3 defenders only.")
    # T3: reward ablation, 1 vs 1, v6
    rows = []
    for lab, nm in (("1 vs 1", "Full reward"), ("no_progress", "No progress term"), ("no_milestones", "No milestones"),
                    ("no_time_penalty", "No time penalty"), ("sparse_only", "Outcome reward only")):
        e = C.get((lab, "v6", "policy", "own"), [])
        rows.append([nm] + [pct(*pooled(e, k)) for k in ("touch", "tackle", "empty")])
    out += write_table("T3_reward_ablation", "Reward ablation, 1 vs 1 against the v6 gate (%)",
                       ["Reward", "Touch", "Tackle", "Empty"], rows, "3 seeds x 1000 raids.")
    return out


# ------------------------------------------------------------- matches (C4)
def t4_matches():
    rows = []
    for f in sorted(glob.glob(str(ROOT / "results/marl/match/*.json"))):
        d = json.load(open(f))
        M = d["matches"]
        w = Counter(m["winner"] for m in M)
        raids = [r for m in M for r in m["raids"]]
        kinds = Counter(r["kind"] for r in raids)
        rows.append([Path(f).stem, len(M), w.get("A", 0), w.get("B", 0), w.get("draw", 0),
                     f"{sum(m['A'] for m in M) / len(M):.1f}", f"{sum(m['B'] for m in M) / len(M):.1f}",
                     len(raids), ", ".join(f"{k} {v}" for k, v in kinds.most_common())])
    return write_table("T4_matches", "Matches between teams of three (C4 match rules), simulator",
                       ["Setup (team A-team B)", "Matches", "A wins", "B wins", "Draws", "A points", "B points",
                        "Raids", "Raid results"], rows)


# ---------------------------------------------------------------- robot runs
CASES = [("A-rule", "learned + rule vs learned", "defence_deep", 0),
         ("A", "learned vs learned", "defence_none", 0),
         ("C", "scripted vs learned", "defence_scripted", 0),
         ("B-rule", "learned + rule vs gate v8", "gates_deep", 2)]


def case_of(key):
    if key[1] == "scripted":
        return "C"
    if key[2] == "scripted":
        return "B-rule"
    return "A-rule" if key[4] else "A"


SIDE = lambda o: "raider" if o in ("touch", "foul") else ("empty" if o == "empty" else "defence")


def robot_tables():
    cases = RH.load()
    logged = RH.rescore(cases, 4.0, 0.19)                  # reproduces the logged results (checked)
    hold2 = RH.rescore(cases, 2.0, 0.25, 0.25)
    out = []
    for name, rob, simdir, title in (
            ("T5_robot_runs", logged, "rule_margin", "Robot runs (as conducted: held after 4 s within 5 cm, "
             "body contact) against the simulator on the same seeds"),
            ("T6_robot_runs_hold2", hold2, "hold2", "Extra observation: the robot raids re-scored with held after "
             "2 s within 25 cm (a defender within 0.25 m), and the simulator with the same rule")):
        rows = []
        for k in (1, 2, 3):
            for cname, desc, f, idx in CASES:
                key = next(kk for kk in rob if kk[0] == k and case_of(kk) == cname)
                R = rob[key]
                sim = json.load(open(ROOT / f"results/marl/{simdir}/d{k}_{f}.json"))
                sim = sim[idx] if len(sim) > idx and sim[idx].get("gate", sim[idx].get("gate_style")) in ("v8",) else sim[0]
                so = [sim["outcomes"][x["seed"] - 50000] for x in R]
                pts = sum(SIDE(x["rescored"]) == "raider" for x in R)
                emp = sum(x["rescored"] == "empty" for x in R)
                spts = sum(SIDE(o) == "raider" for o in so)
                same = sum(SIDE(x["rescored"]) == SIDE(o) for x, o in zip(R, so))
                n = sim["n"]
                sk = sim["touch"]["count"] + sim.get("foul", {}).get("count", 0)
                rows.append([f"{k}", f"{cname}: {desc}", f"{pts}/15 ({pct(pts, 15)})", f"{emp}",
                             f"{spts}/15", f"{pct(sk, n)} (n={n})", f"{same}/15"])
        out += write_table(name, title, ["Def.", "Case", "Robots: raider points", "Robots: empty",
                                         "Sim, same seeds", "Sim, all raids", "Same side, raid by raid"], rows,
                           "Raider point = touch or defender foul. Defenders start 3.0-3.5 m from the raider's end "
                           "line; simulator with varying diagonal speed (0.35-0.85) and the 3 cm return margin.")
    return out, logged, hold2


# ------------------------------------------------------------------ figures
def figures(logged, hold2):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    OUT.mkdir(exist_ok=True)
    # F1: training curves, PPO 1 vs 3 (gate stage after the pursuit third), 3 seeds
    fig, ax = plt.subplots(figsize=(6.5, 3.4))
    for s, c in zip(SEEDS, ("C0", "C1", "C2")):
        xs, ys = [], []
        for run in (f"main_1v3_{s}", f"main_1v3_{s}_ext"):
            p = ROOT / f"results/c1/{run}/metrics.csv"
            if p.exists():
                for r in csv.DictReader(open(p)):
                    if r["stage"] == "gate":
                        xs.append(int(r["timestep"]) / 1e6); ys.append(100 * float(r["touch_rate"]))
        if xs:
            k = 5
            sm = [sum(ys[max(0, i - k):i + 1]) / len(ys[max(0, i - k):i + 1]) for i in range(len(ys))]
            ax.plot(xs, sm, color=c, label=s.replace("seed", "seed "))
    ax.set(xlabel="decisions (millions)", ylabel="training touch % (gate v6)",
           title="PPO raider, 1 vs 3 against the gate")
    ax.set_ylim(0, 100); ax.legend(frameon=False); fig.tight_layout()
    for e in ("pdf", "png"):
        fig.savefig(OUT / f"F1_training_curves.{e}", dpi=200)
    plt.close(fig)
    # F2: robots against the simulator, raider points of 15
    fig, axes = plt.subplots(1, 2, figsize=(9, 3.6), sharey=True)
    for ax, rob, simdir, ttl in ((axes[0], logged, "rule_margin", "held 4 s / 5 cm (as run)"),
                                 (axes[1], hold2, "hold2", "held 2 s / 25 cm (extra)")):
        labels, rv, sv = [], [], []
        for k in (1, 2, 3):
            for cname, desc, f, idx in CASES:
                key = next(kk for kk in rob if kk[0] == k and case_of(kk) == cname)
                R = rob[key]
                sim = json.load(open(ROOT / f"results/marl/{simdir}/d{k}_{f}.json"))
                sim = sim[idx] if len(sim) > idx and sim[idx].get("gate", sim[idx].get("gate_style")) in ("v8",) else sim[0]
                labels.append(f"{k}{cname}")
                rv.append(sum(SIDE(x["rescored"]) == "raider" for x in R))
                sv.append(sum(SIDE(sim["outcomes"][x["seed"] - 50000]) == "raider" for x in R))
        x = range(len(labels))
        ax.bar([i - 0.2 for i in x], rv, 0.4, label="robots", color="#3b6ea5")
        ax.bar([i + 0.2 for i in x], sv, 0.4, label="simulator", color="#c9a227")
        ax.set_xticks(list(x)); ax.set_xticklabels(labels, rotation=60, fontsize=8)
        ax.set_title(ttl, fontsize=10); ax.set_ylim(0, 15)
    axes[0].set_ylabel("raider points of 15"); axes[0].legend(frameon=False, fontsize=8)
    fig.tight_layout()
    for e in ("pdf", "png"):
        fig.savefig(OUT / f"F2_robots_vs_simulator.{e}", dpi=200)
    plt.close(fig)


# ----------------------------------------------------------- MARL (C3, C4)
M = ROOT / "results/marl"
F_CK = "/checkpoints/mappo_3999744_steps.pt"
# the accepted pairs (match.py PAIRS): defenders -> (defence, raider, files)
PAIRS = {3: ("alt_wo_init/defence_s0", "alt_wo_r2/r2_raider_s42", "alt_wo_r2/eval_r2_raiders_vs_all_defences_200.json",
             "alt_wo_init/eval_defence_200.json", "alt_wo_r2/eval_gates_r2_1000.json"),
         2: ("two_def/defence_s1", "two_def_r2/raider_s123", "two_def_r2/eval_raiders_vs_all_defences_2def_200.json",
             "two_def/eval_defence_2def_200.json", "two_def_r2/eval_gates_2def_1000.json"),
         1: ("one_def_copy/defence_s0", "one_def_copy/raider_s456",
             "one_def_copy/eval_raiders_vs_all_defences_1def_200.json",
             "one_def_copy/eval_defence_1def_200.json", "one_def_copy/eval_gates_1def_1000.json")}


def _ck(x):
    return x.get("ckpt", "").replace(str(M) + "/", "").replace("results/marl/", "").replace(F_CK, "")


def _rd(x):
    return str(x.get("raider", "")).replace(str(M) + "/", "").replace("results/marl/", "")


def _find(path, ck=None, raider=None, gate=None):
    for x in json.load(open(M / path)):
        if (ck is None or _ck(x) == ck) and (raider is None or _rd(x) == raider) and (gate is None or x.get("gate") == gate):
            return x
    raise KeyError(f"{path}: {ck} / {raider} / {gate}")


def _cell(x, kind):
    return pct(x[kind]["count"], x["n"]) if kind in x else "—"


def marl_tables():
    out = []
    # T7: learned defences (MAPPO, first stage) against fixed raiders, 3 defenders
    rows = []
    for run, nm in (("mappo_scripted_ent03_s0", "trained vs scripted raider, seed 0"),
                    ("mappo_scripted_ent03_s1", "trained vs scripted raider, seed 1"),
                    ("mappo_hard_ent03_s0", "trained vs PPO raider (seed 42)"),
                    ("mappo_pool3_ent03_s0", "trained vs a pool of PPO raiders")):
        d = [x for x in json.load(open(M / f"eval_{run}_200.json")) if _ck(x) == run]
        scr = next(x for x in d if _rd(x) == "scripted")
        ppo = [x for x in d if _rd(x).startswith("main_1v3_")]
        k = lambda kind: (sum(x[kind]["count"] for x in ppo), sum(x["n"] for x in ppo))
        rows.append([nm, _cell(scr, "touch"), _cell(scr, "tackle"), _cell(scr, "foul"),
                     pct(*k("touch")), pct(*k("tackle")), pct(*k("foul"))])
    out += write_table("T7_learned_defences", "Learned defences (MAPPO, 3 defenders) against fixed raiders, %",
                       ["Defence", "Scripted raider: touch", "tackle", "foul",
                        "PPO raiders (3 seeds): touch", "tackle", "foul"], rows,
                       "200 raids per raider; foul = a defender over a line (the raider's point).")
    # T8: the accepted pairs after alternating training (nominal: defenders in the back row, speed 1.0)
    rows, grows = [], []
    for k in (1, 2, 3):
        dfn, rdr, f_rv, f_dv, f_g = PAIRS[k]
        own = _find(f_rv, ck=dfn, raider=rdr)
        scr = _find(f_dv, ck=dfn, raider="scripted")
        rows.append([f"{k}", f"{rdr} vs {dfn}"] + [_cell(own, c) for c in ("touch", "tackle", "foul", "empty")]
                    + [_cell(scr, "touch"), _cell(scr, "tackle")])
        grows.append([f"{k}"] + [_cell(_find(f_g, raider=rdr, gate=g), "touch") for g in GATES])
    out += write_table("T8_marl_pairs", "The learned pairs after alternating training: the raider against its own "
                       "defence, and the defence against the scripted raider (%)",
                       ["Def.", "Pair (raider vs defence)", "Raider vs own defence: touch", "tackle", "foul", "empty",
                        "Defence vs scripted raider: touch", "tackle"], rows,
                       "200 raids each; defenders start in the back row; speed 1.0.")
    out += write_table("T9_marl_raider_gates", "The learned raiders of the pairs against the gate defences, touch %",
                       ["Def.", *GATES], grows, "1000 raids per cell.")
    return out


def main():
    C = c1_runs()
    md = t1_t2_t3(C)
    md += [""] + t4_matches()
    md += [""] + marl_tables()
    rmd, logged, hold2 = robot_tables()
    md += [""] + rmd
    figures(logged, hold2)
    (OUT / "ALL_TABLES.md").write_text("# Tables for the resubmission (generated by analysis/make_paper_tables.py)\n\n"
                                       + "\n".join(md) + "\n\nFigures: F1_training_curves, F2_robots_vs_simulator\n")
    print(f"written to {OUT}")


if __name__ == "__main__":
    main()
