# sim_robot_gap.py (author, 19 Nov): why the real learned defences do worse in 2 def C and 3 def A.
# The same 15 seeds replayed in the simulator (positions every 0.05 s) against the logged robot raids.
import csv, json, math, sys
from pathlib import Path
import numpy as np
ROOT = Path.home() / "Downloads/robo_kabaddi"
sys.path[:0] = [str(ROOT / "marl"), str(ROOT), str(ROOT / "physical_experiment/rig/hold_rescore")]
from defender_env import DefenderEnv, diag_kwargs, start_kwargs, margin_kwargs
from eval_defenders import load
import rescore_hold as RH
R = ROOT / "results/marl"; F = "checkpoints/mappo_3999744_steps.pt"
OFF = np.array([1.3, 2.0])
CASES = {"2 def C": (2, "scripted", f"{R}/two_def/defence_s1/{F}", (2, "scripted", "learned", "", "")),
         "3 def A": (3, f"{R}/alt_wo_r2/r2_raider_s42", f"{R}/alt_wo_init/defence_s0/{F}", (3, "learned", "learned", "", ""))}
KW = {**diag_kwargs("0.35,0.85", 3), **start_kwargs("1.0,1.5"), **margin_kwargs(0.03)}

def sim_raid(k, raider, act, seed):
    env = DefenderEnv(raider, k, pad_to=getattr(act, "n_in", None), **KW)
    obs, _ = env.reset(seed=seed); e = env.env; T = []
    while True:
        obs, _, _, done, info = env.step(act(obs))
        T.append([e._step * e.dt, int(e._flags["midline"]), int(e._flags["contact"])]
                 + list(e._get_pos(e.raider_id)[:2] + OFF) + [v for d in e.defender_ids for v in e._get_pos(d)[:2] + OFF])
        if done: break
    start = e._raid_start_step
    out = info.get("outcome", "empty")
    if out == "tackle":
        p = e._get_pos(e.raider_id)
        out = "out" if abs(p[0]) > 1.3 or abs(p[1]) > 2.0 else ("held" if start is not None and e._step - start < e._raid_steps else "timer")
    env.close()
    return out, np.array(T)

def robot_raid(row_steps, k):
    T = [[float(x["t_s"]), int(x["midline"]), int(x["contact"]), float(x["raider_x"]), float(x["raider_y"])]
         + [float(x[f"d{j}_{c}"]) for j in range(1, k + 1) for c in "xy"] for x in row_steps]
    return np.array(T)

def metrics(T, k):
    t, mid, con, r = T[:, 0], T[:, 1], T[:, 2], T[:, 3:5]
    D = T[:, 5:].reshape(len(T), k, 2)
    dist = np.linalg.norm(D - r[:, None, :], axis=2).min(1)
    act = mid > 0                                              # raid running
    dt = np.diff(t, prepend=t[0])
    sp = np.linalg.norm(np.diff(D, axis=0), axis=2) / np.maximum(np.diff(t)[:, None], 1e-6)
    spa = sp[act[1:]] if act[1:].any() else sp
    i0 = int(np.argmax(act)) if act.any() else 0
    first_close = next((t[i] - t[i0] for i in range(i0, len(t)) if dist[i] <= 0.25), None)
    # after the counted touch: is a defender between the raider and the midline (blocking the way home)?
    ic = int(np.argmax(con)) if con.any() else None
    block = None
    if ic is not None:
        b = [(D[i, :, 1] < r[i, 1]) & (np.abs(D[i, :, 0] - r[i, 0]) < 0.30) for i in range(ic, len(t))]
        block = float(np.mean([x.any() for x in b]))
    return dict(def_speed=float(np.mean(spa)), moving=float(np.mean(spa > 0.05)),
                close_s=float(dt[act & (dist <= 0.25)].sum()), first_close=first_close,
                deepest=float(r[act, 1].max() - 2.0) if act.any() else 0.0, block_after_touch=block,
                raid_s=float(t[-1] - t[i0]))

cases = RH.load()
res = {}
for name, (k, raider, ckpt, key) in CASES.items():
    act = load(ckpt)
    rows = cases[key]
    S, Rb = [], []
    for seed in range(50000, 50015):
        so, Ts = sim_raid(k, raider, act, seed)
        sess, r, st = rows[seed]
        Tr = robot_raid(st, k)
        S.append(dict(seed=seed, outcome=so, **metrics(Ts, k)))
        Rb.append(dict(seed=seed, outcome=r["kind"], **metrics(Tr, k)))
    res[name] = dict(sim=S, robots=Rb)
    def summ(L):
        f = lambda key: np.nanmean([x[key] for x in L if x[key] is not None])
        from collections import Counter
        return (f"outcomes {dict(Counter(x['outcome'] for x in L))}\n      defender speed {f('def_speed'):.2f} m/s, moving {100*f('moving'):.0f}% of the raid, "
                f"time within 0.25 m {f('close_s'):.1f} s, first within 0.25 m after {f('first_close'):.1f} s, "
                f"raider deepest {f('deepest'):.2f} m, way home blocked after the touch {100*f('block_after_touch'):.0f}% of the time, raid {f('raid_s'):.1f} s")
    print(f"== {name}\n  robots: {summ(Rb)}\n  sim:    {summ(S)}")
json.dump(res, open(Path(__file__).with_name("gap.json"), "w"), indent=1, default=float)

# --- 2. contact while blocking: is the blocker pushed aside, or does it step away?
def push(T, k):
    t, con, r = T[:, 0], T[:, 2], T[:, 3:5]
    D = T[:, 5:].reshape(len(T), k, 2)
    if not con.any():
        return None
    ic = int(np.argmax(con)); out = dict(block_contact_s=0.0, pushed_back_m=0.0, slid_side_m=0.0, raider_home_m=0.0)
    for i in range(ic, len(t) - 1):
        dts = t[i + 1] - t[i]
        for j in range(k):
            d = np.linalg.norm(D[i, j] - r[i])
            blocking = D[i, j, 1] < r[i, 1] and abs(D[i, j, 0] - r[i, 0]) < 0.30
            if d <= 0.22 and blocking:                         # touching the blocker
                out["block_contact_s"] += dts
                mv = D[i + 1, j] - D[i, j]
                out["pushed_back_m"] += max(0.0, -mv[1])       # blocker moved toward the midline (raider's way)
                out["slid_side_m"] += abs(mv[0])
                out["raider_home_m"] += max(0.0, -(r[i + 1, 1] - r[i, 1]))
    return out

print("\nwhile touching a blocking defender, after the touch (per raid, mean):")
for name, (k, raider, ckpt, key) in CASES.items():
    act = load(ckpt); rows = cases[key]
    for lab in ("robots", "sim"):
        P = []
        for seed in range(50000, 50015):
            T = robot_raid(rows[seed][2], k) if lab == "robots" else sim_raid(k, raider, act, seed)[1]
            m = push(T, k)
            if m: P.append(m)
        f = lambda key: np.mean([x[key] for x in P])
        rate = lambda a, b: f(a) / max(f(b), 1e-9)
        print(f"  {name} {lab:<7} contact with a blocker {f('block_contact_s'):.1f} s; per second of it: raider moves home "
              f"{rate('raider_home_m', 'block_contact_s'):.2f} m, blocker moves back {rate('pushed_back_m', 'block_contact_s'):.2f} m, "
              f"sideways {rate('slid_side_m', 'block_contact_s'):.2f} m")
