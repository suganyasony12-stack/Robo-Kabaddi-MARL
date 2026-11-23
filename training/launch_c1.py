"""
launch_c1.py — run the C1 training set with at most 7 jobs at once
===================================================================

Starts every run of training/train_corrected.py that has no final_model.zip
yet, longest first, keeping at most --jobs running (default 7: the 8-core Mac
keeps one core free). Each run's output goes to results/c1/logs/<run>.log;
progress is written to results/c1/launch_status.json. Safe to stop (Ctrl-C)
and start again: finished runs are skipped, unfinished ones start over.

Usage (from the robo_kabaddi/ folder)
-------------------------------------
    caffeinate -i python3 training/launch_c1.py
    python3 training/launch_c1.py --jobs 4 --only main_1v3_seed42 sac_1v3_seed42
"""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "training"))
from train_corrected import RUNS, OUT           # noqa: E402

# Rough minutes per run (pilot-measured speed), to start the longest first.
COST = {("ppo", 3): 45, ("sac", 3): 60, ("td3", 3): 50, ("ppo", 2): 35,
        ("ppo", 1): 25}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--jobs", type=int, default=7)
    ap.add_argument("--only", nargs="*", help="run only these")
    args = ap.parse_args()
    assert 1 <= args.jobs <= 7, "at most 7 jobs: keep one core free"

    todo = [r for r in (args.only or RUNS) if not (OUT / r / "final_model.zip").exists()]
    todo.sort(key=lambda r: -COST[(RUNS[r]["algo"], RUNS[r]["n_def"])])
    (OUT / "logs").mkdir(parents=True, exist_ok=True)
    env = {**os.environ, "OMP_NUM_THREADS": "1", "MKL_NUM_THREADS": "1"}
    running, done, failed = {}, [], []
    t0 = time.time()
    print(f"{len(todo)} runs to do, {args.jobs} at a time", flush=True)

    def status():
        (OUT / "launch_status.json").write_text(json.dumps({
            "updated": time.strftime("%Y-%m-%d %H:%M:%S"),
            "elapsed_min": round((time.time() - t0) / 60, 1),
            "running": sorted(running), "done": done, "failed": failed,
            "waiting": todo}, indent=2))

    try:
        while todo or running:
            while todo and len(running) < args.jobs:
                r = todo.pop(0)
                log = open(OUT / "logs" / f"{r}.log", "w")
                running[r] = (subprocess.Popen(
                    [sys.executable, str(ROOT / "training" / "train_corrected.py"),
                     "--run", r], cwd=ROOT, env=env, stdout=log,
                    stderr=subprocess.STDOUT), log, time.time())
                print(f"[{time.strftime('%H:%M')}] start {r}", flush=True)
            status()
            time.sleep(20)
            for r, (proc, log, ts) in list(running.items()):
                if proc.poll() is None:
                    continue
                log.close()
                (done if proc.returncode == 0 else failed).append(r)
                del running[r]
                print(f"[{time.strftime('%H:%M')}] {'done' if proc.returncode == 0 else 'FAILED'}"
                      f" {r} ({(time.time() - ts) / 60:.0f} min); "
                      f"{len(done)} done, {len(failed)} failed, {len(todo)} waiting",
                      flush=True)
            status()
    except KeyboardInterrupt:
        for proc, log, _ in running.values():
            proc.terminate()
        print("stopped; unfinished runs will start over next time", flush=True)
    status()
    print(f"finished in {(time.time() - t0) / 60:.0f} min: {len(done)} done, "
          f"{len(failed)} failed {failed}", flush=True)


if __name__ == "__main__":
    main()
