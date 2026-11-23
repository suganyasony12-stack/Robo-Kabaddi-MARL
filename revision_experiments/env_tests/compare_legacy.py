"""Run the ORIGINAL env and the NEW env in legacy mode with identical seeds and
actions; compare every PyBullet call. Identical logs mean the legacy mode is
the submitted code, call for call."""
import importlib.util
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent / 'mock'))
import pybullet as p  # the mock

ENVS = str(Path(__file__).resolve().parents[2] / 'envs') + '/'


def load(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def run(mod, kwargs, n_def, dtype, seed, steps, forward=False):
    p.LOG.clear()
    env = mod.RoboKabaddiEnv(n_defenders=n_def, defender_type=dtype, **kwargs)
    obs, _ = env.reset(seed=seed)
    rng = np.random.default_rng(seed + 1000)
    obs_trace = [obs.copy()]
    for _ in range(steps):
        a = rng.uniform(-1, 1, size=3).astype(np.float32)
        if forward:
            a[1] = 1.0          # drive across the midline
        obs, r, term, trunc, info = env.step(a)
        obs_trace.append(obs.copy())
        if term or trunc:
            break
    env.close()
    return list(p.LOG), obs_trace


old = load('env_old', ENVS + 'kabaddi_env_legacy_backup.py')
new = load('env_new', ENVS + 'kabaddi_env.py')

ok = True
for n_def in (1, 3, 5):
    for dtype in ('static', 'reactive', 'wall'):
        for seed, fwd in ((0, False), (42, False), (7, True), (99, True)):
            la, oa = run(old, {}, n_def, dtype, seed, 200, fwd)
            lb, ob = run(new, {'locomotion': 'legacy'}, n_def, dtype, seed, 200, fwd)
            same_calls = la == lb
            same_obs = len(oa) == len(ob) and all(
                np.array_equal(x, y) for x, y in zip(oa, ob))
            ok &= same_calls and same_obs
            if not (same_calls and same_obs):
                first = next((i for i, (x, y) in enumerate(zip(la, lb))
                              if x != y), min(len(la), len(lb)))
                print(f'  MISMATCH n_def={n_def} {dtype} seed={seed}: '
                      f'{len(la)} vs {len(lb)} calls, first diff at {first}')
                print('    old:', la[first] if first < len(la) else None)
                print('    new:', lb[first] if first < len(lb) else None)
            else:
                tag = 'fwd' if fwd else 'rnd'
                print(f'  identical  n_def={n_def} {dtype:<8} seed={seed:<3} {tag}'
                      f' {len(la):>6} calls, {len(oa)} observations')

print('\nLEGACY MODE IS CALL-FOR-CALL IDENTICAL TO THE SUBMITTED CODE'
      if ok else '\nLEGACY MODE DIFFERS — DO NOT USE')
