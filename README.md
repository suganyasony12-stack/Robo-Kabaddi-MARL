# Robo-Kabaddi

Code and data for *Robo-Kabaddi: Multi-Agent Reinforcement Learning for a
Physical-Contact Robot Sport* (Robotics and Autonomous Systems, submitted).

Robo-Kabaddi is a contact sport for mobile robots: a raider crosses the midline,
touches a defender and returns home, while the defenders try to stop it. Both
teams learn: a PPO raider and a MAPPO defender team, trained in turns in a
PyBullet simulator and run on real robots with an overhead camera as referee.

## Contents

| Path | Contents |
|---|---|
| `envs/kabaddi_env.py` | The simulator (Gymnasium, PyBullet). Use `CORRECTED_CONFIG`. |
| `training/` | Single-agent raider training (PPO, SAC, TD3) |
| `marl/` | Learned defenders (`mappo.py`), raider against learned defenders (`raider_vs_learned.py`), evaluation (`eval_gates.py`, `eval_defenders.py`), matches between teams (`match.py`, `match_rules.py`) |
| `revision_experiments/env_tests/` | Checks of every simulator rule |
| `results/c1/` | Single-agent raiders (best models, normalisation, training curves) and their evaluations |
| `results/marl/` | The learned teams for 1, 2 and 3 defenders, and their evaluations |
| `physical_experiment/rig/` | Robot runner (`raid_runner.py`): camera tracking, Bluetooth control, the simulator's rules on the real court |
| `physical_experiment/firmware/` | Robot firmware (Bluetooth); set your own robot role before flashing |
| `physical_experiment/match_data/` | The robot raids of 13 Nov 2026: every step (`steps.csv`), every result (`raids.csv`) and pictures of the counted raids |
| `analysis/make_paper_tables.py` | Rebuilds every table and figure of the paper into `paper_tables/` |

## Install

```bash
conda env create -f environment.yml
conda activate kabaddi
```

## Rebuild the paper's tables and figures

```bash
python3 analysis/make_paper_tables.py
```

## Check the simulator rules

```bash
for f in revision_experiments/env_tests/check_*.py; do python3 "$f"; done
python3 marl/test_match_rules.py
```

## Evaluate a learned team

```bash
python3 marl/eval_defenders.py --ckpt results/marl/alt_wo_init/defence_s0/checkpoints/mappo_3999744_steps.pt \
    --raiders results/marl/alt_wo_r2/r2_raider_s42,scripted --n-def 3 --episodes 200 --out out.json
python3 marl/eval_gates.py --raiders results/marl/alt_wo_r2/r2_raider_s42 --n-def 3 --episodes 1000 --out gates.json
```

## Run the robots

See `physical_experiment/rig/README.md`. A run without robots or camera:

```bash
python3 physical_experiment/rig/raid_runner.py --raider learned --defence learned --n-def 3 --raids 2 --fake --outdir /tmp/rk
```

The robot runs in the paper used `--def-start 1.0,1.5` and, from the
2-defender gate runs on, `--return-margin 0.03`; `--raider-rule deep` is the
go-deep rule reported with the learned raider.

## Licence

MIT, see `LICENSE`.
