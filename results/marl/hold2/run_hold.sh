#!/bin/bash
# Author, 14 Nov: the robot-run simulator tests with the new held rule
# (--hold 2,0.25,0.25: held after 2 s within 25 cm with a defender centre within
# 0.25 m, the touch rule's distance; robots slip and turn while held). Otherwise
# as results/marl/rule_margin: accepted pairs, varying speed, defenders 3.0-3.5 m,
# 3 cm return margin. Gate v8 500 raids (+ rule); the pair's defence 200 raids,
# policy alone and + rule; scripted raider vs the defence. 6 jobs.
set -u
cd ~/Downloads/robo_kabaddi
PY=/opt/homebrew/Caskroom/miniconda/base/envs/kabaddi/bin/python3
R=results/marl; W=$R/hold2; F=checkpoints/mappo_3999744_steps.pt
S=(--def-start 1.0,1.5 --diag-range 0.35,0.85 --diag-period 3 --return-margin 0.03 --hold 2,0.25,0.25)
log() { echo "$(date '+%F %T') $*" >> $W/driver.log; }
pair() {
  local k=$1 RA=$2 D=$3
  caffeinate -i $PY marl/eval_gates.py --raiders $RA --gates v8 --n-def $k --episodes 500 --procs 1 "${S[@]}" --raider-rule deep \
    --out $W/d${k}_gates_deep.json > $W/d${k}_gates_deep.log 2>&1 &
  caffeinate -i $PY marl/eval_defenders.py --ckpt $D --raiders $RA --n-def $k --episodes 200 --procs 1 "${S[@]}" \
    --out $W/d${k}_defence_none.json > $W/d${k}_defence_none.log 2>&1 &
  caffeinate -i $PY marl/eval_defenders.py --ckpt $D --raiders $RA --n-def $k --episodes 200 --procs 1 "${S[@]}" --raider-rule deep \
    --out $W/d${k}_defence_deep.json > $W/d${k}_defence_deep.log 2>&1 &
  caffeinate -i $PY marl/eval_defenders.py --ckpt $D --raiders scripted --n-def $k --episodes 200 --procs 1 "${S[@]}" \
    --out $W/d${k}_defence_scripted.json > $W/d${k}_defence_scripted.log 2>&1 &
  wait
  log "$k defenders: done"
}
log "start"
pair 1 $R/one_def_copy/raider_s456 $R/one_def_copy/defence_s0/$F
pair 2 $R/two_def_r2/raider_s123   $R/two_def/defence_s1/$F
pair 3 $R/alt_wo_r2/r2_raider_s42  $R/alt_wo_init/defence_s0/$F
log "STOPPED for the author"
