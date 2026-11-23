#!/bin/bash
# Author, 13 Nov: the robot-run simulator tests again with the 3 cm return
# margin (--return-margin 0.03: Empty on return only 3 cm behind the midline),
# as the robot runs now use it. Same as results/marl/rule/run_rule.sh and
# run_scripted.sh otherwise: accepted pairs, varying speed, defenders 3.0-3.5 m;
# gates v6-v11 500 raids, the pair's defence 200 raids, policy alone and + rule,
# scripted raider vs the defence. 2 jobs (robot runs on this computer).
set -u
cd ~/Downloads/robo_kabaddi
PY=/opt/homebrew/Caskroom/miniconda/base/envs/kabaddi/bin/python3
R=results/marl; W=$R/rule_margin; F=checkpoints/mappo_3999744_steps.pt
S=(--def-start 1.0,1.5 --diag-range 0.35,0.85 --diag-period 3 --return-margin 0.03)
log() { echo "$(date '+%F %T') $*" >> $W/driver.log; }
pair() {
  local k=$1 RA=$2 D=$3
  local m; for m in none deep; do
    local RR=(); [ $m = deep ] && RR=(--raider-rule deep)
    caffeinate -i $PY marl/eval_gates.py --raiders $RA --n-def $k --episodes 500 --procs 2 "${S[@]}" ${RR[@]+"${RR[@]}"} \
      --out $W/d${k}_gates_$m.json > $W/d${k}_gates_$m.log 2>&1 || log "gates $k $m failed"
    caffeinate -i $PY marl/eval_defenders.py --ckpt $D --raiders $RA --n-def $k --episodes 200 --procs 2 "${S[@]}" ${RR[@]+"${RR[@]}"} \
      --out $W/d${k}_defence_$m.json > $W/d${k}_defence_$m.log 2>&1 || log "defence $k $m failed"
  done
  caffeinate -i $PY marl/eval_defenders.py --ckpt $D --raiders scripted --n-def $k --episodes 200 --procs 1 "${S[@]}" \
    --out $W/d${k}_defence_scripted.json > $W/d${k}_defence_scripted.log 2>&1 || log "scripted $k failed"
  log "$k defenders: done"
}
log "start"
pair 1 $R/one_def_copy/raider_s456    $R/one_def_copy/defence_s0/$F
pair 2 $R/two_def_r2/raider_s123      $R/two_def/defence_s1/$F
pair 3 $R/alt_wo_r2/r2_raider_s42     $R/alt_wo_init/defence_s0/$F
log "STOPPED for the author"
