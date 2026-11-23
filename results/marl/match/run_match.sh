#!/bin/bash
# Author, 5 Nov: C4 matches under the author's rules (revision_experiments/C4_MATCH_RULES.md),
# accepted pairs for 3/2/1 defenders, 10 raids per team, 200 matches per setup.
set -u
cd ~/Downloads/robo_kabaddi
PY=/opt/homebrew/Caskroom/miniconda/base/envs/kabaddi/bin/python3
W=results/marl/match
log() { echo "$(date '+%F %T') $*" >> $W/driver.log; }
for s in learned-learned learned-scripted scripted-scripted; do
  log "$s: start"
  caffeinate -i $PY marl/match.py --setup $s --matches 200 --procs 7 --out $W/$s.json > $W/$s.log 2>&1
  log "$s: done"
done
log "STOPPED for the author"
