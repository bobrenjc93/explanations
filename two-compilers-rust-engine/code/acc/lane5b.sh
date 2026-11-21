#!/bin/bash
# lane5b.sh: probe4.py on every recorded state that does not have it yet (trees linked to build_api15), two at a time.
H=/home/dev/local/hill2
cd $H
run() { t=$1; s=${t%%_*}
  [ -s acc/out/probe4_$t.json ] || HILL_CACHE=$H/acc/cache_$s HILL_CORES=$2 ./acc/gpu2 python -I acc/probe4.py --tree acc/trees/$t --config acc/trees/$t/hill_config.py --out acc/out/probe4_$t.json --label $t > acc/out/probe4_$t.log 2>&1
  echo "$t: $( [ -s acc/out/probe4_$t.json ] && echo done || echo FAILED )"; }
(for t in $(ls acc/trees | grep '^tc_'); do run $t 12-19; done) &
(for t in $(ls acc/trees | grep '^vm_'); do run $t 24-31; done) &
wait
echo "LANE5B DONE"
