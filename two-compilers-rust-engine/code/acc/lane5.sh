#!/bin/bash
# lane5.sh: probe4.py (the stronger probes of kept results) on the best, final and fast states of both sides, with
# the trees linked to the binaries the experiment ran on (build_api15).
H=/home/dev/local/hill2
cd $H
for t in tc_13 tc_15 tc_16 tc_10 vm_04 vm_07 vm_17 vm_20; do s=${t%%_*}
  [ -s acc/out/probe4_$t.json ] || HILL_CACHE=$H/acc/cache_$s HILL_CORES=12-19 ./acc/gpu2 python -I acc/probe4.py --tree acc/trees/$t --config acc/trees/$t/hill_config.py --out acc/out/probe4_$t.json --label $t > acc/out/probe4_$t.log 2>&1
  echo "$t: $(python3 -I -c "
import json,os
p='acc/out/probe4_$t.json'
print({k[:44]:('raised' if 'raised' in v else v.get('followed')) for k,v in json.load(open(p))['probes'].items()} if os.path.exists(p) else 'FAILED')")"
done
echo "LANE5 DONE"
