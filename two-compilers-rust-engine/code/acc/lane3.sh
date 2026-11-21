#!/bin/bash
# lane3.sh: the unannounced probes (probe3.py) on every recorded state, beside the other per-checkpoint checks.
# Starts when the timed training runs are done; no lock, its own cores.
H=/home/dev/local/hill2
cd $H
until grep -q "TRAINING DONE" acc/run_all.out; do sleep 20; done
HILL_RUNNER=$H/acc/gpu2 HILL_CORES=12-19 python3 -I acc/validate.py --what probe3 > acc/lane3.out 2>&1
echo "LANE3 DONE" >> acc/lane3.out
