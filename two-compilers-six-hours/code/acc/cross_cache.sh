#!/bin/bash
# cross_cache.sh TREE CONFIG LABEL: does a state stay right when Inductor's cache was filled by the same model with
# other weights? With a fresh cache directory: first the benchmark's random-weight model (after.py), then the
# pretrained one (infer.py), which finds what the first left in the cache.
H=/home/dev/local/hill
TREE=$1; CONFIG=$2; LABEL=$3
C=$H/acc/cache_x_$LABEL
mkdir -p $C
cd $H
HILL_CACHE=$C ./acc/gpu python -I acc/after.py --tree $TREE --config $CONFIG --out acc/out/xcache_${LABEL}_random.json --label "$LABEL" > acc/out/xcache_${LABEL}_random.log 2>&1
HILL_CACHE=$C ./acc/gpu python -I acc/infer.py --tree $TREE --config $CONFIG --out acc/out/xcache_${LABEL}_pretrained.json --label "$LABEL" > acc/out/xcache_${LABEL}_pretrained.log 2>&1
python3 -I - $LABEL <<'PY'
import json, os, sys
p = f"/home/dev/local/hill/acc/out/xcache_{sys.argv[1]}_pretrained.json"
if os.path.exists(p):
    r = json.load(open(p))
    print(sys.argv[1], "after a cache filled with other weights: mAP", round(r["compiled_bfloat16"]["mAP@0.5"], 4), "(eager", round(r["eager_bfloat16"]["mAP@0.5"], 4), ") rmse vs float32", round(r["compiled_against_float32"]["raw_output"]["rmse"], 3))
else:
    print(sys.argv[1], "FAILED")
PY
