#!/bin/bash
# run_all.sh: the coordinator's validation of the run, after the deadline, in order of importance. Every step
# skips what is already done, so it can be started again.
H=/home/dev/local/hill2
cd $H
mkdir -p acc/out acc/trees
sed -i 's#^H=/home/dev/local/hill$#H=/home/dev/local/hill2#' acc/cross_cache.sh

# 0. The final checkpoint of each side: predictions, probes, the attribute probe.
python3 -I acc/validate.py --what infer,probe,probe2 --last
eval "$(python3 -I - <<'PY'
import json
H = "/home/dev/local/hill2"
for side in ("tc", "vm"):
    off = [r for r in (json.loads(l) for l in open(f"{H}/{side}/log.jsonl")) if r["kind"] == "official"]
    ok = [(i, r) for i, r in enumerate(off) if r["eligible"]]
    best = max(ok, key=lambda t: t[1]["qps"])[0]
    last = ok[-1][0]
    print(f"{side.upper()}_BEST={side}_{best:02d}; {side.upper()}_LAST={side}_{last:02d}; {side.upper()}_BEST_SHA={off[best]['sha']}")
PY
)"
for t in $TC_BEST $VM_BEST; do s=${t%%_*}
  [ -d acc/trees/$t ] || python3 -I tools/mktree.py $s $( [ $s = tc ] && echo $TC_BEST_SHA || echo $VM_BEST_SHA ) acc/trees/$t > /dev/null
done
[ -d acc/trees/tc_00 ] || python3 -I tools/mktree.py tc $(git -C tc/tree rev-list --max-parents=0 HEAD) acc/trees/tc_00 > /dev/null
[ -d acc/trees/vm_00 ] || python3 -I tools/mktree.py vm $(git -C vm/tree rev-list --max-parents=0 HEAD) acc/trees/vm_00 > /dev/null
echo "best: $TC_BEST $VM_BEST; last: $TC_LAST $VM_LAST"

# 1. The same state compiled for the random-weight model first and for the pretrained one second, with one cache.
for t in $TC_BEST $TC_LAST $VM_BEST $VM_LAST; do
  [ -s acc/out/xcache_${t}_pretrained.json ] || ./acc/cross_cache.sh $H/acc/trees/$t $H/acc/trees/$t/hill_config.py $t
done

# 2. Predictions and probes of the best states, then speed after events for best and final states.
for t in $TC_BEST $VM_BEST; do s=${t%%_*}
  for kind in infer probe probe2; do
    [ -s acc/out/${kind}_$t.json ] || HILL_CACHE=$H/acc/cache_$s ./acc/gpu python -I acc/$kind.py --tree acc/trees/$t --config acc/trees/$t/hill_config.py --out acc/out/${kind}_$t.json --label $t > acc/out/${kind}_$t.log 2>&1
  done
done
for t in $TC_BEST $TC_LAST $VM_BEST $VM_LAST; do s=${t%%_*}
  [ -s acc/out/after_$t.json ] || HILL_CACHE=$H/acc/cache_$s ./acc/gpu python -I acc/after.py --tree acc/trees/$t --config acc/trees/$t/hill_config.py --out acc/out/after_$t.json --label $t > acc/out/after_$t.log 2>&1
  echo "after $t: $( [ -s acc/out/after_$t.json ] && echo done || echo FAILED )"
done
echo "TIMED DONE"

# 3. Training with the starting and the final sources, against eager.
train() {  # train NAME TREE MODE FLAGS...
  name=$1; tree=$2; mode=$3; shift 3
  [ -s acc/out/train_$name.json ] && return
  HILL_CACHE=$H/acc/cache_train ./acc/gpu python -I acc/train.py --tree $tree --mode $mode --steps 300 --out acc/out/train_$name.json --label "$name" "$@" > acc/out/train_$name.log 2>&1
  echo "train $name: $( [ -s acc/out/train_$name.json ] && echo done || echo FAILED )"
}
P=$H/acc/trees/tc_00
train det_eager_a $P eager --deterministic
train det_eager_b $P eager --deterministic
train det_eager_nudge_1e-6 $P eager --deterministic --nudge 1e-6
train det_tc_final $H/acc/trees/$TC_LAST tc --deterministic
train det_vm_final $H/acc/trees/$VM_LAST vm --deterministic
train det_tc_start $H/acc/trees/tc_00 tc --deterministic
train det_vm_start $H/acc/trees/vm_00 vm --deterministic
train eager_a $P eager
train eager_b $P eager
train tc_final $H/acc/trees/$TC_LAST tc
train vm_final $H/acc/trees/$VM_LAST vm
train amp_eager_a $P eager --amp
train amp_tc_final $H/acc/trees/$TC_LAST tc --amp
train amp_vm_final $H/acc/trees/$VM_LAST vm --amp
echo "TRAINING DONE"

# 4. Every other checkpoint that counted: predictions, probes, the attribute probe.
python3 -I acc/validate.py --what infer,probe,probe2
echo "CHECKPOINTS DONE"

# 5. Each side's tests on its final source.
O=$H/acc/out
(cd $H/vm/tree && $H/run vm python test/test_vm.py > $O/tests_vm_cpu.log 2>&1; $H/run vm --gpu python test/test_vm.py > $O/tests_vm_gpu.log 2>&1)
(cd /home/dev/local/g/pytorch && $H/run tc python test/dynamo/test_modules.py > $O/tests_tc_modules.log 2>&1; $H/run tc python test/dynamo/test_misc.py > $O/tests_tc_misc.log 2>&1; \
  $H/run tc python test/dynamo/test_guard_manager.py > $O/tests_tc_guard_manager.log 2>&1; \
  $H/run tc --gpu python test/inductor/test_torchinductor.py GPUTests -k conv -k max_pool -k cat > $O/tests_tc_inductor_gpu.log 2>&1)
for f in $O/tests_*.log; do echo "$(basename $f): $(grep -E '^(Ran|OK|FAILED)' $f | tr '\n' ' ')"; done
echo "ALL DONE"
