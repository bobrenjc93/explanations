#!/bin/bash
# lane4.sh: each side's tests on every recorded checkpoint (the rules: a checkpoint that breaks its side's tests is
# struck). tc: the Inductor GPU tests of convolutions, max pools, cat and the stride test, per checkpoint tree;
# vm: test/test_vm.py on the CPU from each checkpoint's own tree. Untimed; its own cores.
H=/home/dev/local/hill2
O=$H/acc/out
export PATH=/home/dev/local/g/pytorch-env/bin:$HOME/.cargo/bin:$PATH
boot() {  # boot TREE -> a directory whose sitecustomize puts TREE first
  local b=$H/acc/boots/$(basename $1); mkdir -p $b
  printf 'import sys\nsys.meta_path[:] = [f for f in sys.meta_path if type(f).__name__ != "ScikitBuildRedirectingFinder"]\nsys.path.insert(0, "%s")\n' "$1" > $b/sitecustomize.py
  echo $b
}
for t in $(ls -d $H/acc/trees/tc_* | sort); do n=$(basename $t)
  [ -s $O/ckpt_tests_$n.log ] && continue
  (cd /home/dev/local/g/pytorch && PYTHONPATH=$(boot $t) TORCHINDUCTOR_CACHE_DIR=$H/acc/cache_tests_tc timeout 1500 taskset -c 24-31 python3 -I /home/dev/local/cfgloop/guard/safe_run.py 48 \
     python test/inductor/test_torchinductor.py GPUTests -k require_stride_expanded -k conv -k max_pool -k cat > $O/ckpt_tests_$n.log 2>&1)
  echo "$n: $(grep -E '^(Ran|OK|FAILED)' $O/ckpt_tests_$n.log | tr '\n' ' ')"
done
for t in $(ls -d $H/acc/trees/vm_* | sort); do n=$(basename $t)
  [ -s $O/ckpt_tests_$n.log ] && continue
  (cd $t && CUDA_VISIBLE_DEVICES="" PYTHONPATH=$(boot $t) timeout 1500 taskset -c 24-31 python3 -I /home/dev/local/cfgloop/guard/safe_run.py 48 python test/test_vm.py > $O/ckpt_tests_$n.log 2>&1)
  echo "$n: $(grep -E '^(Ran|OK|FAILED)' $O/ckpt_tests_$n.log | tr '\n' ' ')"
done
echo "LANE4 DONE"
