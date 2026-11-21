#!/bin/bash
# lane4c.sh: the rest of the torch.compile checkpoints (those lane4.sh had not reached) with the one Inductor GPU
# test that tells the states apart (test_require_stride_expanded), so that the GPU is free sooner. The full set of
# convolution, pool and cat tests ran on tc_00 to tc_11, and runs on the final source in run_all.sh.
H=/home/dev/local/hill2
O=$H/acc/out
export PATH=/home/dev/local/g/pytorch-env/bin:$HOME/.cargo/bin:$PATH
for i in $(seq -w 0 16); do n=tc_$i; t=$H/acc/trees/$n
  [ -d $t ] || continue
  [ -e $O/ckpt_tests_$n.log ] && continue
  b=$H/acc/boots/$n; mkdir -p $b
  printf 'import sys\nsys.meta_path[:] = [f for f in sys.meta_path if type(f).__name__ != "ScikitBuildRedirectingFinder"]\nsys.path.insert(0, "%s")\n' "$t" > $b/sitecustomize.py
  (cd /home/dev/local/g/pytorch && PYTHONPATH=$b TORCHINDUCTOR_CACHE_DIR=$H/acc/cache_tests_tc timeout 900 taskset -c 24-31 python3 -I /home/dev/local/cfgloop/guard/safe_run.py 48 \
     python test/inductor/test_torchinductor.py GPUTests -k require_stride_expanded > $O/ckpt_tests_$n.log 2>&1)
  echo "$n (the stride test only): $(grep -E '^(Ran|OK|FAILED)' $O/ckpt_tests_$n.log | tr '\n' ' ')"
done
echo "LANE4 DONE"
