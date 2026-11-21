#!/bin/bash
# lane4b.sh: the torch.vm half of lane4 (test/test_vm.py on the CPU from each checkpoint's tree), on other cores,
# so that it does not wait for the torch.compile half.
H=/home/dev/local/hill2
O=$H/acc/out
export PATH=/home/dev/local/g/pytorch-env/bin:$HOME/.cargo/bin:$PATH
for t in $(ls -d $H/acc/trees/vm_* | sort -r); do n=$(basename $t)
  [ -e $O/ckpt_tests_$n.log ] && continue
  b=$H/acc/boots/$n; mkdir -p $b
  printf 'import sys\nsys.meta_path[:] = [f for f in sys.meta_path if type(f).__name__ != "ScikitBuildRedirectingFinder"]\nsys.path.insert(0, "%s")\n' "$t" > $b/sitecustomize.py
  (cd $t && CUDA_VISIBLE_DEVICES="" PYTHONPATH=$b timeout 1500 taskset -c 12-19 python3 -I /home/dev/local/cfgloop/guard/safe_run.py 48 python test/test_vm.py > $O/ckpt_tests_$n.log 2>&1)
  echo "$n: $(grep -E '^(Ran|OK|FAILED)' $O/ckpt_tests_$n.log | tr '\n' ' ')"
done
echo "LANE4B DONE"
