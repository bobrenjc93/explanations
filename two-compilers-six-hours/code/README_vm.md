# Side vm: torch.vm

- Your tree: `/home/dev/local/g/pytorch`. `torch.vm` is `torch/vm/*.py` (the API, the passes, the regions
  that are handed to AOTAutograd and Inductor) and a Rust engine in `torch/csrc/cfg/rust/src/*.rs` (the
  frontend from bytecode to a control-flow graph, the VM, its passes). Build the Rust after a change:
  `bash /home/dev/local/cfgloop/build_rs_g.sh` (under a minute; it installs the library the tree loads).
  The C++ under `torch/csrc/cfg/*.cpp` cannot be rebuilt. You may also change the Python of AOTAutograd
  and Inductor in this tree. `/home/dev/local/hill/run vm ...` imports this tree (check: `torch.__file__`).
- Documentation: `/home/dev/local/g/pytorch/docs/source/vm.md`, the docstring of `torch/vm/__init__.py`,
  and `/home/dev/local/hill/vm/docs/report.md` (the engineering report: what is in the engine, what it
  does and does not do, how it performs).
- Tests (run from `/home/dev/local/g/pytorch`): `/home/dev/local/hill/run vm python test/test_vm.py`
  on the CPU (about 35 s, 408 tests), `/home/dev/local/hill/run vm --gpu python test/test_vm.py` on the GPU.
- Useful: `torch.vm.regions(fn)` (the graphs and their variants), `fn.stats()`, `fn.dump()`,
  `torch.vm.passes`, `torch.vm.DEFAULT_PASSES`, `torch.profiler`.
- The model: `/home/dev/local/g/benchmark/torchbenchmark/models/yolov3` (read only).
