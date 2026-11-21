# Side vm: torch.vm

- Your tree: `/home/dev/local/hill2/vm/tree`. `tree/torch` is a complete copy of the Python of `torch` at the
  commit both sides start from; the compiled libraries are links to the shared build. `torch.vm` is
  `tree/torch/vm/*.py` (the API, the passes, the export of a graph to FX for passes written in Python) and a Rust
  engine whose crate is `tree/rust` (`src/*.rs`: the frontend from bytecode to a control-flow graph, the VM, its
  passes, and in `graph.rs` and `vm.rs` the builder that takes down graphs of tensor operations along the path a
  call takes). Build the Rust after a change: `/home/dev/local/hill2/vm/build_rs.sh` (about a minute; it installs
  the library your tree loads). `tree/torch/csrc` is a link to the shared build's sources: do not edit those; the
  C++ cannot be rebuilt. You may also change the Python of AOTAutograd and Inductor in your tree.
  `/home/dev/local/hill2/run vm ...` makes Python import this tree (check: `torch.__file__`). Edits of Python take
  effect at the next process.
- Documentation: `/home/dev/local/hill2/vm/docs/vm.md` (the user documentation: passes, graphs, paths, effects,
  limits), the docstring of `tree/torch/vm/__init__.py`, and `/home/dev/local/hill2/vm/docs/builder_state.md` (the
  working notes of whoever wrote the graph builder: its structures, the rules that keep it right, what ends a
  path, measurements, and what was found wrong and fixed).
- Tests (run from `/home/dev/local/hill2/vm/tree`): `/home/dev/local/hill2/run vm python test/test_vm.py` on the
  CPU and `/home/dev/local/hill2/run vm --gpu python test/test_vm.py` with the GPU.
- Useful: `torch.vm.regions(fn)` (the graphs, their forms, where and why each ends), `fn.stats()`,
  `fn.dump()`, `torch.vm.passes`, `torch.vm.DEFAULT_PASSES`, `torch.vm.is_exact`, `TORCH_VM_GRAPH_DEBUG=1`,
  `TORCH_VM_GRAPH_TIMES=1`, `torch.profiler`.
- The model: `/home/dev/local/g/benchmark/torchbenchmark/models/yolov3` (read only).
