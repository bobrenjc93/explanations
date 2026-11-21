# Side tc: torch.compile

- Your tree: `/home/dev/local/hill2/tc/tree`. `tree/torch` is a complete copy of the Python of `torch` at the
  commit both sides start from (Dynamo in `_dynamo`, AOTAutograd in `_functorch`, Inductor in `_inductor`); the
  compiled libraries are links to the shared build. `/home/dev/local/hill2/run tc ...` makes Python import this
  tree (check: `torch.__file__`). Edits take effect at the next process. `tree/torch/csrc` is a link to the shared
  build's sources: do not edit those; the C++ cannot be rebuilt.
- Tests (run from `/home/dev/local/g/pytorch`, which has the test files):
  `/home/dev/local/hill2/run tc python test/dynamo/test_misc.py` and
  `/home/dev/local/hill2/run tc python test/dynamo/test_modules.py` on the CPU;
  `/home/dev/local/hill2/run tc --gpu python test/inductor/test_torchinductor.py GPUTests -k PATTERN` for
  what you changed in Inductor (pick patterns that fit in the 15 minutes).
- Useful: `TORCH_LOGS` (graph_breaks, recompiles, guards, output_code, perf_hints),
  `TORCH_COMPILE_DEBUG=1`, `torch._dynamo.utils.counters`, `torch.profiler`, the documentation in
  `/home/dev/local/g/pytorch/docs/source` (torch.compiler*.md).
- The model: `/home/dev/local/g/benchmark/torchbenchmark/models/yolov3` (read only).
