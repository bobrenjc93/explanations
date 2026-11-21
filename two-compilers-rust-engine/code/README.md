# One model, two compilers, run again: harness, checks and page generator

The second run of the experiment of `../../two-compilers-one-model` and `../../two-compilers-six-hours`. Everything is
as it ran, on one machine with one H100; paths are that machine's.

## The experiment

- `RULES.md`: the rules both agents were given (the same file), with the checks that were announced; `README_tc.md` and
  `README_vm.md` for each side's tree and tools.
- `bench.py`: one process of the measurement. TorchBench yolov3 as the performance dashboard sets it up (eval, batch 8
  of 3x384x512, bfloat16, `no_grad`), through whatever `build(model, example_inputs)` in a side's `config.py` returns;
  warm-up, 400 timed calls on four inputs, outputs checked against eager, three probes.
- `measure.py` (`measure`): a recorded measurement: a snapshot of the side's state as a commit, three processes of
  `bench.py`, the median, a row in the side's `log.jsonl`. `check.py` (`check`): the announced probes and the
  predictions on a side's current state. `run`: a command in one side's environment, with or without the GPU lock.
- `tools/overlay.py`: a standalone copy of the Python of `torch` for a side. `tools/mktree.py`: the standalone tree of
  a recorded snapshot. `tools/tokens.py`: what each agent had spent (tokens, minutes) at each recorded measurement,
  from the agents' transcripts.
- `tc/` and `vm/`: each side's `start_config.py` and final `config.py`, its `journal.md`, its `summary.md` (the
  agent's final report) and `final.patch` (its changes to the compiler's source, last snapshot against the first,
  without the rebuilt engine library).

## The checks made afterwards (`acc/`)

- `infer.py`: predictions with the original pretrained yolov3-spp weights on the 128 COCO images that ship with the
  benchmark. `probe.py`, `probe2.py`: what a compiled callable still follows once it is warm (announced to the agents).
  `probe4.py`: three probes that were not announced (a weight written through `.data`; every tensor of a result kept
  across two calls with other inputs, as tensors and through DLPack). `probe3.py` is their first version, kept for the
  record and not used by the page: two of its probes were too weak (they kept only the first tensor of a result, and
  the last call repeated the input of the kept one). `after.py`: speed after events. `cross_cache.sh`: a state compiled
  for the random-weight model first and the pretrained one second, with one compile cache. `train.py`: the loss curve
  of training through eager, `torch.compile` or `torch.vm`.
- `validate.py` runs the per-row checks over every recorded row; `run_all.sh` and the `lane*.sh` scripts are the order
  everything was run in after the deadline (`lane3.sh`: `probe3.py`; `lane4*`: each side's tests on every row's
  source, `../results/ckpt_tests_NOTE.txt` says which tests ran on which row; `lane5*`: `probe4.py`).

## The page

`make.py` reads the logs, the snapshots and the outputs of the checks and writes the page; `text.py` is its prose,
`appendix.py` the appendix (every step explained), `charts.py` the small SVG chart kit, `verify.py` a listing of what
the generator read for every row, `stage.py` what made this folder, `shot*.js` the screenshots used to look at the
result.

## Results (`../results`)

- `tc/log.jsonl`, `vm/log.jsonl`: every measurement. `tokens.json`: time and tokens at every recorded measurement.
- `out/`: the outputs of the checks, one JSON per row and check. `out_probe3_first_version/`: the outputs of the first
  version of the unannounced probes, which the page does not use. `tests.json`: what each side's tests gave on the
  source of every row, and which rows are struck for it; `ckpt_tests_NOTE.txt`: which tests those were.
- `by_time.png`, `by_cost.png`: the two charts.
