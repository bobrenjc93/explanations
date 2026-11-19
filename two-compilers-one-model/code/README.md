# One model, two compilers: harness, checks and page generator

Everything here is as it ran, on one machine with one H100; paths are that machine's.

## The experiment

- `RULES.md`: the rules both agents were given (the same file), with `README_tc.md` and `README_vm.md` for each side's tree and tools.
- `bench.py`: one process of the measurement. TorchBench yolov3 as the performance dashboard sets it up (eval, batch 8 of 3x384x512,
  bfloat16, `no_grad`), through whatever `build(model, example_inputs)` in a side's `config.py` returns; warm-up, 400 timed calls on
  four inputs, outputs checked against eager, three probes.
- `measure.py` (`measure`): a recorded measurement: snapshot of the side's state, three processes of `bench.py`, the median, a row in the
  side's `log.jsonl`. Holds the GPU lock.
- `run`: runs a command in one side's environment, with or without the GPU lock.
- `tools/overlay.py`: a standalone copy of the Python of `torch` for the `torch.compile` side (Python files copied, binaries linked), so the
  two sides do not share source. `tools/mktree.py`: the standalone tree of any recorded checkpoint. `tools/tokens.py`: what each agent had
  spent (tokens, minutes) at each recorded measurement, from the agents' transcripts.

## The checks made afterwards (`acc/`)

- `infer.py`: predictions with the original pretrained yolov3-spp weights (Darknet release) on the 128 COCO images that ship with the
  benchmark: mAP@0.5 against the ground truth, the detections against eager's, the raw output against float32.
- `probe.py`: what a compiled callable still follows once it is warm (weights updated in place, a replaced parameter, `model.train()`,
  other shapes). `after.py`: its speed after such events.
- `train.py`: the loss curve of training the model through eager, `torch.compile` or `torch.vm`, with `--deterministic` and `--nudge`.
- `validate.py`: runs `infer.py` and `probe.py` over every recorded checkpoint.

## The page

`make.py` reads the logs and the outputs of the checks and writes the page; `appendix.py` is the appendix (every step explained), `charts.py` the small SVG chart kit.

## Results (`../results`)

- `tc/` and `vm/`: each side's final `config.py`, its journal, its `log.jsonl` (every measurement) and `final.patch` (its changes to the
  compiler's source against the starting tree).
- `tokens.json`: time and tokens at every recorded measurement. `out/`: the outputs of the checks, one JSON per checkpoint and check.
