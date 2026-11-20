# The experiment, and its rules (the same for both sides)

Two agents work at the same time, each on one compiler, on the same model, with the same time budget:
side `tc` improves `torch.compile`, side `vm` improves `torch.vm`. What is measured is the throughput of
TorchBench `yolov3` inference (images per second: eval mode, batches of 8 images of 3x384x512, bfloat16,
`torch.no_grad()`, one H100), as the PyTorch performance dashboard sets the model up. The result of the
experiment is the curve of throughput against the time and tokens an agent had spent when it reached it.
So what counts is every official measurement on the way, not only the last one: **take an official
measurement as soon as something is better, then go on.** Small steps recorded early are worth more
than one big step recorded at the end.

Where each side started (official measurements, same machine): eager 800 images/s; `torch.compile`
2013 images/s; `torch.vm` 784 images/s.

## Measuring

- `/home/dev/local/hill/measure SIDE` : a quick look (one process, about 1 to 4 minutes). Not a result.
- `/home/dev/local/hill/measure SIDE --official "what changed, in a few words"` : a result (three processes: run it in the background if it may take more than a few minutes, and wait for it). Snapshots your
  state, runs three processes, takes the median, checks the outputs against eager, runs three probes
  (a weight updated in place, a batch of another size, the timed shape again) and appends a row to
  `/home/dev/local/hill/SIDE/log.jsonl`. It prints whether the row "counts as a result".
- What is timed is whatever `build(model, example_inputs)` in `/home/dev/local/hill/SIDE/config.py`
  returns: it is called as `fn(x)` under `torch.no_grad()`, after a warm-up of at least 300 calls.
- Official measurements are refused after the deadline in `/home/dev/local/hill/SIDE/deadline`
  (seconds since the epoch; `date -u +%s` is now). When it has passed, stop and write your summary.

## What you may change

1. `/home/dev/local/hill/SIDE/config.py`: anything a user of your compiler could write. Options, modes,
   settings of PyTorch, passes.
2. The source code of your compiler, in your side's tree (see your side's README). Python, and for
   `torch.vm` its Rust engine. No C++: `pip install` and any other build of PyTorch are off limits.

## What you may not change, or do

- The model (TorchBench's yolov3, its weights, its inputs), the harness (`bench.py`, `measure`,
  `measure.py`, `run`, anything in `/home/dev/local/hill` outside your side's directory), eager PyTorch.
- The other side's directory, and any git repository other than through `measure` (no commits, no
  stash, no checkout in `/home/dev/local/g/pytorch` or any other PyTorch checkout; leave your changes
  in the working tree, `measure --official` snapshots them).
- Your compiler must be the one doing the work: side `tc` runs the model through `torch.compile` with
  the `inductor` backend; side `vm` runs it through `torch.vm`, graphs through its Inductor backend.
  Neither uses the other, and neither uses `torch.jit`, `torch.export`/AOTInductor, ONNX or TensorRT.
- The callable must compute what the model computes: no remembering outputs for inputs seen, no lower
  precision than bfloat16, nothing skipped or approximated. Results are checked against eager with the
  dashboard's rule for bfloat16.
- A change to the compiler must be a correct change to the compiler: right for programs in general,
  with a guard or a fallback where its assumption does not hold, and not keyed on this model's name or
  identity. Your side's tests (see README) must still pass; run them before an official measurement
  that changes source code. At the end every checkpoint's tests are run again by the coordinator, and
  a checkpoint that breaks them is struck from the result.
- If a step gives up something eager does (for example: an in-place weight update is no longer
  followed, which the probe reports), it still counts, but say so in the note of that measurement.

## The machine

- One GPU for both sides. Anything that touches the GPU goes through
  `/home/dev/local/hill/run SIDE --gpu COMMAND...`, which waits for the lock and stops the command after
  15 minutes; `measure` takes the lock itself. Keep GPU jobs short: the other side waits while you hold it.
- Everything else runs through `/home/dev/local/hill/run SIDE COMMAND...` (no GPU visible, your side's
  12 cores, your side's environment). Use `run` for every Python command: it is what selects your tree.
- The container is killed as a whole when it runs out of memory. Nothing wider than 12 processes; wrap
  anything that may hold tens of GB in `python3 -I /home/dev/local/cfgloop/guard/safe_run.py 48 COMMAND...`.
- Never `pkill -f`; stop processes by pid. Scratch files go in `/home/dev/local/hill/SIDE/work/`.

## What to leave behind

- `/home/dev/local/hill/SIDE/journal.md`: one line per attempt as you go (UTC time, what you tried, what
  it measured, kept or dropped). Dead ends too: the path is part of the result.
- When the deadline has passed: a summary as your final message. In order: each step that raised the
  throughput (what it was, why it helped, the measured gain), what you tried that did not help, and
  what you would do next with more time. Plain statements, with numbers.

## The second half (hours 4 to 6)

The experiment was extended by three hours for both sides. The rules above are unchanged; this section adds what
both sides are told for the second half.

- The new deadline is in `/home/dev/local/hill/SIDE/deadline`. Your tree, your `config.py`, your `work/` and your
  journal are as you left them; the log goes on.
- After the first half the coordinator checked every recorded state of both sides:
  1. its predictions with the original pretrained yolov3-spp weights on 128 COCO images (with the benchmark's random
     weights the output hardly depends on the convolutions, so the harness's own check against eager says little);
  2. what it still follows once it is warm: weights updated in place (a convolution's weight, a batch norm's bias
     and running mean, the last bias), a parameter replaced by a new one, `model.train()`, another batch size,
     another image size;
  3. the side's tests, and training loss curves against eager with the side's final source.
- The same checks will be run on everything recorded in the second half. The report will give two numbers for each
  side: the best recorded throughput, and the best recorded throughput among states that pass every check of 1 and 2.
  Both count; how you divide your time between them is yours to decide.
- `/home/dev/local/hill/check SIDE` runs 1 and 2 on your current state: about five minutes of the GPU, behind the
  lock, nothing recorded. Use it when a step could change results or what is followed.
- Do not read the other side's directory or tree, or anything published about this experiment: each side works from
  what it finds itself.
