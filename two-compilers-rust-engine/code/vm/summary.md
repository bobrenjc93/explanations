The deadline has passed. Final state: 21 official rows, all eligible; the tree equals the last snapshot; `config.py` is the safe configuration.

## Result

| | images/s | state |
|---|---|---|
| start (dashboard configuration) | 1825 | |
| best recorded | **3384** | snapshot `da8b72ab`; gives up two things (step 5) |
| best recorded that follows everything I know of | **3218** | snapshot `719739b1` |
| last row (final source, safe configuration) | 3158 | snapshot `2f3d4fe2` |

The 3218 was not reproduced later: the same snapshot re-run through `bench.py` gave 3125, and every later safe measurement gave 3145–3162. A cool idle GPU and an empty Inductor cache made no difference; I did not find the cause.

## Steps that raised the throughput

1. **CUDA graph around what Inductor compiled: 1825 → 2405, at 22 min.** New option `Inductor(cuda_graphs=True)`, in `torch/vm/_cudagraphs.py`. The callable was CPU-bound: launching the 328-operation graph took 4.10 ms per call while the GPU was busy only 2.53 ms. Weights are read in place and checked by address each call, the input is copied in, results are copied out. Graphs that write to their inputs or record gradients run as before.
2. **Graph started ahead of the engine's checks: 2405 → 2869, at 30 min.** The engine spent 0.54 ms per call replaying the path's 13,604 instructions before the graph and 0.13 ms after, with the GPU idle. It now starts the replay first and runs the checks while the GPU works; a call that turns out not to match discards the run. The callable became GPU-bound.
3. **`SplitPools` pass: 2869 → 2971, at 43 min.** A max pool over a window of 16 or more elements becomes two one-dimensional pools, bit-exact. The three pools of the SPP block went from 123 µs to 15 µs.
4. **Values a form keeps: 2971 → 3218, at 77 min.** The three detection convolutions were graph outputs only because `p.shape` and `p.device` were read from them afterwards. The engine now keeps such per-form constants itself, so those outputs disappear, with their extra NCHW copies and 16 MB of clones. Kernel time went from 2600 µs to 2465 µs.
   - The same snapshot fixes a case where `torch.meshgrid`'s once-only warning, shown inside the first profiled call, split the graph for good (2300 images/s in a process that had not run eager first).
5. **Fast configuration: 3338 at 115 min, 3384 at 166 min; later samples 3353–3378.** Two options, both off by default:
   - `weight_layout="keep"` keeps channels-last copies of the 3×3 conv weights instead of making them every call (116 µs). **Gives up:** a write through `weight.data` moves no version counter and is not followed. In-place ops, `copy_`, views, `detach()`, replaced or moved weights are followed.
   - `hand_out=True` hands large results out without a copy (about 40 µs). **Gives up:** a raw address of a result kept across calls (DLPack, `__cuda_array_interface__`) sees the next call write it.
   - The 3384 row had hand-out always on, and its note names only the `.data` hole; the journal records both.

`max_autotune_pointwise=True` in `config.py` gave +0.7% in my own loop but nothing visible officially (safe 3159 against 3146–3152); it is kept.

## What did not help

- `cudnn.benchmark_limit=0`: convolutions 1519 µs against 1525 µs.
- `cudnn.benchmark=False`: 1805 images/s.
- Two CUDA graphs, the second launched while the first runs: 2388–2410 µs per call against 2356 µs.
- First convolution as a fused Triton kernel: 718 µs against 102 + 67 µs.
- Deep convolutions in NCHW to skip the weight copies: 58 µs against 30 + 12 µs.
- GPU fingerprints of the weights, to make kept layouts safe: 130–160 µs, against 144 µs for the copies.
- Channels-last input buffer: 3125 against 3164 images/s.
- Batch norm with precomputed scale and shift: no change; those kernels are memory-bound.
- `weight_layout="view"` (1×1 weights only): nothing; Inductor did not copy those anyway.

## Checks I ran

- **`check vm`** on the snapshot behind the 2869 row, the 3218 state, the final safe state and the fast configuration: all ten probes followed; mAP 0.7526–0.7535 against eager's 0.7529–0.7536. I did not run it on the first CUDA-graph snapshot (the 2405 row) or the SplitPools snapshot (the 2971 row).
- **My own probe** (`work/myprobe.py`, 46 checks, pretrained weights): the safe configuration passes all; the fast one fails only the `.data.mul_` case it gives up. It does not test a kept raw address, the other thing the fast configuration gives up.
- **Speed afterwards** (`acc/after.py`): safe 3139 at first, then 3159 / 3163 / 3168; fast 3354, then 3355 / 3360 / 3362.
- **Cross-cache** on my own paths: mAP 0.7530 against 0.7536 (safe) and 0.7530 against 0.7531 (fast).
- **Training** (`acc/train.py --mode vm`, 40 steps): runs; it compiles the same two small graphs as the start tree.
- **Tests:** two tests error on both CPU and GPU (`test_a_pass_written_in_rust_*`) because this tree layout has no `torch/csrc/cfg/rust/sdk`. They errored before any change of mine and pass when the path resolves. Everything else passes, including 11 new tests in `TestVmGraphExtras`.

## What I would do next

- **Make `hand_out` safe.** Leave a still-held result its memory and re-record the graph, switching to copies after two such calls. This passed my whole probe in the dev copy (3182 images/s while results are dropped) but is not in the tree; it is worth about 1.3–1.6% for the safe configuration.
- **A path through `zip(*list)`**, so the final `torch.cat` is inside the graph: about 0.5–1%.
- **A faster engine front** for calls that are not started ahead. Attribute reads are 37% of its instructions at about 50 ns each.
- **A write counter on storages** (needs C++) would make kept weight layouts safe: about 5%.
- **Find what made 21:26–21:28 UTC 2–3% faster.**

Everything is in `/home/dev/local/hill2/vm`: `journal.md` has one line per attempt with corrected UTC times, and `docs/vm.md` and `docs/builder_state.md` describe what was added.