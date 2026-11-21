# Side `tc` (torch.compile): summary

The deadline has passed. The start was 1954 images/s. The final state measures **3014 and 3015 images/s** with the checked config, and **3551** with the variant that gives up eager behaviour.

Higher rows exist (3048 checked, 3612 unchecked), but they sit on a source that breaks one Inductor GPU test; I found and fixed that late. Details below.

## Read these before the numbers

- **Six rows are on a source with a test regression of mine.** The checked rows 2966, 3046, 3048 and the unchecked rows 3524, 3570, 3612 fail `test_require_stride_expanded_cuda`. I only found it at 23:55 by running the whole `GPUTests` suite. The fixed-source rows are 3014, 3015 (checked) and 3551 (unchecked).
- **The final checked state was measured twice, and the second run was triggered by the first.** I re-measured because 3014 read below 3048. It came back 3015. Treat them as two samples of one state, not as a best-of.
- **Noise is about 1–2.5%.** The three runs inside one official row spread up to 2.5%, and single processes of identical code differed by about 2% in GPU time. Only the +32.6%, +6.4%, +7.5% and +2.7% steps are clearly outside that.
- **One official attempt does not count.** The 3416 row was refused by the harness: my guard filter dropped every guard and a batch of 3 raised.
- **The 2757 row is partly contaminated.** Its third process ran while I was editing `lowering.py`; the two clean runs were 2757 and 2754.
- **All checked rows from 2757 on do not follow one thing eager does.** Reassigning `__class__` of a nested module is not followed. It comes with `use_recursive_dict_tags_for_guards` itself (the unmodified tree behaves the same with that flag, and follows it without).
- **A `model.train()`/`model.eval()` round trip costs the final checked config about 10% afterwards.** One process: 3010 before, 2713 after. Without hook guards the loss is about 5% (3052 to 2906).
- **Rows 2591 and 2757 lose speed after another input shape, as the start state does.** Coordinator's `after.py`, one process each: start config 2066 → 1845 → 1189; the 2591 config 2600 → 1936 → 1245. Rows from 2964 on keep their speed (final tree: 3060 → 3049 → 3047 → 3046).
- **I could not run check 6** (the unlisted things), and I do not know the coordinator's thresholds.

## Steps that raised the throughput (checked config)

| Row | Change | Step |
|---|---|---|
| 2591 | CUDA graphs, outputs cloned in config | +32.6% |
| 2757 | unfused `torch.cat` in the YOLO head; recursive dict tags for guards | +6.4% |
| 2964 | separable max pool; `module.__class__` tag-safe; hook guards on; `dynamic=False` | +7.5% |
| 2992 | 1-d pool passes unrolled; `automatic_dynamic_shapes=False` instead of `dynamic=False` | +0.9% (noise) |
| 2966 | RGB input of the first conv padded to 4 channels | −0.9% (noise) |
| 3046 | filter count rounded up to a multiple of 8 | +2.7% |
| 3048 | `copy_` for the single CUDA-graph input | +0.0% |
| 3014, 3015 | RGB pad only when the input is copied to channels-last anyway (the fix) | same graph, noise |

1. **CUDA graphs.** One profile showed 3.78 ms of the 3.92 ms call was CPU time launching about 330 kernels; GPU work was about 2.6 ms. Outputs are cloned so earlier results stay valid, which costs 0.9% (in-process A/B).
2. **Cat option and dict tags.** `max_complex_pointwise_cat_inputs=0` splits a head kernel that evaluated every branch for every element (0.134 → 0.04 ms); A/B on the final tree gives +2.7%. Dict tags cut the guard check from 202 to 130 µs at the time.
3. **Max pool, guards, hooks, static shapes.**
   - Inductor: a max pool with unused indices runs as one 1-d pass per dimension (SPP pools were 0.124 ms).
   - Dynamo: `module.__class__` and `cls.__name__` count as tag-safe, so the model's `module.__class__.__name__` no longer keeps every Sequential off the fast path. This is a CPU-lab number only (120 → 78 µs); I never isolated it in the real flow.
   - Hook guards make hooks registered after compilation followed, which stock torch.compile does not do. They cost 1.4% (A/B). They are only that cheap because of dict tags: with hook guards on, turning dict tags off costs 11.7%.
   - Static recompiles add nothing at the timed shape; they keep the speed after other shapes.
4. **Unrolled pool passes.** Kernel profile: 0.07 → 0.014 ms. The official step is inside noise.
5. **cuDNN paddings** in the conv lowering for channels-last half-precision convs.
   - RGB input padded to 4 channels: first conv 0.101 → 0.076 ms in the kernel profile.
   - Filter count rounded up to a multiple of 8 with zero filters: the three 255-channel head convs took 55/30/22 µs alone in cuDNN and 9/5/4 µs with 256 filters.
6. **`copy_` for the CUDA-graph input.** No measurable gain.

## The unchecked line

These rows give up eager behaviour, and each official note says so.

| Row | Config | Source |
|---|---|---|
| 3327 | freezing + `install_free_tensors` + tensor guards only, no output clones | before the conv changes |
| 3405 | guards only on the call's arguments and global state | before the conv changes |
| 3524 | same, with the padded convs | pre-fix |
| 3570 | + `triton.cudagraph_trees=False` | pre-fix |
| 3612 | + `max_pointwise_cat_inputs=0` | pre-fix |
| 3551 | same as 3612 | fixed |

On the coordinator's `probe.py` this variant follows 4 of 9 probes: the last conv's bias, the batch of 3, the other image size, and the timed shape again. The harness's `follows_weight_update: true` is only that unfolded bias. After `train()`/`eval()` it is also not right afterwards.

## What I verified on the final state

- **`check tc`:** 8 runs across the checked states, each 10 of 10 probes followed; mAP 0.7526–0.7532 against eager's 0.7529–0.7536. The final state: 0.7529 against 0.7529.
- **Cross-cache:** mAP 0.7532 against eager's 0.7535 on a cache filled by the random-weight model.
- **Training:** torch.compile is no further from eager than eager is from itself. Mean loss difference over the last 50 steps is 0.34 and 0.44 against eager, and 0.53 between two eager runs (one pair each).
- **Tests on the final tree:**
  - `test_misc.py`: 1013 tests, 1 failure.
  - `test_modules.py`: 211 OK.
  - `GPUTests`: 1633 tests, 9 not passing.
  - `test_cudagraph_trees.py`: 1 error.
  
  All eleven non-passing tests reproduce on the unmodified tree.
- **Earlier states were tested less.** The 2964 and 2992 trees only saw the max-pool GPU tests and the two Dynamo files.
- **My own probes on the checked config** (pretrained weights): `.data` updates, `load_state_dict`, a hook added later, outputs held across calls, another thread, `augment=True`, `inference_mode` and autocast all match eager.

## What did not help

- **`cudagraph_trees_generation_cloning`:** slower (2519), because the model stores grid tensors on modules and three storages get cloned every call.
- **`freezing` alone:** 2371; nothing is folded and all 375 inputs are copied per call.
- **`install_free_tensors` in the checked config:** no gain, and a replaced Parameter is no longer followed.
- **Combo kernels:** 1234 plain, 1.001 with benchmarking.
- **Other config knobs by A/B:**
  - `aggressive_fusion`: 0.997
  - `max_autotune_pointwise`: 0.987
  - `efficient_conv_bn_eval_fx_passes`: 0.964
  - `max_pointwise_cat_inputs=0` in the checked config: 0.982
- **Weight layout copies as one foreach launch:** slower for the big weights (0.161 against 0.057 ms).
- **Python GC:** no effect (2982 against 2986).
- **Plain CUDA graphs for the checked config:** rejected without measuring; that mode never notices a moved parameter.

## What I would do next

- **The 0.3 ms of CPU before the graph launch.** Without a sync between calls the same callable runs 3451 images/s. The cost is guards (about 0.1 ms), collecting 374 parameters, the static-pointer check and wrappers. A single C++ pass over static inputs would cover most of it.
- **Keep the dict-tag fast path alive across `train()`/`eval()`**, and close the nested `__class__` hole; both are in the C++ guard code.
- **The 38 weight layout copies** (0.14 ms): fuse them with per-kernel block sizes.
- **The batch-norm + LeakyReLU passes** (0.59 ms) are at memory bandwidth; only a conv kernel with that epilogue removes them.
- **Run the whole `GPUTests` suite before every source measurement**, not a name pattern. That would have caught the regression before six rows were recorded on it.

## Where things are

Everything is in `/home/dev/local/hill2/tc`:
- `journal.md` (69 lines)
- `config.py` (the checked config)
- `log.jsonl`
- `work/` (scripts, saved outputs, the alternative configs)

The four changed source files are under `tree/torch`:
- `_inductor/lowering.py`
- `_inductor/kernel/conv.py`
- `_inductor/cudagraph_trees.py`
- `_dynamo/guards.py`