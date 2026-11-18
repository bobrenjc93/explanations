# The chunks of CPython's frame stack under a compiled graph

What built the page in `../index.html`, and what it is built from.

| file | what it is |
|---|---|
| `make_page.py` | the prose of the page and what each figure is made of; `python make_page.py ../index.html` rebuilds the page from `../results/` |
| `charts.py` | the static-SVG chart kit the page uses (the one of `../../cfg-compile/code/`) |
| `tools/` | what measured: the plain-Python reproduction, the depth scan, the frame listing, the allocator wrappers and the `sitecustomize.py` that installs one without adding a frame; see `tools/README.md` |

`../results/`:

| file | what it is |
|---|---|
| `depth.json` | one training step of tf_efficientnet_b0 in the dashboard runner, timed at growing depths of the caller: `torch.rewrite` before the freed chunks were kept (every depth to 139, and every fourth to 396), `torch.compile`, and `torch.rewrite` with the chunks kept |
| `cliff.json` | `tools/chunk_cliff.py` under CPython 3.11.16, 3.12.14, 3.13.15 and 3.14.7: page faults and nanoseconds a call at each of 200 depths |
| `chains.json` | the frames on the stack at the first kernel launch of the forward graph: for the two runs that map chunks, as the interpreter has them at the moment a chunk is asked for (with the bytes the chunk had left, which lines of the generated code asked, and how many chunks each repetition asked for); for tf_efficientnet_b0 under `torch.compile`, from `sys._getframe` |
| `chunks_training.log`, `chunks_inference.log` | per run of the dashboard's suites, the chunks its timed loop asked for (see `tools/README.md` for the fields); the last eight lines of the training log are the second pairs of `torch.compile`'s four models, with the chunks kept for its process |
| `pairs.json` | the geometric means of the suites, and the six models' steps before the fix, with it, and in their second pairs |

The per-model results of the suites are in `../../cfg-compile/results/suites_r25e/` and `suites_r25e_train/`.
