# torch.rewrite: the report

Code behind https://bobrenjc93.github.io/explanations/cfg-compile/ . The compiler itself is the stack of draft PRs
pytorch/pytorch#204373 (bottom) to pytorch/pytorch#204763 (top); nothing of it is in this folder.

| file | what it is |
|---|---|
| `make_page.py` | the prose of the page and what each figure is made of; `python make_page.py ../index.html` rebuilds the page |
| `data.py` | reads `../results/*.jsonl` and derives every per-model number, geometric mean and count the page states |
| `charts.py` | the small static-SVG chart kit shared with the other pages of this site, plus the bars against a baseline and the stat tiles this page adds |

## Results (`../results`)

One JSON line per (model, test, compiler), as written by `benchmarks/rewrite/torchbench.py` of the stack
(`benchmarks/cfg_compile/torchbench.py` before the sixth version; the compiler's mode is still called `cfg` in the rows):
`call_s` is the median time of a call in the steady state, `compile_s` what the compiler reports for itself,
`warmup_s` the cost of the calls before the timed ones, `difference` the worst relative difference from eager's
outputs (losses for training), `graphs` the number of graphs, `paired_*` the two compilers timed in turns inside
the `cfg_compile` process, `peak_rss_mb` the most memory the process held, `peak_cuda_mb` (GPU runs) the most the process had
allocated on the GPU (`torch.cuda.max_memory_allocated`).

| file | what it is |
|---|---|
| `suites_r25c/` | v7 on one H100, on a build with cuDNN 9.26: what `benchmarks/dynamo/{huggingface,timm_models,torchbench}.py --performance --cold-start-latency --inference --bfloat16 --disable-cudagraphs` wrote under `--backend inductor` (`inductor_*.csv`), `--backend inductor --freezing` (`freezing_*.csv`) and `--rewrite` (`rewrite_*.csv`); `suites_r25d/` is TorchBench for inference again and `suites_r25d_train/` the three suites with `--training --amp`, both run model by model, `torch.compile` and `torch.rewrite` one right after the other |
| `gpu_r24.jsonl` | v7 in the stack's harness on one H100, inference only (its training rows are v6's), on a build WITHOUT cuDNN, like `gpu_r23.jsonl` and `gpu_first.jsonl` |
| `gpu_r23.jsonl` | v6 on one H100: inference (59 models) and training (47), each (model, compiler) in a process of its own, one at a time; the rows of hf_Longformer and hf_Reformer are from a rerun on the build with the iterator fix |
| `gpu_first.jsonl` | the first GPU run, on the build of v5, 39 models of inference before it was stopped; what the GPU section's "before" is |
| `final_r22.jsonl` | the CPU run (v5 on the page; v6 was not measured again on the CPU with the machine to itself, so the CPU sections keep these rows): the top of the stack, nine workers for inference and five for training under the memory guard; inference measured again in all three modes, so that every row has its peak memory, training rows of eager and `torch.compile` taken from the run of v4 |
| `memory_r22.log` | memory in use in the container during that run, every 30 s |
| `final_r21f.jsonl` | v4: memory kept per function, weights built in, one-row products. Thirteen of its training rows for `cfg_compile` have no `process_peak_rss_mb`: their `peak_rss_mb` includes `torch.compile` in the same process and is not comparable |
| `memory_r21f.log` | memory in use in the container during that run, every 30 s |
| `final_r21a.jsonl` | v4 before the weights were built in (memory and one-row products only), inference and the first training models |
| `final_r20.jsonl` | v3: sizes left open, effects replayed |
| `final_r19.jsonl`, `rerun_r19.jsonl`, `rerun2_r19.jsonl` | v2: regions across basic blocks; the reruns replace models that were timed before they had settled |
| `final_r18.jsonl` | v1: one region per basic block; its compile times are from a run where the compilers shared Inductor's kernel cache and are not used on the page |

The numbers on the page that are not in these files (the single-model comparisons of section 4, the micro
benchmarks, the CPython conformance run) are quoted from the descriptions of the PRs and from the report in the
description of the top PR.

`cpython_r23.json` has the outcome of every test of the CPython conformance run on the top of the stack (per
module: the plain interpreter's run and the compiled one), with the frame hook on; `cpython_r22.json` is the run of v5.
