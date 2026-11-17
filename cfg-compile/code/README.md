# torch.cfg_compile: the report

Code behind https://bobrenjc93.github.io/explanations/cfg-compile/ . The compiler itself is the stack of draft PRs
pytorch/pytorch#204373 (bottom) to pytorch/pytorch#204763 (top); nothing of it is in this folder.

| file | what it is |
|---|---|
| `make_page.py` | the prose of the page and what each figure is made of; `python make_page.py ../index.html` rebuilds the page |
| `data.py` | reads `../results/*.jsonl` and derives every per-model number, geometric mean and count the page states |
| `charts.py` | the small static-SVG chart kit shared with the other pages of this site, plus the bars against a baseline and the stat tiles this page adds |

## Results (`../results`)

One JSON line per (model, test, compiler), as written by `benchmarks/cfg_compile/torchbench.py` of the stack:
`call_s` is the median time of a call in the steady state, `compile_s` what the compiler reports for itself,
`warmup_s` the cost of the calls before the timed ones, `difference` the worst relative difference from eager's
outputs (losses for training), `graphs` the number of graphs, `paired_*` the two compilers timed in turns inside
the `cfg_compile` process, `peak_rss_mb` the most memory the process held.

| file | what it is |
|---|---|
| `final_r22.jsonl` | the final run (v5 on the page): the top of the stack, nine workers for inference and five for training under the memory guard; inference measured again in all three modes, so that every row has its peak memory, training rows of eager and `torch.compile` taken from the run of v4 |
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

`cpython_r22.json` has the outcome of every test of the CPython conformance run on the top of the stack (per
module: the plain interpreter's run and the compiled one).
