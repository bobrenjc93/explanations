# The calculator at 99%

Code behind https://bobrenjc93.github.io/explanations/calculator-99/ (follow-up to `../thinking-tokens`).

| file | what it is |
|---|---|
| `mathgen.py` | the calculator curriculum and the test-set split (shared with the first write-up) |
| `mathgen2.py` | worked-solution scratchpads: guard digits, partial products, long division, Euclid, Newton, mantissa tables. Every change is a switch (`OPT`, `RUNGS`) so the path can be replayed |
| `check_gen2.py` | re-verifies every arithmetic step in the worked solutions and their agreement with the true answers |
| `build_data2.py` | generates the training pool for any scratchpad version, bucketed by solution length (static shapes, little padding) |
| `model.py` | the transformer: positional encodings (RoPE / none / learned), LayerNorm / RMSNorm, looped middle block, quantised weights |
| `train2.py` | bucketed training on worked solutions, optional thinking tokens in front, loss and held-out answer accuracy logged during training; `rescore.py` re-evaluates a checkpoint; `bench2.py` measures throughput |
| `errs.py` | generates every test solution and reports the first step that differs from the reference, by kind |
| `quant.py` | post-training weight quantisation sweep |
| `token_stats.py` | true mean target tokens per training row of each pool (`results/ladder/target_tokens.json`); the published logs divide the loss by an estimate from the head of each bucket, and the page rescales them with this |
| `effort_eval.py` | the thinking-effort dial: accuracy and compute at each forced effort, and the confidence cascade |
| `sched.py`, `jobs_*.txt` | run queues (`jobs_ladder.txt`, `jobs_ladder2.txt`: the timeline, one idea per run, plus the thinking-token, thinking-block and effort runs) |
| `figs2.py`, `figs3.py`, `prose2.py`, `make_page2.py`, `charts.py`, `pagedata.py` | generate the page from the results |
| `../results/` | per-run results and training logs |

Written and run with an AI assistant (Claude).
