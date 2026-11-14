# The calculator at 99%

Code behind https://bobrenjc93.github.io/explanations/calculator-99/ (follow-up to `../thinking-tokens`).

| file | what it is |
|---|---|
| `mathgen.py` | the calculator curriculum and the test-set split (shared with the first write-up) |
| `mathgen2.py` | worked-solution scratchpads: guard digits, partial products, long division, Euclid, Newton, mantissa tables |
| `check_gen2.py` | re-verifies every arithmetic step in the worked solutions and their agreement with the true answers |
| `build_data2.py` | generates the training pool, bucketed by solution length (static shapes, little padding) |
| `model.py` | the transformer: positional encodings (RoPE / none / learned), LayerNorm / RMSNorm, looped middle block, quantised weights |
| `train2.py` | bucketed training on worked solutions; `rescore.py` re-evaluates a checkpoint |
| `errs.py` | generates every test solution and reports the first step that differs from the reference, by kind |
| `quant.py` | post-training weight quantisation sweep |
| `sched.py`, `jobs_*.txt` | run queues |
| `figs2.py`, `prose2.py`, `make_page2.py`, `charts.py`, `pagedata.py` | generate the page from the results |
| `../results/` | per-run results and training logs |

Written and run with an AI assistant (Claude).
