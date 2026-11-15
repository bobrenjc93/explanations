# A calculator benchmark that takes longer to saturate

Code behind https://bobrenjc93.github.io/explanations/calculator-harder/ (follow-up to `../calculator-99`).

| file | what it is |
|---|---|
| `mathgen.py`, `mathgen2.py` | the seven calculator levels and their worked-solution scratchpads (from the previous write-ups) |
| `mathgen3.py` | the seven harder tiers: big numbers, long expressions, linear and quadratic equations, linear systems, sums, Newton's method. Problem generators, exact answers, and worked solutions built from the same arithmetic procedures |
| `check_gen3.py`, `stepcheck.py` | re-verify every arithmetic step of the new worked solutions and measure how often they end in the true answer (the ceiling) |
| `build_data3.py` | training pool and the 28,000-problem test set for the 14 levels, in three versions (full worked solutions, values only, answer only), bucketed by prompt and solution length |
| `model.py` | the transformer (unchanged architecture; adds a fused attention pass and a decoder with a preallocated cache for outputs of thousands of tokens) |
| `train3.py` | training on the 14 levels; logs loss, per-level held-out accuracy and per-problem outcomes during training |
| `eval_old.py` | a model trained on the old seven levels, zero-shot on the new tiers it can read |
| `build_stretch.py`, `eval_stretch.py` | a test set beyond the training range (every tier's size knob turned up) and a checkpoint's score on it |
| `horizon.py` | accuracy against solution length, and the solution-length horizon |
| `errs3.py` | re-checks the model's own worked solutions and names the first fault |
| `sched.py`, `jobs_hard.txt`, `run_hard.sh` | the run queue |
| `figs4.py`, `prose3.py`, `make_page3.py`, `charts.py` | generate the page from the results |
| `../results/` | per-run results, training logs, the test set (`test_set.json`: problem, true answer, sub-type, reference solution length) and the stretch set (`stretch_set.json`) |

Written and run with an AI assistant (Claude).
