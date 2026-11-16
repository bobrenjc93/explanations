# Scaling laws for a calculator

Code behind https://bobrenjc93.github.io/explanations/calculator-scaling/ (follow-up to `../calculator-harder` and `../calculator-generalize`).

| file | what it is |
|---|---|
| `scale_run.py` | the whole study: two queues of runs side by side on one GPU, every run resumable from its checkpoint |
| `train3.py` | training; `--d --L --heads` set the size, `--steps` the length, `--data_frac` the share of the pool used, `--ckpt_every` makes a run resumable. Writes held-out loss, exact match per level and the loss on its own training rows |
| `eval_stretch.py` | one size up: a checkpoint on the stretch set (here the first 100 problems per tier) |
| `eval_loss.py` | held-out loss for checkpoints trained before the trainer recorded it |
| `mathgen*.py`, `build_data3.py`, `build_stretch.py` | the benchmark, its worked solutions, the pool and the test sets (see `../calculator-harder`) |
| `figs6.py`, `prose5.py`, `make_page5.py`, `figs4.py`, `charts.py` | fits and figures, and the page |
| `../results/` | per-run results and training logs |

Written and run with an AI assistant (Claude).
