# Does the calculator generalise?

Code behind https://bobrenjc93.github.io/explanations/calculator-generalize/ (follow-up to `../calculator-harder`).

| file | what it is |
|---|---|
| `mathgen3.py` | the seven harder tiers, now with size levels (`SIZE`: original range, one size up, two sizes up) and pool regimes (`REGIME`: train, stretch, stretch2, wide, gap) |
| `mathgen2.py` | worked solutions; `OPT["reduce"]` adds argument reduction for sin and cos beyond the table (`Tracer.reduce`) |
| `mathgen.py` | the calculator levels; `SIN_RANGE` draws direct sin/cos arguments beyond the grid |
| `build_data3.py` | pool builder; regime `wide` = original sizes plus one size up, minus the gap sizes, with sine reduction and solutions to 4,096 tokens |
| `build_stretch.py` | the outside test sets: `stretch`, `stretch2` (two sizes up), `gap` (the sizes the wide pool leaves out) |
| `train3.py`, `model.py` | training (here: a fine-tune with `--init`) |
| `eval_stretch.py` | a checkpoint on one of the outside sets |
| `sin_probe.py` | sin(x) on a regular grid, any range |
| `figs5.py`, `prose4.py`, `make_page4.py`, `figs4.py`, `charts.py` | generate the page |
| `../results/` | results of the fine-tuned run and the model it started from, the two sine probes, and the gap and two-sizes-up sets |

Written and run with an AI assistant (Claude).
