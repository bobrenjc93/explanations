# Calculator LLM + thinking-token ablation

Code behind https://bobrenjc93.github.io/explanations/thinking-tokens/

| file | what it is |
|---|---|
| `mathgen.py` | the calculator curriculum: expression trees, evaluator, display rule, printer, independent parser |
| `build_data.py` | generates the 10.5M-problem training pool and the 14k held-out test set (`data.npz`) |
| `check_gen.py` | cross-checks printer, parser, evaluator and Python's `eval` on 280k problems |
| `model.py` | one causal transformer with every thinking mode (um / disc / soft / direct / cont, loop range, memory window, workspace slots, looped middle block) |
| `train.py` | trains one variant; curriculum, variable-K training, scratchpad targets, staged scratchpad removal |
| `sched.py`, `jobs_main.txt` | the sweep |
| `probes.py`, `analyze.py`, `subtypes.py` | post-hoc probes on checkpoints: K sweeps, memory knock-outs, thought interventions, logit lens, confidence halting, sub-skill accuracy |
| `charts.py`, `figs.py`, `pagedata.py`, `prose.py`, `prose_static.py`, `make_page.py` | generate the page from the results |
| `../results/` | per-run results (`*.json`), training logs (`*.jsonl`) and `probes.json` |

```
uv venv .venv && uv pip install --python .venv/bin/python torch numpy
python build_data.py 1500000 && python check_gen.py
python train.py --name cont8_s0 --mode cont --K 8
python sched.py jobs_main.txt runs/main 3
python probes.py runs/main && python make_page.py index.html
```

Written and run with an AI assistant (Claude).
