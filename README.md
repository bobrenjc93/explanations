# Explanations

A place for explanations: write-ups of PyTorch PRs, design discussions, and
anything else worth explaining in more depth than a chat reply allows. Each
explanation is a self-contained page in its own folder.

Published at https://bobrenjc93.github.io/explanations/

## Adding an explanation

Put the page in its own folder as `<folder>/index.html` (with a `<h1>`, an optional `<p class="sub">` summary, and
the favicon `<link>` tags using `../`), commit it, then run `python3 tools/build_index.py` and commit the regenerated
`index.html` and `README.md`. Created and updated dates come from git history; site-wide chore commits listed in
`.index-ignore-revs` don't count as updates. Tags are optional, in the `TAGS` table in `tools/build_index.py`.

## Index

- Nov 16, 2026: [Scaling laws for a calculator: parameters, data, and where it stops](https://bobrenjc93.github.io/explanations/calculator-scaling/)
- Nov 16, 2026: [torch.cfg_compile: a control-flow-graph compiler for PyTorch, measured against torch.compile](https://bobrenjc93.github.io/explanations/cfg-compile/)
- Nov 16, 2026: [Does the calculator generalise? Gaps, a wider pool, and a procedure for sine](https://bobrenjc93.github.io/explanations/calculator-generalize/)
- Nov 15, 2026: [A calculator benchmark that takes longer to saturate](https://bobrenjc93.github.io/explanations/calculator-harder/)
- Nov 14, 2026: [A calculator LLM at 99%: the scratchpad mattered more than the weights](https://bobrenjc93.github.io/explanations/calculator-99/)
- Nov 14, 2026: [Should a transformer say "um"? A calculator LLM and a thinking-token ablation](https://bobrenjc93.github.io/explanations/thinking-tokens/)
- Nov 2, 2026: [Ten days of H200 kernel optimization for the Muse Spark Open Kernel Challenge](https://bobrenjc93.github.io/explanations/okc-kernels/)
- Oct 1, 2026: [Should matmul's backward honor grad_dtype?](https://bobrenjc93.github.io/explanations/matmul-grad-dtype/)
- Sep 23, 2026: [pytorch#198214: what "the continuation reaches no tensor" means](https://bobrenjc93.github.io/explanations/pr198214/)
