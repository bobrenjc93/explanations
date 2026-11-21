"""stage.py [DEST]: the page's folder for the site (default report/site/two-compilers-rust-engine): index.html,
code/ (the rules, the harness, the checks, each side's configurations, journal, summary and final source changes,
this generator) and results/ (the logs, the token table, the outputs of the checks that the page uses).
Run make.py first. Copies a fixed list of files, writes over what is there, removes nothing, and ends with a
search of what it staged for things that must not be published."""
import json, os, re, shutil, subprocess, sys

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
argv, sys.argv = sys.argv, sys.argv[:1]
import make as M  # the rows and what the checks found for each

H = M.H
DEST = argv[1] if len(argv) > 1 else f"{HERE}/site/two-compilers-rust-engine"
copied = []


def put(src, rel, edit=None):
    dst = os.path.join(DEST, rel)
    os.makedirs(os.path.dirname(dst), exist_ok=True)
    if edit:
        open(dst, "w").write(edit(open(src).read()))
        shutil.copymode(src, dst)
    else:
        shutil.copy2(src, dst)
    copied.append(rel)


def write(rel, text):
    dst = os.path.join(DEST, rel)
    os.makedirs(os.path.dirname(dst), exist_ok=True)
    open(dst, "w").write(text)
    copied.append(rel)


README = """# One model, two compilers, run again: harness, checks and page generator

The second run of the experiment of `../../two-compilers-one-model` and `../../two-compilers-six-hours`. Everything is
as it ran, on one machine with one H100; paths are that machine's.

## The experiment

- `RULES.md`: the rules both agents were given (the same file), with the checks that were announced; `README_tc.md` and
  `README_vm.md` for each side's tree and tools.
- `bench.py`: one process of the measurement. TorchBench yolov3 as the performance dashboard sets it up (eval, batch 8
  of 3x384x512, bfloat16, `no_grad`), through whatever `build(model, example_inputs)` in a side's `config.py` returns;
  warm-up, 400 timed calls on four inputs, outputs checked against eager, three probes.
- `measure.py` (`measure`): a recorded measurement: a snapshot of the side's state as a commit, three processes of
  `bench.py`, the median, a row in the side's `log.jsonl`. `check.py` (`check`): the announced probes and the
  predictions on a side's current state. `run`: a command in one side's environment, with or without the GPU lock.
- `tools/overlay.py`: a standalone copy of the Python of `torch` for a side. `tools/mktree.py`: the standalone tree of
  a recorded snapshot. `tools/tokens.py`: what each agent had spent (tokens, minutes) at each recorded measurement,
  from the agents' transcripts.
- `tc/` and `vm/`: each side's `start_config.py` and final `config.py`, its `journal.md`, its `summary.md` (the
  agent's final report) and `final.patch` (its changes to the compiler's source, last snapshot against the first,
  without the rebuilt engine library).

## The checks made afterwards (`acc/`)

- `infer.py`: predictions with the original pretrained yolov3-spp weights on the 128 COCO images that ship with the
  benchmark. `probe.py`, `probe2.py`: what a compiled callable still follows once it is warm (announced to the agents).
  `probe4.py`: three probes that were not announced (a weight written through `.data`; every tensor of a result kept
  across two calls with other inputs, as tensors and through DLPack). `probe3.py` is their first version, kept for the
  record and not used by the page: two of its probes were too weak (they kept only the first tensor of a result, and
  the last call repeated the input of the kept one). `after.py`: speed after events. `cross_cache.sh`: a state compiled
  for the random-weight model first and the pretrained one second, with one compile cache. `train.py`: the loss curve
  of training through eager, `torch.compile` or `torch.vm`.
- `validate.py` runs the per-row checks over every recorded row; `run_all.sh` and the `lane*.sh` scripts are the order
  everything was run in after the deadline (`lane3.sh`: `probe3.py`; `lane4*`: each side's tests on every row's
  source, `../results/ckpt_tests_NOTE.txt` says which tests ran on which row; `lane5*`: `probe4.py`).

## The page

`make.py` reads the logs, the snapshots and the outputs of the checks and writes the page; `text.py` is its prose,
`appendix.py` the appendix (every step explained), `charts.py` the small SVG chart kit, `verify.py` a listing of what
the generator read for every row, `stage.py` what made this folder, `shot*.js` the screenshots used to look at the
result.

## Results (`../results`)

- `tc/log.jsonl`, `vm/log.jsonl`: every measurement. `tokens.json`: time and tokens at every recorded measurement.
- `out/`: the outputs of the checks, one JSON per row and check. `out_probe3_first_version/`: the outputs of the first
  version of the unannounced probes, which the page does not use. `tests.json`: what each side's tests gave on the
  source of every row, and which rows are struck for it; `ckpt_tests_NOTE.txt`: which tests those were.
- `by_time.png`, `by_cost.png`: the two charts.
"""

# The rules, the harness and the checks.
put(f"{H}/RULES.md", "code/RULES.md")
put(f"{H}/tc/README.md", "code/README_tc.md")
put(f"{H}/vm/README.md", "code/README_vm.md")
for name in ("bench.py", "measure.py", "measure", "check.py", "check", "run"):
    put(f"{H}/{name}", f"code/{name}")
for name in ("after.py", "common.py", "cross_cache.sh", "gpu", "gpu2", "infer.py", "probe.py", "probe2.py", "probe3.py", "probe4.py", "train.py", "validate.py", "run_all.sh",
             "lane3.sh", "lane4.sh", "lane4b.sh", "lane4c.sh", "lane5.sh", "lane5b.sh"):
    if os.path.exists(f"{H}/acc/{name}"):
        put(f"{H}/acc/{name}", f"code/acc/{name}")
put(f"{H}/tools/mktree.py", "code/tools/mktree.py")
put(f"{H}/tools/overlay.py", "code/tools/overlay.py")
# The transcripts' directory names the session; as on the earlier pages it is replaced.
put(f"{H}/tools/tokens.py", "code/tools/tokens.py",
    edit=lambda text: re.sub(r'^SUB = ".*"$', 'SUB = "/path/to/the/session/subagents"  # where the agent harness keeps the two agents\' transcripts', text, flags=re.M))
write("code/README.md", README)

# Each side: configurations, journal, summary, final source changes.
for side in ("tc", "vm"):
    for name in ("start_config.py", "config.py", "journal.md", "summary.md"):
        put(f"{H}/{side}/{name}", f"code/{side}/{name}")
    first = M.git(side, "rev-list", "--max-parents=0", "HEAD").split()[0]
    last = M.points[side][-1]["sha"]
    binary = [line.split("\t")[2] for line in M.git(side, "diff", "--numstat", first, last).splitlines() if line.startswith("-\t-\t")]
    patch = M.git(side, "diff", first, last, "--", ".", ":!hill_config.py", *[f":!{path}" for path in binary])
    write(f"code/{side}/final.patch", patch)
    put(f"{H}/{side}/log.jsonl", f"results/{side}/log.jsonl")

# The generator.
for name in ("make.py", "text.py", "appendix.py", "charts.py", "stage.py", "verify.py", "shot.js", "shot_fig.js", "shot_id.js", "shot_full.js", "shot_hover.js"):
    put(f"{HERE}/{name}", f"code/{name}")

# Results: the token table, the outputs of the checks the page reads, the tests per row.
put(f"{HERE}/tokens.json", "results/tokens.json")
tests = {}
for side in ("tc", "vm"):
    tests[side] = []
    for p in M.points[side]:
        for kind in ("infer", "probe", "probe2", "probe4", "after"):
            path = f"{M.OUT}/{kind}_{side}_{p['i']:02d}.json"
            if os.path.exists(path):
                put(path, f"results/out/{os.path.basename(path)}")
        # The first version of the unannounced probes, which the page does not use: kept, under a name that says so.
        path = f"{M.OUT}/probe3_{side}_{p['i']:02d}.json"
        if os.path.exists(path):
            put(path, f"results/out_probe3_first_version/{os.path.basename(path)}")
        path = f"{M.OUT}/xcache_{side}_{p['i']:02d}_pretrained.json"
        if os.path.exists(path):
            put(path, f"results/out/{os.path.basename(path)}")
        tests[side].append({"row": p["i"], "images_per_s": p["qps"], "sha": p["sha"], "counted_by_the_harness": p["eligible"], "tests_on_this_row": p["own_tests"],
                            "runs_on_rows_with_the_same_source": [{"row": i, "tests_run": n} for i, n in p["runs"]], "not_passing_on_this_source": p["bad"],
                            "fails_what_the_starting_source_passes": p["broken"], "struck": p["struck"], "not_followed": p["failed"], "passes_every_check": p["passes"]})
for prefix, _ in M.TRAIN:
    for key, _ in M.RUNS:
        path = f"{M.OUT}/train_{prefix}{key}.json"
        if os.path.exists(path):
            put(path, f"results/out/{os.path.basename(path)}")
write("results/tests.json", json.dumps(tests, indent=1))
if os.path.exists(f"{M.OUT}/ckpt_tests_NOTE.txt"):
    put(f"{M.OUT}/ckpt_tests_NOTE.txt", "results/ckpt_tests_NOTE.txt")
if os.path.exists(f"{HERE}/index.html"):
    put(f"{HERE}/index.html", "index.html")

# Nothing that must not be published: no token files, and no secret in what was staged.
found = []
for root, _, names in os.walk(DEST):
    for name in names:
        path = os.path.join(root, name)
        if name in (".hf" + "_token", "others.log") or "token" in name and not name.startswith("tokens."):
            found.append(f"file name: {path}")
        if name.endswith(".png"):
            continue
        text = open(path, errors="replace").read()
        # The prefixes are put together here so that this file does not match a search for them itself.
        for pattern in ("hf" + r"_[A-Za-z0-9]{20,}", "gh" + r"o_[A-Za-z0-9]+", "gh" + r"p_[A-Za-z0-9]+"):
            for hit in re.findall(pattern, text):
                found.append(f"{path}: {hit[:8]}...")
print(f"staged {len(copied)} files in {DEST}")
print("search for access tokens (Hugging Face and GitHub prefixes) and for token files:", "nothing found" if not found else found)
