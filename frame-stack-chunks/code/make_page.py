"""Builds the page on the chunks of CPython's frame stack: prose here, figures from charts.py, numbers from ../results.

    python make_page.py ../index.html
"""
import json
import os
import re
import sys

import charts as C

RESULTS = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "results")
PR = "https://github.com/pytorch/pytorch/pull/"
CPY = "https://github.com/python/cpython/"
TITLE = "A quarter of a training step, lost to one Python frame: the chunks of CPython's frame stack under a compiled graph"
SUB = ("A training step of tf_efficientnet_b0 took 30.5 ms under torch.rewrite in the perf dashboard's runner and 23.3 ms in a probe of the "
       "same runner. The difference was where a 16 KB chunk of the interpreter's frame stack ends: how that was found, what it is, "
       "that torch.compile has it too, which CPython versions have it, and what was done. Nov 18, 2026.")


def load(name):
    with open(os.path.join(RESULTS, name)) as f:
        return json.load(f)


DEPTH, CLIFF, CHAINS, PAIRS = load("depth.json"), load("cliff.json"), load("chains.json"), load("pairs.json")
REWRITE_CASE, COMPILE_CASE = CHAINS["tf_efficientnet_b0 under torch.rewrite"], CHAINS["ElectraForCausalLM under torch.compile"]
SPARED_CASE = CHAINS["tf_efficientnet_b0 under torch.compile"]
CHUNK, START = 16 * 1024, 32  # a chunk, and where the frames of a thread's first chunk start in it


def pr(n, text=None):
    return f'<a href="{PR}{n}">{text or "#" + str(n)}</a>'


def chunks(name, configs, least=100):
    """The models whose timed loop asked for at least `least` chunks a repetition under one of `configs`: [(model, n)]."""
    found = {}
    with open(os.path.join(RESULTS, name)) as f:
        for line in f:
            if "keep 1" in line.split("|")[0]:
                continue
            config = "rewrite" if "--rewrite" in line else "freezing" if "--freezing" in line else "inductor"
            steady = re.search(r"(-?\d+) after its first repetition", line)
            per = int(steady.group(1)) / 29 if steady else int(line.split()[0]) / 30
            model = re.search(r"--only (\S+)", line).group(1)
            if config in configs and per >= least:
                found[model] = max(found.get(model, 0), per)
    return sorted(found.items(), key=lambda kv: -kv[1])


def lines(name):
    with open(os.path.join(RESULTS, name)) as f:
        return sum(1 for _ in f)


# 150 first runs and the second pairs of four models; 86 models in three configurations.
assert lines("chunks_training.log") == 158 and lines("chunks_inference.log") == 258


# --------------------------------------------------------------------------- figures


def fig_depth():
    before, fixed, compiled = DEPTH["rewrite_before"], DEPTH["rewrite_with_kept_chunks"], DEPTH["compile"]
    xs = before["depth"]
    series = [{"name": "torch.rewrite", "short": "torch.rewrite", "c": "--s1", "y": before["ms"]},
              {"name": "torch.compile", "short": "torch.compile", "c": "--s2", "y": compiled["ms"]},
              {"name": "torch.rewrite, freed chunks kept", "short": "chunks kept", "c": "--s3", "y": fixed["ms"]}]
    body = C.line_chart(series, xs, xlabels=[str(x) for x in xs], h=300, ymin=20, ymax=34, yticks=(20, 24, 28, 32), xtick_idx=range(0, len(xs), 20),
                        xlab="small Python frames between the runner and the step", right=150, yfmt=lambda v: f"{v:.0f}", vfmt=lambda v: f"{v:.1f} ms",
                        tv_header="frames added")
    slow = [d for d, ms in zip(xs, before["ms"]) if ms > 27]
    return C.figure("The same training step, called from 140 depths of the stack",
                    "Milliseconds for one training step of tf_efficientnet_b0 (batch 128, AMP, one H100), called through a chain of 0 to 139 "
                    "small Python frames of 120 bytes each; median of three calls at each depth. Under <code>torch.rewrite</code> the step is "
                    f"slow at depth {slow[0]} and at depths {slow[1]} to {slow[-1]}, and nowhere else; "
                    "<code>torch.compile</code> has no such window for this model; with the freed chunks kept (measured later the same day) "
                    "neither has <code>torch.rewrite</code>. Single depths that are 2 to 3 ms slow (14 under <code>torch.compile</code>, 43 under "
                    "<code>torch.rewrite</code>) were not looked into.", body)


def fig_depth_coarse():
    coarse = DEPTH["rewrite_before_coarse"]
    xs = coarse["depth"]
    body = C.line_chart([{"name": "torch.rewrite", "c": "--s1", "y": coarse["ms"]}], xs, xlabels=[str(x) for x in xs], h=220, ymin=20, ymax=34,
                        yticks=(20, 24, 28, 32), xtick_idx=range(0, len(xs), 10), xlab="small Python frames between the runner and the step", right=110,
                        yfmt=lambda v: f"{v:.0f}", vfmt=lambda v: f"{v:.1f} ms", show_legend=False, tv_header="frames added")
    slow = [d for d, ms in zip(xs, coarse["ms"]) if ms > 27]
    return C.figure("It comes back every 16 KB",
                    "The first scan, every fourth depth up to 396 (milliseconds a step, under <code>torch.rewrite</code>, before the fix). The "
                    f"slow depths are {', '.join(str(d) for d in slow)}: {slow[1] - slow[0]} and {slow[2] - slow[1]} frames of 120 bytes apart, "
                    "about 16 KB.", body)


def fig_cliff():
    names = {"3.11": "--s1", "3.12": "--s2", "3.13": "--s3", "3.14": "--s4"}
    xs = list(range(len(CLIFF[0]["ns"])))
    series = [{"name": f"CPython {d['python']}", "short": d["python"], "c": names[d["python"][:4]], "y": d["ns"]} for d in CLIFF]
    body = C.line_chart(series, xs, xlabels=[str(x) for x in xs], h=300, ymin=20, ymax=20000, yticks=(30, 100, 1000, 10000), ylog=True,
                        xtick_idx=range(0, len(xs), 20), xlab="depth of the caller (frames of 128 bytes)", right=110, yfmt=lambda v: f"{v:g}",
                        vfmt=lambda v: f"{v:,.0f} ns", tv_header="depth")
    return C.figure("Plain Python: a call that takes 41 ns takes 7.8 µs at one depth",
                    "Nanoseconds for one call of a small Python function, made 2000 times from each of 200 depths of the stack, on four CPython "
                    "releases (<code>tools/chunk_cliff.py</code>; one pinned core, no PyTorch in the process; logarithmic scale). At depth 130 the "
                    "callee's frame is the first that does not fit in the chunk: under 3.11 and 3.12 each of the 2000 calls is a page fault.", body)


def chunk_diagram():
    """Three stacks of frames drawn into their 16 KB chunks: what is ordinary, what is large, and what is left."""
    def split(frames, big):
        """[(label, bytes, kind)] from frames outermost first: the frames before the first large one, each large one, the ones between."""
        out, run_bytes = [], 0
        for name, _, size in frames:
            if name in big:
                if run_bytes:
                    out.append(("the runner's frames and wrappers", run_bytes, "plain"))
                    run_bytes = 0
                out.append((big[name], size, "call" if "generated" in big[name] else "resume"))
            else:
                run_bytes += size
        if run_bytes:
            out.append(("the runner's frames and wrappers", run_bytes, "plain"))
        return out

    a = split(REWRITE_CASE["frames_innermost_first"][::-1], {"Runner.call": "the graph's generated code"})
    b = split(SPARED_CASE["frames_outermost_first"], {"torch_dynamo_resume_in_forward_and_backward_pass_at_420": "Dynamo's resume function", "call": "the graph's generated code"})
    c = split(COMPILE_CASE["frames_innermost_first"][::-1], {"Runner.call": "the graph's generated code",
                                                              "HuggingfaceRunner.torch_dynamo_resume_in_forward_and_backward_pass_at_565": "Dynamo's resume function"})
    # Dynamo runs its code in a frame of its own next to the function's; those are in the chunk but not on the chain of frames.
    hidden = CHUNK - START - COMPILE_CASE["bytes_left_in_the_chunk"] - sum(size for _, size, _ in c)
    c.insert(-1, ("frames Dynamo keeps beside its own (not on the chain of frames)", hidden, "plain"))
    b_first, b_call = b[:-1], b[-1]
    rows = [("tf_efficientnet_b0", "torch.rewrite", a, f"{REWRITE_CASE['bytes_left_in_the_chunk']} bytes left: no launch fits"),
            ("tf_efficientnet_b0", "torch.compile", b_first, "11,176 bytes do not fit here"),
            ("", "its next chunk", [b_call], "5,184 bytes left: launches fit"),
            ("ElectraForCausalLM", "torch.compile", c, f"{COMPILE_CASE['bytes_left_in_the_chunk']} bytes left: no launch fits")]
    w, L, pw, bar, gap = 760, 150, 400, 22, 30
    h = 26 + len(rows) * (bar + gap)
    X = lambda v: L + pw * v / CHUNK
    colors = {"plain": "--base", "resume": "--s2", "call": "--s1"}
    o, table = [], []
    for t in (0, 4096, 8192, 12288, 16384):
        o.append(f'<line class="grid" x1="{X(t):.1f}" x2="{X(t):.1f}" y1="14" y2="{h - 8}"/>')
        o.append(f'<text x="{X(t):.1f}" y="10" text-anchor="middle">{t // 1024} KB</text>')
    for i, (model, how, parts, note) in enumerate(rows):
        y = 26 + i * (bar + gap)
        o.append(f'<text class="lab" x="{L - 10}" y="{y + 9}" text-anchor="end">{C.esc(model)}</text>')
        o.append(f'<text x="{L - 10}" y="{y + 22}" text-anchor="end">{C.esc(how)}</text>')
        o.append(f'<rect x="{X(0):.1f}" y="{y}" width="{pw}" height="{bar}" rx="3" fill="none" stroke="var(--axis)"/>')
        at = START if how != "its next chunk" else 24
        for label, size, kind in parts:
            x0, x1 = X(at), X(at + size)
            tip = {"t": label, "r": [{"c": colors[kind], "n": "bytes", "v": f"{size:,}"}]}
            o.append(f'<rect class="bar" x="{x0 + 1:.1f}" y="{y + 1}" width="{max(x1 - x0 - 2, 1):.1f}" height="{bar - 2}" rx="3" fill="var({colors[kind]})" '
                     f'tabindex="0" data-tip="{C.esc(json.dumps(tip))}"/>')
            if x1 - x0 > 46:
                o.append(f'<text x="{(x0 + x1) / 2:.1f}" y="{y + bar + 13}" text-anchor="middle">{size:,}</text>')
            table.append([f"{model or 'tf_efficientnet_b0'}, {how}", label, f"{size:,}"])
            at += size
        table.append([f"{model or 'tf_efficientnet_b0'}, {how}", "left in the chunk", f"{CHUNK - at:,}"])
        o.append(f'<text class="lab" x="{X(CHUNK) + 8:.1f}" y="{y + 15}">{C.esc(note)}</text>')
    svg = f'<div class="scroll"><svg viewBox="0 0 {w} {h}" role="img" style="min-width:640px">' + "".join(o) + "</svg></div>"
    legend = C.legend([("the runner's frames and wrappers", "--base", "box"), ("Dynamo's resume function", "--s2", "box"), ("the graph's generated code", "--s1", "box")])
    return C.figure("Where the graph's frame ends in its chunk",
                    "The frames on the stack when the forward graph launches its first kernel, in the dashboard runner as it is, drawn into the "
                    "16 KB chunks they occupy (bytes under each part; the outline is the chunk). A kernel launch needs 288 bytes for the first "
                    "frame it calls. For the first and the last row the sizes and what is left were read from the interpreter's own frames at "
                    "the moment a launch asked for a chunk; the second row is from <code>sys._getframe</code> in a probe.",
                    legend + svg + C.table_view(["stack", "part", "bytes"], table))


def fig_who():
    train_rewrite, train_compile = chunks("chunks_training.log", ("rewrite",)), chunks("chunks_training.log", ("inductor",))
    items = [{"name": f"{m}, torch.rewrite", "v": n, "c": "--s1", "tip": [("compiler", "torch.rewrite")]} for m, n in train_rewrite]
    items += [{"name": f"{m}, torch.compile", "v": n, "c": "--s2", "tip": [("compiler", "torch.compile")]} for m, n in train_compile]
    items.sort(key=lambda it: -it["v"])
    body = C.legend([("torch.rewrite", "--s1", "box"), ("torch.compile", "--s2", "box")]) + C.value_bars(
        items, 560, (0, 100, 200, 300, 400, 500), label_w=250, vfmt=lambda v: f"{v:.0f}", tv_header=("model and compiler", "chunks a step"))
    return C.figure("Chunks asked for in one training step, in the dashboard runner",
                    "Of 150 training runs (75 models under two compilers) and 258 inference runs "
                    "(86 models under <code>torch.compile</code>, <code>torch.compile</code> with freezing and <code>torch.rewrite</code>), these six "
                    "ask for a hundred or more 16 KB chunks in every repetition of the timed loop, and all six are training runs. The next are "
                    "cm3leon_generate under <code>torch.compile</code> for inference (65) and the two T5 models under <code>torch.rewrite</code> for "
                    "training (18); every other run asks for 8 or fewer.", body)


# --------------------------------------------------------------------------- the page


def tiles():
    m = PAIRS["models"]["tf_efficientnet_b0"]
    before, after = m["before the fix"]["torch.rewrite"], m["with the fix"]["torch.rewrite"]
    slow = next(d for d in CLIFF if d["python"].startswith("3.12"))
    return C.tiles([
        ("A training step of tf_efficientnet_b0 under torch.rewrite", f"{before['ms']:.1f} → {after['ms']:.1f} ms",
         f"{before['speedup']:.2f}x to {after['speedup']:.2f}x over eager; torch.compile: {m['with the fix']['torch.compile']['speedup']:.2f}x"),
        ("Left in the chunk after the graph's frame", f"{REWRITE_CASE['bytes_left_in_the_chunk']} bytes", "of 16 KB; a kernel launch needs 288 for its first frame"),
        ("Chunks mapped and unmapped in one step", f"{REWRITE_CASE['asked_in_each_repetition'][1]}", "368 kernel launches, 81 checks of a tensor's metadata, 3 others"),
        ("One Python call at the wrong depth, CPython 3.12", f"{slow['ns_a_call_there'] / 1000:.1f} µs", f"{slow['ns_a_call_usually']:.0f} ns at any other; 3.13.15 and 3.14.7: no such depth"),
    ])


def version_table():
    rows = []
    for d in CLIFF:
        there = d["ns_a_call_there"]
        rows.append([d["python"], ", ".join(str(x) for x in d["depths_with_a_fault_per_call"]) or "none", f"{d['ns_a_call_usually']:.0f} ns",
                     f"{there:,.0f} ns" if there else "", f"{max(d['faults']):,}"])
    return C.table(["CPython", "depths (of 200) where every call is a page fault", "a call usually", "a call there", "most page faults in 2000 calls"], rows)


def model_table():
    rows = []
    for name in ("tf_efficientnet_b0", "repvgg_a2"):
        m = PAIRS["models"][name]
        b, a = m["before the fix"], m["with the fix"]
        rows.append([name, "torch.rewrite", f"{b['torch.rewrite']['ms']:.1f} ms ({b['torch.rewrite']['speedup']:.2f}x)", f"{a['torch.rewrite']['ms']:.1f} ms ({a['torch.rewrite']['speedup']:.2f}x)",
                     f"{a['torch.compile']['ms']:.1f} ms ({a['torch.compile']['speedup']:.2f}x)"])
    for name in ("ElectraForCausalLM", "BertForMaskedLM", "RobertaForCausalLM", "nfnet_l0"):
        m = PAIRS["models"][name]
        a, k = m["with the fix"], m["second pair, chunks kept for torch.compile too"]
        rows.append([name, "torch.compile", f"{a['torch.compile']['ms']:.1f} ms ({a['torch.compile']['speedup']:.2f}x)", f"{k['torch.compile']['ms']:.1f} ms ({k['torch.compile']['speedup']:.2f}x)",
                     f"{k['torch.rewrite']['ms']:.1f} ms ({k['torch.rewrite']['speedup']:.2f}x)"])
    return C.table(["model", "who maps the chunks", "a step while it does (over eager)", "a step with the chunks kept", "the other compiler, same pair"], rows, num_from=2)


def cost_table():
    per = dict(chunks("chunks_training.log", ("rewrite", "inductor")))
    rows = []
    m = PAIRS["models"]
    rows.append(["tf_efficientnet_b0", "torch.rewrite", f"{per['tf_efficientnet_b0']:.0f}",
                 f"{m['tf_efficientnet_b0']['before the fix']['torch.rewrite']['ms'] - m['tf_efficientnet_b0']['with the fix']['torch.rewrite']['ms']:.1f} ms"])
    rows.append(["repvgg_a2", "torch.rewrite", f"{per['repvgg_a2']:.0f}",
                 f"{m['repvgg_a2']['before the fix']['torch.rewrite']['ms'] - m['repvgg_a2']['with the fix']['torch.rewrite']['ms']:.1f} ms"])
    for name in ("ElectraForCausalLM", "BertForMaskedLM", "RobertaForCausalLM", "nfnet_l0"):
        saved = m[name]["with the fix"]["torch.compile"]["ms"] - m[name]["second pair, chunks kept for torch.compile too"]["torch.compile"]["ms"]
        rows.append([name, "torch.compile", f"{per[name]:.0f}", f"{saved:.1f} ms"])
    for row in rows:
        ms = float(row[3].split()[0])
        if ms <= 0:
            row[3] = "not shorter"
        row.append(f"{ms / float(row[2]) * 1000:.0f} µs" if ms > 0 else "none")
    return C.table(["model", "compiler", "chunks a step", "a step is shorter by", "per chunk"], rows, num_from=2)


def suite_table():
    s = PAIRS["suites"]
    rows = []
    for title, key in (("HuggingFace", "huggingface"), ("timm", "timm_models"), ("TorchBench", "torchbench")):
        a, f, t = s[f"inference {key} vs inductor"], s[f"inference {key} vs freezing"], s[f"training {key} vs inductor"]
        rows.append([title, f"{a['ratio']:.3f} ({a['models']})", f"{f['ratio']:.3f}", f"{t['ratio']:.3f} ({t['models']})", f"{s[f'training {key} before']['ratio']:.3f}"])
    return C.table(["suite", "inference, over torch.compile (models)", "inference, over torch.compile with freezing", "training, over torch.compile (models)",
                    "training, in the pairs before the fix"], rows)


def body():
    m = PAIRS["models"]
    asked = REWRITE_CASE["asked_per_30_repetitions"]
    electra = COMPILE_CASE["asked_per_30_repetitions"]
    rewrite_bytes = sum(size for _, _, size in REWRITE_CASE["frames_innermost_first"])
    o = []
    A = o.append
    A(f"<h1>{TITLE}</h1>")
    A(f'<p class="sub">{SUB}</p>')
    A(f"""<div class="tldr"><p><strong>In short.</strong> On the perf dashboard's training suites, <code>torch.rewrite</code> (a prototype
compiler, <a href="../cfg-compile/">written up here</a>) was level with <code>torch.compile</code> in the mean, and on two timm models
it took a quarter longer for a reason that could not be named: the same graphs, the same kernels on the GPU, as many of them. The reason is in neither
compiler. CPython 3.11 and 3.12 keep the frames of a thread in chunks of 16 KB, and give a chunk back to the system the moment the frame
at its start returns. The code Inductor generates for a graph is one function with one large frame, which launches the graph's kernels
one after the other. Where the dashboard's runner calls that function for tf_efficientnet_b0 under <code>torch.rewrite</code>, its frame
ends {REWRITE_CASE['bytes_left_in_the_chunk']} bytes before the end of a chunk; a kernel launch needs 288 bytes for the first frame it
calls, so each launch maps a new chunk and unmaps it again, {REWRITE_CASE['asked_in_each_repetition'][1]} times in a step. One more Python
frame anywhere above, and the generated function's frame no longer fits either and starts a chunk of its own, with room to spare.</p>
<p>It is an accident of the depth of the caller, to the byte. <code>torch.compile</code> has it on four other training models in the same
runner. CPython fixed it in March 2026 for 3.13 and 3.14 (<a href="{CPY}issues/142183">python/cpython#142183</a>), not for 3.11 and 3.12,
and the perf dashboard's jobs run 3.11. The prototype's runtime now keeps the chunks that the interpreter frees ({pr(205502)}), and the
suites were measured again with a count of the chunks every run asks for.</p></div>""")
    A(tiles())

    A("<h2>1. The symptom</h2>")
    A(f"""<p>The perf dashboard measures <code>torch.compile</code> with <code>benchmarks/dynamo</code>: a runner that loads a model, warms the
compiled function up, and then times thirty compiled calls against thirty eager ones, in turns. {pr(205357)} lets that runner measure
<code>torch.rewrite</code> the same way. For training (<code>--training --amp</code>) the two compilers came out level in the mean
once the update of the optimizer was compiled too: 1.00 of the time of <code>torch.compile</code> on HuggingFace, 1.01 on timm, 0.97 on
TorchBench. On timm, two models stood out:</p>""")
    b = m["tf_efficientnet_b0"]["before the fix"]
    r = m["repvgg_a2"]["before the fix"]
    A(C.table(["model (batch 128)", "torch.compile", "torch.rewrite", "time of torch.rewrite over torch.compile"],
              [["tf_efficientnet_b0", f"{b['torch.compile']['ms']:.1f} ms a step ({b['torch.compile']['speedup']:.2f}x eager)", f"{b['torch.rewrite']['ms']:.1f} ms ({b['torch.rewrite']['speedup']:.2f}x)",
                f"{b['torch.compile']['speedup'] / b['torch.rewrite']['speedup']:.2f}"],
               ["repvgg_a2", f"{r['torch.compile']['ms']:.1f} ms ({r['torch.compile']['speedup']:.2f}x)", f"{r['torch.rewrite']['ms']:.1f} ms ({r['torch.rewrite']['speedup']:.2f}x)",
                f"{r['torch.compile']['speedup'] / r['torch.rewrite']['speedup']:.2f}"]]))
    A("""<p>Both compilers hand the same forward and loss to Inductor as one graph, and AOTAutograd derives the same backward. A profile showed
the same kernels on the GPU under both, as many launches a step, and 6.5 ms more of the CPU's time inside the call of the forward graph under
<code>torch.rewrite</code>. The report of that day said so, and said "why was not found". And a probe that replaced the runner's timed loop
with a copy of it, in the same process, on the same warmed-up function, measured 23.3 ms where the runner measured 30.5 ms.</p>""")

    A("<h2>2. The path</h2>")
    A("""<p>Each step is given with what it was meant to rule out and what it showed. Times are of one training step of tf_efficientnet_b0
under <code>torch.rewrite</code> unless said otherwise.</p>""")
    A("<h3>Not the GPU, not the run</h3>")
    A("""<p>The machine is shared, and an earlier set of numbers had been spoiled by other work on the GPU. So the pair was run three more
times with nothing else on it: 24.5 to 24.6 ms under <code>torch.compile</code>, 29.9 to 30.5 ms under <code>torch.rewrite</code>, and
every one of the thirty timed calls of a run within a millisecond of the others. The probe, run again: 23.3 to 23.8 ms, every call. Two
stable numbers for one function.</p>""")
    A("<h3>The runner's own loop inside the probe</h3>")
    A("""<p>If the probe's copy of the timed loop differs from the runner's, running the runner's loop inside the probe should show it. It
did the opposite: the probe's loop measured 30.2 ms, then the runner's own loop, called right after it in the same process, 23.7 ms, then
the probe's loop again 30.1 ms. A wrapper was then put around the runner's <code>timed</code> function to log each call's time, and with
it every loop measured 23.4 ms. A measurement that changes what it measures; but what a logging wrapper changes is only that there is one
more Python function between the loop and the step.</p>""")
    A("<h3>One thing at a time</h3>")
    A(C.table(["how the runner is started", "speedup over eager"], [
        ["<span>as it is: <code>python benchmarks/dynamo/timm_models.py ...</code></span>", "1.77x, 1.79x"],
        ["<span>by the absolute path of the script</span>", "1.77x"],
        ["<span>imported as a module and called (<code>python -c</code>)</span>", "1.76x"],
        ["<span>with <code>torch.rewrite</code>'s modules imported before the suite</span>", "1.77x"],
        ["<span>with the experiment function behind a wrapper that only calls it</span>", "2.26x, 2.29x, 2.31x"],
    ], num_from=1))
    A("""<p>Eager takes 54 ms a step, so 1.77x is 30.5 ms and 2.28x is 23.7 ms. Nothing about how the process starts matters, and one
extra frame of 160 bytes on the stack does. The runner's 1.77x is as real as the probe's 2.28x: each is what that stack gives.</p>""")
    A("<h3>A sampler, and two wrong ideas</h3>")
    A("""<p><code>py-spy</code> was attached to the runner during a long timed loop, once as it is and once with the wrapper. With native
stacks it could not unwind through the Rust VM that runs the rewritten function, and lost most samples of the compiled call. With Python
stacks only, the two runs differ in one place: the samples whose innermost frame is a line of the generated function of the forward graph.</p>""")
    A(C.table(["share of all samples", "runner as it is (1.72x under the sampler)", "with the wrapper (2.28x)"], [
        ["under the forward graph's call", "10.0%", "6.9%"],
        ["innermost frame is a line of the generated function", "9.1%", "4.8%"],
        ["of those, lines that call a convolution", "5.5%", "4.0%"],
        ["lines that launch a Triton kernel", "2.3%", "1.5%"],
        ["lines that allocate an output", "1.3%", "0.7%"],
    ]))
    A("""<p>Every kind of line was slower by about half, which looked like the interpreter itself running slower there. Two ideas came from
that, and both were wrong. The first: <code>torch.rewrite</code> installs a frame evaluation function (PEP 523) while a rewritten function
runs, and takes it out around the call of a compiled graph; if it stayed in, every Python call under the graph would go through it. The
code takes it out, and a convolution is a C function, which no evaluation function touches. The second: another Python thread holding the
GIL. The samples have no such thread. What the table shows in hindsight: the time is at the lines of the generated function, which
is where a sampler of Python frames sees the interpreter while it makes room for a callee's frame. But the lines that call a convolution
or allocate an output call C functions, which make no frame, and they were slower too. Unmapping memory several hundred times a step
has costs beyond the call that does it; that is as far as this was taken.</p>""")
    A("<h3>The same call from 140 depths</h3>")
    A("""<p>If one frame more makes the step fast, how many frames make it slow again? The step was called through a chain of small functions
of growing length, each adding a frame of 120 bytes.</p>""")
    A(fig_depth_coarse())
    A(fig_depth())
    A("""<p>A period of 16 KB in the depth of the Python stack is the size of a chunk of CPython's frame stack. The window is four or five
frames of 120 bytes wide: about 3% of all depths.</p>""")
    A("<h3>The frames, and a count</h3>")
    A(f"""<p>The interpreter's own frames were then read at the moment the forward graph launches its first kernel (a wrapper around CPython's
arena allocator, installed without adding a frame to any stack, notes which function is running when a chunk is asked for, and how much the
current chunk has left). Under <code>torch.rewrite</code> there are {len(REWRITE_CASE['frames_innermost_first'])} frames from the script's
module down to the generated function, {rewrite_bytes:,} bytes, of which the generated function's frame is 11,176: it has 1,027 local
variables and room for 361 values on its stack. The frames of a thread's first chunk start {START} bytes into it, so the chunk has
{CHUNK:,} − {START} − {rewrite_bytes:,} = {REWRITE_CASE['bytes_left_in_the_chunk']} bytes left. The first frame a kernel launch calls
(<code>CachingAutotuner.run</code>) is 288 bytes.</p>""")
    A(chunk_diagram())
    A(f"""<p>The count, over the thirty repetitions of the runner's timed loop: {sum(asked.values()):,} chunks asked for,
{REWRITE_CASE['asked_in_each_repetition'][1]} in every repetition: {asked['kernel launch'] // 30} kernel launches,
{asked['assert_tensor_metadata'] // 30} calls of <code>assert_tensor_metadata</code> (a Python function the generated code calls after an
extern kernel), and three others. With the allocator keeping the freed chunks and handing them back, and nothing else changed, the same
runner measures 2.25x where it measured 1.77x. That closed it.</p>""")
    A("""<p>It also explains the probe that disagreed with itself: its loop and the runner's loop are two functions with frames of different
sizes, and the difference put one of them inside the window and the other outside.</p>""")

    A("<h2>3. What the interpreter does</h2>")
    A(f"""<p>Since 3.11 CPython does not allocate a frame per call. A thread has a stack of frames made of chunks of 16 KB
(<code>DATA_STACK_CHUNK_SIZE</code>); a call bumps a pointer, and a return bumps it back. When a frame does not fit in what is left of the
current chunk, the interpreter asks its arena allocator for a new chunk (an <code>mmap</code>), puts the frame at its start, and links
the chunk to the one before. When that frame returns, <code>_PyThreadState_PopFrame</code> sees that it was the first of its chunk and
frees the chunk (a <code>munmap</code>). The next call from the same place does it all again: two system calls and a page fault where a
call otherwise moves a pointer.</p>
<p>It takes no PyTorch to see it. <code>tools/chunk_cliff.py</code> calls a small function 2000 times from each of 200 depths of the stack
and reads the page faults of the process around each:</p>""")
    A(version_table())
    A(fig_cliff())
    A(f"""<p>The CPython developers know it as <a href="{CPY}issues/142183">"Calls across stack chunks perform badly"</a> (December 2025).
The change that keeps one freed chunk per thread (<a href="{CPY}pull/145789">gh-145789</a>, "Cache one datachunk per tstate to prevent
alloc/dealloc thrashing") was merged in March 2026 and backported to 3.14 and to 3.13 that month; the 3.13.15 and 3.14.7 on this machine
have the field (<code>datastack_cached_chunk</code>) and no slow depth. 3.11 and 3.12 take security fixes only, and the newest of each,
3.11.16 and 3.12.14, have the cliff. The build these measurements were made on is CPython 3.12; the perf dashboard's jobs
(<code>inductor-perf-test-nightly-h100.yml</code>) set <code>python-version: "3.11"</code>.</p>""")
    A("""<p>Why a compiled graph is where it shows: the cliff needs a function that makes many calls and whose own frame ends just before a
chunk's end. Ordinary functions have frames of a few hundred bytes and make a handful of calls each. The function Inductor generates for
a graph has a frame of several kilobytes to tens of kilobytes (every buffer is a local variable) and makes one call per kernel, hundreds
to thousands in a row.</p>""")

    A("<h2>4. Whom it hits</h2>")
    A("""<p>The count was then made for every run of the dashboard's suites, with the runner started as the dashboard starts it. Each model
was run under the compilers one right after the other, on the build that keeps the freed chunks, so a chunk that <code>torch.rewrite</code>
asks for is counted but costs nothing, and one that <code>torch.compile</code> asks for is mapped as it would be on the dashboard.</p>""")
    A(fig_who())
    spared = SPARED_CASE["frames_outermost_first"]
    resume = next(size for name, _, size in spared if name.endswith("_at_420"))
    A(f"""<p><code>torch.compile</code> does not have it on tf_efficientnet_b0 because Dynamo splits the runner's training step where the
model is called, and the function that resumes after the split has a frame of {resume:,} bytes: with it on the stack the generated
function never fits in the first chunk, starts its own, and has 5 KB left for what it calls (second row of the figure above). On
ElectraForCausalLM the generated function is smaller (7,368 bytes), fits, and ends {COMPILE_CASE['bytes_left_in_the_chunk']} bytes
before the end of the chunk: {electra['kernel launch'] // 30} kernel launches a step each map a chunk. Six of 150 training runs, none of 258
inference runs: which ones is decided by the sizes of two dozen frames, and would be other models from another script, another Python
version, or one more decorator.</p>""")
    A("""<p>What the mapping costs was measured both ways: for <code>torch.rewrite</code> as the difference between the pairs before and
after the runtime kept its chunks, for <code>torch.compile</code> by running its four models a second time with the same allocator wrapper
keeping chunks in its process (which changes nothing else about <code>torch.compile</code>).</p>""")
    A(model_table())
    A(cost_table())
    A("""<p>What a chunk costs a step is not a constant: from nothing measurable to 14 µs, where mapping and unmapping one takes 7 µs in a
small single-threaded process. The low numbers have a likely reason that was not measured: a step that waits for the GPU anyway hides CPU
time spent on chunks (nfnet_l0's convolutions are long), and a step of many short kernels does not. The 14 µs are twice what the small
process pays; a process with a GPU context has more threads and a larger address space to unmap from, and that is only a guess.</p>""")

    A("<h2>5. What was done</h2>")
    A(f"""<ul>
<li><strong>The runtime keeps the chunks</strong> ({pr(205502)}). CPython asks its arena allocator for chunks, and that allocator can be
replaced through a public function (<code>PyObject_SetArenaAllocator</code>). When the first function is handed to
<code>torch.rewrite</code>, the runtime puts a small allocator in front of it that answers a request for exactly 16 KB with a chunk it
kept, keeps up to four chunks of that size when they are freed, and hands every other request through. At most 64 KB are held. A test
counts the page faults of 2000 calls from each of 160 depths: 2000 at one depth without the change, 1 with it. Under CPython 3.13 and 3.14
the interpreter's own cache answers first and this is seldom asked for anything.</li>
<li><strong>Not done: anything for <code>torch.compile</code>.</strong> The same wrapper would take the accident out of its runs on 3.11
and 3.12, and so could generated code that makes sure of some room before it starts launching. Neither is part of this stack. In these
runs the accident made four of its 75 training models' steps longer by nothing to 12%.</li>
<li><strong>The numbers were measured again</strong>, all of them, in pairs, with the count next to each (below).</li>
</ul>""")
    A("<h3>Found on the way: a compile inside the timed loop</h3>")
    A(f"""<p>The same count showed thousands of chunks in inference runs of <code>torch.rewrite</code>, all of them in the first repetition
of the timed loop and none after: a compile. The runner warms a function up with its default arguments and then times it with
<code>collect_outputs=False</code>; that flag is an argument of the region <code>torch.rewrite</code> compiles, part of its key, though no
graph uses it, so the first timed call compiled the same graph a second time. The medians of thirty repetitions were not affected, but the
compile time the runner reported left that compile out. The warm-up for <code>--rewrite</code> now calls the function as the timed loop
does ({pr(205357)}), and its compile time is what the warm-up took beyond as many calls of the settled function. That an unused argument is
in a key at all is a weakness of <code>torch.rewrite</code>, and is still there.</p>""")

    A("<h2>6. The suites again</h2>")
    A("""<p>Time of <code>torch.rewrite</code> over the time of <code>torch.compile</code>, geometric means; below 1 is faster. One H100,
CPython 3.12, cuDNN 9.26; every model under the compilers back to back, 408 runs, no other process on the GPU during any of them.</p>""")
    A(suite_table())
    kept = PAIRS["models"]["ElectraForCausalLM"]
    A(f"""<p>timm for training went from 1.011 to 0.997 with the two models back in line. HuggingFace for training did not move, because the
accident there is <code>torch.compile</code>'s: with its four models run again and their chunks kept, its step of ElectraForCausalLM is
{(1 - kept['second pair, chunks kept for torch.compile too']['torch.compile']['ms'] / kept['with the fix']['torch.compile']['ms']):.0%}
shorter, HuggingFace goes from 1.002 to 1.008 and training as a whole from 0.988 to 0.990. Inference did not move at all.</p>
<p>A caution that the second set of pairs made possible: between the two runs of the training pairs, five hours apart, the ratio of a
model moved by 1.3% at the median and by up to 11% for the 57 models whose step takes 8 ms or more, and by 5.5% and up to 26% for the 16
with shorter steps. Part of that is the machine and part is <code>torch.rewrite</code> itself, which decides by measurement which
functions to leave to the interpreter and does not always decide the same. The means are steady; a single model's ratio is not, and a gap
of a few percent on one model is not a finding until it has been seen twice.</p>""")

    A("<h2>7. What it took</h2>")
    A(C.table(["step", "runs on the GPU, about", "wall-clock time, about"], [
        ["three more pairs, the probe again, the runner's loop inside the probe", "15", "30 min"],
        ["one thing at a time: five ways to start the runner", "8", "12 min"],
        ["py-spy, native and Python stacks, slow and fast", "4", "10 min"],
        ["the depth scans, the frames, the count, the count with chunks kept", "8", "12 min"],
        ["the fix, its test, the plain runner on the two models", "3", "4 min"],
        ["a first count for fifteen models under both compilers", "30", "16 min"],
        ["training pairs: 75 models, two compilers", "150", "1 h 58 min"],
        ["inference pairs: 86 models, three configurations (and 17 min of a first pass, discarded when the compile in the loop was found)", "258", "1 h 50 min"],
        ["the second pairs of torch.compile's four models", "8", "7 min"],
        ["exact frames and bytes left for this page; the depth scan on the fixed build", "6", "8 min"],
    ]))
    A("""<p>About an hour from "two stable numbers for one function" to the cause, most of it on ideas that were wrong, and four hours of
GPU time to measure everything again. The step that found it took four minutes and no theory: change the depth and plot the time. The
measurements that did not find it were the ones that looked for where the time goes; the one that did asked what the time depends on.</p>""")

    A("<h2>8. What is not known</h2>")
    A("""<ul>
<li>Whether the perf dashboard's own numbers carry this for some models. Its jobs run CPython 3.11, which has the cliff, and its runner is
the one measured here; but frame sizes differ between 3.11 and 3.12, so the models would be other ones. Nothing was run on 3.11 with
PyTorch.</li>
<li>Why a chunk costs a step 14 µs in one process and nothing measurable in another (section 4 has a guess).</li>
<li>The backward graphs run on the autograd engine's thread, which has a frame stack of its own. The count covers all threads, and nothing of size showed on that one in these suites; a model whose backward function ends near the end of
that thread's first chunk would have it.</li>
<li>Other callers. Everything here is the dashboard's runner. A training loop of a user's has its own depth, and on CPython 3.11 or 3.12
about one position in thirty is a bad one for a given graph.</li>
</ul>""")

    A("<h2>9. Reproducing</h2>")
    A(f"""<p>The <a href="https://github.com/bobrenjc93/explanations/tree/main/frame-stack-chunks">folder of this page</a> has the data
(<code>results/</code>) and the tools (<code>code/tools/</code>, with a README): the plain-Python script, the depth scan, the allocator
wrappers in C, and the <code>sitecustomize.py</code> that installs a wrapper at the start of the runner's timed loop without adding a frame.
The shortest way to see the cliff is <code>python3.12 code/tools/chunk_cliff.py</code>. For a benchmark of one's own on CPython 3.11 or
3.12: count the chunks, or time it once more with a frame added above it.</p>
<pre><code>TOOLS=/path/to/explanations/frame-stack-chunks/code/tools
gcc -O2 -shared -fPIC -I$(python -c "import sysconfig; print(sysconfig.get_paths()['include'])") \\
  -o /tmp/arena_cache.so $TOOLS/arena_cache.c
cd /path/to/pytorch
PYTHONPATH=$TOOLS CHUNK_LIB=/tmp/arena_cache.so CHUNK_COUNTS=/tmp/chunks.log \\
  python benchmarks/dynamo/timm_models.py --performance --cold-start-latency --training --amp \\
  --disable-cudagraphs --device cuda --backend inductor --only nfnet_l0
cat /tmp/chunks.log   # on the build measured here: "7800 asked, 0 reused, keep 0, 7800 in the loop,
                      #                               7540 after its first repetition | ..."</code></pre>""")
    A("""<footer>Measured on one NVIDIA H100, CUDA 12.8, cuDNN 9.26, CPython 3.12.14, PyTorch main of Nov 17, 2026 with the
<code>torch.rewrite</code> stack on it. Written with an AI assistant (Claude Code).</footer>""")
    return "\n".join(o)


def main(out):
    html = f"""<!doctype html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>{TITLE}</title>
<meta name="description" content="{SUB}">
<link rel="icon" href="../favicon.svg" type="image/svg+xml">
<link rel="icon" href="../favicon.ico" sizes="any">
<link rel="apple-touch-icon" href="../apple-touch-icon.png">
<style>{C.CSS}{C.EXTRA_CSS}</style>
</head>
<body>
{body()}
<script>{C.JS}</script>
</body>
</html>
"""
    open(out, "w").write(html)
    print("wrote", out, len(html) // 1024, "KB")


if __name__ == "__main__":
    main(sys.argv[1] if len(sys.argv) > 1 else "draft.html")
