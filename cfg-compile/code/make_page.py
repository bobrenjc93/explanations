"""Builds the torch.rewrite report page: prose here, figures from charts.py, numbers from data.py.

    python make_page.py OUT.html
"""
import sys

import charts as C
import data as D

PR = "https://github.com/pytorch/pytorch/pull/"
E, T = D.summary(D.FINAL["eval"]), D.summary(D.FINAL["train"])
EV, TR = D.FINAL["eval"], D.FINAL["train"]


def pr(n, text=None):
    return f'<a href="{PR}{n}">{text or "#" + str(n)}</a>'


def prs(a, b):
    return f"{pr(a)} to {pr(b)}"


def x(v):
    return f"{v:.2f}x"


def names(test, which, limit=None, fmt="{n} {v:.2f}"):
    s = E if test == "eval" else T
    items = sorted((s["alone"][n], n) for n in which if n in s["alone"])
    if limit:
        items = items[:limit]
    return ", ".join(fmt.format(n=n, v=v) for v, n in items)


# --------------------------------------------------------------------------- figures


def fig_path():
    out = []
    for test, label in (("eval", "Inference"), ("train", "Training")):
        ss = [D.summary(by[test]) for _, _, by in D.VERSIONS]
        series = [
            {"name": "torch.rewrite", "short": "torch.rewrite", "c": "--s1", "y": [s["speed"]["cfg"] for s in ss]},
            {"name": "torch.compile", "short": "torch.compile", "c": "--s2", "y": [s["speed"]["compile"] for s in ss]},
        ]
        body = C.line_chart(series, list(range(len(D.VERSIONS))), xlabels=[v for v, _, _ in D.VERSIONS], w=400, h=250, ymin=1.0, ymax=2.0,
                            yticks=(1.0, 1.25, 1.5, 1.75, 2.0), right=128, yfmt=lambda v: f"{v:.2f}x", vfmt=x, tv_header="version",
                            xlab="")
        n = len(ss[-1]["names"])
        out.append(C.figure(f"{label}: speedup over eager", f"Geometric mean over the {n} models both compilers ran, by version of the graph backend.", body))
    return '<div class="grid2">' + "".join(out) + "</div>"


def model_rows(test):
    s, models = (E, EV) if test == "eval" else (T, TR)
    items, rows = [], []
    for v, n in sorted((s["alone"][n], n) for n in s["names"]):
        d = models[n]
        p = D.turns(d)
        tip = [("eager", D.ms(d["eager"]["call_s"])), ("torch.compile", D.ms(d["compile"]["call_s"])), ("torch.rewrite", D.ms(d["cfg"]["call_s"])),
               ("in turns", "-" if p is None else f"{p:.2f}"),
               ("graphs, torch.compile / torch.rewrite", f"{d['compile'].get('graphs', '-')} / {d['cfg'].get('graphs', '-')}")]
        items.append({"name": n, "v": v, "tip": tip})
        rows.append([n, D.ms(d["eager"]["call_s"]), D.ms(d["compile"]["call_s"]), D.ms(d["cfg"]["call_s"]), f"{v:.2f}", "-" if p is None else f"{p:.2f}",
                     x(d["eager"]["call_s"] / d["compile"]["call_s"]), x(d["eager"]["call_s"] / d["cfg"]["call_s"]),
                     f"{d['compile'].get('compile_s', 0):.1f} / {d['cfg'].get('compile_s', 0):.1f}",
                     f"{d['compile'].get('graphs', '-')} / {d['cfg'].get('graphs', '-')}",
                     f"{d['compile'].get('difference', 0):.0e} / {d['cfg'].get('difference', 0):.0e}"])
    header = ["model", "eager", "torch.compile", "torch.rewrite", "time ratio, alone", "in turns", "torch.compile over eager", "torch.rewrite over eager",
              "compile s (tc / cfg)", "graphs (tc / cfg)", "difference from eager (tc / cfg)"]
    return items, header, rows


def fig_models(test):
    items, header, rows = model_rows(test)
    s = E if test == "eval" else T
    lo = 0.30 if test == "eval" else 0.42
    body = C.ratio_bars(items, lo, 1.25, (0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0, 1.1, 1.2) if test == "eval" else (0.5, 0.6, 0.7, 0.8, 0.9, 1.0, 1.1, 1.2),
                        tv_header=header, tv_rows=rows)
    what = "inference" if test == "eval" else "a training step"
    cap = (f"Time of one call under <code>torch.rewrite</code> over the time under <code>torch.compile</code>, each in a process of its own; "
           f"{what}, {len(items)} models, logarithmic scale. Below 1 is faster. Geometric mean {s['alone_geo']:.2f}. Hover or focus a row for the times.")
    return C.figure(f"{'Inference' if test == 'eval' else 'Training'}, per model", cap, body)


def fig_memory(test="train"):
    h = D.held(TR if test == "train" else EV)
    items = [{"name": n, "v": r, "tip": [("torch.rewrite", f"{a:.1f} GB"), ("torch.compile", f"{b:.1f} GB"), ("eager", f"{e:.1f} GB")]} for r, n, a, b, e in h]
    rows = [[n, f"{e:.1f}", f"{b:.1f}", f"{a:.1f}", f"{r:.2f}"] for r, n, a, b, e in h]
    top = max(2.0, max(r for r, *_ in h) * 1.05)
    body = C.ratio_bars(items, 0.7, top, (0.8, 1.0, 1.25, 1.5, 2.0), band=0.05, names=("torch.rewrite holds less", "within 5%", "torch.rewrite holds more"),
                        tv_header=["model", "eager GB", "torch.compile GB", "torch.rewrite GB", "torch.rewrite over torch.compile"], tv_rows=rows)
    what = "the training process" if test == "train" else "the inference process"
    cap = (f"Peak resident memory of {what} under <code>torch.rewrite</code>, before <code>torch.compile</code> joins it for the calls in turns, over that "
           f"of the process of <code>torch.compile</code>; {len(h)} models, logarithmic scale. Geometric mean {D.geo(r for r, *_ in h):.2f}.")
    return C.figure("What the processes hold while training" if test == "train" else "What the processes hold in inference", cap, body), h


def fig_gpu(test):
    rows = sorted(D.GPU[test], key=lambda r: r["ratio"])

    def ms(v):
        return f"{v:.2f} ms" if v < 100 else f"{v:.0f} ms"

    items = [{"name": r["name"], "v": r["ratio"], "tip": [("torch.rewrite", ms(r["rewrite"])), ("torch.compile", ms(r["compile"])), ("eager", ms(r["eager"]))]} for r in rows]
    table = [[r["name"], f"{r['eager']:.2f}", f"{r['compile']:.2f}", f"{r['rewrite']:.2f}", f"{r['ratio']:.2f}", f"{r['held'][1]}", f"{r['held'][2]}"] for r in rows]
    lo, hi = min(0.5, rows[0]["ratio"] * 0.85), max(1.3, rows[-1]["ratio"] * 1.05)
    body = C.ratio_bars(items, lo, hi, (0.5, 0.6, 0.7, 0.8, 0.9, 1.0, 1.1, 1.2, 1.4, 1.6), tv_header=[
        "model", "eager ms", "torch.compile ms", "torch.rewrite ms", "torch.rewrite over torch.compile", "torch.compile MB on the GPU", "torch.rewrite MB on the GPU"], tv_rows=table)
    what = "inference" if test == "eval" else "a training step"
    cap = (f"Time of one call under <code>torch.rewrite</code> over the time under <code>torch.compile</code> on an H100, each in a process of its own; {what}, "
           f"{len(rows)} models, logarithmic scale. Below 1 is faster. Geometric mean {D.geo(r['ratio'] for r in rows):.2f}. Hover or focus a row for the times.")
    return C.figure(f"On a GPU: {'inference' if test == 'eval' else 'training'}, per model", cap, body)


def gpu_summary(test):
    rows = D.GPU[test]
    ahead, behind = [r for r in rows if r["ratio"] < 0.97], [r for r in rows if r["ratio"] > 1.03]
    held = [r["held"][2] / r["held"][1] for r in rows if r["held"][1] and r["held"][2]]
    return {"n": len(rows), "geo": D.geo(r["ratio"] for r in rows), "ahead": sorted(ahead, key=lambda r: r["ratio"]), "behind": sorted(behind, key=lambda r: -r["ratio"]),
            "level": len(rows) - len(ahead) - len(behind), "compile": D.geo(r["eager"] / r["compile"] for r in rows), "rewrite": D.geo(r["eager"] / r["rewrite"] for r in rows),
            "held": D.geo(held)}


def gpu_moved(name, test="eval"):
    before = [r for r in D.GPU_FIRST[test] if r["name"] == name]
    now = [r for r in D.GPU[test] if r["name"] == name]
    if not before or not now:
        return f"{name}"
    return f"{name} from {before[0]['ratio']:.2f} to {now[0]['ratio']:.2f} of the time of <code>torch.compile</code>"


SUITE_NAMES = (("HuggingFace", "huggingface"), ("timm", "timm_models"), ("TorchBench", "torchbench"))


def suite_rows(suite, base, table=None):
    """The models of a suite that ran under `base` (inductor or freezing) and under torch.rewrite, with the time of
    torch.rewrite over the base's: the ratio of two speedups over eager, each measured in turns with eager."""
    if table is None:
        # TorchBench under torch.compile as it comes was measured again, each model next to torch.rewrite.
        table = D.SUITES_PAIRED if (suite, base) == ("torchbench", "inductor") else D.SUITES
    plain, rewrite = table.get((suite, base), {}), table.get((suite, "rewrite"), {})
    rows = []
    for name in plain:
        if name in rewrite:
            p, r = plain[name], rewrite[name]
            held = float(r["dynamo_peak_mem"]) / float(p["dynamo_peak_mem"]) if float(p["dynamo_peak_mem"]) > 0 and float(r["dynamo_peak_mem"]) > 0 else None
            rows.append({"name": name, "ratio": float(p["speedup"]) / float(r["speedup"]), "base": float(p["speedup"]), "rewrite": float(r["speedup"]), "batch": p["batch_size"],
                         "held": held, "compile": (max(float(p["compilation_latency"]), 0.01), max(float(r["compilation_latency"]), 0.01))})
    return sorted(rows, key=lambda r: r["ratio"])


def suite_summary(base, table=None):
    out = {}
    for title, suite in SUITE_NAMES:
        rows = suite_rows(suite, base, table)
        if rows:
            out[suite] = {"title": title, "n": len(rows), "geo": D.geo(r["ratio"] for r in rows), "ahead": [r for r in rows if r["ratio"] < 0.97],
                          "behind": [r for r in rows if r["ratio"] > 1.03][::-1], "base": D.geo(r["base"] for r in rows), "rewrite": D.geo(r["rewrite"] for r in rows),
                          "held": D.geo(r["held"] for r in rows if r["held"]), "compile": (D.geo(r["compile"][0] for r in rows), D.geo(r["compile"][1] for r in rows))}
    rows = [r for _, suite in SUITE_NAMES for r in suite_rows(suite, base, table)]
    if rows:
        out["all"] = {"n": len(rows), "geo": D.geo(r["ratio"] for r in rows), "ahead": sum(r["ratio"] < 0.97 for r in rows), "behind": sum(r["ratio"] > 1.03 for r in rows),
                      "base": D.geo(r["base"] for r in rows), "rewrite": D.geo(r["rewrite"] for r in rows), "held": D.geo(r["held"] for r in rows if r["held"]),
                      "compile": (D.geo(r["compile"][0] for r in rows), D.geo(r["compile"][1] for r in rows))}
    return out


def listed(rows, limit=5):
    return ", ".join(f"{r['name']} {r['ratio']:.2f}" for r in rows[:limit]) or "none"


def fig_suite(suite, title, training=False):
    """One suite, per model: against torch.compile with freezing for inference, against torch.compile for training."""
    table = D.SUITES_TRAIN if training else D.SUITES
    rows = suite_rows(suite, "inductor" if training else "freezing", table)
    plain = {r["name"]: r for r in suite_rows(suite, "inductor", table)}
    items, cells = [], []
    for r in rows:
        p = plain.get(r["name"])
        tip = [("torch.rewrite", x(r["rewrite"]) + " eager")]
        if training:
            tip.append(("torch.compile", x(r["base"]) + " eager"))
        else:
            tip.append(("torch.compile, freezing", x(r["base"]) + " eager"))
            if p:
                tip.append(("torch.compile", x(p["base"]) + " eager"))
        tip.append(("batch size", r["batch"]))
        items.append({"name": r["name"], "v": r["ratio"], "tip": tip})
        held = f"{r['held']:.2f}" if r["held"] else ""
        if training:
            cells.append([r["name"], r["batch"], f"{r['base']:.2f}", f"{r['rewrite']:.2f}", f"{r['ratio']:.2f}", held])
        else:
            cells.append([r["name"], r["batch"], f"{p['base']:.2f}" if p else "", f"{r['base']:.2f}", f"{r['rewrite']:.2f}", f"{p['ratio']:.2f}" if p else "", f"{r['ratio']:.2f}", held])
    # Room left of the longest bar for its value, and ticks that do not crowd on a wide scale.
    lo, hi = min(0.6, rows[0]["ratio"] * 0.75), max(1.3, min(rows[-1]["ratio"] * 1.08, 3.6))
    marks = (0.25, 0.35, 0.5, 0.7, 1.0, 1.4, 2.0, 3.0) if hi / lo > 5 else (0.5, 0.6, 0.7, 0.8, 0.9, 1.0, 1.1, 1.2, 1.3, 1.4)
    ticks = tuple(t for t in marks if lo <= t <= hi)
    against = "torch.compile" if training else "torch.compile with freezing"
    header = (["model", "batch size", "torch.compile over eager", "torch.rewrite over eager", "time over torch.compile", "peak GPU memory over torch.compile's"] if training else
              ["model", "batch size", "torch.compile over eager", "with freezing, over eager", "torch.rewrite over eager", "time over torch.compile", "time over torch.compile with freezing",
               "peak GPU memory over freezing's"])
    label_w = max(214, 7 * max(len(r["name"]) for r in rows) + 18)
    body = C.ratio_bars(items, lo, hi, ticks, label_w=label_w, names=("torch.rewrite faster", "within 3%", against + " faster"), tv_header=header, tv_rows=cells)
    what = "a training step" if training else "an inference call"
    cap = (f"Time of {what} under <code>torch.rewrite</code> over the time under <code>{'torch.compile' if training else 'torch.compile</code> with <code>freezing'}</code>, on an H100 with the "
           f"dashboard's runner and settings; {len(rows)} models, logarithmic scale. Below 1 is faster. Geometric mean {D.geo(r['ratio'] for r in rows):.2f}. "
           f"Hover or focus a row for the speedups over eager" + ("." if training else "; the table has the comparison with <code>torch.compile</code> as it comes."))
    return C.figure(f"On a GPU: {title}, {'training' if training else 'inference'}, per model", cap, body)


def gpu_since(name, test="eval"):
    """Where a model stood in the harness's GPU run of the sixth version and where in the seventh's (no cuDNN in either)."""
    before = [r for r in D.GPU[test] if r["name"] == name]
    now = [r for r in D.GPU7[test] if r["name"] == name]
    if not before or not now:
        return name
    return f"{name} from {before[0]['ratio']:.2f} to {now[0]['ratio']:.2f}"


def fig_micro():
    items = [
        {"name": "chain of 9 ops", "v": 1.52, "tip": [("eager", "18,377 ns"), ("torch.rewrite", "12,059 ns")]},
        {"name": "for loop, 10 x (3 ops + branch)", "v": 1.50, "tip": [("eager", "94,558 ns"), ("torch.rewrite", "62,965 ns")]},
        {"name": "while loop on a tensor value", "v": 1.48, "tip": [("eager", "47,186 ns"), ("torch.rewrite", "31,840 ns")]},
        {"name": "2-layer MLP (addmm)", "v": 1.11, "tip": [("eager", "9,130 ns"), ("torch.rewrite", "8,246 ns")]},
        {"name": "Python loop over 8 tensors", "v": 0.96, "tip": [("eager", "10,930 ns"), ("torch.rewrite", "11,419 ns")]},
    ]
    body = C.value_bars(items, 2.0, (0, 0.5, 1.0, 1.5, 2.0), ref=1.0, ref_label="eager", tv_header=("workload", "speedup over eager"))
    return C.figure("The VM alone, without a graph backend: speedup over eager",
                    "64-element tensors on one CPU thread, under <code>no_grad</code>; best of five interleaved rounds. "
                    "These are small on purpose: what the specialized tier removes is the cost per operation.", body)


# --------------------------------------------------------------------------- tables with numbers measured by hand

TUNED = [
    ("resnet50", 745, 682, 683), ("mobilenet_v2", 89.8, 65.9, 64.8), ("densenet121", 1243, 1198, 1204), ("mobilenet_v3_large", 125.2, 120.0, 121.8),
    ("shufflenet_v2_x1_0", 177.5, 153.6, 161.7), ("mnasnet1_0", 152.9, 138.4, 126.8), ("squeezenet1_1", 70.8, 57.7, 39.0), ("vgg16", 356, 366, 348),
    ("alexnet", 609, 609, 630), ("resnext50_32x4d", 208.2, 200.9, 200.7), ("resnet18", 89.4, 84.7, 84.4), ("hf_Bert", 469, 468, 469),
    ("hf_BigBird", 1101, 1094, 959), ("functorch_dp_cifar10", 18.7, 14.4, 14.3),
]

GROUPS = [
    ("The dispatcher hook and what Python sees of it", 5, "1,352", "A sink in the dispatcher that sees every kernel call with its dispatch keys, and a Python binding for it. These five run CI."),
    ("Call sites: calling a kernel directly", 3, "1,681", "From one recorded operation to an object that calls the post-dispatch kernel again, boxed or typed."),
    ("The Rust crate: IR, frontend, VM, specialization, passes", 6, "7,644", "Bytecode to SSA, the three tiers of the VM, guards, the pass pipeline (passes in Rust or Python)."),
    ("Graph transforms and calls", 4, "1,553", "Inlining under a guard, unrolling of constant loops, compiling what a function calls."),
    ("More of Python", 9, "1,929", "Closures, exceptions, <code>with</code>, imports, classes, <code>del</code>, <code>match</code>."),
    ("Behaving like the interpreter", 11, "1,394", "Frames and tracebacks, object lifetimes, the recursion limit, error messages, signals, threads, garbage collection."),
    ("Speed", 4, "1,329", "Shortcuts through <code>nn.Module</code>'s call and attribute code; measuring compiled code against the interpreter and keeping the faster."),
    ("A graph compiler behind the VM", 7, "3,561", "<code>backend=\"inductor\"</code>: regions of a block traced to one ATen graph, compiled forward and backward."),
    ("Regions across basic blocks, and measuring them", 8, "2,041", "A region follows the path its trace takes through loops and branches; the benchmark harness."),
    ("Sizes left open, and state a region changes", 11, "2,367", "Sizes and integers that vary become inputs of one graph under conditions checked in Rust; effects on lists, dicts, objects, module tensors."),
    ("Memory, weights and one-row products", 14, "1,651", "The fourth version, described below, with two leak fixes and three fixes to the benchmark."),
    ("The variables of the frame, and memory again", 8, "1,151", "The fifth version: frames that have their variables, <code>eval</code> and <code>exec</code> compiled, trace functions, and what the processes hold."),
    ("A list of passes, the frame hook, passes of the user's, CPython 3.14, the GPU", 17, "4,484", "The sixth version: <code>torch.rewrite(fn, passes=[...])</code>, the functions that C code calls, <code>swap</code> and passes in Rust, the port to 3.14, and three fixes found on the GPU."),
    ("The GPU again: sizes that depend on data, the cost of a region's call, the dashboard's runner", 18, "2,374", "The seventh version: a size that only data decides stays in a graph, a call like the one before is told in the runtime, the perf dashboard's runner measures <code>torch.rewrite</code>, and what its suites found, the update of an optimizer and the chunks of the interpreter's frame stack among it."),
    ("Tests, benchmarks and tools", 4, "510", "The micro benchmarks and the runner for CPython's own test suite."),
]


def tuned_table():
    rows = [[n, f"{a:g}", f"{b:g}", f"{c:g}", f"{c / min(a, b):.2f}"] for n, a, b, c in TUNED]
    return C.table(["model (inference, ms per call)", "torch.compile", "torch.compile with freezing", "torch.rewrite, defaults", "torch.rewrite over the better of the two"], rows)


def groups_table():
    body = "".join(f"<tr><td>{g}</td><td class=num>{n}</td><td class=num>{lines}</td><td>{what}</td></tr>" for g, n, lines, what in GROUPS)
    head = "<tr><th>part of the stack</th><th class=num>PRs</th><th class=num>inserted lines</th><th>what it is</th></tr>"
    return f'<div class="scroll"><table><thead>{head}</thead><tbody>{body}</tbody></table></div>'


# --------------------------------------------------------------------------- the page

TITLE = "torch.rewrite: a function as a control-flow graph and a list of rewrites, measured against torch.compile"
SUB = ("A prototype that compiles the bytecode of a Python function to one control-flow graph, runs it in a Rust VM and hands the tensor work "
       "to Inductor: what was built over 129 draft PRs, how it compares with torch.compile on TorchBench and on the perf dashboard's suites in accuracy, speed, compile time and "
       "memory, where it is ahead, where it is behind, and what to do next. Nov 11 to Nov 18, 2026.")


def body():
    mem_fig, h = fig_memory("train")
    mem_fig_eval, he = fig_memory("eval")
    held_geo, held_geo_eval = D.geo(r for r, *_ in h), D.geo(r for r, *_ in he)
    worst, worst_eval = h[::-1][:4], he[::-1][:3]
    h4 = D.held(D.V4["train"])
    E4, T4 = D.summary(D.V4["eval"]), D.summary(D.V4["train"])

    def gb(v):
        return f"{v:.1f} GB" if v >= 2 else f"{v * 1024:.0f} MB"

    # Against the last version, for the models whose row was measured the same way there.
    before = {n: d["cfg"]["peak_rss_mb"] for n, d in D.V4["train"].items() if d.get("cfg", {}).get("ok") and d["cfg"].get("process_peak_rss_mb") is not None}
    change = sorted((a * 1024 / before[n], n) for r, n, a, b, e in h if n in before)
    fewer, more = [c for c in change if c[0] < 0.97], [c for c in change if c[0] > 1.03][::-1]
    o = []
    A = o.append
    A(f"<h1>{TITLE}</h1>")
    A(f'<p class="sub">{SUB}</p>')

    A(f"""<div class="tldr">
<p><strong>TL;DR.</strong> <code>torch.rewrite</code> is a prototype of a different front end for PyTorch's compiler. It turns the
bytecode of a function into one control-flow graph in SSA form, runs that graph in a VM written in Rust, traces on real tensors, and hands the
tensor work of each path through the graph to AOTAutograd and Inductor, the compiler <code>torch.compile</code> uses behind Dynamo. It exists as
a stack of 129 draft pull requests against pytorch/pytorch, about 35,000 lines. It was called <code>torch.cfg_compile</code> until the
sixth version, which renamed it for what the one who calls it does: hand over a function and a list of rewrites.</p>
<p>On the TorchBench models that run on this CPU box, it takes <strong>{E['alone_geo']:.2f} of the time of <code>torch.compile</code> for
inference</strong> ({len(E['names'])} models) and <strong>{T['alone_geo']:.2f} for training</strong> ({len(T['names'])} models), geometric
means. That is {x(E['speed']['cfg'])} and {x(T['speed']['cfg'])} eager, where <code>torch.compile</code> is {x(E['speed']['compile'])} and
{x(T['speed']['compile'])}. Its outputs agree with eager's as closely as <code>torch.compile</code>'s do, and it compiles in less time.</p>
<p>Three things produce the margin, none of them in a kernel: the compiled function keeps the memory of
a call for the next one, the weights of a model are built into the graphs of inference behind a check on every call, and matrix products
with one row are rewritten so that they use every core. <strong>It is a margin over <code>torch.compile</code> with its defaults.</strong>
Given an allocator that keeps memory and its own <code>freezing</code> switch, <code>torch.compile</code> is level on most models, because
the kernels are the same.</p>
<p>It holds more memory: {held_geo:.2f} times what the process of <code>torch.compile</code> holds at its peak in training and
{held_geo_eval:.2f} times in inference (geometric means; the last version had single models at four times). It is behind on models whose
call takes under a few milliseconds, on plain Python in the VM, and in everything that was not built: generators,
modules with hooks.</p>
<p><strong>On a GPU</strong> (one H100, the three suites of the PyTorch perf dashboard at its settings) it takes
<strong>{suite_summary("inductor")["all"]["geo"]:.2f} of the time of <code>torch.compile</code> for inference</strong> over
{suite_summary("inductor")["all"]["n"]} models, and {suite_summary("freezing")["all"]["geo"]:.2f} of the time of <code>torch.compile</code>
with its <code>freezing</code> switch, which is the fair comparison there. It pays for that in memory: {suite_summary("inductor")["all"]["held"]:.2f}
times the peak GPU memory of <code>torch.compile</code> while a model is warmed up and compiled. For training it takes
{suite_summary("inductor", D.SUITES_TRAIN)["all"]["geo"]:.2f} of the time of <code>torch.compile</code> over
{suite_summary("inductor", D.SUITES_TRAIN)["all"]["n"]} models. The GPU numbers this page had before Nov 18 were measured on a build of
PyTorch without cuDNN (section 9).</p>
<p>The sixth version also made what is done to a function one list of passes, to which a user adds passes of their own, in Python or
in Rust; rewrites the functions that C code calls (constructors, properties, operators, callbacks), through the frame hook Dynamo uses;
and runs on CPython 3.14 as well as 3.12.</p>
<p>A rewritten function also behaves like an interpreted one where tools look: its frame has its variables, so
tracebacks, debuggers, <code>eval</code> and <code>exec</code> see them, and of 27,028 tests of CPython's own suite five end
differently under the compiler, where it was 21.</p>
</div>""")

    A(C.tiles([
        ("Inference: time over torch.compile", f"{E['alone_geo']:.2f}", f"{len(E['names'])} models, each compiler alone; {E['paired_geo']:.2f} with calls in turns"),
        ("Training: time over torch.compile", f"{T['alone_geo']:.2f}", f"{len(T['names'])} models; {T['paired_geo']:.2f} in turns over {len(T['paired'])}"),
        ("Speedup over eager, inference", x(E['speed']['cfg']), f"torch.compile: {x(E['speed']['compile'])}"),
        ("Speedup over eager, training", x(T['speed']['cfg']), f"torch.compile: {x(T['speed']['compile'])}"),
        ("Models ahead / level / behind, inference", f"{len(E['ahead'])} / {len(E['level'])} / {len(E['behind'])}", "ahead and behind: more than 3% either way"),
        ("Compile time, inference", f"{E['compile_geo']['cfg']:.1f} s", f"geometric mean, cold cache; torch.compile: {E['compile_geo']['compile']:.1f} s"),
    ]))

    # ---------------------------------------------------------------- 1
    A("<h2>1. What it is</h2>")
    A("""<p><code>torch.compile</code> traces a function symbolically: Dynamo interprets the bytecode on stand-ins for the values, builds
an FX graph of the tensor operations, and where it meets something it cannot put in a graph (a branch on the data of a tensor, a call it
does not model, a side effect) it ends the graph, runs that piece in the interpreter and starts another graph behind it. Guards on
everything the trace looked at decide whether a compiled piece may be used again.</p>
<p><code>torch.rewrite</code> starts from the other end. The whole function is compiled, once, to a control-flow graph whose
instructions are the Python operations themselves; a branch on a tensor is a branch of that graph, a side effect is an instruction, and
nothing ends anything. What gets specialized is a basic block at a time, from what its first run on real inputs did:</p>""")
    A("""<div class="pipe">
<div><b>Frontend</b>CPython 3.12 bytecode to a control-flow graph in SSA form. Inlining and loop unrolling are transforms of that graph.</div>
<span>&rarr;</span>
<div><b>VM, three tiers</b>A generic tier runs any block. A profiling run records what each tensor operation dispatched to. The specialized tier
calls those kernels directly, below the Python bindings, under guards.</div>
<span>&rarr;</span>
<div><b>Regions</b>With <code>backend="inductor"</code>: a run of instructions, and the path its trace takes through the blocks after it, is
traced on fake tensors to one ATen graph.</div>
<span>&rarr;</span>
<div><b>AOTAutograd + Inductor</b>The same backend as <code>torch.compile</code>: forward and backward compiled; a training region returns
one compiled autograd function.</div>
</div>""")
    A(f"""<p>Two decisions shape everything else. First, <strong>the VM measures</strong>: compiled code is timed against the interpreter
on the user's own calls, and what the interpreter runs faster is handed back to it ({pr(204861)}, {pr(204878)}), which is why compiling a
function rarely makes it slower than it was. Second, <strong>a region is tried from the outside in</strong>: the region around <code>model(x)</code> first,
and only where an instruction does not trace is it split and the VM descends into what that instruction calls. Control flow that depends
on data stays control flow of the VM, with compiled regions on either side.</p>""")
    A("<h3>What is in the stack</h3>")
    A(f"""<p>Bottom {pr(204373)}, top {pr(204763)} (its description carries the full engineering report with per-PR tables). The five PRs at
the bottom are sized for review and run CI; everything above is a draft marked <code>[no-ci]</code>, built and tested at the top of the
stack.</p>""")
    A(groups_table())

    # ---------------------------------------------------------------- 2
    A("<h2>2. The path: seven versions</h2>")
    A(f"""<p>The VM came first and took the first three of the six days: the language, the fidelity to CPython, the measured choice between
VM and interpreter. The comparison with <code>torch.compile</code> starts with the graph backend, which went through four versions, and a fifth that left speed alone. Each was
benchmarked on the same {len(E['names'])} models; the numbers of the earlier versions are from their own runs on a shared machine, so read
the trend, not the second decimal.</p>""")
    A(fig_path())
    A(f"""<table>
<tr><th>version</th><th>what it added</th><th>where it stood</th></tr>
<tr><td>v1<br><span class="muted">Nov 14</span></td><td>Regions: per basic block, a run of effect-free instructions traced with what it calls and compiled
({prs(204877, 204883)}).</td><td>Level where a model is one graph, behind where Python between tensor code cut a model into pieces: doctr_det_predictor took
1.27 of <code>torch.compile</code>'s time as three graphs, llama was 33 graphs.</td></tr>
<tr><td>v2<br><span class="muted">Nov 14</span></td><td>A region follows the path its trace takes across blocks: loops over layers, branches on flags
({prs(204932, 204948)}).</td><td>0.98 of <code>torch.compile</code>'s time with calls in turns. Behind on generation loops (hf_T5_generate 1.15,
cm3leon_generate 1.08), which were compiled once per sequence length.</td></tr>
<tr><td>v3<br><span class="muted">Nov 15</span></td><td>Sizes and integers that differ from call to call are left open in one graph; what a region does
to the caches and objects it is handed is replayed ({prs(204981, 204991)}).</td><td>Parity: hf_T5_generate 1.00, cm3leon_generate 1.02, llama 1.00, with
fewer graphs than <code>torch.compile</code> (6 against 10 for T5). At this point both compilers ran the same kernels on the same graphs.</td></tr>
<tr><td>v4<br><span class="muted">Nov 16</span></td><td>Memory kept per function, weights built in behind a check, one-row products
({prs(205043, 205052)}, {pr(205088)} to {pr(205103)}).</td><td>{E4['alone_geo']:.2f} for inference and {T4['alone_geo']:.2f} for training, and
memory to match: in training {D.geo(r for r, *_ in h4):.2f} times what <code>torch.compile</code> holds over the {len(h4)} models where it was
recorded, hf_Reformer at 4.3 times.</td></tr>
<tr><td>v5<br><span class="muted">Nov 17</span></td><td>The variables of a function in its frame, so that <code>eval</code>, <code>exec</code>,
tracebacks and debuggers find them; calls under a trace function left to the interpreter; a recording that kept tensors alive; a workspace
that cuts memory to size ({prs(205188, 205195)}).</td><td>{E['alone_geo']:.2f} for inference and {T['alone_geo']:.2f} for training; memory
{held_geo:.2f} in training and {held_geo_eval:.2f} in inference in that version.</td></tr>
<tr><td>v6<br><span class="muted">Nov 17</span></td><td>The name <code>torch.rewrite</code> and one list of passes; the functions that C
code calls rewritten through a frame hook; passes of the user's in Python and in Rust; CPython 3.14; the first runs on a GPU and three
fixes they led to, and one wrong result found on the way ({prs(205237, 205246)}, {prs(205258, 205263)}, {pr(205272)}).</td><td>On the
CPU as the fifth version, whose numbers the CPU sections of this page show: the sixth changed what runs around the graphs, not the graphs.
On the GPU {gpu_summary("eval")["geo"]:.2f} for inference and {gpu_summary("train")["geo"]:.2f} for training, in the stack's
harness on a build without cuDNN.</td></tr>
<tr><td>v7<br><span class="muted">Nov 18</span></td><td>The GPU as the perf dashboard measures it. A size that only data decides stays in
the graph; a call that is like the one before is told so in the runtime, with the dicts of the modules watched; the dashboard's runner
measures <code>torch.rewrite</code>; and what its suites found: which weights are built in, and what a call through a method of an
object costs; the update of an optimizer as a region; the chunks of the interpreter's frame stack kept ({prs(205326, 205335)},
{pr(205357)}, {pr(205397)}, {pr(205405)}, {pr(205409)}, {pr(205410)}, {pr(205436)}, {pr(205437)}, {pr(205502)}).</td><td>On the GPU, on the dashboard's suites with cuDNN:
{suite_summary("inductor")["all"]["geo"]:.2f} of the time of <code>torch.compile</code> for inference,
{suite_summary("freezing")["all"]["geo"]:.2f} of <code>torch.compile</code> with freezing, and
{suite_summary("inductor", D.SUITES_TRAIN)["all"]["geo"]:.2f} for training. The CPU was not measured again.</td></tr>
</table>""")
    A(f"""<p>The fifth version was about fidelity and memory, not speed, and the ratio between the two compilers did not move
({E4['alone_geo']:.2f} and {E['alone_geo']:.2f} for inference). Inference was measured again in all three modes for it, and eager ran
about 5% faster in that run than in the one of v4, which is why both lines of the left chart end lower than they were.</p>""")
    A("""<p>The step from v3 to v4 is the answer to a direct question: with the two compilers at parity, is there anything the architecture
can do that Dynamo does not? Profiles said the remaining time was in library kernels (82% of resnet50 in <code>convolution</code>, 85% of
hf_Bert in <code>addmm</code>), so the search went to what surrounds the kernels.</p>""")

    # ---------------------------------------------------------------- 3
    A("<h2>3. Results on TorchBench</h2>")
    A(f"""<p><strong>Setup.</strong> <a href="https://github.com/pytorch/benchmark">TorchBench</a>, run the way
<code>benchmarks/dynamo/torchbench.py</code> runs it: the module and example inputs of each model; inference under <code>no_grad</code> in
eval mode; a training step of <code>zero_grad</code>, forward, <code>reduce_to_scalar_loss</code>, <code>backward</code> and an SGD step.
The same function goes to <code>torch.compile(fn)</code> and to <code>torch.rewrite(fn, backend="inductor")</code>, with default
options for both. CPU only (AMD EPYC 7R13), 4 threads per run, pinned to 4 physical cores; every (model, test, compiler) in a process of
its own with an empty Inductor cache, so compile times are cold. PyTorch main of Nov 9, 2026 plus the stack, Python 3.12.14.</p>
<p>Of TorchBench's 104 models, {E['eager_ran']} load and run in eager mode here for inference and {T['eager_ran']} for training; the others
need datasets, TorchScript or CUDA and are left out for both compilers. Both compilers ran all {len(E['names'])} for inference and
{len(T['names'])} of the {T['eager_ran']} for training (opacus_cifar10 fails for both: the harness compares against a deep copy of the
model, and Opacus' hooks do not survive the copy).</p>""")
    A(fig_models("eval"))
    A(fig_models("train"))
    A(f"""<p><strong>Alone or in turns.</strong> The harness measures twice: each compiler in its own process, and, inside the
<code>torch.rewrite</code> process, the two compiled functions called in turns. While both ran the same code, in turns was the number to
trust, because two processes on a shared machine differed by more than the compilers did. With v4 that reversed: <code>torch.compile</code>
is slower in a process that <code>torch.rewrite</code> has allocated in than in a process of its own wherever page faults matter
(mobilenet_v3_large: 157 ms there, 109 ms alone), which flatters <code>torch.rewrite</code>. So the final run was made on an idle machine
and this page reads the alone numbers; both are in the table views, and the geometric means of the two are close
({E['alone_geo']:.2f} and {E['paired_geo']:.2f} for inference, {T['alone_geo']:.2f} and {T['paired_geo']:.2f} for training).</p>""")

    # ---------------------------------------------------------------- 4
    A("<h2>4. Where the margin comes from</h2>")
    A("""<p>Three changes, each of which the control-flow-graph design made a natural default rather than a setting.</p>""")
    A(f"""<h3>Memory that belongs to the compiled function</h3>
<p>A model allocates the same tensors on every call. glibc gives the large ones back to the system as soon as they are freed, so the next
call maps them again and the kernel faults every page in: 2.67 microseconds per fault on this (virtual) machine. For a convolutional
model that is more time than is left to win in its kernels. The code <code>torch.compile</code> itself makes for resnet50 runs in 734 ms
instead of 937 ms when the allocator under it keeps the memory; mobilenet_v2 in 54 ms instead of 102 ms.</p>
<p>A graph compiler has nowhere to keep that memory. A compiled function does: it is what the user calls again and again with inputs of
the same sizes. Every compiled function now has a <em>workspace</em> ({pr(205043)}, {pr(205044)}): while it runs, the CPU tensors made on its
thread, by a compiled region, by a kernel the VM calls or by the backward, take their memory from it and give it back to it, and the next
call finds blocks of the sizes it needs. No graph backend is needed to see the effect:</p>
<pre><code>conv1, conv2 = torch.nn.Conv2d(3, 64, 3, padding=1), torch.nn.Conv2d(64, 64, 3, padding=1)
def f(x):
    return conv2(torch.relu(conv1(x))).mean()
g = torch.rewrite(f)                 # no backend: the eager kernels
x = torch.randn(32, 3, 128, 128)
# per call, 4 threads:   eager 323 ms    g 138 ms    g with keep_memory=False 302 ms</code></pre>
<p>This is where most of the margin is: {names('eval', ['densenet121', 'phlippe_densenet', 'mobilenet_v2', 'mnasnet1_0', 'squeezenet1_1', 'basic_gnn_gin', 'basic_gnn_sage', 'basic_gnn_gcn', 'pyhpc_turbulent_kinetic_energy'])}
of <code>torch.compile</code>'s time, and all of the margin in training, where weights cannot be built in.</p>""")
    A(f"""<h3>Weights built into the graph, behind a check</h3>
<p>With the parameters as inputs of a graph, Inductor cannot fold a batch norm into the convolution before it or lay convolution weights
out the way the kernels want them. <code>torch.compile</code> has a switch for that, <code>torch._inductor.config.freezing</code>, off by
default for a reason: nothing checks that the weights stay what they were, so it is a promise the user makes.</p>
<p>A compiler that already keeps variants of a region behind checks can make it the default. In a call of inference (no gradients
recorded, every module in eval mode) the parameters and buffers a region's graph uses are compiled into it ({prs(205046, 205051)}). Every
call then checks a stamp over the object, the memory address and the version counter of each such tensor: 4 microseconds for the 320
tensors of resnet50. The version counter alone is not enough, since <code>p.data.mul_(2)</code> changes the memory without advancing it,
so each built-in tensor also gets a lazy clone that is never read: its storage becomes copy-on-write, and PyTorch moves the tensor to
other memory before anything writes to it, which the stamp sees. A tensor that changed becomes an input of the region's next graph; the
rest stay built in.</p>
<p>resnet50 in one process, each compiler with its own copy of the model: <code>torch.compile</code> 1013 ms, <code>torch.compile</code>
with <code>freezing</code> 799 ms, <code>torch.rewrite</code> 675 ms. Matrices that a graph only multiplies with or looks rows up in are left
as inputs ({pr(205050)}): nothing folds, and a copy of them cost hf_Bert 400 MB.</p>""")
    A(f"""<h3>One-row matrix products</h3>
<p>A decoder that makes one token at a time multiplies a single row of activations with every weight matrix of the model. MKL runs
<code>mm</code> of a 1&times;K row with a K&times;N matrix on one core however many threads it has, and since the matrices do not fit
the cache, that core waits for memory. Rewritten as <code>(matrix * row.T).sum(0)</code> before the graph goes to Inductor ({pr(205045)}),
the product is a loop Inductor splits over the cores. A 768-to-32,128 output layer on one row: eager 7.3 ms, <code>torch.compile</code>
6.8 ms, <code>torch.rewrite</code> 2.5 ms. hf_T5_generate takes {D.ms(EV['hf_T5_generate']['cfg']['call_s'])} against
{D.ms(EV['hf_T5_generate']['compile']['call_s'])}. This one is a graph rewrite that any front end could do.</p>""")
    A(f"""<h3>What torch.compile does when asked</h3>
<p>Two of the three are things <code>torch.compile</code> can be given by hand: glibc can be told to keep what is freed
(<code>MALLOC_TRIM_THRESHOLD_</code>, <code>MALLOC_MMAP_THRESHOLD_</code>, <code>MALLOC_TOP_PAD_</code>), and <code>freezing</code> can
be switched on. In a process set up that way, with the calls of the compilers in turns:</p>""")
    A(tuned_table())
    A("""<p>So the margin of section 3 is over <code>torch.compile</code> as it comes, not over what Inductor can do. With both set by hand
it runs the same kernels on the same memory. What is left then: blocks above 32 MB, which glibc unmaps whatever it is told (squeezenet1_1;
another malloc was not tried), models that are fewer graphs here (hf_BigBird: 1 graph against 28), and one-row products. What the design
adds is that neither is a setting: the memory belongs to a function and goes when the function does, and the weights sit behind a check
where <code>freezing</code> is a promise.</p>""")

    # ---------------------------------------------------------------- 5
    A("<h2>5. Accuracy and fidelity</h2>")
    oc, og = T["over"]["compile"], T["over"]["cfg"]
    A(f"""<p><strong>Model outputs.</strong> For every run the harness compares the compiled outputs with eager's on the same inputs (the
worst relative difference over all outputs; for training, over the losses of the first eight steps).</p>
<table>
<tr><th></th><th>torch.compile</th><th>torch.rewrite</th></tr>
<tr><td>Inference: within 1e-2 of eager</td><td>{E['within']['compile']} of {len(E['names'])}</td><td>{E['within']['cfg']} of {len(E['names'])}</td></tr>
<tr><td>Inference: within 1e-5</td><td>{E['within5']['compile']}</td><td>{E['within5']['cfg']}</td></tr>
<tr><td>Inference: largest difference</td><td>{E['worst']['compile'][0]:.1e} ({E['worst']['compile'][1]})</td><td>{E['worst']['cfg'][0]:.1e} ({E['worst']['cfg'][1]})</td></tr>
<tr><td>Training: within 1e-2 of eager</td><td>{T['within']['compile']} of {len(T['names'])}</td><td>{T['within']['cfg']} of {len(T['names'])}</td></tr>
<tr><td>Training: within 1e-5</td><td>{T['within5']['compile']}</td><td>{T['within5']['cfg']}</td></tr>
<tr><td>Training: bit-identical losses</td><td>{T['exact']['compile']}</td><td>{T['exact']['cfg']}</td></tr>
<tr><td>Training: over 1e-2</td><td>{', '.join(f'{n} {v:.1e}' for v, n in oc)}</td><td>{', '.join(f'{n} {v:.1e}' for v, n in og)}</td></tr>
</table>
<p>Inference agrees to float rounding for both. In training <code>torch.rewrite</code> is the closer of the two to eager, because the
optimizer and what the measurement hands back run eager's own code. The four training runs over 1e-2 are all under 2e-2: differences of the size
that a differently ordered floating-point operation grows into over eight optimizer steps. Their cause was not looked into further. An earlier
sweep with <code>backend="aot_eager"</code>, which builds the same graphs and runs them with the eager kernels, was bit-exact on 99 of 102
configurations, and the other three fail in eager mode too.</p>""")
    A(f"""<p><strong>Python semantics.</strong> A compiler that replaces the interpreter has to behave like it. The stack's runner
(<code>benchmarks/rewrite/cpython_suite.py</code>) compiles every test method of a module of CPython 3.12's own <code>Lib/test</code>,
lets the VM compile whatever those methods call, and compares the outcome of each test with its outcome under the plain interpreter.</p>""")
    A(C.tiles([
        ("CPython test modules run through the VM", "183", "27,028 tests"),
        ("Test methods compiled", "17,121", "25 left to the interpreter: generators, <code>except*</code>, functions that enter a debugger"),
        ("Outcomes that differ from the interpreter", "5", "it was 21 two versions ago, and 81 with a segmentation fault at the start"),
        ("Tests of the stack itself", "397", "under CPython 3.12; 373 on a CPU build under 3.14 before the last eight PRs; those of the last five rounds each checked to fail without their change"),
    ]))
    A(f"""<p>These are from a run on the top of the stack, with the frame hook on: what C code calls under a test method (comparisons, hashes,
<code>__init__</code>, properties, callbacks) runs in the VM too. None of the five is in the result of a computation: one counts objects
around the first call of a function that C code calls, whose rewritten form is made in between; two are a disassembly that shows
which loops the interpreter's own specializer saw, one is a code object corrupted by hand, one a garbage collection that comes one object
earlier. Getting from 81 to 21 was a round of its own: real frames for compiled and inlined calls so tracebacks have columns, CPython's exact
error messages, locals released when the interpreter releases them, signals and thread switches inside compiled loops, a recursion limit
that means the same depth. Getting from 21 to 4 was the fifth version. Seventeen of the 21 had one of two causes: the variables of a
compiled function lived in the VM's registers and its frame was empty, and a trace function was not told about what compiled code called.
Now every assignment also binds the variable in the frame ({pr(205189)}), which is what let functions that call <code>eval</code>,
<code>exec</code> or <code>super()</code> be compiled at all ({pr(205190)}: 231 of the 246 test methods that had been turned down), and
while <code>sys.settrace</code> is set, calls run in the interpreter ({pr(205191)}). On tensor code the frame costs nothing that can be
measured, because a tensor that only kernels touch is bound to its variable when something can look, not before; on plain Python loops that
the VM is forced to run it costs 6 to 12%.</p>
<p><strong>What "matches" does not cover.</strong> A region checks its inputs (tensors, modules and their structure, values, modes) and
what its own path read from modules and globals. It does not check what a callee that the trace ran in the interpreter read from an
object that is not an input. Dynamo guards those reads; this does not yet.</p>""")

    # ---------------------------------------------------------------- 6
    A("<h2>6. Compile time and graphs</h2>")
    A(f"""<table>
<tr><th></th><th>torch.compile</th><th>torch.rewrite</th></tr>
<tr><td>Inference: compile time, geometric mean</td><td>{E['compile_geo']['compile']:.1f} s</td><td>{E['compile_geo']['cfg']:.1f} s</td></tr>
<tr><td>Inference: compile time, sum over {len(E['names'])} models</td><td>{E['compile_sum']['compile']:.0f} s</td><td>{E['compile_sum']['cfg']:.0f} s</td></tr>
<tr><td>Inference: time to steady state, geometric mean</td><td>{E['warm_geo']['compile']:.1f} s</td><td>{E['warm_geo']['cfg']:.1f} s</td></tr>
<tr><td>Inference: models that are one graph</td><td>{E['one_graph']['compile']}</td><td>{E['one_graph']['cfg']}</td></tr>
<tr><td>Training: compile time, geometric mean</td><td>{T['compile_geo']['compile']:.1f} s</td><td>{T['compile_geo']['cfg']:.1f} s</td></tr>
<tr><td>Training: compile time, sum over {len(T['names'])} models</td><td>{T['compile_sum']['compile']:.0f} s</td><td>{T['compile_sum']['cfg']:.0f} s</td></tr>
<tr><td>Training: time to steady state, geometric mean</td><td>{T['warm_geo']['compile']:.1f} s</td><td>{T['warm_geo']['cfg']:.1f} s</td></tr>
<tr><td>Training: models that are one graph</td><td>{T['one_graph']['compile']}</td><td>{T['one_graph']['cfg']}</td></tr>
</table>
<p>"Compile time" is what each compiler reports for itself: Dynamo's <code>compile_inner</code>, and tracing plus backend per region.
The backend is the same and is most of it. What differs is the front: a region is traced by running its instructions once on fake
tensors, which costs one eager call on fake tensors, and its guards are a description of its inputs rather than a tree built while
tracing. "Time to steady state" is what the calls before the timed ones cost over the same number of steady calls; for
<code>torch.rewrite</code> that includes the calls it spends measuring compiled code against the interpreter, which is why it gives back
in training what it saves on compiling.</p>
<p>Graph counts differ most where control flow or side effects sit between tensor code. hf_BigBird is {EV['hf_BigBird']['compile']['graphs']}
graphs for <code>torch.compile</code> and {EV['hf_BigBird']['cfg']['graphs']} here; in training, mobilenet_v2_quantized_qat is
{TR['mobilenet_v2_quantized_qat']['compile']['graphs']} against {TR['mobilenet_v2_quantized_qat']['cfg']['graphs']}. It goes the other way
too: hf_Bart trains as {TR['hf_Bart']['cfg']['graphs']} graphs here against {TR['hf_Bart']['compile']['graphs']}, and demucs as
{TR['demucs']['cfg']['graphs']} against {TR['demucs']['compile']['graphs']}, where an instruction inside the model does not trace and
the pieces around it become regions of their own.</p>""")

    # ---------------------------------------------------------------- 7
    A("<h2>7. What it costs: memory</h2>")
    A(f"""<p>The workspace holds what a call freed for the next call, and the process has that memory whether or not a call is running. In
the last version that was the one cost that should not have shipped: whole blocks were kept, up to one and a half times the peak of a
call on top of what was in use, and single models held four times what <code>torch.compile</code> holds. Two changes went in
({pr(205194)}, {pr(205195)}), and this is where it stands:</p>""")
    A(mem_fig)
    A(mem_fig_eval)
    A(f"""<p>Training, {len(h)} models: geometric mean {held_geo:.2f}; the most are
{', '.join(f'{n} {gb(a)} against {gb(b)}' for r, n, a, b, e in worst)}. Inference, {len(he)} models: {held_geo_eval:.2f}; the most are
{', '.join(f'{n} {gb(a)} against {gb(b)}' for r, n, a, b, e in worst_eval)}.</p>
<p><strong>What the two changes were.</strong> The largest number of the last version, hf_Reformer at 18 GB against 4 GB, was not the
workspace at all. The first times the VM runs a call of a C function it records the kernel calls underneath, to see whether the call is one
kernel that it can call directly from then on, and the record kept the arguments and results of all of them alive until the call returned.
<code>torch.autograd.Function.apply</code> is such a C function, and in hf_Reformer what runs under it is the whole reversible encoder: the
first training step held every tensor of every layer at once, 18 GB, and every later step 4. A record now lets go of its tensors as soon
as it sees a second kernel call. The other change is the workspace itself: instead of keeping whole blocks and handing one out when it is
between the size asked for and twice that size, it cuts what is asked for out of the memory it holds, closest fit first, joins what comes
back with the free memory next to it, and holds at most the peak of its tensors plus half of that, 256 MB at most. Replaying the requests
of eleven models under both policies, the most that is held over the most the tensors ask for at once went from 1.55 to 1.24, and the
share of a call's bytes that has to come from the system, which is what costs page faults, from 4.9% to 1.1%: less memory and no less
speed.</p>
<p><strong>What is left.</strong> A function still holds about the peak of its tensors between calls, where <code>torch.compile</code>
gives everything back: that is what keeping memory means, and <code>keep_memory=False</code> is there for a process that cannot afford
it. And the first calls of a function run before its graphs exist, one kernel at a time, and can hold more at once than the calls after
them: a training step of vgg16 asks for 4.95 GB at once in its first two calls and for 3.02 GB from the third on, and the memory of the
first calls stays in use because the later ones are cut out of it. A limit that follows the last few calls was tried in four variants and
did not get vgg16 below 5.9 GB (6.3 GB as it is; 4.7 GB without a workspace), so it is not in the stack.</p>
<p><strong>Not better everywhere.</strong> Of the {len(change)} training models that were measured the same way in the last version,
{len(fewer)} hold less now ({', '.join(f'{n} {v:.2f} of what it held' for v, n in fewer[:3])}) and {len(more)} hold more
({', '.join(f'{n} {v:.2f}' for v, n in more[:4])}). Keeping whole blocks suited a model whose sizes repeat exactly from call to call;
cutting to size suits the others, and costs those a few percent.</p>""")

    # ---------------------------------------------------------------- 8
    A("<h2>8. The VM without a graph backend</h2>")
    A("""<p>Without <code>backend=</code> the VM runs the function itself and calls ATen kernels directly. What that buys is the overhead
per operation, so it shows on small tensors and disappears on large ones.</p>""")
    A(fig_micro())
    A("""<table>
<tr><th>workload</th><th>eager</th><th>torch.rewrite</th><th>time over eager</th></tr>
<tr><td>nn.Module forward, 2 Linear layers</td><td>21.4 &micro;s</td><td>14.7 &micro;s</td><td>0.68</td></tr>
<tr><td>nn.Module forward, tracking gradients</td><td>31.0 &micro;s</td><td>23.3 &micro;s</td><td>0.75</td></tr>
<tr><td>training step: forward, backward, SGD</td><td>329 &micro;s</td><td>336 &micro;s</td><td>1.02</td></tr>
<tr><td>plain Python loop, per iteration</td><td>21.0 ns</td><td>21.0 ns</td><td>1.00 (handed back to the interpreter)</td></tr>
<tr><td>the same loop, VM forced to run it</td><td>21.0 ns</td><td>66.6 ns</td><td>3.17</td></tr>
</table>
<p>These were measured before the graph backend existed, and the rounds since do not touch this path. The last two rows are the
honest picture of the VM on plain Python: its generic tier is an interpreter without CPython 3.12's inline
caches and specialized instructions, three to four times slower, and the reason a program does not see that is the measurement that
hands such code back. On CUDA the tensor workloads ran 1.00x to 1.30x eager when they were last measured, early in the work; nothing
since has been measured on a GPU.</p>""")

    # ---------------------------------------------------------------- 9
    lj = EV["lennard_jones"]
    # ---------------------------------------------------------------- 9
    GE, GT = gpu_summary("eval"), gpu_summary("train")
    DS, FS, TS = suite_summary("inductor"), suite_summary("freezing"), suite_summary("inductor", D.SUITES_TRAIN)
    A("<h2>9. On a GPU</h2>")
    A("""<p class="note"><strong>What changed in this section on Nov 18.</strong> It used to report TorchBench at its default batch sizes,
measured by the stack's own harness on a build of PyTorch that had been configured without cuDNN: convolutions ran PyTorch's own kernels,
20 to 30 times slower than cuDNN's. Eager, <code>torch.compile</code> and <code>torch.rewrite</code> ran on that same build, so the ratios
compared like with like, but for convolutional models they were not of the workload people run. The section now reports the three suites
of the PyTorch perf dashboard, run by the dashboard's runner at its settings, on a build with cuDNN 9.26. What the harness measured is
kept below where it explains a change, and says so.</p>""")
    A(f"""<p>Everything above is a CPU. On one NVIDIA H100 (CUDA 12.8, cuDNN 9.26, Triton 3.8.0) the seventh version was measured the way the
perf dashboard measures <code>torch.compile</code>: <code>benchmarks/dynamo</code> runs HuggingFace, timm and TorchBench at batch sizes of
8 to 1024, in bfloat16 for inference and under AMP for training, and times compiled calls against eager ones in turns in one process.
{pr(205357)} lets that runner measure <code>torch.rewrite</code> (<code>--rewrite</code>) where <code>--backend inductor</code> has it
measure <code>torch.compile</code>. For inference <code>torch.compile</code> was run as the dashboard runs it and once more with Inductor's
<code>freezing</code>, which is the comparison to make: <code>torch.rewrite</code> builds weights into its graphs, as freezing does. The
HuggingFace models that go through <code>generate</code> are left out (<code>--rewrite</code> has no path for them), and the TorchBench
models that fail under both compilers on this machine.</p>
<p>The machine is shared. Every model was measured under the compilers one right after the other, in an order that takes turns, on
the build of {pr(205502)}; earlier runs had probes of this report's own on the GPU for part of the time and are not used. Between two
such runs of the training suites, five hours apart, the ratio of a model moved by a percent or two at the median and by up to a tenth,
and by more for models whose step takes a few milliseconds: a single model says less than a mean.</p>""")
    tiles = [
        ("GPU inference: time over torch.compile", f"{DS['all']['geo']:.2f}", f"{DS['all']['n']} models of three suites; {DS['all']['ahead']} ahead, {DS['all']['behind']} behind"),
        ("GPU inference: time over torch.compile with freezing", f"{FS['all']['geo']:.2f}", f"{FS['all']['n']} models; {FS['all']['ahead']} ahead, {FS['all']['behind']} behind"),
        ("Speedup over eager, GPU inference", x(DS['all']['rewrite']), f"torch.compile: {x(DS['all']['base'])}; with freezing: {x(FS['all']['base'])}"),
        ("Peak GPU memory over torch.compile", f"{DS['all']['held']:.2f}", f"while a model is warmed up and compiled; {FS['all']['held']:.2f} over freezing"),
    ]
    if "all" in TS:
        tiles.insert(2, ("GPU training: time over torch.compile", f"{TS['all']['geo']:.2f}", f"{TS['all']['n']} models; {TS['all']['ahead']} ahead, {TS['all']['behind']} behind"))
    A(C.tiles(tiles))
    rows = []
    for title, suite in SUITE_NAMES:
        d, f = DS[suite], FS[suite]
        rows.append([title, str(d["n"]), f"{x(d['base'])} / {x(f['base'])} / {x(d['rewrite'])}", f"{d['geo']:.2f} ({len(d['ahead'])} ahead, {len(d['behind'])} behind)",
                     f"{f['geo']:.2f} ({len(f['ahead'])} ahead, {len(f['behind'])} behind)", f"{d['held']:.2f} / {f['held']:.2f}",
                     f"{d['compile'][0]:.0f} s / {f['compile'][0]:.0f} s / {d['compile'][1]:.0f} s"])
    A(C.table(["inference", "models", "speedup over eager: torch.compile / with freezing / torch.rewrite", "time of torch.rewrite over torch.compile",
               "over torch.compile with freezing", "peak GPU memory over torch.compile / over freezing", "compile time: torch.compile / with freezing / torch.rewrite"], rows))
    if "all" in TS:
        rows = []
        for title, suite in SUITE_NAMES:
            if suite in TS:
                t = TS[suite]
                rows.append([title, str(t["n"]), f"{x(t['base'])} / {x(t['rewrite'])}", f"{t['geo']:.2f} ({len(t['ahead'])} ahead, {len(t['behind'])} behind)", f"{t['held']:.2f}"])
        A(C.table(["training", "models", "speedup over eager: torch.compile / torch.rewrite", "time of torch.rewrite over torch.compile",
                   "peak GPU memory over torch.compile"], rows))
    else:
        A("<p>The training suites (<code>--training --amp</code>) were running when this page was made.</p>")
    for title, suite in SUITE_NAMES:
        A(fig_suite(suite, title))
    if "all" in TS:
        for title, suite in SUITE_NAMES:
            if suite in TS:
                A(fig_suite(suite, title, training=True))
    hf, timm, tb = (FS[s] for _, s in SUITE_NAMES)
    A(f"""<p><strong>Reading it.</strong> With weights as constants of the graph, Inductor folds batch norms into convolutions and makes
one product of the three projections of an attention layer, for <code>torch.rewrite</code> as for <code>torch.compile</code> with freezing:
the two then run nearly the same kernels, and timm, which is convolutions at large batches, is level ({timm['geo']:.2f}). What margin there
is comes from what runs around the kernels and from how many graphs a call is.</p>
<ul>
<li><strong>Ahead of torch.compile with freezing</strong>: {listed(tb['ahead'], 4)} on TorchBench; {listed(hf['ahead'], 4)} on HuggingFace.
basic_gnn_gcn is one graph where Dynamo makes six, because a size that only data decides stays in the graph ({pr(205326)}); the generation
loop of cm3leon_generate stays instructions of the VM, where Dynamo ends a graph at each step that depends on data.</li>
<li><strong>Against torch.compile as the dashboard runs it</strong>, without freezing: {DS['all']['geo']:.2f} of its time over {DS['all']['n']} models;
ahead on {DS['all']['ahead']}, behind on {DS['all']['behind']} ({listed(DS['torchbench']['behind'], 3)}).</li>
<li><strong>Behind</strong>: {listed(tb['behind'], 4)} on TorchBench, {listed(hf['behind'], 2)} on HuggingFace, {listed(timm['behind'], 2)} on timm.
opacus_cifar10 puts a hook on every layer; a region refuses a module with hooks, so the model runs at the speed of eager, where
<code>torch.compile</code> compiles through them. The pyhpc models are calls of under a millisecond that the runner times right after an
eager call of the same model, which costs <code>torch.rewrite</code> more than <code>torch.compile</code>: 0.099 ms against 0.084 ms for
pyhpc_equation_of_state, and 0.069 ms for both when the calls are back to back. AllenaiLongformerBase is four graphs, and its encoder's graph leaves a size open that only data decides, where Dynamo ends its
graph there and compiles what follows for the size it saw; that this is what costs the 9% was not measured.</li>
<li><strong>Memory is the price</strong>: the runner records the most that was allocated on the GPU while a model is warmed up and
compiled, and <code>torch.rewrite</code> has {DS['huggingface']['held']:.2f} times what <code>torch.compile</code> has there on HuggingFace,
{DS['timm_models']['held']:.2f} on timm and {DS['torchbench']['held']:.2f} on TorchBench (where the smallest models hold tens of megabytes and the ratio moves with them: it was 1.33 a run earlier). The first calls of a rewritten function run the
model kernel by kernel on real tensors, so its peak is never under eager's, where <code>torch.compile</code> never runs the model that way;
and a weight that is built in is held a second time (XGLM: 6.0 GB, where it was 4.9 GB before the matrices of its products were built in).</li>
<li><strong>Training</strong>: {TS['all']['geo']:.2f} of the time of <code>torch.compile</code> over {TS['all']['n']} models
({TS['huggingface']['geo']:.2f} on HuggingFace, {TS['timm_models']['geo']:.2f} on timm, {TS['torchbench']['geo']:.2f} on TorchBench); ahead on
{TS['all']['ahead']}, behind on {TS['all']['behind']} ({listed(TS['torchbench']['behind'], 3)} on TorchBench, {listed(TS['huggingface']['behind'], 3)} on
HuggingFace, {listed(TS['timm_models']['behind'], 3)} on timm). The forward and the loss are one graph under both, and the backward is AOTAutograd's.
What <code>torch.rewrite</code> did not have when the suites were first run was the step of the optimizer as a graph:
<code>torch.compile</code> compiles it, and the regions pass did not look at <code>torch/optim/</code>. With the update a region
({pr(205437)}), a step of deit_tiny takes 12.4 ms, as under <code>torch.compile</code> in the same process, where it took 14.3 ms.
Getting there costs more: a training step is called some hundred times while its levels of calls are measured one after another,
{TS['all']['compile'][1]:.0f} s of warm-up where <code>torch.compile</code> compiles for {TS['all']['compile'][0]:.0f} s (geometric means).</li>
<li><strong>An accident of the interpreter had two models a quarter behind.</strong> CPython keeps the frames of a thread in chunks of
16 KB and unmaps a chunk the moment the frame at its start returns. The code Inductor generates for a graph has one large frame and
launches the graph's kernels from it. Where the runner calls tf_efficientnet_b0 for training under <code>torch.rewrite</code>, that
frame ended 112 bytes short of the end of a chunk, and each of the 452 kernel launches of a step mapped a chunk and unmapped it again:
30.5 ms a step, and 23.3 ms with one more Python frame anywhere above. The runtime now keeps the chunks that are freed
({pr(205502)}), and the numbers here are of that build. It is not <code>torch.rewrite</code>'s accident alone. The timed loops ask
for these many chunks a step: under <code>torch.rewrite</code>, which now has them at hand, {D.chunks("suites_r25e_train", ("rewrite",))}
in training and {D.chunks("suites_r25e", ("rewrite",))} in inference; under <code>torch.compile</code>, which maps them as it would
on the dashboard, {D.chunks("suites_r25e_train", ("inductor",))} in training and {D.chunks("suites_r25e", ("inductor", "freezing"))}
in inference. With those four run a second time and the chunks kept for <code>torch.compile</code> as well, its step of
ElectraForCausalLM is 12% shorter (<code>torch.rewrite</code> is then at 0.96 of its time, not 0.89) and those of the other three 0% to 5%; training as a whole is then at
{D.kept(D.SUITES_TRAIN, "inductor")[0]:.3f} of the time of <code>torch.compile</code>, where it is at {TS['all']['geo']:.3f} with the first runs.</li>
<li><strong>Compile time</strong>: {DS['huggingface']['compile'][1]:.0f} s against {DS['huggingface']['compile'][0]:.0f} s on HuggingFace,
{DS['timm_models']['compile'][1]:.0f} s against {DS['timm_models']['compile'][0]:.0f} s on timm, {DS['torchbench']['compile'][1]:.0f} s
against {DS['torchbench']['compile'][0]:.0f} s on TorchBench (geometric means, as the runner reports them).</li>
</ul>
<p><strong>What the dashboard's suites found.</strong> The first run of the suites had <code>torch.rewrite</code> behind
<code>torch.compile</code> with freezing on HuggingFace and timm, behind on every model whose call is short, and behind in training.
Five causes, each in <code>torch.rewrite</code> and none in Inductor:</p>
<ol>
<li><strong>A method of an object as what is called.</strong> The runner calls a model through <code>runner.forward_pass</code>, a method
of an object that holds its parsed command line, 127 attributes. The pass that tells a call that is like the one before in the runtime
did not know a bound method or an object with attributes, so every call described that object in Python: 52 of the 96 microseconds of a
call of pyhpc_equation_of_state ({pr(205405)}).</li>
<li><strong>A loss module in training.</strong> Weights were built in when no module of the call was in training, and the timm runner
holds a <code>CrossEntropyLoss</code>, which is in training like any module nobody called <code>eval()</code> on: no timm model had its
weights built in. A module in training now keeps only its own weights as inputs ({pr(205409)}): tf_efficientnet_b0 went from 5.3 ms to
4.0 ms a call.</li>
<li><strong>The matrix of a product.</strong> A matrix that a graph only multiplies with was left as an input, to not hold it twice.
On a GPU, Inductor fuses the products of one input only if their matrices are constants; they are built in there now ({pr(205410)}), at
the price in memory above.</li>
<li><strong>What stands in during a trace.</strong> maml failed: a function that stands in for <code>torch.autograd.grad</code> while
a region is traced was handed back by a region and called ({pr(205397)}).</li>
<li><strong>The step of an optimizer.</strong> The regions pass left <code>torch/optim/</code> alone, as code that moves tensors about.
The update of Adam or SGD is operations on lists of tensors, which a region holds as it is; it is a region now ({pr(205437)}). A
learning rate is a float in the key of that region, so one on a schedule would have been a graph per step: a region whose first calls
each come with other numbers gets no graph ({pr(205436)}). Under Adam the weights are no longer bit for bit the interpreter's, as they
are not under <code>torch.compile</code>: the corrections for the first steps are float32 in the graph.</li>
</ol>""")
    A("""<p><strong>What the stack does to <code>torch.compile</code>.</strong> The first PR of the stack adds a check to the dispatcher,
which every operator call of every program passes. The perf dashboard's own workflow (the three suites, inference in bfloat16 and training
under AMP, thirteen configurations, on H100 runners) was run on the commit the stack starts from, twice, and on the top of the stack. Its
build has no Rust runtime, so it measures <code>torch.compile</code>, not <code>torch.rewrite</code>. Over 681 inference and 423 training
model runs the speedup over eager is within 0.3% of the starting commit's in every comparison. Compiled latency on the stack is 0.6% and
0.9% above the starting commit's first run and 0.5% and 0.1% above its second, and the second run is itself 0.2% and 0.8% above the
first: within what two runs of one commit differ by. The one place where the stack is slower than both is HuggingFace for inference
(0.8% and 1.5%), where the two runs of the starting commit differ by 0.7% the other way; two runs cannot tell whether that is the
dispatcher.</p>""")
    A(f"""<p><strong>Before the dashboard's runner: the stack's harness, on the build without cuDNN.</strong> The ratios in the rest of this
section are of TorchBench at its default batch sizes in float32, each compiler in a process of its own, on that build. They are kept
because they are what each change was measured by. The first run of all models on the GPU, on the fifth version, had
<code>torch.rewrite</code> at 1.10 times <code>torch.compile</code> for inference. Profiling single models found three causes, none of
them in a kernel, and each became a PR:</p>
<ol>
<li><strong>A default device is a Python mode.</strong> <code>torch.set_default_device</code> is a torch function mode: Python is
entered for every <code>torch</code> function called while it is active, the few hundred calls of the code Inductor generates included.
A graph made under a default device is now called with the mode off the stack, as Dynamo does ({pr(205245)}): {gpu_moved('llama')}.</li>
<li><strong>Left to the interpreter too early.</strong> A callee is timed in the VM and in the interpreter and left to the
interpreter if it is faster there. A callee whose region had not yet decided to become a graph ran its instructions one by one, lost
that comparison, and never came back, so the region never got its graph: cm3leon_generate ran at the speed of eager with 32 callees
left to the interpreter, its decoder layers among them. A region that is still deciding now keeps the measurement open
({pr(205246)}): {gpu_moved('cm3leon_generate')}.</li>
<li><strong>Three passes over a module's tensors on every call.</strong> Before its graph ran, a region read each tensor of the
module out of its dict, described them all for the key of the call, and checked the ones the graph has built in: 193 of the 1,670
microseconds of a call of hf_DistilBert. One pass in Rust now checks that nothing changed since the tensors were last looked at
({pr(205258)}): 118 microseconds.</li>
</ol>
<p>That was the sixth version: level for inference in that harness ({gpu_summary("eval")["geo"]:.2f}). What kept it level was
looked for model by model, and became the first ten PRs of the seventh version:</p>
<ol>
<li><strong>A size that only data decides.</strong> A region with <code>nonzero</code> or a mask as an index in it was traced, because
the trace gives such a size a symbol, and then given up as a whole: Inductor was handed the graph without the ShapeEnv the symbol
belongs to. The trace is now made again in one the backend shares ({pr(205326)}): {gpu_since('basic_gnn_gcn')}, one graph where Dynamo
makes six, and {gpu_since('hf_Reformer')}.</li>
<li><strong>What a compiled call costs before its graph runs.</strong> A region made the key of every call by walking, in Python, what it
was handed, after a walk over the dicts of every module to see that none had changed: 83 microseconds for a ResNet-50 with a backend
that computes nothing. The dicts are now watched, so one number says that none changed ({pr(205328)}); the places of the tensors are not
looked at while it says so ({pr(205329)}); and a call that is like the one before is told so in one pass in the runtime, which returns
its key and inputs ({prs(205330, 205332)}): 21 microseconds. {gpu_since('hf_Bart')}, {gpu_since('lennard_jones')}.</li>
<li><strong>Objects made again by their <code>__init__</code>.</strong> What a model returns that a region made (a
<code>ModelOutput</code>, a distribution) was made again on every call by its class, in the interpreter. It is a new object with the same
attributes now ({pr(205327)}).</li>
<li><strong>A clock that does not wait for the GPU.</strong> A function is left to the interpreter when it is faster there, but a kernel
on a GPU is started and not waited for: a graph that waits somewhere inside looked slower than kernels that were only started. A call
that is measured now ends when the GPU is idle ({pr(205334)}).</li>
<li><strong>The same program in every mode.</strong> Dynamo turns the validation of the arguments of distributions off for its process;
the harness measured eager and <code>torch.rewrite</code> with it on. It is off in all three now ({pr(205335)}):
{gpu_since('soft_actor_critic')}.</li>
</ol>
<p>In that harness the seventh version took {D.geo(r["ratio"] for r in D.GPU7["eval"]):.2f} of the time of <code>torch.compile</code> for
inference over {len(D.GPU7["eval"])} models, where the sixth took {gpu_summary("eval")["geo"]:.2f}.</p>""")

    # ---------------------------------------------------------------- 10
    A("<h2>10. One list of passes, and passes of your own</h2>")
    A(f"""<p>The entry point used to be <code>torch.cfg_compile</code>, with nine keywords for what it should do. It is
<code>torch.rewrite(fn, passes=[...])</code> now ({pr(205237)}, {pr(205238)}). The list is what is done to the function, in its order.
With an empty list the function runs with exactly the semantics of the interpreter, so nothing is compiled: a function is handed over
together with a list of rewrites, and each of them can be looked at.</p>
<pre><code>torch.rewrite.canonical
# ('inline', 'unroll', 'shortcuts', 'resolve_dispatch', 'tensor_accessors', 'method_calls', 'fold', 'dce')

@torch.rewrite(passes=[torch.rewrite.inductor, *torch.rewrite.canonical])   # graphs by AOTAutograd and Inductor
def forward(model, x): ...</code></pre>
<p><strong>A kernel in place of operations.</strong> A pass sees which ATen operator each instruction was seen to be, so
<code>x.pow(2)</code>, <code>x ** 2</code> and <code>torch.pow(x, 2)</code> are the same thing to it.
<code>torch.rewrite.swap(ops, kernel)</code> finds a run of operators in a block, puts one call of the kernel in its place, and for the
first calls runs the replaced instructions on the same inputs and compares ({pr(205242)}):</p>
<pre><code>def rms_norm(x, weight):
    return x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + 1e-6) * weight

fuse = torch.rewrite.swap(["pow", "mean", "add", "rsqrt", "mul"], my_kernel)
g = torch.rewrite(rms_norm, passes=[fuse, *torch.rewrite.canonical])

# a kernel that is wrong:
# torch.rewrite.Mismatch: wrong: result 0 differs by up to 0.0217 on call 1
#   (the operations it replaced were run on the same inputs)</code></pre>
<p><strong>A pass in Rust.</strong> The same pass can be written in Rust, in a library of its author's that links against neither
Python nor PyTorch and is built with plain <code>cargo build</code> ({pr(205243)}). The kernel stays a Python object that the pass is
bound to; where it goes is decided in Rust:</p>
<pre><code>impl Pass for FuseRmsNorm {{
    fn name(&amp;self) -&gt; &amp;'static str {{ "fuse_rmsnorm" }}
    fn run(&amp;self, region: &amp;mut Region&lt;'_&gt;, kernel: Option&lt;Bound&gt;) -&gt; Result&lt;bool, Error&gt; {{
        let Some(kernel) = kernel else {{ return region.fail("fuse_rmsnorm needs a kernel") }};
        match region.find(&amp;["pow", "mean", "add", "rsqrt", "mul"]) {{
            Some((start, end)) =&gt; region.swap(start, end, kernel, 4),   // checked for 4 calls
            None =&gt; Ok(false),
        }}
    }}
}}
export_passes!(FuseRmsNorm);

# Python
passes = torch.rewrite.load_passes("target/release/libmine.so")
g = torch.rewrite(rms_norm, passes=[passes["fuse_rmsnorm"].bind(my_kernel), *torch.rewrite.canonical])</code></pre>
<p><strong>Functions that C code calls.</strong> The VM used to rewrite a callee only when its own call instruction made the call. The
<code>__init__</code> of a class, a property, an operator, the <code>key</code> of <code>sorted</code>, a function under
<code>functools.partial</code> and everything a generator calls ran in the interpreter, with all they called. While a rewritten function
runs, the interpreter now starts its frames through a hook of the runtime (PEP 523, the mechanism Dynamo uses), and the hook runs a
frame's function in its rewritten form, in the frame the interpreter made ({pr(205240)}). The IR stays the VM's own: the hook is only how
frames that the VM did not start come back to it. A function that takes the hook pays about 25 ns a call for it, and code that is not
under a rewritten function pays nothing.</p>
<p><strong>CPython 3.14.</strong> The stack was 3.12-only in three places: the pyo3 version, the instruction set the frontend knew, and
the layout of interpreter frames. All three are ported ({prs(205259, 205263)}): the tests of the stack pass under 3.12 and under 3.14,
and over the 181,327 functions of the PyTorch tree that exist under both, the frontend accepts and turns down the same ones.</p>""")

    A("<h2>11. Ahead, level, behind</h2>")
    A(f"""<div class="three">
<div><h3>Ahead</h3><ul>
<li><strong>Convolutional models</strong>, from memory and built-in weights: {names('eval', ['phlippe_densenet', 'mnasnet1_0', 'squeezenet1_1', 'densenet121', 'mobilenet_v2', 'mobilenet_v3_large', 'resnet50', 'shufflenet_v2_x1_0', 'resnet152', 'vgg16', 'alexnet'])}.</li>
<li><strong>Models with large temporaries</strong> (graph networks, stencil codes): {names('eval', ['basic_gnn_gin', 'basic_gnn_sage', 'basic_gnn_gcn', 'pyhpc_turbulent_kinetic_energy', 'pyhpc_isoneutral_mixing'])}.</li>
<li><strong>Generation loops</strong>: {names('eval', ['hf_T5_generate', 'cm3leon_generate'])}.</li>
<li><strong>Models that are many graphs under Dynamo</strong>: hf_BigBird {E['alone']['hf_BigBird']:.2f} (1 graph against 28).</li>
<li><strong>Rarely slower than eager</strong>: <code>torch.compile</code> is more than 5% slower than eager on {len(E['slower_than_eager']['compile'])}
inference models ({', '.join(E['slower_than_eager']['compile'][:3])}, ...) and {len(T['slower_than_eager']['compile'])} training models;
<code>torch.rewrite</code> on none for inference and one for training ({', '.join(T['slower_than_eager']['cfg'])}), because it times compiled
code against the interpreter and keeps the faster.</li>
<li><strong>Training</strong>: {names('train', T['ahead'], limit=8)}.</li>
<li><strong>Compile time</strong>: {E['compile_geo']['cfg']:.1f} s against {E['compile_geo']['compile']:.1f} s for inference, {T['compile_geo']['cfg']:.1f} s against {T['compile_geo']['compile']:.1f} s for training.</li>
<li><strong>Safety of the fast path</strong>: built-in weights are checked on every call; a write through <code>.data</code> is seen.</li>
</ul></div>
<div><h3>Level, or nearly</h3><ul>
<li><strong>Transformers on batches</strong>, whose time is in BLAS, gain a few percent at most: {names('eval', ['hf_Bert', 'hf_Bert_large', 'hf_GPT2_large', 'hf_T5_large', 'hf_distil_whisper', 'moondream', 'llama', 'llava'])}.</li>
<li><strong>Against a tuned torch.compile</strong> (allocator that keeps memory, <code>freezing</code> on): within a few percent on most
convolutional models.</li>
<li><strong>Accuracy</strong>: both within float rounding of eager for inference.</li>
<li><strong>Time to steady state in training</strong>: {T['warm_geo']['cfg']:.1f} s against {T['warm_geo']['compile']:.1f} s.</li>
</ul></div>
<div><h3>Behind</h3><ul>
<li><strong>Memory</strong>: {held_geo:.2f} times <code>torch.compile</code>'s peak in training, {held_geo_eval:.2f} in inference; most on
{worst[0][1]} ({gb(worst[0][2])} against {gb(worst[0][3])}).</li>
<li><strong>Very short calls</strong>: soft_actor_critic ({D.ms(EV['soft_actor_critic']['cfg']['call_s'])} against
{D.ms(EV['soft_actor_critic']['compile']['call_s'])}) is {E['alone']['soft_actor_critic']:.2f} alone, and lennard_jones
({D.ms(lj['cfg']['call_s'])}) no better than level ({E['alone']['lennard_jones']:.2f}); entering a region costs a walk over its
arguments in Python.</li>
<li><strong>One training model</strong>: resnet152 at {T['alone']['resnet152']:.2f} alone ({T['paired']['resnet152']:.2f} in turns).</li>
<li><strong>Plain Python in the VM</strong>: three to four times slower than CPython when forced; hidden by handing it back.</li>
<li><strong>Around the update of the optimizer</strong>: the update is a region; <code>zero_grad</code> and the bookkeeping of a step run in the VM or the interpreter.</li>
<li><strong>Guards</strong>: reads made by interpreted callees from objects that are not inputs are not checked.</li>
<li><strong>Not built</strong>: generators, coroutines, <code>except*</code>; regions over modules with hooks; Python other than 3.12 and 3.14.</li>
<li><strong>On a GPU</strong> (section 9): {suite_summary("inductor")["all"]["held"]:.2f} times the peak memory of <code>torch.compile</code>;
a model whose modules have hooks runs at the speed of eager (opacus_cifar10); calls of under a tenth of a millisecond.</li>
<li><strong>Maturity</strong>: a week old; 124 of the 129 PRs have had neither review nor CI.</li>
</ul></div>
</div>""")

    # ---------------------------------------------------------------- 10
    A("<h2>12. What went wrong on the way</h2>")
    A(f"""<p>Most of the errors were in measuring, and several flattered the prototype until they were found. They are listed because the
final numbers are only as good as the list is complete.</p>
<ul>
<li><strong>A shared kernel cache.</strong> Inductor's on-disk cache is shared between processes, so whichever compiler ran second found
the kernels the first had built. The first run's "7.5 s against 4.3 s" compile time was that; cold, resnet50 was 17.4 s against 15.2 s.
Every process now gets an empty cache ({pr(204943)}).</li>
<li><strong>Timing before the measurement had settled.</strong> Calls were timed while <code>torch.rewrite</code> was still comparing
compiled code with the interpreter, which made it look slower than it was. The harness now waits ({pr(204941)}).</li>
<li><strong>Wrappers left on nn.Module.</strong> An early version wrapped methods of <code>nn.Module</code> process-wide; with them
<code>torch.compile</code> failed on one model in the same process and ran 2% slower on another, which flattered the comparison.
Removed ({pr(204948)}).</li>
<li><strong>A cache that grew, with the right answer.</strong> An object made inside a trace was handed out again on every call, so T5's
key-value cache grew across calls to <code>generate</code> while the tokens stayed correct. Found from a profile showing the same kernel
at 514 &micro;s and 6 &micro;s ({pr(204984)}).</li>
<li><strong>A benchmark model with no weights.</strong> cm3leon_generate is built without initializing its weights: 2.6 GB of zeros in
memory that was never written, which the kernel serves from one 4 KB page. Every compiler looked faster on it than a model with weights
allows (eager 5.7 s; 9.9 s once the weights are written), and it made the one-row rewrite look like 0.43 where it is
{E['alone']['cm3leon_generate']:.2f}. The harness now writes every weight once ({pr(205088)}).</li>
<li><strong>Weights copied on every decoder step.</strong> A small region had built in 145 matrices it never used; the decoder-step graph
read the same matrices through writable pointers, so each was copied on every step, and the measurement gave up on the compiled code.
Only what a graph uses is built in now.</li>
<li><strong>Too many workers for the container.</strong> Nine training processes at once went past the 256 GiB of the container the
benchmark ran in, which killed it as a whole, four times, before the cause was understood. The harness now reads the cgroup limit, starts
nothing above a fraction of it, and stops its own largest process above a second mark ({pr(205096)}). The final run peaked at 161 GiB.</li>
<li><strong>A leak, found by that guard.</strong> The first process the guard stopped was hf_Bart's training under <code>torch.rewrite</code>,
at 55 GB where eager peaks at 8 GB. A region that is handed a loss module built inside <code>forward</code>, as every model of
transformers does, made a variant per step, and each variant kept that step's logits alive: 393 MB per step. Fixed, with the price that
the small region around such a loss now runs uncompiled ({pr(205102)}, {pr(205103)}).</li>
<li><strong>A trace that used up a generator.</strong> A region that spans blocks is traced by running its instructions on copies of
what it was handed; a generator cannot be copied, and where a region began with <code>tuple(gen)</code> the trace emptied it, so the
call got <code>()</code>: a wrong result, once, without an error, since the third version. It surfaced in the sixth, when the frame hook
brought <code>ModelOutput.to_tuple</code> of transformers into the VM and two models failed in a re-run on the CPU ({pr(205272)}).</li>
<li><strong>Handed back to the interpreter before it had a graph.</strong> The measurement that leaves slow code to the interpreter was
taken while regions were still running their instructions one by one, so on the GPU, where the VM's own margin is small, whole
transformer layers were left to the interpreter and never came back (section 9).</li>
<li><strong>The VM's own call, caught by its own hook.</strong> To let the interpreter look at signals and other threads, the VM calls
an empty Python function. With the frame hook that call came back to the VM, and other threads starved; one test caught it.</li>
<li><strong>A CPU re-run under load.</strong> The sixth version was run again on the CPU while the GPU run shared the machine. Models
bound by memory bandwidth took up to 1.8 times as long as in the fifth version's run; alone, they take what they took (dlrm 49.3 and
49.7 ms). The CPU numbers of this page are therefore still the fifth version's.</li>
<li><strong>A memory table that mixed two measurements.</strong> Thirteen training rows of the last version's results were from a run made
before the harness recorded the peak of the <code>torch.rewrite</code> process ahead of the calls in turns, so their peak included
<code>torch.compile</code> and its copy of the model in the same process. They were left out of the geometric mean that was reported
(1.07 over 35 models), but they stood in the table, and densenet121 looked like 68 GB against 33 GB there. Every row of this version is
measured the same way.</li>
<li><strong>A recording that kept a model's tensors alive.</strong> The 18 GB of hf_Reformer was attributed to the workspace for a
day. It was the VM's recording of a C function's kernel calls (section 7), found only when the same peak showed with the workspace
switched off.</li>
<li><strong>Tests that passed without the feature.</strong> Small functions are inlined into the compiled function, so three new tests
never exercised the code they were written for. Every new test is now run against the parent commit, and must fail there.</li>
</ul>""")

    # ---------------------------------------------------------------- 11
    A("<h2>13. Suggested next steps</h2>")
    A(f"""<ol>
<li><strong>Make the graphs before the first call runs.</strong> What is left of the memory cost is the first calls of a function, which run
kernel by kernel and hold more at once than its graphs do later (section 7). Tracing a region on the arguments of the first call, instead
of after a few calls that are surveyed, would take that away, and with it most of the time to the steady state.</li>
<li><strong>Run generators in the VM where it pays.</strong> With the variables in the frame, a generator's state can live where the
interpreter keeps it. On its own the measurement would hand a generator back to the interpreter; it pays where the body is inlined into the
loop that drives it. The <code>hold</code> instruction, which keeps a value alive as long as its variable, is redundant now that the frame
does, and removing it takes values out of what regions are handed.</li>
<li><strong>Keep the loss region of transformer models compiled.</strong> Name a module that has no tensors by its class and attribute
values instead of its identity, so that a loss built on every call is one variant and not one per call. The leak fix retires that region
instead, which is safe and gives up a little speed on every HuggingFace training step.</li>
<li><strong>Modules with hooks, and branches on data.</strong> A region refuses a module with hooks, and with real convolutions that is
the largest gap there is: opacus_cifar10 runs at the speed of eager where <code>torch.compile</code> gets 2.8 times. Lifting the refusal
is not enough (tried): the results stay right, a hook that changes an output is traced, but a hook that appends to a list is run at the
trace and never again. A branch on what a tensor holds still ends a region (hf_Bart checks every layer's output for infinities), and a
loop over the layers of a <code>ModuleList</code> is not unrolled.</li>
<li><strong>Enter a region from a cold cache.</strong> The dicts of the modules tell of their changes now, and a call that is like the
one before is told so in one pass in the runtime. What is left shows on calls of under a tenth of a millisecond right after an eager call
of the same model, which is how the dashboard's runner times: a look at every tensor of the modules a region is handed, about 230
nanoseconds a tensor on memory that was pushed out of the cache. A region also cannot hold a <code>with</code> whose manager a
<code>functools.partial</code> makes (the runner's <code>autocast</code>), so the VM runs it around the graph.</li>
<li><strong>Look for more kernels that do not use the machine.</strong> With memory and weights handled, what is left on the CPU is in
the kernels: BLAS for transformers on batches, MKLDNN for convolutions. The one-row product was found by asking, per kernel, how many
cores it used. A profile of every model in that form would find the others, if there are any.</li>
<li><strong>Close the guard gap.</strong> Trace callees through the VM's own graphs instead of running them in the interpreter during
a trace, so that every read is seen and guarded. This is the main soundness difference from Dynamo.</li>
<li><strong>Compile what training still leaves to the interpreter, and decide sooner.</strong> Frames that C++ enters (a custom
<code>autograd.Function</code>'s <code>forward</code>, as in hf_Reformer), which the VM cannot see today. And the measurement of a
function against the interpreter goes one level of calls after the other, about thirty calls a level: a training step, which goes
through an optimizer, is still being measured after several hundred calls.</li>
<li><strong>Make the VM fast on Python</strong>, or keep handing it back: inline caches for attribute loads and method calls are the
first candidates.</li>
<li><strong>Move the walk over a region's arguments into Rust.</strong> It is 0.4 ms per decoder step of T5 and the reason very short
calls are behind.</li>
<li><strong>Review and land from the bottom.</strong> The five PRs with CI are the dispatcher hook; everything above depends on a
decision about whether this front end is wanted. Several PRs above are larger than a reviewer should be asked to read.</li>
<li><strong>Report the cm3leon_generate finding upstream.</strong> A benchmark whose weights are unwritten zeros reads one page of
the kernel over and over, for every compiler.</li>
</ol>""")

    # ---------------------------------------------------------------- 12
    A("<h2>14. Reproducing</h2>")
    A(f"""<pre><code># build (the Rust runtime needs cargo on PATH)
USE_CFG_COMPILE=1 pip install -e . -v --no-build-isolation

# tests of the stack, and CPython's own
PYTORCH_TEST_WITH_SLOW=1 python test/test_rewrite.py
python benchmarks/rewrite/cpython_suite.py --lib &lt;CPython 3.12&gt;/Lib --verbose &lt;modules&gt;

# TorchBench, within the memory of the machine or container
python benchmarks/rewrite/torchbench.py --torchbench &lt;pytorch/benchmark checkout&gt; \\
    --tests eval,train --cpus 9,10,11,12/14,15,16,17/... --memory-start 0.40 --memory-stop 0.65 \\
    --results results.jsonl</code></pre>
<p>The result files behind the charts and the per-model tables are in
<a href="https://github.com/bobrenjc93/explanations/tree/main/cfg-compile/results">cfg-compile/results</a> (one JSON line per model,
test and compiler; <code>final_r22.jsonl</code> is the final CPU run; <code>suites_r25e/</code> and <code>suites_r25e_train/</code> have what the dashboard's runner wrote on the GPU, one CSV per suite and compiler (<code>-kept</code>: the second runs, with the chunks kept for <code>torch.compile</code> too), and <code>chunks.log</code> the chunks each timed loop asked for; <code>suites_r25c/</code>, <code>suites_r25d/</code> and <code>suites_r25d_train/</code> are the runs before those; <code>gpu_r24.jsonl</code> and <code>gpu_r23.jsonl</code> are the stack's harness on the GPU, on a build without cuDNN; the others are the earlier versions), and the generator of this page
in <a href="https://github.com/bobrenjc93/explanations/tree/main/cfg-compile/code">cfg-compile/code</a>. Numbers quoted from
single-model runs (section 4, the micro benchmarks, the conformance run) are from the descriptions of the PRs that made each change and
from the report in the top PR. The top of the stack is {pr(204763)}.</p>""")
    A("""<footer>Written with an AI assistant (Claude Code), which also wrote the stack and ran the measurements. One machine, one
GPU, one run per configuration: a difference of a few percent between two models, or between two runs, is not a finding. Results and
the generator of this page:
<a href="https://github.com/bobrenjc93/explanations/tree/main/cfg-compile">cfg-compile</a>.</footer>""")
    return "\n".join(o)


def wrap_tables(html):
    """The tables written inline in the prose get the same sideways-scrolling box as the generated ones."""
    return html.replace("<table>\n<tr>", '<div class="scroll"><table>\n<tr>').replace("</tr>\n</table>", "</tr>\n</table></div>")


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
{wrap_tables(body())}
<script>{C.JS}</script>
</body>
</html>
"""
    open(out, "w").write(html)
    print("wrote", out, len(html) // 1024, "KB")


if __name__ == "__main__":
    main(sys.argv[1] if len(sys.argv) > 1 else "draft.html")
