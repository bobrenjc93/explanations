"""Builds the cfg_compile report page: prose here, figures from charts.py, numbers from data.py.

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
            {"name": "torch.cfg_compile", "short": "cfg_compile", "c": "--s1", "y": [s["speed"]["cfg"] for s in ss]},
            {"name": "torch.compile", "short": "torch.compile", "c": "--s2", "y": [s["speed"]["compile"] for s in ss]},
        ]
        body = C.line_chart(series, list(range(4)), xlabels=[v for v, _, _ in D.VERSIONS], w=400, h=250, ymin=1.0, ymax=2.0,
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
        tip = [("eager", D.ms(d["eager"]["call_s"])), ("torch.compile", D.ms(d["compile"]["call_s"])), ("cfg_compile", D.ms(d["cfg"]["call_s"])),
               ("in turns", "-" if p is None else f"{p:.2f}"),
               ("graphs, torch.compile / cfg_compile", f"{d['compile'].get('graphs', '-')} / {d['cfg'].get('graphs', '-')}")]
        items.append({"name": n, "v": v, "tip": tip})
        rows.append([n, D.ms(d["eager"]["call_s"]), D.ms(d["compile"]["call_s"]), D.ms(d["cfg"]["call_s"]), f"{v:.2f}", "-" if p is None else f"{p:.2f}",
                     x(d["eager"]["call_s"] / d["compile"]["call_s"]), x(d["eager"]["call_s"] / d["cfg"]["call_s"]),
                     f"{d['compile'].get('compile_s', 0):.1f} / {d['cfg'].get('compile_s', 0):.1f}",
                     f"{d['compile'].get('graphs', '-')} / {d['cfg'].get('graphs', '-')}",
                     f"{d['compile'].get('difference', 0):.0e} / {d['cfg'].get('difference', 0):.0e}"])
    header = ["model", "eager", "torch.compile", "cfg_compile", "time ratio, alone", "in turns", "torch.compile over eager", "cfg_compile over eager",
              "compile s (tc / cfg)", "graphs (tc / cfg)", "difference from eager (tc / cfg)"]
    return items, header, rows


def fig_models(test):
    items, header, rows = model_rows(test)
    s = E if test == "eval" else T
    lo = 0.30 if test == "eval" else 0.42
    body = C.ratio_bars(items, lo, 1.25, (0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0, 1.1, 1.2) if test == "eval" else (0.5, 0.6, 0.7, 0.8, 0.9, 1.0, 1.1, 1.2),
                        tv_header=header, tv_rows=rows)
    what = "inference" if test == "eval" else "a training step"
    cap = (f"Time of one call under <code>torch.cfg_compile</code> over the time under <code>torch.compile</code>, each in a process of its own; "
           f"{what}, {len(items)} models, logarithmic scale. Below 1 is faster. Geometric mean {s['alone_geo']:.2f}. Hover or focus a row for the times.")
    return C.figure(f"{'Inference' if test == 'eval' else 'Training'}, per model", cap, body)


def fig_memory():
    h = D.held(TR)
    items = [{"name": n, "v": r, "tip": [("cfg_compile", f"{a:.1f} GB"), ("torch.compile", f"{b:.1f} GB"), ("eager", f"{e:.1f} GB")]} for r, n, a, b, e in h]
    rows = [[n, f"{e:.1f}", f"{b:.1f}", f"{a:.1f}", f"{r:.2f}"] for r, n, a, b, e in h]
    body = C.ratio_bars(items, 0.7, 5.0, (0.8, 1.0, 1.5, 2.0, 3.0, 4.0), band=0.05, names=("cfg_compile holds less", "within 5%", "cfg_compile holds more"),
                        tv_header=["model", "eager GB", "torch.compile GB", "cfg_compile GB", "cfg_compile over torch.compile"], tv_rows=rows)
    cap = (f"Peak resident memory of the training process under <code>torch.cfg_compile</code> over that under <code>torch.compile</code>, "
           f"for the {len(h)} models where it was recorded before the two share a process; logarithmic scale. Geometric mean {D.geo(r for r, *_ in h):.2f}.")
    return C.figure("What the processes hold while training", cap, body), h


def fig_micro():
    items = [
        {"name": "chain of 9 ops", "v": 1.52, "tip": [("eager", "18,377 ns"), ("cfg_compile", "12,059 ns")]},
        {"name": "for loop, 10 x (3 ops + branch)", "v": 1.50, "tip": [("eager", "94,558 ns"), ("cfg_compile", "62,965 ns")]},
        {"name": "while loop on a tensor value", "v": 1.48, "tip": [("eager", "47,186 ns"), ("cfg_compile", "31,840 ns")]},
        {"name": "2-layer MLP (addmm)", "v": 1.11, "tip": [("eager", "9,130 ns"), ("cfg_compile", "8,246 ns")]},
        {"name": "Python loop over 8 tensors", "v": 0.96, "tip": [("eager", "10,930 ns"), ("cfg_compile", "11,419 ns")]},
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
    ("Memory, weights and one-row products", 14, "1,651", "The last round, described below, with two leak fixes and three fixes to the benchmark."),
    ("Tests, benchmarks and tools", 4, "510", "The micro benchmarks and the runner for CPython's own test suite."),
]


def tuned_table():
    rows = [[n, f"{a:g}", f"{b:g}", f"{c:g}", f"{c / min(a, b):.2f}"] for n, a, b, c in TUNED]
    return C.table(["model (inference, ms per call)", "torch.compile", "torch.compile with freezing", "cfg_compile, defaults", "cfg_compile over the better of the two"], rows)


def groups_table():
    body = "".join(f"<tr><td>{g}</td><td class=num>{n}</td><td class=num>{lines}</td><td>{what}</td></tr>" for g, n, lines, what in GROUPS)
    head = "<tr><th>part of the stack</th><th class=num>PRs</th><th class=num>inserted lines</th><th>what it is</th></tr>"
    return f'<div class="scroll"><table><thead>{head}</thead><tbody>{body}</tbody></table></div>'


# --------------------------------------------------------------------------- the page

TITLE = "torch.cfg_compile: a control-flow-graph compiler for PyTorch, measured against torch.compile"
SUB = ("A prototype that compiles the bytecode of a Python function to one control-flow graph, runs it in a Rust VM and hands the tensor work "
       "to Inductor: what was built over 86 draft PRs, how it compares with torch.compile on TorchBench in accuracy, speed, compile time and "
       "memory, where it is ahead, where it is behind, and what to do next. Nov 11 to Nov 16, 2026.")


def body():
    mem_fig, h = fig_memory()
    held_geo = D.geo(r for r, *_ in h)
    worst = h[::-1][:4]
    o = []
    A = o.append
    A(f"<h1>{TITLE}</h1>")
    A(f'<p class="sub">{SUB}</p>')

    A(f"""<div class="tldr">
<p><strong>TL;DR.</strong> <code>torch.cfg_compile</code> is a prototype of a different front end for PyTorch's compiler. It turns the
bytecode of a function into one control-flow graph in SSA form, runs that graph in a VM written in Rust, traces on real tensors, and hands the
tensor work of each path through the graph to AOTAutograd and Inductor, the compiler <code>torch.compile</code> uses behind Dynamo. It exists as
a stack of 86 draft pull requests against pytorch/pytorch, about 27,000 lines.</p>
<p>On the TorchBench models that run on this CPU box, it takes <strong>{E['alone_geo']:.2f} of the time of <code>torch.compile</code> for
inference</strong> ({len(E['names'])} models) and <strong>{T['alone_geo']:.2f} for training</strong> ({len(T['names'])} models), geometric
means. That is {x(E['speed']['cfg'])} and {x(T['speed']['cfg'])} eager, where <code>torch.compile</code> is {x(E['speed']['compile'])} and
{x(T['speed']['compile'])}. Its outputs agree with eager's as closely as <code>torch.compile</code>'s do, and it compiles in less time.</p>
<p>Three things produce the margin, all of them new in the last round and none of them in a kernel: the compiled function keeps the memory of
a call for the next one, the weights of a model are built into the graphs of inference behind a check on every call, and matrix products
with one row are rewritten so that they use every core. <strong>It is a margin over <code>torch.compile</code> with its defaults.</strong>
Given an allocator that keeps memory and its own <code>freezing</code> switch, <code>torch.compile</code> is level on most models, because
the kernels are the same.</p>
<p>It is behind on memory in training for a few models (the worst holds 18 GB where <code>torch.compile</code> holds 4 GB), on models whose
call takes under a few milliseconds, on plain Python in the VM, and in everything that was not built or not measured: CUDA, the optimizer
step, generators, sizes that only data decides.</p>
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
<p><code>torch.cfg_compile</code> starts from the other end. The whole function is compiled, once, to a control-flow graph whose
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
    A("<h2>2. The path: four versions of the graph backend</h2>")
    A(f"""<p>The VM came first and took the first three of the six days: the language, the fidelity to CPython, the measured choice between
VM and interpreter. The comparison with <code>torch.compile</code> starts with the graph backend, which went through four versions. Each was
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
({prs(205043, 205052)}, {pr(205088)} to {pr(205103)}).</td><td>{E['alone_geo']:.2f} for inference and {T['alone_geo']:.2f} for training. The rest
of this page is about this version.</td></tr>
</table>""")
    A("""<p>The step from v3 to v4 is the answer to a direct question: with the two compilers at parity, is there anything the architecture
can do that Dynamo does not? Profiles said the remaining time was in library kernels (82% of resnet50 in <code>convolution</code>, 85% of
hf_Bert in <code>addmm</code>), so the search went to what surrounds the kernels.</p>""")

    # ---------------------------------------------------------------- 3
    A("<h2>3. Results on TorchBench</h2>")
    A(f"""<p><strong>Setup.</strong> <a href="https://github.com/pytorch/benchmark">TorchBench</a>, run the way
<code>benchmarks/dynamo/torchbench.py</code> runs it: the module and example inputs of each model; inference under <code>no_grad</code> in
eval mode; a training step of <code>zero_grad</code>, forward, <code>reduce_to_scalar_loss</code>, <code>backward</code> and an SGD step.
The same function goes to <code>torch.compile(fn)</code> and to <code>torch.cfg_compile(fn, backend="inductor")</code>, with default
options for both. CPU only (AMD EPYC 7R13), 4 threads per run, pinned to 4 physical cores; every (model, test, compiler) in a process of
its own with an empty Inductor cache, so compile times are cold. PyTorch main of Nov 9, 2026 plus the stack, Python 3.12.14.</p>
<p>Of TorchBench's 104 models, {E['eager_ran']} load and run in eager mode here for inference and {T['eager_ran']} for training; the others
need datasets, TorchScript or CUDA and are left out for both compilers. Both compilers ran all {len(E['names'])} for inference and
{len(T['names'])} of the {T['eager_ran']} for training (opacus_cifar10 fails for both: the harness compares against a deep copy of the
model, and Opacus' hooks do not survive the copy).</p>""")
    A(fig_models("eval"))
    A(fig_models("train"))
    A(f"""<p><strong>Alone or in turns.</strong> The harness measures twice: each compiler in its own process, and, inside the
<code>cfg_compile</code> process, the two compiled functions called in turns. While both ran the same code, in turns was the number to
trust, because two processes on a shared machine differed by more than the compilers did. With v4 that reversed: <code>torch.compile</code>
is slower in a process that <code>cfg_compile</code> has allocated in than in a process of its own wherever page faults matter
(mobilenet_v3_large: 157 ms there, 109 ms alone), which flatters <code>cfg_compile</code>. So the final run was made on an idle machine
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
g = torch.cfg_compile(f)                 # no backend: the eager kernels
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
with <code>freezing</code> 799 ms, <code>cfg_compile</code> 675 ms. Matrices that a graph only multiplies with or looks rows up in are left
as inputs ({pr(205050)}): nothing folds, and a copy of them cost hf_Bert 400 MB.</p>""")
    A(f"""<h3>One-row matrix products</h3>
<p>A decoder that makes one token at a time multiplies a single row of activations with every weight matrix of the model. MKL runs
<code>mm</code> of a 1&times;K row with a K&times;N matrix on one core however many threads it has, and since the matrices do not fit
the cache, that core waits for memory. Rewritten as <code>(matrix * row.T).sum(0)</code> before the graph goes to Inductor ({pr(205045)}),
the product is a loop Inductor splits over the cores. A 768-to-32,128 output layer on one row: eager 7.3 ms, <code>torch.compile</code>
6.8 ms, <code>cfg_compile</code> 2.5 ms. hf_T5_generate takes {D.ms(EV['hf_T5_generate']['cfg']['call_s'])} against
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
<tr><th></th><th>torch.compile</th><th>torch.cfg_compile</th></tr>
<tr><td>Inference: within 1e-2 of eager</td><td>{E['within']['compile']} of {len(E['names'])}</td><td>{E['within']['cfg']} of {len(E['names'])}</td></tr>
<tr><td>Inference: within 1e-5</td><td>{E['within5']['compile']}</td><td>{E['within5']['cfg']}</td></tr>
<tr><td>Inference: largest difference</td><td>{E['worst']['compile'][0]:.1e} ({E['worst']['compile'][1]})</td><td>{E['worst']['cfg'][0]:.1e} ({E['worst']['cfg'][1]})</td></tr>
<tr><td>Training: within 1e-2 of eager</td><td>{T['within']['compile']} of {len(T['names'])}</td><td>{T['within']['cfg']} of {len(T['names'])}</td></tr>
<tr><td>Training: within 1e-5</td><td>{T['within5']['compile']}</td><td>{T['within5']['cfg']}</td></tr>
<tr><td>Training: bit-identical losses</td><td>{T['exact']['compile']}</td><td>{T['exact']['cfg']}</td></tr>
<tr><td>Training: over 1e-2</td><td>{', '.join(f'{n} {v:.1e}' for v, n in oc)}</td><td>{', '.join(f'{n} {v:.1e}' for v, n in og)}</td></tr>
</table>
<p>Inference agrees to float rounding for both. In training <code>cfg_compile</code> is the closer of the two to eager, because the
optimizer and what the measurement hands back run eager's own code. The four training runs over 1e-2 are all under 2e-2: differences of the size
that a differently ordered floating-point operation grows into over eight optimizer steps. Their cause was not looked into further. An earlier
sweep with <code>backend="aot_eager"</code>, which builds the same graphs and runs them with the eager kernels, was bit-exact on 99 of 102
configurations, and the other three fail in eager mode too.</p>""")
    A(f"""<p><strong>Python semantics.</strong> A compiler that replaces the interpreter has to behave like it. The stack's runner
(<code>benchmarks/cfg_compile/cpython_suite.py</code>) compiles every test method of a module of CPython 3.12's own <code>Lib/test</code>,
lets the VM compile whatever those methods call, and compares the outcome of each test with its outcome under the plain interpreter.</p>""")
    A(C.tiles([
        ("CPython test modules run through the VM", "183", "27,028 tests"),
        ("Test methods compiled", "16,900", "246 left to the interpreter: <code>eval</code>, <code>exec</code>, generators"),
        ("Outcomes that differ from the interpreter", "21", "it was 81, and a segmentation fault, on the first 83 modules"),
        ("Tests of the stack itself", "317", "those of the last two rounds each checked to fail without their change"),
    ]))
    A("""<p>These are from the last conformance run, on the version of Nov 15; the two rounds since changed the graph backend and the
benchmark, not the VM or the frontend. None of the 21 is in the result of a computation: nine are about <code>f_locals</code> (locals live in registers, not in the
frame), seven about <code>sys.settrace</code> line events that compiled code does not raise, the rest about reference counts and
specializer internals. Getting from 81 to 21 was a round of its own: real frames for compiled and inlined calls so tracebacks have
columns, CPython's exact error messages, locals released when the interpreter releases them, signals and thread switches inside compiled
loops, a recursion limit that means the same depth.</p>
<p><strong>What "matches" does not cover.</strong> A region checks its inputs (tensors, modules and their structure, values, modes) and
what its own path read from modules and globals. It does not check what a callee that the trace ran in the interpreter read from an
object that is not an input. Dynamo guards those reads; this does not yet.</p>""")

    # ---------------------------------------------------------------- 6
    A("<h2>6. Compile time and graphs</h2>")
    A(f"""<table>
<tr><th></th><th>torch.compile</th><th>torch.cfg_compile</th></tr>
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
<code>cfg_compile</code> that includes the calls it spends measuring compiled code against the interpreter, which is why it gives back
in training what it saves on compiling.</p>
<p>Graph counts differ most where control flow or side effects sit between tensor code. hf_BigBird is {EV['hf_BigBird']['compile']['graphs']}
graphs for <code>torch.compile</code> and {EV['hf_BigBird']['cfg']['graphs']} here; in training, mobilenet_v2_quantized_qat is
{TR['mobilenet_v2_quantized_qat']['compile']['graphs']} against {TR['mobilenet_v2_quantized_qat']['cfg']['graphs']}. It goes the other way
too: hf_Bart trains as {TR['hf_Bart']['cfg']['graphs']} graphs here against {TR['hf_Bart']['compile']['graphs']}, and demucs as
{TR['demucs']['cfg']['graphs']} against {TR['demucs']['compile']['graphs']}, where an instruction inside the model does not trace and
the pieces around it become regions of their own.</p>""")

    # ---------------------------------------------------------------- 7
    A("<h2>7. What it costs: memory</h2>")
    A(f"""<p>The workspace holds what a call freed for the next call: up to one and a half times what the call had in use at its peak, on
top of what is in use. In inference that is modest: the process of resnet50 grows by 566 MB over its calls, against 224 MB for
<code>torch.compile</code> (406 MB with <code>freezing</code>); hf_Bert by 219 MB against 192 MB. In training it is usually about the
same as <code>torch.compile</code>, and on a few models far more:</p>""")
    A(mem_fig)
    A(f"""<p>Over these {len(h)} models the geometric mean is {held_geo:.2f}. The outliers are models whose step makes and frees many large
tensors of different sizes: {', '.join(f'{n} {a:.0f} GB against {b:.0f} GB' for r, n, a, b, e in worst)}. The memory is held, not lost
(demucs sat at 55.7 GB for the eight minutes it was watched), but it is too much, and section 11 says what to do about it.</p>""")

    # ---------------------------------------------------------------- 8
    A("<h2>8. The VM without a graph backend</h2>")
    A("""<p>Without <code>backend=</code> the VM runs the function itself and calls ATen kernels directly. What that buys is the overhead
per operation, so it shows on small tensors and disappears on large ones.</p>""")
    A(fig_micro())
    A("""<table>
<tr><th>workload</th><th>eager</th><th>cfg_compile</th><th>time over eager</th></tr>
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
    A("<h2>9. Ahead, level, behind</h2>")
    A(f"""<div class="three">
<div><h3>Ahead</h3><ul>
<li><strong>Convolutional models</strong>, from memory and built-in weights: {names('eval', ['phlippe_densenet', 'mnasnet1_0', 'squeezenet1_1', 'densenet121', 'mobilenet_v2', 'mobilenet_v3_large', 'resnet50', 'shufflenet_v2_x1_0', 'resnet152', 'vgg16', 'alexnet'])}.</li>
<li><strong>Models with large temporaries</strong> (graph networks, stencil codes): {names('eval', ['basic_gnn_gin', 'basic_gnn_sage', 'basic_gnn_gcn', 'pyhpc_turbulent_kinetic_energy', 'pyhpc_isoneutral_mixing'])}.</li>
<li><strong>Generation loops</strong>: {names('eval', ['hf_T5_generate', 'cm3leon_generate'])}.</li>
<li><strong>Models that are many graphs under Dynamo</strong>: hf_BigBird {E['alone']['hf_BigBird']:.2f} (1 graph against 28).</li>
<li><strong>Rarely slower than eager</strong>: <code>torch.compile</code> is more than 5% slower than eager on {len(E['slower_than_eager']['compile'])}
inference models ({', '.join(E['slower_than_eager']['compile'][:3])}, ...) and {len(T['slower_than_eager']['compile'])} training models;
<code>cfg_compile</code> on none for inference and one for training ({', '.join(T['slower_than_eager']['cfg'])}), because it times compiled
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
<li><strong>Memory in training on a few models</strong>: hf_Reformer 18 GB against 4 GB, demucs 60 GB against 36 GB.</li>
<li><strong>Very short calls</strong>: lennard_jones ({D.ms(lj['cfg']['call_s'])} against {D.ms(lj['compile']['call_s'])}) is
{E['alone']['lennard_jones']:.2f} alone and {E['paired']['lennard_jones']:.2f} in turns, and soft_actor_critic (2.4 ms) no better than level
({E['alone']['soft_actor_critic']:.2f}); entering a region costs a walk over its arguments in Python.</li>
<li><strong>One training model</strong>: resnet152 at {T['alone']['resnet152']:.2f} alone ({T['paired']['resnet152']:.2f} in turns).</li>
<li><strong>Plain Python in the VM</strong>: three to four times slower than CPython when forced; hidden by handing it back.</li>
<li><strong>The optimizer step is not compiled</strong>; <code>torch.compile</code> compiles it.</li>
<li><strong>Guards</strong>: reads made by interpreted callees from objects that are not inputs are not checked.</li>
<li><strong>Not built</strong>: generators, coroutines, <code>except*</code>; sizes that only data decides; Python other than 3.12.</li>
<li><strong>Not measured</strong>: anything on CUDA. The workspace is for CPU memory only.</li>
<li><strong>Maturity</strong>: six days old; 81 of the 86 PRs have had neither review nor CI.</li>
</ul></div>
</div>""")

    # ---------------------------------------------------------------- 10
    A("<h2>10. What went wrong on the way</h2>")
    A(f"""<p>Most of the errors were in measuring, and several flattered the prototype until they were found. They are listed because the
final numbers are only as good as the list is complete.</p>
<ul>
<li><strong>A shared kernel cache.</strong> Inductor's on-disk cache is shared between processes, so whichever compiler ran second found
the kernels the first had built. The first run's "7.5 s against 4.3 s" compile time was that; cold, resnet50 was 17.4 s against 15.2 s.
Every process now gets an empty cache ({pr(204943)}).</li>
<li><strong>Timing before the measurement had settled.</strong> Calls were timed while <code>cfg_compile</code> was still comparing
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
<li><strong>A leak, found by that guard.</strong> The first process the guard stopped was hf_Bart's training under <code>cfg_compile</code>,
at 55 GB where eager peaks at 8 GB. A region that is handed a loss module built inside <code>forward</code>, as every model of
transformers does, made a variant per step, and each variant kept that step's logits alive: 393 MB per step. Fixed, with the price that
the small region around such a loss now runs uncompiled ({pr(205102)}, {pr(205103)}).</li>
<li><strong>Tests that passed without the feature.</strong> Small functions are inlined into the compiled function, so three new tests
never exercised the code they were written for. Every new test is now run against the parent commit, and must fail there.</li>
</ul>""")

    # ---------------------------------------------------------------- 11
    A("<h2>11. Suggested next steps</h2>")
    A(f"""<ol>
<li><strong>Bound what the workspace holds.</strong> Limit the sum of what is in use and what is kept to about 1.25 to 1.5 times the
peak, accept a kept block only if it is at most a quarter larger than the request (it is twice now), and evict on a miss. This needs
re-measuring densenet121, hf_T5_generate, hf_Reformer and demucs, since the same slack is what makes those fast. It is the one known
cost that should not ship as it is.</li>
<li><strong>Keep the loss region of transformer models compiled.</strong> Name a module that has no tensors by its class and attribute
values instead of its identity, so that a loss built on every call is one variant and not one per call. The leak fix retires that region
instead, which is safe and gives up a little speed on every HuggingFace training step.</li>
<li><strong>Measure on CUDA.</strong> Nothing here says what happens on a GPU. Built-in weights and fewer graphs carry over; the
workspace does not (the CUDA allocator already caches), and page faults are not the bottleneck there, so the margin is unknown and probably smaller.</li>
<li><strong>Look for more kernels that do not use the machine.</strong> With memory and weights handled, what is left on the CPU is in
the kernels: BLAS for transformers on batches, MKLDNN for convolutions. The one-row product was found by asking, per kernel, how many
cores it used. A profile of every model in that form would find the others, if there are any.</li>
<li><strong>Close the guard gap.</strong> Trace callees through the VM's own graphs instead of running them in the interpreter during
a trace, so that every read is seen and guarded. This is the main soundness difference from Dynamo.</li>
<li><strong>Compile what training still leaves to the interpreter.</strong> The optimizer step, and frames that C++ enters (a custom
<code>autograd.Function</code>'s <code>forward</code>, as in hf_Reformer), which the VM cannot see today.</li>
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
    A("<h2>12. Reproducing</h2>")
    A(f"""<pre><code># build (the Rust runtime needs cargo on PATH)
USE_CFG_COMPILE=1 pip install -e . -v --no-build-isolation

# tests of the stack, and CPython's own
PYTORCH_TEST_WITH_SLOW=1 python test/test_cfg_compile.py
python benchmarks/cfg_compile/cpython_suite.py --lib &lt;CPython 3.12&gt;/Lib --verbose &lt;modules&gt;

# TorchBench, within the memory of the machine or container
python benchmarks/cfg_compile/torchbench.py --torchbench &lt;pytorch/benchmark checkout&gt; \\
    --tests eval,train --cpus 9,10,11,12/14,15,16,17/... --memory-start 0.40 --memory-stop 0.65 \\
    --results results.jsonl</code></pre>
<p>The result files behind the charts and the per-model tables are in
<a href="https://github.com/bobrenjc93/explanations/tree/main/cfg-compile/results">cfg-compile/results</a> (one JSON line per model,
test and compiler; <code>final_r21f.jsonl</code> is the final run, the others are the earlier versions), and the generator of this page
in <a href="https://github.com/bobrenjc93/explanations/tree/main/cfg-compile/code">cfg-compile/code</a>. Numbers quoted from
single-model runs (section 4, the micro benchmarks, the conformance run) are from the descriptions of the PRs that made each change and
from the report in the top PR. The top of the stack is {pr(204763)}.</p>""")
    A("""<footer>Written with an AI assistant (Claude Code), which also wrote the stack and ran the measurements. One machine, CPU
only, one run per configuration: a difference of a few percent between two models, or between two runs, is not a finding. Results and
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
