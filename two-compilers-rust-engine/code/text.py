"""text.py: the prose of the page. body(G, n) gets make.py's module (the data, the charts, the tables) and its
headline numbers. Figures of the first run are quoted from the two earlier pages."""
TITLE = "One model, two compilers, run again: torch.vm with its graphs built in the Rust engine"
FIRST = {"start": {"tc": 2013, "vm": 784}, "best3": {"tc": 3973, "vm": 4507}, "best6": {"tc": 3973, "vm": 4614}, "passing6": {"tc": 3634, "vm": 4614}}
PR = "https://github.com/pytorch/pytorch/pull/204763"
FOOTER = ('<footer>Written with an AI assistant (Claude Code), which also set up the experiment, ran the checks and made the charts; the two agents were instances of the same model. '
          'One model, one run per side, one GPU: an example, not a benchmark. The harness, the rules both agents were given, their journals, summaries and final changes, the results and the generator of this page: '
          '<a href="https://github.com/bobrenjc93/explanations/tree/main/two-compilers-rust-engine">two-compilers-rust-engine</a>.</footer>')
CSS = """
table s { color:var(--muted); }
table.wide { font-size:.86em; }
table.wide td:nth-child(-n+5), table.wide td:nth-child(7) { white-space:nowrap; }
table.wide td:nth-child(6) { min-width:170px; }
table.wide td:nth-child(8) { min-width:200px; overflow-wrap:anywhere; }
tr.off td { color:var(--muted); }
tr.off td a { color:var(--muted); }
"""


def head(C, A):
    description = ("TorchBench yolov3 inference under torch.compile and torch.vm, second run: two fresh instances of the same AI agent get six hours each to improve one compiler, after torch.vm's "
                   "graph compiler was rewritten inside its Rust engine. Throughput against the time and tokens spent; the best recorded state and the best that passes every check for each side; "
                   "predictions checked with pretrained weights on COCO images, probes of what each state still follows, and training loss curves against eager. Nov 21, 2026.")
    return (f'<!doctype html>\n<html lang="en">\n<head>\n<meta charset="utf-8">\n<meta name="viewport" content="width=device-width, initial-scale=1">\n<title>{C.esc(TITLE)}</title>\n'
            f'<meta name="description" content="{C.esc(description)}">\n<link rel="icon" href="../favicon.svg" type="image/svg+xml">\n<link rel="icon" href="../favicon.ico" sizes="any">\n'
            f'<link rel="apple-touch-icon" href="../apple-touch-icon.png">\n<style>{C.CSS}{C.EXTRA_CSS}{A.CSS}{CSS}</style>\n</head>\n<body>\n')


def body(G, n):
    C, A, points, NAME = G.C, G.A, G.points, G.NAME
    start, best, passing, total, minutes = n["start"], n["best"], n["passing"], n["total"], n["minutes"]
    pct = lambda a, b: f"{100 * (a / b - 1):.0f}%"
    ch = G.CHANGED
    word = ["zero", "one", "two", "three", "four", "five", "six", "seven", "eight", "nine", "ten", "eleven", "twelve", "thirteen", "fourteen"]
    Word = [w.capitalize() for w in word]
    n_probes = len(G.PROBES)
    told = sum(announced for _, _, _, announced in G.PROBES)
    q = lambda v: f"{v:,.0f}"
    t = {s: G.tokens[s]["total"] for s in points}
    row = lambda s, i: points[s][i]
    struck = {s: [p for p in points[s] if p["struck"]] for s in points}
    struck_best = {s: max([p["qps"] for p in struck[s]] or [0]) for s in points}
    struck_pass = {s: max([p["qps"] for p in struck[s] if not p["failed"]] or [0]) for s in points}
    counted = {s: [p for p in points[s] if p["counts"]] for s in points}
    maps = [p["infer"]["compiled_bfloat16"]["mAP@0.5"] for s in points for p in points[s] if p["infer"]]
    eager_maps = [p["infer"]["eager_bfloat16"]["mAP@0.5"] for s in points for p in points[s] if p["infer"]]
    safe_later = [p["qps"] for p in points["vm"] if p["kind"] == "safe" and p["i"] > 4]
    fast = [p for p in points["vm"] if p["kind"] == "fast"]
    fast_later = [p["qps"] for p in fast if p["qps"] != best["vm"] and p["i"] > 5]
    unchecked = [p for p in points["tc"] if p["kind"] == "unchecked" and p["eligible"]]
    lost = lambda ps: sorted({len(p["failed"]) for p in ps})
    n_states = sum(p["eligible"] for s in points for p in points[s])
    n_pass = {s: sum(G.holds(p, True) for p in points[s]) for s in points}
    first_pass = {s: next(p for p in points[s] if G.holds(p, True) and p["qps"] == passing[s]) for s in points}
    first_best = {s: next(p for p in points[s] if p["counts"] and p["qps"] == best[s]) for s in points}
    # The fast states of torch.vm, by what the probes of kept results found, and how many of a result's tensors later calls changed.
    long_name = {short: name for name, short, _, _ in G.PROBES}
    handed = [p for p in fast if "results kept through DLPack" in p["failed"]]
    keep_only = [p for p in fast if "results kept through DLPack" not in p["failed"]]

    def changed(states, short):
        counts = {(p["raw"][long_name[short]]["tensors"], p["raw"][long_name[short]]["unchanged"]) for p in states if p["raw"][long_name[short]] and "tensors" in p["raw"][long_name[short]]}
        if len(counts) != 1:
            return "tensors"
        n, same = next(iter(counts))
        return f"all {word[n]} tensors" if same == 0 else f"{word[n - same]} of the {word[n]} tensors"

    if not G.INCOMPLETE:
        # The sentences below name what is missed; stop if the files say something else.
        expect = [(unchecked, 9), (handed, 2), (keep_only, 1)]
        for states, count in expect:
            if not states or lost(states) != [count]:
                raise SystemExit(f"text.py: the probes no longer say what the page's sentences say: {[(p['i'], p['failed']) for p in states]}")
        if any(set(p["failed"]) != {"write through .data", "results kept through DLPack"} for p in handed) or any(p["failed"] != ["write through .data"] for p in keep_only):
            raise SystemExit("text.py: torch.vm's fast states miss other probes than the page says")
    tests_wide = word[sum(1 for p in points["tc"] if p["own_tests"] and p["own_tests"]["ran"] > 1)]
    tests_one = word[sum(1 for p in points["tc"] if p["own_tests"] and p["own_tests"]["ran"] == 1)]
    final_note = ""
    misc, vm_cpu = G.final_tests("tests_tc_misc.log"), G.final_tests("tests_vm_cpu.log")
    if misc and vm_cpu:
        parts = []
        if vm_cpu["bad"] and vm_cpu["bad"] == points["vm"][0]["bad"]:
            parts.append(f"The {word[len(vm_cpu['bad'])]} errors of <code>torch.vm</code>'s file are the {word[len(vm_cpu['bad'])]} tests that error on every one of its sources.")
        if misc["bad"] == ["test_debugmode_respects_custom_op_aliasing_env_override"] and G.MISC_IS_SETUP:
            parts.append("The one failure in <code>test_misc.py</code> is <code>test_debugmode_respects_custom_op_aliasing_env_override</code>, the test that failed in the first run as well, on its starting tree too. "
                         "It starts a Python subprocess, and by the log that subprocess imports <code>torch</code> from the shared checkout instead of the side's tree and stops at a missing library: "
                         "the experiment's tree setup, before any of the side's code runs. <code>torch.compile</code>'s agent reports the same failure on the unmodified tree of this run; that tree was not run for this page.")
        final_note = "<p>" + " ".join(parts) + "</p>" if parts else ""
    incomplete = ""
    if G.INCOMPLETE:
        incomplete = ('<div class="note"><p><b>Checks still running when this page was made.</b> For these states one or more checks had no result yet; they are counted as passing as long as no check says otherwise: '
                      + "; ".join(f"{NAME[s]} row {i} ({', '.join(m)})" for s, i, m in G.INCOMPLETE) + ".</p></div>")
    tiles = C.tiles([
        ("torch.compile, six hours of one agent", f"{q(start['tc'])} → {q(best['tc'])}",
         f"images/s, best recorded; ${total['tc']:.0f} of tokens at list price. Passing every check: {q(passing['tc'])}. Six rows are struck, among them {q(struck_best['tc'])} and {q(struck_pass['tc'])}."),
        ("torch.vm, six hours of one agent", f"{q(start['vm'])} → {q(best['vm'])}",
         f"images/s, best recorded; ${total['vm']:.0f} of tokens. Passing every check: {q(passing['vm'])}, measured once; the same configuration later: {q(min(safe_later))} to {q(max(safe_later))}."),
        ("The start, torch.vm over torch.compile", f"{FIRST['start']['vm'] / FIRST['start']['tc']:.2f}x → {start['vm'] / start['tc']:.2f}x",
         f"First run: {q(FIRST['start']['vm'])} against {q(FIRST['start']['tc'])} images/s. This run, after the rewrite: {q(start['vm'])} against {q(start['tc'])}."),
        ("Predictions, real weights and images", f"{min(maps):.3f} to {max(maps):.3f}",
         f"mAP@0.5 of the {len(maps)} recorded states that were run on 128 COCO images. Eager in the same precision, in the same processes: {min(eager_maps):.3f} to {max(eager_maps):.3f}."),
    ])
    named = [("tc", 1, "torch.compile: CUDA graphs, an option"), ("vm", 2, "torch.vm: its CUDA graph started ahead of its checks"), ("tc", 4, "torch.compile: weights built in, guards dropped")]
    by_time = C.figure("Throughput against the agent's time",
                       "TorchBench yolov3 inference, batches of 8 images in bfloat16 on one H100. Every mark is an official measurement (median of three processes). For each compiler the solid line is the best recorded so far and the dashed line the best so far among states that pass every check; struck rows count for neither. The y axis starts at 1,500 images/s.",
                       G.LEGEND + G.step_chart("minutes", "minutes of agent time", 370, [0, 60, 120, 180, 240, 300, 360], lambda v: f"{v}", labels=named))
    by_cost = C.figure("The same against what the agent had spent",
                       f"x: the agent's tokens at list price. Over the six hours torch.compile's agent spent ${total['tc']:.0f} and torch.vm's ${total['vm']:.0f}; the lines end there.",
                       G.LEGEND + G.step_chart("cost", "agent tokens spent, $ at list price", 70, [0, 10, 20, 30, 40, 50, 60, 70], lambda v: f"${v}", labels=named))
    compare = G.table(["images/s", "first run, six hours", "this run, six hours"], [
        ["torch.compile, start", q(FIRST["start"]["tc"]), q(start["tc"])],
        ["torch.compile, best recorded", q(FIRST["best6"]["tc"]), f"{q(best['tc'])} (struck: {q(struck_best['tc'])})"],
        ["torch.compile, best passing the checks of its run", q(FIRST["passing6"]["tc"]), f"{q(passing['tc'])} (struck: {q(struck_pass['tc'])})"],
        ["torch.vm, start", q(FIRST["start"]["vm"]), q(start["vm"])],
        ["torch.vm, best recorded", q(FIRST["best6"]["vm"]), q(best["vm"])],
        ["torch.vm, best passing the checks of its run", q(FIRST["passing6"]["vm"]), q(passing["vm"])],
    ], num=(1, 2))
    amp = {k: G.train_ms("amp_", k) for k in ("eager_a", "tc_final", "vm_final")}
    f32 = {k: G.train_ms("", k) for k in ("eager_a", "tc_final", "vm_final")}
    det = {k: G.train_ms("det_", k) for k in ("eager_a", "tc_final", "vm_final")}
    train_speed = ""
    if all(amp.values()) and all(f32.values()) and all(det.values()):
        train_speed = (f"<p><b>Speed.</b> Training was not what the agents worked on, and it shows: <code>torch.vm</code>'s final source gains little on this model in training. With default kernels in float32 a step takes "
                       f"{f32['vm_final']:.0f} ms against eager's {f32['eager_a']:.0f} and <code>torch.compile</code>'s {f32['tc_final']:.0f}; under autocast {amp['vm_final']:.0f} against {amp['eager_a']:.0f} and {amp['tc_final']:.0f}; "
                       f"with deterministic kernels all three are within 5% of each other ({det['vm_final']:.0f}, {det['eager_a']:.0f}, {det['tc_final']:.0f}). The reason is in the last column: in training mode the engine does not take the model's forward down as one graph. "
                       "It compiles two small graphs (16 operations in float32), as it did with the starting source, and runs the rest kernel by kernel. <code>torch.compile</code> compiles the forward and the backward.</p>")
    return f"""<h1>{TITLE}</h1>
<p class="sub">The experiment of the two earlier pages, repeated from the start after torch.vm's graph compiler was rewritten: two new instances of the same AI agent, six hours each, one improving torch.compile and one torch.vm on one model. Throughput against the time and tokens spent, every recorded state checked on real weights, and two numbers per side: the best recorded, and the best that passes every check. Nov 21, 2026.</p>
<p><a href="../two-compilers-one-model/">The first page</a> gave two instances of the same AI coding agent three hours each to raise the throughput of TorchBench's yolov3 in inference, one through <code>torch.compile</code> and one through
<a href="../cfg-compile/"><code>torch.vm</code></a>, a prototype compiler for PyTorch whose engine is written in Rust, and plotted throughput against the time and tokens spent. <a href="../two-compilers-six-hours/">The second</a> gave the same two agents three more hours.
This page is a second run of the whole experiment: the same model, harness and rules, six hours in one go, two fresh agents that had seen nothing of the first run, and a <code>torch.vm</code> whose graph compiler had been rewritten in between.
As before, each agent could change its compiler's options and its source; the model and the measurement were not theirs to touch, every gain was recorded by a harness neither could change, and every recorded state was checked again after the deadline.</p>
<div class="tldr"><p><b>In short.</b> The two compilers now start close together: <code>torch.compile</code> at {q(start['tc'])} images/s and <code>torch.vm</code> at {q(start['vm'])} (eager does {G.EAGER:,.0f}; in the first run it was {q(FIRST['start']['tc'])} against {q(FIRST['start']['vm'])}).
<code>torch.compile</code> moved first, by switching CUDA graphs on ({q(row('tc', 1)['qps'])} after {row('tc', 1)['minutes']:.0f} minutes). <code>torch.vm</code>'s agent had to build them, was ahead after {row('vm', 2)['minutes']:.0f} minutes ({q(row('vm', 2)['qps'])})
and was at {q(row('vm', 4)['qps'])} after {row('vm', 4)['minutes']:.0f}, when <code>torch.compile</code> had {q(row('tc', 3)['qps'])}. After that neither line moved much.
The page reports two numbers per side. Best recorded: <code>torch.compile</code> {q(best['tc'])}, <code>torch.vm</code> {q(best['vm'])}. Best recorded among states that pass every check: <code>torch.vm</code> {q(passing['vm'])}, <code>torch.compile</code> {q(passing['tc'])}.
So the expectation these charts were first drawn for, a higher start for <code>torch.compile</code> and a steeper slope for <code>torch.vm</code>, holds in this run only in a weak form: the start is {pct(start['tc'], start['vm'])} higher,
the slope is steeper for the first 80 minutes, and the lead it buys is {pct(passing['vm'], passing['tc'])} on the checked line and is reversed on the other.</p>
<p>Three things qualify the end points. Six <code>torch.compile</code> rows are struck, its two highest among them ({q(struck_best['tc'])} and, of the checked ones, {q(struck_pass['tc'])}): its agent introduced a test failure,
found it itself after about 3 hours 45 minutes and fixed it, and the rule was that a state whose source breaks its side's tests does not count. <code>torch.vm</code>'s {q(passing['vm'])} was measured once;
the same configuration gave {q(min(safe_later))} to {q(max(safe_later))} in nine later rows, and its agent did not find out why. And the two best-recorded states give up different amounts. <code>torch.compile</code>'s misses {word[len(first_best['tc']['failed'])]} of the {word[n_probes]} probes: it follows no in-place weight update except one,
no replaced parameter, no <code>model.train()</code> and no edited module attribute, and a result kept across calls is overwritten. <code>torch.vm</code>'s misses {word[len(first_best['vm']['failed'])]}: a weight written through <code>.data</code>,
and a result held by its raw address across later calls, which is the hole its agent had reported. Neither agent wrote a convolution kernel of its own, which is what carried <code>torch.vm</code> past 4,500 in the first run; both ended at the speed of the kernels that Inductor and cuDNN provide.</p></div>
{incomplete}
{tiles}
{by_time}
{by_cost}
<h2 id="changed">What changed since the first run</h2>
<p>In the first run, <code>torch.vm</code>'s graph compiler was Python code. It took a stretch of tensor code that the VM had run, traced it a second time on fake tensors (tensors that have a shape and a type but no data),
and handed the graph it got to AOTAutograd and Inductor, the backend <code>torch.compile</code> uses. Whatever the trace read on its way, a flag or an attribute of a module, was fixed into the graph, and a separate check of the modules decided whether a later call could use it.</p>
<p>Now graphs are built inside the Rust engine, from the path a call actually took on real tensors: its instructions in the order that call ran them, through the branches as they went and round the loops as often as they went round.
Every value Python read on the way is read again and compared on every call. AOTAutograd and Inductor are no longer the compiler but optional passes over the graph the engine exports;
without them the engine runs the graph's kernels itself and the results are eager's bit for bit. The stack is described in its top pull request, <a href="{PR}">pytorch/pytorch#204763</a>.</p>
<p>Two consequences are on this page before either agent did anything. <code>torch.vm</code> starts at {q(start['vm'])} images/s where it started at {q(FIRST['start']['vm'])}: the model is one graph of 328 operations out of the box,
where the first run began with a refused store into a module's buffer and the model running kernel by kernel. And an edit of a module attribute that is not a tensor (the slope of a LeakyReLU),
which every one-graph <code>torch.vm</code> state of the first run missed, is followed by every <code>torch.vm</code> state of this run: the slope is one of the values that are read again on each call.</p>
<p>Also different in this run:</p>
<ul>
<li>Six hours in one go, not two halves, with fresh agents. They were told not to read the first run's directory or anything published about it.</li>
<li>The checks were announced from the start, with a command that runs them on an agent's current state, and both agents knew that the report would give two numbers per side.
In the first run the checks came after the first three hours.</li>
<li>Each side worked on its own copy of the source tree, with eight CPU cores of its own.</li>
<li>Other work ran on other cores of the same machine during the run. It used no GPU.</li>
</ul>
<h2 id="show">What the charts show</h2>
<ul>
<li><b>The start.</b> <code>torch.compile</code> is {pct(start['tc'], start['vm'])} ahead ({q(start['tc'])} against {q(start['vm'])}), where it was 2.6 times ahead. Both now begin with the whole model as one compiled graph,
and both agents' first profiles say the same thing in almost the same numbers: about 4 ms of CPU time per call to launch 330 kernels that need 2.5 ms of the GPU.</li>
<li><b>The first step is the same step.</b> Both went to CUDA graphs. For <code>torch.compile</code> that is an option, recorded after {row('tc', 1)['minutes']:.0f} minutes. <code>torch.vm</code> had none, and its agent wrote one in {row('vm', 1)['minutes']:.0f} minutes ({q(row('vm', 1)['qps'])}).
What put <code>torch.vm</code> ahead eight minutes later is specific to it: its checks of a call cost more than <code>torch.compile</code>'s guards (0.54 ms against 0.2), and instead of thinning them out its agent
<a href="#vm-2">started the GPU's work before them</a>, so that they run while the kernels do.</li>
<li><b>The slope.</b> On states that pass every check, <code>torch.vm</code> gained {q(round(passing['vm']) - round(start['vm']))} images/s in its first {first_pass['vm']['minutes']:.0f} minutes and ${first_pass['vm']['cost']:.0f}, and nothing after.
<code>torch.compile</code> gained {q(round(row('tc', 3)['qps']) - round(start['tc']))} in its first {row('tc', 3)['minutes']:.0f} minutes and ${row('tc', 3)['cost']:.0f}, and {q(round(passing['tc']) - round(row('tc', 3)['qps']))} more by the end.
From {row('vm', 2)['minutes']:.0f} minutes on <code>torch.vm</code> is ahead on that line at every time and every budget, by {pct(passing['vm'], passing['tc'])} at the end; measured against the level its configuration held later (about 3,150), by 4.5%.</li>
<li><b>The best recorded line is the other way round.</b> Both agents kept a second, faster configuration that gives up things eager does, and said so in each row's note. <code>torch.compile</code>'s reached {q(row('tc', 4)['qps'])} after {row('tc', 4)['minutes']:.0f} minutes
and {q(best['tc'])} at the end; <code>torch.vm</code>'s reached {q(row('vm', 5)['qps'])} after {row('vm', 5)['minutes']:.0f} and {q(best['vm'])}. <code>torch.compile</code> ends {pct(best['tc'], best['vm'])} higher, and gives up far more for it (<a href="#pass">below</a>).</li>
<li><b>Both lines are flat after an hour and a half.</b> The steps after that are a few percent each and many are inside the noise. Both agents were then bound by the same kernels: cuDNN's convolutions (1.5 ms per call) and the generated
batch-norm-and-LeakyReLU kernel behind each of them (0.5 to 0.6 ms). In the first run <code>torch.vm</code>'s agent replaced that pair with one cuDNN kernel through an interface PyTorch does not expose; its line stood at 3,261 before that kernel and ended at 4,614.
In this run <code>torch.compile</code>'s agent named that fusion as what would help and did not build it, and <code>torch.vm</code>'s tried a generated kernel for the first convolution, found it four times slower and stopped there.
The steep second half of the first run was one agent's kernel, not a property of the compiler it was written for.</li>
<li><b>Per token.</b> <code>torch.compile</code>'s agent spent more: ${total['tc']:.0f} against ${total['vm']:.0f}, with about the same output ({t['tc']['output']/1e3:.0f}k and {t['vm']['output']/1e3:.0f}k tokens) and more re-reading of its own history
({t['tc']['cache_read']/1e6:.0f} against {t['vm']['cache_read']/1e6:.0f} million cached input tokens, over {G.tokens['tc']['responses']} against {G.tokens['vm']['responses']} responses). At equal dollars the picture is the one at equal minutes.</li>
</ul>
{G.ahead_table()}
<p>And next to the first run. The two columns are not the same test: the first run's agents were a different pair making different choices, its checks were fewer (no attribute probe and none of the three unannounced ones), and by the stricter count used on its second page <code>torch.vm</code> had no passing state above its start, because of the attribute edit it now follows.</p>
{compare}
<h2 id="rows">Every recorded measurement</h2>
<p>Time and dollars are the agent's own at the moment of the measurement; the dollars are its tokens at Anthropic's list price for the model ($8 per million input, $40 output, $10 cache write, $0.20 cache read; what a cloud provider charges may differ).
The two start rows were measured before the agents began. Each step's name links to what it is, in the <a href="#appendix">appendix</a>. The mAP column is the state's predictions with the pretrained weights next to eager's in the same process;
the last column is from the checks made after the deadline, explained in the <a href="#pass">next section</a>.</p>
<h3>torch.compile: {q(start['tc'])} to {q(best['tc'])}; passing every check, {q(passing['tc'])}</h3>
<p>{Word[len(ch['tc'])]} source files changed, {sum(ch['tc'].values())} lines added: Inductor's lowering of max pools ({ch['tc']['torch/_inductor/lowering.py']}) and of convolutions ({ch['tc']['torch/_inductor/kernel/conv.py']}), Dynamo's guards ({ch['tc']['torch/_dynamo/guards.py']}) and Inductor's CUDA-graph code ({ch['tc']['torch/_inductor/cudagraph_trees.py']}). The configuration is {G.CONFIG_LINES['tc']} lines.
The agent alternated between two configurations, the checked one (rows 0 to 3, 5, 8 to 10, 14 and 16) and unchecked variants of it.</p>
{G.steps_table('tc')}
<h3>torch.vm: {q(start['vm'])} to {q(best['vm'])}; passing every check, {q(passing['vm'])}</h3>
<p>{Word[len(ch['vm'])]} source files changed, {sum(ch['vm'].values()):,} lines added: a new module of {ch['vm']['torch/vm/_cudagraphs.py']} lines that runs what Inductor compiled as a CUDA graph, {ch['vm']['torch/vm/passes.py']} lines in the passes, {sum(v for k, v in ch['vm'].items() if k.startswith('rust/'))} lines of Rust in the engine, and {ch['vm']['test/test_vm.py']} lines of tests. The configuration is {G.CONFIG_LINES['vm']} lines.
From row 5 on the agent alternated between a safe and a fast configuration and measured each again after every change; the many rows at the same height are those repeats.</p>
{G.steps_table('vm')}
<h2 id="pass">Which states pass every check</h2>
<p>The rules both agents had named the checks and added: "Other things eager does that are not listed here. A state that is found not to do one of them does not pass." On this page a state passes every check when</p>
<ul>
<li>the harness counted its measurement (outputs equal to eager's on the benchmark's inputs, its three probes);</li>
<li>its predictions with the pretrained weights are eager's (mAP@0.5 within 0.01);</li>
<li>it follows all {word[told]} announced probes and the {word[n_probes - told]} that were not announced;</li>
<li>its source passes its side's tests. A row whose source fails one that the starting source passes is struck, as the rules said it would be.</li>
</ul>
<p>Two more checks were run on four states only, each side's highest row and its last: the compile cache filled by a model with other weights, and the speed after events. They are reported <a href="#checks">below</a>; none of the four fails the first.
By this count {n_pass['tc']} of <code>torch.compile</code>'s {len(points['tc'])} rows pass and {n_pass['vm']} of <code>torch.vm</code>'s {len(points['vm'])}.</p>
<h3 id="follow">What the states follow</h3>
<p>With the pretrained weights, after the callable is warm, one thing is done to the callable's model and to an eager copy, and the two outputs are compared; then it is undone and they are compared again.
For the last two probes the caller keeps every tensor of one call's result and looks at them again after two calls with other inputs. The table has one column for each group of states that came out alike.</p>
<p>The three probes that were not announced are a second version. In the first, the two probes of kept results held only the first tensor of a result, and their last call repeated the input of the call whose result was kept,
so memory that had been overwritten compared equal again. Both were too weak and were replaced before this page was written; the first version's script and outputs are kept with the results and are not used here.</p>
{G.probe_table()}
<p><b>What <code>torch.compile</code>'s unchecked variants give up.</b> {Word[lost(unchecked)[0]]} of the {word[n_probes]} probes, the same {word[lost(unchecked)[0]]} on all {len(unchecked)} states.{' (They differ between states: see the rows.)' if len(lost(unchecked)) > 1 else ''} Their weights and batch-norm statistics are constants of the graph, so an in-place update of a convolution's weight, a batch norm's bias or a running mean is not followed;
the one weight update that is followed is the last convolution's bias, which nothing is folded into. Their guards cover only the call's arguments (in the first variant, only tensors), so a replaced parameter, <code>model.train()</code> and the edited slope go unnoticed,
and after <code>model.train()</code> and <code>model.eval()</code> the callable stays wrong. A write through <code>.data</code> is not followed either. And their results are handed out in memory that later calls write again: kept across two calls with other inputs, as tensors or through DLPack,
{changed(unchecked, 'results kept as tensors')} of a result hold other values afterwards.
The agent's own notes on these rows say all of this except the <code>.data</code> write, which follows from the first item.</p>
<p><b>What <code>torch.vm</code>'s fast configuration gives up.</b> {Word[lost(handed)[0]]} of the {word[n_probes]} probes on the {word[len(handed)]} states that hand large results out without a copy (rows {G.rows_text([p['i'] for p in handed])}),
and {word[lost(keep_only)[0]]} on the state that only keeps the weight layouts (row {G.rows_text([p['i'] for p in keep_only])}).
On all {word[len(fast)]}, a 3x3 convolution's weight written through <code>.data</code> is not followed, because the kept channels-last copy of the weight is refreshed only when the weight's version counter moves.
On the {word[len(handed)]}, a result held by its raw address, through a DLPack capsule, is changed by later calls: {changed(handed, 'results kept through DLPack')} of a result, by their sizes the two raw head outputs above the option's threshold of 1 MB,
which are handed out on the recording's memory. That is the hole its agent reported for these states. Held as tensors, the same results stay what they were: before the recording runs again, a handed-out tensor that somebody still holds is given memory of its own.
Everything else is followed, including the in-place updates and the attribute edit.</p>
<p><b>The checked and the safe lines</b> follow all {word[n_probes]}. What their agents say they still give up is in the section on <a href="#own">what the agents found</a>.</p>
<h3 id="struck">Struck rows</h3>
<p>After the deadline each side's tests were run on the source of every recorded row. For <code>torch.compile</code> that is the Inductor GPU tests of convolutions, max pools and cat together with the stride test (471 tests) on {tests_wide} rows,
and on the other {tests_one} the stride test alone, the one test that tells its sources apart. For <code>torch.vm</code> it is its test file on the CPU, from each row's own tree. The table of sources below says what ran on which.</p>
{G.struck_table()}
<p>That is what <code>torch.compile</code>'s agent reported about itself in its summary, row for row. The failure came in with <a href="#tc-5">step 5</a> and went out with <a href="#tc-8">step 8</a>; the sources before and after pass.
With these rows struck, <code>torch.compile</code>'s best recorded row is {q(best['tc'])} instead of {q(struck_best['tc'])}, and its best that passes every check is {q(passing['tc'])} instead of {q(struck_pass['tc'])}.
The corrected source compiles yolov3 to the same graph as the struck one, and its agent reads the differences ({q(struck_pass['tc'])} against {q(passing['tc'])}, {q(struck_best['tc'])} against {q(best['tc'])}) as noise between runs. The struck rows stay on the charts, as crosses.</p>
<p>No <code>torch.vm</code> row is struck. Two tests of its file error on every one of its sources, the starting one included: they build a pass written in Rust and need a directory that this experiment's copy of the tree does not have.</p>
{G.tests_table()}
<h2 id="checks">The other checks</h2>
<h3>Predictions</h3>
<p>The benchmark runs the model with random weights, and with those the output hardly depends on the convolutions, so the harness's comparison with eager says little. This check loads the original pretrained yolov3-spp weights into the same model
and runs the 128 COCO images that ship with the benchmark, letterboxed to the benchmark's 384x512 and in its batches of 8.</p>
{G.acc_table()}
<p>All {len(maps)} recorded states that were run through this have a mAP@0.5 between {min(maps):.4f} and {max(maps):.4f}; eager in bfloat16, in the same processes, has {min(eager_maps):.4f} to {max(eager_maps):.4f} (it differs from process to process in the fourth digit).
The value for each state is in the tables above. The unchecked <code>torch.compile</code> variants are a little closer to float32 than eager's bfloat16 is, as they should be where batch norms are folded in float32 before the cast.</p>
<h3>A compile cache filled by another model</h3>
<p>In the first run a <code>torch.compile</code> change let a model compute with a weight of another model that had been compiled before it, through Inductor's cache on disk. So this check compiles a state first for the benchmark's random-weight model and then,
with the same cache, for the pretrained one, which must compute with its own weights. It was run on each side's highest row and its last.</p>
{G.cache_table()}
<h3>Speed after events</h3>
<p>The callable is timed in the benchmark's setup, then again after a batch of another size, after images of another size, and after a weight was updated in place. One process per state.</p>
{G.after_table()}
<p>All four keep their speed. That was not so at the start of <code>torch.compile</code>'s line: by its agent's own runs of this script, the dashboard's configuration and its first two rows lose about half their speed after an input of another shape,
until <a href="#tc-3">step 3</a> gave every shape a graph of its own.</p>
<h3>Tests on the final sources</h3>
{G.final_tests_table() or "<p>The runs of each side's test suites on its final source had not finished when this page was made; the tests per recorded row are in the table of struck rows above.</p>"}
{final_note}
<h3 id="training">Training</h3>
<p>The experiment was inference. This checks that the compilers, as their agents left them, still train the model as eager does: yolov3-spp from a seeded random start, the model's own detection loss and SGD settings, the 128 images in a fixed order, 300 steps of 16.
The forward, and so the backward, goes through eager, <code>torch.compile</code> or <code>torch.vm</code> with Inductor graphs, each as the dashboard would configure it, on the side's starting and final source.</p>
{G.loss_figure()}
{G.train_table()}
<p><b>Loss.</b> With deterministic kernels two eager runs are equal bit for bit, so the distances in the first block are exact. The compiled curves leave eager's at the first or second step and then stay as close to it as eager's own curve does
when one layer's weights are changed in the sixth digit: this network, at a random start with batch norm in training mode, amplifies a last-digit difference to a percent within ten steps.
<code>torch.compile</code>'s final source gives the same 300 losses as its starting source, to the last bit. With default kernels two eager runs already differ from each other by as much as the compiled runs differ from eager.</p>
{train_speed}
<h2 id="own">What the agents found about their own work</h2>
<p>Both agents' summaries say, near the top, what is wrong with their own numbers. These are their findings, not this page's; where a check here bears on one, it says so.</p>
<ul>
<li><b><code>torch.compile</code>: a late regression.</b> Six rows were recorded on a source that fails one Inductor test, three of the checked line and three unchecked. The agent had run the tests whose names match what it changed; it found the failure when it ran the whole GPU suite,
fixed it, measured both configurations again and listed the six rows as not to be used. The checks here confirm each of them. Its note for next time: run the whole suite before every measurement of a source change.</li>
<li><b><code>torch.compile</code>: the final state was measured twice,</b> and the second measurement was started because the first read low. The agent wrote that down before the second result came, and asks for both to be read as samples of one state: {q(row('tc', 14)['qps'])} and {q(row('tc', 16)['qps'])}.</li>
<li><b><code>torch.compile</code>: one thing its checked line does not follow.</b> With the dictionary tags of <a href="#tc-2">step 2</a>, a new <code>__class__</code> assigned to a module nested inside another is not noticed. The agent found it with a small model on the CPU, in the last hour,
and traced it to the setting, not to its own change. No check here tests it. Under the rule quoted above it would count against every checked row from {q(row('tc', 2)['qps'])} on, and the fastest <code>torch.compile</code> row with nothing known against it would be {q(row('tc', 1)['qps'])}.
The page does not apply that, because it has not measured it.</li>
<li><b><code>torch.compile</code>: speed after <code>model.train()</code> and back.</b> About 10% lower for the rest of the process on the final checked configuration (one process). The speed-after-events check here has no such event.</li>
<li><b><code>torch.compile</code>: one row partly spoiled.</b> The third process of the {q(row('tc', 2)['qps'])} row ran while a source file was being edited. The median is one of the two clean runs.</li>
<li><b><code>torch.vm</code>: the {q(passing['vm'])} that did not come back.</b> The snapshot of that row, run again later, gave 3,125, and the same configuration on later sources {q(min(safe_later))} to {q(max(safe_later))}. Temperature and the compile cache were ruled out. The cause was not found.
It is the number on this page's checked line because it is what was recorded and the state passes; the level to expect from that configuration is about 3,150.</li>
<li><b><code>torch.vm</code>: one note that says too little.</b> The {q(best['vm'])} row was measured with large results handed out without a copy in every case, before that became an option, and its note names only the <code>.data</code> hole. The agent's journal and summary name both.</li>
<li><b><code>torch.vm</code>: journal times.</b> From 33 minutes on the agent stamped its journal entries by estimate, and the estimates ran ahead, by three hours at the end. It corrected them from the logs before the deadline and says so at the top of the journal.
The times on this page are the harness's.</li>
<li><b>Noise.</b> Both agents put it at 1 to 2.5% between runs of the same state. In the logs, the three processes of one row are {G.SPREAD['tc'][0]:.1f}% (<code>torch.compile</code>) and {G.SPREAD['vm'][0]:.1f}% (<code>torch.vm</code>) apart at the median.
The widest are the two start rows ({G.SPREAD['tc'][1]:.1f}% and {G.SPREAD['vm'][1]:.1f}%); after them, {G.SPREAD['tc'][2]:.1f}% and {G.SPREAD['vm'][2]:.1f}%. Between two measurements of the same state the agents saw 2 to 3%. <code>torch.compile</code>'s agent lists which of its steps are clearly outside that: +32.6%, +6.4%, +7.5% and +2.7%. The rest of its checked line, and every difference between two rows of the same configuration on either side, is not.</li>
</ul>
<h2 id="limits">What this does and does not show</h2>
<ul>
<li><b>One model, one run per side.</b> An example of two paths, not a measurement of two slopes. Run again, either agent might have written a kernel, or missed a step it found here, and the end points would differ by more than the gaps on this page.</li>
<li><b>It shows where the rewrite put the start, and little about the ceiling.</b> The start moved from 0.39 to 0.93 of <code>torch.compile</code>'s, and what the first run's agent spent its first hour on is no longer needed. Past 3,000 images/s both compilers hand the same graph to the same backend and wait for the same kernels;
the differences there are small, partly noise, and partly a choice of what to give up.</li>
<li><b>"Passes every check" is as strong as the checks.</b> Thirteen probes, one model, one input pipeline. <code>torch.compile</code>'s agent found something about its own checked line that no probe here tests, and the first version of two of the unannounced probes saw less than it was written for and had to be replaced.
A state that passes is not thereby proven equal to eager.</li>
<li><b>The two "best recorded" numbers are not the same kind of state.</b> {q(best['tc'])} is a frozen model behind an input-only guard; {q(best['vm'])} is a model that is still followed in everything the probes tried but {word[len(first_best['vm']['failed'])]} things. Comparing them as speeds says little without the table of probes next to them.</li>
<li><b>The agent knew one compiler and not the other.</b> The model has read <code>torch.compile</code>'s source in training and had never seen <code>torch.vm</code>'s; it was given the tree and its documentation.</li>
<li><b>Not on the x axis:</b> the tokens that built <code>torch.vm</code> up to this start, the rewrite included, and those of the session that set this up and checked it.</li>
<li><b>Training is not covered by the result.</b> On this model <code>torch.vm</code>'s graphs do not yet reach the training forward as a whole, and nothing in these six hours changed that.</li>
</ul>
<h2 id="how">How it was run</h2>
<ul>
<li>Model and setup as the PyTorch performance dashboard has them: eval mode, batches of 8 images of 3x384x512, bfloat16, <code>no_grad</code>, one H100. Each side starts from the dashboard's configuration:
<code>torch.compile</code> of a function that calls the model, and <code>torch.vm</code> with its default passes, <code>Graphs()</code> and <code>Inductor()</code>.</li>
<li>Each agent may change its compiler's options and Python source (and Rust, for <code>torch.vm</code>), in its own copy of the tree; not the model, not the harness, not eager, no C++ build. Neither may use the other's compiler, read the other's directory, or read the first run.</li>
<li>A measurement: the state is snapshotted as a commit, three processes each warm the callable up (at least 300 calls and 20 s) and time 400 calls on four different inputs; the median counts if the outputs match eager's
by the dashboard's rule for bfloat16 and the harness's probes pass (a weight updated in place, another batch size, the timed shape again). Measurements are refused after the deadline, six hours after the start.</li>
<li>One GPU for both sides, behind a lock, so each side's GPU jobs waited for the other's. Other, CPU-only work ran on other cores of the machine during the run.</li>
<li>The checks on this page were made after the deadline by the session that set the experiment up, on the snapshot of every row the harness had counted: predictions, the ten announced probes, three that were not announced,
and each side's tests; the compile-cache check and the speed after events on each side's highest and last row; training on the starting and final sources. Each agent could run the announced ones on its own state during the run, and did
({G.CHECK_RUNS['tc']} and {G.CHECK_RUNS['vm']} times).</li>
<li>Tokens are read from each agent's transcript, by the timestamp of each response: ${total['tc']:.2f} for <code>torch.compile</code>'s agent over {minutes['tc']:.0f} minutes ({t['tc']['output']/1e3:.0f}k output tokens, {t['tc']['cache_read']/1e6:.0f} million cached input tokens read)
and ${total['vm']:.2f} for <code>torch.vm</code>'s over {minutes['vm']:.0f} ({t['vm']['output']/1e3:.0f}k and {t['vm']['cache_read']/1e6:.0f} million).</li>
<li>One slip of isolation, from <code>torch.compile</code>'s journal: a process listing it ran to make sure none of its own jobs was left printed five command lines of the other side's GPU job. It says it opened no file of theirs and did not act on it.</li>
</ul>
{A.APPENDIX}
"""
