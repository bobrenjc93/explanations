"""text6.py: the prose of the six-hour page. body(G, n) gets make6's module and its headline numbers."""
FIRST_HALF_COST = {"tc": 29.94, "vm": 40.69}  # what each agent had spent when the first half ended


def body(G, n):
    C, A, M, points = G.C, G.A, G.M, G.points
    b3, b6, p3, p6, total = n["b3"], n["b6"], n["p3"], n["p6"], n["total"]
    second = {s: total[s] - FIRST_HALF_COST[s] for s in total}
    gap = lambda a, b: f"{100 * (a / b - 1):.0f}%"
    t = {s: M.tokens[s]["total"] for s in total}
    new = {s: [p for p in points[s] if p["half"] == 2] for s in points}
    maps = [p["infer"]["compiled_bfloat16"]["mAP@0.5"] for s in new for p in new[s] if p["infer"]]
    map_range = f"{min(maps):.3f} to {max(maps):.3f}" if maps else "…"
    tiles = C.tiles([
        ("Best recorded, torch.compile", f"{b3['tc']:,.0f} → {b6['tc']:,.0f}", "images/s after three and after six hours. No row of the second half is higher; the same configuration on the corrected source gives 3,861."),
        ("Best recorded, torch.vm", f"{b3['vm']:,.0f} → {b6['vm']:,.0f}", "images/s. One step, in the first 40 minutes of the second half."),
        ("Passing every check, torch.compile", f"{p3['tc']:,.0f} → {p6['tc']:,.0f}", f"images/s. torch.vm's states all pass: {p3['vm']:,.0f} → {p6['vm']:,.0f}."),
        ("The gap, torch.vm over torch.compile", f"{gap(b3['vm'], b3['tc'])} → {gap(b6['vm'], b6['tc'])}", f"on the best recorded. On what passes every check: {gap(p3['vm'], p3['tc'])} → {gap(p6['vm'], p6['tc'])}."),
    ])
    labels = [("tc", 19, "freezing that notices weight updates", 0, 75, "middle", True), ("tc", 22, "narrow guard filter, static shapes", 30, 115, "end", True),
              ("vm", 9, "fused cuDNN kernel", -8, -62, "end", True), ("vm", 18, "kernels started early", 0, -12, "middle", False)]
    by_time = C.figure("Six hours: throughput against the agent's time",
                       "TorchBench yolov3 inference, batches of 8 images in bfloat16 on one H100. Every mark is an official measurement (median of three processes). The dashed orange line counts only torch.compile states that pass every check; for torch.vm that line is the solid one.",
                       G.LEGEND + G.chart("minutes", "minutes of agent time", 370, [0, 60, 120, 180, 240, 300, 360], lambda v: f"{v}", labels=labels))
    by_cost = C.figure("The same against what the agent had spent",
                       f"x: the agent's tokens at list price. The second half starts at ${FIRST_HALF_COST['tc']:.0f} for torch.compile and ${FIRST_HALF_COST['vm']:.0f} for torch.vm, and cost ${second['tc']:.0f} and ${second['vm']:.0f}: more than the first, because every step of an agent re-reads a conversation that is now twice as long.",
                       G.LEGEND + G.chart("cost", "agent tokens spent, $ at list price", 100, [0, 20, 40, 60, 80, 100], lambda v: f"${v}"))
    gap_fig = C.figure("The gap: torch.vm's throughput over torch.compile's, at equal agent time",
                       "Above 1.0x torch.vm is ahead. Each line divides the two sides' best so far: once over all recorded states, once over those that pass every check.", G.gap_chart())
    return f"""<h1>Three more hours: does the gap between torch.compile and torch.vm widen?</h1>
<p class="sub">The same two AI agents continue for a second three hours on one model, one improving torch.compile and one torch.vm. The best recorded throughput barely moves; what moves is how much of torch.compile's speed holds up when it is checked. Nov 20, 2026.</p>
<p><a href="../two-compilers-one-model/">The first page</a> gave two instances of the same AI coding agent three hours each to raise the throughput of TorchBench's yolov3 in inference, one through <code>torch.compile</code> and one through
<a href="../cfg-compile/"><code>torch.vm</code></a>, and plotted throughput against the time and tokens spent. It ended at 3,973 against 4,507 images/s, with a caveat on each number. This page is the same experiment continued: the same two agents, resumed where they had stopped,
with the same rules and the same harness, for three more hours each. One thing was added, the same for both: each agent was told what the checks after the first half had found for its own side, was given a command that runs those checks on its current state,
and was told that this page would report two numbers per side, the best recorded throughput and the best recorded throughput among states that pass every check.</p>
<div class="tldr"><p><b>In short.</b> On the best recorded throughput the gap widened a little, from {gap(b3['vm'], b3['tc'])} to {gap(b6['vm'], b6['tc'])}: <code>torch.vm</code> went from {b3['vm']:,.0f} to {b6['vm']:,.0f} images/s with one step,
and <code>torch.compile</code> recorded nothing above its {b3['tc']:,.0f}. Both were already at the speed of the GPU's kernels, and neither agent built the next kernel it had named.
On the throughput that passes every check the gap closed by more than half, from {gap(p3['vm'], p3['tc'])} to {gap(p6['vm'], p6['tc'])}: <code>torch.compile</code> went from {p3['tc']:,.0f} to {p6['tc']:,.0f},
because its agent made freezing notice a weight update, which is the thing <code>torch.vm</code> had and it did not. On the way that agent found a bug in one of its own changes of the first half, which the checks on the first page had not caught:
through the compile cache, a model could compute with the first convolution's weight of another model. And a probe added for this page, which neither agent was told about, goes against <code>torch.vm</code>:
once the whole model is one graph it does not notice an edit of a module attribute that is not a tensor, which <code>torch.compile</code> with all its guards does. The second three hours cost more than the first (${second['tc']:.0f} and ${second['vm']:.0f} of tokens, after ${FIRST_HALF_COST['tc']:.0f} and ${FIRST_HALF_COST['vm']:.0f}) and bought less speed.</p></div>
{tiles}
{by_time}
{gap_fig}
{by_cost}
<h2>Does the gap widen?</h2>
<ul>
<li><b>On the best recorded number: slightly, and not because anything pulled away.</b> <code>torch.vm</code> gained 2% in the first 40 minutes (the replay of the recorded kernels is now started before the call has been checked, <a href="#v-early">step 1</a>) and nothing after.
<code>torch.compile</code>'s best is still the first half's row; the same configuration measured again after the bug fix gives 3,861. A call is 1.73 ms for <code>torch.vm</code>, 1.70 of it on the GPU, and about 2.0 ms for <code>torch.compile</code>.
What separates them is what separated them after three hours: <code>torch.vm</code>'s convolutions run through the fused cuDNN kernel its agent wrote in the first half, and <code>torch.compile</code>'s 3x3 convolutions still make a second pass over memory for the bias and the LeakyReLU.
Each agent had named its next kernel-level step (that fusion for <code>torch.compile</code>; one kernel for a 1x1 convolution and the 3x3 behind it for <code>torch.vm</code>). Neither started it.</li>
<li><b>On what passes every check: it narrowed, from {gap(p3['vm'], p3['tc'])} to {gap(p6['vm'], p6['tc'])}.</b> This is where <code>torch.compile</code>'s three hours went. After the first half, its only states that followed a weight update were the first two.
Its agent made freezing sound (<a href="#t-version">step 2</a>), then won back most of what the guards cost (<a href="#t-narrow">steps 4</a> and <a href="#t-static">6</a>): {p6['tc']:,.0f}, which is 94% of its fastest configuration.
What is left between the two lines is 0.11 ms of guards per call.</li>
<li><b>Told what the checks had found, both agents worked on what the checks had found.</b> Most of the {len(new['tc']) + len(new['vm'])} rows recorded in the second half are repairs, not speed: <code>torch.compile</code>'s freezing and its bug, <code>torch.vm</code>'s speed after a change of image size, a module state that differed from eager's, and training, which is now one graph for <code>torch.vm</code>.</li>
<li><b>The slope flattened for both.</b> The first three hours bought <code>torch.vm</code> 3,723 images/s and <code>torch.compile</code> 1,960; the second three bought {b6['vm'] - b3['vm']:,.0f} and, on the line that moved, {p6['tc'] - p3['tc']:,.0f}.
In tokens the second half was the more expensive one for both, because an agent that continues re-reads everything it has done so far at every step.</li>
</ul>
<h2>Every step of the second half</h2>
<p>Time and dollars continue from the first half; the pause between the halves is not counted. Each step's name links to what it is, in the <a href="#appendix">appendix</a>. The steps of the first half are on <a href="../two-compilers-one-model/">the first page</a>.</p>
<h3>torch.compile: {p3['tc']:,.0f} to {p6['tc']:,.0f} passing every check; best recorded unchanged at {b6['tc']:,.0f}</h3>
<p>Eight source files changed against the start, 186 lines added (the first half's three files, and Dynamo's guards and Inductor's freezing for the version guard), and a 65-line configuration.</p>
{G.steps_table('tc')}
<h3>torch.vm: {b3['vm']:,.0f} to {b6['vm']:,.0f}</h3>
<p>All source changes of the second half are in one file, the regions (774 lines added against the start over both halves); the configuration changed in one helper.</p>
{G.steps_table('vm')}
<h2>The bug, and what it means for the first page</h2>
<p>The first page said that every recorded state of both sides predicted as well as eager on real weights. That was measured, and it was not the whole truth for <code>torch.compile</code>'s states from its seventh step on.
Those states were right as long as Inductor's compile cache had been filled by the same model. Compiled once for a model with other weights and the same shapes, they took the first convolution's weight from that other model
(<a href="#t-bug">what happened, in the appendix</a>). The checks on the first page compiled every state for the pretrained model only, so they could not see it; the agent saw it in the second half because it ran the pretrained-weights check
on configurations it had first compiled for the benchmark's random weights.</p>
<p>The check below does that on purpose: with an empty cache, a state is compiled for the benchmark's random-weight model first and for the pretrained model second.</p>
{G.cache_table()}
<h2>The checks</h2>
<h3>Predictions</h3>
<p>As on the first page: the pretrained yolov3-spp weights, the 128 COCO images that ship with the benchmark. The {len(maps)} states recorded in the second half have a mAP@0.5 of {map_range}; eager in the same precision has 0.753 to 0.754.
Each is in the tables above.</p>
<h3>What the states still follow</h3>
{G.probe_table()}
<p>The first column is <code>torch.compile</code>'s fastest configuration, as at the end of the first half and again, on the corrected source, in the second (3,861). The last row is a probe that the agents were not told about and that the first page did not have: an attribute of a module that is not a tensor is edited after the callable is warm (the slope of one LeakyReLU in the middle of the model).
It is what <code>torch.compile</code>'s narrow guard filter says it gives up, and the filter does give it up. It also shows a hole in <code>torch.vm</code> that no check had exercised: its starting state follows the edit, because the model's loop then runs in the interpreter,
but from the first step of the first half on, when the whole model became one graph, the slope the trace read is in the graph and nothing checks the module's attribute again. <code>torch.compile</code> with every guard does check it.
"Passing every check" on this page means the checks both agents were given, which are the rows above the last. Counting the last row too, the fastest state that follows everything is <code>torch.compile</code>'s 3,404, and <code>torch.vm</code> has none above its start.</p>
<h3>Speed after events</h3>
<p>The callable is timed, then again after a batch of another size, after images of another size, and after a weight was updated in place.</p>
{G.after_table()}
<h3>Training</h3>
{G.loss_figure()}
{G.train_table()}
<p>The compiled curves are as far from eager's as eager's own is when it is run again or nudged in the last digit, as on the first page. What changed is <code>torch.vm</code>'s speed in training: at the end of the first half its training forward was a graph per layer and ran at eager's speed;
with the final source it is one graph, and a step takes what it takes under <code>torch.compile</code>.</p>
<h3>Tests</h3>
{C.table(["on each side's final source", "tests", "result"], G.TESTS, num_from=1)}
<h2>What this does and does not show</h2>
<ul>
<li><b>One model, one run per side, and now one continuation.</b> It is still an example. A different second half is easy to imagine: had either agent started on its next kernel, the best recorded numbers would have moved instead.</li>
<li><b>The second half was not a blind continuation.</b> Each agent was told what the checks had found for its side and that the checked number would be reported. That is why both spent their time on soundness; an agent told only to raise the throughput might not have.</li>
<li><b>"Passing every check" is as strong as the checks.</b> They did not catch the cache bug until an agent tripped over it, and one probe added afterwards was enough to find something neither side's fast states follow, and that <code>torch.vm</code> has no guard for at all.
A state that passes them is not thereby proven equal to eager, and the comparison of the two "checked" lines is only fair for the checks both agents knew.</li>
<li><b>Agents write bugs into compilers.</b> Both sides shipped one in the first half that the tests and the checks let through: a stale weight through the cache on one side, module attributes left at another image size's values on the other. Both were found by the agent that wrote them, in the second half, once it had a stronger check or a reason to look.</li>
<li><b>Not on the x axis,</b> as before: the tokens that built <code>torch.vm</code> up to the start, and those of the session that set this up and checked it.</li>
</ul>
<h2>How the second half was run</h2>
<ul>
<li>The two agents of the first half were resumed with their whole history, about three hours after they had stopped; the pause is not counted as their time. The trees and configurations were their final states, measured again before the start (3,860 and 4,506 images/s).</li>
<li>Rules, harness and measurement as before. Added, in the same words for both: the list of checks, that they would be run again, that both numbers would be reported, a command to run the checks, and that neither side reads the other's directory or the first page.</li>
<li>Each agent got the findings for its own side only: for <code>torch.compile</code>, which states followed what, the error at a recompilation, the cuDNN error in one run; for <code>torch.vm</code>, that every state passed, that the speed did not come back after events, and that training ran a graph per layer.</li>
<li><code>torch.vm</code>'s agent stopped once with 23 minutes left and was told to go on, as both had been in the first half.</li>
<li>Tokens at list price, as before: ${total['tc']:.0f} for <code>torch.compile</code>'s agent over the six hours ({t['tc']['output']/1e3:.0f}k output tokens, {t['tc']['cache_read']/1e6:.0f} million cached input tokens read) and ${total['vm']:.0f} for <code>torch.vm</code>'s
({t['vm']['output']/1e3:.0f}k and {t['vm']['cache_read']/1e6:.0f} million).</li>
</ul>
{A.APPENDIX}
"""
