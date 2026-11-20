"""appendix6.py: the appendix of the six-hour page: every recorded step of the second half, explained. Figures
inside a step are the agent's own, from its journal; the images/s in a heading are the recorded measurements."""
import html


def code(text):
    return f"<pre><code>{html.escape(text.strip(chr(10)))}</code></pre>"


def step(anchor, title, before, after, body):
    arrow = f' <span class="gain">{before} &rarr; {after} images/s</span>' if after else (f' <span class="gain">{before}</span>' if before else "")
    return f'<h4 class="step" id="{anchor}">{title}{arrow}</h4>\n{body}\n'


INTRO = """
<h2 id="appendix">Appendix: every step of the second half, in detail</h2>
<p>In the order of the tables above. The figures inside a step are from the agent's own profiles, as its journal has them; the images/s in a heading are the recorded measurements.
The steps of the first three hours, and the terms used here (host time, CUDA graph, weights built in, guards), are explained in the
<a href="../two-compilers-one-model/#appendix">appendix of the first page</a>.</p>
"""

TC = "".join([
    '<h3 id="appendix-tc">torch.compile</h3>\n',
    step("t-clean", "1. The state that already passed everything, recorded", "2,784 images/s", "", """
<p>The checks after the first half had found one <code>torch.compile</code> state faster than its start that passed all of them: the agent's final source with CUDA graphs and nothing else, no freezing and no guard filter.
The agent recorded it first, as the floor of its second line.</p>"""),
    step("t-version", "2. Freezing that notices a weight update", "2,784", "3,404", """
<p><b>The problem.</b> Freezing builds the weights into the graph, and after that nothing looked at them again: every frozen state of the first half kept computing with the old values after a weight was updated in place.</p>
<p><b>Change</b>, in five files of Dynamo and Inductor. Every tensor in PyTorch carries a version counter that goes up on each in-place write (autograd uses it to notice that a tensor it saved was modified).
Freezing now records which tensors it built into the graph, and Dynamo guards the version counter of exactly those. When one of them has been written to, the guard fails, the function is compiled again and the new values are built in.
Tensors the graph writes itself (a batch norm's running statistics in training mode) are not frozen and get no such guard, so a training-mode call does not recompile every time.</p>
<p><b>Cost.</b> One recompilation per weight update, about 9 s here; the configuration raises Dynamo's limit on recompilations from 8 to 32, since the ninth would otherwise send the function back to eager for good.
And 370 more guards to check on every call (step 4).</p>
<p><b>What it still misses.</b> A write through <code>tensor.data</code> does not move the version counter and is not followed.</p>
<p>The first row recorded with this change (3,413) is struck, because of the bug in the next entry; 3,404 is the same configuration on the corrected source.</p>"""),
    step("t-bug", "3. A bug in a first-half change: a weight that came from another model", "", "", """
<p><b>How it showed.</b> With the pretrained-weights check now at hand, the agent ran it on a new candidate and got a wrong result outright: 2.88 from eager where a correct state is at 0.05, with the first convolutions' outputs off by half their size.
On the benchmark's random weights the same candidate had looked right to the last bit. For most of an hour it looked like a matter of settings ("it takes all four of these together"), because whether a configuration was right depended on something else:
which model had been compiled with it first.</p>
<p><b>The cause</b> was the agent's own step 7 of the first half, the fourth input channel for the first convolution. For a frozen weight that change computed the padded weight while compiling and registered it as a new constant of the graph.
Inductor keeps compiled graphs in a cache on disk. For a frozen graph, the cache's key covers the shapes and types of the frozen weights but not their values, since those are supplied again from the model on a cache hit; constants that are created later, while the graph is turned into kernels, are stored in the cache entry as they are.
So when the same architecture was compiled a second time with other weights, the cache handed back the first model's padded weight for the first convolution, and everything else from the second.</p>
<p><b>The fix.</b> The weight is padded when the code runs (one small kernel per call) and no constant is created.</p>
<p><b>What it means for the rows.</b> The first-half rows from that step on, and the 3,126 row of this half, were measured with a cache that the same model had filled, so their numbers are for the right function; but those states would have computed with a stale weight after another set of weights had been compiled.
The 3,413 row was measured in exactly that situation (error against float64 4.97 times eager's, inside the harness's limit of 10) and is struck. The check that reproduces this is in the section on checks above.</p>"""),
    step("t-narrow", "4. A narrower guard filter", "3,404", "3,549", """
<p><b>What the guards cost now.</b> With every guard and the 370 version guards, the check before each call takes 197 microseconds; the first half's unsafe filter, which drops every guard on a module, takes 3.</p>
<p><b>Change.</b> A filter of the agent's own that keeps the guards on tensors (their identity and, new, their version) and on each module's <code>training</code> flag, and drops the guards on the modules' other attributes: 114 microseconds, nearly all of it the version guards.</p>
<p><b>What it gives up.</b> An edit of an attribute of a module that is not a tensor, after compilation (a stride, a slope, a hook), is not noticed. The probes of the first half had no such edit; one was added for this page, and the table of probes above has the outcome.</p>"""),
    step("t-fast", "5. The fastest configuration again, on the corrected source", "3,861 images/s", "", """
<p>The configuration of the first half's best row (3,973: freezing, every guard on a module dropped, CUDA graphs without trees), measured on the source without the bug. It follows neither weight updates nor a replaced parameter nor <code>model.train()</code>, as before.
The 3% to the earlier number is the small kernel that now pads the weight on every call, and the spread between processes these rows have.</p>"""),
    step("t-static", "6. CUDA graphs without trees, and static shapes", "3,549", "3,634", """
<p><b>A slowdown the checks had not looked for.</b> After the whole sequence of probes the same callable ran at about 1,480 images/s instead of 3,560. The agent timed it step by step: the speed held through the weight updates and the other shapes and fell when the last bias was replaced.
By then the function had seen three input shapes, and Dynamo's automatic dynamic shapes compile the next version with symbolic sizes; that version ran without a CUDA graph.
<code>torch._dynamo.config.automatic_dynamic_shapes = False</code> gives every shape a graph of its own with fixed sizes: the callable stays at 3,600 to 3,650 through the whole sequence, and a recompilation takes 9 to 20 s instead of up to 130.</p>
<p><b>Without trees.</b> As in the first half, the older CUDA-graph path saves about 40 microseconds of bookkeeping per call, and gives up the error that the tree manager raises when a result is read after a later call has overwritten it.
It also sidesteps an error the tree manager raises for this model (re-recording a graph after other shapes have run: "storage data ptrs are not allocated in pool"), which the agent reproduced without freezing and did not fix.</p>
<p>3,619, and 3,634 for the final state, which is the same source and configuration measured again.</p>"""),
    step("t-one", "7. One guard for all version counters", "3,626 images/s, no gain", "", """
<p>The 370 version guards are 106 of the narrow filter's 114 microseconds; reading the same 370 counters in one Python call takes 20. The agent replaced the 370 guards by one that checks all the counters.
It measured the same: the loop over 370 weak references costs what the guards did. Reverted; the agent's note for later is one call in C over the frozen tensors, which would recover most of the 0.11 ms that separates 3,634 from 3,861.</p>"""),
])

VM = "".join([
    '<h3 id="appendix-vm">torch.vm</h3>\n',
    step("v-early", "1. The kernels started before the call is checked", "4,507", "4,608", """
<p><b>What was left.</b> At the start of a call the GPU sat idle for about 60 microseconds, 41 of them the runtime's look over the model's 370 tensors that tells whether this call is like the last one.</p>
<p><b>Change.</b> If the input is in memory that one of the recordings reads in place, the recording that served the previous call is started as soon as the region is called, and the look is done while the GPU already runs.
The results are handed out only if the look then confirms that the call was for that recording; otherwise they are dropped and the call is handled the ordinary way, which is safe because a recording writes only to memory of its own.
A call goes from 1.766 to 1.727 ms. This is the only step of the second half that raised <code>torch.vm</code>'s throughput.</p>"""),
    step("v-events", "2. The speed after another image size", "4,599 and 4,614 images/s, unchanged", "", """
<p><b>The finding it started from.</b> The checks after the first half timed the final state again after a batch of another size, images of another size and a weight update: 4,533, 4,387, 4,205, 4,194 images/s. The speed did not come back.</p>
<p><b>First cause.</b> To see where a graph's inputs lie in memory, the CUDA-graph code asked each for a pointer one can write through. For a watched tensor that request counts as a write, so weights that happened to be inputs of some graph lost their watch,
and the graph for the timed shape was built again with 24 convolution weights as inputs instead of built in: their batch norms were folded on every call, 7% of it. The code now asks for the address without the right to write.</p>
<p><b>Second cause.</b> For a new image size a YOLO layer stores two integers, <code>self.nx, self.ny = ng</code>. A region allowed stores of tensors (step 1 of the first half) but not of plain values, so for the new size the VM ran the model instruction by instruction once, with a graph per layer.
A region may now store a plain value (a number, a string, <code>None</code>) into a module attribute, after its graph; what those attributes held before the call is part of what a compiled variant is keyed on, so a variant is only used for the state it was traced from.
An attribute that is overwritten with a new value on every call, a counter, is still refused.</p>
<p><b>A wrong state this also fixed.</b> Coming back to the first image size had left <code>nx</code> and <code>ny</code> at the other size's values, which eager does not do. The outputs were right, so none of the checks had seen it.</p>"""),
    step("v-train", "3. Training as one graph", "4,602 images/s, unchanged", "", """
<p>The first page noted that <code>torch.vm</code> was no faster than eager in training on this model. What kept the training-mode forward from being one region was one statement: in training mode the YOLO layer sets <code>self.grid = None</code> on every call, a store of <code>None</code> into a buffer that already holds <code>None</code>.
With that allowed the training forward is one region of 709 operations. The measurements of training are in the section on checks.</p>"""),
    step("v-rest", "4. Tried and dropped, and the rows without a change", "", "", """
<p><b>Results as copy-on-write clones.</b> Each call copies its results out of the recording (about 30 microseconds). Handing them out as lazy clones would copy only when someone writes, but on this build a lazy clone of a tensor on the GPU copies at once, so nothing was saved.</p>
<p><b>Building a changed weight in again after a quiet spell.</b> After a weight update the new graph has that weight as an input. The agent tried building it in again once it had not changed for 64 calls: the speed after a weight update got worse, not better (4,479 against 4,535), and each time uses one of the eight graphs a function may have with weights built in. Reverted; the 4,611 row is that state and 4,606 the state put back.</p>
<p><b>Hardening.</b> The agent's own tests of its changes grew from 16 to 22, among them three threads calling the same compiled function, and it ran random sequences of image sizes, batch sizes, <code>train()</code>, <code>eval()</code> and weight updates against eager, checking as well that no result a caller kept was changed by a later call.</p>
<p><b>Where a call's time is at the end.</b> 1.73 ms, of which about 1.70 is the GPU's; 91 to 93% of that is the fused cuDNN kernels.</p>"""),
])

APPENDIX = INTRO + TC + VM
ANCHOR = {
    "tc": {16: "t-clean", 17: "t-version", 18: "t-bug", 19: "t-version", 20: "t-narrow", 21: "t-fast", 22: "t-static", 23: "t-one", 24: "t-static"},
    "vm": {18: "v-early", 19: "v-events", 20: "v-events", 21: "v-train", 22: "v-rest", 23: "v-rest", 24: "v-rest"},
}
SHORT = {
    "tc": {16: "the state that passed everything: CUDA graphs only", 17: "freezing that notices a weight update (struck: stale weight)", 18: "the same without autotune (state still has the bug)",
           19: "corrected source: freezing that notices a weight update", 20: "a narrower guard filter", 21: "(the fastest configuration, on the corrected source)",
           22: "CUDA graphs without trees, static shapes", 23: "one guard for all version counters (no gain, reverted)", 24: "(final state: the 3,619 row again)"},
    "vm": {18: "kernels started before the call is checked", 19: "plain values stored into modules", 20: "the early start also finds model tensors as inputs",
           21: "None stored into a buffer: training is one graph", 22: "(docstrings)", 23: "a changed weight built in again (dropped)", 24: "(source put back)"},
}
