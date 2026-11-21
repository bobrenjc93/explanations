"""appendix.py: the appendix of the page: every recorded step of both sides, explained. The figures quoted inside a
step (microseconds per kernel, milliseconds per call, A/B ratios) are the agent's own, from its journal and summary;
the images/s in each heading are the recorded measurements. Which rows are struck and what the probes found is
filled in by make.py from the checks."""
import html


def code(text):
    return f"<pre><code>{html.escape(text.strip(chr(10)))}</code></pre>"


def step(anchor, title, gain, body):
    arrow = f' <span class="gain">{gain}</span>' if gain else ""
    return f'<h4 class="step" id="{anchor}">{title}{arrow}</h4>\n{body}\n'


INTRO = """
<h2 id="appendix">Appendix: every step, in detail</h2>
<p>What each agent found, what it changed, why that is faster, what it measured and what it gives up, in the order of the tables above.
The figures inside a step (microseconds per kernel, milliseconds per call, the ratios of an A/B comparison in one process) are the agent's own, from its journal and its final summary;
they were not measured again for this page. The images/s in a heading are the recorded measurements before and after, each the median of three processes.
What a state follows and whether its tests pass is from the checks made afterwards, not from the agents' notes.</p>
<p><b>The model, for orientation.</b> One call is a batch of 8 images through 76 convolutions. 73 of them are followed by a batch norm and a LeakyReLU, 23 results are added to an earlier one
(the residual connections), one tensor goes through three max pools with windows of 5, 9 and 13, and three detection heads turn their convolution's output into boxes with a few sigmoids and exponentials;
their three results are concatenated. Compiled, that is about 330 GPU kernels in this version of PyTorch: the convolutions are cuDNN's, everything between them is kernels that Inductor generates, and together they take about 2.5 ms.</p>
<p><b>Terms that come back.</b></p>
<ul>
<li><b>Host time.</b> Each kernel is launched from the CPU, through Python, PyTorch and the driver, at 10 microseconds or more apiece. For 330 small kernels that is more than the GPU needs to run them, so the GPU waits for the CPU.</li>
<li><b>CUDA graph.</b> A recording of all the kernel launches of one call. Replaying it starts them with one call into the driver. The recording is tied to the memory the recorded call used:
it reads its inputs and writes its results at fixed addresses. So each call's input is copied to where the recording reads it, the results sit in memory that the next replay writes again, and each new input shape needs a recording of its own.</li>
<li><b>Guards, and the checks of a call.</b> Compiled code is right only for the situation it was compiled in: these modules, in this mode, holding tensors of these shapes. Before it runs the compiled code, <code>torch.compile</code>
evaluates guards, one small check per assumption (about 3,700 for this model). <code>torch.vm</code> does the equivalent by going through the instructions of the path again without the tensor operations,
re-reading every value Python read and comparing it (13,604 instructions for this model).</li>
<li><b>Weights built in.</b> A compiler can treat the model's weights as inputs of the graph or as constants of it. As constants they can be combined ahead of time (a batch norm folds into its convolution) and stored in the layout cuDNN prefers.
The price is that the graph is right only while the weights hold what they held.</li>
<li><b>Channels last.</b> cuDNN's fast half-precision convolutions want images and weights stored with the channel as the last dimension. PyTorch stores them with the channel second. With the weights as inputs,
the compiled code copies each 3x3 weight into the other layout on every call (a 1x1 weight is the same bytes in both).</li>
</ul>
"""

TC = "".join([
    '<h3 id="appendix-tc">torch.compile</h3>\n',
    step("tc-0", "Start: the dashboard's configuration", "1,954 images/s", """
<p><code>torch.compile</code> of a function that calls the model, with the Inductor backend in its default mode. The agent's first profile: a call takes 3.92 ms, and for 3.78 ms of it the CPU is launching about 330 kernels from Python;
the kernels themselves need about 2.6 ms of the GPU (cuDNN's convolutions 1.54 ms, the kernels between them about 0.8, the copies of the weights into channels last about 0.15, the detection heads 0.13).
The host is the bottleneck, not the GPU.</p>"""),
    step("tc-1", "1. CUDA graphs, with the results copied out", "1,954 &rarr; 2,591 images/s, 7 minutes in", """
<p><b>Change.</b> One option, <code>mode="reduce-overhead"</code>, later written as <code>options={"triton.cudagraphs": True}</code>: Inductor records the compiled graph's kernels as a CUDA graph and replays it.
And a few lines in the configuration that clone every tensor the compiled function returns.</p>
<p><b>Why it helps.</b> The 330 launches become one replay.</p>
<p><b>Why the clones.</b> The results of a replay lie in the recording's memory, and the next call writes them again. Eager hands out new tensors on every call, and a caller may keep them.
The agent decided at the start that its main line would hand out copies; it later measured their cost at 0.9% (3,020 against 3,046 images/s in one process).</p>
<p><b>What it gives up.</b> Nothing that the checks look at. One thing it inherits from the start: after an input of another shape, the dashboard's configuration recompiles with symbolic sizes and the new graph then also serves the timed shape, more slowly.
The agent ran the speed-after-events script on the starting configuration and on this one, one process each: 2,066 &rarr; 1,845 &rarr; 1,189 and 2,600 &rarr; 1,936 &rarr; 1,245 images/s. Step 3 removes that.</p>"""),
    step("tc-2", "2. The heads' concatenation unfused, and a cheaper guard check", "2,591 &rarr; 2,757 images/s, 36 minutes in", """
<p><b>The concatenation.</b> The model's result is the <code>torch.cat</code> of the three heads' outputs. Inductor had fused it with the arithmetic of its inputs into one kernel, in which every element of the result evaluates the expressions of all three heads
and then picks one: 0.134 ms. <code>"max_complex_pointwise_cat_inputs": 0</code> makes each head write its own slice of the result: 0.04 ms. In one process on the final tree the option is worth 2.7% (2,907 against 2,985).</p>
<p><b>The guards.</b> Of the 0.46 ms the CPU still spent per call, about 0.2 was Dynamo's guards. <code>torch._dynamo.config.use_recursive_dict_tags_for_guards = True</code> lets Dynamo skip all guards under a module
when neither its attribute dictionary nor those of its submodules have been modified since the last check, which Python can tell from a version tag on each dictionary: 202 to 130 microseconds.</p>
<p><b>What it gives up</b>, by the agent's own later finding: assigning a new <code>__class__</code> to a module nested inside such a module does not touch any dictionary, and is not noticed.
The agent reproduced that on the unmodified tree with this flag, on a small model on the CPU; without the flag it is noticed. The checks made for this page have no such probe.</p>
<p><b>About this row.</b> Its third process ran while the agent was already editing a source file for the next step, and recompiled (2,806). The two clean processes gave 2,757 and 2,754, and the median is one of them.
From here on the agent developed in a copy of its tree.</p>"""),
    step("tc-3", "3. Max pool one dimension at a time; module guards on the fast path; hooks followed; a graph per shape", "2,757 &rarr; 2,964 images/s, 71 minutes in", """
<p>Four things in one row; only their sum was measured officially.</p>
<p><b>Max pool, in Inductor</b> (<code>torch/_inductor/lowering.py</code>). The three max pools took 0.124 ms. Inductor computes each output as the maximum over the whole window, so the 13-wide pool reads 169 values per output element.
The maximum over a rectangle is the maximum over its rows of the maximum along each row, so when only the pooled values are used (not the positions of the maxima) the pool can run as one pass per dimension: 13 + 13 reads.
The lowering does that where it at least halves the reads. A maximum involves no rounding, so the result is the same bits.</p>
<p><b>Module guards, in Dynamo</b> (<code>torch/_dynamo/guards.py</code>). The model's loop over its 114 layers reads <code>module.__class__.__name__</code> to decide what to do with each. The guards on that read kept every one of those modules off the fast path of step 2.
The change counts <code>module.__class__</code> and <code>cls.__name__</code> as safe for that path, under an existing setting that says such attributes are assumed not to change. In a test on the CPU the guard check went from 120 to 78 microseconds;
the agent did not isolate its share of this row.</p>
<p><b>Hooks.</b> <code>torch.compile</code> by default does not notice a forward hook that is registered on a module after compilation. The agent set <code>skip_nnmodule_hook_guards = False</code>, so that it does, as eager does.
Cost: 1.4% (2,999 against 3,043 in one process), and only that little because of step 2: with the hook guards on, switching the dictionary tags off takes the callable from 2,999 to 2,686 images/s.</p>
<p><b>A graph per shape.</b> Run on its own state, the speed-after-events script gave 2,954 images/s at first, 2,121 after a batch of 3 and 1,380 after another image size. With <code>dynamic=False</code> every input shape gets a graph of its own with fixed sizes:
2,877, 2,943, 2,941, 2,952. This adds nothing at the timed shape; it keeps the speed afterwards.</p>"""),
    step("tc-4", "4. The pool passes as unrolled kernels", "2,964 &rarr; 2,992 images/s, 96 minutes in", """
<p>The 9- and 13-wide one-dimensional passes of step 3 had been generated as reduction kernels, a loop with a running maximum per output element: 16 microseconds each. A kernel that reads the 9 or 13 values and takes their maximum in one expression is faster.
Inductor already does that for windows below a threshold; inside the new path the threshold is raised to 25. The three pools now take 0.014 ms, from 0.124 at the start.
<code>dynamic=False</code> was replaced by <code>torch._dynamo.config.automatic_dynamic_shapes = False</code>, which has the same effect without patching a setting on every call (about 11 microseconds).</p>
<p>The recorded step, 0.9%, is inside the noise. This is the highest row of the line that passes every check until the fix of step 8.</p>"""),
    step("tc-5", "5. A fourth input channel for the first convolution, and the regression in it", "2,992 &rarr; 2,966 images/s, 133 minutes in; struck", """
<p><b>What was measured.</b> The first convolution reads images with 3 channels, and cuDNN's half-precision channels-last kernel is slow for exactly that: 0.1015 ms, against 0.0786 with the input padded to 4 channels (8 channels: no better).</p>
<p><b>Change</b>, in Inductor's convolution lowering (<code>torch/_inductor/kernel/conv.py</code>): a half-precision channels-last convolution on CUDA with 3 input channels gets a channel of zeros appended to its input and zero weights for it.
A zero times a zero adds nothing to any sum; what changes is which kernel cuDNN picks. In the full graph the first convolution went from 0.101 to 0.076 ms. The recorded row did not move: 2,966 after 2,992, which the agent put down to noise between sessions.</p>
<p><b>The regression.</b> As first written, the change padded every such convolution, also one whose input and weight are already stored channels last. There it adds padding kernels where none are needed.
PyTorch has a test for exactly that case, <code>test_require_stride_expanded</code>, which asserts that such a convolution generates no kernel, and whose half-precision run now failed.
Before this measurement the agent had run the Inductor GPU tests whose names contain <code>conv</code> (352 pass); the name of this test does not. It found the failure about 3 hours 45 minutes in, when it ran the whole suite.
By then six rows had been recorded on sources that contain it. They are struck (<a href="#struck">the table above</a>); step 8 is the fix.</p>"""),
    step("tc-6", "6. Filter counts rounded up to a multiple of 8", "2,966 &rarr; 3,046 images/s, 152 minutes in; struck", """
<p><b>What was measured.</b> The three detection-head convolutions have 255 output channels. cuDNN's fast channels-last half-precision kernels want a multiple of 8 and otherwise fall back to much slower ones:
55, 30 and 22 microseconds for the three heads, against 9, 5 and 4 with 256 filters.</p>
<p><b>Change</b>, in the same lowering: for a half-precision convolution with 64 filters or more whose count is not a multiple of 8, the weight is padded with all-zero filters up to the next multiple, and the extra output channels are sliced off.
The channels that are kept are computed from the same numbers as before.</p>
<p><b>Effect.</b> +2.7%, the last step of this line that is clearly outside the noise. The row is struck because its source still has the regression of step 5; the change itself is in the final source, where the same graph measures 3,014 and 3,015.</p>"""),
    step("tc-7", "7. One input copied with <code>copy_</code>", "3,046 &rarr; 3,048 images/s, 162 minutes in; struck", """
<p>Inductor's CUDA-graph code copies each call's inputs into the recording's memory with one batched kernel for all of them. For a single input the agent replaced it by a plain <code>copy_</code>, which goes through the device's memcpy
(<code>torch/_inductor/cudagraph_trees.py</code>). No measurable gain; the agent's summary lists it as an estimate that was never confirmed.</p>"""),
    step("tc-8", "8. The fix: the input is padded only where it is copied anyway", "3,014 and 3,015 images/s, 248 and 277 minutes in", """
<p><b>Change.</b> The zero channel of step 5 is now added only when the convolution's input has to be copied into channels last anyway. The copy then writes the padded input instead, so no kernel is added, and a convolution whose input is already channels last is left alone.
<code>test_require_stride_expanded</code> passes again. For yolov3 the graph is the same as before the fix: 127 generated kernels, the first convolution at 0.0765 ms.</p>
<p><b>The two rows.</b> 3,014 is the corrected source with the checked configuration. Because it read lower than the struck 3,048 of the same graph, the agent measured the unchanged state once more: 3,015. Its journal says, before the second result was in,
that the decision to measure again depended on the first result, so the two are two samples of one state and not a best of two. It also measured two processes of identical code 2% apart in GPU time, which is the size of the difference to the struck rows.</p>"""),
    step("tc-u", "The unchecked line: weights built in, guards dropped, results not copied", "3,327 &rarr; 3,405 &rarr; 3,551 images/s; 3,524, 3,570 and 3,612 struck", """
<p>From 86 minutes on the agent kept a second configuration next to the first, for the report's first number, and switched its <code>config.py</code> back after each measurement. Every one of these rows says in its note what it gives up.</p>
<p><b>Variant 1</b> (3,327). <code>freezing</code> builds the weights into the graph: the batch norms fold into their convolutions and the weights are stored channels last once. By itself it was slower (2,371): Dynamo passes a model's parameters to the graph as inputs,
so nothing was folded, and the CUDA-graph code copied all 375 inputs on every call. With <code>torch._dynamo.config.install_free_tensors = True</code> the parameters become constants of the graph and freezing works.
The guard filter <code>torch.compiler.keep_tensor_guards_unsafe</code> keeps only the guards on tensors, and the results are handed out without a copy.</p>
<p><b>Variant 2</b> (3,405; again on a later source: 3,524, struck). A guard filter of the agent's own that keeps only the guards on the call's arguments and on global state.
Its first attempt (3,416) is the one row the harness refused: a mismatch in how guards are named made the filter drop every guard, so a batch of 3 images was fed to the graph for 8 and raised an error.</p>
<p><b>Variant 3</b> (3,570, struck). <code>"triton.cudagraph_trees": False</code> selects Inductor's older CUDA-graph path, without the bookkeeping that tracks which results are still alive and where the weights lie: +1.4% in one process.</p>
<p><b>Variant 4</b> (3,612, struck; on the corrected source 3,551). <code>"max_pointwise_cat_inputs": 0</code> makes every concatenation a kernel of its own: +1.4% in the frozen graph (in the checked configuration the same option is 1.8% slower).</p>
<p><b>What these states give up, as the probes found it</b> (all six states alike; <a href="#follow">the table above</a>). Not followed: an in-place update of a convolution's weight, of a batch norm's bias or of its running mean;
a parameter replaced by a new one; <code>model.train()</code>, and after <code>model.eval()</code> the callable is still wrong (the agent's reading: eager's training-mode call moved the batch norms' statistics and the frozen graph has the old ones);
the edited slope of a LeakyReLU; a weight written through <code>.data</code>; and all four tensors of a result that the caller kept, as tensors or through DLPack, hold other values two calls later.
Followed: an update of the last convolution's bias, which is folded into nothing, so the graph still reads the tensor itself (it is why the harness's own weight probe passes on these rows), other batch and image sizes, and the timed shape again.
Their predictions on the pretrained weights are as good as eager's.</p>"""),
    step("tc-rest", "What did not help, and where a call's time is at the end", "", """
<p><b>Tried and dropped</b>, with the agent's figures: letting the CUDA-graph code copy live results itself instead of the clones in the configuration (2,519: the model stores grid tensors on its modules, and three storages were cloned on every call);
<code>install_free_tensors</code> in the checked configuration (no gain, and a replaced parameter is then not followed); combo kernels (1,234, and 1.001 with their benchmarking on);
<code>aggressive_fusion</code> (0.997), <code>max_autotune_pointwise</code> (0.987), folding batch norms into the convolution weights at run time (0.964); the 38 weight-layout copies as one batched launch per shape
(slower for the large weights: 0.161 against 0.057 ms); Python's garbage collector (no effect); the older CUDA-graph path for the checked configuration (rejected without measuring: it never notices a parameter that moved).</p>
<p><b>Where a call's time is at the end</b>, checked configuration: the replay is 2.31 ms of GPU work and a call takes 2.73 ms. Of the rest, about 0.3 ms is CPU work in front of the launch: the guards (about 0.1 ms), handing 374 parameters to the graph,
the check that they still lie where the recording reads them, and wrappers. Without waiting for the GPU between calls the same callable runs 3,451 images/s. On the GPU, cuDNN's convolutions are 1.5 ms, the batch-norm and LeakyReLU kernels behind them 0.59 ms
("at memory bandwidth; only a conv kernel with that epilogue removes them", in the agent's words) and the weight-layout copies 0.14 ms.</p>
<p><b>A cost the agent measured on its final checked configuration, in one process.</b> After a <code>model.train()</code> and <code>model.eval()</code> round trip it is about 10% slower for the rest of the process (3,010 images/s before, 2,713 after):
the toggle switches the dictionary-tag fast path of step 2 off, so every guard runs in full. With the hook guards of step 3 off, the loss is about 5%. The cause is in the C++ of the guards, which this experiment could not rebuild.</p>"""),
])

VM = "".join([
    '<h3 id="appendix-vm">torch.vm</h3>\n',
    step("vm-0", "Start: the dashboard's configuration", "1,825 images/s", """
<p>The default passes, <code>Graphs()</code> and <code>Inductor()</code>. The engine takes down the path a call takes through the model as one graph of 328 tensor operations and hands it to Inductor.
The agent's first profile: launching that graph's kernels keeps the CPU busy for 4.10 ms per call while the GPU works for only 2.53 ms (cuDNN's convolutions 1.55 ms, the generated kernels 0.97).
Of the CPU's time, 0.54 ms is the engine in front of the graph, going through the path's 13,604 instructions to re-read and check what the path depends on, and 0.13 ms is the engine behind it.</p>"""),
    step("vm-1", "1. A CUDA graph around what Inductor compiled, built as a new option", "1,825 &rarr; 2,405 images/s, 24 minutes in", """
<p><b>The problem</b> is <code>torch.compile</code>'s: 4.1 ms of launching for 2.5 ms of kernels. <code>torch.compile</code> has an option for it; this version of <code>torch.vm</code> had none, so the agent wrote one:
<code>Inductor(cuda_graphs=True)</code>, in a new file <code>torch/vm/_cudagraphs.py</code>.</p>
<p><b>How it works.</b> The first call runs the compiled function as it is, which loads the kernels, lets cuDNN choose its algorithms and shows whether the function writes to what it is given. The second call is recorded as a CUDA graph.
A tensor that lay at the same address in both calls (a weight) is read where it lies, and every later call checks that it still lies there; what it holds is read when the kernels run, so a weight updated in place is followed.
The other tensors (the input) are copied into memory of the recording, and the results are copied out, so a caller gets tensors of its own.
When a tensor that was read in place is found somewhere else (a parameter that was replaced), the graph is recorded again and reads it where it now lies; a tensor that moves a second time is copied from then on.</p>
<p><b>When it does not apply.</b> Calls that record gradients, graphs that write to their inputs, that compute a tensor on the CPU from anything but numbers, or that hand back one of their inputs run the compiled function as before.</p>
<p><b>Effect.</b> The CPU's launch work fell from 4.10 to 1.47 ms per call. A call now took 3.3 ms against 2.74 ms of GPU work: what remained was the engine's walk in front of the graph. Nothing is given up: the state passes every check.</p>"""),
    step("vm-2", "2. The graph started ahead of the engine's checks", "2,405 &rarr; 2,869 images/s, 32 minutes in", """
<p><b>The problem.</b> Before the graph could start, the engine went through the path: 0.54 ms on the CPU with the GPU idle, and 0.13 ms more behind the graph. That walk is what makes <code>torch.vm</code> notice an edited attribute or a module switched to training mode,
so it cannot be dropped the way a guard filter drops guards.</p>
<p><b>Change</b>, in the Rust engine and in the new file. When a function is called, the engine first asks the graph that served the previous call to start, with the arguments it already has in hand.
That graph checks that the tensors it reads in place are still where they were, copies the input in and launches the replay. Only then does the engine walk the path, while the GPU works.
If the walk confirms that this call is one the graph was made for, the results are copied out. If not, the run is discarded and the call is handled the ordinary way.</p>
<p><b>Why that is safe.</b> Until its results are handed out, a replay has written to nothing but the recording's own memory, so a run that is thrown away leaves nothing behind. The tensors it reads in place are kept alive for as long as it may read them.
A graph that draws random numbers is not started ahead, because the numbers of a discarded run would be missing from the generator. After a start that was for nothing, the graph waits longer before it is started ahead again (twice as many calls each time, up to 32).</p>
<p><b>Effect.</b> A call takes 2.79 ms against 2.70 ms of GPU work: the callable is bound by the GPU from here on. This is <code>torch.vm</code>'s largest step after CUDA graphs, and it has no counterpart on the other side,
where the 0.1 ms of guards still runs in front of the launch at the end.</p>"""),
    step("vm-3", "3. A pass: max pools as two one-dimensional pools", "2,869 &rarr; 2,971 images/s, 45 minutes in", """
<p>The agent's profile of one replay, in microseconds: convolutions 1,526, batch norm and LeakyReLU kernels 522, max pools 123, weight copies into channels last 121, the heads about 150, copies of input and results 62.</p>
<p><b>Change.</b> A new pass over the graph the engine exports, <code>torch.vm.passes.SplitPools()</code>, which the configuration puts between <code>Graphs()</code> and <code>Inductor()</code>: a max pool over a window of 16 elements or more becomes two pools,
one along each dimension of the window. It is the rewrite <code>torch.compile</code>'s agent put into Inductor's lowering (its step 3), done here on the graph before Inductor sees it, and exact for the same reason.
The three pools went from 123 to 14.5 microseconds.</p>"""),
    step("vm-4", "4. Values that a compiled graph keeps for itself", "2,971 &rarr; 3,218 images/s, 78 minutes in", """
<p><b>What the agent found.</b> It listed what the engine still runs after the graph. Each detection layer reads the size and the device of its convolution's output:</p>
""" + code("""
def forward(self, p, out):
    bs, _, ny, nx = p.shape                              # read from the convolution's output
    self.grid = self.create_grids((nx, ny), p.device)    # stores nx and ny on the module
""") + """
<p>The stores into the module happen after the graph, and they need <code>p.shape</code> and <code>p.device</code>. So the three convolution outputs were results of the graph for no other reason than to be asked for their size afterwards.
Inductor therefore wrote each of them twice, once more in PyTorch's own layout (68, 19 and 7 microseconds), and the CUDA-graph wrapper copied 16 MB more per call.</p>
<p><b>Change</b>, in the Rust engine. The engine compiles a graph once for each combination of input sizes and types it meets. A value that is computed after the graph, is the same for every call of one such combination, and cannot change
(a number, a string, a device, a dtype, the sizes of a tensor, a tuple of these) is now kept with the compiled graph. Its instruction does not run, and the tensor it was read from is no longer a result.
The kernels of a call went from 2,600 to 2,465 microseconds.</p>
<p><b>Two fixes in the same snapshot</b>, both found by a probe script the agent wrote for itself. A process that had not run the model in eager first showed <code>torch.meshgrid</code>'s once-only warning inside the first call the engine watched.
The Python frames that print a warning made that call look as if it entered other Python code, and the path ended there for good: a graph of 205 operations and pieces, 2,300 images/s. Frames entered to show a warning are now counted apart.
And a weight that had moved twice, and was therefore copied, had stopped the early start of step 2 for good; it no longer does.</p>
<p><b>The number.</b> 3,218 is the highest row of <code>torch.vm</code> that passes every check, and its agent could not get it again (<a href="#vm-safe">below</a>).</p>"""),
    step("vm-5", "5. The fast configuration: weight layouts kept, large results handed out", "3,338 images/s at 117 minutes, 3,384 at 168; later samples 3,353 to 3,378", """
<p>Two options, both off by default, each of which gives something up. Like the other side, the agent recorded them as a second configuration and kept the first.</p>
<p><b><code>weight_layout="keep"</code></b> (the 3,338 row). The compiled code copies the 38 weights of the 3x3 convolutions into channels last on every call: 116 microseconds. With the weights converted beforehand, in an experiment, the callable ran 5% faster.
The option hands those weights to Inductor as extra inputs that are already laid out channels last. For a 1x1 weight that is a view of the same memory, which is always right. For the others it is a copy that is kept,
and made again when the weight is another tensor or its version counter has moved. PyTorch raises that counter on every in-place operation, so <code>w.mul_()</code>, <code>w.copy_()</code> and writes through a view are followed.
<b>What it gives up:</b> a write through <code>w.data</code> raises no counter and is not followed. The probe made for this page confirms that, on all seven states that use the option.</p>
<p><b>Why not make that safe.</b> The agent looked at a fingerprint of each weight computed on the GPU on every call: 130 to 160 microseconds, as much as the copies it would save (144 in the same setup). A counter of writes on the memory itself would need C++.</p>
<p><b><code>hand_out=True</code></b> (the 3,384 row, where it was always on, and as an option from the next row). Copying the results out of the recording takes about 40 microseconds. With the option, a result of 1 MB or more is handed out as a tensor on the recording's memory.
Before the recording runs again it looks at the reference counts of that tensor and of its memory: if somebody still holds the tensor, the tensor is given memory of its own first; while a view of it is alive, the recording does not run at all and the compiled function does.
<b>What it gives up:</b> somebody who kept the raw address of such a result (a DLPack capsule, <code>__cuda_array_interface__</code>) is invisible to reference counts and sees the next call write that memory.
The 3,384 row's note names only the <code>.data</code> hole; the journal names both, and the later rows' notes do.</p>
<p><b>What the probes made for this page found.</b> On the six states that hand results out, a result kept through DLPack capsules across two calls with other inputs came back with two of its four tensors changed:
by their sizes, the two raw head outputs above 1 MB. The third is smaller and is copied, and the concatenated detections are made after the graph and are a new tensor in every call.
Kept as tensors, all four stayed what they were, which is what the reference counts are for. On the state with only the kept weight layouts (the 3,338 row) both probes pass.</p>
<p>The same snapshot moved the address check of step 2 into the engine: one call for all tensors read in place, 3.4 microseconds in front of the launch instead of 23.</p>"""),
    step("vm-safe", "The safe configuration on the later sources, and the 3,218 that did not come back", "3,145 to 3,162 images/s, nine rows from 124 minutes on", """
<p>After each change to the source the agent measured the configuration of step 4 again, with both options off. All nine rows are between 3,145 and 3,162, 2% below the 3,218 of the same configuration at 78 minutes.</p>
<p><b>What the agent checked.</b> The snapshot of the 3,218 row, run again through the benchmark script later: 3,125. Its own timing loop had given 3,110 to 3,130 for this configuration all along, also at the time of that row.
Only the processes of two minutes around that measurement were about 3% faster (2,365 microseconds of GPU time from the launch, against 2,408 to 2,450). A GPU that had been idle and cool made no difference, and neither did an empty compile cache.
Its summary: "I did not find the cause."</p>
<p>So the record says 3,218 for a state that passes every check, and the level this configuration held afterwards is about 3,150. Both are on the page.</p>"""),
    step("vm-7", "A setting: more launch configurations for the generated kernels", "3,353 and 3,159 images/s at 254 and 259 minutes: no gain", """
<p><code>torch._inductor.config.max_autotune_pointwise = True</code> lets Inductor try more launch configurations for its own kernels. In the agent's loop the batch norm and LeakyReLU kernels took 553 microseconds instead of 577 to 580, +0.7% over two rounds.
In the official rows nothing shows (fast 3,353 after 3,355; safe 3,159 after 3,152). It stayed in the configuration because it never measured slower. On the other side the same setting measured 0.987.</p>"""),
    step("vm-rest", "What did not help, what was not built, and where a call's time is", "", """
<p><b>Tried and dropped</b>, with the agent's figures: <code>cudnn.benchmark_limit = 0</code> (convolutions 1,519 against 1,525 microseconds); <code>cudnn.benchmark = False</code> (1,805 images/s);
the kernels as two CUDA graphs, the second started while the first runs (2,388 to 2,410 microseconds per call against 2,356); the first convolution written out so that Inductor fuses it with its batch norm and LeakyReLU into one generated kernel
(718 microseconds against 102 + 67); the deep convolutions in PyTorch's own layout to save the weight copies (58 against 30 + 12); a channels-last input buffer (3,125 against 3,164 images/s);
a batch norm handed a precomputed scale and shift (no change: those kernels are bound by memory traffic).</p>
<p><b>A fix outside the benchmark.</b> The agent's own probe registered a forward hook and removed it again. The short path recorded while the hook stood kept being tried first, so the full path was never used again: 804 images/s.
The engine was like that before the run. Now a path with more operations than the one that ran last is tried first again after 16 calls, then 32, up to 1,024. The agent added 11 tests for what it built.</p>
<p><b>Not built.</b> A <code>hand_out</code> that is safe: leave a result that is still held its memory, and record the graph again on new memory. In the agent's development copy this passed its whole probe at 3,182 images/s, 25 minutes before the deadline;
with no time left for tests, a measurement and a check, it stayed out of the tree. And a path through <code>zip(*list)</code>, so that the final concatenation is inside the graph.</p>
<p><b>Where a call's time is</b>, in the agent's breakdown after step 4, for the safe configuration. A call takes 2,490 microseconds and the GPU works for 2,486 of them. The graph is on its way 125 microseconds after the call arrives; the engine's walk of the path (515 microseconds) and the CPU's part of copying the results out (110) run while the GPU works.
On the GPU: convolutions 1,520 to 1,540, batch norm and LeakyReLU kernels 520 to 575, weight copies 116, copies of the results about 40.</p>"""),
])

APPENDIX = INTRO + TC + VM
CSS = """
h4.step { margin:1.9em 0 .2em; font-size:1.02em; scroll-margin-top:12px; }
h4.step .gain { color:var(--ink2); font-weight:400; font-size:.92em; margin-left:.5em; }
td.state-pass { white-space:nowrap; }
tr.struck td.qps { text-decoration:line-through; color:var(--muted); }
tr.refused td.qps { color:var(--muted); }
.mk { display:inline-block; width:12px; height:12px; vertical-align:-1px; }
"""
# Which appendix entry each recorded row belongs to, and its short name in the tables and tooltips.
ANCHOR = {
    "tc": ["tc-0", "tc-1", "tc-2", "tc-3", "tc-u", "tc-4", "tc-u", "tc-u", "tc-5", "tc-6", "tc-7", "tc-u", "tc-u", "tc-u", "tc-8", "tc-u", "tc-8"],
    "vm": ["vm-0", "vm-1", "vm-2", "vm-3", "vm-4", "vm-5", "vm-safe", "vm-5", "vm-5", "vm-safe", "vm-safe", "vm-5", "vm-safe", "vm-7", "vm-7", "vm-5", "vm-safe", "vm-5",
           "vm-safe", "vm-safe", "vm-safe"],
}
SHORT = {
    "tc": ["start", "CUDA graphs, results copied out", "heads' concatenation unfused; cheaper guard check", "max pool one dimension at a time; hooks followed; a graph per shape",
           "unchecked variant 1: weights built in, only tensor guards, no copies", "pool passes as unrolled kernels", "unchecked variant 2, first attempt (refused: its filter dropped every guard)",
           "unchecked variant 2: guards only on the call's arguments", "a fourth input channel for the first convolution", "filter counts rounded up to a multiple of 8",
           "one input copied with copy_", "unchecked variant 2 on this source", "unchecked variant 3: CUDA graphs without the tree bookkeeping",
           "unchecked variant 4: every concatenation a kernel of its own", "the fix: input padded only where it is copied anyway", "unchecked variant 4 on the corrected source",
           "(the same state, measured again)"],
    "vm": ["start", "CUDA graph around what Inductor compiled, as a new option", "the graph started ahead of the engine's checks", "pass: max pools as two one-dimensional pools",
           "values a compiled graph keeps: detection convolutions no longer results", "weight layouts kept (gives up writes through .data)", "(safe configuration on that source; tests added)",
           "weight layouts kept, large results handed out", "fast configuration, both as options", "(safe configuration)", "(safe configuration, GPU idle and cool)", "(fast configuration again)",
           "(safe configuration)", "fast configuration + max_autotune_pointwise", "safe configuration + max_autotune_pointwise", "(fast configuration again)", "(safe configuration again)",
           "(fast configuration again)", "(safe configuration again)", "(safe configuration again)", "(safe configuration, last row)"],
}
