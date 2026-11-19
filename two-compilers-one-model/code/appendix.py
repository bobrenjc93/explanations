"""appendix.py: the appendix of the page: every recorded step of both sides, explained. The numbers quoted inside a
step (microseconds per kernel, milliseconds per call) are the agent's own profiles from its journal; the images/s in
each heading are the recorded measurements."""
import html


def code(text):
    return f"<pre><code>{html.escape(text.strip(chr(10)))}</code></pre>"


def step(anchor, title, before, after, body):
    arrow = f' <span class="gain">{before} &rarr; {after} images/s</span>' if after else ""
    return f'<h4 class="step" id="{anchor}">{title}{arrow}</h4>\n{body}\n'


INTRO = """
<h2 id="appendix">Appendix: every step, in detail</h2>
<p>What each agent found, what it changed and why that is faster, in the order of the tables above. The figures inside a step (microseconds per kernel, milliseconds per call)
are from the agent's own profiles, as its journal has them; the images/s in a heading are the recorded measurements before and after.</p>
<p><b>The model, for orientation.</b> One call is a batch of 8 images through 76 convolutions. 73 of them are followed by a batch norm and a LeakyReLU, 23 results are added to an earlier one
(the residual connections), one tensor goes through three max pools with windows of 5, 9 and 13, and three detection heads turn their convolution's output into boxes with a few sigmoids and exponentials;
their three results are concatenated. Compiled, that is about 200 GPU kernels: the convolutions are cuDNN's, everything between them is kernels that Inductor generates, and they take about 2.5 ms together.</p>
<p><b>Three terms that come back.</b></p>
<ul>
<li><b>Host time.</b> Each kernel is launched from the CPU, through Python, PyTorch and the driver, at 10 to 25 microseconds apiece. For 200 small kernels that is more than the GPU needs to run them, so the GPU waits for the CPU.</li>
<li><b>CUDA graph.</b> A recording of all the kernel launches of one call. Replaying it starts them with one call into the driver. The recording is tied to the memory the recorded call used, so each call's input is copied into it
and each new shape needs a new recording.</li>
<li><b>Weights built in.</b> A compiler can treat the model's weights as inputs of the graph (any tensor may show up there next call) or as constants of it. As constants they can be combined ahead of time: an inference-mode batch norm is a
multiply and an add per channel, and folds into the convolution's weights and bias, so the batch norm's arithmetic leaves the call. The price is that the graph is only right while the weights are what they were.
<code>torch.compile</code> calls this freezing and does not notice a weight that is updated in place afterwards; <code>torch.vm</code> does it by default and watches the weights.</li>
</ul>
"""

TC = "".join([
    '<h3 id="appendix-tc">torch.compile</h3>\n',
    step("tc-0", "Start: the dashboard's configuration", "", "", """
<p>2,013 images/s, 3.98 ms per batch. The agent's first profile: a call takes 3.90 ms, the CPU is busy launching kernels for 3.83 ms of it, and the kernels themselves add up to 2.54 ms
(the 76 cuDNN convolutions are 61% of that). The host is the bottleneck, not the GPU.</p>"""),
    step("tc-1", "1. CUDA graphs", "2,013", "2,641", """
<p><b>Change.</b> One option: <code>mode="reduce-overhead"</code>, later written as <code>options={"triton.cudagraphs": True}</code>. Inductor records the compiled graph's kernels as a CUDA graph and replays it.</p>
<p><b>Why it helps.</b> The 200 launches become one. A call goes from 3.9 ms to 3.0 ms, not far from what the kernels themselves take.</p>
<p><b>Cost.</b> The memory of the recorded call stays allocated. Nothing eager does is given up: this state follows every probe.</p>"""),
    step("tc-2", "2. Freezing, and the fix in Dynamo it needed", "2,641", "2,909", """
<p><b>Change.</b> <code>torch._inductor.config.freezing = True</code> builds the weights into the graph: the 73 batch norms fold into their convolutions, and the weights are converted to cuDNN's preferred memory layout (channels last) once, at compile time, instead of being handed over in the other layout on every call.</p>
<p><b>Why it did not work at first.</b> With freezing on, the model fell back to eager (748 images/s): Dynamo recompiled on every call until it hit its limit. Each of the model's three YOLO layers rebuilds a small grid tensor on every call and assigns it to a buffer:</p>
""" + code("""
def forward(self, p, out):
    bs, _, ny, nx = p.shape
    self.grid = self.create_grids((nx, ny), p.device)   # a registered buffer, replaced on every call
""") + """
<p>Freezing guards the identity of every tensor it builds in, so that a different tensor in its place triggers a recompilation. Assigning to a buffer goes through <code>nn.Module.__setattr__</code>, which first looks the name up;
in Dynamo that lookup took the old buffer's value far enough to install its identity guard, although the code never reads the old buffer, it only overwrites it. The guard then failed on every call, because the buffer is a new tensor every time.</p>
<p><b>The fix</b> is 11 lines in <code>torch/_dynamo/variables/user_defined.py</code>: when that lookup finds a value that has not been used yet and is not a submodule, leave it unused, so no guard is installed for it. On a small reproduction that is one graph where there were six.</p>
<p><b>What it gives up.</b> The compiled code computes with the weights as they were at compile time. The probes confirm it: an in-place update of a convolution's weight, a batch norm's bias or running mean is not followed.</p>"""),
    step("tc-3", "3. Max pool one dimension at a time, in Inductor", "2,909", "3,051", """
<p><b>What the profile said.</b> The three max pools took 19, 26 and 99 microseconds: 145 of the 2,430 the kernels take. Inductor computes each output as the maximum over the whole window, so the 13-window reads 169 values for every output element.</p>
<p><b>Change</b>, in <code>torch/_inductor/lowering.py</code>: the maximum over a rectangle is the maximum over its rows of the maximum along each row. So when only the pooled values are used (not the positions of the maxima), the pool is computed one dimension at a time:
13 values along one dimension, written to memory, then 13 along the other. That is 26 reads and one extra write per element instead of 169. It applies only where it at least halves the reads, and the path that also returns indices is unchanged.</p>
<p><b>Second change, a setting.</b> Inductor can compute a <code>torch.cat</code> in two ways: as a kernel of its own into which each part is written (a "concat kernel"), or fused into one kernel with the arithmetic of its parts, where every output element first works out which part it belongs to.
The fused form was the slowest kernel that is not a convolution, for the model's final concatenation of the three heads. <code>max_pointwise_cat_inputs = 1</code> makes every concatenation a kernel of its own. (Step 6 refines this.)</p>"""),
    step("tc-4", "4. Guard filter (unsafe)", "3,051", "3,270", """
<p><b>What the profile said.</b> Of the 0.32 ms the CPU still spends per call, launching the CUDA graph is 58 microseconds and copying the input 17. The rest, about 250, is Dynamo's guards and the Python around them.
Before it runs compiled code, Dynamo checks that everything the code assumed still holds; for this model that is about 4,700 checks on the attributes of its 225 modules (is this module still in eval mode, is this still the same parameter, are there still no hooks).</p>
<p><b>Change.</b> <code>torch.compile(..., options={"guard_filter_fn": torch.compiler.skip_guard_on_all_nn_modules_unsafe})</code> drops the guards on module attributes. The guard check goes from 149 microseconds to 3.</p>
<p><b>What it gives up</b>, as the name says: a change to a module after compilation is not noticed. In the probes, <code>model.train()</code> and a replaced parameter are not followed. The agent tried two sound alternatives first
(<code>use_recursive_dict_tags_for_guards</code>: 149 to 126 microseconds; skipping only the guards of PyTorch's built-in modules: 59 microseconds, 3,180 images/s) and recorded two reference rows without the filter at the end: it is worth about 8%.</p>"""),
    step("tc-5", "5. Pool stages as unrolled kernels", "3,270", "3,328", """
<p>The two stages of the 9- and 13-wide pools from step 3 were generated as reduction kernels: a loop with a running maximum for each output element, 16 microseconds each. For a window this short, a kernel that simply reads the 9 or 13 values and takes their maximum
in one expression is faster. Inductor already does that below a threshold (<code>unroll_reductions_threshold</code>, 8 by default); the lowering now raises it to 26 for these stages.</p>"""),
    step("tc-6", "6. Concatenation and in-place settings", "3,328", "3,420", """
<p><b>Two settings.</b> <code>max_pointwise_cat_inputs = 2</code>: concatenations of two parts are fused with their parts again (the model has small ones where that is cheaper), those of three or four stay kernels of their own.
<code>inplace_buffers = False</code>: by default a generated kernel may write its result over its input when nothing else needs the input, which is what the bias-and-LeakyReLU kernel after each convolution did. Writing to a fresh buffer measured faster here:
the largest such kernel (the first layer's, 100 MB) took 65 microseconds instead of 91. The agent measured this and did not find out why.</p>"""),
    step("tc-7", "7. A fourth input channel for the first convolution", "3,420", "3,453", """
<p><b>What was measured.</b> The first convolution reads images with 3 channels, and cuDNN's half-precision kernels are slow for exactly that: 101 microseconds, against 74 when the input has a fourth channel of zeros (8 channels: no further gain; 16: slower).</p>
<p><b>Change</b>, in <code>torch/_inductor/kernel/conv.py</code>: a half-precision, channels-last convolution with 3 input channels gets a zero channel appended to its input and zero weights for it. A zero times a zero adds exactly nothing to any sum;
what changes is which kernel cuDNN picks, and the agent's checks agree with eager to within one unit in the last place. The input is padded in the kernel that already converts it to channels last, and a built-in weight is padded once at compile time.</p>"""),
    step("tc-8", "8. A smaller inlining threshold", "3,453", "3,467", """
<p>Inductor keeps an expression inside the kernels that use it until it has more than 30 operations, and only then computes it into memory of its own. Each detection head ends with four assignments to slices of one tensor:</p>
""" + code("""
io[..., :2] = torch.sigmoid(io[..., :2]) + self.grid
io[..., 2:4] = torch.exp(io[..., 2:4]) * self.anchor_wh
io[..., :4] *= self.stride
torch.sigmoid_(io[..., 4:])
""") + """
<p>Each one wraps the expression before it in a condition ("if this element is in the slice, the new value, else the old one"), so the inlined expression doubles four times, and an element can evaluate a sigmoid or an exponential several times.
<code>realize_opcount_threshold = 20</code> cuts the chain earlier. The gain is within the noise of the measurement.</p>"""),
    step("tc-10", "9. Autotune: Inductor's own kernels for the 1x1 convolutions", "3,467", "3,806", """
<p><b>Change.</b> <code>torch._inductor.config.max_autotune = True</code>. For every convolution and matrix product, Inductor generates candidates (its own Triton kernels in several tile sizes, next to cuDNN's and cuBLAS's), times them while compiling, and keeps the fastest. Compiling takes about 50 s longer.</p>
<p><b>What it picked.</b> A 1x1 convolution is a matrix product. For 36 of the model's 38 such convolutions Inductor's own kernel won, and that kernel adds the bias and applies the LeakyReLU before it writes its result, so the separate pass over the output afterwards is gone.
The 38 3x3 convolutions stayed with cuDNN, which still needs that second pass. Per call afterwards: cuDNN's 3x3 convolutions 1,143 microseconds, the 36 generated kernels 317, the bias-and-LeakyReLU passes after cuDNN 383.</p>
<p><b>Only with freezing.</b> The agent had tried the same option at the start, with the weights as inputs, and it gave nothing (2,659 against 2,653).</p>"""),
    step("tc-13", "10. CUDA graphs without the tree bookkeeping", "3,816", "3,973", """
<p>Inductor's default CUDA-graph support keeps a tree of recordings that share one pool of memory, tracks which results of earlier calls are still alive, and raises an error when a program touches a result that a later call has overwritten.
That costs about 40 microseconds per call. <code>"triton.cudagraph_trees": False</code> selects the older path with one recording per graph: the CPU's share of a call goes from 0.160 to 0.125 ms.</p>
<p><b>What it gives up.</b> Results are fixed buffers that the next call overwrites without a word. A program that keeps a result across calls reads the wrong numbers instead of getting an error.</p>"""),
    step("tc-rest", "The rows that are not steps", "", "", """
<p><b>Hardening</b> (3,446): the agent reviewed its convolution change and added a case it had missed (a built-in weight that lives on another device than it is used on is padded at run time). No change in speed was expected.
<b>Reference rows</b> (3,539 and 3,664): the two best configurations measured again without the guard filter, so that the result has points that do not depend on it. <b>Repeats</b> (3,816 and 3,887): the same state measured again.
The last rows differ by up to 8% between processes; the agent traced that to cuDNN timing each convolution algorithm once per process and sometimes picking a slower one, and did not fix it.</p>
<p><b>Tried without gain:</b> kernel autotuning before freezing; coordinate-descent tuning of the generated kernels; 1x1 convolutions as matrix products without autotune; padding the first convolution to 8 or 16 channels;
other inlining thresholds; realizing the detection head's chain inside the lowering (the head's kernels got faster, the total did not).</p>"""),
])

VM = "".join([
    '<h3 id="appendix-vm">torch.vm</h3>\n',
    step("vm-0", "Start: the dashboard's configuration", "", "", """
<p>784 images/s: eager's speed, after 130 s of warm-up. <code>torch.vm</code> hands stretches of tensor code ("regions") to Inductor as graphs. <code>torch.vm.regions(fn)</code> showed that the region around the whole model had been refused,
with the reason "<code>__setattr__</code> on a module the region was given": the three YOLO layers assign a grid and a size to buffers of theirs on every call (the code under step 2 of <code>torch.compile</code>), and a region allowed such a store into a plain attribute but not into a registered buffer.
Without the outer region the model fell apart into a region per layer, 73 graphs of a handful of operations, only 8 of them with weights built in. The VM then measured those against the interpreter, found them no faster, and handed the model's loop back to the interpreter.</p>"""),
    step("vm-1", "1. Stores into module buffers allowed: one graph", "784", "1,983", """
<p><b>Change</b>, in <code>torch/vm/_regions.py</code>: a region may store a tensor it has computed into an existing buffer of a module, the way it already could into a plain attribute. The graph returns the tensor, and the region does the store after the graph has run, so the module ends up holding what eager would have left there.</p>
<p><b>Effect.</b> The whole model is one region: one Inductor graph of 645 operations with the weights built in, compiled in 20 s. That is where <code>torch.compile</code> started (2,013).</p>"""),
    step("vm-2", "2. CUDA graphs, built as a new option", "1,983", "2,456", """
<p><b>What the profile said.</b> The same as on the other side: a call returns after 3.9 ms while its kernels need 2.4. <code>torch.vm</code> had no CUDA-graph support, so the agent wrote it: <code>Graphs("inductor", cuda_graphs=True)</code>.</p>
<p><b>How it works.</b> The compiled code runs the first call. The second call is recorded as a CUDA graph and replayed once, and the recording is kept only if the replay gives the code's result bit for bit, writes to no input,
and returns nothing that shares memory with an input or another result. Otherwise the compiled code goes on running. Each call's input is copied into the recording and each result is copied out, so a caller gets tensors of its own, as from eager.</p>
<p><b>When it does not apply.</b> The graph must run on one GPU, with sizes that are all known, without operations whose work depends on a tensor's contents and without random numbers; and the code runs instead of the recording when gradients are being recorded,
under a torch function or dispatch mode, while the caller is recording a CUDA graph of its own, or while another thread is replaying. The option is off by default; its cost is the recorded call's GPU memory.</p>"""),
    step("vm-3", "3. The fast path kept after the region's own stores", "2,456", "2,863", """
<p><b>Background.</b> Before a region runs its graph, it has to establish that this call is like the one the graph was made for: the same modules, holding tensors of the same shapes and types, the same settings. Doing that from scratch takes about 0.6 ms of Python for this model.
There is a fast path: if nothing in the modules has changed since the previous call, which the runtime can tell in one pass (about 40 microseconds), the previous answer stands.</p>
<p><b>The problem.</b> This region changes the modules itself: it stores a new grid in the YOLO layers on every call (step 1). So every call found the modules changed and took the 0.6 ms path, before any kernel started.</p>
<p><b>Change.</b> After a call that stored tensors into modules, the region establishes the answer again right away, for the modules as they now are. On a GPU that is after the kernels have been launched and while they run, so the 0.6 ms no longer delays anything;
the next call finds the modules as that look left them and takes the fast path.</p>"""),
    step("vm-4", "4. The weights check moved behind the kernel launch", "2,863", "3,108", """
<p><b>Background.</b> Because <code>torch.vm</code> builds the weights into the graph, it has to make sure on every call that none of the 290 built-in tensors has been written to or replaced since; that is what lets it follow a weight update where freezing does not.
For the same reason as in step 3 this check ran the slow way on every call: 0.20 of the 0.27 ms that passed before the kernels were launched.</p>
<p><b>Change.</b> The check for the next call is done behind the launch too, while the GPU is busy, and the next call only has to confirm that nothing was touched in between, which the fast path of step 3 already tells it.
The time before the launch goes from 0.27 to 0.07 ms.</p>
<p><b>Is it still safe?</b> This is the step that could have cost correctness: a weight written between two calls must still be noticed. The probes were run on this checkpoint for that reason, and every one is followed.</p>"""),
    step("vm-5", "5. Pass: max pools of 5, 9 and 13 as a cascade of 5s", "3,108", "3,205", """
<p><b>What a pass is.</b> A function in the user's configuration that <code>torch.vm</code> calls with the graph of a region before the graph goes to Inductor. This one is 50 lines.</p>
<p><b>The rewrite.</b> The model pools one tensor three times, with windows of 5, 9 and 13, all at stride 1. At stride 1, the maximum over a window of 9 is the maximum over a window of 5 of the maxima over windows of 5 (5 + 5 - 1 = 9), and 13 is one more round. So the pass computes
the 5-pool, feeds its result to another 5-pool to get the 9, and that to a third to get the 13:</p>
""" + code("""
p5  = max_pool(x,  5)        # was: p5  = max_pool(x, 5)
p9  = max_pool(p5, 5)        #      p9  = max_pool(x, 9)     81 values per output
p13 = max_pool(p9, 5)        #      p13 = max_pool(x, 13)   169 values per output
""") + """
<p>A maximum involves no rounding, so the results are the same bits. The three pools had taken 145 microseconds. (<code>torch.compile</code>'s agent attacked the same 145 microseconds inside Inductor, step 3 there.)</p>"""),
    step("vm-9", "6. Pass: convolution, batch norm and LeakyReLU as one cuDNN kernel", "3,261", "3,476", """
<p><b>What the profile said.</b> A call was now 2.41 ms of kernels and under 0.2 ms of everything else. 62% of the kernel time is cuDNN's convolutions, and 26% is the kernel after each one that adds the bias and applies the LeakyReLU.
That second kernel does almost no arithmetic: its time is reading the convolution's whole result from memory and writing it back.</p>
<p><b>The idea.</b> If the convolution applied the bias and the activation before writing its result, the second pass over memory would not exist. The cuDNN interface that PyTorch calls can fuse a ReLU into a convolution but not a LeakyReLU.
cuDNN also has a newer interface (its "backend" or graph API) that takes a small graph of operations (convolution, add a bias, a ReLU with a slope for negative values, add another tensor) and builds one kernel for it at run time.
PyTorch does not expose it.</p>
<p><b>Change</b>, all in the configuration file, about 500 lines. The agent called that interface with <code>ctypes</code> on the cuDNN library PyTorch has already loaded (no build), wrapped it as a custom operator
<code>y = leaky_relu(conv(x, w) + b) [+ z]</code>, and wrote a pass that replaces convolution &rarr; batch norm &rarr; LeakyReLU [&rarr; add the residual] in the region's graph by that operator. The batch norm is folded into the weights and bias in float32.</p>
<p><b>Per layer, by measurement.</b> The fused kernel is not always faster: one stride-2 layer took 270 microseconds fused against 63 + 32 apart. So the pass times both ways for each kind of layer when it compiles, and fuses only where that wins.
72 layers were fused at this step; the first layer, close to a tie, was not. Graphs with sizes that are not known are left as they were.</p>"""),
    step("vm-10", "7. The first layer and the detection heads fused too", "3,476", "3,743", """
<p><b>Three refinements of the pass.</b> A layer left unfused between fused ones has its tensors copied between two memory layouts, so the pass now fuses unless the fused kernel is more than 10% slower; that brings in the first layer
(104 + 68 microseconds plus an 81-microsecond layout copy before; 166 after). The three detection-head convolutions, which have a bias but no batch norm or activation, are fused with a slope of 1, which makes the LeakyReLU do nothing.
And those heads have 255 output channels, where cuDNN's fastest kernels want a multiple of 8: the pass adds one all-zero filter to make 256 and cuts the extra channel off afterwards. The largest head went from 61 to 13 microseconds.</p>"""),
    step("vm-11", "8. The fastest of cuDNN's first five kernel variants per layer", "3,743", "3,968", """
<p><b>Background.</b> For each small graph it is given, cuDNN has several implementations, which it calls engines, each with settings of its own. Asked for one, it returns a list of up to ten candidates ordered by a heuristic: its guess of the fastest first, without timing anything.</p>
<p><b>What was measured.</b> The guess is often wrong. For a 3x3 convolution from 128 to 256 channels with the residual add, the first candidate took 35.4 microseconds and the second 31.3; for one from 64 to 128 channels, 61.3 against 42.9 for the third;
for a 1x1, 9.7 against 8.2 for the fifth.</p>
<p><b>Change.</b> When the pass sets a layer up, it runs the first five candidates on tensors of that layer's shapes and keeps the fastest. This is what <code>torch.backends.cudnn.benchmark = True</code> does for the interface PyTorch calls.
The kernels of a call went from 2.02 to 1.88 ms.</p>"""),
    step("vm-12", "9. Pass: the final concatenation as a kernel of its own", "3,968", "4,156", """
<p>The model's result is the concatenation of the three heads' outputs. Inductor had fused it with the heads' arithmetic into one kernel, in which every element of the result evaluates the expressions of all three heads under conditions
(the chain of slice assignments shown under step 8 of <code>torch.compile</code>): 126 reads, 39 sigmoids and 18 exponentials per element, 131 microseconds. The pass replaces a concatenation that is a result of the graph with an operator Inductor cannot fuse through,
so each head is computed for its own elements only and the concatenation is a copy: 61 microseconds. (<code>torch.compile</code>'s agent got the same effect with a setting, steps 3 and 6 there.)</p>"""),
    step("vm-13", "10. Pass: each pool as two one-dimensional pools", "4,156", "4,253", """
<p>On top of the cascade of step 5, each 5x5 pool is split into a 5x1 pool followed by a 1x5 pool: 10 values per output instead of 25, exact for the same reason. This is the rewrite <code>torch.compile</code>'s agent put into Inductor's lowering.</p>"""),
    step("vm-14", "11. Input channels padded to a multiple of 4", "4,253", "4,372", """
<p>The same finding as <code>torch.compile</code>'s step 7, for the fused kernel: with the image's 3 channels padded by a channel of zeros, the first layer takes 116 microseconds instead of 165. The pass pads the input channels of a fused convolution to a multiple of 4, with zero weights for the added ones.</p>"""),
    step("vm-15", "12. The fastest of all ten kernel variants per layer", "4,372", "4,451", """
<p>Step 8 timed the first five of cuDNN's candidates. For the 3x3 convolutions from 128 to 256 channels, of which the model has eight, the tenth turned out fastest: 25.9 microseconds against 31.3 for the best of the first five.
The pass now times all ten. Compiling takes 37 s instead of 17.</p>
<p>The agent then tried the tuning settings inside the winning engine as well (56 values of one, a range of two others): none beat what the best of the ten already had.</p>"""),
    step("vm-16", "13. The input read in place instead of copied", "4,451", "4,505", """
<p>Replaying a CUDA graph meant copying each call's input into the memory the recording reads from: 30 microseconds. A caller that hands over the same memory again and again (one buffer it refills, a few it rotates through) now gets, at the third sighting of an address,
a second recording made on its own tensor, which reads the input where it is. That recording is only made once the first one has shown that the code writes to no input and returns no memory of one, and it is checked by a bit-for-bit replay like the first.
This is a change to the compiler, not the configuration.</p>"""),
    step("vm-rest", "The rows that are not steps", "", "", """
<p><b>Hardening and documentation</b> (3,165, 3,261, 3,234): the CUDA-graph option refuses graphs that draw random numbers (recording and then replaying to check would advance the generator twice) and steps aside when the caller is recording a CUDA graph itself;
the look-ahead of step 4 was restricted to the part of the check that changes nothing; docstrings. These three rows have the same configuration, and show the noise: up to 1.5% from one measurement to the next, 0.3% between the three processes of one.
<b>Steadier timing</b> (4,507): the timings that pick a kernel variant were lengthened, to see whether the remaining spread came from different picks. It did not.</p>
<p><b>Where a call's time is at the end.</b> 1.77 ms. The kernels start about 60 microseconds after the call (40 of them the fast-path look over the modules' 370 tensors) and run for 1.7 ms, 91% of it in the fused cuDNN kernels;
meanwhile the CPU copies the results out (90 microseconds) and prepares the next call (step 3, about 1 ms, hidden behind the GPU here; for a model with a shorter graph it would not be).</p>
<p><b>Tried without gain:</b> coordinate-descent tuning of the generated kernels; convolutions in the other memory layout (3.64 ms against 2.53); 1x1 convolutions as matrix products; splitting the batch in two halves replayed on two streams at once (2.50 ms against 2.45).</p>"""),
])

APPENDIX = INTRO + TC + VM
CSS = """
h4.step { margin:1.9em 0 .2em; font-size:1.02em; scroll-margin-top:12px; }
h4.step .gain { color:var(--ink2); font-weight:400; font-size:.92em; margin-left:.5em; white-space:nowrap; }
"""
# Which appendix entry each recorded row belongs to.
ANCHOR = {
    "tc": ["tc-0", "tc-1", "tc-2", "tc-3", "tc-4", "tc-5", "tc-6", "tc-7", "tc-8", "tc-rest", "tc-10", "tc-rest", "tc-rest", "tc-13", "tc-rest", "tc-rest"],
    "vm": ["vm-0", "vm-1", "vm-2", "vm-3", "vm-4", "vm-5", "vm-rest", "vm-rest", "vm-rest", "vm-9", "vm-10", "vm-11", "vm-12", "vm-13", "vm-14", "vm-15", "vm-16", "vm-rest"],
}
