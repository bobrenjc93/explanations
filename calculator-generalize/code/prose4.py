"""Fourth write-up: three attempts to make the calculator generalise."""
import re
import numpy as np
import mathgen3 as m3
import figs4 as G
import figs5 as F
from charts import pct, table, esc, figure

TITLE = "Does the calculator generalise? Gaps, a wider pool, and a procedure for sine"
DESCRIPTION = ("Three tests on a small calculator transformer that scores 98% inside its training range and 27% one size beyond it: sizes left "
               "out in the middle of the range, a wider pool tested two sizes up, and sine computed by argument reduction instead of recalled "
               "from a table.")
PREV = '<a href="../calculator-harder/">the harder benchmark</a>'
B, A = F.BEFORE, F.AFTER
SIX = list(range(F.OLD, F.NL - 1))


def body():
    if not F.have():
        return f"<h1>{TITLE}</h1><p>Results are not in yet.</p>"
    parts = [head(), tldr(), setup(), gaps(), edge(), sine(), meaning(), compute(), repro(), FOOT]
    return "\n".join(parts)


def head():
    return (f"<h1>{TITLE}</h1>\n"
            f'<p class="sub">Follow-up to {PREV}, which ended with a model that scores 97.7% inside the sizes it trained on, 27% one size '
            "beyond them, and knows sine as a table of 1,999 values. The question left open there was whether these models generalise at all. "
            "Three tests, run as one fine-tune: leave sizes out in the middle of the range, widen the range and test past the new edge, and "
            "replace the sine table with a procedure.</p>")


def tldr():
    fa, fb = F.final(B), F.final(A)
    g, n = F.gap_summary()
    s1b, s1a = F.mean_em(B, "stretch", SIX), F.mean_em(A, "stretch", SIX)
    s2b, s2a = F.mean_em(B, "stretch2"), F.mean_em(A, "stretch2")
    sb, sa = F.sin_range_stats("before"), F.sin_range_stats("after")
    b = []
    b.append(f"<li><strong>Between the sizes it has seen: yes.</strong> Sizes left out of training in the middle of the range (a 9-term sum when 8 "
             f"and 10 were trained) are solved {pct(g)} of the time, against {pct(n)} for the sizes on either side.</li>")
    b.append(f"<li><strong>Past the largest size it has seen: the edge moves, it does not go away.</strong> After the stretch sizes are added to "
             f"training they go from {pct(s1b)} to {pct(s1a)}. The sizes one step beyond the new pool go from {pct(s2b)} to {pct(s2a)}, about "
             f"where one size up stood before the pool was widened ({pct(s1b)}).</li>")
    b.append(f"<li><strong>Sine by procedure works where it was trained and fades beyond, instead of stopping dead.</strong> With argument "
             f"reduction written into the thinking block and trained for |x| up to 100, exact match on sin(x) between 10 and 100 goes from "
             f"{pct(np.mean([sb[1][1], sb[2][1]]))} to {pct(np.mean([sa[1][1], sa[2][1]]))}, on both sides of zero, which is the ceiling of the procedure. "
             f"Between 100 and 300 it is {pct(sa[3][1])}, and beyond 300 {pct(sa[4][1])}, where the table had {pct(sb[3][1])} and {pct(sb[4][1])}.</li>")
    d = [100 * (y - x) for x, y in zip(fa["final"]["em"], fb["final"]["em"])]
    w = min(range(len(d)), key=lambda l: d[l])
    ma, mb = float(np.mean(fa["final"]["em"])), float(np.mean(fb["final"]["em"]))
    if mb >= ma - 0.002:
        b.append(f"<li><strong>The original range is unharmed.</strong> The standard 14-level test goes from {pct(ma)} to {pct(mb)}.</li>")
    else:
        b.append(f"<li><strong>The original range pays a little.</strong> The standard 14-level test goes from {pct(ma)} to {pct(mb)}, most of it "
                 f"on {m3.LEVELS[w]} ({d[w]:+.1f} points), whose training rows are now shared with a second size range.</li>")
    b.append("<li><strong>So: these models interpolate over size and extrapolate poorly.</strong> What looks like a learned procedure is a "
             "procedure for the sizes covered, plus a fading margin. Each of the three fixes worked about as far as its training data went.</li>")
    return '<div class="tldr"><p><strong>TL;DR.</strong></p><ul>' + "".join(b) + "</ul></div>"


def setup():
    fb = F.final(A)
    o = ["<h2>1. One run, three tests</h2>"]
    o.append(f"<p>The starting point is the 3.2M-parameter model from {PREV} after 30,000 steps: 97.7% on the 14-level test, 26.6% on the stretch "
             f"set (every tier one size past its training range). It is fine-tuned for {fb['args']['steps']:,} more steps on a <em>wide pool</em> "
             "that differs from the original in three ways.</p>")
    o.append("<ul><li><strong>The stretch sizes are added.</strong> Every tier is drawn half from its original range and half from one size up. "
             "The 2,100 stretch-set problems themselves are kept out, so that set is still a test.</li>"
             "<li><strong>A few sizes are left out on purpose.</strong> Long expressions never have 7 or 12 operations, linear equations never "
             "6 or 10 terms, sums never 9 or 17 terms. Those sizes sit inside the range, with trained sizes on both sides.</li>"
             "<li><strong>Sine and cosine get a procedure.</strong> In the original pool sin(x) beyond the 0.01 grid on [-9.99, 9.99] was a "
             "recalled value. Now the worked solution reduces the argument first, and half of the direct sin and cos problems have |x| between "
             "10 and 100, both signs.</li></ul>")
    o.append(F.pool_table())
    o.append("<p>The reduction is how a calculator does it. Divide x by 2&pi; to one decimal and round to the nearest whole number of turns k; "
             "subtract k times 2&pi;, with 2&pi; split as 6.28 + 0.00318531 so the difference keeps its digits; then take the sine of what is "
             "left, which is now inside the table's range:</p>")
    o.append("<pre><code>sin(37.42)=[37.42/6.28319:5p3141595r600405,9p5654871r349179~5.9; k=6; 6*6.28=37.68; 37.42-37.68=-0.26;\n"
             "            6*0.00318531=0.0191119; -0.26-0.0191119=-0.279112; sin(-0.28)=-0.276356; cos(-0.28)=0.961056;\n"
             "            0.000888*0.961056:...=0.000853418; -0.276356+0.000853418=-0.275503; sin(-0.279112)=-0.275503;\n"
             "            sin(37.42)=-0.275503]-0.2755</code></pre>")
    c = fb["compute"]
    cut = (" The machine restarted under the run three times; it was resumed from its step-7,000 checkpoint, and the last 3,000 steps drew "
           "from a fresh 3.75M-row sample of the same generator." if fb["args"]["steps"] >= 10000 else
           " The run was planned for 10,000 steps with a decaying rate at the end; the machine "
           "restarted three times under it, and what is reported is its last saved checkpoint, at step 7,000 and still at the full rate.")
    o.append(f"<p>The pool has 15M rows, a mean solution of {c['mean_target_tokens']:.0f} tokens and solutions of up to 4,096. Fine-tuning used half "
             f"the original learning rate.{cut} Three test sets sit outside it: the stretch set (now inside the pool's sizes, problems unseen), a "
             "<em>gap</em> set of 300 problems per tier at the left-out sizes, and a <em>two sizes up</em> set of 200 per tier beyond the wide "
             "pool's largest size.</p>")
    return "\n".join(o)


def gaps():
    g, n = F.gap_summary()
    o = ["<h2>2. Sizes left out in the middle</h2>"]
    o.append("<p>If the model had learned one way of adding nine terms and another of adding ten, a size it never trained on would fail even "
             "with trained sizes on both sides. It does not fail.</p>")
    o.append(F.gap_table())
    o.append(f"<p>Across the six left-out sizes exact match is {pct(g)}, and {pct(n)} at the sizes next to them. Whatever the model has learned "
             "about the number of terms, it is not a separate routine per count. Between sizes it has seen, it generalises.</p>")
    return "\n".join(o)


def edge():
    fa, fb = F.final(B), F.final(A)
    s1b, s1a = F.mean_em(B, "stretch", SIX), F.mean_em(A, "stretch", SIX)
    s2b, s2a = F.mean_em(B, "stretch2"), F.mean_em(A, "stretch2")
    t2 = F.set_em(A, "stretch2")
    best = max(t2, key=t2.get)
    worst = min(t2, key=t2.get)
    o = ["<h2>3. Past the edge</h2>"]
    o.append(f"<p>The previous write-up guessed that training on the stretch range would teach that range and leave the next one out of reach, "
             "and said it had not run it. This is that run.</p>")
    o.append(F.fig_edges())
    o.append(F.before_after_table())
    o.append(f"<p>The sizes that were added are learned: the stretch set goes from {pct(s1b)} to {pct(s1a)} on the six tiers, on problems that were "
             f"never in the pool. The sizes one step past the new pool go from {pct(s2b)} to {pct(s2a)}: best on {m3.LEVELS[best]} ({pct(t2[best])}), "
             f"worst on {m3.LEVELS[worst]} ({pct(t2[worst])}). The curves show where the fall is.</p>")
    o.append(F.fig_sizes())
    sa, sb = F.size_acc(A, 12), F.size_acc(B, 12)
    gone = lambda acc, edge: next((s for s in sorted(acc) if s > edge and all(acc[t][0] <= 0.05 for t in acc if t >= s)), None)
    half = lambda acc, edge: next((s for s in sorted(acc) if s > edge and acc[s][0] < 0.5), None)
    o.append(f"<p>For sums the picture is the same as before, moved over. The original model is at {pct(sb[12][0], 0)} for 12 terms, the most it "
             f"had trained on, first drops below half at {half(sb, 12)} and has nothing left from {gone(sb, 12)} on. The fine-tuned one is at "
             f"{pct(sa[24][0], 0)} for 24 terms, first drops below half at {half(sa, 24)} and has nothing left from {gone(sa, 24)} on. There is a "
             "margin of a few sizes past the edge in both cases, and then nothing. Long expressions and linear equations show the same shape: "
             "a wider pool buys a wider range and the same slope at its end.</p>")
    return "\n".join(o)


def sine():
    sb, sa = F.sin_range_stats("before"), F.sin_range_stats("after")
    o = ["<h2>4. Sine with a procedure</h2>"]
    o.append(f"<p>In {PREV} the model's sine was exact on the 1,999 grid points it was trained on and about 1% right outside them. For "
             "positive x it landed near the right value where training had left scattered samples; for negative x it had nothing. The table "
             "could not be extended, because nothing in a table says the function repeats. A procedure does say it. The probe is the same as "
             "before: <code>sin(x)</code> on a regular grid, now out to &plusmn;1,000.</p>")
    o.append(F.fig_sin_curve())
    o.append(F.fig_sin_ranges())
    o.append(F.sin_table())
    neg_b, neg_a = F.sin_side("before", 10, 100, -1), F.sin_side("after", 10, 100, -1)
    o.append(f"<p>Inside the range the procedure was trained on, it works: {pct(sa[1][1])} between 10 and 40 and {pct(sa[2][1])} between 40 and 100, "
             f"where the table had {pct(sb[1][1])} and {pct(sb[2][1])}. The ceiling is not 100%: the first-order correction after the reduction is "
             f"good to about five digits, and a perfect solver following it gets {pct(sa[1][3], 0)} to four. The negative side, which had nothing, "
             f"goes from {pct(neg_b)} to {pct(neg_a)}: the reduction handles the sign, so there is no symmetry to discover.</p>")
    o.append(f"<p>Past 100 the model has to do the same steps on a number with one more digit: a quotient with two or three digits before "
             f"the point, and a k of up to 159. Exact match is {pct(sa[3][1])} between 100 and 300 and {pct(sa[4][1])} beyond 300 "
             f"(within 0.1 of the true value: {pct(sa[3][2], 0)} and {pct(sa[4][2], 0)}). That is a long way below the trained range and a long way "
             f"above the table, which had {pct(sb[3][1])} and {pct(sb[4][1])} there. A procedure does not stop dead at the edge of its training the "
             "way recall does: the long division and the multiplication by k are skills the model has at other sizes, and some of that carries. "
             "But it fades with every extra digit, the same way the big-number tier did.</p>")
    return "\n".join(o)


def meaning():
    o = ["<h2>5. What this says</h2>"]
    o.append("<p>The three tests give one answer. These models generalise <em>between</em> the cases they have trained on and not "
             "<em>beyond</em> them, and that holds whether the case is a problem, a size, or the range of an argument:</p>")
    o.append("<ul><li>A held-out problem of a trained size: solved (the 98% of the previous write-up).</li>"
             "<li>A held-out size between trained sizes: solved (section 2).</li>"
             "<li>A size past the largest trained size: a margin of a few sizes, then nothing, wherever the largest size happens to be "
             "(section 3).</li>"
             "<li>A function recalled from a table: exact on the table, approximate between scattered samples, nothing elsewhere.</li>"
             "<li>The same function computed by a procedure: at the procedure's ceiling as far as its inputs were trained, then fading as "
             "the inputs outgrow them (section 4).</li></ul>")
    o.append("<p>The worked solutions look like algorithms, and inside the trained range the model executes them like one. But an algorithm "
             "for adding terms does not care whether there are 24 or 25, and the model does. What it has is closer to a very good interpolator "
             "over worked solutions: dense where the pool is dense, smooth across small holes, and empty past the boundary. That makes the "
             "practical recipe simple and a little disappointing. To get a range, put it in the pool. A procedure in the thinking block is "
             "what makes a range learnable at all (the table could never have covered sine out to 100), and it is not what makes it open-ended.</p>")
    o.append("<p>What was not tried: positional schemes built for length generalisation, index hints inside the solutions, and randomised "
             "positions, which are the usual suspects for exactly this failure in the arithmetic-transformer literature. They change the model; "
             "everything here changed only the data.</p>")
    return "\n".join(o)


def compute():
    fa, fb = F.final(B), F.final(A)
    c = fb["compute"]
    o = ["<h2>6. Compute</h2>"]
    pf = fb["args"]["steps"] * c["padded_positions_per_step"] * c["layer_applications"] * 3 * G.fpt(fb["model"]["d"]) / 1e15
    o.append(f"<p>The fine-tune is {fb['args']['steps']:,} steps at batch 2,048 on rows averaging {c['mean_target_tokens']:.0f} target tokens: "
             f"{c['padded_positions_per_step']/1e3:,.0f}k positions per step, {pf:.0f} PFLOP, {fb['train_time']/3600:.1f} hours on one H100. The "
             f"30,000 steps before it were 687 PFLOP. A wider pool costs in proportion to its solutions: the same step count on the original pool "
             f"is {229 * fb['args']['steps'] / 10000:.0f} PFLOP.</p>")
    return "\n".join(o)


def repro():
    return """
<h2>7. Reproduce</h2>
<pre><code>python build_stretch.py hard/gap.json 300 gap                 # the left-out sizes
python build_stretch.py hard/stretch2.json 200 stretch2       # two sizes up
python build_data3.py 1000000 hard/wide.npz full 1,1,1,1,1,1.5,1.5,1,1,1,1,1,1,1 wide
python train3.py --name W_wide_3m --data hard/wide.npz --init runs/hard/H_full_3m_30k.pt --steps 10000 --lr 5e-4 --ckpt_every 1000
python eval_stretch.py runs/hard/W_wide_3m.pt hard/stretch.json stretch      # also: gap.json gap, stretch2.json stretch2
python sin_probe.py runs/hard/sin_wide_fine.json range=-13000:13000:5 after=runs/hard/W_wide_3m.pt before=runs/hard/H_full_3m_30k.pt
python sin_probe.py runs/hard/sin_wide_far.json range=-100000:100000:37 after=... before=...</code></pre>
"""


FOOT = """
<footer>Written with an AI assistant (Claude), which also designed and ran the experiments. Code and per-run results:
<a href="https://github.com/bobrenjc93/explanations/tree/main/calculator-generalize/code">calculator-generalize/code</a>.</footer>
"""
