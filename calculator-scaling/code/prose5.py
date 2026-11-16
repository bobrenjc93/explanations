"""Fifth write-up: scaling laws for the calculator."""
import math, re
import numpy as np
import mathgen3 as m3
import figs4 as G
import figs6 as S
from charts import pct, table, esc, figure

TITLE = "Scaling laws for a calculator: parameters, data, and where it stops"
DESCRIPTION = ("How a small calculator transformer's error falls with parameters, training length, compute and distinct training problems, "
               "where the curves bend, and whether any of it buys problems one size larger than the training range.")
HARDER = '<a href="../calculator-harder/">the 14-level benchmark</a>'
GEN = '<a href="../calculator-generalize/">the generalisation follow-up</a>'
nf = S.nfmt


def body():
    parts, num = [], {}
    for name in ["setup", "size", "length", "data", "asymptote", "stretch", "laws", "repro"]:
        html = globals()[name]()
        if html.strip():
            num[name] = len(num) + 1
            parts.append(re.sub(r"<h2>\d+\. ", f"<h2>{num[name]}. ", html, count=1))
    text = "\n".join([head(), tldr(), pending()] + parts + [FOOT])
    return re.sub(r"@(\w+)@", lambda mm: str(num.get(mm.group(1), "?")), text)


def head():
    return (f"<h1>{TITLE}</h1>\n"
            f'<p class="sub">Follow-up to {HARDER} and {GEN}. The question this time: if the model and the data are scaled up, does performance '
            "follow, is there an asymptote, or can it just keep going? One task, one architecture family, three knobs turned one at a time: "
            "parameters, training length, and the number of distinct problems in the training pool.</p>")


def pending():
    todo = [n for n in S.ALL if not S.have(n)]
    if not todo:
        return ""
    return (f'<p class="note">Runs still training as this version goes up: {len(todo)} of {len(S.ALL)}. The figures show what has finished; the '
            "text and the fits will be updated as the rest come in.</p>")


def tldr():
    b = []
    rs = S.runs(S.SIZE)
    if len(rs) >= 4:
        (a, al), _ = S.size_fit()
        lo, hi = rs[0], rs[-1]
        b.append(f"<li><strong>Parameters.</strong> At a fixed 10,000 steps, in-range error falls from {pct(1 - lo['acc'])} at {nf(lo['N'])} parameters to "
                 f"{pct(1 - hi['acc'])} at {nf(hi['N'])}. Above the floor it follows a power law in parameters with exponent {al:.2f}: ten times the "
                 f"parameters cuts the excess error by a factor of {10 ** al:.1f}.</li>")
    (sf, srs) = S.steps_fit()
    if sf:
        b.append(f"<li><strong>Training length.</strong> For the 3.2M model the excess error falls with steps to the power {sf[1]:.2f}. In this regime "
                 "every batch is new problems, so steps and data seen are the same axis.</li>")
    if len(rs) >= 4:
        (ca, cg), crs, miss = S.compute_fit()
        b.append(f"<li><strong>Compute.</strong> Size and length are close to interchangeable: over all {len(crs)} runs, excess error is training "
                 f"compute to the power -{cg:.2f}, and no run misses that line by more than {100 * (miss - 1):.0f}%.</li>")
    ds = sorted(S.runs(S.DATA), key=lambda r: r["unique"])
    if len(ds) >= 3:
        b.append(f"<li><strong>Distinct data.</strong> With the same compute, a pool of {nf(ds[0]['unique'])} problems gives {pct(ds[0]['acc'])}, "
                 f"{nf(ds[1]['unique'])} gives {pct(ds[1]['acc'])} and {nf(ds[-1]['unique'])} gives {pct(ds[-1]['acc'])}. More distinct problems help "
                 "until each is seen only a few times; past that the run is limited by compute, not data.</li>")
    if len(rs) >= 4:
        b.append(f"<li><strong>The asymptote in range is the scratchpad's, not the model's.</strong> A perfect solver following the worked solutions "
                 f"scores {pct(S.ceiling())}. The curves bend toward that floor and nothing else stops them.</li>")
    st = [r for r in rs if r["stretch_mean"] is not None]
    if len(st) >= 4:
        top = max(st[:-1], key=lambda r: r["stretch_mean"])
        if st[-1]["stretch_mean"] < top["stretch_mean"] + 0.03:
            b.append(f"<li><strong>One size up does have an asymptote, and it is low.</strong> Accuracy on problems one size beyond the training "
                     f"range rises from {pct(st[0]['stretch_mean'])} at {nf(st[0]['N'])} parameters to {pct(top['stretch_mean'])} at {nf(top['N'])} and is no "
                     f"higher at {nf(st[-1]['N'])} ({pct(st[-1]['stretch_mean'])}), while in-range error keeps falling. Scale buys accuracy inside the "
                     "range it is trained on; it does not buy the next range.</li>")
        else:
            b.append(f"<li><strong>One size up scales much more slowly.</strong> Across the same sweep, accuracy on problems one size beyond the "
                     f"training range goes from {pct(st[0]['stretch_mean'])} to {pct(st[-1]['stretch_mean'])}.</li>")
    return '<div class="tldr"><p><strong>TL;DR.</strong></p><ul>' + "".join(b) + "</ul></div>" if b else ""


def setup():
    o = ["<h2>1. What is scaled, and what is measured</h2>"]
    o.append(f"<p>Everything is {HARDER}: 14 levels from two-digit addition to Newton's method on cubics, a thinking block with a full worked "
             "solution before every answer, 2,000 held-out problems per level. The training pool has 28M problems, 2M per level. The model is the "
             "same pre-norm transformer with rotary positions and a looped middle block at every size; only width and depth change.</p>")
    o.append("<ul><li><strong>Parameters:</strong> six sizes from 0.2M to 19.0M (width 64 to 512, 4 layers up to 3.2M and 6 above), each trained "
             "for 10,000 steps of batch 2,048.</li>"
             "<li><strong>Training length:</strong> the 3.2M model for 2,500, 5,000, 10,000 and 30,000 steps, and the 0.8M model for 10,000 and "
             "30,000. Each run has its own warm-up and decay, so these are separate runs and not points read off one curve.</li>"
             "<li><strong>Distinct data:</strong> the 3.2M model for 10,000 steps on a random 0.2%, 2%, 20% and 100% of the pool. The number of "
             "problems drawn is the same 20.5M in every run; what changes is how many different ones there are to draw.</li></ul>")
    o.append("<p>Three things are measured at the end of every run. <em>In-range error</em>: the share of the 28,000 test problems answered "
             "wrong, generated greedily. <em>Held-out loss</em>: cross-entropy per token on the test problems' reference solutions. And "
             f"<em>one size up</em>: exact match on 100 problems per tier from the stretch set of {HARDER}, where every tier's size is one step past "
             "its training range. One seed per run; in earlier write-ups a second seed moved level-averaged accuracy by up to 2 points, so small "
             "differences between neighbouring points mean little and the slopes mean more.</p>")
    o.append(S.run_table())
    return "\n".join(o)


def size():
    rs = S.runs(S.SIZE)
    if len(rs) < 4:
        return ""
    (a, al), _ = S.size_fit()
    o = ["<h2>2. Parameters</h2>"]
    o.append(S.fig_size())
    worst = min(range(S.NL), key=lambda l: rs[-1]["em"][l])
    o.append(f"<p>Error over the 14 levels goes from {pct(1 - rs[0]['acc'])} at {nf(rs[0]['N'])} parameters to {pct(1 - rs[-1]['acc'])} at "
             f"{nf(rs[-1]['N'])}. Measured above the floor, the fall is close to a straight line on log-log axes: a power law with exponent "
             f"{al:.2f}, so each tenfold increase in parameters divides the excess error by about {10 ** al:.1f}. The new tiers start further "
             f"behind ({pct(1 - rs[0]['new'])} error against {pct(1 - rs[0]['old'])} on the old levels at {nf(rs[0]['N'])}) and close the gap as the "
             f"model grows ({pct(1 - rs[-1]['new'])} against {pct(1 - rs[-1]['old'])} at {nf(rs[-1]['N'])}).</p>")
    o.append(S.fig_size_levels())
    small = rs[0]
    dead = [G.SHORT[l] for l in range(S.NL) if small["em"][l] < 0.05]
    o.append(f"<p>By level the picture is a set of thresholds more than a smooth slide. The {nf(small['N'])} model has "
             f"{'nothing on ' + ', '.join(dead) if dead else 'something on every level'}; each tier switches on at its own size and then "
             f"saturates. The last to come in is {m3.LEVELS[worst]} ({pct(rs[-1]['em'][worst], 0)} at {nf(rs[-1]['N'])}). The smooth power law is what "
             "these thresholds add up to.</p>")
    st = S.slip_table()
    if st:
        s0, s1 = S.slips(rs[0]), S.slips(rs[-1])
        o.append("<p>Exact match on a long solution is a harsh measure, because one wrong token anywhere is fatal. Turning it around gives the "
                 "rate the model is actually being asked for. If a solution of T tokens is right with probability (1 - p)<sup>T</sup>, then p is "
                 "the rate of fatal slips per token:</p>")
        o.append(st)
        o.append(f"<p>From {s0:,.0f} fatal slips per million tokens at {nf(rs[0]['N'])} parameters to {s1:,.0f} at {nf(rs[-1]['N'])}. A 1,700-token "
                 "Newton solution is right 99% of the time only if fewer than 6 tokens in a million go wrong, and that is the regime the larger "
                 "models are in. This is why the longest tier is not the hardest: its tokens are easy ones, and the slip rate on easy tokens falls "
                 "fastest.</p>")
    fl = S.fig_loss_size()
    if fl:
        ls = [r for r in rs if r["loss"]]
        (la, lal) = S.fit_power([r["N"] for r in ls], [r["loss"] for r in ls])
        o.append(fl)
        o.append(f"<p>Held-out loss tells the same story on the axis scaling laws are usually stated on: {ls[0]['loss']:.3f} nats per token at "
                 f"{nf(ls[0]['N'])} down to {ls[-1]['loss']:.4f} at {nf(ls[-1]['N'])}, an exponent of {lal:.2f} over the sweep, flattening at the top "
                 "where the unpredictable root estimates start to dominate what is left.</p>")
    return "\n".join(o)


def length():
    sf, srs = S.steps_fit()
    fc = S.fig_compute()
    if not fc:
        return ""
    o = ["<h2>3. Training length and compute</h2>"]
    o.append(fc)
    if sf:
        o.append(f"<p>For the 3.2M model the excess error falls with training length as steps to the power {sf[1]:.2f}: "
                 + ", ".join(f"{pct(1 - r['acc'])} at {r['steps']/1000:g}k" for r in srs) + ". The pool is larger than the number of problems drawn, "
                 "so almost every batch is new problems: in this regime training length, data seen and compute are one axis.</p>")
    (a, g), crs, miss = S.compute_fit()
    o.append(f"<p>Put every run on one axis, the compute it used, and they line up. Across {len(crs)} runs that differ in size by a factor of "
             f"{max(r['N'] for r in crs) / min(r['N'] for r in crs):.0f} and in length by a factor of {max(r['steps'] for r in crs) / min(r['steps'] for r in crs):.0f}, "
             f"excess error is compute to the power -{g:.2f}, and no run is off that line by more than {100 * (miss - 1):.0f}%. To a first "
             "approximation the error depends on how much compute was spent and not on whether it went into parameters or into steps.</p>")
    a3, a11 = (S.m("H_full_3m_30k") if S.have("H_full_3m_30k") else None), (S.m("H_full_11m") if S.have("H_full_11m") else None)
    if a3 and a11:
        o.append(f"<p>The cleanest pair is at {a3['pflop']:.0f} PFLOP: the 3.2M model for 30,000 steps ({pct(1 - a3['acc'])} error) and the 10.7M model "
                 f"for 10,000 ({pct(1 - a11['acc'])}).")
        if S.have("S_n08_30k") and S.have("S_n18"):
            b1, b2 = S.m("S_n08_30k"), S.m("S_n18")
            o.append(f" At the small end the 0.8M model for 30,000 steps ({b1['pflop']:.0f} PFLOP, {pct(1 - b1['acc'])}) sits next to the 1.8M model for "
                     f"10,000 ({b2['pflop']:.0f} PFLOP, {pct(1 - b2['acc'])}).")
        o.append(" The trade is not free at inference: the larger model costs more per problem every time it is used, the longer training is "
                 "paid once.</p>")
    return "\n".join(o)


def data():
    ds = sorted(S.runs(S.DATA), key=lambda r: r["unique"])
    if len(ds) < 3:
        return ""
    o = ["<h2>4. Distinct data</h2>"]
    o.append("<p>Synthetic problems are free, so every other run here sees each problem about once. To find out how much of that data is "
             "needed, the 3.2M model is trained with the same compute on smaller pools, revisiting the same problems.</p>")
    o.append(S.fig_data())
    o.append(S.data_table())
    small = ds[0]
    o.append(f"<p>With {nf(small['unique'])} distinct problems, each seen {small['seen'] / small['unique']:.0f} times, the model "
             + (f"reproduces {pct(small['train_tf'], 0)} of its training solutions token for token and " if small["train_tf"] is not None else "")
             + f"answers {pct(small['acc'])} of the test problems: it has memorised a pool it cannot generalise from. "
             + " ".join(f"At {nf(r['unique'])} it is at {pct(r['acc'])}." for r in ds[1:])
             + " The curve flattens once each problem is seen only a handful of times. Past that point distinct data is not what limits the "
             "run; steps and parameters are.</p>")
    return "\n".join(o)


def asymptote():
    rs = S.runs(S.SIZE)
    if len(rs) < 5:
        return ""
    o = ["<h2>5. Is there an asymptote?</h2>"]
    floor = 1 - S.ceiling()
    o.append(f"<p>In range there is one, and it is known in advance. The worked solutions carry six significant digits, and a solver that "
             f"executed them without a single slip would still get {pct(floor)} of the test problems wrong in the fourth digit. That is the dashed "
             "line in the figures, and it belongs to the scratchpad, not to the model. Above it, nothing in the curves looks like a second "
             "plateau.</p>")
    big = rs[-1]
    (a, al), used = S.size_fit(upto=rs[-2]["N"])
    pred = a * big["N"] ** (-al) + floor
    o.append(f"<p>The check is to fit on the smaller models and predict the largest. A power law through the excess error of the "
             f"{len(used)} sizes up to {nf(used[-1]['N'])} (exponent {al:.2f}) predicts {pct(pred)} error for the {nf(big['N'])} model. It measured "
             f"{pct(1 - big['acc'])}." +
             (" The law holds one step further out than it was fitted on." if abs(math.log((1 - big['acc'] - floor + 1e-4) / (pred - floor))) < 0.5 else
              " The largest model falls short of the line: at a fixed 10,000 steps a model this size is limited by training length, and more "
              "parameters alone no longer pay at the same rate.") + "</p>")
    (a2, al2), _ = S.size_fit()
    need = lambda err: (a2 / max(err - floor, 1e-6)) ** (1 / al2)
    o.append(f"<p>Read forwards, the fit over all sizes says 1% error in range would take about {nf(need(0.01))} parameters at this training "
             f"length, and 0.5% about {nf(need(0.005))}. Those are extrapolations of a line through {len(rs)} points, with one seed per point, and the last points "
             "sit close enough to the floor that the exponent is sensitive to where the floor is put. The qualitative answer is firmer than the "
             "numbers: in range, more parameters and more steps keep paying until the scratchpad's own precision is what is left.</p>")
    return "\n".join(o)


def stretch():
    rs = [r for r in S.runs(S.SIZE) if r["stretch_mean"] is not None]
    if len(rs) < 4:
        return ""
    o = ["<h2>6. Does scale buy the next size up?</h2>"]
    o.append('<p>The <a href="../calculator-generalize/">generalisation follow-up</a> found that these models interpolate over problem '
             "size and fall off a few sizes past the largest one they trained on. The scaling question is whether that margin grows with the "
             "model.</p>")
    o.append(S.fig_stretch_scale())
    o.append(S.stretch_levels_table())
    lo, hi = rs[0], rs[-1]
    top = max(rs[:-1], key=lambda r: r["stretch_mean"])
    flat = hi["stretch_mean"] < top["stretch_mean"] + 0.03
    seq = ", ".join(f"{pct(r['stretch_mean'])} at {nf(r['N'])}" for r in rs)
    dead = [m3.LEVELS[i + S.OLD] for i in range(7) if max(r["stretch"][i] for r in rs) < 0.25]
    moved = [m3.LEVELS[i + S.OLD] for i in range(7) if hi["stretch"][i] - lo["stretch"][i] > 0.4]
    o.append(f"<p>Up to a point. One size up reads {seq}. " +
             (f"The {nf(hi['N'])} model cuts in-range error by a further factor of {(1 - top['acc']) / (1 - hi['acc']):.1f} over the {nf(top['N'])} one and is "
              "no better out of range. " if flat else "") +
             f"In-range error fell by a factor of {(1 - lo['acc']) / (1 - hi['acc']):.0f} over the sweep; out of range, error went from "
             f"{pct(1 - lo['stretch_mean'])} to {pct(1 - hi['stretch_mean'])}.</p>")
    o.append("<p>The tiers explain the shape. " +
             (f"{' and '.join(moved)} respond to scale: there a larger problem is the same procedure on larger numbers or a longer chain of "
              "familiar steps, and a model whose arithmetic is more reliable gets further. " if moved else "") +
             (f"{', '.join(dead)} stay under 25% at every size: there the stretch asks for more digits than any carry it has done, more terms to collect, "
              "more terms to sum, or a combination never seen, and no amount of in-range accuracy supplies that. " if dead else "") +
             "With 100 problems per tier a single tier's number can move by several points from sampling alone, so the mean is the thing to read. "
             "Two points do not make an asymptote, and this is one seed each; what they do show is that the last factor of in-range error bought "
             "nothing out of range. If it is a plateau, it is the out-of-range counterpart of the in-range floor, set by the training pool's "
             "coverage instead of the scratchpad's precision. Widening the pool by one size, as in the follow-up, moved it from 30% to 92% with 10,000 steps of "
             "fine-tuning.</p>")
    return "\n".join(o)


def laws():
    rs = S.runs(S.SIZE)
    if len(rs) < 5:
        return ""
    (a, al), _ = S.size_fit()
    sf, _ = S.steps_fit()
    o = ["<h2>7. The laws, stated plainly</h2>"]
    o.append("<ul>"
             f"<li>In range, excess error falls as a power of parameters (exponent about {al:.1f} at 10,000 steps)"
             + (f" and as a power of training length (exponent about {sf[1]:.1f} for the 3.2M model)" if sf else "") + ".</li>"
             "<li>Parameters and steps are close to interchangeable at equal training compute over the range tested.</li>"
             "<li>Distinct data matters until each problem is seen a few times, and not after.</li>"
             f"<li>The in-range asymptote is the worked solutions' own precision ({pct(1 - S.ceiling())} error), set by the scratchpad's design.</li>"
             "<li>One size beyond the training range improves with scale only up to a plateau around a third, and only in tiers where "
             "larger means larger numbers rather than more structure. Range comes from the pool; scale buys accuracy inside it.</li></ul>")
    o.append("<p>Limits of all this: one task, one architecture family, models between 0.2M and 19M parameters, a single seed per point, and "
             "width and depth changed together. The exponents describe this calculator. The shape (power laws to a floor that the data format "
             "sets, and a much flatter curve out of range) is the part I would expect to carry.</p>")
    return "\n".join(o)


def repro():
    return """
<h2>8. Reproduce</h2>
<pre><code>python build_data3.py 2000000 hard/full.npz full              # the 28M-problem pool and the 28k test set
python scale_run.py                                          # all runs, two at a time, resumable (jobs are listed in the script)
python train3.py --name S_n08 --data hard/full.npz --out runs/scale --d 128 --heads 4            # one size
python train3.py --name S_d560k --data hard/full.npz --out runs/scale --data_frac 0.02           # one pool size
python eval_stretch.py runs/scale/S_n08.pt hard/stretch.json stretch100 100                      # one size up, 100 per tier</code></pre>
"""


FOOT = """
<footer>Written with an AI assistant (Claude), which also designed and ran the experiments. Code and per-run results:
<a href="https://github.com/bobrenjc93/explanations/tree/main/calculator-scaling/code">calculator-scaling/code</a>.</footer>
"""
