"""Third write-up: a calculator benchmark that takes longer to saturate."""
import re
import numpy as np
import mathgen3 as m3
import figs4 as G
from charts import pct, table, esc, figure

TITLE = "A calculator benchmark that takes longer to saturate"
DESCRIPTION = ("Seven harder tiers on top of the calculator benchmark that reached 99%: big numbers, long expressions, equations, linear systems, "
               "sums and Newton's method, with worked solutions of up to 2,300 tokens, and a solution-length horizon as the score.")
PREV = '<a href="../calculator-99/">the calculator at 99%</a>'
A, A30, BIG, NONE, VALS, CONT = "H_full_3m", "H_full_3m_30k", "H_full_11m", "H_none_3m", "H_values_3m", "H_um8_3m"
has = G.have
R = G.R


def lv(n, l):
    return pct(G.em(n)[l - 1])


def new(n, d=1):
    return pct(G.new_mean(n), d)


def old(n, d=1):
    return pct(G.old_mean(n), d)


def allm(n, d=1):
    return pct(float(np.mean(G.em(n))), d)


SECTIONS = ["used_up", "tiers", "block", "results", "saturation", "beyond", "sine", "length", "scale", "errors", "thinking", "compute", "repro"]


def body():
    """Sections that have their runs are numbered in order; a '@name@' in the text becomes that section's number."""
    parts, num = [], {}
    for name in SECTIONS:
        html = globals()[name]()
        if html.strip():
            num[name] = len(num) + 1
            parts.append(re.sub(r"<h2>\d+\. ", f"<h2>{num[name]}. ", html, count=1))
    text = "\n".join([head(), tldr(), pending()] + parts + [FOOT])
    return re.sub(r"@(\w+)@", lambda m: str(num.get(m.group(1), "?")), text)


def pending():
    todo = [lab for n, lab in ((A30, "three times the steps"), (BIG, "3.4 times the parameters"), (CONT, "thinking tokens in front of the block")) if not has(n)]
    if not todo:
        return ""
    return ('<p class="note">Still training as this version goes up: runs with ' + ", ".join(todo) +
            ". Their sections will be added when they finish; nothing below depends on them.</p>")


def head():
    return (f"<h1>{TITLE}</h1>\n"
            f'<p class="sub">Follow-up to {PREV}. That benchmark is used up: seven levels, 99.1% exact match. The ask this time was harder and '
            "harder problems, equations included, and an eval that takes longer to saturate. Same small transformer, same recipe (a supervised "
            "thinking block before the answer), 14 levels instead of 7, and worked solutions of up to 2,300 tokens.</p>"
            '<p class="note"><strong>Follow-up:</strong> <a href="../calculator-generalize/">Does the calculator generalise?</a> leaves sizes '
            "out in the middle of the range, widens the pool and tests past its new edge, and gives sine a procedure instead of a table.</p>")


def tldr():
    b = []
    z = G.jload(f"{G.DIR}/old_11m.hard.json")
    b.append("<li><strong>The benchmark.</strong> The seven old levels plus seven new tiers: big numbers, long expressions, linear equations, quadratics, "
             "linear systems, sums, and Newton's method. 28,000 held-out problems. A reference solution runs from 15 tokens (L1) to 2,300 (L14), "
             "and a solver that executes it perfectly scores 98.6% or more on every level.</li>")
    if z:
        b.append(f"<li><strong>The old eval said nothing about harder problems.</strong> The model that scores 99.1% on L1 to L7 gets {pct(z['em'][7])} on L8 and "
                 f"{pct(z['em'][8])} on L9, which are written in the grammar it already knows and are only bigger.</li>")
    if has(A):
        hard = sorted(range(G.OLD, G.NL), key=lambda l: G.em(A)[l])
        b.append(f"<li><strong>Trained on all 14 levels</strong> (3.2M parameters, 10,000 steps), the same architecture reaches {old(A)} on the old levels and "
                 f"{new(A)} on the new ones, from {pct(G.em(A)[hard[-1]])} ({G.SHORT[hard[-1]]}) down to {pct(G.em(A)[hard[0]])} ({G.SHORT[hard[0]]}).</li>")
    if has(NONE, VALS, A):
        b.append(f"<li><strong>The thinking block matters more as problems get harder.</strong> On the new tiers: answer only {new(NONE)}, a block with "
                 f"values only {new(VALS)}, a block with full procedures {new(A)}.</li>")
    if has(A30):
        s90 = G.steps_to(A30, 0.9)
        slow = [G.SHORT[l] for l in range(G.NL) if s90[l] is None]
        b.append(f"<li><strong>It saturates level by level, and it does saturate.</strong> In a 30,000-step run the levels pass 90% one after another, the "
                 f"first at step {min(s for s in s90 if s):,} and the last at {max(s for s in s90 if s):,}" +
                 (f"; {', '.join(slow)} never do" if slow else "") + f". It ends at {old(A30)} on the old levels and {new(A30)} on the new: about three "
                 "times as long as the old benchmark took, and then this one is nearly used up too.</li>")
    names = [n for n in (A, A30, BIG) if G.stretch(n)]
    if names:
        n = A30 if A30 in names else names[0]
        b.append("<li><strong>What does not saturate: one size up.</strong> Every tier has a size knob. A stretch set turns each knob past its training "
                 f"range (more digits, more operations, more terms, more iterations). There the {R[n]['args']['steps']//1000}k-step model scores "
                 f"{pct(G.stretch_mean(n))}, against {new(n)} inside the range" +
                 (f", and the 10.7M model {pct(G.stretch_mean(BIG))}" if BIG in names and n != BIG else "") +
                 ". Sums stop at about the number of terms a training sum had; Newton's method breaks where the output gets longer than any "
                 "training solution.</li>")
    if has(A30, BIG):
        b.append(f"<li><strong>Steps or parameters.</strong> At about the same training compute, 3x the steps and 3.4x the parameters end in the same "
                 f"place inside the range ({new(A30)} and {new(BIG)} on the new tiers)" +
                 (f"; one size up the larger model does better ({pct(G.stretch_mean(BIG))} against {pct(G.stretch_mean(A30))})" if G.stretch(A30) and G.stretch(BIG) else "") +
                 ".</li>")
    if has(CONT, A):
        d = 100 * (float(np.mean(G.em(CONT))) - float(np.mean(G.em(A))))
        b.append(f"<li><strong>Thinking tokens: no clear effect.</strong> Eight filler tokens in front of the full block: {allm(CONT)} against "
                 f"{allm(A)} over the 14 levels ({d:+.1f} points), a difference of the size a different seed makes.</li>")
    return '<div class="tldr"><p><strong>TL;DR.</strong></p><ul>' + "".join(b) + "</ul></div>"


def used_up():
    o = ["<h2>1. The old benchmark is used up</h2>"]
    z = G.jload(f"{G.DIR}/old_11m.hard.json")
    o.append(f"<p>The last write-up ended with a 10.7M-parameter model at 99.1% exact match over seven levels, from <code>57+38</code> to composite "
             "TI-84 expressions. A test set where the best model misses one problem in a hundred cannot rank anything that comes after it. The first "
             "question is whether that 99% means the model can calculate, or only that it can do the seven things it was trained on.</p>")
    if z:
        o.append(f"<p>Two of the new tiers are written in the grammar the old model already reads. L8 is ordinary arithmetic with bigger operands "
                 "(8 to 14 digits for sums, 4 to 8 for products, long division and Euclid on 7 to 12 digit numbers). L9 is the same kind of expression as "
                 "L7 with 4 to 9 operations instead of 2 or 3. The old model has never seen a problem this long, and it was not trained on either tier:</p>")
        o.append(G.fig_zero())
        o.append(f"<p>{pct(z['em'][7])} and {pct(z['em'][8])}. The procedures are the ones it executes at 99% when the problem is short; it does not carry "
                 "them further than it was shown. So the old number was a statement about a distribution, and anything harder needs its own test.</p>")
    return "\n".join(o)


def tiers():
    o = ["<h2>2. Seven harder tiers</h2>"]
    o.append("<p>The new tiers were built to three rules. Each needs strictly more work than the ones before it. The true answer comes from exact "
             "arithmetic (rationals, or double precision where a root is involved) and never from the worked solution. And the worked solution, "
             "executed perfectly, has to end in the true answer almost always, so the benchmark measures the model and not the scratchpad: the "
             "<em>ceiling</em> column is that rate. (In the previous write-up the first scratchpad capped a perfect model at 94.9%, and nobody "
             "noticed until the error analysis.)</p>")
    o.append(G.tier_table())
    o.append("<p>The display rule is unchanged (integers in full, otherwise 4 significant digits), except that the new tiers show integers in "
             "full up to 10<sup>18</sup>. Equations with several answers list them separated by commas, roots in ascending order; a quadratic with "
             "no real root answers <code>none</code>. Each level has 2,000 test problems, 28,000 in all.</p>")
    o.append("<p><strong>What is held out, and two versions that were thrown away.</strong> Test problems never appear in training. For most "
             "tiers that means the problem text. The first version of the Newton tier showed why that is not always enough: a model trained to "
             "give the answer with no thinking at all reached 60% on it after 3,000 steps. The same polynomial appeared in training with other "
             "starting points, and the root could simply be remembered. Holding out whole polynomials did not fix it (57%): most of the tier was "
             "quadratics with small integer coefficients, and their roots are a table of a few hundred square roots plus a little arithmetic. So "
             "the tier is cubics only, held out by polynomial, and quadratics (L11) are held out by root set: no rearrangement, multiple or "
             "restart of a test equation is in the training pool. An answer-only baseline turned out to be a good test of a benchmark.</p>")
    return "\n".join(o)


def block():
    o = ["<h2>3. A thinking block for an equation</h2>"]
    o.append(f"<p>The recipe is the one that worked in {PREV}: before the answer the model writes a thinking block, "
             "<code>&lt;think&gt;</code> a worked solution <code>&lt;/think&gt;</code>, and in training it is shown what goes in the block. Every arithmetic "
             "step in the new tiers is the same kind of step as before: partial products with a running sum, long division digit by digit, a Newton "
             "step for a square root, Euclid for a gcd. What is new is the algebra between them, which takes two kinds of line:</p>")
    o.append("<ul><li><strong>Restated equations.</strong> After multiplying out brackets, after collecting terms, and after moving x to one side, the "
             "equation is written again in its new form: <code>3*x-6+4*x=5*x+5-7</code>, then <code>7*x-6=5*x-2</code>, then <code>2*x=4</code>.</li>"
             "<li><strong>Labelled values.</strong> <code>d=49</code> for a discriminant or a determinant, <code>dx=</code> and <code>dy=</code> for "
             "Cramer's rule, <code>k=3</code> for the index of a sum, <code>x=1.333</code> for the current Newton iterate.</li></ul>")
    o.append("<p>The procedures: linear equations are expanded, collected, moved and divided. Quadratics go through the discriminant; the root "
             "where <code>-b</code> and the square root add up comes from the formula and the other from <code>x1*x2 = c/a</code>, which avoids the "
             "cancellation that cost the textbook formula 3% of its answers at 6 digits. Systems use Cramer's rule, 3x3 determinants by cofactors "
             "of the first row. A sum evaluates its term for every k and keeps a running total. Newton's method (cubics, from a given starting "
             "point) writes the derivative once, then repeats: f(x) and f'(x) by Horner's rule, their quotient, the new x. The iterate carries 2 "
             "significant digits after the first step, 3 after the second and 4 from then on, since each step roughly doubles the correct digits "
             "anyway, and the loop stops when x no longer changes at 4 digits. Click one to see it written out:</p>")
    o.append(G.worked_examples())
    o.append("<p>Three versions of every training pool were built from the same problems: the full block, a block with <em>values only</em> (each "
             "operation and its result, without partial products, division digits or Newton steps), and no block at all.</p>")
    return "\n".join(o)


def results():
    o = ["<h2>4. Results inside the training range</h2>"]
    if not has(A):
        return ""
    o.append("<p>Every run trains on a pool of 28M rows, 2M per level, drawn uniformly, at batch 2,048. The model is the 3.2M-parameter transformer "
             "of the earlier write-ups (4 layers, the middle two applied twice, rotary positions) unless a row says otherwise. Exact match of the final "
             "answer, 2,000 test problems per level:</p>")
    o.append(G.run_heat())
    worst = min(range(G.OLD, G.NL), key=lambda l: G.em(A)[l])
    o.append(f"<p>Trained on all 14 levels for 10,000 steps, the full-procedure model holds {old(A)} on the old levels (the single-benchmark model "
             f"had 96.6% at this budget, with all of its training on those seven) and reaches {new(A)} on the new ones. The spread is the point: "
             f"{', '.join(f'{G.SHORT[l]} {pct(G.em(A)[l], 0)}' for l in range(G.OLD, G.NL))}. The weakest is {m3.LEVELS[worst]}.</p>")
    if has(NONE, VALS):
        o.append(G.fig_variants())
        o.append(f"<p>The thinking block does more work here than it did before. Without one the model scores {old(NONE)} on the old levels and "
                 f"{new(NONE)} on the new tiers; with values only, {old(VALS)} and {new(VALS)}; with procedures, {old(A)} and {new(A)}. On the old levels "
                 f"procedures were worth {100*(G.old_mean(A)-G.old_mean(VALS)):.0f} points over values; on the new tiers "
                 f"{100*(G.new_mean(A)-G.new_mean(VALS)):.0f}.</p>")
    return "\n".join(o)


def saturation():
    o = ["<h2>5. How long each level takes to saturate</h2>"]
    n = A30 if has(A30) else A
    if not has(n):
        return ""
    o.append("<p>The request was an eval that takes longer to saturate, so here is the saturation itself: held-out accuracy of every level as "
             "training proceeds, all 14 trained together.</p>")
    o.append(G.fig_saturation(n))
    o.append(G.saturation_table(n))
    s90 = G.steps_to(n, 0.9)
    order = sorted((l for l in range(G.NL) if s90[l]), key=lambda l: s90[l])
    never = [G.SHORT[l] for l in range(G.NL) if s90[l] is None]
    first, last = [l for l in order if s90[l] == s90[order[0]]], [l for l in order if s90[l] == s90[order[-1]]]
    cur = [float(np.mean(v)) for _, v in G.ans_curve(n)]
    steps = [s for s, _ in G.ans_curve(n)]
    dips = [steps[i] for i in range(1, len(cur)) if cur[i] < cur[i - 1] - 0.05]
    names = lambda ls: " and ".join(G.SHORT[l] for l in ls)
    verb = lambda ls: "passes" if len(ls) == 1 else "pass"
    f90 = lambda l: f"step {s90[l]:,}" if s90[l] else "never"
    o.append(f"<p>Steps are counted at the evaluation points (every {R[n]['args']['gen_every']:,} steps) on 200 problems per level; the final "
             f"column is the full 2,000. {names(first)} {verb(first)} 90% first, at step {s90[order[0]]:,}; {names(last)} last, at step {s90[order[-1]]:,}" +
             (f"; {', '.join(never)} are still short of 90% when the run ends" if never else "") +
             f". The order is not the order of solution length. Sums (409 tokens on average) are there at {f90(12)} and Newton's method (1,700 "
             f"tokens) at {f90(13)}, while composite expressions, big numbers and long expressions take until {f90(6)}, {f90(7)} and {f90(8)}. "
             "What takes longest is long arithmetic on long numbers and values that have to be recalled exactly, where one slip in one digit is "
             "fatal and nothing later in the solution corrects it. A Newton iteration, by contrast, forgives: the next iterate repairs a small "
             "error in the last." +
             (f" (The dip at step {dips[0]:,} is the optimiser at its peak learning rate, not the eval: every level falls together and "
              "recovers by the next evaluation.)" if dips else "") + "</p>")
    o.append("<p>So the answer to the request, inside the training range, is a qualified yes. The old seven levels were past 96% after 10,000 "
             f"steps of training on them alone. This benchmark needs about three times that to reach {allm(n)}, the tiers arrive in a visible order, "
             "and four levels (scientific functions, composite expressions, big numbers, long expressions) are still at 92 to 95% at the end. But "
             "it is the same kind of benchmark as the old one, and with more steps or a larger model it will go the same way. The part that "
             "does not is the next section.</p>" if n == A30 else "")
    return "\n".join(o)


def beyond():
    o = ["<h2>6. One size up: the stretch set</h2>"]
    names = [n for n in (A, A30, BIG) if G.stretch(n)]
    if not names:
        return ""
    n = A30 if A30 in names else names[0]
    se, ie = G.stretch_em(n), G.em(n)[G.OLD:]
    o.append("<p>Section 1 retired the old benchmark with one test: problems of a familiar kind, a size larger than any the model had trained on. "
             "The new model gets the same test. Every tier has a size knob, and the <em>stretch set</em> turns each one past its training range: "
             "300 problems per tier, never trained on, with reference solutions of up to 4,000 tokens. Nothing about the procedure changes. A "
             "13-term sum is evaluated exactly like a 12-term one.</p>")
    o.append(G.stretch_def_table())
    o.append(G.fig_stretch_bars(names))
    o.append(G.stretch_table(names))
    best = max(range(7), key=lambda i: se[i])
    worst = min(range(7), key=lambda i: se[i])
    o.append(f"<p>The {R[n]['args']['steps']//1000}k-step model falls from {pct(float(np.mean(ie)))} inside the range to {pct(float(np.mean(se)))} one size up. It "
             f"holds up best on {m3.LEVELS[best + G.OLD]} ({pct(se[best])}) and worst on {m3.LEVELS[worst + G.OLD]} ({pct(se[worst])}). "
             f"The old model fell from 99% on its own levels to 12% on L8; this one falls {100*(float(np.mean(ie))-float(np.mean(se))):.0f} points "
             "going from the range it trained on to the next.</p>")
    o.append(G.fig_knobs(n))
    sa = G.knob_acc(G.STRETCH["kind"], G.STRETCH["level"], G.stretch(n)["ok"], 12)
    if sa:
        ks = sorted(sa)
        far = [k for k in ks if sa[k] <= 0.02]
        o.append(f"<p>The knob curves show how close the edge is. A sum of {ks[0]} terms, one more than any sum in the pool, is solved "
                 f"{pct(sa[ks[0]], 0)} of the time; {ks[3]} terms, {pct(sa[ks[3]], 0)}" +
                 (f"; from {far[0]} terms on, almost never" if far else "") +
                 ". Long expressions and linear equations step down the same way at their first unseen size.</p>")
    o.append(G.fig_stretch_len(n))
    o.append("<p>The figure separates two things that are easy to confuse. Inside the range, accuracy hardly depends on how long the solution "
             "is. Beyond it, accuracy is low at almost every length, short solutions included: a stretch problem whose solution takes 300 tokens "
             "is much harder than an in-range problem of 300 tokens. What fails is not stamina. It is doing a familiar procedure at an unfamiliar "
             "size.</p>")
    kt = G.stretch_kinds_table(n)
    if kt:
        o.append("<p>The first fault in each wrong stretch solution, found the same way as for the in-range errors in section @errors@:</p>")
        o.append(kt)
        pos14, _ = G.stretch_fault_stats(n, G.NL - 1)
        share_stop, wrote, has_terms = G.sum_stop_stats(n)
        o.append("<p>Each tier breaks in its own way. Big numbers go wrong in the very first step: a few more digits than it has seen are "
                 "enough to break a carry, a partial product or a division. Linear equations lose track while collecting a dozen terms: the "
                 "equation they restate is no longer the one the arithmetic produced. Two-digit 3x3 systems go wrong a tenth of the way in, "
                 "usually by multiplying numbers the determinant does not call for. The sums are the clearest case of a procedure that was not "
                 f"really learned: {pct(share_stop, 0)} of the failures are correct solutions that simply stop early, after a median of "
                 f"{wrote:.0f} terms where the problem has {has_terms:.0f}, without looking at the upper limit. A training sum never had more than 12. "
                 "And Newton's method fails where the output becomes longer than anything the model ever "
                 f"wrote: {pct(float(np.mean(pos14 >= 2000)), 0)} of its first faults fall after token 2,000 of the solution, next to the 2,304 "
                 "where every training solution had ended.</p>")
    ft = G.stretch_formats_table([NONE, VALS, A])
    if ft:
        o.append("<p>The other formats, at the same size and 10,000 steps. Values-only is worse almost everywhere, with one instructive "
                 "exception: it beats full procedures on stretch Newton problems. Its solutions are shorter (about 1,500 tokens where the full "
                 "ones take 2,800), so 97% of them stay inside the range of output lengths it was trained on.</p>")
        o.append(ft)
    o.append("<p>This is the part of the benchmark that should take longest to saturate, because it is defined relative to what was trained on. "
             "Adding the stretch range to the training pool would presumably teach that range, the way the old model's failure on L8 and L9 "
             "disappeared once they were trained on; but the generator has knobs, and the set one size above the new pool is then the new stretch "
             "set. (<a href=\"../calculator-generalize/\">The follow-up</a> runs that step: the added range is learned and the next one mostly is not.) A fixed "
             "test set measures whether a model has learned a distribution. A stretch set measures "
             "whether it has learned the procedure, and for these models the answer is: not yet.</p>")
    return "\n".join(o)


def sine():
    if not G.SIN or not G.SINW:
        return ""
    s = G.sin_stats("big")
    bins = {r[1]: r for r in G.sin_bins("big")}
    old_bins = {r[1]: r for r in G.sin_bins("old")}
    o = ["<h2>7. The simplest case: does it know sine?</h2>"]
    o.append("<p>After the stretch set, a fair question is whether these models have learned any function at all, or only how to act on inputs "
             "like the ones they have met. Sine is the cleanest probe in the benchmark, because the answer outside the trained range is fully "
             "determined by the answers inside it: sin(x) = sin(x - 2&pi;), and sin(-x) = -sin(x). In training, sine is asked for directly only "
             "for x between -9.99 and 9.99 on a grid of 0.01. That is 1,999 arguments, and in the worked solution each one is a single recalled "
             "line (<code>sin(1.5)=0.997494</code>). The probe is that one question, <code>sin(x)</code>, for x far outside the grid.</p>")
    o.append(G.fig_sin_curve("big"))
    o.append(f"<p>Inside the range the 10.7M model is right on all {s['n_in']:,} grid points, to four digits. One step outside, exact match is "
             f"{pct(s['near'])} for |x| between 10 and 20 and {pct(s['far'])} beyond 20. So the short answer is no: it does not calculate sine once x "
             "leaves the range. But the dots are not noise either, and the two sides of the figure are different.</p>")
    o.append(f"<p><strong>To the left</strong>, below -10, the answers have nothing to do with the true value: within 0.1 of it "
             f"{pct(bins['-10-20'][3], 0)} of the time, which is what unrelated numbers get, and the right sign less than half the time. If "
             f"anything they lean toward sin(|x|), the answer with the minus sign ignored ({pct(G.sin_flip('big'), 0)} within 0.1 of that between "
             f"-20 and -10, {pct(G.sin_flip('old'), 0)} for the old model). The model computes sin(3) and sin(-3) perfectly and has not picked up "
             "that one is minus the other. <strong>To the right</strong>, above 10, the answers "
             f"follow the wave: the sign is right {pct(bins['+10-20'][4], 0)} of the time between 10 and 20 and the value is within 0.1 "
             f"{pct(bins['+10-20'][3], 0)} of the time. They sit near the right peak or trough and miss the digits.</p>")
    o.append(G.fig_sin_within())
    d = G.SIND
    o.append("<p>The asymmetry is in the training data. Composite problems take sines of computed values, <code>sin(5.5^2)</code>, "
             "<code>sin(7.5+4.8)</code>, <code>sin(31*pi)</code>, and those values are always positive, mostly below 100, and few: "
             f"{d['distinct_outside']:,} distinct arguments beyond the grid in {d['problems']:,} composite problems. The table lines the two up.</p>")
    o.append(G.sin_table("big"))
    o.append(f"<p>Three regimes, and they follow how densely the function was sampled. On the dense grid, every value many times: exact. Where "
             f"training left scattered samples: the right neighbourhood, fading as the samples thin out, and gone past 100 "
             f"({pct(bins['+70-100'][3], 0)} within 0.1 at 70 to 100, {pct(bins['+100-150'][3], 0)} at 100 to 150). Where it left none: nothing. "
             "The old model saw about ten times as many composite problems as this one and is better in the middle regime "
             f"({pct(old_bins['+10-20'][3], 0)} within 0.1 between 10 and 20), identical on the grid, and on the left wrong in the same way, more consistently: it mostly "
             "answers sin(|x|).</p>")
    pr = d["probe"]
    mb, mo = pr["models"]["big"], pr["models"]["old"]
    o.append(f"<p>The split between recalling and interpolating can be made exact. Of the {pr['n']:,} probe points between 10 and 40, {pr['met']} are "
             f"arguments that occur somewhere in the solutions of {d['problems']:,} sampled composite problems. The 10.7M model gets {pct(mb['met']['exact'], 0)} of them to four digits and "
             f"{pct(mb['unmet']['exact'])} of the other {pr['n'] - pr['met']:,}; the old model {pct(mo['met']['exact'], 0)} and "
             f"{pct(mo['unmet']['exact'])}. Exact answers are entries it has met. But the points it never met are still within 0.1 "
             f"{pct(mb['unmet']['within'], 0)} of the time ({pct(mo['unmet']['within'], 0)} for the old model) with the right sign "
             f"{pct(mb['unmet']['sign'], 0)} of the time, nearly the same as the ones it did meet.</p>")
    o.append("<p>So the models do generalise, in the way a smooth interpolator does: between samples, an argument they have never seen gets an "
             "answer near the right one. What they do not do is use the structure of the function. The period and the odd symmetry are both "
             "sitting in the 1,999 values they know exactly, and neither is applied one step outside them. That is the stretch set's result in "
             "its plainest form: accuracy inside the range says the model has the entries, and only the step outside says whether it has the "
             "rule.</p>")
    return "\n".join(o)


def length():
    o = ["<h2>7. Is it the length?</h2>"]
    if not has(A):
        return ""
    o.append("<p>When this started I expected the natural score to be a horizon: the length of worked solution at which accuracy falls to some "
             "threshold, read off a curve of accuracy against solution length. Every problem has such a length, independent of any model: the "
             "number of tokens in its reference solution. Here is that curve for each run, over all 28,000 test problems.</p>")
    o.append(G.fig_horizon())
    hn, hv = G.horizons(NONE) if has(NONE) else (None, None), G.horizons(VALS) if has(VALS) else (None, None)
    o.append("<p>For the weaker formats the horizon is real. " +
             (f"Answering with no block, accuracy crosses 50% at solutions of about {hn[1]:,.0f} tokens; " if hn[1] else "") +
             (f"with values only, at about {hv[1]:,.0f}. " if hv[1] else "") +
             "With full procedures the curve does not fall off inside the range at all: it dips where the long arithmetic lives (big numbers, long "
             "expressions) and comes back up for sums and Newton's method, whose solutions are the longest in the benchmark. Per token, a Newton "
             "iteration is easier than a 14-digit carry. So a single horizon number would describe the mix of tiers more than the model, and the "
             "honest summaries are the two in the sections above: accuracy per level inside the range, and accuracy one size up.</p>")
    o.append(G.horizon_table())
    return "\n".join(o)


def scale():
    o = ["<h2>8. More steps or more parameters</h2>"]
    if not has(A, A30, BIG):
        return ""
    rows = []
    for n, lab in ((A, "3.2M parameters, 10k steps"), (A30, "3.2M parameters, 30k steps"), (BIG, "10.7M parameters, 10k steps")):
        rows.append([lab, f"{G.train_pflops(n):.0f}", old(n), new(n), pct(G.stretch_mean(n)) if G.stretch(n) else "", f"{G.infer_gflops(n):.2f}"])
    o.append("<p>Two ways to spend about three times the training compute: three times the steps, or a model with 3.4 times the parameters (6 layers of "
             "width 384, the middle two applied twice) for the same number of steps.</p>")
    o.append(table(["run", "train PFLOP", "L1\u20117", "L8\u201114", "stretch set", "infer GFLOP / problem"], rows))
    gap = 100 * abs(G.new_mean(A30) - G.new_mean(BIG))
    txt = (f"<p>Inside the range the two are {'indistinguishable' if gap < 1 else 'close'}: {new(A30)} on the new tiers with more steps, {new(BIG)} "
           f"with more parameters, and the larger model costs {G.infer_gflops(BIG)/G.infer_gflops(A30):.1f} times as much per problem at inference. ")
    if G.stretch(A30) and G.stretch(BIG) and G.stretch(A):
        s0, s1, s2 = G.stretch_mean(A), G.stretch_mean(A30), G.stretch_mean(BIG)
        e1, e2 = G.stretch_em(A30), G.stretch_em(BIG)
        txt += (f"One size up they differ. Tripling the steps moves the stretch score from {pct(s0)} to {pct(s1)}, tripling the parameters to {pct(s2)}, "
                f"and the difference is two tiers: Newton's method ({pct(e2[6], 0)} against {pct(e1[6], 0)}) and quadratics ({pct(e2[3], 0)} against "
                f"{pct(e1[3], 0)}), where a stretch problem is the same procedure on larger numbers and the larger model's arithmetic is simply more "
                "reliable. Where the stretch is structural (more terms to collect, more terms to sum, two-digit entries in a 3x3 determinant) "
                "neither moves: three times the compute buys the range, and very little of the next one.")
    o.append(txt + "</p>")
    o.append(G.run_heat([A, A30, BIG], zero=False))
    return "\n".join(o)


def errors():
    o = ["<h2>9. Where it goes wrong</h2>"]
    n = A30 if G.errs(A30) else A
    if not G.errs(n):
        return ""
    o.append(f"<p>Every wrong answer of the {R[n]['args']['steps']//1000}k-step model on the new tiers was traced back through the model's own worked solution. "
             "Each arithmetic step is re-checked independently; the first one that is wrong names the failure. If all the arithmetic is right, the first "
             "place the solution departs from the reference does.</p>")
    o.append(G.fig_err_kinds(n))
    o.append(G.err_table(n))
    e = G.errs(n)["kinds"]
    sub = {(k["level"].split()[0], k["kind"]): k["em"] for k in e}
    ar, op, look, rnd = G.err_shares(n)
    o.append(f"<p>{pct(ar, 0)} are a wrong digit somewhere in the arithmetic: a quotient digit, a carry, a partial product. {pct(op, 0)} are steps "
             f"whose arithmetic is right but whose operands are not the ones the solution calls for, a number copied wrong from an earlier line. "
             f"{pct(look, 0)} are recalled values (a logarithm, a sine), and {pct(rnd, 0)} are not the model's fault at all: it reproduces the "
             "reference solution exactly, and the reference's 6-digit arithmetic rounds to a different fourth digit than the true answer (the "
             "ceiling in section 2).</p>")
    o.append(f"<p>Inside the tiers, difficulty follows size the way it should. Long expressions go from {pct(sub[('L9', '4 operations')], 0)} at 4 "
             f"operations to {pct(sub[('L9', '9 operations')], 0)} at 9; 3x3 systems are at {pct(sub[('L12', '3x3')], 0)} against "
             f"{pct(sub[('L12', '2x2')], 0)} for 2x2; and within big numbers the hardest sub-type is plain addition and subtraction of "
             f"8-to-14-digit numbers ({pct(sub[('L8', 'add/sub')], 0)}), the one operation whose worked solution shows no work: the model has to "
             "carry across 14 digits in one step.</p>")
    o.append("<details class=tv><summary>Accuracy by sub-type, every new tier</summary>" + G.kind_table([x for x in (A, A30, BIG) if G.errs(x)]) + "</details>")
    return "\n".join(o)


def thinking():
    o = ["<h2>10. Thinking tokens, again</h2>"]
    if not has(CONT, A):
        return ""
    o.append("<p>The idea these write-ups started from, positions before the answer with no supervised content, was tested on the old benchmark "
             "three times and did nothing each time. One more try on the harder one, where there is more to think about: eight filler thinking "
             "tokens (one learned embedding, repeated) between the problem and the thinking block.</p>")
    o.append(G.run_heat([A, CONT], zero=False))
    d = 100 * (float(np.mean(G.em(CONT))) - float(np.mean(G.em(A))))
    up = sum(1 for a, b in zip(G.em(CONT), G.em(A)) if a > b)
    o.append(f"<p>{allm(CONT)} against {allm(A)} over the 14 levels ({d:+.1f} points), and {new(CONT)} against {new(A)} on the new tiers; "
             f"{up} of the 14 levels are higher with the tokens. That is one seed each, and in the first write-up the same configuration run "
             "twice differed by about 2 points, so this is a difference I cannot tell from noise. It is the same size as the "
             "thinking-token results before it: about a point at most, never the tens of points that supervised content in the block is worth.</p>")
    o.append("<p>The continuous-thought variant (the model's last hidden state fed back as its next input) is not in this write-up: it needs a "
             "sequential loop inside every training step, and with 21 length buckets its compilation did not finish in the time I gave it.</p>")
    return "\n".join(o)


def compute():
    o = ["<h2>11. Compute</h2>"]
    if not has(A):
        return ""
    c = R[A]["compute"]
    o.append("<p>Accounting as before: a token passing through one layer application costs 24d<sup>2</sup> FLOPs forward (the matmuls), training "
             "three times that per padded position. The model applies 6 layers per token (4 layers, two of them twice).</p>")
    o.append(G.compute_table())
    f = R[A]["final"]
    share = [f["tle_level"][l] for l in range(G.NL)]
    o.append(f"<p>Harder is longer, and longer is what costs. A training row now averages {c['mean_target_tokens']:.0f} target tokens against 75 on the old "
             f"benchmark, and one optimiser step processes {c['padded_positions_per_step']/1e3:.0f}k positions against 254k. At inference the 14 "
             f"levels span a factor of {max(share)/min(share):.0f} in compute per problem:</p>")
    o.append(G.level_compute_table(A30 if has(A30) else A))
    o.append("<p>Two implementation notes for sequences this long. The teacher-forced pass uses fused attention over prompt and output instead of "
             "building the attention matrix explicitly, which is the difference between fitting a 2,300-token bucket in memory and not. And decoding "
             "writes keys and values into a preallocated buffer; re-concatenating the cache at every token is quadratic in copies.</p>")
    return "\n".join(o)


def repro():
    return """
<h2>12. Reproduce</h2>
<pre><code>python build_data.py 1500000                                  # the old 14k test problems (data.npz)
python check_gen3.py 3000                                     # new tiers: every arithmetic step re-checked, ceilings, lengths
python build_data3.py 2000000 hard/full.npz full              # 28M rows + the 28k test set; also: values, none
python train3.py --name H_full_3m --data hard/full.npz        # 3.2M parameters, 10k steps
SCHED_SCRIPT=train3.py python sched.py jobs_hard.txt runs/hard 2   # all runs
python eval_old.py runs/v2/v23_11m_s0.pt hard/full.npz        # the previous model, zero-shot
python errs3.py runs/hard/H_full_3m_30k hard/full.npz         # first fault in each wrong solution
python build_stretch.py hard/stretch.json 300                 # the beyond-the-range test set
python eval_stretch.py runs/hard/H_full_3m_30k.pt             # a checkpoint on it
python sin_probe.py runs/hard/sin_probe.json big=runs/hard/H_full_11m.pt small30k=runs/hard/H_full_3m_30k.pt old=runs/v2/v23_11m_s0.pt
python sin_probe.py runs/hard/sin_probe_wide.json range=-30000:30000:7 big=... small30k=... old=...   # out to +-300
python sin_density.py runs/hard/sin_density.json              # how often training solutions recall sin(x), by size of x</code></pre>
"""


FOOT = """
<footer>Written with an AI assistant (Claude), which also designed and ran the experiments. Code and per-run results:
<a href="https://github.com/bobrenjc93/explanations/tree/main/calculator-harder/code">calculator-harder/code</a>.</footer>
"""
