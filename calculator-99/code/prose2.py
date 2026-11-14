"""Second write-up: getting the calculator past 95%."""
import json, os
import numpy as np
import mathgen as mg, mathgen2 as m2
import pagedata as pd
import figs2 as F
from charts import pct, table, esc, figure

TITLE = "A calculator LLM at 99%: the scratchpad mattered more than the weights"
DESCRIPTION = ("Scaling a calculator transformer from 83% to 99% exact match: a worked-solution scratchpad with guard digits and real "
               "procedures, then positional encodings, norms, weight quantisation and model size.")
ALL = {**F.ABL, **F.V2}
OKS = {}
for d in ("runs/v2abl", "runs/v2", "runs/arch"):
    for n in list(ALL) + list(F.ARCH):
        if os.path.exists(f"{d}/{n}.ok.npy"):
            OKS[n] = np.load(f"{d}/{n}.ok.npy")
MAIN, SMALL, PILOT = "v23_11m_s0", "v23_3m_s0", "v2_3m_s0"
OLD_CHARS = float(((np.load("data.npz")["test_C"] != 0).sum(1) - 1).mean())  # old scratchpad + answer, characters per problem


def E(n, d=1):
    return pct(F.em(ALL, n), d)


def C(n, d=1):
    return pct(F.close(ALL, n), d)


def has(*names):
    return all(n in ALL for n in names)


def sp(t):
    return f"<span>{t}</span>"


def body():
    o = [f"<h1>{TITLE}</h1>", sub(), tldr(), diagnosis(), scratchpad(), results(), fixes(), knobs(), quant(), closing(), repro(), FOOT]
    return "\n".join(x for x in o if x)


def sub():
    return ('<p class="sub">Follow-up to <a href="../thinking-tokens/">the thinking-token ablation</a>, which left the calculator at 83.3% exact match. '
            "The ask was to scale it out to 95%+, with more weights and whatever architecture changes help. Same task, same 14,000 held-out test "
            "problems, same display rule (integers in full, otherwise 4 significant digits). It ends at 99.1%, and most of the distance was not covered by weights.</p>")


def tldr():
    b = []
    if has(MAIN):
        b.append(f"<li><strong>{E(MAIN)} exact match</strong> ({C(MAIN)} within 1%) with 10.7M parameters, up from 83.3% at the same size.</li>")
    small = SMALL if has(SMALL) else PILOT
    b.append(f"<li><strong>The scratchpad did it, not the scale.</strong> A 3.2M-parameter model goes from 77.7% to {E(small)} when only what it is "
             "trained to write down changes.</li>")
    b.append("<li><strong>The old scratchpad capped a perfect model at 94.9%</strong>: it wrote intermediate values to 4 significant digits, the precision "
             "of the answer. The new one carries 6 digits and spells out procedures: partial products, long division, Euclid, a Newton step.</li>")
    b.append("<li><strong>Three bugs in the training signal</strong>, each found by reading wrong solutions: a Newton step the model learned to skip, table "
             "entries that existed only in the test set, and a division step that asked for a multiply and a subtract at once.</li>")
    if has("b_base_s0", "b_nope_s0", "b_abs_s0", "b_rms_s0"):
        x = (f"<li><strong>Architecture knobs.</strong> Rotary positions matter for long worked solutions: no positional encoding costs "
             f"{100*(F.em(ALL,'b_base_s0')-F.em(ALL,'b_nope_s0')):.1f} points and learned absolute positions {100*(F.em(ALL,'b_base_s0')-F.em(ALL,'b_abs_s0')):.1f}. "
             f"RMSNorm and LayerNorm are indistinguishable ({E('b_rms_s0')} vs {E('b_base_s0')}).")
        if has("b_noloop_s0"):
            x += f" The looped middle block is worth {100*(F.em(ALL,'b_base_s0')-F.em(ALL,'b_noloop_s0')):.1f} points."
        b.append(x + "</li>")
    q = F.jload(f"runs/v2/{MAIN}.quant.json")
    if q:
        x = (f"<li><strong>Quantisation.</strong> Rounded after training, the 10.7M model keeps {pct(float(np.mean(q['4'])))} at 4 bits per weight and "
             f"falls apart at 3 ({pct(float(np.mean(q['3'])))}).")
        if has("b_w2_s0"):
            x += f" Trained with ternary weights from the start, a 3.2M model reaches {E('b_w2_s0')} (full precision, same budget: {E('b_base_s0')})."
        b.append(x + "</li>")
    if has(MAIN):
        b.append(f"<li><strong>The price is inference length.</strong> A worked solution averages {ALL[MAIN]['mean_solution_chars']:.0f} characters against "
                 f"{OLD_CHARS:.0f} for the old scratchpad: {ALL[MAIN]['final']['tle'] / pd.tle('big2'):.1f} times the compute per problem.</li>")
    return '<div class="tldr"><p><strong>TL;DR.</strong></p><ul>' + "".join(b) + "</ul></div>"


def diagnosis():
    o = ["<h2>1. Why more weights alone could not get there</h2>"]
    o.append("<p>Two measurements on the 83.3% model (10.7M parameters, written scratchpad, looped middle block) set the direction.</p>")
    o.append("<p><strong>It failed where the scratchpad had nothing to say.</strong> The first scratchpad listed intermediate <em>values</em> and gave a "
             "procedure only for multiplication. Per sub-skill, the failures follow that line exactly:</p>")
    k = lambda l, n: pct(pd.sub_one("big2", l, n), 0)
    o.append(table(["sub-skill", "the 83.3% model"], [
        ["sums of 3-4 operands; multiplication by 2 digits (the scratchpad shows the steps)", f"{k(1, '3-4 operands')}, {k(2, 'a*b, smaller has 2 digits')}"],
        ["division to 4 digits", k(2, "a/b inexact")], ["mod(a,b)", k(2, "mod(a,b)")], ["powers", k(4, "powers")],
        ["gcd / lcm", k(5, "gcd/lcm")], ["e^x, 10^x, 2^x", k(5, "e^x, 10^x, 2^x")], ["nCr / nPr", k(5, "nCr/nPr")], ["composite expressions", k(6, "composite")]]))
    o.append("<p><strong>And the format had a ceiling.</strong> Intermediate values were written to 4 significant digits, the same precision as the final "
             "answer, so round-off in the intermediates reaches the last digit of the answer. An oracle that evaluates every test expression exactly, "
             "except that each intermediate value is first rounded to n digits, scores:</p>")
    o.append(table(["digits kept in intermediate values", "oracle exact match, all levels", "composite expressions"],
                   [["4 (the old scratchpad)", "94.9%", "71.7%"], ["5", "99.2%", "94.9%"], ["6 (the new scratchpad)", "99.8%", "99.2%"], ["7", "99.9%", "99.7%"]]))
    o.append("<p>A perfect model trained on the old scratchpad would still have missed 95%. The first write-up's conclusion, that the training signal is the "
             "bottleneck, had more to give before any scaling.</p>")
    return "\n".join(o)


def scratchpad():
    o = ["<h2>2. A worked solution instead of a list of values</h2>"]
    o.append("<p>Every step is now <code>operation on printed values : work = value</code>. Steps are separated by <code>;</code>, values carry 6 "
             "significant digits (integers are never rounded), and every kind of operation has a procedure that can be executed one small step at a time:</p>")
    o.append(table(["operation", "what is written out"], [
        ["a*b", sp("one partial product per multiplier digit, most significant first, with a running sum: p1, p2, s2 = 10&middot;p1 + p2, p3, s3, ...")],
        ["a/b", sp("long division on the digits, one <code>digit r remainder</code> per quotient digit to 7 significant digits; with the product "
                   "written too (<code>9p7767r101</code>) when the divisor has 3 or more digits")],
        ["mod, gcd, lcm", sp("integer long division kept for its remainder; Euclid's algorithm; lcm = a &middot; (b / gcd)")],
        ["sqrt, cube root", "a 4-digit estimate, then one Newton step: divide, average"],
        ["a^n", "repeated multiplication for whole n up to 7; then 1/p if n is negative"],
        ["ln, log", sp("ln of the mantissa (9,000 table entries instead of 27,000) plus exponent &times; ln 10")],
        ["10^x, 2^x, e^x", "a small table for the fractional part, then a shift or one multiplication for the whole part"],
        ["n!, nPr, nCr", sp("(n-1)! &times; n; running products; nPr / r!")],
        ["sin, cos off the 0.01 grid; ln off its table", "nearest table entry plus a first-order correction"],
        ["sin, cos, tan, asin, acos, atan on the grid", "one looked-up value"]], num_from=9))
    o.append(F.worked_examples())
    o.append("<p><strong>Checks.</strong> All 23,860 arithmetic steps the generator writes for the test set were re-verified by an independent checker: zero "
             "errors. Because every value is derived only from what is printed, the final answer is the last value rounded to 4 digits, and that agrees "
             "with the true answer on 99.78% of test problems (98.65% on composite ones; the rest is 6-digit round-off). One detail mattered: an "
             "inexact 6-digit value is never allowed to end in 0 or 5 (it takes the neighbour on the true value's side), so rounding it again to 4 digits "
             "cannot be fooled by a false ...50. That moved scientific-function agreement from 99.1% to 100%.</p>")
    if has(MAIN) and ALL[MAIN].get("mean_solution_chars"):
        o.append(f"<p><strong>Cost.</strong> The model's worked solutions average {ALL[MAIN]['mean_solution_chars']:.0f} characters, answer included (the "
                 f"longest reference is 552), against {OLD_CHARS:.0f} for the old scratchpad. With a KV cache that is {ALL[MAIN]['final']['tle']:.0f} token-layer evaluations per problem, "
                 f"against {pd.tle('big2'):.0f} for the 83.3% model of the same size. Training batches are drawn per length bucket, so shapes stay "
                 "static and padding stays small.</p>")
    return "\n".join(x for x in o if x)


def results():
    o = ["<h2>3. Results</h2>"]
    rows = [("3.2M, old scratchpad", F.em(F.ARCH, "a_base_s0"), F.close(F.ARCH, "a_base_s0"), True),
            ("10.7M, old scratchpad", pd.em("big2")[0], pd.close("big2"), True)]
    if has(PILOT):
        rows.append(("3.2M, worked solutions, first version", F.em(ALL, PILOT), F.close(ALL, PILOT), False))
    if has(SMALL):
        rows.append(("3.2M, worked solutions, final version", F.em(ALL, SMALL), F.close(ALL, SMALL), False))
    if has(MAIN):
        rows.append(("10.7M, worked solutions, final version", F.em(ALL, MAIN), F.close(ALL, MAIN), False))
    o.append(F.fig_progress(rows))
    if has(MAIN):
        r = ALL[MAIN]["final"]
        o.append("<p>Per level, for the final 10.7M model (6 layers, width 384, middle two layers applied twice, 30,000 steps at batch 2,048, 2.2 hours on "
                 "one H100):</p>")
        o.append(table(["level", "exact", "within 1%", "83.3% model, exact"],
                       [[mg.LEVELS[l], pct(r["em_gen"][l]), pct(r["close"][l]), pct(pd.levels("big2")[l])] for l in range(mg.NLEVELS)]))
    cols = [("3.2M old", OKS.get("a_base_s0")), ("10.7M old", pd.OK["big2_s0"]), ("3.2M new", OKS.get(SMALL if has(SMALL) else PILOT)), ("10.7M new", OKS.get(MAIN))]
    o.append(F.fig_subskills([c for c in cols if c[1] is not None]))
    sizes = [("c_d128_s0", "0.8M (width 128)"), ("c_d192_s0", "1.8M (width 192)"), ("c_base_s0", "3.2M (width 256)")]
    if any(has(n) for n, _ in sizes):
        o.append("<p><strong>Does it need the weights?</strong> The same recipe at smaller sizes, 10,000 steps each (a third of the main run's budget):</p>")
        rws = [[lab, pct(F.em(ALL, n)), pct(F.close(ALL, n))] + [pct(x, 0) for x in ALL[n]["final"]["em_gen"]] for n, lab in sizes if has(n)]
        if has(SMALL):
            rws.append(["3.2M, 20,000 steps", E(SMALL), C(SMALL)] + [pct(x, 0) for x in ALL[SMALL]["final"]["em_gen"]])
        if has(MAIN):
            rws.append(["10.7M, 30,000 steps", E(MAIN), C(MAIN)] + [pct(x, 0) for x in ALL[MAIN]["final"]["em_gen"]])
        o.append(table(["parameters", "exact", "within 1%"] + [f"L{i+1}" for i in range(7)], rws))
    return "\n".join(o)


def fixes():
    """Three failure modes found by reading wrong solutions, and what each fix bought at a fixed size and budget."""
    before = F.jload("runs/v2abl/b_base_s0.errs.json")
    after = F.jload("runs/v2abl/c_base_s0.errs.json")
    pilot = F.jload("runs/v2/v2_3m_s0.errs.json")
    if not before:
        return ""
    o = ["<h2>4. Three things that broke, found by reading wrong solutions</h2>"]
    o.append("<p>The first version of the worked solutions already put the 3.2M model above 95%. A tool that generates every test solution and "
             "finds the first step that differs from the reference showed three separate ways the rest went wrong. None was a capacity problem.</p>")
    r = before["root"]
    o.append("<h3>A Newton step the model never took</h3>")
    x = (f"<p>Square roots were written as a 4-digit estimate, one long division, and the average of the two. In the training data the estimate was always "
         f"the correctly rounded root, so the quotient was always a hair away from it, and the model stopped dividing: it wrote plausible quotient "
         f"digits near whatever it had guessed. When its estimate matched the reference, the answer was right {r['est_same'][0]} times out of "
         f"{r['est_same'][1]}; when the estimate was off by even one unit in the last place, {r['est_diff'][0]} out of {r['est_diff'][1]}. The step that "
         "should repair a rough guess was being copied through. <strong>Fix:</strong> jitter the estimate by up to &plusmn;5 units of its last digit in "
         "training, so the division has to be real and the average has to do its job.")
    if after:
        a = after["root"]
        x += (f" Afterwards the model's estimate almost never equals the reference one, and it does not need to: {a['est_diff'][0]} right out of "
              f"{a['est_diff'][1]} when it differs.")
    o.append(x + "</p>")
    o.append("<h3>Table entries that existed only in the test set</h3>")
    x = ("<p>Test problems are held out by their whole expression. For most functions that is harmless: <code>sin(1.2)</code> the problem is held out, but "
         "<code>sin(1.2)</code> the sub-expression still turns up inside composite training problems. For a function whose every use is its own "
         "top-level problem, a held-out entry is never seen at all: e<sup>x</sup> at negative x, 10<sup>x</sup>, 2<sup>x</sup>, nCr, cube roots.")
    if pilot:
        x += (f" The first version scored {pct(pilot['sub']['5|nCr/nPr'], 0)} on nCr and {pct(pilot['sub']['5|e^x, 10^x, 2^x'], 0)} on the exponentials "
              f"while getting {pct(pilot['sub']['5|sin/cos/tan'], 0)} on sin/cos/tan.")
    x += (" <strong>Fix:</strong> split each lookup into pieces that are shared across problems. 10<sup>5.33</sup> is 10<sup>0.33</sup> (a 100-entry table) "
          "shifted; e<sup>8.4</sup> is e<sup>8</sup>&middot;e<sup>0.4</sup>; nCr is a running product and one division. A held-out problem is then made of "
          "parts that were all seen.</p>")
    o.append(x)
    d = before["div_len"]
    rate = lambda dd, k: pct(dd[k][0] / dd[k][1], 1) if k in dd and dd[k][1] else "n/a"
    o.append("<h3>Multiply and subtract in one breath</h3>")
    x = (f"<p>A long-division step was <code>digit r remainder</code>: choose the digit, multiply it by the divisor, subtract, with the product never "
         f"written. How often a division was the first step to go wrong depended almost entirely on the divisor's length: {rate(d, '1')} for 1 digit, "
         f"{rate(d, '2')} for 2, {rate(d, '3')} for 3, {rate(d, '4')} for 4 and {rate(d, '6')} for 6 (3.2M model, 10,000 steps). In 86% of those the digit was "
         "right and the remainder was not. <strong>Fix:</strong> for divisors of 3 or more digits, write the product: <code>9p7767r101</code>. The "
         "remainder becomes a plain subtraction.")
    if after:
        x += (f" Afterwards, at the same size and budget: {rate(after['div_len'], '3')} for 3 digits and {rate(after['div_len'], '6')} for 6. "
              "(Four-digit divisors are nearly all Newton steps, which the model was not really computing before, so there is no fair \"before\" for them.)")
    final = F.jload(f"runs/v2/{MAIN}.errs.json")
    if final:
        x += f" In the final 10.7M model a division is the first wrong step {final['first'].get('long division', 0)} times in {final['total']['long division']:,}."
    o.append(x + "</p>")
    if after and has("c_base_s0", "b_base_s0"):
        sb, sa = before["sub"], after["sub"]
        keys = ["4|sqrt(x)", "4|powers", "5|e^x, 10^x, 2^x", "5|nCr/nPr", "5|gcd/lcm", "5|ln/log", "2|a/b inexact", "6|composite"]
        o.append("<p>Same size (3.2M), same 10,000 steps: the second version of the scratchpad, which already split 10<sup>x</sup> and 2<sup>x</sup>, "
                 "against the final one (whose pool also gives the harder levels more of the training rows):</p>")
        o.append(table(["", "second version", "final version"], [["all levels, exact", E("b_base_s0"), E("c_base_s0")]] +
                       [[k.split("|")[1], pct(sb[k], 0), pct(sa[k], 0)] for k in keys if k in sb and k in sa]))
    return "\n".join(o)


def knobs():
    if not has("b_base_s0"):
        return ""
    o = ["<h2>5. Positional encodings, norms, the loop</h2>"]
    order = [("b_base_s0", "RoPE + LayerNorm + looped middle (baseline)"), ("b_rms_s0", "RMSNorm instead of LayerNorm"),
             ("b_noloop_s0", "no looped middle block"), ("b_abs_s0", "learned absolute positions"), ("b_nope_s0", "no positional encoding (NoPE)")]
    o.append(F.fig_abl(order, "Architecture knobs on worked solutions",
                       "3.2M parameters, 10,000 steps each, one seed; exact match. In the first write-up's experiments one seed moved results by about 2 points."))
    o.append(F.abl_table(order))
    t = []
    if has("b_nope_s0", "b_abs_s0"):
        t.append(f"<strong>Positions.</strong> With the old, short scratchpad, dropping positional information cost {100*(F.em(F.ARCH,'a_base_s0')-F.em(F.ARCH,'a_nope_s0')):.1f} "
                 f"points ({pct(F.em(F.ARCH,'a_base_s0'))} vs {pct(F.em(F.ARCH,'a_nope_s0'))}). With worked solutions it costs "
                 f"{100*(F.em(ALL,'b_base_s0')-F.em(ALL,'b_nope_s0')):.1f}, and learned absolute positions cost {100*(F.em(ALL,'b_base_s0')-F.em(ALL,'b_abs_s0')):.1f}. "
                 "The procedures are all relative lookups, the digit under this one, the remainder two items back, at offsets that slide along a "
                 "sequence of varying length. That is what rotary positions encode directly, and the losses are concentrated in multiplication and division "
                 f"(level 3: {pct(ALL['b_base_s0']['final']['em_gen'][2],0)} with RoPE, {pct(ALL['b_abs_s0']['final']['em_gen'][2],0)} learned, "
                 f"{pct(ALL['b_nope_s0']['final']['em_gen'][2],0)} none).")
    if has("b_rms_s0"):
        t.append(f"<strong>Norm.</strong> RMSNorm and LayerNorm give the same result ({E('b_rms_s0')} vs {E('b_base_s0')}; it was "
                 f"{pct(F.em(F.ARCH,'a_rms_s0'))} vs {pct(F.em(F.ARCH,'a_base_s0'))} with the old scratchpad). Nothing here depends on mean-centring.")
    if has("b_noloop_s0"):
        t.append(f"<strong>Loop.</strong> Applying the two middle layers twice, the trick that won the first write-up's ablation, is still worth "
                 f"{100*(F.em(ALL,'b_base_s0')-F.em(ALL,'b_noloop_s0')):.1f} points with no extra parameters.")
    o.append("".join(f"<p>{x}</p>" for x in t))
    return "\n".join(o)


def quant():
    o = ["<h2>6. Weight quantisation</h2>"]
    small = f"runs/v2abl/{SMALL}.quant.json" if os.path.exists(f"runs/v2abl/{SMALL}.quant.json") else f"runs/v2/{PILOT}.quant.json"
    o.append(F.fig_ptq([(small, "3.2M parameters", F.ORANGE), (f"runs/v2/{MAIN}.quant.json", "10.7M parameters", F.BLUE)]))
    q, qs = F.jload(f"runs/v2/{MAIN}.quant.json"), F.jload(small)
    if q and qs:
        m = lambda d, b: pct(float(np.mean(d[str(b)])))
        drop = lambda d, b: 100 * (float(np.mean(d["fp"])) - float(np.mean(d[str(b)])))
        o.append(f"<p>Rounding the transformer-block weights after training, with no retraining: 8 and 6 bits are free at both sizes, and 5 bits costs "
                 f"{drop(q, 5):.2f} points at 10.7M and {drop(qs, 5):.1f} at 3.2M. At 4 bits the 10.7M model keeps {m(q, 4)} and the 3.2M model drops to "
                 f"{m(qs, 4)}; at 3 bits both break ({m(q, 3)}, {m(qs, 3)}); ternary is noise. A worked solution is a few hundred consecutive correct "
                 "tokens, so a small per-token error rate is fatal, and the larger model has the margin to absorb more rounding. In bits, the 10.7M model "
                 "at 4 bits (5.3 MB) is smaller than the 3.2M model at 16 (6.4 MB), and more accurate.</p>")
    order = [("b_base_s0", "full precision (baseline)"), ("b_w4_s0", "4-bit weights, trained quantised"), ("b_w2_s0", "ternary weights, trained quantised")]
    if any(has(n) for n, _ in order[1:]):
        o.append(F.fig_abl(order, "Training with quantised weights",
                           "3.2M parameters, 10,000 steps; block weights quantised in every forward pass, straight-through gradient. Exact match."))
        o.append(F.abl_table(order))
        if has("b_w4_s0", "b_w2_s0"):
            base = F.em(ALL, "b_base_s0")
            x = (f"<p>Training with the quantiser in the loop does better than rounding afterwards. Four bits costs {100*(base-F.em(ALL,'b_w4_s0')):.1f} "
                 f"points ({E('b_w4_s0')} against {E('b_base_s0')} at full precision for the same budget). Ternary weights (&minus;1, 0, +1 and one scale "
                 f"per row), useless when applied after training, reach {E('b_w2_s0')}: {100*(base-F.em(ALL,'b_w2_s0')):.0f} points under full precision.")
            if has("d_w2_11m_s0"):
                x += (f" More weights buy part of that back: a ternary 10.7M model on the same data and budget scores {E('d_w2_11m_s0')}, with about a "
                      "third of the bits of the full-precision 3.2M model.")
            o.append(x + "</p>")
    return "\n".join(o)


def closing():
    o = ["<h2>7. What is left, and what it cost</h2>"]
    e = F.jload(f"runs/v2/{MAIN}.errs.json")
    if e:
        tot = e["total"]
        f = e["first"]
        g = lambda k: f"{f.get(k, 0)} of {tot[k]:,}"
        o.append(f"<p>The final model's remaining errors, by the first step that goes wrong: looked-up trig values ({g('trig table')}), powers and their "
                 f"tables ({g('power (table or summary)')}), long division ({g('long division')}), multi-digit multiplication ({g('multiplication with work')}) "
                 f"and addition or subtraction ({g('add/subtract')}). By problem type, what is still missing is mostly what has no procedure yet: "
                 "tan and atan away from their tables, the fractional part of a negative exponent (the model computes 10<sup>-1.16</sup> as "
                 "10<sup>-2</sup>&middot;10<sup>0.86</sup>), and the longest nCr chains. The format's own ceiling is 99.8%.</p>")
    o.append("<ul>"
             "<li><strong>More weights helped, after the format was right.</strong> " + (f"At the final format, 3.2M parameters score {E(SMALL)} and 10.7M "
             f"(with a longer run) {E(MAIN)}" if has(SMALL, MAIN) else "At the final format, 3.2M to 10.7M parameters took the model from the mid-90s to 99%") +
             ", mostly through better-memorised tables and steadier long procedures. At the old format the same step went from 77.7% to 83.3%, and "
             "a 0.8M model on the new format beats the 10.7M model on the old one.</li>"
             "<li><strong>Reading failures beat adding capacity.</strong> Each of the three fixes in section 4 came from looking at where solutions first "
             "diverged, and each was a property of the data, not of the network.</li>"
             "<li><strong>It is slower.</strong> The model now writes out arithmetic the way a person does on paper, and pays for every character.</li>"
             "</ul>")
    o.append("<p><strong>Caveats.</strong> One seed for most runs. The architecture ablations ran at 3.2M parameters and a third of the main budget, on "
             "the second of three scratchpad versions; the final format differs from it only in the fixes of section 4. Two runs were stopped early when "
             "a fix made them obsolete. Scores are exact string match on the calculator display; the within-1% numbers are there for scale.</p>")
    return "\n".join(o)


def repro():
    return """
<h2>8. Reproduce</h2>
<pre><code>python build_data.py 1500000                              # v1 pool and the 14k held-out test set (data.npz)
python build_data2.py 1000000 data23.npz 1,2,3,3,3,5,5    # worked solutions, 22M rows, harder levels weighted up
python check_gen2.py                                      # re-verifies every arithmetic step; agreement with true answers
python train2.py --name v23_11m_s0 --data data23.npz --d 384 --L 6 --heads 6 --lo 2 --hi 4 --steps 30000
python errs.py runs/v2/v23_11m_s0.pt 2 data23.npz         # first wrong step, by kind
python quant.py runs/v2/v23_11m_s0.pt v2 data23.npz       # post-training quantisation sweep</code></pre>
"""


FOOT = """
<footer>Written with an AI assistant (Claude), which also designed and ran the experiments. Code and per-run results:
<a href="https://github.com/bobrenjc93/explanations/tree/main/calculator-99/code">calculator-99/code</a>.</footer>
"""
