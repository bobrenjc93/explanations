"""The write-up. Numbers are pulled from the finished runs so the text cannot drift from the data."""
import json
import os
import numpy as np
import mathgen as mg
import pagedata as pd
import figs
import prose_static as ps
from charts import pct, table, esc, figure

TITLE = 'Should a transformer say "um"? A calculator LLM and a thinking-token ablation'
DESCRIPTION = ("A 3M-parameter transformer learns a calculator curriculum from arithmetic to TI-84 expressions; filler, discrete, soft and "
               "continuous thinking tokens, looped layers and scratchpads are compared at matched inference compute.")


def E(p, d=1):
    return pct(pd.em(p)[0], d)


def SL(p, d=1):
    """Per-seed exact match, e.g. '52.5%, 53.4%, 56.4%'."""
    return ", ".join(pct(v, d) for v in sorted(pd.em(p)[1]))


def C(p, d=1):
    return pct(pd.close(p), d)


def CL(p, d=1):
    return ", ".join(pct(float(np.mean(pd.R[n]["final"]["close"])), d) for n in pd.seeds(p))


def S(p, level, name, d=0):
    return pct(pd.sub_one(p, level, name), d)


def L(p, i, d=0):
    return pct(pd.levels(p)[i], d)


def cl_level(p, i, d=0):
    return pct(float(np.mean([pd.R[n]["final"]["close"][i] for n in pd.seeds(p)])), d)


def body():
    o = [f"<h1>{TITLE}</h1>",
         '<p class="sub">A 3-million-parameter transformer is taught a calculator, from <code>47+85</code> up to <code>sin(1.2)</code> and the '
         f"quadratic formula. Then {len(pd.R)} training runs compare ways for it to think before it answers: filler tokens, discrete and "
         "continuous thinking tokens, looped layers, scratchpads. Every number below is measured on those runs (one H100, 20,000 steps each, "
         "14,000 held-out test problems).</p>",
         tldr(), ps.idea(), ps.task_section(), ps.model_section(), results(), flagship(), takeaways(), repro(), FOOT]
    return "\n".join(x for x in o if x)


def tldr():
    b = []
    b.append(f"<li><strong>Thinking tokens trained end to end barely move exact match.</strong> No thinking: {E('none')} "
             f"(seeds {SL('none')}). Eight \"um\" tokens: {E('um8')}. Eight discrete thinking tokens: {E('disc8')}. "
             f"Eight continuous thoughts: {E('cont8')}. The gaps are no bigger than the seed-to-seed spread.</li>")
    b.append(f"<li><strong>Continuous thoughts do make the model a better estimator.</strong> Answers within 1% of the truth go from "
             f"{C('none')} to {C('cont8')} (multiplication/division: {cl_level('none', 2)} to {cl_level('cont8', 2)}), in every seed. "
             f"Filler and discrete tokens do not do this ({C('um8')}, {C('disc8')}): the gain needs a continuous channel.</li>")
    if pd.has("ut2"):
        b.append(f"<li><strong>The same compute works better as depth than as thoughts.</strong> Looping the two middle layers twice at every "
                 f"position, with no thinking positions and no new parameters, costs what 8 thoughts cost and gets <strong>{E('ut2')}</strong> "
                 f"(seeds {SL('ut2')})" + (f"; two extra ordinary layers get {E('noneL6')}" if pd.has("noneL6") else "") + ".</li>")
    if pd.has("cont16") and pd.has("win1") and pd.has("krand_cont"):
        byk = pd.R[pd.seeds("krand_cont")[0]]["by_K"]
        b.append(f"<li><strong>More thoughts and more memory changed nothing.</strong> 16 thoughts score {E('cont16')}; a single overwritten slot "
                 f"scores {E('win1')}; a model trained with a variable budget goes {pct(byk['0']['mean_em'])} &rarr; {pct(byk['1']['mean_em'])} with "
                 f"its first thought and is flat from there to 16.</li>")
    if pd.has("cot"):
        x = (f"<li><strong>The training signal is the bottleneck.</strong> Teaching the same 3M network to write its intermediate values (a supervised "
             f"scratchpad) gives <strong>{E('cot')}</strong>, at lower average inference cost than eight thoughts, and takes 3-4 operand sums "
             f"from {S('none', 1, '3-4 operands')} to {S('cot', 1, '3-4 operands')}.")
        if pd.has("coco_cont8"):
            x += f" Distilling that scratchpad into continuous thoughts in four short stages lost the whole gain ({E('coco_cont8')})."
        b.append(x + "</li>")
    if pd.has("nocurr_none"):
        b.append(f"<li><strong>The curriculum did not help.</strong> Arithmetic first, TI-84 last: {E('none')}. All seven levels mixed from step 0, "
                 f"same budget: {E('nocurr_none')}.</li>")
    if pd.has("big_cot"):
        r = pd.R["big_cot_s0"]["final"]
        b.append(f"<li><strong>The calculator.</strong> Scratchpad recipe at 10.7M parameters: {pct(float(np.mean(r['em_gen'])))} exact and "
                 f"{pct(float(np.mean(r['close'])))} within 1% on held-out problems; near-perfect on add/subtract, {pct(r['em_gen'][3],0)} on "
                 f"multi-operation expressions, {pct(r['em_gen'][5],0)} on scientific functions.</li>")
    return '<div class="tldr"><p><strong>TL;DR.</strong></p><ul>' + "".join(b) + "</ul></div>"


def results():
    o = ["<h2>4. Results</h2>"]
    add = o.append
    lv = pd.levels("none")

    # ---------------------------------------------------------------- 4.1
    add("<h3>4.1 The curriculum, and how far a 3M model gets</h3>")
    add(figs.fig_curriculum())
    add(f"<p>Each level takes off at the step it is introduced and keeps improving while it is replayed. After 20,000 steps the plain model "
        f"is a solid four-function calculator on short problems and a poor scientific one: {L('none',0)} on small sums, {L('none',1)} on long sums, "
        f"{L('none',2)} on multiplication and division, {L('none',4)} on decimals/powers/roots, {L('none',5)} on scientific functions, "
        f"{L('none',3)} on multi-operation expressions and {L('none',6)} on composite ones. Level-averaged: <strong>{E('none')}</strong>.</p>")
    add(f'<p class="note"><strong>How much is noise?</strong> Three seeds of this exact model finish at {SL("none")}; rerunning seed 0 on the same '
        "code gave 55.2% instead of 53.4% (GPU nondeterminism alone). A gap under about 2 points between two single runs is not a finding. "
        "Where a variant has several seeds the charts show each one as a dot.</p>")
    if pd.has("nocurr_none"):
        n = len(pd.seeds("nocurr_none"))
        add(f"<p><strong>Does teaching in order help? No.</strong> Mixing all seven levels uniformly from step 0, with the same 20,000 steps, ends at "
            f"<strong>{E('nocurr_none')}</strong>" + (f" (seeds {SL('nocurr_none')})" if n > 1 else " (one seed)") +
            f", above every curriculum seed ({SL('none')}). The staged schedule spends its early steps on problems the model masters within a few "
            "hundred updates and shows it the hard levels late: composite problems only exist for the last 31% of training. The staircase makes "
            "a nice picture; as a recipe at a fixed budget it loses to plain mixing. Every comparison below was run under the curriculum, so the "
            "variants are comparable with each other, but none of them is the best this network can do.</p>")

    # ---------------------------------------------------------------- 4.2
    add("<h3>4.2 What should a thought be made of?</h3>")
    a, b, tab = figs.fig_media()
    add(a)
    add(f"<p>Eight thinking steps cost 49% more inference compute. In exact match they buy almost nothing: every variant is inside or just "
        f"above the plain model's seed range ({SL('none')}).</p>")
    add(b)
    add(f"<p>The second chart is where a real difference shows. Feeding back a <em>continuous</em> vector makes the printed number land within "
        f"1% of the truth far more often: {C('none')} without thinking, {C('cont8')} with continuous thoughts ({CL('cont8')} per seed)"
        + (f", {C('soft8')} with soft tokens" if pd.has("soft8") else "") +
        f". Filler tokens ({C('um8')}) and discrete tokens ({C('disc8')}) stay at the baseline. So the extra steps are doing arithmetic of a kind, "
        "an estimate of magnitude and leading digits, and that estimate only survives the trip to the next step if the channel is continuous. "
        "What they do not deliver is the last digit.</p>")
    re_ = pd.PROBES.get("relerr", {})
    if all(k in re_ for k in ("none_s0", "none_s1", "none_s2", "cont8_s0", "cont8_s1", "cont8_s2")):
        w = lambda names, t: float(np.mean([np.mean(re_[n][t]) for n in names]))
        nn, cc = ["none_s0", "none_s1", "none_s2"], ["cont8_s0", "cont8_s1", "cont8_s2"]
        add(f"<p>The same holds at a looser tolerance: within 10% of the truth, {pct(w(nn, '0.1'))} without thinking and {pct(w(cc, '0.1'))} with "
            f"continuous thoughts. By level the within-1% gain is largest on multiplication/division ({cl_level('none', 2)} &rarr; {cl_level('cont8', 2)}) "
            f"and multi-operation expressions ({cl_level('none', 3)} &rarr; {cl_level('cont8', 3)}), the problems whose exact digits the model cannot get.</p>")
    add(tab)

    # ---------------------------------------------------------------- 4.3
    add("<h3>4.3 Which skills move</h3>")
    add(figs.fig_subskills())
    k34 = (1, "3-4 operands")
    per = lambda p: ", ".join(pct(float(pd.OK[n][pd.SUB_IDX[k34]].mean()), 0) for n in pd.seeds(p) if n in pd.OK)
    add("<p>Reading across the rows, the problem types fall into three groups.</p><ul>"
        f"<li><strong>One step, needs precision.</strong> Reciprocals ({S('none',4,'1/x')} &rarr; {S('cont8',4,'1/x')} with continuous thoughts), "
        f"exact division ({S('none',2,'a/b exact')} &rarr; {S('cont8',2,'a/b exact')}), 6-8 digit sums "
        f"({S('none',1,'2 operands, 6-8 digits')} &rarr; {S('cont8',1,'2 operands, 6-8 digits')}), <code>abs</code> of a signed sum "
        f"({S('none',4,'abs(a±b)')} &rarr; {S('cont8',4,'abs(a±b)')}). This is where continuous thoughts pay off in exact match.</li>"
        f"<li><strong>Several steps chained.</strong> Sums of 3-4 operands, expressions with 3-4 operations, multiplication by a 2- or 3-digit "
        f"number. No latent variant moves these reliably. Sums of 3-4 operands look like a counter-example for discrete tokens, but it is a "
        f"lottery every model plays: per seed, no thinking gets {per('none')}; discrete tokens {per('disc8')}; continuous thoughts {per('cont8')}.</li>"
        + (f"<li><strong>The scratchpad column.</strong> Writing the intermediate values fixes exactly the chained group: 3-4 operand sums "
           f"{S('none',1,'3-4 operands')} &rarr; {S('cot',1,'3-4 operands')}, 2-digit multipliers {S('none',2,'a*b, smaller has 2 digits')} &rarr; "
           f"{S('cot',2,'a*b, smaller has 2 digits')}, 3-digit multipliers {S('none',2,'a*b, smaller has 3 digits')} &rarr; "
           f"{S('cot',2,'a*b, smaller has 3 digits')}, 4-operation expressions {S('none',3,'4 operations')} &rarr; {S('cot',3,'4 operations')}. "
           f"(Its {S('cot',5,'sin/cos/tan')} on sin/cos/tan is partly a lookup table: test problems are held out as whole expressions, but "
           f"<code>sin(-0.14)</code> also appears inside composite training problems, where the scratchpad spells out its value.)</li>"
           if pd.has("cot") else "") + "</ul>")

    # ---------------------------------------------------------------- 4.4
    add("<h3>4.4 Where to spend the extra compute</h3>")
    add(figs.fig_compute())
    t = []
    if pd.has("cont16"):
        x = f"<strong>More thoughts.</strong> Doubling to 16 continuous thoughts changes nothing: {E('cont16')} exact and {C('cont16')} within 1%, against {E('cont8')} and {C('cont8')} for 8"
        if pd.has("cont4"):
            x += f" (4 thoughts: {E('cont4')}, {C('cont4')})"
        t.append(x + ". Whatever the thoughts do, they are done doing it early.")
    if pd.has("mid16"):
        t.append(f"<strong>Whole model or middle layers?</strong> Sixteen thoughts that pass only through layers 2-3 cost the same as eight "
                 f"full ones. They score {E('mid16')} exact and {C('mid16')} within 1%: the estimation gain of whole-model thoughts ({C('cont8')}) is gone. "
                 "In a 4-layer model the first and last layer are evidently part of the computation a thought needs (one seed).")
    if pd.has("ut2"):
        n = len(pd.seeds("ut2"))
        t.append(f"<strong>Looped transformer.</strong> No thinking positions at all; instead layers 2-3 are applied twice at <em>every</em> "
                 f"position, prompt and answer alike. Same parameters as the plain model, {pd.tle('ut2'):.0f} token-layer evaluations (the price of 8 thoughts). "
                 f"It scores <strong>{E('ut2')}</strong> exact" + (f" (seeds {SL('ut2')})" if n > 1 else " (one seed)") +
                 f" and {C('ut2')} within 1%: the best exact match of anything trained without scratchpad labels"
                 + (f", and looping three times gives {E('ut3')}" if pd.has("ut3") else "") + ".")
    if pd.has("noneL6"):
        t.append(f"<strong>Or just add layers.</strong> Six distinct layers instead of four, no thinking: the same inference cost again, but 50% more "
                 f"parameters. {E('noneL6')} exact, {C('noneL6')} within 1%.")
    add("".join(f"<p>{x}</p>" for x in t))
    if pd.has("ut2"):
        add("<p>The conversation ended on this fork: \"Do you actually want to create new token positions, or do you want to give the model "
            "additional computation without advancing its sequence position?\" For exact arithmetic on this task the results side with the "
            "second option. Extra depth applied where the digits are, at every prompt and answer character, beats extra positions appended after the "
            "<code>=</code> sign.</p>")

    # ---------------------------------------------------------------- 4.5
    add("<h3>4.5 Working memory, or just more forward passes?</h3>")
    ko = pd.PROBES.get("knockout", {}).get("cont8_s0")
    if ko:
        add("<p>Two ways to ask. First, take the trained 8-thought model and break things at inference:</p>")
        add(figs.knockout_table())
        add(f"<p>Everything collapses. Hiding all but the newest thought from the answer drops it from {pct(ko['full']['mean'])} to "
            f"{pct(ko['window1']['mean'])}; giving it another problem's thoughts, {pct(ko['shuffle']['mean'])}. So in the model as trained the "
            "thoughts are not filler: they carry problem-specific content, and the answer reads many of them, not only the last.</p>")
    if pd.has("win1"):
        add(f"<p>Second, the question the knock-out cannot answer: does the model <em>need</em> that memory, or does it merely use what it is "
            f"given? Train with a single slot from the start, where each thought overwrites the previous one, so there are 8 serial steps of "
            f"compute but nothing accumulates. Result: {E('win1')} exact, {C('win1')} within 1%, against {E('cont8')} and {C('cont8')} for the "
            f"growing scratchpad"
            + (f"; a fixed workspace of 4 vectors refined together for 2 rounds gets {E('slots4')} and {C('slots4')}" if pd.has("slots4") else "")
            + ".</p>")

    # ---------------------------------------------------------------- 4.6
    add("<h3>4.6 The training signal</h3>")
    if pd.has("cot"):
        add(f"<p>The conversation flagged it: \"There's one major challenge, though: training.\" Everything above trains the thinking from the "
            f"answer loss alone, and it mostly fails to discover multi-step procedures. Give the same 3.18M-parameter network the intermediate "
            f"values as text to imitate and it reaches <strong>{E('cot')}</strong> exact ({C('cot')} within 1%). On this test set the scratchpad "
            f"is not even the expensive option: most problems have little or nothing to write, so it averages {pd.tle('cot'):.0f} token-layer "
            f"evaluations against {pd.tle('cont8'):.0f} for eight thoughts. It costs more only where it helps most (composite problems: 146 vs 125).</p>")
    add(figs.fig_bars("Same network, three training signals", "Exact match, level average. The scratchpad rows use intermediate-value labels the others never see.",
                      ["none", "cont8", "coco_none", "coco_um8", "coco_cont8", "cot_cont8", "cot"], label_w=330))
    if pd.has("coco_cont8"):
        add(f"<p>The obvious next move is to get the scratchpad's benefit without paying for its tokens: train with the scratchpad, then remove "
            f"it one item at a time while handing the model continuous thoughts to hold what the removed text used to hold (the idea behind "
            f"Coconut and stepwise internalisation). Here that <strong>did not work</strong>: the distilled model ends at {E('coco_cont8')}, back at "
            f"the no-thinking level.</p>")
        add(figs.fig_distill())
        add("<p>The curve shows where it goes: each removal costs accuracy that the next 2,000 steps do not win back. The honest reading is "
            "narrow. With 2,000 steps per stage and two thoughts per removed item, this network did not learn to carry the removed values in "
            "its thoughts. Published successes use much longer schedules; this run says the transfer is not free, not that it is impossible.</p>")

    # ---------------------------------------------------------------- 4.7
    add("<h3>4.7 One model, adjustable thinking time</h3>")
    ks = pd.PROBES.get("ksweep", {}).get("cont8_s0")
    if ks:
        add(f"<p>A model trained with exactly 8 thoughts is welded to that number. Ask it to answer after 4 and it scores "
            f"{pct(ks['4']['mean'])}; after 0, {pct(ks['0']['mean'])}; let it run to 16 and it drifts down to {pct(ks['16']['mean'])}. "
            "It never learned to answer early, and nothing told it when it was done.</p>")
    add(figs.fig_ksweep())
    if pd.has("krand_cont"):
        byk = pd.R[pd.seeds("krand_cont")[0]]["by_K"]
        g = lambda k: pct(byk[str(k)]["mean_em"])
        add(f"<p>Training with a random number of thoughts each step (0, 1, 2, 4, 8 or 16) removes the brittleness: one set of weights answers at "
            f"any budget it saw and only degrades beyond it ({g(24)} at 24 thoughts, {g(32)} at 32). But look at the shape. {g(0)} with no "
            f"thoughts, {g(1)} with one, and then nothing: {g(2)}, {g(4)}, {g(8)}, {g(16)}. The model does whatever it is going to do in the first "
            f"thought and treats the rest as a waiting room. The conversation predicted this (\"You'd also need to prevent the model from learning "
            f"to skip the additional computation\"), and the answer loss alone does not prevent it. The flexibility also costs accuracy: {g(8)} "
            f"at 8 thoughts against {E('cont8')} for the model that only ever used 8.</p>")
        h = pd.PROBES.get("halting", {}).get("krand_cont_s0")
        if h:
            c99 = [c for c in h["curve"] if c["tau"] == 0.99][0]
            add(f"<p>That also settles \"some mechanism for deciding when to stop\" for this model. Decoding after each budget and stopping once the "
                f"answer's probability clears 0.99 spends {c99['mean_K']:.1f} thoughts per problem on average for {pct(c99['acc'])}, no better than "
                f"always thinking once. Even an oracle that picks the best budget per problem would reach only {pct(h['oracle_acc'])}. There is "
                "no extra accuracy at higher budgets for a stopping rule to find.</p>")

    # ---------------------------------------------------------------- 4.8
    add("<h3>4.8 What the thoughts say</h3>")
    lens = pd.PROBES.get("lens")
    if lens:
        m = lens["match_char"][0]
        add("<p>The thoughts are never supervised, so nothing forces them to look like tokens. Still, pushing each one through the ordinary "
            "output head shows the answer condensing step by step:</p>")
        add(figs.fig_lens())
        add(f"<p>After the first thought the top character matches the answer's first character on {pct(m[0],0)} of problems; by the fifth, "
            f"{pct(m[4],0)}; by the eighth, {pct(m[7],0)} (the eighth is the position the first character is actually read from). In the "
            "examples you can watch a leading digit being revised: <code>5*(8*83)</code> goes 5, 5, 4, 4, 3, 3, 3, 3 on its way to 3320. "
            f"The second and third characters never show up this way (they stay near {pct(lens['match_char'][1][-1],0)}, about chance): "
            "whatever else the thoughts carry is not written in the output head's alphabet. This fits section 4.2: the thinking steps settle "
            "sign, magnitude and leading digit.</p>")
    return "\n".join(x for x in o if x)


def flagship():
    if not pd.has("big_cot"):
        return ""
    r = pd.R["big_cot_s0"]["final"]
    o = ["<h2>5. The calculator</h2>"]
    o.append(f"<p>The recipe that works is the unglamorous one, so the final model uses it: written scratchpad, the same curriculum and 20,000 steps, "
             f"scaled up to 6 layers and width 384 ({pd.R['big_cot_s0']['model']['params']/1e6:.1f}M parameters). It prints exactly the right display on "
             f"<strong>{pct(float(np.mean(r['em_gen'])))}</strong> of held-out problems and lands within 1% on <strong>{pct(float(np.mean(r['close'])))}</strong>.</p>")
    rows = [[mg.LEVELS[l], pct(pd.levels('none')[l], 0), pct(float(np.mean([pd.R[n]['final']['close'][l] for n in pd.seeds('none')])), 0),
             pct(r["em_gen"][l], 0), pct(r["close"][l], 0)] for l in range(mg.NLEVELS)]
    o.append(table(["level", "3M, no thinking: exact", "within 1%", "10.7M, scratchpad: exact", "within 1%"], rows))
    o.append("<p>A session with it, on test problems it has never seen (four per level, picked at random, not by outcome):</p>")
    o.append(figs.calculator_table("big_cot_s0"))
    o.append("<p>So: a reliable four-function calculator with parentheses, a usable one for powers, roots and trig, and an unreliable one for "
             "anything that needs four correct digits of a transcendental function or a procedure it has no scratchpad for (gcd, nCr, "
             "e<sup>x</sup>). It was still improving when training stopped.</p>")
    return "\n".join(o)


def takeaways():
    o = ["<h2>6. What this says about the idea</h2>", "<ul>"]
    o.append("<li><strong>The \"um\" itself does nothing.</strong> Extra positions holding a fixed filler bought no accuracy at this size. "
             "Whatever a pause does for a person, here a thinking step has to carry content to matter.</li>")
    o.append("<li><strong>Continuous beats discrete as a channel, for estimating.</strong> A vector fed back unquantised lets the model refine "
             "sign, magnitude and leading digit over a few steps; ten bits per step and filler do not. None of them produced exact multi-step "
             "arithmetic.</li>")
    if pd.has("ut2"):
        o.append(f"<li><strong>New positions or more compute per position? More compute per position.</strong> A looped middle block "
                 f"({E('ut2')}) beat every thinking-token variant ({E('cont8')} at best with several seeds) at the same inference cost and the same "
                 "parameter count. Thoughts add compute only after the <code>=</code> sign; the loop adds it under every digit of the "
                 "prompt and the answer, which is where arithmetic happens.</li>")
    if pd.has("win1") and pd.has("cont16"):
        o.append(f"<li><strong>Growing scratchpad or fixed state? It did not matter.</strong> One overwritten slot ({E('win1')}), a growing "
                 f"scratchpad of 8 ({E('cont8')}) and of 16 ({E('cont16')}) are the same within noise, and the variable-budget model stopped "
                 "using thoughts after the first. The trained model does read its whole scratchpad (hide it and accuracy collapses), but "
                 "it does not need one.</li>")
    if pd.has("cot"):
        o.append(f"<li><strong>Supervision of the intermediate steps is what limits it.</strong> The same weights go from {E('none')} to "
                 f"{E('cot')} when shown what to write down. End-to-end gradient through eight latent steps did not find those procedures, and a "
                 "short distillation did not transfer them.</li>")
    if pd.has("nocurr_none"):
        o.append(f"<li><strong>And the easy-to-hard curriculum cost accuracy</strong> at a fixed budget ({E('none')} with it, {E('nocurr_none')} "
                 "without). Harder skills need the steps more than they need the ordering.</li>")
    o.append("</ul>")
    o.append("<p><strong>Caveats.</strong> One small model family, 20,000 steps, one task. Many variants have a single seed and the seed spread "
             "is about &plusmn;2 points, so trust the large or replicated effects: the within-1% gain of continuous thoughts (every continuous "
             "seed beats every baseline seed), the looped-transformer gain (likewise), the scratchpad gain, the knock-out collapses, the "
             "fixed-K brittleness and the flat variable-K curve. Discrete tokens were trained with straight-through Gumbel-softmax, one of several "
             "options. Thoughts were only allowed before the first answer character, never between digits. The distillation stages were short. "
             "A 4-layer model has little \"middle\", so the middle-layer-thoughts result may not carry to deep models.</p>")
    o.append("<p><strong>What I would run next.</strong> The loop and the thoughts together (continuous thoughts through a looped block); "
             "longer distillation stages with more thoughts per removed item; an auxiliary loss that asks each thought to predict a scratchpad "
             "value, so latent steps get supervision without spending tokens at inference; thoughts interleaved between answer digits; and a "
             "penalty or curriculum that stops a variable-budget model from front-loading everything into its first thought.</p>")
    return "\n".join(o)


def repro():
    return """
<h2>7. Reproduce</h2>
<pre><code>uv venv .venv &amp;&amp; uv pip install --python .venv/bin/python torch numpy
python build_data.py 1500000          # 10.5M training problems + 14k held-out, ~20 s on 28 cores
python check_gen.py                   # printer/parser/evaluator cross-check
python train.py --name cont8_s0 --mode cont --K 8          # one variant, ~8 GPU-minutes on an H100
python sched.py jobs_main.txt runs/main 3                  # the whole sweep
python probes.py runs/main &amp;&amp; python make_page.py index.html</code></pre>
"""


FOOT = """
<footer>Written with an AI assistant (Claude), which also designed and ran the experiment. Code, per-run results and the generator for
this page: <a href="https://github.com/bobrenjc93/explanations/tree/main/thinking-tokens/code">thinking-tokens/code</a>.</footer>
"""
