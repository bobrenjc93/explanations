"""Sections of the write-up that describe the setup: the idea, the task, the model."""
from charts import seq, table, esc
import figs
import pagedata as pd

def idea():
    head = """
<h2>1. The idea, and the questions it turned into</h2>

<p>People say "um" when the next word needs more time. A transformer has no such option: every token gets exactly one trip
through the layers, and whatever was not computed on that trip has to be squeezed into the following tokens. The proposal
(from a voice conversation, paraphrased) is to give the model the "um": let it take extra steps before it speaks, and ask
<em>what those steps should be made of</em>.</p>

<p>The conversation walked through a design space. Each fork became a set of models here:</p>
"""
    rows = [
        ["What is a thought? A filler token, one of 1,024 special thinking tokens, a probability-weighted mix of their embeddings "
         "(z = pE), or a free vector from a dedicated head (z = W·h)?",
         "<span><b>um</b>, <b>disc</b>, <b>soft</b>, <b>cont</b>"
         + (", plus <b>direct</b> (the last hidden state fed straight back, as in Coconut)" if pd.has("direct8") else "") + "</span>"],
        ["Loop the whole model, or only its middle layers?",
         "<span><b>cont</b> (thought passes through all 4 layers) vs <b>mid</b> (layers 2-3 only, half the cost per thought) vs a "
         "<b>looped transformer</b> (layers 2-3 applied twice at every position, no new positions)</span>"],
        ["Should the latent state be one vector, a fixed matrix of vectors, or a scratchpad that grows like a context window?",
         "<span>append-only scratchpad vs <b>1 slot</b> (each thought overwrites the last)"
         + (" vs a <b>workspace</b> of 4 vectors refined together for 2 rounds" if pd.has("slots4") else "") + "</span>"],
        ["Are latent tokens working memory, or just a way to get more forward passes?",
         "<span>knock-outs on the trained model (hide old thoughts, zero them, swap in another problem's thoughts), and the 1-slot model, which has the compute but not the memory</span>"],
        ["Does more inference compute buy more accuracy, at equal parameters?",
         "<span>" + ("4, 8 and 16" if pd.has("cont4") else "8 and 16") + " thoughts; one model trained with a random number of thoughts; stopping by confidence</span>"],
        ["\"There's one major challenge, though: training.\"",
         "<span>end-to-end from the answer loss alone vs a supervised written scratchpad (<b>CoT</b>) vs writing the scratchpad first and then "
         "<b>distilling</b> it into the thoughts stage by stage</span>"],
    ]
    return head + table(["question raised", "what was trained to answer it"], rows, num_from=9)


TASK_HEAD = """
<h2>2. The task: a calculator, taught in order</h2>

<p>The model reads an expression one character at a time and must print what a calculator would show. The rule for the display is
fixed: <strong>integers in full, everything else rounded to 4 significant digits</strong> (rational arithmetic is exact, so
<code>2.5*4</code> prints <code>10</code>, and <code>7/3</code> prints <code>2.333</code>). Answers are written most-significant digit
first, like a display, which is the hard direction: the first character already depends on every carry.</p>

<p>Seven levels, introduced one after another. A new level takes half of every batch while the earlier ones are replayed in the
other half; the last eighth of training mixes all seven evenly.</p>
"""

TASK_TAIL = """
<p>Every problem is an expression tree with one evaluator, and a separate recursive-descent parser re-reads the printed string to
check that the printer's parentheses mean what the tree meant (280,000 random problems, zero mismatches; Python's own
<code>eval</code> agrees on the 274,064 it can express). Training draws from a pool of 10.5 million generated problems. The test
set is 2,000 problems per level whose expression strings hash into a bucket that training never sees, so every reported number is
on <strong>unseen expressions</strong>. Scoring is exact string match; "within 1%" is also reported because a display that reads
<code>0.9319</code> instead of <code>0.932</code> is wrong by exact match and nearly right as a number.</p>
"""


def model_section():
    p = [("4", ""), ("7", ""), ("+", ""), ("8", ""), ("5", ""), ("=", "")]
    ans = [("1", "an"), ("3", "an"), ("2", "an"), ("⏎", "an")]
    o = ["""
<h2>3. One transformer, many ways to think</h2>

<p>The base model is deliberately small so that dozens of them could be trained: a causal transformer with 4 layers, width 256,
4 heads, rotary positions and a 44-character vocabulary, <strong>3.18M parameters</strong>. Every variant below is that same network;
the thinking mechanisms add between 0 and 0.5M parameters (the 1,024-word thinking vocabulary is the largest). All are trained
for 20,000 steps at batch 2,048 (41M problems) with the same data order, optimiser and schedule.</p>

<p>A sequence is the prompt, then the thinking, then the answer. Only the answer characters are ever supervised; the thinking
positions get their gradient through the answer.</p>
"""]
    o.append("<p><strong>No thinking.</strong> The position holding <code>=</code> must produce the first digit.</p>")
    o.append(seq(*p, "answer:", *ans))
    o.append('<p><strong>"um" tokens.</strong> K copies of one learned filler embedding. Extra positions, so extra parallel compute, '
             "but the filler itself carries nothing forward.</p>")
    o.append(seq(*p, *[("um", "um")] * 4, "answer:", *ans))
    o.append("<p><strong>Discrete thinking tokens.</strong> At each step the model picks one of 1,024 thinking words and its embedding "
             "is fed back in (trained with straight-through Gumbel-softmax). 10 bits per step. <strong>Soft tokens</strong> feed back the "
             "probability-weighted average of those embeddings instead of picking one.</p>")
    o.append(seq(*p, ("t417", "dt"), ("t33", "dt"), ("t901", "dt"), ("t5", "dt"), "answer:", *ans))
    o.append("<p><strong>Continuous thoughts.</strong> The final hidden state goes through a small linear head and comes back as the next "
             "input vector: 256 unconstrained numbers per step, never forced through a vocabulary. Each thought is a new position, "
             "so later thoughts and the answer can attend to all earlier ones.</p>")
    o.append(seq(*p, ("z1", "th"), ("z2", "th"), ("z3", "th"), ("z4", "th"), "answer:", *ans))
    o.append("<p><strong>Written scratchpad</strong> (the supervised reference). The model is trained to write the intermediate values in "
             "evaluation order, then the answer. This uses labels the other variants never see, and many more tokens.</p>")
    o.append(seq(("5", ""), ("2", ""), ("*", ""), ("1", ""), ("7", ""), ("=", ""), "scratch:", *[(c, "sc") for c in "[364,520]"], "answer:",
                 ("8", "an"), ("8", "an"), ("4", "an"), ("⏎", "an")))
    o.append("""
<p>The answer's first character is always read off the last thinking position (or off <code>=</code> when there is none). Thoughts and
answer characters both count their positions from the <code>=</code> sign, so the digit geometry the answer sees does not shift with K.</p>

<p><strong>Counting compute.</strong> With a KV cache, the cost of inference is the number of times a transformer block is applied to a
position. Call that a <em>token-layer evaluation</em>: about 12·d² multiply-adds each, with attention over the cache adding under 2% at
these lengths. A problem with p prompt tokens, K thoughts and a answer characters costs (p + K + a) × 4 of them; a thought that only
passes through the two middle layers costs 2 instead of 4. The average test problem costs 65 with no thinking, so 8 full thoughts are
+49% compute and a looped middle block is +50% at every position.</p>
""")
    return "".join(o)


def task_section():
    return TASK_HEAD + figs.level_table() + TASK_TAIL
