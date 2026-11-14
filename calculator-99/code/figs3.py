"""Figures for the change-by-change ladder, the thinking-token test and the compute accounting."""
import json, os, re
import numpy as np
import mathgen as mg, mathgen2 as m2
import pagedata as pd
import figs2 as F
from charts import *

GRAY, BLUE, ORANGE, AQUA = "--base", "--s1", "--s2", "--s3"
LAD, LADLOG, LADOK = F.load_dir("runs/ladder")

# The timeline: every idea in the order it was tried. (run, label, what it is, run it is compared with, kept?)
TIMELINE = [
    ("U0_answer_s0", "answer only", "No thinking of any kind: the first answer character is produced right after the <code>=</code> sign.", None, True),
    ("U1_um8_s0", '+ 8 "um" thinking tokens', "Thinking tokens with no content: eight copies of one learned filler embedding before the answer.", "U0_answer_s0", False),
    ("U2_cont8_s0", "+ 8 continuous thinking tokens", "Thinking tokens with latent content: the model's last hidden state is fed back as its next input, eight times, before the answer.", "U0_answer_s0", False),
    ("U3_disc8_s0", "+ unsupervised thinking block", "A thinking block of 8 free tokens from a 1,024-word thinking vocabulary. The model chooses them; only the answer is graded (straight-through Gumbel-softmax).", "U0_answer_s0", False),
    ("U4_loop_s0", "+ looped middle block", "Layers 2-3 applied twice at every position. No new parameters, no thinking positions.", "U0_answer_s0", True),
    ("L0_v1_s0", "+ supervised thinking block: values", "<code>&lt;think&gt;</code> the intermediate values, to 4 significant digits <code>&lt;/think&gt;</code> then the answer. This is the first write-up's scratchpad.", "U4_loop_s0", True),
    ("L1_values6_s0", "+ 6-digit values, one step per operation", "Inside the block every operation is restated with its result (<code>93*88=8184</code>) and values carry 6 significant digits. No procedure is written out.", "L0_v1_s0", True),
    ("L2_v20_s0", "+ procedures in the block", "Partial products with running sums, long division, Euclid for gcd/lcm/mod, a Newton step for square roots, repeated multiplication for powers, mantissa tables for ln/log.", "L1_values6_s0", True),
    ("L3_v21_s0", "+ shared lookups, off-grid corrections", "10<sup>x</sup> and 2<sup>x</sup> from a 100-entry table for the fractional part; sin, cos and ln away from their tables by a first-order correction.", "L2_v20_s0", True),
    ("L4_v22_s0", "+ jittered Newton, e^x split, nCr chains", "Root estimates perturbed in training; e<sup>x</sup> split into whole and fractional part; n!, nPr, nCr as running products; cube roots by Newton.", "L3_v21_s0", True),
    ("L5_v23_s0", "+ products written in long division", "<code>9p7767r101</code> instead of <code>9r101</code> when the divisor has 3 or more digits.", "L4_v22_s0", True),
    ("L6_weighted_s0", "+ pool weighted to hard levels", "Training rows per level in the ratio 1:2:3:3:3:5:5 instead of equal.", "L5_v23_s0", True),
    ("T_um8_s0", '+ 8 "um" tokens before the block', "The filler thinking tokens again, now in front of a full thinking block.", "L6_weighted_s0", False),
    ("T_cont8_s0", "+ 8 continuous thoughts before the block", "The latent thinking tokens again, in front of a full thinking block.", "L6_weighted_s0", False),
    ("E_effort_s0", "+ thinking-effort dial (at full effort)", "One model trained to answer with no block, a brief block (values only) or a full block, selected by the token that opens the block. Shown here with the block forced to full.", "L6_weighted_s0", False),
]
RUNGS = [(n, l, w) for n, l, w, _, _ in TIMELINE]  # (run, short label, what changed)


TOK = F.jload("runs/ladder/target_tokens.json") or {}


def target_tokens(R):
    """Mean target tokens per training row, counted over the whole pool (token_stats.py)."""
    return TOK.get(R["args"]["data"], R["compute"]["mean_target_tokens"])


def recs(name, key):
    """[(step, value)] of one kind of log record. The training loss was logged as summed loss / (batch x an estimate of the tokens per row
    taken from the head of each bucket, which is level 1 only for the short formats); it is rescaled here to the true token count."""
    scale = LAD[name]["compute"]["mean_target_tokens"] / target_tokens(LAD[name]) if key == "train_loss" and name in LAD else 1.0
    return [(r["step"], r[key] * scale if key == "train_loss" else r[key]) for r in LADLOG.get(name, []) if key in r]


def fpt(d):
    """Forward FLOPs for one token through one layer application (the matmuls: 12 d^2 multiply-adds)."""
    return 24.0 * d * d


def train_pflops(R):
    c = R["compute"]
    return R["args"]["steps"] * c["padded_positions_per_step"] * c["layer_applications"] * 3 * fpt(R["model"]["d"]) / 1e15


def infer_gflops(R):
    return R["final"]["tle"] * fpt(R["model"]["d"]) / 1e9


def bucket_positions(data_file, bs):
    """Padded positions per optimiser step for a bucketed pool (prompt field + bucket width per row)."""
    Z = np.load(data_file)
    W = [int(w) for w in Z["widths"]]
    n = np.array([len(Z[f"L{b}"]) for b in range(len(W))], float)
    share = n / n.sum()
    return float(sum(bs * s * (mg.P_MAX + w) for s, w in zip(share, W))), float(bs * (mg.P_MAX + W[-1]))


def ladder_list():
    cap = lambda t: t[0].upper() + t[1:]
    out = []
    for i, (name, lab, what, base, kept) in enumerate(TIMELINE):
        if name in LAD:
            out.append(f'<li value="{i}"><strong>{cap(lab.lstrip("+ "))}.</strong> {what}' + ("" if kept else " <em>Not kept.</em>") + "</li>")
    return "<ol>" + "".join(out) + "</ol>"


def ladder_table():
    rows = []
    for i, (name, lab, what, base, kept) in enumerate(TIMELINE):
        if name not in LAD:
            continue
        R = LAD[name]
        e = F.em(LAD, name)
        loss = [v for _, v in recs(name, "train_loss")][-1]
        mt = target_tokens(R)
        gain = "" if base is None or base not in LAD else f"{100*(e-F.em(LAD, base)):+.1f}"
        rows.append([f"{i}. {lab}", pct(e), gain, "yes" if kept else "no", pct(F.close(LAD, name)), f"{mt:.0f}", f"{loss:.4f}", f"{loss*mt:.2f}",
                     f"{train_pflops(R):.0f}", f"{infer_gflops(R):.2f}"])
    return table(["idea (each on top of the kept ones before it)", "exact", "gain", "kept", "within 1%", "tokens / problem", "loss / token", "loss / problem",
                  "train PFLOP", "infer GFLOP"], rows)


def _panel(title, sub, series, xs, xl, **kw):
    chart = line_chart(series, xs, xl, w=400, h=180, end_labels=False, show_legend=False, tv=False, **kw)
    return f'<div><div style="font-size:.85em;color:var(--ink2);margin-top:10px"><b>{esc(title)}</b><br>{sub}</div>{chart}</div>'


def fig_ladder(metric):
    """Small multiples: each panel is one idea, the run with it (blue) against the run it was added to (gray)."""
    panels, tv_rows = [], []
    for i, (cn, cl, _, pn, kept) in enumerate(TIMELINE):
        if pn is None or cn not in LAD or pn not in LAD:
            continue
        title = f"{i}. {cl}" + ("" if kept else " (not kept)")
        if metric == "ans":
            a, b = recs(pn, "ans"), recs(cn, "ans")
            xs = [s for s, _ in b]
            series = [{"name": "before this idea", "c": GRAY, "y": [float(np.mean(v)) for _, v in a]},
                      {"name": "with this idea", "c": BLUE, "y": [float(np.mean(v)) for _, v in b]}]
            sub = f"full test set: {pct(F.em(LAD, pn))} &rarr; {pct(F.em(LAD, cn))}"
            kw = dict(ymin=0.0, ymax=1.0, yticks=(0, .25, .5, .75, 1.0), xtick_idx=[1, 4, 9])
        else:
            a, b = recs(pn, "train_loss"), recs(cn, "train_loss")
            xs = [s for s, _ in b]
            series = [{"name": "before this idea", "c": GRAY, "y": [v for _, v in a]}, {"name": "with this idea", "c": BLUE, "y": [v for _, v in b]}]
            sub = f"final {series[0]['y'][-1]:.4f} &rarr; {series[1]['y'][-1]:.4f} per token"
            kw = dict(ylog=True, ymin=0.002, ymax=4.0, yticks=(0.01, 0.1, 1.0), yfmt=lambda v: f"{v:g}", vfmt=lambda v: f"{v:.4f}", xtick_idx=[7, 19, 39])
        panels.append(_panel(title, sub, series, xs, [f"step {s/1000:g}k" for s in xs], **kw))
        tv_rows.append([title] + [("%.4f" % series[k]["y"][-1]) if metric != "ans" else pct(series[k]["y"][-1]) for k in (0, 1)])
    body = legend([("before this idea", GRAY, "line"), ("with this idea", BLUE, "line")]) + '<div class="grid2">' + "".join(panels) + "</div>" + \
        table_view(["idea", "before", "after"], tv_rows)
    if metric == "ans":
        return figure("Held-out answer accuracy during training, one idea per panel",
                      "Exact match of the final answer on 1,400 held-out problems (200 per level), generated greedily every 1,000 steps. "
                      "Same 3.2M-parameter model and 10,000 steps in every run; each panel compares a run with the run the idea was added to.", body)
    return figure("Training loss, one idea per panel",
                  "Cross-entropy per target token (log scale), averaged over 250 steps. Each format has its own mix of easy and hard tokens, so the "
                  "level of two curves in a panel is not comparable when the format changes; their shape is.", body)


def fig_ladder_bars(extra=()):
    items = []
    for i, (name, lab, _, _, kept) in enumerate(TIMELINE):
        if name in LAD:
            items.append({"name": f"{i}. {lab}", "v": F.em(LAD, name), "c": BLUE if kept else GRAY, "tip": [("within 1%", pct(F.close(LAD, name))), ("kept", "yes" if kept else "no")]})
    for lab, e, c in extra:
        items.append({"name": lab, "v": e, "c": BLUE, "tip": [("within 1%", pct(c))]})
    body = legend([("kept", BLUE, "box"), ("tried, not kept", GRAY, "box")]) + hbar_chart(items, label_w=330)
    return figure("Exact match as the ideas are layered on", "Full 14,000-problem test set. Rows 0 to 14 are the same 3.2M-parameter model trained for 10,000 steps; "
                  "each idea is added to the kept ones above it. The last rows add training steps, then parameters.", body)


def fig_think():
    runs = [("L6_weighted_s0", "worked solutions only", GRAY), ("T_cont8_s0", "8 continuous thoughts, then the worked solution", BLUE),
            ("T_um8_s0", '8 "um" tokens, then the worked solution', ORANGE)]
    runs = [r for r in runs if r[0] in LAD]
    if len(runs) < 2:
        return ""
    xs = [s for s, _ in recs(runs[0][0], "ans")]
    series = [{"name": lab, "short": lab.split(",")[0], "c": c, "y": [float(np.mean(v)) for _, v in recs(n, "ans")]} for n, lab, c in runs]
    chart = figure("Thinking tokens in front of a worked solution",
                   "Held-out answer accuracy during training (1,400 problems). 3.2M parameters, final scratchpad format, 10,000 steps.",
                   line_chart(series, xs, [f"step {s/1000:g}k" for s in xs], h=260, ymin=0.0, ymax=1.0, right=170))
    rows = [[lab, pct(F.em(LAD, n)), pct(F.close(LAD, n)), f"{LAD[n]['final']['tle']:.0f}", f"{train_pflops(LAD[n]):.1f}"] + [pct(x, 0) for x in LAD[n]["final"]["em_gen"]]
            for n, lab, _ in runs]
    return chart + table(["variant", "exact", "within 1%", "token-layer evals per problem", "training PFLOP"] + [f"L{i+1}" for i in range(7)], rows)


def kind(step):
    head = step.split("=")[0].split(":")[0]
    if "~" in step:
        return "root estimates"
    if re.match(r"^(sin|cos|tan|asin|acos|atan|ln|log)\(", head) or "!" in head or "nCr" in head or "nPr" in head:
        return "looked-up values"
    if re.match(r"^(sqrt|gcd|lcm|mod|abs)\(", head) or "^" in head:
        return "summaries and power tables"
    if "%" in head or "/" in head:
        return "long division"
    if "*" in head:
        return "multiplication"
    return "addition and subtraction"


def token_budget():
    """Where the characters of a worked solution go, over the test set."""
    Z = np.load("data.npz")
    exprs = [mg.decode(r)[:-1] for r in Z["test_P"]]
    m2.configure("v2.3")
    tot, n_steps = {}, {}
    answer = 0
    for e in exprs:
        sc, a = m2.solve(e)
        answer += len(a) + 3
        for st in sc.split(";") if sc else []:
            k = kind(st)
            tot[k] = tot.get(k, 0) + len(st) + 1
            n_steps[k] = n_steps.get(k, 0) + 1
    all_chars = sum(tot.values()) + answer
    rows = [[k, f"{n_steps[k]:,}", f"{v / n_steps[k]:.0f}", pct(v / all_chars)] for k, v in sorted(tot.items(), key=lambda kv: -kv[1])]
    rows.append(["brackets, final answer, end marker", f"{len(exprs):,}", f"{answer / len(exprs):.0f}", pct(answer / all_chars)])
    return table(["kind of step", "steps in the test set", "characters per step", "share of all characters"], rows)


def fig_train_compute():
    """Held-out answer accuracy against cumulative training compute for three rungs of the ladder."""
    fams = []
    for name, lab, c in [("L0_v1_s0", "old scratchpad", GRAY), ("L1_values6_s0", "6-digit values, no procedures", ORANGE), ("L6_weighted_s0", "final format", BLUE)]:
        if name in LAD:
            per_step = train_pflops(LAD[name]) / LAD[name]["args"]["steps"]
            fams.append({"name": lab, "c": c, "pts": [(s * per_step, float(np.mean(v)), f"step {s/1000:g}k") for s, v in recs(name, "ans")]})
    if len(fams) < 2:
        return ""
    xmax = max(p[0] for f in fams for p in f["pts"])
    return figure("Answer accuracy against training compute",
                  "x: cumulative forward-plus-backward FLOPs (PFLOP = 10^15). Same model and number of steps; the formats differ in how many positions a step processes.",
                  xy_chart(fams, [], xlab="training compute so far, PFLOP", ylab="held-out answer accuracy", ymin=0.0, ymax=1.0, yticks=[0, .25, .5, .75, 1.0],
                           xticks=[t for t in (0, 10, 20, 30, 40, 50, 60, 70, 80, 90) if t <= xmax * 1.05], xmin=0, xmax=xmax * 1.05, xfmt=lambda v: f"{v:.0f}", right=190))


def effort_data():
    return F.jload("runs/ladder/E_effort_s0.effort.json")


def effort_match(e, key):
    """The cheapest threshold of a routing rule that is at least as accurate as always thinking fully."""
    full = e["efforts"]["full"]["mean"]
    return min((c for c in e.get(key, []) if c["mean"] >= full - 0.0005 and c["tau"] <= 1), key=lambda c: c["gflop"], default=None)


def effort_table():
    e = effort_data()
    if not e:
        return ""
    names = [("none", "no thinking: <think></think> answer"), ("brief", "brief: one value per operation"), ("full", "full: procedures written out")]
    rows = [[lab, pct(e["efforts"][k]["mean"]), f"{e['efforts'][k]['tokens']:.0f}", f"{e['efforts'][k]['gflop']:.2f}"] + [pct(x, 0) for x in e["efforts"][k]["exact"]] for k, lab in names]
    for key, lab in [("cascade2", "routed: no thinking, full if unsure"), ("cascade", "routed: no thinking, brief if unsure, then full")]:
        c = effort_match(e, key)
        if c:
            rows.append([f"{lab} (threshold {c['tau']:g})", pct(c["mean"]), f"{c['tokens']:.0f}", f"{c['gflop']:.2f}"] + [pct(x, 0) for x in c["exact"]])
    return table(["thinking effort", "exact", "tokens written", "GFLOP per problem"] + [f"L{i+1}" for i in range(7)], rows)


def fig_effort():
    e = effort_data()
    if not e:
        return ""
    ef = e["efforts"]
    line = lambda key: [(c["gflop"], c["mean"], f"threshold {c['tau']:g}") for c in sorted(e[key], key=lambda c: c["tau"]) if 0 < c["tau"] <= 1]
    fam = [{"name": "no thinking, then a full block if unsure", "c": BLUE, "pts": line("cascade2")},
           {"name": "no thinking, then brief, then full", "c": ORANGE, "pts": line("cascade")}]
    pts = [{"name": "always full", "c": GRAY, "x": ef["full"]["gflop"], "y": ef["full"]["mean"], "side": "r", "dy": -15}]
    return figure("Thinking harder only when unsure, against always thinking fully",
                  "x: forward FLOPs per problem (billions), counting every attempt. Each line sweeps the confidence threshold from 0.3 to 0.999: the model "
                  "answers with no thinking first and thinks harder when the probability of what it wrote is under the threshold. The axes are zoomed on "
                  f"always-full; always-none ({pct(ef['none']['mean'])} at {ef['none']['gflop']:.2f}) and always-brief ({pct(ef['brief']['mean'])} at "
                  f"{ef['brief']['gflop']:.2f}) are off the chart to the lower left.",
                  xy_chart(fam, pts, xlab="inference, GFLOP per problem", ylab="exact match", ymin=0.85, ymax=0.93, yticks=[0.86, 0.88, 0.90, 0.92],
                           xticks=[0.5, 0.6, 0.7, 0.8], xmin=0.5, xmax=0.87, xfmt=lambda v: f"{v:.2f}", right=30, end_labels=False))
