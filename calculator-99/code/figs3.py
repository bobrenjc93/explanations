"""Figures for the change-by-change ladder, the thinking-token test and the compute accounting."""
import json, os, re
import numpy as np
import mathgen as mg, mathgen2 as m2
import pagedata as pd
import figs2 as F
from charts import *

GRAY, BLUE, ORANGE, AQUA = "--base", "--s1", "--s2", "--s3"
LAD, LADLOG, LADOK = F.load_dir("runs/ladder")

# the ladder, in order: (run, short label, what changed)
RUNGS = [
    ("L0_v1_s0", "old scratchpad", "The first write-up's scratchpad: a list of intermediate values at 4 significant digits; partial products for multiplication only."),
    ("L1_values6_s0", "+ 6-digit values, one step per operation", "Every operation is restated with its result (<code>93*88=8184</code>) and values carry 6 significant digits. No procedure is written out."),
    ("L2_v20_s0", "+ procedures", "Partial products with running sums, long division, Euclid for gcd/lcm/mod, a Newton step for square roots, repeated multiplication for powers, mantissa tables for ln/log."),
    ("L3_v21_s0", "+ shared lookups, off-grid corrections", "10<sup>x</sup> and 2<sup>x</sup> from a 100-entry table for the fractional part; sin, cos and ln away from their tables by a first-order correction."),
    ("L4_v22_s0", "+ jittered Newton, e^x split, nCr chains", "Root estimates perturbed in training; e<sup>x</sup> split into whole and fractional part; n!, nPr, nCr as running products; cube roots by Newton."),
    ("L5_v23_s0", "+ products written in long division", "<code>9p7767r101</code> instead of <code>9r101</code> when the divisor has 3 or more digits."),
    ("L6_weighted_s0", "+ pool weighted to hard levels", "Training rows per level in the ratio 1:2:3:3:3:5:5 instead of equal."),
]


def recs(name, key):
    """[(step, value)] of one kind of log record."""
    return [(r["step"], r[key]) for r in LADLOG.get(name, []) if key in r]


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
    return "<ol start=\"0\">" + "".join(f"<li><strong>{cap(lab.lstrip('+ '))}.</strong> {what}</li>" for name, lab, what in RUNGS if name in LAD) + "</ol>"


def ladder_table():
    rows, prev = [], None
    for i, (name, lab, what) in enumerate(RUNGS):
        if name not in LAD:
            continue
        R = LAD[name]
        e = F.em(LAD, name)
        loss = [v for _, v in recs(name, "train_loss")][-1]
        mt = R["compute"]["mean_target_tokens"]
        rows.append([f"{i}. {lab}", pct(e), ("" if prev is None else f"{100*(e-prev):+.1f}"), pct(F.close(LAD, name)),
                     f"{mt:.0f}", f"{loss:.4f}", f"{loss*mt:.2f}", f"{train_pflops(R):.0f}", f"{infer_gflops(R):.2f}"])
        prev = e
    return table(["version", "exact", "gain", "within 1%", "tokens / problem", "loss / token", "loss / problem", "train PFLOP", "infer GFLOP"], rows)


def _panel(title, sub, series, xs, xl, **kw):
    chart = line_chart(series, xs, xl, w=400, h=180, end_labels=False, show_legend=False, tv=False, **kw)
    return f'<div><div style="font-size:.85em;color:var(--ink2);margin-top:10px"><b>{esc(title)}</b><br>{sub}</div>{chart}</div>'


def fig_ladder(metric):
    """Small multiples: each panel is one change, this version (blue) against the one before it (gray)."""
    panels, tv_rows = [], []
    have = [(n, l, w) for n, l, w in RUNGS if n in LAD]
    for i in range(1, len(have)):
        (pn, pl, _), (cn, cl, _) = have[i - 1], have[i]
        if metric == "ans":
            a, b = recs(pn, "ans"), recs(cn, "ans")
            xs = [s for s, _ in b]
            series = [{"name": "before this change", "c": GRAY, "y": [float(np.mean(v)) for _, v in a]},
                      {"name": "with this change", "c": BLUE, "y": [float(np.mean(v)) for _, v in b]}]
            sub = f"full test set: {pct(F.em(LAD, pn))} &rarr; {pct(F.em(LAD, cn))}"
            kw = dict(ymin=0.0, ymax=1.0, yticks=(0, .25, .5, .75, 1.0), xtick_idx=[1, 4, 9])
        else:
            a, b = recs(pn, "train_loss"), recs(cn, "train_loss")
            xs = [s for s, _ in b]
            series = [{"name": "before this change", "c": GRAY, "y": [v for _, v in a]}, {"name": "with this change", "c": BLUE, "y": [v for _, v in b]}]
            sub = f"final {series[0]['y'][-1]:.4f} &rarr; {series[1]['y'][-1]:.4f} per token"
            kw = dict(ylog=True, ymin=0.002, ymax=4.0, yticks=(0.01, 0.1, 1.0), yfmt=lambda v: f"{v:g}", vfmt=lambda v: f"{v:.4f}", xtick_idx=[7, 19, 39])
        panels.append(_panel(cl, sub, series, xs, [f"step {s/1000:g}k" for s in xs], **kw))
        tv_rows.append([cl] + [("%.4f" % series[k]["y"][-1]) if metric != "ans" else pct(series[k]["y"][-1]) for k in (0, 1)])
    body = legend([("before this change", GRAY, "line"), ("with this change", BLUE, "line")]) + '<div class="grid2">' + "".join(panels) + "</div>" + \
        table_view(["change", "before", "after"], tv_rows)
    if metric == "ans":
        return figure("Held-out answer accuracy during training, one change per panel",
                      "Exact match of the final answer on 1,400 held-out problems (200 per level), generated greedily every 1,000 steps. "
                      "Same 3.2M-parameter model and 10,000 steps in every run; each panel compares a version with the one before it.", body)
    return figure("Training loss, one change per panel",
                  "Cross-entropy per target token (log scale), averaged over 250 steps. Each format has its own mix of easy and hard tokens, so the "
                  "level of two curves in a panel is not comparable; their shape is.", body)


def fig_ladder_bars(extra=()):
    items = []
    for name, lab, _ in RUNGS:
        if name in LAD:
            items.append({"name": lab, "v": F.em(LAD, name), "c": GRAY if name.startswith("L0") else BLUE, "tip": [("within 1%", pct(F.close(LAD, name)))]})
    for lab, e, c in extra:
        items.append({"name": lab, "v": e, "c": BLUE, "tip": [("within 1%", pct(c))]})
    return figure("Exact match after each change", "Full 14,000-problem test set. The first seven bars are the same 3.2M model trained for 10,000 steps; "
                  "the last two add training steps and then parameters.", hbar_chart(items, label_w=310))


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
