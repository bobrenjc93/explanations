"""Figures for the fourth write-up: the wide pool (gaps, two sizes up, sine with argument reduction)."""
import json, math, os, re
import numpy as np
import mathgen as mg, mathgen2 as m2, mathgen3 as m3
import figs4 as G
from charts import *

GRAY, BLUE, ORANGE, AQUA = "--base", "--s1", "--s2", "--s3"
DIR = "runs/hard"
BEFORE = "H_full_3m_30k"
# the full fine-tune once it and all of its evaluations exist, otherwise its step-7,000 checkpoint
FULL = [f"{DIR}/W_wide_3m.json"] + [f"{DIR}/W_wide_3m.{s}.json" for s in ("gap", "stretch", "stretch2")] + [f"{DIR}/sin_final_fine.json", f"{DIR}/sin_final_far.json"]
AFTER = "W_wide_3m" if all(os.path.exists(p) for p in FULL) else "W_wide_7k"
SETS = {"stretch": "hard/stretch.json", "gap": "hard/gap.json", "stretch2": "hard/stretch2.json"}
SET = {k: G.jload(p) for k, p in SETS.items()}
OLD, NL = m3.OLD, m3.NLEVELS


def res(run, suffix):
    return G.jload(f"{DIR}/{run}.{suffix}.json")


def final(run):
    return G.jload(f"{DIR}/{run}.json")


def ok_test(run):
    p = f"{DIR}/{run}.ok.npy"
    return np.load(p) if os.path.exists(p) else None


def have():
    return final(AFTER) is not None and all(res(AFTER, s) for s in SETS)


def set_em(run, suffix):
    """Per level exact match of a run on one of the outside sets: {level index: accuracy}."""
    r = res(run, suffix)
    return {m3.LEVELS.index(k): v for k, v in r["em"].items()} if r else {}


def mean_em(run, suffix, levels=None):
    e = set_em(run, suffix)
    ls = [l for l in (levels or sorted(e)) if l in e]
    return float(np.mean([e[l] for l in ls])) if ls else None


# ------------------------------------------------------------------ sizes
def lin_terms(text):
    """Number of top-level terms of a linear equation 'solve(lhs=rhs)' (a bracketed term counts once)."""
    n = 0
    for side in text[6:-1].split("="):
        depth, n = 0, n + 1
        for i, c in enumerate(side):
            depth += (c == "(") - (c == ")")
            if c in "+-" and depth == 0 and i > 0:
                n += 1
    return n


def size_of(level, text, kind):
    if level == 9:
        return lin_terms(text)
    return int(re.findall(r"\d+", kind)[0])


KNOB = {8: "operations", 9: "terms", 12: "terms"}
TRAINED_TO = {8: (9, 14), 9: (7, 13), 12: (12, 24)}  # largest size in the original pool, in the wide pool


def size_acc(run, level, min_n=12):
    """{size: (accuracy, n)} for one level over every test set the run was scored on (standard test, stretch, gap, two sizes up)."""
    acc = {}
    ok = ok_test(run)
    if ok is not None:
        for i in np.where(G.LV == level)[0]:
            acc.setdefault(size_of(level, G.SIDE["text"][i], G.SIDE["kind"][i]), []).append(bool(ok[i]))
    for suffix, S in SET.items():
        r = res(run, suffix)
        if not r or not S:
            continue
        for j, l in enumerate(S["level"]):
            if l == level:
                acc.setdefault(size_of(level, S["text"][j], S["kind"][j]), []).append(bool(r["ok"][j]))
    return {s: (float(np.mean(v)), len(v)) for s, v in acc.items() if len(v) >= min_n}


def fig_sizes():
    """Accuracy against size for the three tiers with gaps: before and after fine-tuning on the wide pool."""
    panels, rows = [], []
    for level in (8, 9, 12):
        a, b = size_acc(BEFORE, level), size_acc(AFTER, level)
        xs = sorted(set(a) | set(b))
        series = [{"name": "before", "c": GRAY, "y": [a.get(x, (None,))[0] for x in xs]},
                  {"name": "after", "c": BLUE, "y": [b.get(x, (None,))[0] for x in xs]}]
        t0, t1 = TRAINED_TO[level]
        marks = [(xs.index(g), "") for g in m3.GAPS[level] if g in xs]
        vl = [(xs.index(t0) + 0.5, "") if t0 in xs else None, (xs.index(t1) + 0.5, "") if t1 in xs else None]
        chart = line_chart(series, xs, [str(x) for x in xs], w=760, h=230, end_labels=False, show_legend=False, tv=False, xlab=KNOB[level],
                           xtick_idx=list(range(0, len(xs), 2 if len(xs) > 14 else 1)), vlines=[v for v in vl if v] + marks)
        gaps = " and ".join(str(g) for g in m3.GAPS[level])
        panels.append(f'<div><div style="font-size:.85em;color:var(--ink2);margin-top:14px"><b>{esc(m3.LEVELS[level])}</b><br>'
                      f"original pool up to {t0} {KNOB[level]}, wide pool up to {t1} with {gaps} left out; beyond {t1}: never trained</div>{chart}</div>")
        for x in xs:
            where = "left out of the wide pool" if x in m3.GAPS[level] else "original pool" if x <= t0 else "added by the wide pool" if x <= t1 else "beyond the wide pool"
            rows.append([m3.LEVELS[level], f"{x} {KNOB[level]}", where, pct(a[x][0]) if x in a else "", pct(b[x][0]) if x in b else ""])
    body = legend([("before: trained on the original pool", GRAY, "line"), ("after: fine-tuned on the wide pool", BLUE, "line")]) + "".join(panels) + \
        table_view(["level", "size", "status", "before", "after"], rows)
    return figure("Accuracy against size, before and after widening the pool",
                  "Vertical lines: the largest size in the original pool, the largest in the wide pool, and the two sizes the wide pool leaves "
                  "out. Sizes with fewer than 12 test problems are not drawn.", body)


def gap_table():
    """The left-out sizes against their trained neighbours, after fine-tuning."""
    rows = []
    for level in (8, 9, 12):
        b = size_acc(AFTER, level)
        for g in m3.GAPS[level]:
            nb = [b[x][0] for x in (g - 1, g + 1) if x in b]
            if g in b:
                rows.append([m3.LEVELS[level], f"{g} {KNOB[level]}", f"{b[g][1]}", pct(b[g][0]), pct(float(np.mean(nb))) if nb else ""])
    return table(["level", "size left out", "test problems", "exact match", "mean of the two neighbouring sizes"], rows)


def gap_summary():
    """(mean accuracy at the gap sizes, mean accuracy at their neighbours) after fine-tuning."""
    g_, n_ = [], []
    for level in (8, 9, 12):
        b = size_acc(AFTER, level)
        for g in m3.GAPS[level]:
            if g in b:
                g_.append(b[g][0])
                n_ += [b[x][0] for x in (g - 1, g + 1) if x in b]
    return float(np.mean(g_)), float(np.mean(n_))


def pool_table():
    what = [
        ("L8 big numbers", "sums of 8-14 digit numbers, products of 4-8 digits", "to 20 and 11 digits", "", "21-26 and 12-14 digits"),
        ("L9 long expressions", "4 to 9 operations", "to 14 operations", "7, 12", "15 to 19 operations"),
        ("L10 linear equations", "2 to 7 terms", "to 13 terms", "6, 10", "15 to 19 terms"),
        ("L11 quadratics", "coefficients to about 40", "to 400", "", "to 4,000"),
        ("L12 linear systems", "3x3 with one-digit coefficients", "3x3 with coefficients to 30", "", "to 99"),
        ("L13 sums", "3 to 12 terms", "to 24 terms", "9, 17", "25 to 36 terms"),
        ("L14 Newton's method", "solutions to 2,304 tokens", "to 4,096 tokens", "", ""),
        ("sin, cos", "|x| to 9.99, from a table", "|x| to 100, by argument reduction", "", "|x| to 1,000"),
    ]
    return table(["", "original pool", "wide pool adds", "sizes left out", "two sizes up (test only)"], [list(r) for r in what], num_from=9)


def before_after_table():
    fa, fb = final(BEFORE), final(AFTER)
    rows = []
    for l in range(OLD, NL):
        s1, s2 = set_em(BEFORE, "stretch"), set_em(AFTER, "stretch")
        t1, t2 = set_em(BEFORE, "stretch2"), set_em(AFTER, "stretch2")
        rows.append([m3.LEVELS[l], pct(fa["final"]["em"][l]), pct(fb["final"]["em"][l]), pct(s1[l]) if l in s1 else "", pct(s2[l]) if l in s2 else "",
                     pct(t1[l]) if l in t1 else "", pct(t2[l]) if l in t2 else ""])
    f = lambda v: "" if v is None else pct(v)
    six = list(range(OLD, NL - 1))
    rows.append(["mean, L8 to L13", pct(float(np.mean(fa["final"]["em"][OLD:NL - 1]))), pct(float(np.mean(fb["final"]["em"][OLD:NL - 1]))),
                 f(mean_em(BEFORE, "stretch", six)), f(mean_em(AFTER, "stretch", six)), f(mean_em(BEFORE, "stretch2")), f(mean_em(AFTER, "stretch2"))])
    return table(["level", "original range: before", "after", "one size up: before", "after", "two sizes up: before", "after"], rows)


def fig_edges():
    """Three ranges, before and after: the six tiers that have all three."""
    six = list(range(OLD, NL - 1))
    fa, fb = final(BEFORE), final(AFTER)
    items = []
    for lab, a, b in (("original range", float(np.mean(fa["final"]["em"][OLD:NL - 1])), float(np.mean(fb["final"]["em"][OLD:NL - 1]))),
                      ("one size up (now in the pool)", mean_em(BEFORE, "stretch", six), mean_em(AFTER, "stretch", six)),
                      ("two sizes up (never trained)", mean_em(BEFORE, "stretch2"), mean_em(AFTER, "stretch2"))):
        items.append({"name": f"{lab}: before", "v": a, "c": GRAY})
        items.append({"name": "after", "v": b, "c": BLUE})
    body = legend([("before: trained on the original pool", GRAY, "box"), ("after: fine-tuned on the wide pool", BLUE, "box")]) + hbar_chart(items, label_w=300)
    return figure("The edge moves with the pool", "Exact match, mean of L8 to L13 (the six tiers with a test set at every size).", body)


# ------------------------------------------------------------------ sine
_SIN = "sin_final" if AFTER == "W_wide_3m" else "sin_wide"
SINF = G.jload(f"{DIR}/{_SIN}_fine.json")   # every 0.05 out to +-130
SINL = G.jload(f"{DIR}/{_SIN}_far.json")    # every 0.37 out to +-1000
SIN_RANGES = [(0, 10), (10, 40), (40, 100), (100, 300), (300, 1000)]
_CEIL = {}


def sin_ceiling(texts):
    """Whether the reference solution (argument reduction, 6-digit arithmetic) ends in the true answer, per probe point."""
    key = id(texts)
    if key not in _CEIL:
        saved = dict(m2.OPT)
        m2.configure("v2.3")
        m2.OPT["reduce"] = True
        out = []
        for t in texts:
            try:
                out.append(m2.solve(t)[1] == mg.calc(t))
            except mg.Reject:
                out.append(False)
        m2.OPT.update(saved)
        _CEIL[key] = np.array(out)
    return _CEIL[key]


def sin_range_stats(name):
    """[(label, exact, within 0.1, ceiling, n)] by |x|, both signs together, from the two probes."""
    rows = []
    for a, b in SIN_RANGES:
        S = SINF if b <= 100 else SINL
        x = np.array(S["x"])
        valid = np.array([t is not None for t in S["truth"]])
        m = (np.abs(x) >= a) & (np.abs(x) < b) & valid
        ok = np.array(S["models"][name]["ok"])
        val = np.array([np.nan if G._num(v) is None else G._num(v) for v in S["models"][name]["ans"]])
        rows.append((f"{a} to {b}", float(ok[m].mean()), float(np.nanmean((np.abs(val - np.sin(x)) < 0.1)[m])), float(sin_ceiling(S["text"])[m].mean()), int(m.sum())))
    return rows


def sin_side(name, a, b, sign):
    """Exact match for x in [a, b) on one side of zero."""
    S = SINF if b <= 100 else SINL
    x = np.array(S["x"])
    m = (sign * x >= a) & (sign * x < b) & np.array([t is not None for t in S["truth"]])
    return float(np.array(S["models"][name]["ok"])[m].mean())


def fig_sin_ranges():
    if not SINF or not SINL:
        return ""
    a, b = sin_range_stats("before"), sin_range_stats("after")
    labs = [r[0] for r in a]
    series = [{"name": "a perfect solver following the worked solution (ceiling)", "short": "ceiling", "c": GRAY, "y": [r[3] for r in b], "dash": True},
              {"name": "before: sine as a table", "short": "before", "c": ORANGE, "y": [r[1] for r in a]},
              {"name": "after: sine by argument reduction, trained to |x| = 100", "short": "after", "c": BLUE, "y": [r[1] for r in b]}]
    return figure("Exact match on sin(x), by size of x",
                  "Both signs of x. The first three ranges are inside what the fine-tuned model was trained on; the last two are beyond it.",
                  line_chart(series, labs, labs, h=300, xlab="|x|", right=84, vlines=[(2.5, "end of the wide pool")], tv_header="|x|"))


def sin_table():
    a, b = sin_range_stats("before"), sin_range_stats("after")
    rows = [[ra[0], f"{ra[4]:,}", pct(ra[1]), pct(rb[1]), pct(rb[3]), pct(ra[2], 0), pct(rb[2], 0)] for ra, rb in zip(a, b)]
    return table(["|x|", "probe points", "exact: before", "exact: after", "ceiling", "within 0.1: before", "within 0.1: after"], rows)


def fig_sin_curve(name="after", lo=70, hi=130, step=25, edge=100):
    """The fine-tuned model's sin(x) as dots over the true curve, across the end of its training range."""
    S = SINF
    if not S:
        return ""
    x, truth, ans = S["x"], S["truth"], S["models"][name]["ans"]
    idx = [i for i in range(len(x)) if lo <= x[i] <= hi and round(x[i] * 100) % step == 0]
    w, h = 760, 300
    L, R_, T, B = 40, 14, 24, 34
    pw, ph = w - L - R_, h - T - B
    X = lambda v: L + pw * (v - lo) / (hi - lo)
    Y = lambda v: T + ph * (1 - (max(-1.15, min(1.15, v)) + 1.15) / 2.3)
    o = [f'<rect x="{X(lo):.1f}" y="{T}" width="{X(edge) - X(lo):.1f}" height="{ph}" fill="var(--grid)" opacity=".45"/>',
         f'<text x="{(X(lo) + X(edge)) / 2:.1f}" y="{T - 8}" text-anchor="middle" class="lab">in the wide pool: |x| up to {edge}</text>',
         f'<text x="{(X(edge) + X(hi)) / 2:.1f}" y="{T - 8}" text-anchor="middle" class="lab">never trained</text>']
    for t in (-1, -0.5, 0, 0.5, 1):
        o.append(f'<line class="grid" x1="{L}" x2="{L + pw}" y1="{Y(t):.1f}" y2="{Y(t):.1f}"/>')
        o.append(f'<text x="{L - 6}" y="{Y(t) + 4:.1f}" text-anchor="end">{t:g}</text>')
    for t in range(lo, hi + 1, 10):
        o.append(f'<text x="{X(t):.1f}" y="{T + ph + 15}" text-anchor="middle">{t}</text>')
    o.append(f'<text x="{L + pw / 2:.1f}" y="{h - 3}" text-anchor="middle" class="lab">x</text>')
    fine = [lo + j * 0.05 for j in range(int((hi - lo) / 0.05) + 1)]
    o.append(f'<path class="ln" d="M{" L".join(f"{X(v):.1f},{Y(math.sin(v)):.1f}" for v in fine)}" stroke="var({GRAY})"/>')
    for i in idx:
        v = G._num(ans[i])
        if v is not None:
            o.append(f'<circle cx="{X(x[i]):.1f}" cy="{Y(v):.1f}" r="3" fill="var({BLUE})" stroke="var(--surface)" stroke-width="1"/>')
    o.append(f'<line class="xh" x1="0" x2="0" y1="{T}" y2="{T + ph}"/>')
    data = {"w": w, "px": [round(X(x[i]), 1) for i in idx], "xl": [S["text"][i] for i in idx],
            "s": [{"n": "true value", "c": GRAY, "v": [truth[i] or "0" for i in idx]}, {"n": "the model's answer", "c": BLUE, "v": [ans[i] for i in idx]}]}
    svg = f'<svg viewBox="0 0 {w} {h}" role="img" data-xh="{esc(json.dumps(data))}">' + "".join(o) + "</svg>"
    return figure("The fine-tuned model's sin(x) across the end of its training range",
                  f"x from {lo} to {hi}, one dot per 0.{step:02d}. A right answer sits on the curve. Hover for the values.",
                  legend([("sin(x)", GRAY, "line"), ("the model's answer", BLUE, "box")]) + svg)
